"""Train the Phase 11A shared-PVB-H-block adapter on exact protein views.

This runner deliberately reuses the Phase 9 exact batching/evaluation helpers.
It trains only ``shared_hblock_adapter`` and ``shared_hblock_gate`` after a
complete ``pvb_full`` source load, selects one checkpoint using validation
reconstruction, and writes an explicit lock manifest.  It never touches test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping

import numpy as np
import torch

from data.collate import collate_fn
from data.mmap_dataset import UniDataset
from data.protein_view import make_protein_only_item
from scripts.phase9_train_eval import (
    DEFAULT_FUSED_ROOT,
    BatchRecord,
    _accumulate,
    _checksums,
    _collate,
    _device,
    _empty_accumulator,
    _evaluate_model,
    _exact_batches,
    _metric_values,
    _seed,
    _summarize,
)
from scripts.profile_training_paths import _load_roles, build_model
from utils.fusion_training import (
    configure_fusion_parameters,
    fusion_gradient_norms,
    fusion_parameter_groups,
)
from utils.phase10_diagnostics import summarize_tensor
from utils.phase13_ablation import phase13_adapter_variant_names


DEFAULT_PVB_FULL_STATE = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/"
    "checkpoints/pvb_state_dict.pt"
)
DEFAULT_PHASE11_ROOT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11"
)
EXPECTED_SHARED_TRAINABLE = frozenset(
    {
        "shared_hblock_gate",
        "shared_hblock_adapter.projection.0.bias",
        "shared_hblock_adapter.projection.0.weight",
        "shared_hblock_adapter.projection.1.bias",
        "shared_hblock_adapter.projection.1.weight",
        "shared_hblock_adapter.projection.3.bias",
        "shared_hblock_adapter.projection.3.weight",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--validation-seed", type=int, default=20260810)
    parser.add_argument("--fused-data-root", default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--pvb-checkpoint", default=DEFAULT_PVB_FULL_STATE)
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--last-checkpoint", default=None)
    parser.add_argument("--lock-out", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-budget", type=int, default=4_000_000)
    parser.add_argument("--max-atoms-per-batch", type=int, default=4096)
    parser.add_argument("--max-items-per-batch", type=int, default=8)
    parser.add_argument("--pvb-lr", type=float, default=1.0e-3)
    parser.add_argument("--anew-lr", type=float, default=1.0e-5)
    parser.add_argument("--projector-lr", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--shared-hblock-variant", choices=phase13_adapter_variant_names(), default="real")
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Debug limit; zero means the complete train/valid views.",
    )
    return parser


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_roles(reports: Mapping) -> dict:
    return {
        role: {
            "coverage": report.coverage,
            "matched": len(report.matched_keys),
            "expected": report.expected_key_count,
            "source_keys": len(report.source_keys),
            "missing": len(report.missing_keys),
            "unexpected": len(report.unexpected_keys),
            "shape_mismatches": len(report.shape_mismatches),
        }
        for role, report in sorted(reports.items())
    }


def _add_rec_total(summary: dict) -> dict:
    if "per_seed" in summary:
        for seed_result in summary["per_seed"]:
            seed_result["rec_total"] = {
                weighting: float(
                    seed_result["rec_vel"][weighting]
                    + seed_result["rec_drf"][weighting]
                )
                for weighting in ("batch_mean", "atom_weighted_mean")
            }
        for weighting in ("batch_mean", "atom_weighted_mean"):
            values = [
                result["rec_total"][weighting]
                for result in summary["per_seed"]
            ]
            summary["aggregate"].setdefault("rec_total", {})[weighting] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=0)) if values else None,
            }
    else:
        summary["rec_total"] = {
            weighting: float(
                summary["rec_vel"][weighting]
                + summary["rec_drf"][weighting]
            )
            for weighting in ("batch_mean", "atom_weighted_mean")
        }
    return summary


def _view_counts(
    dataset: UniDataset,
    transform: Callable[[dict], dict],
    *,
    budget: int,
    max_atoms: int,
    max_items: int,
    shuffle: bool,
    seed: int,
    max_items_total: int,
) -> dict[str, int]:
    counts = {
        "items": 0,
        "atoms": 0,
        "batches": 0,
        "padded_cost": 0,
        "oversized": 0,
    }
    for record in _exact_batches(
        dataset,
        transform,
        budget=budget,
        max_atoms=max_atoms,
        max_items=max_items,
        shuffle=shuffle,
        seed=seed,
        max_items_total=max_items_total,
    ):
        counts["items"] += len(record.indices)
        counts["atoms"] += sum(len(item["atype"]) for item in record.items)
        counts["batches"] += 1
        counts["padded_cost"] += record.padded_cost
        counts["oversized"] += int(record.oversized)
    return counts


def _diagnostic_batch(
    dataset: UniDataset, device: torch.device, index: int = 0
) -> dict:
    item = make_protein_only_item(dataset[index])
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in collate_fn([[item]]).items()
    }


def _shared_diagnostics(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> dict:
    """Summarize PVB posterior and shared adapter state without gradients."""

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            pvb = model._pvb_encoder_diagnostics(
                batch["atype"],
                batch["btype"],
                batch["x0"],
                batch["abid"],
                batch["mask"],
                batch["edge_mask"],
                batch["bond_index"],
            )
            shared = model.shared_hblock_adapter(
                pvb["H_pvb"],
                batch["atom_block_id"],
                batch["block_lengths"],
                block_batch=batch.get("block_batch"),
                variant=model.shared_hblock_variant,
            )
            gate = float(model.shared_hblock_gate.detach().cpu().item())
            projection_norm = math.sqrt(
                sum(
                    float(parameter.detach().float().pow(2).sum().cpu())
                    for parameter in model.shared_hblock_adapter.parameters()
                )
            )
            return {
                "fusion_mode": model.fusion_mode,
                "shared_hblock_variant": model.shared_hblock_variant,
                "anew_encoder_constructed": model.anew_block_encoder is not None,
                "pvb_log_var": summarize_tensor(pvb["log_var_pvb"]),
                "pvb_kl": float(pvb["kl_loss_pvb"].detach().cpu()),
                "pvb_embedding_norm": float(
                    torch.linalg.vector_norm(pvb["H_pvb"].detach().float()).cpu()
                ),
                "h_block_norm": float(
                    torch.linalg.vector_norm(shared["H_block"].detach().float()).cpu()
                ),
                "projected_condition_block_norm": float(
                    torch.linalg.vector_norm(
                        shared["condition_block"].detach().float()
                    ).cpu()
                ),
                "projected_condition_atom_norm": float(
                    torch.linalg.vector_norm(
                        shared["condition_atom"].detach().float()
                    ).cpu()
                ),
                "block_gate": gate,
                "tanh_block_gate": math.tanh(gate),
                "projector_parameter_norm": projection_norm,
                "shared_hblock_log_var_used_in_loss": False,
            }
    finally:
        model.train(was_training)


def _save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    valid: dict,
    valid_rec_total: float,
    source_checksums: dict[str, str],
    source_reports: dict,
    trainable_names: list[str],
    protocol: dict,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 5,
        "checkpoint_kind": "phase11_adapter_only",
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if key not in source_checksums
        },
        # Phase 11 checkpoints are inference/evaluation locks.  The shared
        # output volume has room for one model archive but not a second copy
        # plus optimizer moments, so optimizer state is intentionally omitted.
        # The training report records the exact optimizer membership/protocol.
        "optimizer_state_saved": False,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "valid": valid,
        "valid_rec_total": float(valid_rec_total),
        "source_checksums_before": source_checksums,
        "source_checksums_after": _checksums(model, source_checksums),
        "source_role_reports": source_reports,
        "source_checkpoint_state_is_external": True,
        "fusion_stage": "source_frozen",
        "phase11": {
            "fusion_mode": "pvb_shared_hblock",
            "shared_hblock_variant": protocol.get("shared_hblock_variant", "real"),
            "pvb_role": "pvb_full",
            "checkpoint_kind": "phase11_adapter_only",
            "selection_metric": "valid_rec_total_batch_mean",
            "trainable_names": trainable_names,
            "anew_encoder_constructed": False,
            "test_evaluated": False,
            "protocol": protocol,
        },
    }
    # The output volume can have only one full Phase 11 checkpoint free.  A
    # direct overwrite needs old and new archives simultaneously, so remove
    # only this task-owned target before writing the replacement.  Phase 9/10
    # artifacts are never targets of this runner.
    if target.exists():
        target.unlink()
    torch.save(payload, target)


def _write_lock(
    path: str,
    checkpoint: str,
    *,
    epoch: int,
    global_step: int,
    valid: dict,
    valid_rec_total: float,
    source_checksums: dict[str, str],
    source_reports: dict,
    source_checkpoint: str,
    trainable_names: list[str],
    protocol: dict,
) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = {
        "format_version": 1,
        "status": "locked",
        "test_evaluated": False,
        "source_checkpoint": str(Path(source_checkpoint)),
        "source_checkpoint_sha256": _file_sha256(source_checkpoint),
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_kind": "phase11_adapter_only",
        "checkpoint_sha256": _file_sha256(checkpoint),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "valid_rec_total_batch_mean": float(valid_rec_total),
        "valid": valid,
        "pvb_role": "pvb_full",
        "fusion_mode": "pvb_shared_hblock",
        "checkpoint_kind": "phase11_adapter_only",
        "trainable_names": trainable_names,
        "source_checksums": source_checksums,
        "source_role_reports": source_reports,
        "protocol": protocol,
        "selection_rule": "minimum valid rec_vel + rec_drf; test not used",
    }
    target.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock["lock_path"] = str(target)
    lock["lock_sha256"] = _file_sha256(target)
    return lock


def _eval_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        eval_seeds=[args.validation_seed],
        original_data_root="/data/pvb_cross_dataset_20260810/blocks",
        fused_data_root=args.fused_data_root,
        batch_budget=args.batch_budget,
        max_atoms_per_batch=args.max_atoms_per_batch,
        max_items_per_batch=args.max_items_per_batch,
        max_items=args.max_items,
    )


def run(args: argparse.Namespace) -> dict:
    if args.epochs <= 0 or args.min_epochs <= 0:
        raise ValueError("epochs and min-epochs must be positive")
    if args.min_epochs > args.epochs:
        raise ValueError("min-epochs cannot exceed epochs")
    if args.patience <= 0:
        raise ValueError("patience must be positive")

    device = _device(args.device)
    _seed(args.seed)
    model = build_model(
        "pvb_shared_hblock",
        shared_hblock_variant=args.shared_hblock_variant,
        shared_hblock_seed=args.seed,
    )
    reports = _load_roles(model, args.pvb_checkpoint, None, "pvb_full")
    source_keys = set().union(*(report.matched_keys for report in reports.values()))
    source_reports = _report_roles(reports)
    if source_reports["pvb_full"]["coverage"] != 1.0:
        raise RuntimeError("pvb_full checkpoint coverage is not complete")
    before = _checksums(model, source_keys)
    stage_counts = configure_fusion_parameters(
        model, "source_frozen", source_keys=source_keys
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if set(trainable_names) != set(EXPECTED_SHARED_TRAINABLE):
        raise AssertionError(
            "Phase 11 shared trainable set mismatch: "
            f"{trainable_names}"
        )
    groups = fusion_parameter_groups(
        model,
        pvb_lr=args.pvb_lr,
        anew_lr=args.anew_lr,
        projector_lr=args.projector_lr,
    )
    optimizer = torch.optim.Adam(groups)
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != trainable_ids or len(groups) != 1:
        raise AssertionError("Phase 11 optimizer is not the exact adapter complement")
    model.to(device)

    train_dataset = UniDataset(str(Path(args.fused_data_root) / "train_block"))
    valid_dataset = UniDataset(str(Path(args.fused_data_root) / "valid_block"))
    train_expected = _view_counts(
        train_dataset,
        make_protein_only_item,
        budget=args.batch_budget,
        max_atoms=args.max_atoms_per_batch,
        max_items=args.max_items_per_batch,
        shuffle=False,
        seed=args.seed,
        max_items_total=args.max_items,
    )
    valid_expected = _view_counts(
        valid_dataset,
        make_protein_only_item,
        budget=args.batch_budget,
        max_atoms=args.max_atoms_per_batch,
        max_items=args.max_items_per_batch,
        shuffle=False,
        seed=args.validation_seed,
        max_items_total=args.max_items,
    )
    diagnostic_batch = _diagnostic_batch(train_dataset, device, 0)

    checkpoint_out = args.checkpoint_out or str(
        Path(DEFAULT_PHASE11_ROOT)
        / "checkpoints/pvb_shared_hblock_best.ckpt"
    )
    last_checkpoint = args.last_checkpoint or str(
        Path(checkpoint_out).with_name("pvb_shared_hblock_last.ckpt")
    )
    lock_out = args.lock_out or str(
        Path(checkpoint_out).with_name("phase11_shared_hblock_best.lock.json")
    )
    protocol = {
        "max_epochs": int(args.epochs),
        "minimum_epochs": int(args.min_epochs),
        "patience": int(args.patience),
        "projector_gate_lr": float(args.projector_lr),
        "pvb_source_frozen": True,
        "anew_encoder_constructed": False,
        "gradient_clip": float(args.grad_clip),
        "shared_hblock_variant": args.shared_hblock_variant,
        "selection_metric": "valid rec_vel + rec_drf, batch mean",
        "test_used_for_selection": False,
        "pvb_checkpoint": args.pvb_checkpoint,
        "batch_budget": int(args.batch_budget),
        "max_atoms_per_batch": int(args.max_atoms_per_batch),
        "max_items_per_batch": int(args.max_items_per_batch),
    }

    history = []
    global_step = 0
    completed_steps = 0
    train_oversized = 0
    start = time.perf_counter()
    best_valid_rec_total: float | None = None
    best_epoch: int | None = None
    no_improvement = 0
    stop_reason = "max_epochs"
    eval_args = _eval_args(args)
    last_gradient_norms = {}

    for epoch in range(args.epochs):
        model.train()
        epoch_acc = _empty_accumulator()
        epoch_start = time.perf_counter()
        train_items_seen = 0
        for record in _exact_batches(
            train_dataset,
            make_protein_only_item,
            budget=args.batch_budget,
            max_atoms=args.max_atoms_per_batch,
            max_items=args.max_items_per_batch,
            shuffle=True,
            seed=args.seed + epoch,
            max_items_total=args.max_items,
        ):
            if args.max_steps and completed_steps >= args.max_steps:
                break
            train_oversized += int(record.oversized)
            train_items_seen += len(record.indices)
            batch = _collate(record, device)
            _seed(args.seed + 10_000_019 * (global_step + 1))
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model._train(batch, mode="pretrain")
            metrics = _metric_values(loss, parts)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(
                        f"non-finite Phase 11 gradient: {name}"
                    )
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
            optimizer.step()
            last_gradient_norms = fusion_gradient_norms(model)
            _accumulate(epoch_acc, metrics, int(batch["x0"].shape[0]), record)
            global_step += 1
            completed_steps += 1
            if completed_steps % 100 == 0:
                print(
                    f"phase11 train epoch={epoch} steps={completed_steps} "
                    f"loss={metrics['loss']:.6f} atoms={epoch_acc['atom_count']}",
                    flush=True,
                )
        if epoch_acc["batch_count"] == 0:
            raise RuntimeError("no Phase 11 training batches were processed")
        if not args.max_steps and train_items_seen != train_expected["items"]:
            raise AssertionError(
                "train traversal incomplete: "
                f"{train_items_seen}/{train_expected['items']} items"
            )
        train_summary = _add_rec_total(_summarize(epoch_acc))

        valid = _evaluate_model(
            model,
            mode="fused",
            split="valid",
            args=eval_args,
            device=device,
        )
        valid = _add_rec_total(valid["pdbind_protein_only"])
        valid_aggregate = valid["aggregate"]
        for key, expected_key in (
            ("batch_count", "batches"),
            ("atom_count", "atoms"),
            ("oversized_count", "oversized"),
        ):
            if valid_aggregate[key] != valid_expected[expected_key]:
                raise AssertionError(
                    f"valid {key} mismatch: "
                    f"{valid_aggregate[key]} != {valid_expected[expected_key]}"
                )
        valid_rec_total = float(valid_aggregate["rec_total"]["batch_mean"]["mean"])
        diagnostic = _shared_diagnostics(model, diagnostic_batch)
        improved = (
            best_valid_rec_total is None
            or valid_rec_total < best_valid_rec_total - 1.0e-12
        )
        if improved:
            best_valid_rec_total = valid_rec_total
            best_epoch = epoch
            no_improvement = 0
            _save_checkpoint(
                checkpoint_out,
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                valid=valid,
                valid_rec_total=valid_rec_total,
                source_checksums=before,
                source_reports=source_reports,
                trainable_names=trainable_names,
                protocol=protocol,
            )
        else:
            no_improvement += 1
        # The full model plus optimizer is roughly 40 MiB. On the shared
        # output volume a separate rolling copy can exhaust the remaining
        # quota, so callers may alias last_checkpoint to the selected
        # checkpoint; in that mode keep only the best artifact.
        if Path(last_checkpoint).resolve() != Path(checkpoint_out).resolve():
            _save_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                valid=valid,
                valid_rec_total=valid_rec_total,
                source_checksums=before,
                source_reports=source_reports,
                trainable_names=trainable_names,
                protocol=protocol,
            )
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_summary,
                "valid": valid,
                "valid_rec_total": valid_rec_total,
                "improved": improved,
                "no_improvement_epochs": no_improvement,
                "diagnostics": diagnostic,
                "gradient_norms_last_update": last_gradient_norms,
                "train_items_seen": train_items_seen,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )
        print(
            f"phase11 valid epoch={epoch} rec_total={valid_rec_total:.6f} "
            f"best={best_valid_rec_total:.6f} improved={improved}",
            flush=True,
        )
        if args.max_steps and completed_steps >= args.max_steps:
            stop_reason = "max_steps"
            break
        if epoch + 1 >= args.min_epochs and no_improvement >= args.patience:
            stop_reason = "valid_rec_total_patience"
            break

    after = _checksums(model, source_keys)
    mismatches = sorted(
        key for key in source_keys if before.get(key) != after.get(key)
    )
    if mismatches:
        raise AssertionError(f"source checkpoint tensors changed: {mismatches}")
    if best_valid_rec_total is None or best_epoch is None:
        raise AssertionError("Phase 11 did not produce a validation checkpoint")
    if not Path(checkpoint_out).exists():
        raise AssertionError(f"missing best checkpoint: {checkpoint_out}")
    lock = _write_lock(
        lock_out,
        checkpoint_out,
        epoch=best_epoch,
        global_step=history[best_epoch]["global_step"],
        valid=history[best_epoch]["valid"],
        valid_rec_total=best_valid_rec_total,
        source_checksums=before,
        source_reports=source_reports,
        source_checkpoint=args.pvb_checkpoint,
        trainable_names=trainable_names,
        protocol=protocol,
    )
    result = {
        "format_version": 1,
        "mode": "phase11_train_shared_hblock",
        "fusion_mode": "pvb_shared_hblock",
        "shared_hblock_variant": args.shared_hblock_variant,
        "pvb_role": "pvb_full",
        "device": str(device),
        "data_root": args.fused_data_root,
        "epochs_requested": args.epochs,
        "minimum_epochs": args.min_epochs,
        "patience": args.patience,
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "steps_completed": completed_steps,
        "batch_budget": args.batch_budget,
        "max_atoms_per_batch": args.max_atoms_per_batch,
        "max_items_per_batch": args.max_items_per_batch,
        "train_expected": train_expected,
        "valid_expected": valid_expected,
        "train_oversized_batches": train_oversized,
        "fusion_parameter_counts": stage_counts,
        "source_key_count": len(source_keys),
        "source_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in source_keys
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_names": trainable_names,
        "optimizer_group_count": len(optimizer.param_groups),
        "optimizer_is_exact_trainable_complement": optimizer_ids == trainable_ids,
        "source_checkpoint_unchanged": not mismatches,
        "source_checksum_mismatches": mismatches,
        "anew_encoder_constructed": model.anew_block_encoder is not None,
        "anew_variance_used_in_loss": False,
        "best_epoch": best_epoch,
        "best_global_step": history[best_epoch]["global_step"],
        "best_valid_rec_total": best_valid_rec_total,
        "checkpoint_out": checkpoint_out,
        "checkpoint_out_sha256": _file_sha256(checkpoint_out),
        "last_checkpoint": last_checkpoint,
        "last_checkpoint_sha256": _file_sha256(last_checkpoint),
        "lock": lock,
        "source_reports": source_reports,
        "source_checkpoint": args.pvb_checkpoint,
        "source_checkpoint_sha256": _file_sha256(args.pvb_checkpoint),
        "history": history,
        "test_evaluated": False,
        "checkpoint_kind": "phase11_adapter_only",
        "optimizer_state_saved": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
