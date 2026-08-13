"""Train and evaluate the Phase 10 posterior-preserving H-block adapter.

This runner is intentionally thin: exact materialized batching, evaluation,
metric accumulation, and checkpoint-key utilities come from the Phase 9
runner. Phase 10 adds only the corrected model role/mode, validation
``rec_total`` selection, bounded early stopping, and epoch diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping

import numpy as np
import torch

from data.collate import collate_fn
from data.mmap_dataset import UniDataset
from data.protein_view import make_protein_only_item
from scripts.phase9_train_eval import (
    DEFAULT_ANEW_STATE,
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
from utils.phase10_diagnostics import collect_phase10_diagnostics


DEFAULT_PVB_FULL_STATE = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/"
    "checkpoints/pvb_state_dict.pt"
)
DEFAULT_PHASE10_ROOT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--validation-seed", type=int, default=20260810)
    parser.add_argument("--fused-data-root", default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--pvb-checkpoint", default=DEFAULT_PVB_FULL_STATE)
    parser.add_argument("--anew-checkpoint", default=DEFAULT_ANEW_STATE)
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--last-checkpoint", default=None)
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


def _add_rec_total(summary: dict) -> dict:
    """Add reconstruction total without changing Phase 9 metric semantics."""

    if "per_seed" in summary:
        for seed_result in summary["per_seed"]:
            seed_result["rec_total"] = {
                weighting: {
                    "value": float(
                        seed_result["rec_vel"][weighting]
                        + seed_result["rec_drf"][weighting]
                    )
                }
                for weighting in ("batch_mean", "atom_weighted_mean")
            }
        for weighting in ("batch_mean", "atom_weighted_mean"):
            values = [
                result["rec_total"][weighting]["value"]
                for result in summary["per_seed"]
            ]
            summary["aggregate"].setdefault("rec_total", {})[weighting] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=0)) if values else None,
            }
    else:
        summary["rec_total"] = {
            weighting: float(
                summary["rec_vel"][weighting] + summary["rec_drf"][weighting]
            )
            for weighting in ("batch_mean", "atom_weighted_mean")
        }
    return summary


def _expected_counts(
    dataset: UniDataset,
    transform: Callable[[dict], dict],
    max_items: int,
) -> dict[str, int]:
    limit = len(dataset) if not max_items else min(len(dataset), max_items)
    atoms = 0
    for index in range(limit):
        atoms += len(transform(dataset[index])["atype"])
    return {"items": limit, "atoms": atoms}


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


def _save_phase10_checkpoint(
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
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 3,
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "valid": valid,
        "valid_rec_total": float(valid_rec_total),
        "source_checksums_before": source_checksums,
        "source_checksums_after": _checksums(model, source_checksums),
        "source_role_reports": source_reports,
        "fusion_stage": "source_frozen",
        "phase10": {
            "fusion_mode": "anew_block_pvb_posterior",
            "pvb_role": "pvb_full",
            "selection_metric": "valid_rec_total_batch_mean",
            "trainable_names": trainable_names,
            "protocol": {
                "max_epochs": 5,
                "minimum_epochs": 3,
                "patience": 2,
                "projector_gate_lr": 1.0e-3,
                "pvb_source_frozen": True,
                "anew_source_frozen": True,
                "gradient_clip": 1.0,
            },
        },
    }
    torch.save(payload, target)


def _diagnostic_batch(
    dataset: UniDataset, device: torch.device, index: int = 0
) -> dict:
    item = make_protein_only_item(dataset[index])
    return _collate(
        BatchRecord(indices=[index], items=[item], padded_cost=len(item["atype"]) ** 2, oversized=False),
        device,
    )


def _epoch_diagnostics(
    model: torch.nn.Module,
    batch: dict,
    *,
    seed: int,
    epoch: int,
    global_step: int,
) -> dict:
    """Collect a real fixed-record gradient/variance record without updating."""

    model.train()
    model.zero_grad(set_to_none=True)
    _seed(seed)
    loss, parts = model._train(batch, mode="pretrain")
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("non-finite Phase 10 diagnostic loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"non-finite Phase 10 diagnostic gradient: {name}")
    record = collect_phase10_diagnostics(
        model,
        batch,
        loss=loss,
        parts=parts,
        epoch=epoch,
        global_step=global_step,
    )
    record["source_freezing"] = True
    model.zero_grad(set_to_none=True)
    return record


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
    model = build_model("anew_block_pvb_posterior")
    with torch.no_grad():
        reports = _load_roles(
            model,
            args.pvb_checkpoint,
            args.anew_checkpoint,
            "pvb_full",
        )
    source_keys = set().union(*(report.matched_keys for report in reports.values()))
    source_reports = _report_roles(reports)
    if source_reports["pvb_full"]["coverage"] != 1.0:
        raise RuntimeError("pvb_full checkpoint coverage is not complete")
    before = {
        key: _tensor_sha256(value)
        for key, value in model.state_dict().items()
        if key in source_keys
    }
    stage_counts = configure_fusion_parameters(
        model, "source_frozen", source_keys=source_keys
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    expected_trainable = {
        "block_gate",
        "block_projection.0.bias",
        "block_projection.0.weight",
        "block_projection.1.bias",
        "block_projection.1.weight",
    }
    if set(trainable_names) != expected_trainable:
        raise AssertionError(
            "Phase 10 corrected mode trainable set mismatch: "
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
    if optimizer_ids != trainable_ids:
        raise AssertionError("Phase 10 optimizer is not the exact complement")
    model.to(device)

    train_dataset = UniDataset(str(Path(args.fused_data_root) / "train_block"))
    valid_dataset = UniDataset(str(Path(args.fused_data_root) / "valid_block"))
    expected_train = _expected_counts(
        train_dataset, make_protein_only_item, args.max_items
    )
    expected_valid = _expected_counts(
        valid_dataset, make_protein_only_item, args.max_items
    )
    diagnostic_batch = _diagnostic_batch(train_dataset, device, 0)

    checkpoint_out = args.checkpoint_out or str(
        Path(DEFAULT_PHASE10_ROOT) / "checkpoints/anew_block_pvb_posterior_best.ckpt"
    )
    last_checkpoint = args.last_checkpoint or str(
        Path(checkpoint_out).with_name("last.ckpt")
    )
    history = []
    global_step = 0
    completed_steps = 0
    train_oversized = 0
    start = time.perf_counter()
    best_valid_rec_total: float | None = None
    no_improvement = 0
    stop_reason = "max_epochs"
    eval_args = _eval_args(args)

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
            if not all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                raise FloatingPointError("non-finite Phase 10 training gradient")
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
            optimizer.step()
            _accumulate(epoch_acc, metrics, int(batch["x0"].shape[0]), record)
            global_step += 1
            completed_steps += 1
            if completed_steps % 100 == 0:
                print(
                    f"phase10 train epoch={epoch} steps={completed_steps} "
                    f"loss={metrics['loss']:.6f} atoms={epoch_acc['atom_count']}",
                    flush=True,
                )
        if epoch_acc["batch_count"] == 0:
            raise RuntimeError("no Phase 10 training batches were processed")
        if not args.max_steps and train_items_seen != expected_train["items"]:
            raise AssertionError(
                f"train traversal incomplete: {train_items_seen}/{expected_train['items']} items"
            )
        train_summary = _add_rec_total(_summarize(epoch_acc))

        valid = _evaluate_model(
            model,
            mode="fused",
            split="valid",
            args=eval_args,
            device=device,
        )
        valid_source = valid["pdbind_protein_only"]
        valid = _add_rec_total(valid_source)
        valid_aggregate = valid["aggregate"]
        if (
            valid_aggregate["atom_count"] != expected_valid["atoms"]
            or valid_aggregate["batch_count"] <= 0
        ):
            raise AssertionError(
                "valid traversal mismatch: "
                f"atoms={valid_aggregate['atom_count']} expected={expected_valid['atoms']}"
            )
        valid_rec_total = float(
            valid_aggregate["rec_total"]["batch_mean"]["mean"]
        )
        diagnostic = _epoch_diagnostics(
            model,
            diagnostic_batch,
            seed=args.seed + 2_000_003 * (epoch + 1),
            epoch=epoch,
            global_step=global_step,
        )
        improved = (
            best_valid_rec_total is None
            or valid_rec_total < best_valid_rec_total - 1.0e-12
        )
        if improved:
            best_valid_rec_total = valid_rec_total
            no_improvement = 0
            _save_phase10_checkpoint(
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
            )
        else:
            no_improvement += 1
        _save_phase10_checkpoint(
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
                "gradient_norms": fusion_gradient_norms(model),
                "train_items_seen": train_items_seen,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )
        print(
            f"phase10 valid epoch={epoch} rec_total={valid_rec_total:.6f} "
            f"best={best_valid_rec_total:.6f} improved={improved}",
            flush=True,
        )
        if args.max_steps and completed_steps >= args.max_steps:
            stop_reason = "max_steps"
            break
        if epoch + 1 >= args.min_epochs and no_improvement >= args.patience:
            stop_reason = "valid_rec_total_patience"
            break

    after = {
        key: _tensor_sha256(value)
        for key, value in model.state_dict().items()
        if key in source_keys
    }
    mismatches = sorted(key for key in source_keys if before.get(key) != after.get(key))
    best_hash = _file_sha256(checkpoint_out) if Path(checkpoint_out).exists() else None
    last_hash = _file_sha256(last_checkpoint) if Path(last_checkpoint).exists() else None
    result = {
        "mode": "phase10_train_corrected",
        "fusion_mode": "anew_block_pvb_posterior",
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
        "train_expected": expected_train,
        "valid_expected": expected_valid,
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
        "optimizer_is_exact_trainable_complement": optimizer_ids == trainable_ids,
        "source_checkpoint_unchanged": not mismatches,
        "source_checksum_mismatches": mismatches,
        "best_valid_rec_total": best_valid_rec_total,
        "checkpoint_out": checkpoint_out,
        "checkpoint_out_sha256": best_hash,
        "last_checkpoint": last_checkpoint,
        "last_checkpoint_sha256": last_hash,
        "source_reports": source_reports,
        "history": history,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if mismatches:
        raise AssertionError(f"source checkpoint tensors changed: {mismatches}")
    if best_valid_rec_total is None:
        raise AssertionError("Phase 10 did not produce a validation checkpoint")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
