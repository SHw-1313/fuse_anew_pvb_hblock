"""Exact source-frozen training and full-split evaluation for Phase 9.

This runner deliberately avoids DynamicBatchWrapper's historical silent
oversized-record drop. It materializes each item on CPU, builds an exact
padded-attention budget, yields oversized records as explicit singleton
batches, and records every batch in the result. It supports train_fused,
eval_pvb, and eval_fused modes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from data.collate import collate_fn
from data.mmap_dataset import UniDataset
from data.protein_view import make_protein_only_item
from module import dyVAE
from scripts.profile_training_paths import _load_roles, build_model
from utils.checkpoint import (
    _state_dict_from_payload,
    load_resume_checkpoint,
)
from utils.fusion_training import (
    configure_fusion_parameters,
    fusion_gradient_norms,
    fusion_parameter_groups,
)


DEFAULT_PVB_STATE = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/"
    "checkpoints/pvb_state_dict.pt"
)
DEFAULT_ANEW_STATE = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/"
    "checkpoints/legacy_fused_state_dict.pt"
)
DEFAULT_ORIGINAL_ROOT = "/data/pvb_cross_dataset_20260810/blocks"
DEFAULT_FUSED_ROOT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/data/"
    "pdbind_protein_only"
)
METRIC_NAMES = ("loss", "kl", "rec_vel", "rec_drf")


@dataclass
class BatchRecord:
    indices: list[int]
    items: list[dict]
    padded_cost: int
    oversized: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("train_fused", "eval_pvb", "eval_pvb_protein", "eval_fused"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=[20260810, 20260811, 20260812],
    )
    parser.add_argument("--original-data-root", default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--fused-data-root", default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--pvb-checkpoint", default=DEFAULT_PVB_STATE)
    parser.add_argument("--anew-checkpoint", default=DEFAULT_ANEW_STATE)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-budget", type=int, default=4_000_000)
    parser.add_argument("--max-atoms-per-batch", type=int, default=4096)
    parser.add_argument("--max-items-per-batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--pvb-lr", type=float, default=1.0e-3)
    parser.add_argument("--anew-lr", type=float, default=1.0e-5)
    parser.add_argument("--projector-lr", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-seed", type=int, default=20260810)
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Debug limit; zero means the complete requested split.",
    )
    parser.add_argument("--split", default=None)
    return parser


def _seed(seed: int) -> None:
    # NumPy's legacy global RNG accepts only uint32 seeds; keep the
    # deterministic per-batch schedule valid even on long full-split runs.
    seed = int(seed)
    np_seed = seed % (2**32 - 1)
    torch.manual_seed(seed)
    np.random.seed(np_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        if device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        torch.cuda.set_device(device)
    return device


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _checksums(model: torch.nn.Module, keys: Iterable[str]) -> dict[str, str]:
    state = model.state_dict()
    return {
        key: _tensor_sha256(state[key])
        for key in sorted(keys)
        if key in state
    }


def _build_pvb_model() -> dyVAE:
    return dyVAE(
        256,
        512,
        32,
        8,
        8,
        cutoff_lower=0.0,
        cutoff_upper=10.0,
        cutoff_H=3.5,
        k_neighbors=32,
        coord_prior_var=0.5,
        sigma=0.2,
        additional_noise_scale=0.2,
        kl_weight=0.8,
        re_weight=1.0,
        using_ode=False,
        backbone="torchmdnet",
        fusion_mode="off",
    )


def _load_full_state(model: torch.nn.Module, path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = _state_dict_from_payload(payload)
    target = model.state_dict()
    missing = sorted(set(target) - set(state))
    unexpected = sorted(set(state) - set(target))
    shape_mismatches = sorted(
        key
        for key in set(target).intersection(state)
        if tuple(target[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "full PVB checkpoint does not match the constructed model: "
            f"missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={shape_mismatches}"
        )
    model.load_state_dict(state, strict=True)
    return {
        "path": str(path),
        "matched": len(state),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
    }


def _load_fused_source(model: dyVAE, args: argparse.Namespace):
    if args.resume_checkpoint:
        with contextlib.redirect_stdout(io.StringIO()):
            report, metadata = load_resume_checkpoint(
                model, args.resume_checkpoint, min_coverage=1.0
            )
        return {
            "mode": "resume",
            "reports": {
                "resume": {
                    "coverage": report.coverage,
                    "matched": len(report.matched_keys),
                    "missing": len(report.missing_keys),
                    "unexpected": len(report.unexpected_keys),
                    "shape_mismatches": len(report.shape_mismatches),
                }
            },
            "source_keys": set(),
            "resume_metadata": metadata,
        }

    with contextlib.redirect_stdout(io.StringIO()):
        reports = _load_roles(model, args.pvb_checkpoint, args.anew_checkpoint)
    source_keys = set().union(
        *(report.matched_keys for report in reports.values())
    )
    return {
        "mode": "source_roles",
        "reports": {
            role: {
                "coverage": report.coverage,
                "matched": len(report.matched_keys),
                "source_keys": len(report.source_keys),
                "missing": len(report.missing_keys),
                "unexpected": len(report.unexpected_keys),
                "shape_mismatches": len(report.shape_mismatches),
            }
            for role, report in reports.items()
        },
        "source_keys": source_keys,
        "resume_metadata": {},
    }


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _item_count(item: Mapping) -> int:
    count = len(item["atype"])
    if count <= 0:
        raise ValueError("empty records are not valid training/evaluation items")
    return count


def _exact_batches(
    dataset: UniDataset,
    transform: Callable[[dict], dict],
    *,
    budget: int,
    max_atoms: int,
    max_items: int,
    shuffle: bool,
    seed: int,
    max_items_total: int = 0,
) -> Iterator[BatchRecord]:
    indices = list(range(len(dataset)))
    if max_items_total:
        indices = indices[:max_items_total]
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)

    current_indices: list[int] = []
    current_items: list[dict] = []
    current_max = 0
    current_cost = 0

    def emit_current() -> BatchRecord | None:
        nonlocal current_indices, current_items, current_max, current_cost
        if not current_items:
            return None
        record = BatchRecord(
            indices=current_indices,
            items=current_items,
            padded_cost=current_cost,
            oversized=current_cost > budget,
        )
        current_indices = []
        current_items = []
        current_max = 0
        current_cost = 0
        return record

    for index in indices:
        item = transform(dataset[index])
        atoms = _item_count(item)
        singleton_cost = atoms * atoms
        if not current_items:
            current_indices = [index]
            current_items = [item]
            current_max = atoms
            current_cost = singleton_cost
            if atoms > max_atoms or singleton_cost > budget:
                record = emit_current()
                assert record is not None
                record.oversized = True
                yield record
            continue

        new_max = max(current_max, atoms)
        new_cost = (len(current_items) + 1) * new_max * new_max
        exceeds = (
            new_cost > budget
            or new_max > max_atoms
            or len(current_items) >= max_items
        )
        if exceeds:
            record = emit_current()
            assert record is not None
            yield record
            current_indices = [index]
            current_items = [item]
            current_max = atoms
            current_cost = singleton_cost
            if atoms > max_atoms or singleton_cost > budget:
                record = emit_current()
                assert record is not None
                record.oversized = True
                yield record
        else:
            current_indices.append(index)
            current_items.append(item)
            current_max = new_max
            current_cost = new_cost

    record = emit_current()
    if record is not None:
        yield record


def _collate(record: BatchRecord, device: torch.device) -> dict:
    return _move_batch(collate_fn([record.items]), device)


def _metric_values(loss, parts) -> dict[str, float]:
    values = {
        "loss": loss,
        "kl": parts[0],
        "rec_vel": parts[1],
        "rec_drf": parts[2],
    }
    result = {}
    for name, value in values.items():
        scalar = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        if not math.isfinite(scalar):
            raise FloatingPointError(f"non-finite {name}: {scalar}")
        result[name] = scalar
    return result


def _empty_accumulator() -> dict:
    return {
        "batch_count": 0,
        "atom_count": 0,
        "padded_cost": 0,
        "oversized_count": 0,
        "values": {name: [] for name in METRIC_NAMES},
    }


def _accumulate(acc: dict, metrics: dict[str, float], atoms: int, record: BatchRecord) -> None:
    acc["batch_count"] += 1
    acc["atom_count"] += atoms
    acc["padded_cost"] += record.padded_cost
    acc["oversized_count"] += int(record.oversized)
    for name, value in metrics.items():
        acc["values"][name].append((value, atoms))


def _summarize(acc: dict) -> dict:
    summary = {
        "batch_count": acc["batch_count"],
        "atom_count": acc["atom_count"],
        "padded_cost": acc["padded_cost"],
        "oversized_count": acc["oversized_count"],
    }
    for name in METRIC_NAMES:
        values = acc["values"][name]
        if not values:
            summary[name] = {"batch_mean": None, "atom_weighted_mean": None}
            continue
        raw = np.asarray([value for value, _ in values], dtype=np.float64)
        weights = np.asarray([atoms for _, atoms in values], dtype=np.float64)
        summary[name] = {
            "batch_mean": float(raw.mean()),
            "atom_weighted_mean": float(np.average(raw, weights=weights)),
        }
    return summary


def _evaluate_split(
    model: dyVAE,
    dataset: UniDataset,
    transform: Callable[[dict], dict],
    *,
    device: torch.device,
    budget: int,
    max_atoms: int,
    max_items: int,
    seeds: Sequence[int],
    max_items_total: int = 0,
) -> dict:
    per_seed = []
    model.eval()
    with torch.no_grad():
        for seed in seeds:
            _seed(seed)
            acc = _empty_accumulator()
            start = time.perf_counter()
            for batch_number, record in enumerate(
                _exact_batches(
                    dataset,
                    transform,
                    budget=budget,
                    max_atoms=max_atoms,
                    max_items=max_items,
                    shuffle=False,
                    seed=seed,
                    max_items_total=max_items_total,
                )
            ):
                batch = _collate(record, device)
                _seed(seed + 1_000_003 * (batch_number + 1))
                loss, parts = model._train(batch, mode="pretrain")
                metrics = _metric_values(loss, parts)
                _accumulate(acc, metrics, int(batch["x0"].shape[0]), record)
                if (batch_number + 1) % 1000 == 0:
                    print(
                        f"eval progress batches={batch_number + 1} "
                        f"atoms={acc['atom_count']}",
                        flush=True,
                    )
            result = _summarize(acc)
            result["seed"] = int(seed)
            result["elapsed_seconds"] = time.perf_counter() - start
            per_seed.append(result)

    aggregate = {
        "batch_count": int(per_seed[0]["batch_count"]) if per_seed else 0,
        "atom_count": int(per_seed[0]["atom_count"]) if per_seed else 0,
        "oversized_count": int(per_seed[0]["oversized_count"]) if per_seed else 0,
    }
    for name in METRIC_NAMES:
        for weighting in ("batch_mean", "atom_weighted_mean"):
            values = [
                result[name][weighting]
                for result in per_seed
                if result[name][weighting] is not None
            ]
            aggregate.setdefault(name, {})[weighting] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=0)) if values else None,
            }
    return {"per_seed": per_seed, "aggregate": aggregate}


def _dataset_for(
    mode: str, split: str, original_root: str, fused_root: str
) -> tuple[UniDataset, Callable[[dict], dict], str]:
    if mode in {"pvb", "pvb_protein"}:
        raise AssertionError("use _datasets_for_eval for pvb mode")
    path = Path(fused_root) / f"{split}_block"
    return UniDataset(str(path)), make_protein_only_item, "pdbind_protein_only"


def _datasets_for_eval(
    mode: str, split: str, original_root: str, fused_root: str
) -> list[tuple[str, UniDataset, Callable[[dict], dict]]]:
    if mode == "pvb":
        return [
            (
                source,
                UniDataset(str(Path(original_root) / source / f"{split}_block")),
                lambda item: item,
            )
            for source in ("pcqm4mv2", "ani1x", "pdbbind")
        ]
    if mode == "pvb_protein":
        return [
            (
                "pdbind_protein_only",
                UniDataset(str(Path(fused_root) / f"{split}_block")),
                lambda item: item,
            )
        ]
    dataset, transform, name = _dataset_for(
        mode, split, original_root, fused_root
    )
    return [(name, dataset, transform)]


def _evaluate_model(
    model: dyVAE,
    *,
    mode: str,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    results = {}
    for name, dataset, transform in _datasets_for_eval(
        mode, split, args.original_data_root, args.fused_data_root
    ):
        print(
            f"evaluate mode={mode} split={split} source={name} "
            f"items={len(dataset)} max_items={args.max_items}",
            flush=True,
        )
        results[name] = _evaluate_split(
            model,
            dataset,
            transform,
            device=device,
            budget=args.batch_budget,
            max_atoms=args.max_atoms_per_batch,
            max_items=args.max_items_per_batch,
            seeds=args.eval_seeds,
            max_items_total=args.max_items,
        )
        print(
            json.dumps(
                {name: results[name]["aggregate"]}, sort_keys=True
            ),
            flush=True,
        )
    return results


def _save_checkpoint(
    path: str,
    model: dyVAE,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    valid: dict | None,
    source_checksums: dict[str, str],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "valid": valid,
        "source_checksums_before": source_checksums,
        "source_checksums_after": _checksums(model, source_checksums),
        "fusion_stage": "source_frozen",
    }
    torch.save(payload, target)


def _train(args: argparse.Namespace) -> dict:
    device = _device(args.device)
    if not args.anew_checkpoint:
        raise ValueError("--anew-checkpoint is required for train_fused")
    if not args.checkpoint_out:
        raise ValueError("--checkpoint-out is required for train_fused")

    model = build_model()
    load_info = _load_fused_source(model, args)
    source_keys = set(load_info["source_keys"])
    if not source_keys:
        raise RuntimeError("source-frozen training needs source-role checkpoint keys")
    before = _checksums(model, source_keys)
    info = configure_fusion_parameters(
        model, "source_frozen", source_keys=source_keys
    )
    groups = fusion_parameter_groups(
        model, args.pvb_lr, args.anew_lr, args.projector_lr
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
    if trainable_ids != optimizer_ids:
        raise AssertionError("optimizer is not the exact source-frozen complement")

    model.to(device)
    train_dataset = UniDataset(str(Path(args.fused_data_root) / "train_block"))
    history = []
    global_step = 0
    completed_steps = 0
    train_oversized = 0
    start = time.perf_counter()
    best_metric = None
    best_path = str(Path(args.checkpoint_out))
    last_path = str(Path(args.checkpoint_out).with_name("last.ckpt"))

    for epoch in range(max(1, args.epochs)):
        model.train()
        epoch_acc = _empty_accumulator()
        epoch_start = time.perf_counter()
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
                raise FloatingPointError("non-finite gradient in source-frozen training")
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
            optimizer.step()
            _accumulate(epoch_acc, metrics, int(batch["x0"].shape[0]), record)
            global_step += 1
            if completed_steps % 100 == 0:
                print(
                    f"train progress epoch={epoch} steps={completed_steps} "
                    f"loss={metrics['loss']:.6f} atoms={epoch_acc['atom_count']}",
                    flush=True,
                )
            completed_steps += 1

        if epoch_acc["batch_count"] == 0:
            raise RuntimeError("no training batches were processed")
        train_summary = _summarize(epoch_acc)
        valid = _evaluate_model(
            model,
            mode="fused",
            split="valid",
            args=argparse.Namespace(
                **{
                    **vars(args),
                    "eval_seeds": [args.validation_seed],
                    "max_items": args.max_items,
                }
            ),
            device=device,
        )
        valid_summary = valid["pdbind_protein_only"]["aggregate"]
        metric = valid_summary["loss"]["batch_mean"]["mean"]
        if best_metric is None or metric < best_metric:
            best_metric = metric
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                valid=valid,
                source_checksums=before,
            )
        _save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch=epoch,
            global_step=global_step,
            valid=valid,
            source_checksums=before,
        )
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_summary,
                "valid": valid,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "gradient_norms": fusion_gradient_norms(model),
            }
        )
        if args.max_steps and completed_steps >= args.max_steps:
            break

    after = _checksums(model, source_keys)
    mismatches = sorted(
        key for key in source_keys if before.get(key) != after.get(key)
    )
    result = {
        "mode": "train_fused",
        "device": str(device),
        "data_root": args.fused_data_root,
        "epochs_requested": max(1, args.epochs),
        "epochs_completed": len(history),
        "steps_completed": completed_steps,
        "batch_budget": args.batch_budget,
        "max_atoms_per_batch": args.max_atoms_per_batch,
        "max_items_per_batch": args.max_items_per_batch,
        "train_oversized_batches": train_oversized,
        "fusion_parameter_counts": info,
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
        "optimizer_is_exact_trainable_complement": optimizer_ids == trainable_ids,
        "source_checkpoint_unchanged": not mismatches,
        "source_checksum_mismatches": mismatches,
        "checkpoint_out": best_path,
        "last_checkpoint": last_path,
        "best_valid_loss": best_metric,
        "history": history,
        "load": {
            key: value
            for key, value in load_info.items()
            if key not in {"source_keys", "resume_metadata"}
        },
        "elapsed_seconds": time.perf_counter() - start,
    }
    if mismatches:
        raise AssertionError(f"source checkpoint tensors changed: {mismatches}")
    return result


def _eval(args: argparse.Namespace) -> dict:
    device = _device(args.device)
    if args.mode in {"eval_pvb", "eval_pvb_protein"}:
        model = _build_pvb_model()
        checkpoint = _load_full_state(model, args.pvb_checkpoint)
        mode = "pvb" if args.mode == "eval_pvb" else "pvb_protein"
    else:
        model = build_model()
        load_info = _load_fused_source(model, args)
        checkpoint = {
            "mode": load_info["mode"],
            "reports": load_info["reports"],
        }
        mode = "fused"
    model.to(device)
    splits = [args.split] if args.split else ["valid", "test"]
    result = {
        "mode": args.mode,
        "device": str(device),
        "checkpoint": checkpoint,
        "original_data_root": args.original_data_root,
        "fused_data_root": args.fused_data_root,
        "batch_budget": args.batch_budget,
        "max_atoms_per_batch": args.max_atoms_per_batch,
        "max_items_per_batch": args.max_items_per_batch,
        "max_items": args.max_items,
        "splits": {},
    }
    for split in splits:
        result["splits"][split] = _evaluate_model(
            model, mode=mode, split=split, args=args, device=device
        )
    return result


def main() -> None:
    args = _parser().parse_args()
    _seed(args.seed)
    if args.mode == "train_fused":
        result = _train(args)
    else:
        result = _eval(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
