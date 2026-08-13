"""Run the locked Phase 10 paired protein-only valid/test evaluation once."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch

from scripts.phase9_train_eval import (
    DEFAULT_FUSED_ROOT,
    _build_pvb_model,
    _device,
    _evaluate_model,
    _load_full_state,
    _seed,
)
from scripts.profile_training_paths import build_model
from utils.checkpoint import load_resume_checkpoint


DEFAULT_PVB_STATE = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/"
    "checkpoints/pvb_state_dict.pt"
)
DEFAULT_LEGACY_CHECKPOINT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/"
    "checkpoints/source_frozen_epoch1_best.ckpt"
)
DEFAULT_PHASE10_CHECKPOINT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/"
    "checkpoints/anew_block_pvb_posterior_best.ckpt"
)
DEFAULT_LOCK = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10/"
    "checkpoints/phase10_best.lock.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fused-data-root", default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--pvb-checkpoint", default=DEFAULT_PVB_STATE)
    parser.add_argument("--legacy-checkpoint", default=DEFAULT_LEGACY_CHECKPOINT)
    parser.add_argument("--phase10-checkpoint", default=DEFAULT_PHASE10_CHECKPOINT)
    parser.add_argument("--lock", default=DEFAULT_LOCK)
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=[20260810, 20260811, 20260812]
    )
    parser.add_argument("--batch-budget", type=int, default=4_000_000)
    parser.add_argument("--max-atoms-per-batch", type=int, default=4096)
    parser.add_argument("--max-items-per-batch", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    return parser


def _add_rec_total(summary: dict) -> dict:
    for seed_result in summary["per_seed"]:
        for weighting in ("batch_mean", "atom_weighted_mean"):
            seed_result.setdefault("rec_total", {})[weighting] = float(
                seed_result["rec_vel"][weighting]
                + seed_result["rec_drf"][weighting]
            )
    for weighting in ("batch_mean", "atom_weighted_mean"):
        values = [item["rec_total"][weighting] for item in summary["per_seed"]]
        summary["aggregate"].setdefault("rec_total", {})[weighting] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=0)) if values else None,
        }
    return summary


def _eval_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        eval_seeds=list(args.eval_seeds),
        original_data_root="/data/pvb_cross_dataset_20260810/blocks",
        fused_data_root=args.fused_data_root,
        batch_budget=args.batch_budget,
        max_atoms_per_batch=args.max_atoms_per_batch,
        max_items_per_batch=args.max_items_per_batch,
        max_items=args.max_items,
    )


def _load_resume(model: torch.nn.Module, path: str) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        report, metadata = load_resume_checkpoint(model, path, min_coverage=1.0)
    metadata_summary = {
        key: metadata[key]
        for key in ("epoch", "global_step", "valid_global_step", "best_metric", "patience")
        if key in metadata and not torch.is_tensor(metadata[key])
    }
    for key in (
        "optimizer_state_dict",
        "warmup_scheduler_state_dict",
        "scheduler_state_dict",
        "ema_model_state_dict",
        "scaler_state_dict",
    ):
        if key in metadata:
            metadata_summary["has_" + key] = True
    return {
        "coverage": report.coverage,
        "matched": len(report.matched_keys),
        "expected": report.expected_key_count,
        "missing": len(report.missing_keys),
        "unexpected": len(report.unexpected_keys),
        "shape_mismatches": len(report.shape_mismatches),
        "metadata": metadata_summary,
    }


def _assert_same_view(results: Mapping[str, dict], split: str) -> dict:
    views = {}
    for name, report in results.items():
        aggregate = report[split]["aggregate"]
        views[name] = {
            key: aggregate[key]
            for key in ("batch_count", "atom_count", "oversized_count")
        }
    if len({tuple(view.values()) for view in views.values()}) != 1:
        raise AssertionError(f"paired {split} views differ: {views}")
    return views


def main() -> None:
    args = _parser().parse_args()
    if args.splits != ["valid", "test"]:
        raise ValueError("Phase 10 paired evaluation must run valid then test exactly once")
    lock_path = Path(args.lock)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked" or lock.get("test_evaluated"):
        raise RuntimeError("paired evaluation requires an unlocked best-checkpoint manifest")
    phase10_path = Path(args.phase10_checkpoint)
    if str(phase10_path) != lock["checkpoint"]:
        raise RuntimeError("Phase 10 checkpoint differs from the locked checkpoint")
    if _sha256(phase10_path) != lock["checkpoint_sha256"]:
        raise RuntimeError("locked Phase 10 checkpoint SHA256 mismatch")

    device = _device(args.device)
    eval_args = _eval_args(args)
    models = {}
    load_reports = {}

    pvb = _build_pvb_model()
    load_reports["pvb_off"] = _load_full_state(pvb, args.pvb_checkpoint)
    models["pvb_off"] = (pvb, "pvb_protein")

    legacy = build_model("anew_block")
    load_reports["phase9_legacy"] = _load_resume(legacy, args.legacy_checkpoint)
    models["phase9_legacy"] = (legacy, "fused")

    corrected = build_model("anew_block_pvb_posterior")
    load_reports["phase10_corrected"] = _load_resume(corrected, args.phase10_checkpoint)
    models["phase10_corrected"] = (corrected, "fused")

    result = {
        "format_version": 1,
        "mode": "phase10_paired_eval",
        "seeds": list(args.eval_seeds),
        "data_root": args.fused_data_root,
        "splits": list(args.splits),
        "models": {},
        "load_reports": load_reports,
        "artifacts": {
            "pvb_checkpoint": {
                "path": args.pvb_checkpoint,
                "sha256": _sha256(Path(args.pvb_checkpoint)),
            },
            "legacy_checkpoint": {
                "path": args.legacy_checkpoint,
                "sha256": _sha256(Path(args.legacy_checkpoint)),
            },
            "phase10_checkpoint": {
                "path": args.phase10_checkpoint,
                "sha256": _sha256(phase10_path),
            },
            "lock": {"path": str(lock_path), "sha256": _sha256(lock_path)},
        },
    }

    for split in args.splits:
        for name, (model, mode) in models.items():
            model.to(device)
            _seed(args.eval_seeds[0])
            evaluated = _evaluate_model(
                model, mode=mode, split=split, args=eval_args, device=device
            )["pdbind_protein_only"]
            result["models"].setdefault(name, {})[split] = _add_rec_total(evaluated)
            model.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _assert_same_view(result["models"], split)

    result["paired_views"] = {
        split: _assert_same_view(result["models"], split) for split in args.splits
    }
    result["test_evaluated"] = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
