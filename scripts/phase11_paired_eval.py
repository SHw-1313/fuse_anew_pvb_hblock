"""Run the locked Phase 11A paired protein-only valid/test evaluation once."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from collections.abc import Mapping

import torch
from pathlib import Path
from types import SimpleNamespace

from scripts.phase11_shared_train_eval import EXPECTED_SHARED_TRAINABLE
from scripts.phase10_paired_eval import (
    _add_rec_total,
    _assert_same_view,
    _eval_args,
    _load_resume,
    _sha256,
)
from scripts.phase9_train_eval import (
    DEFAULT_FUSED_ROOT,
    _build_pvb_model,
    _device,
    _evaluate_model,
    _load_full_state,
    _seed,
)
from scripts.profile_training_paths import build_model
from utils.checkpoint import _state_dict_from_payload, load_role_checkpoint


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
DEFAULT_PHASE11_CHECKPOINT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/"
    "checkpoints/pvb_shared_hblock_best.ckpt"
)
DEFAULT_PHASE11_LOCK = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/"
    "checkpoints/phase11_shared_hblock_best.lock.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fused-data-root", default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--pvb-checkpoint", default=DEFAULT_PVB_STATE)
    parser.add_argument("--legacy-checkpoint", default=DEFAULT_LEGACY_CHECKPOINT)
    parser.add_argument("--phase10-checkpoint", default=DEFAULT_PHASE10_CHECKPOINT)
    parser.add_argument("--phase11-checkpoint", default=DEFAULT_PHASE11_CHECKPOINT)
    parser.add_argument("--lock", default=DEFAULT_PHASE11_LOCK)
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=[20260810, 20260811, 20260812]
    )
    parser.add_argument("--batch-budget", type=int, default=4_000_000)
    parser.add_argument("--max-atoms-per-batch", type=int, default=4096)
    parser.add_argument("--max-items-per-batch", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    return parser


def _artifact(path: str | Path) -> dict:
    target = Path(path)
    return {"path": str(target), "sha256": _sha256(target)}


def _load_phase11_adapter(model: torch.nn.Module, path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = _state_dict_from_payload(payload)
    expected = set(EXPECTED_SHARED_TRAINABLE)
    actual = set(state)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    shape_mismatches = {}
    target = model.state_dict()
    for key in sorted(expected & actual):
        if tuple(state[key].shape) != tuple(target[key].shape):
            shape_mismatches[key] = {
                "source_shape": tuple(state[key].shape),
                "target_shape": tuple(target[key].shape),
            }
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Phase 11 adapter checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={shape_mismatches}"
        )
    model.load_state_dict(state, strict=False)
    return {
        "kind": payload.get("checkpoint_kind") if isinstance(payload, Mapping) else None,
        "coverage": 1.0,
        "matched": len(actual),
        "expected": len(expected),
        "missing": 0,
        "unexpected": 0,
        "shape_mismatches": 0,
        "source_state_external": True,
    }


def _load_shared_pvb_full(model: torch.nn.Module, path: str) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        report = load_role_checkpoint(model, path, "pvb_full", min_coverage=1.0)
    return {
        "coverage": report.coverage,
        "matched": len(report.matched_keys),
        "expected": report.expected_key_count,
        "missing": len(report.missing_keys),
        "unexpected": len(report.unexpected_keys),
        "shape_mismatches": len(report.shape_mismatches),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.splits != ["valid", "test"]:
        raise ValueError(
            "Phase 11 paired evaluation must run valid then test exactly once"
        )
    lock_path = Path(args.lock)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked" or lock.get("test_evaluated"):
        raise RuntimeError(
            "paired evaluation requires an unlocked Phase 11 best-checkpoint manifest"
        )
    if lock.get("checkpoint_kind") != "phase11_adapter_only":
        raise RuntimeError("Phase 11 lock must identify an adapter-only checkpoint")
    phase11_path = Path(args.phase11_checkpoint)
    if str(phase11_path) != lock["checkpoint"]:
        raise RuntimeError("Phase 11 checkpoint differs from the locked checkpoint")
    if _sha256(phase11_path) != lock["checkpoint_sha256"]:
        raise RuntimeError("locked Phase 11 checkpoint SHA256 mismatch")

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
    load_reports["phase10_corrected"] = _load_resume(
        corrected, args.phase10_checkpoint
    )
    models["phase10_corrected"] = (corrected, "fused")

    shared = build_model("pvb_shared_hblock")
    load_reports["phase11_shared_pvb_full"] = _load_shared_pvb_full(
        shared, args.pvb_checkpoint
    )
    load_reports["phase11_shared_adapter"] = _load_phase11_adapter(
        shared, args.phase11_checkpoint
    )
    models["phase11_shared"] = (shared, "fused")

    result = {
        "format_version": 1,
        "mode": "phase11_paired_eval",
        "seeds": list(args.eval_seeds),
        "data_root": args.fused_data_root,
        "splits": list(args.splits),
        "models": {},
        "load_reports": load_reports,
        "artifacts": {
            "pvb_checkpoint": _artifact(args.pvb_checkpoint),
            "legacy_checkpoint": _artifact(args.legacy_checkpoint),
            "phase10_checkpoint": _artifact(args.phase10_checkpoint),
            "phase11_checkpoint": _artifact(args.phase11_checkpoint),
            "phase11_lock": _artifact(lock_path),
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
                import torch

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
