"""Run the locked Phase 13 ablation evaluation on paired protein views.

This evaluator constructs every model before loading its explicitly scoped
checkpoint roles. It evaluates the PVB ``off`` baseline and selected Phase
13 adapter checkpoints on identical valid/test traversal and seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.phase10_paired_eval import _add_rec_total, _assert_same_view
from scripts.phase11_paired_eval import _load_phase11_adapter, _load_shared_pvb_full
from scripts.phase9_train_eval import (
    DEFAULT_FUSED_ROOT,
    _build_pvb_model,
    _device,
    _evaluate_model,
    _load_full_state,
    _seed,
)
from scripts.profile_training_paths import build_model
from utils.checkpoint import _state_dict_from_payload
from utils.phase13_ablation import get_phase13_variant


DEFAULT_PVB_STATE = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/"
    "checkpoints/pvb_state_dict.pt"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fused-data-root", default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--pvb-checkpoint", default=DEFAULT_PVB_STATE)
    parser.add_argument(
        "--variant-checkpoint",
        action="append",
        required=True,
        metavar="VARIANT=CHECKPOINT",
    )
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=[20260810, 20260811, 20260812]
    )
    parser.add_argument("--batch-budget", type=int, default=4_000_000)
    parser.add_argument("--max-atoms-per-batch", type=int, default=4096)
    parser.add_argument("--max-items-per-batch", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    return parser


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: str | Path) -> dict[str, str]:
    target = Path(path)
    return {"path": str(target), "sha256": _sha256(target)}


def _parse_variant_specs(values: list[str]) -> list[tuple[str, str, str]]:
    parsed = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"variant checkpoint must be VARIANT=PATH, got {value!r}")
        name, path = value.split("=", 1)
        spec = get_phase13_variant(name)
        if spec.adapter_variant in seen:
            raise ValueError(f"duplicate Phase 13 adapter variant: {spec.adapter_variant}")
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        seen.add(spec.adapter_variant)
        parsed.append((spec.name, spec.adapter_variant, path))
    return parsed


def _load_locked_adapter(model: torch.nn.Module, checkpoint: str, adapter_variant: str) -> dict:
    checkpoint_path = Path(checkpoint)
    lock_path = checkpoint_path.with_name(
        checkpoint_path.stem.replace("_best", "") + ".lock.json"
    )
    if not lock_path.is_file():
        raise FileNotFoundError(f"missing lock manifest for {checkpoint}: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked" or lock.get("test_evaluated"):
        raise RuntimeError(f"checkpoint lock is not available for one-time evaluation: {lock_path}")
    if str(Path(lock.get("checkpoint", ""))) != str(checkpoint_path):
        raise RuntimeError(f"lock/checkpoint path mismatch: {lock_path}")
    if lock.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise RuntimeError(f"lock/checkpoint SHA256 mismatch: {lock_path}")
    protocol_variant = lock.get("protocol", {}).get("shared_hblock_variant")
    if protocol_variant != adapter_variant:
        raise RuntimeError(
            f"lock variant mismatch for {checkpoint}: {protocol_variant!r} != {adapter_variant!r}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _state_dict_from_payload(payload)
    report = _load_phase11_adapter(model, str(checkpoint_path))
    return {
        "lock_path": str(lock_path),
        "lock_sha256": _sha256(lock_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_kind": payload.get("checkpoint_kind") if isinstance(payload, Mapping) else None,
        "adapter_coverage": report,
        "lock": lock,
        "state_key_count": len(state),
    }


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


def main() -> None:
    args = _parser().parse_args()
    if args.splits != ["valid", "test"]:
        raise ValueError("Phase 13 paired evaluation must run valid then test exactly once")
    variants = _parse_variant_specs(args.variant_checkpoint)
    device = _device(args.device)
    eval_args = _eval_args(args)
    models = {}
    load_reports = {}

    pvb = _build_pvb_model()
    load_reports["pvb_off"] = _load_full_state(pvb, args.pvb_checkpoint)
    models["pvb_off"] = (pvb, "pvb_protein")
    for public_name, adapter_variant, checkpoint in variants:
        model = build_model(
            "pvb_shared_hblock",
            shared_hblock_variant=adapter_variant,
            shared_hblock_seed=args.eval_seeds[0],
        )
        load_reports[public_name] = {
            "pvb_full": _load_shared_pvb_full(model, args.pvb_checkpoint),
            "adapter": _load_locked_adapter(model, checkpoint, adapter_variant),
        }
        models[public_name] = (model, "fused")

    result = {
        "format_version": 1,
        "mode": "phase13_paired_eval",
        "seeds": list(args.eval_seeds),
        "data_root": args.fused_data_root,
        "splits": list(args.splits),
        "models": {},
        "load_reports": load_reports,
        "artifacts": {"pvb_checkpoint": _artifact(args.pvb_checkpoint)},
        "variants": [
            {
                "public_name": public_name,
                "adapter_variant": adapter_variant,
                "checkpoint": _artifact(checkpoint),
            }
            for public_name, adapter_variant, checkpoint in variants
        ],
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
