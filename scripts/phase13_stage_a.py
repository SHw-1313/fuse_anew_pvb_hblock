"""Stage-A dependency diagnostic for the locked Phase 11A adapter.

The same locked real-adapter weights are evaluated with real, shuffled,
constant, and atom-no-pool inputs on valid only. This is a dependency probe,
not a retrained performance comparison and it does not consume test.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.phase10_paired_eval import _add_rec_total
from scripts.phase11_paired_eval import _load_phase11_adapter, _load_shared_pvb_full
from scripts.phase9_train_eval import (
    DEFAULT_FUSED_ROOT,
    _build_pvb_model,
    _device,
    _evaluate_model,
    _seed,
)
from scripts.profile_training_paths import build_model

DEFAULT_PVB_STATE = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/"
    "checkpoints/pvb_state_dict.pt"
)
DEFAULT_ADAPTER = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/"
    "checkpoints/pvb_shared_hblock_best.ckpt"
)
DEFAULT_LOCK = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/"
    "checkpoints/phase11_shared_hblock_best.lock.json"
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fused-data-root", default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--pvb-checkpoint", default=DEFAULT_PVB_STATE)
    parser.add_argument("--adapter-checkpoint", default=DEFAULT_ADAPTER)
    parser.add_argument("--lock", default=DEFAULT_LOCK)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--batch-budget", type=int, default=4_000_000)
    parser.add_argument("--max-atoms-per-batch", type=int, default=4096)
    parser.add_argument("--max-items-per-batch", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=0)
    return parser


def _args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        eval_seeds=[args.seed],
        original_data_root="/data/pvb_cross_dataset_20260810/blocks",
        fused_data_root=args.fused_data_root,
        batch_budget=args.batch_budget,
        max_atoms_per_batch=args.max_atoms_per_batch,
        max_items_per_batch=args.max_items_per_batch,
        max_items=args.max_items,
    )


def main() -> None:
    args = _parser().parse_args()
    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    if lock.get("status") != "locked" or lock.get("test_evaluated"):
        raise RuntimeError("Stage A requires an unused locked Phase 11A checkpoint")
    if lock.get("checkpoint_sha256") != _sha256(args.adapter_checkpoint):
        raise RuntimeError("Phase 11A lock/checkpoint SHA256 mismatch")
    device = _device(args.device)
    eval_args = _args(args)
    result = {
        "format_version": 1,
        "mode": "phase13_stage_a_locked_adapter_control",
        "valid_only": True,
        "test_evaluated": False,
        "seed": args.seed,
        "data_root": args.fused_data_root,
        "pvb_checkpoint": {"path": args.pvb_checkpoint, "sha256": _sha256(args.pvb_checkpoint)},
        "adapter_checkpoint": {"path": args.adapter_checkpoint, "sha256": _sha256(args.adapter_checkpoint)},
        "lock": {"path": args.lock, "sha256": _sha256(args.lock)},
        "variants": {},
    }
    for variant in ("real", "shuffled", "constant", "atom_no_pool"):
        model = build_model(
            "pvb_shared_hblock",
            shared_hblock_variant=variant,
            shared_hblock_seed=args.seed,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            pvb_report = _load_shared_pvb_full(model, args.pvb_checkpoint)
            adapter_report = _load_phase11_adapter(model, args.adapter_checkpoint)
        model.to(device)
        _seed(args.seed)
        evaluated = _evaluate_model(
            model, mode="fused", split="valid", args=eval_args, device=device
        )["pdbind_protein_only"]
        result["variants"][variant] = {
            "pvb_full_coverage": pvb_report,
            "adapter_coverage": adapter_report,
            "metrics": _add_rec_total(evaluated),
            "shared_hblock_variant": variant,
        }
        model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
