"""Run one corrected-mode step and persist Phase 10 diagnostics.

This thin driver reuses the current profiler model builder, checkpoint loader,
collation, and source-frozen optimizer helpers. It is intended as the fixed
real-batch diagnostic primitive that a multi-epoch Phase 10 runner can call at
each epoch boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data.collate import collate_fn
from data.mmap_dataset import UniDataset
from data.protein_view import make_protein_only_item
from scripts.profile_training_paths import _load_roles, build_model
from utils.fusion_training import configure_fusion_parameters, fusion_parameter_groups
from utils.phase10_diagnostics import collect_phase10_diagnostics


DEFAULT_PVB_CHECKPOINT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/"
    "pvb_state_dict.pt"
)
DEFAULT_ANEW_CHECKPOINT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/"
    "legacy_fused_state_dict.pt"
)


def _seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _checksums(model: torch.nn.Module, keys) -> dict[str, str]:
    state = model.state_dict()
    return {
        key: _tensor_sha256(state[key])
        for key in sorted(keys)
        if key in state
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pvb-checkpoint", default=DEFAULT_PVB_CHECKPOINT)
    parser.add_argument("--anew-checkpoint", default=DEFAULT_ANEW_CHECKPOINT)
    parser.add_argument("--pvb-role", choices=("pvb", "pvb_full"), default="pvb_full")
    parser.add_argument(
        "--fusion-mode",
        choices=("anew_block", "anew_block_pvb_posterior"),
        default="anew_block_pvb_posterior",
    )
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--global-step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--projector-lr", type=float, default=1.0e-3)
    parser.add_argument("--pvb-lr", type=float, default=1.0e-3)
    parser.add_argument("--anew-lr", type=float, default=1.0e-5)
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="backpropagate for gradient diagnostics but do not update the adapter",
    )
    parser.add_argument("--output", required=True)
    return parser


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        if device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        torch.cuda.set_device(device)

    _seed(args.seed)
    dataset = UniDataset(str(Path(args.dataset_root) / f"{args.split}_block"))
    item = make_protein_only_item(dataset[args.record_index])
    batch = collate_fn([[item]])
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    model = build_model(args.fusion_mode)
    reports = _load_roles(
        model,
        args.pvb_checkpoint,
        args.anew_checkpoint,
        args.pvb_role,
    )
    source_keys = set().union(*(report.matched_keys for report in reports.values()))
    model.to(device)
    stage_counts = configure_fusion_parameters(
        model, "source_frozen", source_keys=source_keys
    )
    optimizer = torch.optim.Adam(
        fusion_parameter_groups(
            model,
            pvb_lr=args.pvb_lr,
            anew_lr=args.anew_lr,
            projector_lr=args.projector_lr,
        )
    )
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
        raise AssertionError("diagnostic optimizer is not exact source-frozen complement")

    before = _checksums(model, source_keys)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _seed(args.seed + 1)
    loss, parts = model._train(batch, mode="pretrain")
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("non-finite Phase 10 diagnostic loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"non-finite gradient in diagnostic step: {name}")

    diagnostics = collect_phase10_diagnostics(
        model,
        batch,
        loss=loss,
        parts=parts,
        epoch=args.epoch,
        global_step=args.global_step,
    )
    if not args.no_update:
        optimizer.step()
    after = _checksums(model, source_keys)

    result = {
        "mode": "phase10_diagnostics",
        "device": str(device),
        "dataset_root": args.dataset_root,
        "split": args.split,
        "record_index": args.record_index,
        "atoms": int(batch["x0"].shape[0]),
        "blocks": int(batch["block_type"].shape[0]),
        "bonds": int(batch["bond_index"].shape[1]),
        "epoch": args.epoch,
        "global_step": args.global_step,
        "seed": args.seed,
        "fusion_mode": args.fusion_mode,
        "pvb_checkpoint_role": args.pvb_role,
        "checkpoint_reports": {
            role: {
                "coverage": report.coverage,
                "matched": len(report.matched_keys),
                "expected": report.expected_key_count,
                "missing": len(report.missing_keys),
                "unexpected": len(report.unexpected_keys),
                "shape_mismatches": len(report.shape_mismatches),
            }
            for role, report in sorted(reports.items())
        },
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
        "trainable_parameter_names": [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        "optimizer_is_exact_trainable_complement": optimizer_ids == trainable_ids,
        "stage_counts": stage_counts,
        "source_checkpoint_unchanged": before == after,
        "source_checksum_mismatches": sorted(
            key for key in source_keys if before.get(key) != after.get(key)
        ),
        "updated_adapter": not args.no_update,
        "diagnostics": diagnostics,
    }
    if result["source_checksum_mismatches"]:
        raise AssertionError("source tensors changed during diagnostic step")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
