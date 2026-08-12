"""Run a deterministic source-frozen overfit on one materialized protein batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from data.collate import collate_fn
from data.mmap_dataset import UniDataset
from data.protein_view import make_protein_only_item
from scripts.profile_training_paths import _load_roles, build_model
from utils.fusion_training import (
    configure_fusion_parameters,
    fusion_gradient_norms,
    fusion_parameter_groups,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pvb-checkpoint", required=True)
    parser.add_argument("--anew-checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--pvb-lr", type=float, default=1.0e-4)
    parser.add_argument("--anew-lr", type=float, default=1.0e-5)
    parser.add_argument("--projector-lr", type=float, default=1.0e-4)
    return parser


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _checksums(model: torch.nn.Module, keys) -> dict[str, str]:
    state = model.state_dict()
    return {key: _sha256(state[key]) for key in sorted(keys) if key in state}


def _finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        torch.cuda.set_device(device)

    _seed(args.seed)
    dataset = UniDataset(str(Path(args.dataset_root) / f"{args.split}_block"))
    item = make_protein_only_item(dataset[args.record_index])
    batch = collate_fn([[item]])
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    model = build_model()
    reports = _load_roles(model, args.pvb_checkpoint, args.anew_checkpoint)
    source_keys = set().union(*(report.matched_keys for report in reports.values()))
    source_checksums_before = _checksums(model, source_keys)
    configure_fusion_parameters(model, "source_frozen", source_keys=source_keys)
    groups = fusion_parameter_groups(
        model, args.pvb_lr, args.anew_lr, args.projector_lr
    )
    optimizer = torch.optim.Adam(groups)
    model.to(device).train()

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    losses = []
    gradient_history = []
    finite = True
    for step in range(max(1, args.steps)):
        # Fix the stochastic bridge draw so a loss change measures optimization,
        # not a different sampled time/noise realization.
        _seed(args.seed + 1)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model._train(batch, mode="pretrain")
        finite = finite and bool(torch.isfinite(loss).item())
        if not finite:
            break
        loss.backward()
        finite = finite and _finite_gradients(model)
        if not finite:
            break
        diagnostics = fusion_gradient_norms(model)
        finite = finite and all(math.isfinite(value) for value in diagnostics.values())
        if not finite:
            break
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_history.append(diagnostics)

    source_checksums_after = _checksums(model, source_keys)
    source_unchanged = source_checksums_before == source_checksums_after
    no_gradient_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    result = {
        "device": str(device),
        "split": args.split,
        "record_index": args.record_index,
        "atoms": int(batch["x0"].shape[0]),
        "blocks": int(batch["block_type"].shape[0]),
        "bonds": int(batch["bond_index"].shape[1]),
        "steps_requested": max(1, args.steps),
        "steps_completed": len(losses),
        "losses": losses,
        "gradient_history": gradient_history,
        "finite": finite,
        "decreased": len(losses) >= 2 and losses[-1] < losses[0],
        "source_key_count": len(source_keys),
        "source_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in source_keys
        ),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "trainable_tensor_count": len(trainable_names),
        "optimizer_parameter_count": sum(
            parameter.numel()
            for group in optimizer.param_groups
            for parameter in group["params"]
        ),
        "optimizer_is_exact_trainable_complement": optimizer_ids == trainable_ids,
        "trainable_names": trainable_names,
        "trainable_without_gradient_names": no_gradient_names,
        "source_checkpoint_unchanged": source_unchanged,
        "source_checksum_mismatches": sorted(
            key for key in source_keys
            if source_checksums_before.get(key) != source_checksums_after.get(key)
        ),
        "checkpoint_reports": {
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
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not (
        result["finite"]
        and result["decreased"]
        and result["source_checkpoint_unchanged"]
        and result["optimizer_is_exact_trainable_complement"]
    ):
        raise SystemExit(1)
    return result


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
