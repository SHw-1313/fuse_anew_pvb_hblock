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


_EXPECTED_CORRECTED_TRAINABLE = {
    "block_gate",
    "block_projection.0.bias",
    "block_projection.0.weight",
    "block_projection.1.bias",
    "block_projection.1.weight",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pvb-checkpoint", required=True)
    parser.add_argument("--anew-checkpoint", required=True)
    parser.add_argument("--pvb-role", choices=("pvb", "pvb_full"), default="pvb")
    parser.add_argument(
        "--fusion-mode",
        choices=("anew_block", "anew_block_pvb_posterior"),
        default="anew_block",
    )
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


def _parameter_gradient_norms(model: torch.nn.Module) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            result[name] = None
        else:
            result[name] = float(parameter.grad.detach().float().norm().cpu())
    return result


def _scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        torch.cuda.set_device(device)

    _seed(args.seed)
    dataset = UniDataset(str(Path(args.dataset_root) / f"{args.split}_block"))
    raw_item = dataset[args.record_index]
    item = make_protein_only_item(raw_item)
    raw_atoms = len(raw_item["atype"])
    view_atoms = len(item["atype"])
    dataset_atoms = dataset.get_len(args.record_index)
    no_dropped_or_truncated = dataset_atoms == raw_atoms == view_atoms
    if not no_dropped_or_truncated:
        raise RuntimeError(
            "fixed-batch input is not exact: "
            f"dataset.get_len={dataset_atoms}, raw_atoms={raw_atoms}, view_atoms={view_atoms}"
        )
    batch = collate_fn([[item]])
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    model = build_model(args.fusion_mode)
    reports = _load_roles(
        model, args.pvb_checkpoint, args.anew_checkpoint, args.pvb_role
    )
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
    optimizer_is_exact = optimizer_ids == trainable_ids
    exact_corrected_trainable = (
        args.fusion_mode != "anew_block_pvb_posterior"
        or set(trainable_names) == _EXPECTED_CORRECTED_TRAINABLE
    )

    losses: list[float] = []
    parts_history: list[dict[str, float]] = []
    gradient_history: list[dict] = []
    gate_values_after_step: list[float] = []
    finite = True
    for step in range(max(1, args.steps)):
        # Fix the stochastic bridge draw so a loss change measures optimization,
        # not a different sampled time/noise realization.
        _seed(args.seed + 1)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model._train(batch, mode="pretrain")
        finite = finite and bool(torch.isfinite(loss).item())
        if not finite:
            break
        loss.backward()
        finite = finite and _finite_gradients(model)
        if not finite:
            break
        diagnostics = fusion_gradient_norms(model)
        parameter_gradients = _parameter_gradient_norms(model)
        finite = finite and all(
            math.isfinite(value)
            for value in diagnostics.values()
        ) and all(
            value is None or math.isfinite(value)
            for value in parameter_gradients.values()
        )
        if not finite:
            break
        optimizer.step()
        losses.append(_scalar(loss))
        parts_history.append(
            {
                "kl": _scalar(parts[0]),
                "rec_vel": _scalar(parts[1]),
                "rec_drf": _scalar(parts[2]),
                "rec_total": _scalar(parts[1]) + _scalar(parts[2]),
            }
        )
        gradient_history.append(
            {
                "step": step,
                "fusion": diagnostics,
                "parameter_norms": parameter_gradients,
            }
        )
        gate_values_after_step.append(float(model.block_gate.detach().cpu().item()))

    source_checksums_after = _checksums(model, source_keys)
    source_unchanged = source_checksums_before == source_checksums_after
    no_gradient_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    gate_gradient_steps = [
        entry["step"]
        for entry in gradient_history
        if (entry["parameter_norms"].get("block_gate") or 0.0) > 1.0e-12
    ]
    projector_gradient_steps = [
        entry["step"]
        for entry in gradient_history
        if any(
            name.startswith("block_projection.")
            and (value or 0.0) > 1.0e-12
            for name, value in entry["parameter_norms"].items()
        )
    ]
    rec_total_losses = [part["rec_total"] for part in parts_history]
    kl_losses = [part["kl"] for part in parts_history]
    source_parameter_names = {
        name for name, _ in model.named_parameters() if name in source_keys
    }
    anew_variance_gradient_names = [
        name
        for name, parameter in model.named_parameters()
        if "Wx_log_var" in name and parameter.grad is not None
    ]
    checkpoint_reports = {
        role: {
            "coverage": report.coverage,
            "matched": len(report.matched_keys),
            "source_keys": len(report.source_keys),
            "missing": len(report.missing_keys),
            "unexpected": len(report.unexpected_keys),
            "shape_mismatches": len(report.shape_mismatches),
        }
        for role, report in reports.items()
    }
    corrected_kl_is_pvb_like = (
        args.fusion_mode != "anew_block_pvb_posterior"
        or (bool(kl_losses) and max(kl_losses) < 0.1)
    )
    result = {
        "device": str(device),
        "split": args.split,
        "record_index": args.record_index,
        "fusion_mode": args.fusion_mode,
        "pvb_role": args.pvb_role,
        "atoms": int(batch["x0"].shape[0]),
        "blocks": int(batch["block_type"].shape[0]),
        "bonds": int(batch["bond_index"].shape[1]),
        "dataset_atoms": dataset_atoms,
        "raw_atoms": raw_atoms,
        "view_atoms": view_atoms,
        "no_dropped_or_truncated_items": no_dropped_or_truncated,
        "steps_requested": max(1, args.steps),
        "steps_completed": len(losses),
        "losses": losses,
        "parts_history": parts_history,
        "rec_total_losses": rec_total_losses,
        "kl_losses": kl_losses,
        "gradient_history": gradient_history,
        "gate_values_after_step": gate_values_after_step,
        "finite": finite,
        "decreased": len(rec_total_losses) >= 2 and rec_total_losses[-1] < rec_total_losses[0],
        "gate_gradient_steps": gate_gradient_steps,
        "projector_gradient_steps": projector_gradient_steps,
        "nonzero_gate_gradient": bool(gate_gradient_steps),
        "projector_gradient_after_initial_update": any(step >= 1 for step in projector_gradient_steps),
        "corrected_kl_is_pvb_like": corrected_kl_is_pvb_like,
        "anew_variance_gradient_names": anew_variance_gradient_names,
        "anew_variance_absent_from_gradient_graph": not anew_variance_gradient_names,
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
        "optimizer_is_exact_trainable_complement": optimizer_is_exact,
        "exact_corrected_trainable_set": exact_corrected_trainable,
        "trainable_names": trainable_names,
        "source_trainable_names": sorted(set(trainable_names) & source_parameter_names),
        "trainable_without_gradient_names": no_gradient_names,
        "source_checkpoint_unchanged": source_unchanged,
        "source_checksum_mismatches": sorted(
            key for key in source_keys
            if source_checksums_before.get(key) != source_checksums_after.get(key)
        ),
        "checkpoint_reports": checkpoint_reports,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not (
        result["finite"]
        and result["decreased"]
        and result["source_checkpoint_unchanged"]
        and result["optimizer_is_exact_trainable_complement"]
        and result["exact_corrected_trainable_set"]
        and result["nonzero_gate_gradient"]
        and result["projector_gradient_after_initial_update"]
        and result["corrected_kl_is_pvb_like"]
        and result["anew_variance_absent_from_gradient_graph"]
        and result["no_dropped_or_truncated_items"]
    ):
        raise SystemExit(1)
    return result


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
