"""Compare fused execution paths on a real protein-only PDBBind item.

This is a diagnostic profiler, not a training entrypoint.  It loads the
configured PVB and Anew roles into a freshly constructed fused model, derives
a block-complete protein-only item on CPU, and compares:
- all parameters trainable;
- the strict adapter stage (only projector and gate trainable);
- forward-only evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Dict

import torch

from data.collate import collate_fn
from data.mmap_dataset import UniDataset
from data.protein_view import make_protein_only_item
from module import dyVAE
from utils.checkpoint import CheckpointReport, load_role_checkpoint
from utils.fusion_training import (
    configure_fusion_parameters,
    fusion_gradient_norms,
    fusion_parameter_groups,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pvb-checkpoint", required=True)
    parser.add_argument("--anew-checkpoint", required=True)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    return parser


def build_model() -> dyVAE:
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
        fusion_mode="anew_block",
        anew_encoder_config={
            "hidden_size": 128,
            "ffn_size": 128,
            "edge_size": 16,
            "n_rbf": 16,
            "cutoff": 10.0,
            "n_layers": 2,
            "n_head": 4,
            "k_neighbors": 4,
            "sparse_k": 3,
            "efficient": False,
            "vector_act": "layernorm",
        },
    )


def _load_roles(
    model: dyVAE, pvb_checkpoint: str, anew_checkpoint: str
) -> Dict[str, CheckpointReport]:
    # The loader prints a full mismatch report by design.  The profiler emits
    # machine-readable JSON, so retain the reports while keeping stdout clean.
    with contextlib.redirect_stdout(io.StringIO()):
        pvb = load_role_checkpoint(model, pvb_checkpoint, "pvb", min_coverage=1.0)
        anew = load_role_checkpoint(model, anew_checkpoint, "anew", min_coverage=1.0)
    return {"pvb": pvb, "anew": anew}


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _source_checksums(model, source_keys):
    state = model.state_dict()
    return {
        key: _tensor_sha256(state[key])
        for key in sorted(source_keys)
        if key in state
    }


def _step(model, batch, optimizer, training: bool):
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        loss, parts = model._train(batch, mode="pretrain")
    if training:
        loss.backward()
        optimizer.step()
    return loss, parts


def _measure(model, batch, optimizer, training: bool, warmup: int, steps: int):
    for _ in range(warmup):
        _step(model, batch, optimizer, training)
    if torch.cuda.is_available() and batch["x0"].is_cuda:
        torch.cuda.synchronize(batch["x0"].device)
        torch.cuda.reset_peak_memory_stats(batch["x0"].device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            loss, parts = _step(model, batch, optimizer, training)
        end.record()
        torch.cuda.synchronize(batch["x0"].device)
        elapsed_ms = start.elapsed_time(end) / steps
        peak_allocated = torch.cuda.max_memory_allocated(batch["x0"].device) / 2**30
        peak_reserved = torch.cuda.max_memory_reserved(batch["x0"].device) / 2**30
    else:
        start_time = time.perf_counter()
        for _ in range(steps):
            loss, parts = _step(model, batch, optimizer, training)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / steps
        peak_allocated = 0.0
        peak_reserved = 0.0
    gradients = fusion_gradient_norms(model) if training else {}
    trainable_without_gradient = []
    trainable_with_gradient = []
    if training:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                trainable_without_gradient.append(name)
            else:
                trainable_with_gradient.append(name)
    return {
        "step_ms": float(elapsed_ms),
        "loss": float(loss.detach().cpu()),
        "parts": [
            float(part.detach().cpu()) if torch.is_tensor(part) else float(part)
            for part in parts
        ],
        "gradient_norms": gradients,
        "trainable_with_gradient": len(trainable_with_gradient),
        "trainable_without_gradient": len(trainable_without_gradient),
        "trainable_without_gradient_names": trainable_without_gradient,
        "peak_allocated_gib": float(peak_allocated),
        "peak_reserved_gib": float(peak_reserved),
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        if device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        torch.cuda.set_device(device)

    dataset = UniDataset(str(Path(args.dataset_root) / f"{args.split}_block"))
    raw_item = dataset[args.record_index]
    item = make_protein_only_item(raw_item)
    batch = _move_batch(collate_fn([[item]]), device)
    reports = {}
    for role, report in _load_roles(
        build_model(), args.pvb_checkpoint, args.anew_checkpoint
    ).items():
        reports[role] = {
            "coverage": report.coverage,
            "matched": len(report.matched_keys),
            "source_keys": len(report.source_keys),
            "missing": len(report.missing_keys),
            "unexpected": len(report.unexpected_keys),
            "shape_mismatches": len(report.shape_mismatches),
        }

    results = {
        "data": {
            "split": args.split,
            "record_index": args.record_index,
            "atoms": int(batch["x0"].shape[0]),
            "blocks": int(batch["block_type"].shape[0]),
            "bonds": int(batch["bond_index"].shape[1]),
        },
        "checkpoints": reports,
        "modes": {},
    }
    for mode in ("all_trainable", "adapter", "source_frozen", "forward_only"):
        torch.manual_seed(1000)
        model = build_model()
        role_reports = _load_roles(model, args.pvb_checkpoint, args.anew_checkpoint)
        source_keys = set().union(
            *(report.matched_keys for report in role_reports.values())
        )
        source_checksums_before = (
            _source_checksums(model, source_keys) if mode == "source_frozen" else {}
        )
        model.to(device)
        if mode == "all_trainable":
            configure_fusion_parameters(model, "standard")
            optimizer = torch.optim.Adam(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=1e-4,
            )
            model.train()
            training = True
        elif mode == "adapter":
            configure_fusion_parameters(model, "adapter")
            optimizer = torch.optim.Adam(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=1e-4,
            )
            model.train()
            training = True
        elif mode == "source_frozen":
            configure_fusion_parameters(model, "source_frozen", source_keys=source_keys)
            optimizer = torch.optim.Adam(
                fusion_parameter_groups(model, 1e-4, 1e-5, 1e-4),
            )
            model.train()
            training = True
        else:
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()
            optimizer = None
            training = False
        result = _measure(
            model,
            batch,
            optimizer,
            training,
            max(0, args.warmup_steps),
            max(1, args.steps),
        )
        result["trainable_parameters"] = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        result["trainable_tensors"] = sum(
            parameter.requires_grad for parameter in model.parameters()
        )
        result["source_key_count"] = len(source_keys) if mode == "source_frozen" else 0
        result["source_parameter_count"] = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if mode == "source_frozen" and name in source_keys
        )
        if mode == "source_frozen":
            source_checksums_after = _source_checksums(model, source_keys)
            result["source_checkpoint_unchanged"] = (
                source_checksums_before == source_checksums_after
            )
            result["source_checksum_mismatches"] = sorted(
                key for key in source_keys
                if source_checksums_before.get(key) != source_checksums_after.get(key)
            )
        result["optimizer_parameters"] = (
            sum(
                parameter.numel()
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
            if optimizer is not None
            else 0
        )
        results["modes"][mode] = result
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
