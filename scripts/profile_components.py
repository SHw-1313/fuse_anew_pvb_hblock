#!/usr/bin/env python3
"""Measure PVB graph, encoder, decoder, and backward timings by fusion mode."""

from __future__ import annotations

import json
import math
import time

import torch

import module.model as model_impl
from scripts.profile_train_step import build_model, build_synthetic_batch, make_parser


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_call(fn, device):
    """Run ``fn`` and return ``(result, seconds)`` with CUDA event timing."""

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        _sync(device)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        return result, start.elapsed_time(end) / 1000.0

    start_time = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start_time


def _timed_wrapper(original, name, timings, device):
    def wrapper(*args, **kwargs):
        result, seconds = _timed_call(lambda: original(*args, **kwargs), device)
        timings[name] += seconds
        return result

    return wrapper


def _summary(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((value - mean) ** 2 for value in ordered) / count

    def percentile(q):
        position = (count - 1) * q
        lower = int(position)
        upper = min(lower + 1, count - 1)
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    return {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "cv": math.sqrt(variance) / mean if mean else None,
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "max": ordered[-1],
    }


def run(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    model = build_model(args).to(device).train()
    batch = build_synthetic_batch(args.atoms, args.samples, device, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup_steps = int(getattr(args, "warmup_steps", 0))
    measured_steps = int(args.steps)
    if warmup_steps < 0 or measured_steps < 1:
        raise ValueError("warmup_steps must be non-negative and steps must be positive")
    measurements = []

    for iteration in range(warmup_steps + measured_steps):
        timings = {
            "graph_seconds": 0.0,
            "encoder_seconds": 0.0,
            "decoder_seconds": 0.0,
        }
        original_graph = model_impl.construct_edges
        original_encoder = model.encode
        original_fused_encoder = model._encode_anew_block
        original_decode = model.decode
        model_impl.construct_edges = _timed_wrapper(original_graph, "graph_seconds", timings, device)
        # These are bound methods captured above. Functions assigned to an
        # instance are not rebound, so the wrapper forwards the original call
        # arguments without injecting a second ``self``.
        model.encode = _timed_wrapper(original_encoder, "encoder_seconds", timings, device)
        model._encode_anew_block = _timed_wrapper(
            original_fused_encoder, "encoder_seconds", timings, device
        )
        model.decode = _timed_wrapper(original_decode, "decoder_seconds", timings, device)
        try:
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_result, forward_seconds = _timed_call(
                lambda: model._train(batch, mode="pretrain"), device
            )
            loss, components = train_result
            _, backward_seconds = _timed_call(loss.backward, device)
            _, optimizer_seconds = _timed_call(optimizer.step, device)
            peak_memory_bytes = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else None
            )
            reserved_memory_bytes = (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda" else None
            )
        finally:
            model_impl.construct_edges = original_graph
            model.encode = original_encoder
            model._encode_anew_block = original_fused_encoder
            model.decode = original_decode

        if iteration < warmup_steps:
            continue
        scalar = lambda value: (
            float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        )
        measurements.append(
            {
                "step": iteration - warmup_steps,
                "iteration": iteration,
                "loss": scalar(loss),
                "kl_loss": scalar(components[0]),
                "rec_vel_loss": scalar(components[1]),
                "rec_drf_loss": scalar(components[2]),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds,
                "step_seconds": forward_seconds + backward_seconds + optimizer_seconds,
                **timings,
                "peak_memory_bytes": peak_memory_bytes,
                "reserved_memory_bytes": reserved_memory_bytes,
                "finite": bool(torch.isfinite(loss).item())
                and math.isfinite(forward_seconds)
                and math.isfinite(backward_seconds)
                and math.isfinite(optimizer_seconds),
            }
        )

    summary_fields = (
        "forward_seconds",
        "backward_seconds",
        "optimizer_seconds",
        "step_seconds",
        "graph_seconds",
        "encoder_seconds",
        "decoder_seconds",
        "peak_memory_bytes",
        "reserved_memory_bytes",
    )
    statistics = {
        field: _summary([measurement[field] for measurement in measurements])
        for field in summary_fields
    }
    return {
        "fusion_mode": args.fusion_mode,
        "atoms": args.atoms,
        "samples": args.samples,
        "device": str(device),
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "measurements": measurements,
        "statistics": statistics,
    }


def make_component_parser():
    parser = make_parser()
    parser.set_defaults(steps=20)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="warmup iterations excluded from the reported statistics",
    )
    return parser


if __name__ == "__main__":
    args = make_component_parser().parse_args()
    print(json.dumps(run(args), indent=2))
