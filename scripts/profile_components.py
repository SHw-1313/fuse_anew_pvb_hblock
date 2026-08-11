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


def _timed_wrapper(original, name, timings, device):
    def wrapper(*args, **kwargs):
        _sync(device)
        start = time.perf_counter()
        result = original(*args, **kwargs)
        _sync(device)
        timings[name] += time.perf_counter() - start
        return result

    return wrapper


def run(args):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = build_model(args).to(device).train()
    batch = build_synthetic_batch(args.atoms, args.samples, device, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    measurements = []

    for step in range(args.steps):
        timings = {"graph_seconds": 0.0, "encoder_seconds": 0.0, "decoder_seconds": 0.0}
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
            _sync(device)
            start = time.perf_counter()
            loss, components = model._train(batch, mode="pretrain")
            _sync(device)
            forward_seconds = time.perf_counter() - start
            backward_start = time.perf_counter()
            loss.backward()
            _sync(device)
            backward_seconds = time.perf_counter() - backward_start
            optimizer.step()
            _sync(device)
            peak_memory_bytes = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else None
            )
        finally:
            model_impl.construct_edges = original_graph
            model.encode = original_encoder
            model._encode_anew_block = original_fused_encoder
            model.decode = original_decode
        measurements.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "kl_loss": float(components[0].detach().cpu()) if torch.is_tensor(components[0]) else float(components[0]),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "step_seconds": forward_seconds + backward_seconds,
                **timings,
                "peak_memory_bytes": peak_memory_bytes,
                "finite": bool(torch.isfinite(loss).item()) and math.isfinite(backward_seconds),
            }
        )
    return {
        "fusion_mode": args.fusion_mode,
        "atoms": args.atoms,
        "samples": args.samples,
        "device": str(device),
        "measurements": measurements,
    }


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
