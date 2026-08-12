"""Operator-level profiling for a real protein-only PDBBind batch."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

import torch

import module.model as model_impl
from data.collate import collate_fn
from data.mmap_dataset import UniDataset
from data.protein_view import make_protein_only_item
from module import dyVAE
from utils.checkpoint import load_role_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--record-index", type=int, required=True)
    parser.add_argument("--fusion-mode", choices=("off", "anew_block"), default="anew_block")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pvb-checkpoint", required=True)
    parser.add_argument("--anew-checkpoint")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--row-limit", type=int, default=40)
    return parser


def build_model(fusion_mode: str) -> dyVAE:
    anew_config = {
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
    }
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
        fusion_mode=fusion_mode,
        anew_encoder_config=anew_config if fusion_mode == "anew_block" else None,
    )


def load_roles(model: dyVAE, args: argparse.Namespace) -> dict[str, Any]:
    reports = {}
    with contextlib.redirect_stdout(io.StringIO()):
        pvb = load_role_checkpoint(model, args.pvb_checkpoint, "pvb", min_coverage=1.0)
        reports["pvb"] = {
            "coverage": pvb.coverage,
            "matched": len(pvb.matched_keys),
            "missing": len(pvb.missing_keys),
            "unexpected": len(pvb.unexpected_keys),
            "shape_mismatches": len(pvb.shape_mismatches),
        }
        if args.fusion_mode == "anew_block":
            if not args.anew_checkpoint:
                raise ValueError("--anew-checkpoint is required for anew_block")
            anew = load_role_checkpoint(
                model, args.anew_checkpoint, "anew", min_coverage=1.0
            )
            reports["anew"] = {
                "coverage": anew.coverage,
                "matched": len(anew.matched_keys),
                "missing": len(anew.missing_keys),
                "unexpected": len(anew.unexpected_keys),
                "shape_mismatches": len(anew.shape_mismatches),
            }
    return reports


def _record_wrapper(name: str, fn):
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(name):
            return fn(*args, **kwargs)

    return wrapped


def _scalar(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def _event_value(event, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value or 0.0)
    return 0.0


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("operator profiling requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.set_device(device)

    dataset = UniDataset(str(Path(args.dataset_root) / f"{args.split}_block"))
    item = make_protein_only_item(dataset[args.record_index])
    batch = collate_fn([[item]])
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    model = build_model(args.fusion_mode)
    checkpoint_reports = load_roles(model, args)
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(max(0, args.warmup_steps)):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model._train(batch, mode="pretrain")
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)

    original_graph = model_impl.construct_edges
    original_decode = model.decode
    original_fused_encode = model._encode_anew_block
    original_encode = model.encode
    original_anew_edges = None
    original_anew_ept_forward = None
    model_impl.construct_edges = _record_wrapper("PVB graph construction", original_graph)
    model.decode = _record_wrapper("PVB decoder", original_decode)
    if args.fusion_mode == "anew_block":
        original_anew_edges = model.anew_block_encoder._block_edges
        original_anew_ept_forward = model.anew_block_encoder.encoder.forward
        model.anew_block_encoder._block_edges = _record_wrapper(
            "Anew block KNN construction", original_anew_edges
        )
        model.anew_block_encoder.encoder.forward = _record_wrapper(
            "Anew EPT", original_anew_ept_forward
        )
        model._encode_anew_block = _record_wrapper(
            "Anew block encoder", original_fused_encode
        )
    else:
        model.encode = _record_wrapper("PVB encoder", original_encode)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.fusion_mode}_{len(item['atype'])}atoms_idx{args.record_index}"
    trace_path = output_dir / f"{stem}.json.gz"
    report_path = output_dir / f"{stem}.json"

    try:
        torch.cuda.reset_peak_memory_stats(device)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model._train(batch, mode="pretrain")
            loss.backward()
            optimizer.step()
            profiler.step()
        torch.cuda.synchronize(device)
    finally:
        model_impl.construct_edges = original_graph
        model.decode = original_decode
        if original_anew_edges is not None:
            model.anew_block_encoder._block_edges = original_anew_edges
        if original_anew_ept_forward is not None:
            model.anew_block_encoder.encoder.forward = original_anew_ept_forward
        model._encode_anew_block = original_fused_encode
        model.encode = original_encode

    events = profiler.key_averages(group_by_input_shape=True)
    events = sorted(
        events,
        key=lambda event: _event_value(event, "self_device_time_total", "self_cuda_time_total"),
        reverse=True,
    )
    top_events = []
    for event in events[: max(1, args.row_limit)]:
        top_events.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "device": str(getattr(event, "device_type", "")),
                "self_device_us": _event_value(event, "self_device_time_total", "self_cuda_time_total"),
                "device_us": _event_value(event, "device_time_total", "cuda_time_total"),
                "self_cpu_us": _event_value(event, "self_cpu_time_total"),
                "cpu_memory_bytes": int(getattr(event, "self_cpu_memory_usage", 0)),
                "device_memory_bytes": int(getattr(event, "self_device_memory_usage", getattr(event, "self_cuda_memory_usage", 0))),
                "input_shapes": str(getattr(event, "input_shapes", "")),
            }
        )
    profiler.export_chrome_trace(str(trace_path))
    report = {
        "fusion_mode": args.fusion_mode,
        "split": args.split,
        "record_index": args.record_index,
        "atoms": len(item["atype"]),
        "blocks": len(item["block_type"]),
        "bonds": len(item["bond_index"][0]),
        "device": str(device),
        "warmup_steps": args.warmup_steps,
        "loss": _scalar(loss),
        "parts": [_scalar(part) for part in parts],
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "checkpoint_reports": checkpoint_reports,
        "trace": str(trace_path),
        "top_cuda_events": top_events,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    print(json.dumps({
        "report": str(Path(args.output_dir) / f"{args.fusion_mode}_{report['atoms']}atoms_idx{args.record_index}.json"),
        "trace": report["trace"],
        "atoms": report["atoms"],
        "blocks": report["blocks"],
        "peak_allocated_gib": report["peak_allocated_gib"],
        "top_cuda_events": report["top_cuda_events"][:10],
    }, indent=2))


if __name__ == "__main__":
    main()
