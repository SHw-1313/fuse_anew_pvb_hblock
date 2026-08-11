#!/usr/bin/env python3
"""Profile one or more unchanged PVB training steps on a synthetic batch.

This diagnostic intentionally exercises the public PVB ``dyVAE._train`` path,
including graph construction, encoder, bridge objective, decoder, and
backward pass.  It does not introduce fusion behavior or alter the PVB model.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Dict, Iterable

import torch

from module import dyVAE
from utils.bio_utils import ATOM_TYPE, NUM_BLOCK_TYPE


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_synthetic_batch(
    atoms: int,
    samples: int,
    device: torch.device,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Create a PVB-compatible protein-only batch with explicit sample IDs."""

    if atoms < 2:
        raise ValueError("--atoms must be at least 2 so the bond graph is non-empty")
    if samples < 1 or atoms % samples:
        raise ValueError("--samples must be positive and divide --atoms")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    atoms_per_sample = atoms // samples
    x0 = torch.randn((atoms, 3), generator=generator, dtype=torch.float32)
    # Keep samples separated so accidental cross-sample edges are easy to detect.
    sample_offsets = torch.arange(samples, dtype=torch.float32).repeat_interleave(atoms_per_sample)
    x0[:, 0] += sample_offsets * 20.0

    # PVB uses zero-based periodic-table IDs. C/N/O are valid protein atom IDs.
    atom_choices = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    atype = atom_choices[torch.randint(len(atom_choices), (atoms,), generator=generator)]
    abid = torch.arange(samples, dtype=torch.long).repeat_interleave(atoms_per_sample)
    if atoms_per_sample % 2:
        raise ValueError("--atoms per sample must be divisible by 2 for synthetic complete blocks")
    blocks_per_sample = atoms_per_sample // 2
    local_block_id = torch.arange(atoms_per_sample, dtype=torch.long) // 2
    atom_block_id = local_block_id.repeat(samples) + (
        torch.arange(samples, dtype=torch.long).repeat_interleave(atoms_per_sample) * blocks_per_sample
    )
    residue_start = len(ATOM_TYPE)
    block_type = residue_start + torch.arange(samples * blocks_per_sample, dtype=torch.long) % 20
    btype = block_type[atom_block_id]
    block_batch = torch.arange(samples, dtype=torch.long).repeat_interleave(blocks_per_sample)
    block_lengths = torch.full(
        (samples * blocks_per_sample,), 2, dtype=torch.long
    )

    # Protein-only: edge_mask is false, while mask marks all atoms as valid.
    edge_mask = torch.zeros(atoms, dtype=torch.bool)
    mask = torch.ones(atoms, dtype=torch.bool)

    bonds = []
    for sample_id in range(samples):
        start = sample_id * atoms_per_sample
        stop = start + atoms_per_sample
        for atom_id in range(start, stop - 1):
            bonds.append((atom_id, atom_id + 1))
            bonds.append((atom_id + 1, atom_id))
    bond_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()

    return {
        "x0": x0.to(device),
        "b0": x0.to(device),
        "atype": atype.to(device),
        "btype": btype.to(device),
        "atom_block_id": atom_block_id.to(device),
        "block_type": block_type.to(device),
        "block_batch": block_batch.to(device),
        "block_lengths": block_lengths.to(device),
        "abid": abid.to(device),
        "mask": mask.to(device),
        "edge_mask": edge_mask.to(device),
        "bond_index": bond_index.to(device),
    }


def build_model(args: argparse.Namespace) -> dyVAE:
    """Construct PVB with explicit dimensions for reproducible diagnostics."""

    return dyVAE(
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        rbf_dim=args.rbf_dim,
        heads=args.heads,
        layers=args.layers,
        cutoff_lower=0.0,
        cutoff_upper=args.cutoff_upper,
        cutoff_H=3.5,
        k_neighbors=args.k_neighbors,
        coord_prior_var=0.5,
        sigma=0.2,
        additional_noise_scale=args.additional_noise_scale,
        kl_weight=0.8,
        re_weight=1.0,
        using_ode=args.using_ode,
        backbone="torchmdnet",
        fusion_mode=args.fusion_mode,
        anew_encoder_config=(
            {
                "hidden_size": args.anew_hidden_dim,
                "ffn_size": args.anew_ffn_dim,
                "edge_size": args.anew_edge_size,
                "n_rbf": args.anew_rbf_dim,
                "cutoff": args.anew_cutoff,
                "n_layers": args.anew_layers,
                "n_head": args.anew_heads,
                "k_neighbors": args.anew_k_neighbors,
                "sparse_k": args.anew_sparse_k,
            }
            if args.fusion_mode == "anew_block"
            else None
        ),
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gradient_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(squared)


def run_profile(args: argparse.Namespace) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _device_from_arg(args.device)
    model = build_model(args).to(device)
    model.train()
    batch = build_synthetic_batch(args.atoms, args.samples, device, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    measurements = []
    for step in range(args.steps):
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
        gradient_norm = _gradient_norm(model)
        optimizer.step()
        _sync(device)

        peak_memory = None
        if device.type == "cuda":
            peak_memory = int(torch.cuda.max_memory_allocated(device))
        measurements.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "kl_loss": float(components[0].detach().cpu()) if torch.is_tensor(components[0]) else float(components[0]),
                "rec_vel_loss": float(components[1].detach().cpu()) if torch.is_tensor(components[1]) else float(components[1]),
                "rec_drf_loss": float(components[2].detach().cpu()) if torch.is_tensor(components[2]) else float(components[2]),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "step_seconds": forward_seconds + backward_seconds,
                "gradient_norm": gradient_norm,
                "peak_memory_bytes": peak_memory,
                "finite_loss": bool(torch.isfinite(loss).item()),
            }
        )

    return {
        "device": str(device),
        "atoms": args.atoms,
        "samples": args.samples,
        "model": {
            "hidden_dim": args.hidden_dim,
            "ffn_dim": args.ffn_dim,
            "rbf_dim": args.rbf_dim,
            "heads": args.heads,
            "layers": args.layers,
            "k_neighbors": args.k_neighbors,
            "using_ode": args.using_ode,
            "fusion_mode": args.fusion_mode,
        },
        "measurements": measurements,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=256)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--rbf-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--k-neighbors", type=int, default=32)
    parser.add_argument("--cutoff-upper", type=float, default=10.0)
    parser.add_argument("--additional-noise-scale", type=float, default=0.2)
    parser.add_argument("--using-ode", action="store_true")
    parser.add_argument("--fusion-mode", choices=("off", "anew_block"), default="off")
    parser.add_argument("--anew-hidden-dim", type=int, default=512)
    parser.add_argument("--anew-ffn-dim", type=int, default=512)
    parser.add_argument("--anew-edge-size", type=int, default=64)
    parser.add_argument("--anew-rbf-dim", type=int, default=64)
    parser.add_argument("--anew-cutoff", type=float, default=10.0)
    parser.add_argument("--anew-layers", type=int, default=6)
    parser.add_argument("--anew-heads", type=int, default=8)
    parser.add_argument("--anew-k-neighbors", type=int, default=9)
    parser.add_argument("--anew-sparse-k", type=int, default=3)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    result = run_profile(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
