#!/usr/bin/env python3
"""Run a deterministic one-batch PVB overfit diagnostic."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from scripts.profile_train_step import build_model, build_synthetic_batch


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--rbf-dim", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--cutoff-upper", type=float, default=10.0)
    parser.add_argument("--additional-noise-scale", type=float, default=0.0)
    parser.add_argument("--using-ode", action="store_true")
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _gradient_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(squared)


def main() -> None:
    args = make_parser().parse_args()
    device = _device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(args).to(device)
    model.train()
    batch = build_synthetic_batch(args.atoms, args.samples, device, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    gradient_norms = []
    finite = True
    for _ in range(args.steps):
        # PVB samples both t and latent noise. Resetting both RNGs makes this a
        # fixed-batch regression diagnostic rather than a stochastic benchmark.
        torch.manual_seed(args.seed + 1)
        np.random.seed(args.seed + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + 1)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model._train(batch, mode="pretrain")
        finite = finite and bool(torch.isfinite(loss).item())
        if not finite:
            break
        loss.backward()
        grad_norm = _gradient_norm(model)
        finite = finite and math.isfinite(grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(grad_norm)

    decreased = len(losses) >= 2 and losses[-1] < losses[0]
    result = {
        "device": str(device),
        "atoms": args.atoms,
        "samples": args.samples,
        "steps_requested": args.steps,
        "steps_completed": len(losses),
        "losses": losses,
        "gradient_norms": gradient_norms,
        "finite": finite,
        "decreased": decreased,
    }
    print(json.dumps(result, indent=2))
    if not finite or not decreased:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
