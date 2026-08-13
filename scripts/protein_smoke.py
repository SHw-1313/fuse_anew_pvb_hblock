#!/usr/bin/env python3
"""Run a tiny protein-only train and inference smoke for all fusion modes."""

from __future__ import annotations

import argparse
import json

import torch

from scripts.profile_train_step import build_model, build_synthetic_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fusion-mode", choices=("off", "anew_block", "anew_block_pvb_posterior"), default="anew_block")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    torch.manual_seed(41)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(41)

    class Config:
        hidden_dim = 32
        ffn_dim = 64
        rbf_dim = 8
        heads = 4
        layers = 2
        k_neighbors = 8
        cutoff_upper = 10.0
        additional_noise_scale = 0.0
        using_ode = True
        fusion_mode = args.fusion_mode
        anew_hidden_dim = 32
        anew_ffn_dim = 32
        anew_edge_size = 16
        anew_rbf_dim = 8
        anew_cutoff = 10.0
        anew_layers = 1
        anew_heads = 4
        anew_k_neighbors = 4
        anew_sparse_k = 2

    model = build_model(Config()).to(device).train()
    batch = build_synthetic_batch(16, 2, device, seed=41)
    loss, _ = model._train(batch, mode="pretrain")
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    model.eval()
    with torch.no_grad():
        generated = model.inference(batch, sde_step=2)
    result = {
        "fusion_mode": args.fusion_mode,
        "device": str(device),
        "train_loss": float(loss.detach().cpu()),
        "inference_shape": list(generated.shape),
        "finite": bool(torch.isfinite(loss).item()) and finite_gradients and bool(torch.isfinite(generated).all()),
    }
    print(json.dumps(result, indent=2))
    if not result["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
