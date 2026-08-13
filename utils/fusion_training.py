"""Parameter-freezing and optimizer-group helpers for staged fusion training."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def configure_fusion_parameters(
    model: nn.Module,
    stage: str = "standard",
    unfreeze_ept_layers: int = 2,
    source_keys: Iterable[str] | None = None,
) -> Dict[str, int]:
    """Set ``requires_grad`` for the adapter and staged fusion schedules.

    The ``adapter`` stage freezes all original PVB/Anew network parameters and
    trains only the newly introduced block projector and scalar gate. Stage A
    trains the PVB decoder/heads as well, Stage B adds the last N Anew EPT
    layers, and Stage C unfreezes the complete fused model.
    ``source_frozen`` freezes the union of keys loaded from the PVB/Anew
    checkpoint roles and leaves only non-source-loaded parameters trainable.

    """

    model = unwrap_model(model)
    stage = str(stage).upper()
    if stage not in {"STANDARD", "ADAPTER", "SOURCE_FROZEN", "A", "B", "C"}:
        raise ValueError("fusion training stage must be standard, adapter, source_frozen, A, B, or C")

    if getattr(model, "fusion_mode", "off") not in {"anew_block", "anew_block_pvb_posterior"} or stage == "STANDARD":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return {"trainable": sum(p.requires_grad for p in model.parameters()), "total": sum(1 for _ in model.parameters())}

    for parameter in model.parameters():
        parameter.requires_grad = False

    pvb_prefixes = ("decoder.", "vel_ffn.", "drf_ffn.", "block_projection.")
    if stage == "SOURCE_FROZEN":
        loaded_keys = set(source_keys if source_keys is not None else getattr(model, "_source_checkpoint_keys", ()))
        if not loaded_keys:
            raise ValueError("source_frozen requires matched checkpoint keys on the fused model")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name not in loaded_keys
        total = sum(1 for _ in model.parameters())
        return {"trainable": sum(p.requires_grad for p in model.parameters()), "total": total}
    if stage == "ADAPTER":
        for name, parameter in model.named_parameters():
            if name == "block_gate" or name.startswith("block_projection."):
                parameter.requires_grad = True
        total = sum(1 for _ in model.parameters())
        return {"trainable": sum(p.requires_grad for p in model.parameters()), "total": total}
    for name, parameter in model.named_parameters():
        if name == "block_gate" or name.startswith(pvb_prefixes):
            parameter.requires_grad = True

    if stage in {"B", "C"}:
        if stage == "C":
            for name, parameter in model.named_parameters():
                if name.startswith("anew_block_encoder."):
                    parameter.requires_grad = True
        else:
            layer_pattern = re.compile(r"^anew_block_encoder\.encoder\.encoder\.layer_(\d+)\.")
            layer_ids = sorted(
                {
                    int(match.group(1))
                    for name in (name for name, _ in model.named_parameters())
                    if (match := layer_pattern.match(name))
                }
            )
            if not layer_ids:
                raise ValueError("Could not find Anew EPT layers for stage B")
            count = max(1, min(int(unfreeze_ept_layers), len(layer_ids)))
            allowed = set(layer_ids[-count:])
            for name, parameter in model.named_parameters():
                match = layer_pattern.match(name)
                if match and int(match.group(1)) in allowed:
                    parameter.requires_grad = True

    total = sum(1 for _ in model.parameters())
    return {"trainable": sum(p.requires_grad for p in model.parameters()), "total": total}


def fusion_parameter_groups(
    model: nn.Module,
    pvb_lr: float,
    anew_lr: float,
    projector_lr: float,
) -> List[Dict[str, object]]:
    """Return separate optimizer groups for PVB, Anew, and projection/gate."""

    model = unwrap_model(model)
    grouped = defaultdict(list)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "block_gate" or name.startswith("block_projection."):
            group_name = "projector_gate"
        elif name.startswith("anew_block_encoder."):
            group_name = "anew"
        else:
            group_name = "pvb"
        grouped[group_name].append(parameter)

    learning_rates = {
        "pvb": float(pvb_lr),
        "anew": float(anew_lr),
        "projector_gate": float(projector_lr),
    }
    return [
        {"name": name, "params": params, "lr": learning_rates[name]}
        for name, params in grouped.items()
        if params
    ]


def fusion_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """Return finite per-module gradient norms and the current scalar gate."""

    model = unwrap_model(model)
    squared = defaultdict(float)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name == "block_gate" or name.startswith("block_projection."):
            group_name = "projector_gate"
        elif name.startswith("anew_block_encoder."):
            group_name = "anew"
        else:
            group_name = "pvb"
        squared[group_name] += float(parameter.grad.detach().float().pow(2).sum().cpu())
    result = {name: math.sqrt(value) for name, value in squared.items()}
    if getattr(model, "block_gate", None) is not None:
        result["block_gate"] = float(model.block_gate.detach().cpu().item())
    return result
