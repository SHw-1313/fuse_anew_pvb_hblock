"""Diagnostics for the Phase 10 posterior-preserving fusion path.

The helpers in this module deliberately separate the two variance sources:
PVB ``W_vec_log_var`` is the posterior used by the corrected loss, while Anew
``Wx_log_var`` is reported as a diagnostic and never participates in it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import torch


_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def _scalar(value) -> float:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().item()
    return float(value)


def _norm(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return _scalar(torch.linalg.vector_norm(value.detach().float()))


def summarize_tensor(value: torch.Tensor) -> dict[str, float | int]:
    """Return finite distribution statistics for a tensor."""

    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("non-finite tensor in Phase 10 diagnostics")
    quantile_tensor = torch.tensor(_QUANTILES, dtype=flat.dtype, device=flat.device)
    quantiles = torch.quantile(flat, quantile_tensor).detach().cpu().tolist()
    result: dict[str, float | int] = {
        "count": int(flat.numel()),
        "mean": _scalar(flat.mean()),
        "std": _scalar(flat.std(unbiased=False)),
        "min": _scalar(flat.min()),
        "max": _scalar(flat.max()),
    }
    for name, quantile in zip(("p01", "p10", "p50", "p90", "p99"), quantiles):
        result[name] = float(quantile)
    return result


def posterior_kl(
    log_var: torch.Tensor,
    mask: torch.Tensor,
    coord_prior_var: float,
    atom_block_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute PVB's coordinate-prior KL for atom- or block-level log variance."""

    if log_var.ndim != 2:
        raise ValueError("log_var must have shape [N, C] or [N_block, C]")
    if log_var.shape[0] != mask.shape[0]:
        if atom_block_id is None:
            raise ValueError("block log_var requires atom_block_id")
        log_var = log_var.index_select(0, atom_block_id.long())
    if log_var.shape[1] == 1:
        log_var = log_var.expand(-1, 3)
    if log_var.shape[1] != 3:
        raise ValueError("coordinate posterior log_var must have one or three channels")
    selected = log_var[mask.bool()]
    denominator = mask.sum().clamp_min(1).to(dtype=log_var.dtype)
    return -0.5 * torch.sum(
        1.0
        + selected
        - math.log(float(coord_prior_var))
        - torch.exp(selected) / float(coord_prior_var)
    ) / denominator


def module_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    """Return finite gradient norms grouped by PVB/Anew/projector modules."""

    squared = defaultdict(float)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite gradient for {name}")
        if name == "block_gate" or name.startswith("block_projection."):
            group = "projector_gate"
        elif name.startswith("anew_block_encoder."):
            group = "anew"
        else:
            group = "pvb"
        squared[group] += _scalar(gradient.pow(2).sum())
    return {
        group: math.sqrt(value)
        for group, value in sorted(squared.items())
    }


def collect_phase10_diagnostics(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    loss=None,
    parts: Sequence | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
) -> dict:
    """Collect one epoch/step diagnostic record from a fused model and batch.

    The posterior summaries are evaluated on the clean ``x0`` in evaluation
    mode. ``loss``/``parts`` and gradients, when supplied, are copied from the
    actual training step, so diagnostics do not alter the optimization graph.
    """

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            pvb = model._pvb_encoder_diagnostics(
                batch["atype"],
                batch["btype"],
                batch["x0"],
                batch["abid"],
                batch["mask"],
                batch["edge_mask"],
                batch["bond_index"],
            )
            anew = None
            condition = None
            if getattr(model, "fusion_mode", "off") in {
                "anew_block",
                "anew_block_pvb_posterior",
            }:
                anew = model._run_anew_block_encoder(batch["x0"], batch)
                condition = model._project_block_condition(anew)

            pvb_log_var = pvb["log_var_pvb"]
            pvb_kl = pvb["kl_loss_pvb"]
            record = {
                "epoch": None if epoch is None else int(epoch),
                "global_step": None if global_step is None else int(global_step),
                "fusion_mode": getattr(model, "fusion_mode", "off"),
                "pvb_log_var": summarize_tensor(pvb_log_var),
                "pvb_kl": _scalar(pvb_kl),
                "pvb_embedding_norm": _norm(pvb["H_pvb"]),
                "anew_log_var_used_in_loss": False,
                "anew_wx_log_var_gradient_norm": None,
                "block_gate": None,
                "tanh_block_gate": None,
                "projector_weight_norm": None,
                "projector_parameter_norm": None,
                "projected_condition_norm": None,
                "anew_h_block_norm": None,
                "anew_log_var_block": None,
                "anew_log_var_atom": None,
                "anew_kl_diagnostic": None,
            }
            if anew is not None:
                block_log_var = anew["log_var_block"]
                atom_log_var = block_log_var.index_select(
                    0, anew["atom_block_id"].long()
                ).expand(-1, 3)
                record["anew_log_var_block"] = summarize_tensor(block_log_var)
                record["anew_log_var_atom"] = summarize_tensor(atom_log_var)
                record["anew_kl_diagnostic"] = _scalar(
                    posterior_kl(
                        block_log_var,
                        batch["mask"],
                        model.coord_prior_var,
                        atom_block_id=anew["atom_block_id"],
                    )
                )
                record["anew_h_block_norm"] = _norm(anew["H_block"])
                record["block_gate"] = _scalar(model.block_gate)
                record["tanh_block_gate"] = math.tanh(_scalar(model.block_gate))
                record["projector_weight_norm"] = _norm(
                    model.block_projection[1].weight
                )
                record["projector_parameter_norm"] = math.sqrt(
                    sum(
                        _scalar(parameter.detach().float().pow(2).sum())
                        for parameter in model.block_projection.parameters()
                    )
                )
                record["projected_condition_norm"] = _norm(condition)

            if loss is not None:
                record["loss"] = _scalar(loss)
                if parts is not None:
                    record["kl"] = _scalar(parts[0])
                    record["rec_vel"] = _scalar(parts[1])
                    record["rec_drf"] = _scalar(parts[2])
                    record["rec_total"] = record["rec_vel"] + record["rec_drf"]
                    record["total_loss"] = record["loss"]
            record["gradient_norms"] = module_gradient_norms(model)
            if getattr(model, "anew_block_encoder", None) is not None:
                for name, parameter in model.anew_block_encoder.Wx_log_var.named_parameters():
                    if parameter.grad is not None:
                        record["anew_wx_log_var_gradient_norm"] = _norm(parameter.grad)
                        break
            return record
    finally:
        model.train(was_training)
