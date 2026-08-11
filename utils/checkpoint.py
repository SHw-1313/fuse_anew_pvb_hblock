"""Explicit checkpoint loading and migration for the fused PVB/Anew model.

The upstream PVB trainer serializes a complete ``nn.Module`` while Anew's
training entrypoint loads a serialized model and then calls ``state_dict()``.
The fused repository never treats either object as the fused model.  It first
constructs the configured target model, extracts a state dictionary, translates
only an explicitly supported key set, and reports every mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn


class CheckpointCoverageError(RuntimeError):
    """Raised when a checkpoint does not meet the requested coverage threshold."""


@dataclass
class CheckpointReport:
    role: str
    path: str
    matched_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    shape_mismatches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    expected_key_count: int = 0
    coverage: float = 0.0

    def summary(self) -> str:
        return (
            f"[checkpoint:{self.role}] {self.path}\n"
            f"  coverage: {len(self.matched_keys)}/{self.expected_key_count} "
            f"({self.coverage:.2%})\n"
            f"  matched keys ({len(self.matched_keys)}): {self.matched_keys}\n"
            f"  missing keys ({len(self.missing_keys)}): {self.missing_keys}\n"
            f"  unexpected keys ({len(self.unexpected_keys)}): {self.unexpected_keys}\n"
            f"  shape mismatches ({len(self.shape_mismatches)}): {self.shape_mismatches}"
        )


def _load_payload(path: str | Path) -> Any:
    # weights_only=False is intentional: old PVB/Anew files may contain a
    # serialized Module, from which we extract state_dict without instantiating
    # it as the fused model.
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _state_dict_from_payload(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, nn.Module):
        return {str(k): v for k, v in payload.state_dict().items()}

    if isinstance(payload, Mapping):
        if payload and all(isinstance(k, str) and torch.is_tensor(v) for k, v in payload.items()):
            return {str(k): v for k, v in payload.items()}
        for name in ("state_dict", "model_state_dict", "model", "module", "weights"):
            if name in payload:
                try:
                    return _state_dict_from_payload(payload[name])
                except TypeError:
                    continue
    raise TypeError(
        "Checkpoint must contain a tensor state_dict or a serialized nn.Module; "
        f"received {type(payload).__name__}"
    )


def _strip_wrappers(key: str) -> str:
    key = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "state_dict."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _normalized_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = _strip_wrappers(key)
        if normalized_key in normalized:
            raise ValueError(f"Duplicate normalized checkpoint key: {normalized_key}")
        normalized[normalized_key] = value
    return normalized


def _role_target_keys(model: nn.Module, role: str) -> set[str]:
    target_keys = set(model.state_dict().keys())
    if role == "pvb":
        prefixes = ("decoder.", "vel_ffn.", "drf_ffn.")
        return {key for key in target_keys if key.startswith(prefixes)}
    if role == "anew":
        return {key for key in target_keys if key.startswith("anew_block_encoder.")}
    if role == "resume":
        return target_keys
    raise ValueError(f"Unsupported checkpoint role: {role!r}")


def _translate_key(source_key: str, role: str) -> Optional[str]:
    key = _strip_wrappers(source_key)
    if role == "pvb":
        return key
    if role == "resume":
        return key
    if role == "anew":
        if key.startswith("anew_block_encoder."):
            return key
        prefix_map = (
            ("embedding.", "anew_block_encoder.embedding."),
            ("encoder.", "anew_block_encoder.encoder."),
            ("edge_embedding.", "anew_block_encoder.block_edge_embedding."),
            ("block_edge_embedding.", "anew_block_encoder.block_edge_embedding."),
            ("atom_edge_embedding.", "anew_block_encoder.atom_edge_embedding."),
            ("enc_embed2hidden.", "anew_block_encoder.enc_embed2hidden."),
            ("Wx_log_var.", "anew_block_encoder.Wx_log_var."),
        )
        for source_prefix, target_prefix in prefix_map:
            if key.startswith(source_prefix):
                return target_prefix + key[len(source_prefix):]
        return None
    raise ValueError(f"Unsupported checkpoint role: {role!r}")


def _load_role_payload(
    model: nn.Module,
    payload: Any,
    role: str,
    path: str,
    min_coverage: float,
) -> CheckpointReport:
    source_state = _normalized_state_dict(_state_dict_from_payload(payload))
    target_state = model.state_dict()
    expected_keys = _role_target_keys(model, role)
    report = CheckpointReport(role=role, path=path, expected_key_count=len(expected_keys))

    translated: Dict[str, torch.Tensor] = {}
    for source_key, value in source_state.items():
        target_key = _translate_key(source_key, role)
        if target_key is None or target_key not in expected_keys:
            report.unexpected_keys.append(source_key)
            continue
        if tuple(value.shape) != tuple(target_state[target_key].shape):
            report.shape_mismatches[target_key] = {
                "source_key": source_key,
                "source_shape": tuple(value.shape),
                "target_shape": tuple(target_state[target_key].shape),
            }
            continue
        translated[target_key] = value
        report.matched_keys.append(target_key)

    report.missing_keys = sorted(expected_keys.difference(report.matched_keys))
    report.matched_keys.sort()
    report.unexpected_keys.sort()
    report.coverage = (
        len(report.matched_keys) / report.expected_key_count
        if report.expected_key_count
        else 0.0
    )
    print(report.summary())
    if report.coverage < min_coverage:
        raise CheckpointCoverageError(
            f"{role} checkpoint coverage {report.coverage:.2%} is below "
            f"minimum {min_coverage:.2%}: {path}"
        )
    model.load_state_dict(translated, strict=False)
    return report


def load_role_checkpoint(
    model: nn.Module,
    path: str | Path,
    role: str,
    min_coverage: float = 0.0,
) -> CheckpointReport:
    """Load one explicitly scoped checkpoint role into an already-built model."""

    if role not in {"pvb", "anew", "resume"}:
        raise ValueError(f"Unsupported checkpoint role: {role!r}")
    payload = _load_payload(path)
    return _load_role_payload(
        model,
        payload,
        role=role,
        path=str(path),
        min_coverage=min_coverage,
    )


def load_resume_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    min_coverage: float = 1.0,
) -> Tuple[CheckpointReport, Dict[str, Any]]:
    """Restore fused model state and optional training state from a resume file."""

    payload = _load_payload(path)
    report = _load_role_payload(
        model,
        payload,
        role="resume",
        path=str(path),
        min_coverage=min_coverage,
    )
    metadata: Dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key in (
            "epoch",
            "global_step",
            "valid_global_step",
            "best_metric",
            "patience",
            "config",
        ):
            if key in payload:
                metadata[key] = payload[key]
        for key in (
            "optimizer_state_dict",
            "warmup_scheduler_state_dict",
            "scheduler_state_dict",
            "ema_model_state_dict",
            "ema_initted",
            "ema_step",
        ):
            if key in payload:
                metadata[key] = payload[key]
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in payload:
            scaler.load_state_dict(payload["scaler_state_dict"])
    return report, metadata
