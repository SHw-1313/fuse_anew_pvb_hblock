#!/usr/bin/env python3
"""Audit fused checkpoint provenance and the source-frozen parameter complement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from scripts.profile_training_paths import _load_roles, build_model
from utils.checkpoint import _load_payload, _normalized_state_dict, _state_dict_from_payload
from utils.fusion_training import configure_fusion_parameters, fusion_parameter_groups


DEFAULT_PVB_CHECKPOINT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/"
    "pvb_state_dict.pt"
)
DEFAULT_ANEW_CHECKPOINT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/"
    "legacy_fused_state_dict.pt"
)
DEFAULT_OUTPUT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/"
    "source_frozen_provenance.json"
)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _report_json(report: Any) -> dict[str, Any]:
    return {
        "role": report.role,
        "path": report.path,
        "matched_count": len(report.matched_keys),
        "expected_key_count": report.expected_key_count,
        "coverage": report.coverage,
        "matched_keys": report.matched_keys,
        "source_key_by_target": dict(sorted(report.source_keys.items())),
        "missing_keys": report.missing_keys,
        "unexpected_keys": report.unexpected_keys,
        "shape_mismatches": {
            key: {
                field: list(value) if isinstance(value, tuple) else value
                for field, value in details.items()
            }
            for key, details in sorted(report.shape_mismatches.items())
        },
        "checkpoint_sha256": _file_sha256(report.path),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    model = build_model(args.fusion_mode)
    reports = _load_roles(
        model,
        args.pvb_checkpoint,
        args.anew_checkpoint,
        args.pvb_role,
    )
    loaded_by_role = {
        role: set(report.matched_keys) for role, report in reports.items()
    }
    source_union = set().union(*loaded_by_role.values())
    model._source_checkpoint_keys = set(source_union)
    target_state = model.state_dict()
    source_checksum_mismatches = []
    source_checksum_checked = 0
    for role, report in reports.items():
        source_state = _normalized_state_dict(
            _state_dict_from_payload(_load_payload(report.path))
        )
        for target_key, source_key in report.source_keys.items():
            source_checksum_checked += 1
            target_hash = _tensor_sha256(target_state[target_key])
            source_hash = _tensor_sha256(source_state[source_key])
            if target_hash != source_hash:
                source_checksum_mismatches.append(
                    {
                        "role": role,
                        "target_key": target_key,
                        "source_key": source_key,
                        "target_sha256": target_hash,
                        "source_sha256": source_hash,
                    }
                )
    if source_checksum_mismatches:
        raise AssertionError(
            "source-loaded target tensors do not match checkpoint tensors bitwise"
        )
    stage_counts = configure_fusion_parameters(
        model,
        stage="source_frozen",
        source_keys=source_union,
    )
    optimizer_groups = fusion_parameter_groups(
        model,
        pvb_lr=args.pvb_lr,
        anew_lr=args.anew_lr,
        projector_lr=args.projector_lr,
    )

    parameters = dict(model.named_parameters())
    target_state = model.state_dict()
    target_parameter_keys = set(parameters)
    loaded_parameter_keys = source_union.intersection(target_parameter_keys)
    new_parameter_keys = target_parameter_keys.difference(source_union)
    frozen_parameter_keys = {
        name for name, parameter in parameters.items() if not parameter.requires_grad
    }
    trainable_parameter_keys = {
        name for name, parameter in parameters.items() if parameter.requires_grad
    }
    if frozen_parameter_keys != loaded_parameter_keys:
        raise AssertionError(
            "source_frozen did not freeze exactly the loaded parameter-key union"
        )
    if trainable_parameter_keys != new_parameter_keys:
        raise AssertionError(
            "source_frozen did not expose exactly the non-source parameter complement"
        )

    role_by_target: dict[str, list[str]] = {}
    source_keys_by_target: dict[str, list[str]] = {}
    for role, report in reports.items():
        for target_key, source_key in report.source_keys.items():
            role_by_target.setdefault(target_key, []).append(role)
            source_keys_by_target.setdefault(target_key, []).append(source_key)

    provenance: dict[str, dict[str, Any]] = {}
    for target_key, value in sorted(target_state.items()):
        roles = sorted(role_by_target.get(target_key, []))
        source_keys = sorted(source_keys_by_target.get(target_key, []))
        parameter = parameters.get(target_key)
        provenance[target_key] = {
            "source_roles": roles if roles else ["new"],
            "source_keys": source_keys,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "is_parameter": parameter is not None,
            "requires_grad_after_source_frozen": (
                bool(parameter.requires_grad) if parameter is not None else False
            ),
            "target_sha256": _tensor_sha256(value),
        }

    group_summary = []
    optimizer_parameter_ids = set()
    for group in optimizer_groups:
        params = list(group["params"])
        optimizer_parameter_ids.update(id(parameter) for parameter in params)
        group_summary.append(
            {
                "name": group["name"],
                "lr": float(group["lr"]),
                "tensor_count": len(params),
                "parameter_count": int(sum(parameter.numel() for parameter in params)),
                "parameter_names": sorted(
                    name for name, parameter in parameters.items()
                    if id(parameter) in {id(item) for item in params}
                ),
            }
        )
    if optimizer_parameter_ids != {
        id(parameter) for name, parameter in parameters.items()
        if name in trainable_parameter_keys
    }:
        raise AssertionError("optimizer groups do not equal source-frozen trainable parameters")

    return {
        "model": {
            "fusion_mode": getattr(model, "fusion_mode", None),
            "pvb_checkpoint_role": args.pvb_role,
            "stage": "source_frozen",
            "state_key_count": len(target_state),
            "parameter_tensor_count": len(parameters),
            "parameter_numel": int(sum(parameter.numel() for parameter in parameters.values())),
        },
        "checkpoints": {
            role: _report_json(report) for role, report in sorted(reports.items())
        },
        "source_key_union": {
            "count": len(source_union),
            "keys": sorted(source_union),
            "parameter_count": len(loaded_parameter_keys),
            "parameter_numel": int(
                sum(parameters[name].numel() for name in loaded_parameter_keys)
            ),
        },
        "new_parameter_complement": {
            "count": len(new_parameter_keys),
            "keys": sorted(new_parameter_keys),
            "parameter_numel": int(
                sum(parameters[name].numel() for name in new_parameter_keys)
            ),
        },
        "source_loaded_parameter_checksums": {
            name: _tensor_sha256(target_state[name])
            for name in sorted(loaded_parameter_keys)
        },
        "source_checksum_audit": {
            "checked_state_tensors": source_checksum_checked,
            "mismatches": source_checksum_mismatches,
            "all_source_tensors_match_bitwise": not source_checksum_mismatches,
        },
        "stage_counts": stage_counts,
        "frozen_parameter_keys": sorted(frozen_parameter_keys),
        "trainable_parameter_keys": sorted(trainable_parameter_keys),
        "optimizer_groups": group_summary,
        "assertions": {
            "all_loaded_parameters_frozen": True,
            "all_non_source_parameters_trainable": True,
            "optimizer_equals_trainable_complement": True,
            "all_source_tensors_match_bitwise": not source_checksum_mismatches,
        },
        "per_target_key": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvb-checkpoint", default=DEFAULT_PVB_CHECKPOINT)
    parser.add_argument("--anew-checkpoint", default=DEFAULT_ANEW_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fusion-mode",
        choices=("anew_block", "anew_block_pvb_posterior"),
        default="anew_block",
    )
    parser.add_argument("--pvb-role", choices=("pvb", "pvb_full"), default="pvb")
    parser.add_argument("--pvb-lr", type=float, default=1.0e-4)
    parser.add_argument("--anew-lr", type=float, default=1.0e-5)
    parser.add_argument("--projector-lr", type=float, default=1.0e-4)
    args = parser.parse_args()

    result = audit(args)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()

