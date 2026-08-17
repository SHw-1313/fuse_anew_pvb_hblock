"""Audit Phase 11A PVB-only checkpoint coverage and exact trainable complement."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch

from scripts.profile_training_paths import build_model
from utils.checkpoint import (
    _load_payload,
    _normalized_state_dict,
    _state_dict_from_payload,
    load_role_checkpoint,
)
from utils.fusion_training import configure_fusion_parameters, fusion_parameter_groups


DEFAULT_PVB_CHECKPOINT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/checkpoints/"
    "pvb_state_dict.pt"
)
DEFAULT_OUTPUT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/"
    "t1105_shared_provenance.json"
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


def _group_summary(model: torch.nn.Module, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parameters = dict(model.named_parameters())
    summary = []
    for group in groups:
        tensors = list(group["params"])
        tensor_ids = {id(parameter) for parameter in tensors}
        summary.append(
            {
                "name": str(group["name"]),
                "lr": float(group["lr"]),
                "tensor_count": len(tensors),
                "parameter_numel": int(sum(parameter.numel() for parameter in tensors)),
                "parameter_names": sorted(
                    name for name, parameter in parameters.items()
                    if id(parameter) in tensor_ids
                ),
            }
        )
    return summary


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.pvb_role != "pvb_full":
        raise ValueError("Phase 11A requires --pvb-role pvb_full")

    model = build_model("pvb_shared_hblock")
    # The loader's detailed report is retained in the JSON artifact without
    # flooding the audit log with the full 195-key list.
    with contextlib.redirect_stdout(io.StringIO()):
        report = load_role_checkpoint(
            model,
            args.pvb_checkpoint,
            "pvb_full",
            min_coverage=1.0,
        )

    if report.coverage != 1.0 or report.missing_keys or report.unexpected_keys or report.shape_mismatches:
        raise AssertionError("pvb_full coverage is incomplete for Phase 11A")

    source_state = _normalized_state_dict(
        _state_dict_from_payload(_load_payload(args.pvb_checkpoint))
    )
    target_state = model.state_dict()
    source_checksum_mismatches = []
    for target_key, source_key in report.source_keys.items():
        target_hash = _tensor_sha256(target_state[target_key])
        source_hash = _tensor_sha256(source_state[source_key])
        if target_hash != source_hash:
            source_checksum_mismatches.append(
                {
                    "target_key": target_key,
                    "source_key": source_key,
                    "target_sha256": target_hash,
                    "source_sha256": source_hash,
                }
            )
    if source_checksum_mismatches:
        raise AssertionError("source-loaded target tensors do not match bitwise")

    source_union = set(report.matched_keys)
    model._source_checkpoint_keys = frozenset(source_union)
    parameters = dict(model.named_parameters())
    loaded_parameter_keys = source_union.intersection(parameters)
    new_parameter_keys = set(parameters).difference(source_union)
    source_checksums_before = {
        name: _tensor_sha256(parameters[name]) for name in sorted(loaded_parameter_keys)
    }

    stage_counts = configure_fusion_parameters(
        model,
        stage="source_frozen",
        source_keys=source_union,
    )
    frozen_parameter_keys = {
        name for name, parameter in parameters.items() if not parameter.requires_grad
    }
    trainable_parameter_keys = {
        name for name, parameter in parameters.items() if parameter.requires_grad
    }
    if frozen_parameter_keys != loaded_parameter_keys:
        raise AssertionError("source_frozen did not freeze exactly the loaded PVB keys")
    if trainable_parameter_keys != new_parameter_keys:
        raise AssertionError("source_frozen did not expose exactly the new complement")

    expected_new_keys = {
        "shared_hblock_gate",
        "shared_hblock_adapter.projection.0.weight",
        "shared_hblock_adapter.projection.0.bias",
        "shared_hblock_adapter.projection.1.weight",
        "shared_hblock_adapter.projection.1.bias",
        "shared_hblock_adapter.projection.3.weight",
        "shared_hblock_adapter.projection.3.bias",
    }
    if trainable_parameter_keys != expected_new_keys:
        raise AssertionError(
            "unexpected non-source complement: "
            f"{sorted(trainable_parameter_keys)}"
        )

    groups = fusion_parameter_groups(
        model,
        pvb_lr=args.pvb_lr,
        anew_lr=args.anew_lr,
        projector_lr=args.projector_lr,
    )
    optimizer_parameter_ids = {
        id(parameter) for group in groups for parameter in group["params"]
    }
    expected_optimizer_ids = {id(parameters[name]) for name in trainable_parameter_keys}
    if optimizer_parameter_ids != expected_optimizer_ids:
        raise AssertionError("optimizer does not equal the exact non-source complement")
    if {str(group["name"]) for group in groups} != {"projector_gate"}:
        raise AssertionError("Phase 11A optimizer contains a non-projector group")

    source_checksums_after = {
        name: _tensor_sha256(parameters[name]) for name in sorted(loaded_parameter_keys)
    }
    if source_checksums_before != source_checksums_after:
        raise AssertionError("source-loaded tensors changed during the freeze audit")

    return {
        "task": "T1105",
        "fusion_mode": "pvb_shared_hblock",
        "pvb_checkpoint_role": report.role,
        "pvb_checkpoint": report.path,
        "pvb_checkpoint_sha256": _file_sha256(report.path),
        "checkpoint": {
            "matched": len(report.matched_keys),
            "expected": report.expected_key_count,
            "coverage": report.coverage,
            "missing": report.missing_keys,
            "unexpected": report.unexpected_keys,
            "shape_mismatches": report.shape_mismatches,
            "matched_keys": report.matched_keys,
        },
        "model": {
            "state_tensor_count": len(target_state),
            "parameter_tensor_count": len(parameters),
            "parameter_numel": int(sum(parameter.numel() for parameter in parameters.values())),
            "anew_block_encoder_constructed": model.anew_block_encoder is not None,
        },
        "source_union": {
            "state_tensor_count": len(source_union),
            "parameter_tensor_count": len(loaded_parameter_keys),
            "parameter_numel": int(sum(parameters[name].numel() for name in loaded_parameter_keys)),
        },
        "new_parameter_complement": {
            "tensor_count": len(new_parameter_keys),
            "parameter_numel": int(sum(parameters[name].numel() for name in new_parameter_keys)),
            "keys": sorted(new_parameter_keys),
        },
        "source_checksum_audit": {
            "checked_state_tensors": len(report.source_keys),
            "mismatches": source_checksum_mismatches,
            "before_after_equal": source_checksums_before == source_checksums_after,
        },
        "stage_counts": stage_counts,
        "frozen_parameter_keys": sorted(frozen_parameter_keys),
        "trainable_parameter_keys": sorted(trainable_parameter_keys),
        "optimizer_groups": _group_summary(model, groups),
        "assertions": {
            "pvb_full_coverage": True,
            "all_loaded_parameters_frozen": True,
            "only_shared_adapter_and_gate_trainable": True,
            "optimizer_equals_trainable_complement": True,
            "source_tensors_match_checkpoint_bitwise": True,
            "source_tensors_unchanged_during_audit": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvb-checkpoint", default=DEFAULT_PVB_CHECKPOINT)
    parser.add_argument("--pvb-role", choices=("pvb_full",), default="pvb_full")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--pvb-lr", type=float, default=1.0e-4)
    parser.add_argument("--anew-lr", type=float, default=1.0e-5)
    parser.add_argument("--projector-lr", type=float, default=1.0e-3)
    args = parser.parse_args()

    result = audit(args)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded + "\n", encoding="utf-8")
    print(f"wrote {output}: pvb_full={result['checkpoint']['matched']}/{result['checkpoint']['expected']}, trainable={result['new_parameter_complement']['tensor_count']} tensors / {result['new_parameter_complement']['parameter_numel']} parameters")

