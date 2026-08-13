"""Audit the Phase 9 loss decomposition and posterior provenance.

This is a read-only diagnostic over Phase 9 JSON/manifests. It does not
construct or modify a model and it never writes into the Phase 9 directory.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any


KL_WEIGHT = 0.8
RE_WEIGHT = 1.0
WEIGHTINGS = ("batch_mean", "atom_weighted_mean")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase9-root",
        default="/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Phase 10 diagnostic output path.",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _aggregate(root: Path, filename: str) -> dict[str, Any]:
    data = _read_json(root / "profiles" / filename)
    return data["splits"]["valid"]["pdbind_protein_only"]["aggregate"]


def _metric(aggregate: dict[str, Any], name: str, weighting: str) -> float:
    return float(aggregate[name][weighting]["mean"])


def _git(cwd: str, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", cwd, *args], text=True
    ).strip()


def main() -> None:
    args = _parser().parse_args()
    root = Path(args.phase9_root)
    target_root = Path(__file__).resolve().parents[1]

    pvb_aggregate = _aggregate(root, "pvb_off_protein_valid_test.json")
    fused_aggregate = _aggregate(root, "fused_epoch1_valid_test.json")
    provenance = _read_json(
        root / "checkpoints" / "source_frozen_provenance.json"
    )
    train_report = _read_json(
        root / "profiles" / "source_frozen_train_epoch1.json"
    )

    decomposition: dict[str, Any] = {}
    max_formula_error = 0.0
    for weighting in WEIGHTINGS:
        rows = {}
        for name, aggregate in (
            ("pvb_off", pvb_aggregate),
            ("legacy_fused", fused_aggregate),
        ):
            kl = _metric(aggregate, "kl", weighting)
            rec_vel = _metric(aggregate, "rec_vel", weighting)
            rec_drf = _metric(aggregate, "rec_drf", weighting)
            reported = _metric(aggregate, "loss", weighting)
            recomputed = KL_WEIGHT * kl + RE_WEIGHT * (rec_vel + rec_drf)
            error = recomputed - reported
            max_formula_error = max(max_formula_error, abs(error))
            rows[name] = {
                "loss_reported": reported,
                "loss_recomputed": recomputed,
                "formula_error": error,
                "kl": kl,
                "rec_vel": rec_vel,
                "rec_drf": rec_drf,
                "rec_total": rec_vel + rec_drf,
            }
        decomposition[weighting] = rows

    pvb_batch = decomposition["batch_mean"]["pvb_off"]
    fused_batch = decomposition["batch_mean"]["legacy_fused"]
    pvb_atom = decomposition["atom_weighted_mean"]["pvb_off"]
    fused_atom = decomposition["atom_weighted_mean"]["legacy_fused"]

    target_model = (target_root / "module" / "model.py").read_text()
    anew_encoder = (target_root / "module" / "anew_block_encoder.py").read_text()
    checkpoint_code = (target_root / "utils" / "checkpoint.py").read_text()

    anew_matched = provenance["checkpoints"]["anew"]["matched_keys"]
    pvb_matched = provenance["checkpoints"]["pvb"]["matched_keys"]
    per_target = provenance["per_target_key"]
    anew_variance_keys = [
        key for key in anew_matched if "Wx_log_var" in key
    ]
    pvb_posterior_keys = [
        key for key in pvb_matched
        if key.startswith(("encoder.", "W_vec_mu.", "W_vec_log_var."))
    ]
    pvb_posterior_complement = [
        key for key, record in per_target.items()
        if key.startswith(("encoder.", "W_vec_mu.", "W_vec_log_var."))
        and "new" in record.get("source_roles", [])
    ]

    code_assertions = {
        "pvb_encode_uses_W_vec_log_var": "self.W_vec_log_var(h)" in target_model,
        "pvb_encode_constructs_x_rep": "x_rep = x_mu + torch.exp(x_log_var / 2)" in target_model,
        "legacy_anew_path_uses_block_log_var": 'block_output["log_var_block"]' in target_model,
        "legacy_anew_path_constructs_x_rep": "atom_log_var / 2" in target_model,
        "anew_variance_head_exists": "self.Wx_log_var = nn.Linear" in anew_encoder,
        "legacy_pvb_role_excludes_encoder": 'prefixes = ("decoder.", "vel_ffn.", "drf_ffn.")' in checkpoint_code,
    }
    if not all(code_assertions.values()):
        raise AssertionError(f"loss-path source assertions failed: {code_assertions}")

    source_frozen_checks = {
        "source_key_count": train_report["source_key_count"],
        "source_parameter_count": train_report["source_parameter_count"],
        "trainable_parameter_count": train_report["trainable_parameter_count"],
        "optimizer_is_exact_trainable_complement": train_report[
            "optimizer_is_exact_trainable_complement"
        ],
        "source_checkpoint_unchanged": train_report[
            "source_checkpoint_unchanged"
        ],
        "source_checksum_mismatches": train_report["source_checksum_mismatches"],
        "anew_variance_keys_loaded": anew_variance_keys,
        "anew_variance_keys_frozen": [
            key for key in anew_variance_keys
            if not per_target[key]["requires_grad_after_source_frozen"]
        ],
        "pvb_posterior_keys_loaded_by_legacy_role": pvb_posterior_keys,
        "pvb_posterior_keys_in_legacy_complement": pvb_posterior_complement,
        "effective_projector_gate_tensors": provenance["optimizer_groups"][0][
            "parameter_names"
        ],
    }

    report = {
        "phase": 10,
        "task": "T1001",
        "phase9_root": str(root),
        "inputs": {
            "pvb_off_report": str(
                root / "profiles" / "pvb_off_protein_valid_test.json"
            ),
            "legacy_fused_report": str(
                root / "profiles" / "fused_epoch1_valid_test.json"
            ),
            "provenance_manifest": str(
                root / "checkpoints" / "source_frozen_provenance.json"
            ),
            "training_report": str(
                root / "profiles" / "source_frozen_train_epoch1.json"
            ),
        },
        "loss_contract": {
            "kl_weight": KL_WEIGHT,
            "re_weight": RE_WEIGHT,
            "formula": "loss = kl_weight * KL + re_weight * (rec_vel + rec_drf)",
            "using_ode": False,
        },
        "valid_paired_decomposition": decomposition,
        "diagnosis": {
            "max_formula_error": max_formula_error,
            "formula_reproduced": max_formula_error < 1.0e-5,
            "legacy_kl_ratio_batch": fused_batch["kl"] / pvb_batch["kl"],
            "legacy_kl_ratio_atom": fused_atom["kl"] / pvb_atom["kl"],
            "reconstruction_delta_batch": fused_batch["rec_total"] - pvb_batch["rec_total"],
            "reconstruction_delta_atom": fused_atom["rec_total"] - pvb_atom["rec_total"],
            "total_loss_delta_batch": fused_batch["loss_reported"] - pvb_batch["loss_reported"],
            "total_loss_delta_atom": fused_atom["loss_reported"] - pvb_atom["loss_reported"],
            "kl_dominates_total_loss": (
                fused_batch["kl"] > 100.0 * pvb_batch["kl"]
                and fused_batch["rec_total"] < pvb_batch["rec_total"]
            ),
        },
        "posterior_provenance": {
            "legacy_fusion_mode": "anew_block",
            "legacy_kl_source": (
                "module/model.py::_encode_anew_block: "
                "log_var_block -> atom_log_var -> KL and reparameterized x_rep"
            ),
            "anew_variance_head": (
                "module/anew_block_encoder.py::AnewBlockEncoder.Wx_log_var"
            ),
            "pvb_posterior_source_if_off": (
                "module/model.py::encode: W_vec_log_var(h) -> KL and x_rep"
            ),
            "source_frozen": source_frozen_checks,
            "code_assertions": code_assertions,
        },
        "source_revisions": {
            "target": _git(str(target_root), "rev-parse", "HEAD"),
            "pvb": _git("/workspace/PVB", "rev-parse", "HEAD"),
            "anew": _git("/workspace/AnewOmni", "rev-parse", "HEAD"),
        },
    }
    if not report["diagnosis"]["formula_reproduced"]:
        raise AssertionError("Phase 9 loss formula did not reproduce from reports")
    if not report["diagnosis"]["kl_dominates_total_loss"]:
        raise AssertionError("recorded Phase 9 KL diagnosis did not reproduce")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
