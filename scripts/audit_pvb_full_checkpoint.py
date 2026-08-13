"""Audit the complete PVB checkpoint role used by Phase 10.

The script constructs the current fused target first, then loads the explicitly
scoped pvb_full role and persists the complete coverage/mismatch report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.profile_training_paths import build_model
from utils.checkpoint import load_role_checkpoint


DEFAULT_CHECKPOINT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/"
    "checkpoints/pvb_state_dict.pt"
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvb-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    model = build_model()
    report = load_role_checkpoint(model, args.pvb_checkpoint, "pvb_full", min_coverage=1.0)
    payload = {
        "task": "T1002",
        "role": report.role,
        "checkpoint": report.path,
        "checkpoint_sha256": _sha256(report.path),
        "expected_key_count": report.expected_key_count,
        "matched_key_count": len(report.matched_keys),
        "coverage": report.coverage,
        "matched_keys": report.matched_keys,
        "source_key_by_target": dict(sorted(report.source_keys.items())),
        "missing_keys": report.missing_keys,
        "unexpected_keys": report.unexpected_keys,
        "shape_mismatches": report.shape_mismatches,
        "prefix_counts": {
            prefix: sum(key.startswith(prefix) for key in report.matched_keys)
            for prefix in (
                "encoder.",
                "W_vec_mu.",
                "W_vec_log_var.",
                "decoder.",
                "vel_ffn.",
                "drf_ffn.",
            )
        },
        "source_revisions": {
            "target": _revision("/workspace/fuse_anew_pvb_hblock"),
            "pvb": _revision("/workspace/PVB"),
            "anew": _revision("/workspace/AnewOmni"),
        },
    }
    if report.coverage != 1.0 or report.missing_keys or report.shape_mismatches:
        raise AssertionError("pvb_full coverage is incomplete")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
