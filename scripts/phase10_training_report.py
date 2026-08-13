"""Recover an auditable summary from a Phase 10 training log/checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

import torch


VALID_RE = re.compile(
    r"phase10 valid epoch=(?P<epoch>\d+) rec_total=(?P<rec>[0-9.eE+-]+) "
    r"best=(?P<best>[0-9.eE+-]+) improved=(?P<improved>True|False)"
)
TRAIN_RE = re.compile(
    r"phase10 train epoch=(?P<epoch>\d+) steps=(?P<steps>\d+) "
    r"loss=(?P<loss>[0-9.eE+-]+) atoms=(?P<atoms>\d+)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    log_path = Path(args.log)
    checkpoint_path = Path(args.checkpoint)
    valid_reports = []
    valid_json = None
    train_progress = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{") and "pdbind_protein_only" in line:
            try:
                valid_json = json.loads(line)
            except json.JSONDecodeError:
                valid_json = None
        match = VALID_RE.search(line)
        if match:
            if not isinstance(valid_json, Mapping):
                raise RuntimeError("validation summary is missing before valid marker")
            valid_reports.append(
                {
                    "epoch": int(match["epoch"]),
                    "rec_total": float(match["rec"]),
                    "best": float(match["best"]),
                    "improved": match["improved"] == "True",
                    "valid": valid_json["pdbind_protein_only"],
                }
            )
            valid_json = None
        match = TRAIN_RE.search(line)
        if match:
            train_progress.append(
                {
                    "epoch": int(match["epoch"]),
                    "steps": int(match["steps"]),
                    "loss": float(match["loss"]),
                    "atoms": int(match["atoms"]),
                }
            )

    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    complete_epochs = sorted({record["epoch"] for record in valid_reports})
    last_train_epoch = max((record["epoch"] for record in train_progress), default=None)
    report = {
        "format_version": 1,
        "mode": "phase10_train_corrected",
        "fusion_mode": "anew_block_pvb_posterior",
        "run_status": "interrupted_before_epoch_validation",
        "log": str(log_path),
        "log_sha256": _sha256(log_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_global_step": int(payload["global_step"]),
        "checkpoint_valid_rec_total": float(payload["valid_rec_total"]),
        "epochs_requested": 5,
        "minimum_epochs": 3,
        "patience": 2,
        "complete_epochs_with_validation": complete_epochs,
        "last_epoch_seen_in_train_log": last_train_epoch,
        "valid_history": valid_reports,
        "last_train_progress": train_progress[-1] if train_progress else None,
        "source_checksums_unchanged": (
            payload.get("source_checksums_before") == payload.get("source_checksums_after")
        ),
        "test_evaluated": False,
        "interpretation": (
            "The best checkpoint is selected from complete validation epochs only. "
            "The long-running session ended during the next training epoch before "
            "its validation pass; no test result was used."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
