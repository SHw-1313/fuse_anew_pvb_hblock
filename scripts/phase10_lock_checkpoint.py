"""Lock the Phase 10 checkpoint selected by validation reconstruction loss."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-log", default=None)
    parser.add_argument("--data-root", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing lock: {output}")
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("Phase 10 checkpoint payload must be a mapping")

    phase10 = payload.get("phase10")
    if not isinstance(phase10, Mapping):
        raise RuntimeError("checkpoint has no Phase 10 metadata")
    if phase10.get("fusion_mode") != "anew_block_pvb_posterior":
        raise RuntimeError("checkpoint is not the corrected Phase 10 mode")
    if phase10.get("pvb_role") != "pvb_full":
        raise RuntimeError("checkpoint was not trained with pvb_full")
    if phase10.get("selection_metric") != "valid_rec_total_batch_mean":
        raise RuntimeError("checkpoint selection metric is not valid rec_total")
    if "test" in payload:
        raise RuntimeError("test metrics must not be present in the locked checkpoint")

    valid = payload.get("valid")
    if not isinstance(valid, Mapping):
        raise RuntimeError("checkpoint has no complete validation report")
    aggregate = valid.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise RuntimeError("checkpoint validation report has no aggregate")
    rec_total = aggregate.get("rec_total")
    if not isinstance(rec_total, Mapping):
        raise RuntimeError("validation report has no rec_total")
    batch_mean = rec_total.get("batch_mean")
    if not isinstance(batch_mean, Mapping) or "mean" not in batch_mean:
        raise RuntimeError("validation report has no batch-mean rec_total")
    selected_metric = float(batch_mean["mean"])
    stored_metric = float(payload.get("valid_rec_total"))
    if abs(selected_metric - stored_metric) > 1.0e-12:
        raise RuntimeError(
            "stored selection metric differs from validation aggregate: "
            f"{stored_metric} vs {selected_metric}"
        )

    before = payload.get("source_checksums_before")
    after = payload.get("source_checksums_after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise RuntimeError("checkpoint has no source checksum manifests")
    if dict(before) != dict(after):
        raise RuntimeError("source checksums changed inside the checkpoint")

    reports = payload.get("source_role_reports", {})
    if not isinstance(reports, Mapping):
        raise RuntimeError("checkpoint has no source role reports")
    for role in ("pvb_full", "anew"):
        report = reports.get(role)
        if not isinstance(report, Mapping) or float(report.get("coverage", 0.0)) != 1.0:
            raise RuntimeError(f"incomplete {role} coverage in checkpoint")

    lock = {
        "format_version": 1,
        "status": "locked",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "selection_metric": "valid_rec_total_batch_mean",
        "valid_rec_total": selected_metric,
        "valid": valid,
        "fusion_mode": phase10["fusion_mode"],
        "pvb_role": phase10["pvb_role"],
        "source_checksums": dict(sorted((str(k), str(v)) for k, v in before.items())),
        "source_role_reports": reports,
        "test_evaluated": False,
        "train_log": args.train_log,
        "data_root": args.data_root,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
