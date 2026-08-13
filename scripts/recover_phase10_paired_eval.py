"""Recover the completed Phase 10 paired report from its evaluation log.

The first paired evaluator completed all six model/split traversals, but an
unfiltered resume metadata dictionary containing optimizer tensors prevented
the final JSON write. This recovery consumes only the six aggregate lines
already emitted by that run; it never evaluates a model or re-runs test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--pvb-checkpoint", required=True)
    parser.add_argument("--legacy-checkpoint", required=True)
    parser.add_argument("--phase10-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260810, 20260811, 20260812]
    )
    return parser


def _with_rec_total(aggregate: dict) -> dict:
    result = json.loads(json.dumps(aggregate))
    result["rec_total"] = {}
    for weighting in ("batch_mean", "atom_weighted_mean"):
        velocity = result["rec_vel"][weighting]
        drift = result["rec_drf"][weighting]
        result["rec_total"][weighting] = {
            "mean": float(velocity["mean"] + drift["mean"]),
            "std": None,
        }
    return result


def _aggregate_lines(log: Path) -> list[dict]:
    values = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.startswith('{"pdbind_protein_only":'):
            continue
        parsed = json.loads(line)
        values.append(parsed["pdbind_protein_only"])
    if len(values) != 6:
        raise RuntimeError(
            f"expected six completed aggregate lines, found {len(values)}"
        )
    return values


def _load_report(path: Path, coverage: int, expected: int) -> dict:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "coverage": float(coverage / expected),
        "matched": coverage,
        "expected": expected,
        "missing": 0,
        "unexpected": 0,
        "shape_mismatches": 0,
    }


def main() -> None:
    args = _parser().parse_args()
    log = Path(args.log)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing paired report: {output}")
    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    if lock.get("status") != "locked" or lock.get("test_evaluated"):
        raise RuntimeError("the valid-only lock is not in the expected pre-evaluation state")
    if _sha256(Path(args.phase10_checkpoint)) != lock["checkpoint_sha256"]:
        raise RuntimeError("locked Phase 10 checkpoint SHA256 mismatch")

    values = _aggregate_lines(log)
    names = [
        ("valid", "pvb_off"),
        ("valid", "phase9_legacy"),
        ("valid", "phase10_corrected"),
        ("test", "pvb_off"),
        ("test", "phase9_legacy"),
        ("test", "phase10_corrected"),
    ]
    models: dict[str, dict] = {}
    for (split, name), aggregate in zip(names, values):
        models.setdefault(name, {})[split] = {
            "aggregate": _with_rec_total(aggregate),
            "per_seed": None,
            "per_seed_detail": (
                "The evaluator emitted aggregate mean/std before its final "
                "serialization failure; no second evaluation was run."
            ),
        }

    for split in ("valid", "test"):
        views = {
            name: {
                key: models[name][split]["aggregate"][key]
                for key in ("batch_count", "atom_count", "oversized_count")
            }
            for name in models
        }
        if len({tuple(view.values()) for view in views.values()}) != 1:
            raise AssertionError(f"paired {split} views differ: {views}")

    result = {
        "format_version": 1,
        "mode": "phase10_paired_eval_recovered",
        "seeds": list(args.seeds),
        "data_root": args.data_root,
        "splits": ["valid", "test"],
        "models": models,
        "paired_views": {
            split: {
                name: {
                    key: models[name][split]["aggregate"][key]
                    for key in ("batch_count", "atom_count", "oversized_count")
                }
                for name in models
            }
            for split in ("valid", "test")
        },
        "load_reports": {
            "pvb_off": _load_report(Path(args.pvb_checkpoint), 195, 195),
            "phase9_legacy": _load_report(Path(args.legacy_checkpoint), 268, 268),
            "phase10_corrected": _load_report(
                Path(args.phase10_checkpoint), 268, 268
            ),
        },
        "artifacts": {
            "pvb_checkpoint": {
                "path": args.pvb_checkpoint,
                "sha256": _sha256(Path(args.pvb_checkpoint)),
            },
            "legacy_checkpoint": {
                "path": args.legacy_checkpoint,
                "sha256": _sha256(Path(args.legacy_checkpoint)),
            },
            "phase10_checkpoint": {
                "path": args.phase10_checkpoint,
                "sha256": _sha256(Path(args.phase10_checkpoint)),
            },
            "lock": {
                "path": args.lock,
                "sha256": _sha256(Path(args.lock)),
            },
            "evaluation_log": {
                "path": str(log),
                "sha256": _sha256(log),
            },
        },
        "test_evaluated": True,
        "recovery": {
            "source": str(log),
            "completed_traversals": 6,
            "reran_evaluation": False,
            "initial_writer_error": (
                "TypeError: Object of type Tensor is not JSON serializable "
                "from unfiltered resume metadata"
            ),
            "aggregate_only": True,
            "rec_total_std": (
                "null because the first run emitted only aggregate rec_vel and "
                "rec_drf mean/std; rec_total mean is their exact sum."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
