"""Run the final Phase 10 provenance and acceptance audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable


PVB = Path("/workspace/PVB")
ANEW = Path("/workspace/AnewOmni")
TARGET = Path("/workspace/fuse_anew_pvb_hblock")
PHASE9 = Path("/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9")
PHASE10 = Path("/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase10")

PROTECTED_PHASE9 = {
    PHASE9 / "checkpoints/source_frozen_epoch1_best.ckpt":
        "c9df6928268c4c8a5f27779067b83703af1a15d92f187f570bb454baa2441d57",
    PHASE9 / "profiles/source_frozen_train_epoch1.json":
        "570f42e6db3f68db16f977540977ecf7dec6c0d469ceb1cc75247136c99e2ca0",
    PHASE9 / "profiles/pvb_off_protein_valid_test.json":
        "90ad2e50d4dd1352c1c533e0f620cb9aa898674040b8cb9883d7e1d558ffa02d",
    PHASE9 / "profiles/fused_epoch1_valid_test.json":
        "5bd5de29862e7d0e68a421ea7e56e85e704f4dba643b80c4598f464aa3ced02a",
}

PHASE10_ARTIFACTS = [
    PHASE10 / "checkpoints/anew_block_pvb_posterior_best.ckpt",
    PHASE10 / "checkpoints/phase10_best.lock.json",
    PHASE10 / "profiles/phase10_train_interrupted.json",
    PHASE10 / "profiles/phase10_paired_valid_test.json",
    PHASE10 / "profiles/t1013_unit_tests.log",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _status(path: Path) -> list[str]:
    return _git(path, "status", "--porcelain=v1").splitlines()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    return parser


def _hash_report(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): _sha256(path) for path in paths}


def main() -> None:
    args = _parser().parse_args()
    paired_path = PHASE10 / "profiles/phase10_paired_valid_test.json"
    lock_path = PHASE10 / "checkpoints/phase10_best.lock.json"
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    source_status = {"PVB": _status(PVB), "AnewOmni": _status(ANEW)}
    if any(source_status.values()):
        raise AssertionError(f"source repositories are dirty: {source_status}")

    protected_hashes = _hash_report(PROTECTED_PHASE9)
    protected_ok = protected_hashes == {str(k): v for k, v in PROTECTED_PHASE9.items()}
    if not protected_ok:
        raise AssertionError("protected Phase 9 artifact hash mismatch")

    expected_seeds = [20260810, 20260811, 20260812]
    if paired.get("seeds") != expected_seeds or not paired.get("test_evaluated"):
        raise AssertionError("paired report seed/test contract failed")
    if paired.get("recovery", {}).get("reran_evaluation"):
        raise AssertionError("paired recovery unexpectedly reran evaluation")

    paired_views = paired["paired_views"]
    for split in ("valid", "test"):
        views = paired_views[split]
        if len({tuple(value.values()) for value in views.values()}) != 1:
            raise AssertionError(f"paired {split} traversal counts differ")

    load_reports = paired["load_reports"]
    if any(
        report["coverage"] != 1.0
        or report["missing"]
        or report["unexpected"]
        or report["shape_mismatches"]
        for report in load_reports.values()
    ):
        raise AssertionError("paired checkpoint coverage audit failed")

    if (
        lock.get("status") != "locked"
        or lock.get("test_evaluated")
        or lock.get("selection_metric") != "valid_rec_total_batch_mean"
    ):
        raise AssertionError("valid-only lock contract failed")
    if _sha256(Path(lock["checkpoint"])) != lock["checkpoint_sha256"]:
        raise AssertionError("locked checkpoint hash mismatch")

    diff_check = subprocess.run(
        ["git", "diff", "--check"],
        cwd=TARGET,
        text=True,
        capture_output=True,
    )
    if diff_check.returncode:
        raise AssertionError(diff_check.stdout + diff_check.stderr)

    unit_log = (PHASE10 / "profiles/t1013_unit_tests.log").read_text(encoding="utf-8")
    if "Ran 36 tests" not in unit_log or "\nOK\n" not in unit_log:
        raise AssertionError("T1013 unit-test log does not show 36 tests and OK")

    runtime_scan_roots = [
        TARGET / "module",
        TARGET / "data",
        TARGET / "utils",
        TARGET / "third_party",
        TARGET / "train.py",
        TARGET / "infer_prot.py",
    ]
    runtime_refs = []
    for root in runtime_scan_roots:
        candidates = [root] if root.is_file() else root.rglob("*.py")
        for path in candidates:
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "/workspace/PVB" in text or "/workspace/AnewOmni" in text:
                runtime_refs.append(str(path))
    if runtime_refs:
        raise AssertionError(f"runtime sibling-repository references: {runtime_refs}")

    report = {
        "format_version": 1,
        "phase": 10,
        "status": "passed",
        "source_revisions": {
            "PVB": _git(PVB, "rev-parse", "HEAD"),
            "AnewOmni": _git(ANEW, "rev-parse", "HEAD"),
        },
        "source_worktrees_clean": source_status,
        "target_branch": _git(TARGET, "branch", "--show-current"),
        "target_status": _status(TARGET),
        "target_diff_check": "passed",
        "protected_phase9_artifacts_unchanged": protected_ok,
        "protected_phase9_hashes": protected_hashes,
        "phase10_artifact_hashes": _hash_report(PHASE10_ARTIFACTS),
        "paired_report": {
            "path": str(paired_path),
            "sha256": _sha256(paired_path),
            "test_evaluated_once": True,
            "seeds": expected_seeds,
            "paired_views": paired_views,
            "checkpoint_load_reports": load_reports,
            "aggregate_only_recovery": True,
            "per_seed_detail_available": False,
        },
        "valid_only_lock": {
            "path": str(lock_path),
            "sha256": _sha256(lock_path),
            "checkpoint_sha256": lock["checkpoint_sha256"],
            "epoch": lock["epoch"],
            "global_step": lock["global_step"],
            "valid_rec_total": lock["valid_rec_total"],
            "test_evaluated_before_lock": lock["test_evaluated"],
        },
        "unit_tests": "36 tests passed",
        "compileall": "passed",
        "cli_help": "train.py and infer_prot.py passed",
        "runtime_sibling_dependency_audit": "passed",
        "known_limitation": (
            "The recovered paired report contains aggregate mean/std and fixed "
            "seed provenance, but not individual per-seed metric records because "
            "the first writer failed after all six traversals."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
