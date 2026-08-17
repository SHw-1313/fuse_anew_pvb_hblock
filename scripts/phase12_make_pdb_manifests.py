#!/usr/bin/env python3
"""Create deterministic structure-level PVB manifests from an EPT index.

The EPT processor emits one index row per chain. This helper groups rows by
PDB filename before assigning train/valid/test, preventing chains from the
same structure from crossing splits. It writes one identifier per line, which
is the format consumed by PVB's data/pdb_dataset.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def stable_bucket(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10000


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_index(path: Path) -> dict[str, list[str]]:
    by_structure: dict[str, list[str]] = defaultdict(list)
    with path.open() as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            identifier = line.split("\t", 1)[0]
            if "_" not in identifier:
                raise ValueError(f"EPT index identifier has no chain separator: {identifier}")
            structure = identifier.split("_", 1)[1]
            by_structure[structure].append(identifier)
    return {key: sorted(values) for key, values in sorted(by_structure.items())}


def assign_splits(
    structures: Iterable[str],
    train_cut: int,
    valid_cut: int,
) -> dict[str, list[str]]:
    splits = {"train": [], "valid": [], "test": []}
    for structure in structures:
        bucket = stable_bucket(structure)
        if bucket < train_cut:
            split = "train"
        elif bucket < train_cut + valid_cut:
            split = "valid"
        else:
            split = "test"
        splits[split].append(structure)
    return splits


def choose_half(structures: list[str]) -> list[str]:
    ordered = sorted(
        structures,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    return ordered[: (len(ordered) + 1) // 2]


def write_manifests(
    out_dir: Path,
    by_structure: dict[str, list[str]],
    split_structures: dict[str, list[str]],
    index_hash: str,
    source_index: str,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    item_counts: dict[str, int] = {}
    structure_counts: dict[str, int] = {}
    for split, structures in split_structures.items():
        selected = set(structures)
        identifiers = [
            identifier
            for structure in sorted(selected)
            for identifier in by_structure[structure]
        ]
        (out_dir / f"{split}.txt").write_text(
            "".join(f"{identifier}\n" for identifier in identifiers)
        )
        item_counts[split] = len(identifiers)
        structure_counts[split] = len(structures)

    record = {
        "source_index": source_index,
        "source_index_sha256": index_hash,
        "assignment": "sha256(structure) mod 10000; train=9000, valid=500, test=500",
        "item_counts": item_counts,
        "structure_counts": structure_counts,
        "splits": {key: sorted(value) for key, value in split_structures.items()},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--half-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_structure = read_index(args.index)
    split_structures = assign_splits(by_structure, train_cut=9000, valid_cut=500)
    index_hash = source_sha256(args.index)
    full = write_manifests(
        args.full_dir,
        by_structure,
        split_structures,
        index_hash,
        str(args.index),
    )
    half_structures = {
        split: choose_half(structures)
        for split, structures in split_structures.items()
    }
    half = write_manifests(
        args.half_dir,
        by_structure,
        half_structures,
        index_hash,
        str(args.index),
    )
    print(json.dumps({"full": full["item_counts"], "half": half["item_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()

