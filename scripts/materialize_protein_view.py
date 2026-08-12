#!/usr/bin/env python3
"""Materialize exact-length, block-complete protein-only PVB dataset views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

from data.block_metadata import ensure_item_block_metadata
from data.mmap_dataset import UniDataset, create_mmap
from data.protein_view import make_protein_only_item
from utils.bio_utils import NUM_ATOM_TYPE


DEFAULT_SOURCE_ROOT = "/data/pvb_cross_dataset_20260810/blocks/pdbbind"
DEFAULT_OUTPUT_ROOT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/data/"
    "pdbind_protein_only"
)
DEFAULT_SPLITS = ("train_block", "valid_block", "test_block")


def _stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty split")
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def _hash_ids(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for item_id in ids:
        digest.update(str(item_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_item(item: dict, index: int) -> tuple[int, int]:
    atom_block_id, block_type, block_lengths = ensure_item_block_metadata(
        item, warn_legacy=False
    )
    num_atoms = len(item["atype"])
    if num_atoms != int(block_lengths.sum()):
        raise ValueError(
            f"item {index}: block lengths sum to {int(block_lengths.sum())}, "
            f"but atom count is {num_atoms}"
        )
    if np.any(block_type < NUM_ATOM_TYPE):
        raise ValueError(f"item {index}: protein-only view contains a non-protein block")
    if not np.array_equal(np.asarray(item["btype"], dtype=np.int64), block_type[atom_block_id]):
        raise ValueError(f"item {index}: btype and block metadata disagree")

    bond_index = np.asarray(item.get("bond_index", [[], []]), dtype=np.int64)
    if bond_index.ndim != 2 or bond_index.shape[0] != 2:
        raise ValueError(f"item {index}: invalid bond_index shape {bond_index.shape}")
    if bond_index.size and (
        np.any(bond_index < 0) or np.any(bond_index >= num_atoms)
    ):
        raise ValueError(f"item {index}: bond index is outside [0, {num_atoms})")
    return num_atoms, int(block_type.shape[0])


def materialize_split(source_dir: Path, destination_dir: Path) -> dict:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source split does not exist: {source_dir}")
    if destination_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output path: {destination_dir}"
        )
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_dir.name}.tmp-",
            dir=str(destination_dir.parent),
        )
    )

    source = UniDataset(str(source_dir), name=f"source:{source_dir.name}")
    source_ids = [str(index[0]) for index in source._indexes]
    source_declared_lengths = [source.get_len(i) for i in range(len(source))]
    converted_ids: list[str] = []
    exact_atom_counts: list[int] = []
    block_counts: list[int] = []
    bond_counts: list[int] = []
    legacy_metadata_count = 0

    def iterator():
        nonlocal legacy_metadata_count
        for index in range(len(source)):
            raw_item = source[index]
            if not all(
                key in raw_item
                for key in ("atom_block_id", "block_type", "block_lengths")
            ):
                legacy_metadata_count += 1
            item = make_protein_only_item(raw_item, copy=False)
            num_atoms, num_blocks = _validate_item(item, index)
            properties = list(source._properties[index])
            if not properties:
                raise ValueError(f"item {index}: source has no length property")
            properties[0] = str(num_atoms)
            item_id = str(source._indexes[index][0])
            converted_ids.append(item_id)
            exact_atom_counts.append(num_atoms)
            block_counts.append(num_blocks)
            bond_counts.append(
                int(np.asarray(item.get("bond_index", [[], []])).shape[1])
            )
            yield item_id, item, properties

    try:
        create_mmap(iterator(), str(temporary_dir), total_len=len(source))
        if converted_ids != source_ids:
            raise ValueError("materialized IDs differ from source IDs")
        output = UniDataset(str(temporary_dir), name=f"output:{destination_dir.name}")
        if len(output) != len(source):
            raise ValueError(
                f"materialized length differs from source: {len(output)} != {len(source)}"
            )
        output_atom_counts: list[int] = []
        for index in range(len(output)):
            item = output[index]
            num_atoms, _ = _validate_item(item, index)
            declared = output.get_len(index)
            if declared != num_atoms:
                raise ValueError(
                    f"item {index}: output property length {declared} != {num_atoms}"
                )
            output_atom_counts.append(num_atoms)
        del output
        if output_atom_counts != exact_atom_counts:
            raise ValueError("output atom counts differ from materialization counts")
        os.replace(temporary_dir, destination_dir)
    except Exception as exc:
        raise RuntimeError(
            f"materialization failed for {source_dir}; temporary output retained at "
            f"{temporary_dir}"
        ) from exc
    finally:
        del source

    return {
        "source": str(source_dir),
        "output": str(destination_dir),
        "records": len(source_ids),
        "source_id_sha256": _hash_ids(source_ids),
        "output_id_sha256": _hash_ids(converted_ids),
        "legacy_metadata_records": legacy_metadata_count,
        "source_declared_atom_stats": _stats(source_declared_lengths),
        "exact_atom_stats": _stats(exact_atom_counts),
        "block_stats": _stats(block_counts),
        "bond_stats": _stats(bond_counts),
        "exact_lengths_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    summaries = {}
    for split in args.splits:
        summaries[split] = materialize_split(
            source_root / split,
            output_root / split,
        )

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "splits": summaries,
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True)
    print(encoded)
    if args.manifest is not None:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

