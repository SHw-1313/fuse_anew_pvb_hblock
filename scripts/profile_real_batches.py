#!/usr/bin/env python3
"""Profile real mmap records and dynamic n*n batches without running the model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from data.block_metadata import ensure_item_block_metadata
from data.dataset_wrapper import DynamicBatchWrapper
from data.mmap_dataset import UniDataset
from utils.bio_utils import NUM_ATOM_TYPE


def _quantiles(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def _sample_indices(length: int, max_records: int, seed: int) -> list[int]:
    if max_records <= 0 or max_records >= length:
        return list(range(length))
    if max_records == 1:
        return [0]
    rng = random.Random(seed)
    selected = {0, length - 1}
    selected.update(rng.sample(range(length), k=min(max_records - 2, length - 2)))
    return sorted(selected)


def _record_stats(item: dict[str, Any]) -> dict[str, Any]:
    explicit = all(
        key in item for key in ("atom_block_id", "block_type", "block_lengths")
    )
    atom_block_id, block_type, block_lengths = ensure_item_block_metadata(
        item, warn_legacy=False
    )
    atom_count = len(item["atype"])
    bond_index = np.asarray(item.get("bond_index", []), dtype=np.int64)
    bond_count = int(bond_index.shape[1]) if bond_index.ndim == 2 else 0
    block_atom_types = block_type[atom_block_id]
    protein_mask = block_atom_types >= NUM_ATOM_TYPE
    edge_mask = np.asarray(item.get("edge_mask", [0] * atom_count), dtype=np.int64)
    mixed_edge_mask = 0
    start = 0
    for length in block_lengths.tolist():
        stop = start + int(length)
        if edge_mask[start:stop].size and np.unique(edge_mask[start:stop]).size > 1:
            mixed_edge_mask += 1
        start = stop
    return {
        "atom_count": atom_count,
        "block_count": int(block_lengths.size),
        "bond_count": bond_count,
        "block_lengths": block_lengths.astype(np.int64).tolist(),
        "protein_atom_count": int(protein_mask.sum()),
        "unsupported_atom_count": int((~protein_mask).sum()),
        "mixed_edge_mask_blocks": mixed_edge_mask,
        "metadata_source": "explicit" if explicit else "legacy_cpu_fallback",
    }


def _record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "atom_count",
        "block_count",
        "bond_count",
        "protein_atom_count",
        "unsupported_atom_count",
        "mixed_edge_mask_blocks",
    )
    summary = {field: _quantiles([record[field] for record in records]) for field in fields}
    all_block_lengths = [
        length
        for record in records
        for length in record["block_lengths"]
    ]
    summary["block_length"] = _quantiles(all_block_lengths)
    summary["records_with_legacy_metadata"] = sum(
        record["metadata_source"] == "legacy_cpu_fallback" for record in records
    )
    summary["records_with_mixed_edge_mask_blocks"] = sum(
        record["mixed_edge_mask_blocks"] > 0 for record in records
    )
    return summary


def _representative_group_stats(
    dataset: UniDataset,
    groups: list[list[int]],
    max_groups: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not groups or max_groups <= 0:
        return []
    rng = random.Random(seed)
    selected = {0, len(groups) - 1, len(groups) // 2}
    remaining = max_groups - len(selected)
    if len(groups) > len(selected) and remaining > 0:
        selected.update(rng.sample(range(len(groups)), k=min(remaining, len(groups) - len(selected))))
    selected = sorted(selected)[:max_groups]
    result = []
    for group_id in selected:
        items = [dataset[index] for index in groups[group_id]]
        records = [_record_stats(item) for item in items]
        atom_counts = [record["atom_count"] for record in records]
        max_atoms = max(atom_counts) if atom_counts else 0
        max_padded_atoms = ((max_atoms + 7) // 8) * 8 if max_atoms else 0
        per_graph_padded = [((count + 7) // 8) * 8 for count in atom_counts]
        ideal_attention_work = sum(padded * padded for padded in per_graph_padded)
        result.append(
            {
                "group_id": group_id,
                "record_count": len(records),
                "record_indices": groups[group_id],
                "actual_atoms": int(sum(atom_counts)),
                "max_atoms_per_record": int(max(atom_counts)) if atom_counts else 0,
                "hypothetical_padded_atoms": int(len(atom_counts) * max(atom_counts))
                if atom_counts
                else 0,
                "padding_ratio_if_padded": (
                    float(sum(atom_counts) / (len(atom_counts) * max(atom_counts)))
                    if atom_counts and max(atom_counts)
                    else 1.0
                ),
                "max_padded_atoms": int(max_padded_atoms),
                "dynamic_n2_work": int(sum(count * count for count in atom_counts)),
                "ideal_attention_work": int(ideal_attention_work),
                "batch_padded_attention_work": int(len(atom_counts) * max_padded_atoms * max_padded_atoms),
                "attention_padding_ratio": (
                    float(ideal_attention_work / (len(atom_counts) * max_padded_atoms * max_padded_atoms))
                    if atom_counts and max_padded_atoms else 1.0
                ),
                "blocks": int(sum(record["block_count"] for record in records)),
                "bonds": int(sum(record["bond_count"] for record in records)),
                "protein_atoms": int(sum(record["protein_atom_count"] for record in records)),
                "unsupported_atoms": int(sum(record["unsupported_atom_count"] for record in records)),
            }
        )
    return result


def _dynamic_summary(
    dataset: UniDataset,
    budget: int,
    max_groups: int,
    seed: int,
) -> dict[str, Any]:
    wrapper = DynamicBatchWrapper(
        dataset,
        complexity="n*n",
        ubound_per_batch=budget,
        same_origin=False,
    )
    groups = wrapper.batch_indexes
    group_items = [len(group) for group in groups]
    group_costs = [
        sum(dataset.get_len(index) ** 2 for index in group)
        for group in groups
    ]
    approximate_atoms = [
        sum(dataset.get_len(index) for index in group)
        for group in groups
    ]
    return {
        "budget": budget,
        "dataset_records": len(dataset),
        "dynamic_groups": len(groups),
        "records_selected": sum(group_items),
        "records_skipped_by_budget": len(dataset) - sum(group_items),
        "items_per_group": _quantiles(group_items),
        "budget_cost": _quantiles(group_costs),
        "approximate_atoms_per_group": _quantiles(approximate_atoms),
        "multi_record_groups": sum(item_count > 1 for item_count in group_items),
        "representative_groups": _representative_group_stats(
            dataset, groups, max_groups=max_groups, seed=seed
        ),
    }


def profile_split(
    path: Path,
    max_records: int,
    budgets: list[int],
    max_groups: int,
    seed: int,
) -> dict[str, Any]:
    dataset = UniDataset(str(path))
    indices = _sample_indices(len(dataset), max_records=max_records, seed=seed)
    records = [_record_stats(dataset[index]) for index in indices]
    result = {
        "path": str(path),
        "dataset_records": len(dataset),
        "inspected_records": len(records),
        "inspected_indices": indices,
        "record_summary": _record_summary(records),
        "dynamic_batches": [
            _dynamic_summary(dataset, budget, max_groups=max_groups, seed=seed + offset)
            for offset, budget in enumerate(budgets)
        ],
    }
    del dataset
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/pvb_cross_dataset_20260810/blocks"),
    )
    parser.add_argument("--dataset", default="pdbbind")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_block", "valid_block", "test_block"],
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=100,
        help="records inspected per split; 0 inspects every record",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[2_000, 4_000_000, 8_000_000],
        help="n*n dynamic batch budgets to compare",
    )
    parser.add_argument("--max-groups", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    result = {
        "dataset": args.dataset,
        "dataset_root": str(args.dataset_root),
        "max_records": args.max_records,
        "budgets": args.budgets,
        "splits": {
            split: profile_split(
                args.dataset_root / args.dataset / split,
                max_records=args.max_records,
                budgets=args.budgets,
                max_groups=args.max_groups,
                seed=args.seed + sum(ord(character) for character in split),
            )
            for split in args.splits
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
