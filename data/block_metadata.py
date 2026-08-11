"""Explicit atom-to-block metadata for PVB/Anew batches.

PVB's historical mmap records contain repeated ``b0`` block-center
coordinates but no integer block membership.  New preprocessing records carry
explicit local metadata.  The legacy conversion in this module is deliberately
CPU-side and is used only by collation for old records; model forward code
receives integer IDs and never compares floating-point coordinates.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import numpy as np
import torch


class LegacyBlockMetadataWarning(UserWarning):
    """Raised when old PVB records need CPU-side metadata reconstruction."""


def _as_array(value: Any, dtype=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def derive_local_block_metadata(btype: Any, b0: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive contiguous local block IDs from one legacy PVB item.

    PVB preprocessing writes each atom's block center in ``b0`` and keeps
    atoms belonging to a block contiguous.  Exact row equality is used here
    on CPU as a compatibility bridge for old records only.  New records are
    annotated during preprocessing and do not take this path.
    """

    atom_block_type = _as_array(btype, dtype=np.int64).reshape(-1)
    block_centers = _as_array(b0, dtype=np.float64)
    if block_centers.ndim != 2 or block_centers.shape[1] != 3:
        raise ValueError(f"b0 must have shape [N, 3], got {block_centers.shape}")
    if atom_block_type.shape[0] != block_centers.shape[0]:
        raise ValueError(
            "btype and b0 must contain the same number of atoms: "
            f"{atom_block_type.shape[0]} != {block_centers.shape[0]}"
        )
    if not np.isfinite(block_centers).all():
        raise ValueError("b0 contains NaN or Inf; cannot derive block metadata")
    if atom_block_type.shape[0] == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    starts = np.ones(atom_block_type.shape[0], dtype=bool)
    if atom_block_type.shape[0] > 1:
        same_center = np.all(block_centers[1:] == block_centers[:-1], axis=1)
        same_type = atom_block_type[1:] == atom_block_type[:-1]
        starts[1:] = ~(same_center & same_type)

    atom_block_id = np.cumsum(starts, dtype=np.int64) - 1
    block_starts = np.flatnonzero(starts)
    block_type = atom_block_type[block_starts]
    block_lengths = np.bincount(atom_block_id, minlength=block_type.shape[0]).astype(np.int64)
    return atom_block_id, block_type, block_lengths


def validate_local_block_metadata(
    atom_block_id: Any,
    block_type: Any,
    block_lengths: Any,
    btype: Any,
    num_atoms: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate one item's explicit local metadata and return NumPy arrays."""

    atom_block_id = _as_array(atom_block_id, dtype=np.int64).reshape(-1)
    block_type = _as_array(block_type, dtype=np.int64).reshape(-1)
    block_lengths = _as_array(block_lengths, dtype=np.int64).reshape(-1)
    atom_types = _as_array(btype, dtype=np.int64).reshape(-1)
    if num_atoms is None:
        num_atoms = atom_types.shape[0]
    if atom_block_id.shape[0] != num_atoms or atom_types.shape[0] != num_atoms:
        raise ValueError(
            "atom_block_id, btype, and atom count disagree: "
            f"{atom_block_id.shape[0]}, {atom_types.shape[0]}, {num_atoms}"
        )
    if block_type.shape[0] != block_lengths.shape[0]:
        raise ValueError("block_type and block_lengths must have the same length")
    if num_atoms == 0:
        if block_type.size or block_lengths.size:
            raise ValueError("empty items must not contain block entries")
        return atom_block_id, block_type, block_lengths
    if block_type.shape[0] == 0 or atom_block_id.min() != 0:
        raise ValueError("atom_block_id must start at zero and contain at least one block")
    if np.any(atom_block_id[1:] < atom_block_id[:-1]):
        raise ValueError("atom_block_id must be nondecreasing within each item")
    expected_ids = np.arange(block_type.shape[0], dtype=np.int64)
    observed_ids = np.unique(atom_block_id)
    if not np.array_equal(observed_ids, expected_ids):
        raise ValueError(
            "atom_block_id must contain each contiguous local ID exactly once; "
            f"observed {observed_ids.tolist()}, expected {expected_ids.tolist()}"
        )
    if np.any(block_lengths <= 0) or int(block_lengths.sum()) != num_atoms:
        raise ValueError("block_lengths must be positive and sum to the atom count")
    observed_lengths = np.bincount(atom_block_id, minlength=block_type.shape[0])
    if not np.array_equal(observed_lengths, block_lengths):
        raise ValueError(
            f"block_lengths disagree with atom_block_id: {observed_lengths.tolist()} != {block_lengths.tolist()}"
        )
    if not np.array_equal(block_type[atom_block_id], atom_types):
        raise ValueError("btype must equal block_type[atom_block_id] for explicit metadata")
    return atom_block_id, block_type, block_lengths


def annotate_block_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """Add explicit local metadata to a preprocessing record in place."""

    atom_block_id, block_type, block_lengths = derive_local_block_metadata(item["btype"], item["b0"])
    validate_local_block_metadata(atom_block_id, block_type, block_lengths, item["btype"])
    item["atom_block_id"] = atom_block_id.tolist()
    item["block_type"] = block_type.tolist()
    item["block_lengths"] = block_lengths.tolist()
    return item


def ensure_item_block_metadata(item: Dict[str, Any], warn_legacy: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return validated local metadata, converting one old item on CPU if needed."""

    keys = ("atom_block_id", "block_type", "block_lengths")
    present = [key in item for key in keys]
    if all(present):
        return validate_local_block_metadata(
            item["atom_block_id"],
            item["block_type"],
            item["block_lengths"],
            item["btype"],
            num_atoms=len(item["btype"]),
        )
    if any(present):
        missing = [key for key, exists in zip(keys, present) if not exists]
        raise ValueError(f"incomplete block metadata; missing {missing}")
    if warn_legacy:
        warnings.warn(
            "Legacy PVB record has no explicit block metadata; deriving atom_block_id "
            "from b0 on the CPU during collation. Re-preprocess the dataset.",
            LegacyBlockMetadataWarning,
            stacklevel=2,
        )
    annotate_block_metadata(item)
    return (
        np.asarray(item["atom_block_id"], dtype=np.int64),
        np.asarray(item["block_type"], dtype=np.int64),
        np.asarray(item["block_lengths"], dtype=np.int64),
    )


def collate_block_metadata(items: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Offset local item metadata into global batch metadata."""

    atom_ids = []
    block_types = []
    block_batches = []
    block_lengths = []
    block_offset = 0
    for batch_id, item in enumerate(items):
        local_ids, local_types, local_lengths = ensure_item_block_metadata(item)
        atom_ids.append(torch.from_numpy(local_ids).long() + block_offset)
        block_types.append(torch.from_numpy(local_types).long())
        block_lengths.append(torch.from_numpy(local_lengths).long())
        block_batches.append(torch.full((local_types.shape[0],), batch_id, dtype=torch.long))
        block_offset += int(local_types.shape[0])

    return {
        "atom_block_id": torch.cat(atom_ids, dim=0),
        "block_type": torch.cat(block_types, dim=0),
        "block_batch": torch.cat(block_batches, dim=0),
        "block_lengths": torch.cat(block_lengths, dim=0),
    }
