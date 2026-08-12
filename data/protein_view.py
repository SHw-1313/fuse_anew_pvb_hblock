"""Block-complete protein-only views for mixed PVB records.

PDBBind records contain a protein part plus unsupported molecular/element
blocks.  The first faithful Anew H-block milestone derives a separate view by
removing complete non-protein blocks on the CPU.  It never maps ligand block
IDs into Anew's protein vocabulary.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import numpy as np

from utils.bio_utils import NUM_ATOM_TYPE

from .block_metadata import ensure_item_block_metadata


_ATOM_FIELDS = ("atype", "btype", "x0", "b0", "mask", "edge_mask")
_OPTIONAL_COORDINATE_FIELDS = ("x1", "b1", "x_ref")


def _select_atom_field(item: Dict[str, Any], key: str, atom_keep: np.ndarray) -> None:
    value = item.get(key)
    if value is None:
        return
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[0] != atom_keep.shape[0]:
        raise ValueError(
            f"{key} must have an atom-leading dimension of {atom_keep.shape[0]}, "
            f"got {array.shape}"
        )
    item[key] = array[atom_keep].tolist()


def make_protein_only_item(item: Dict[str, Any], *, copy: bool = True) -> Dict[str, Any]:
    """Return a block-complete protein-only copy of one PVB record.

    Protein residue block IDs start at NUM_ATOM_TYPE in PVB.  Every
    selected block is retained in full, atom and bond indices are remapped,
    and the selected edge mask is required to be protein-like (zero).
    """

    result = deepcopy(item) if copy else item
    atom_block_id, block_type, block_lengths = ensure_item_block_metadata(
        result, warn_legacy=False
    )
    block_keep = block_type >= NUM_ATOM_TYPE
    if not bool(block_keep.any()):
        raise ValueError("record contains no protein residue blocks")

    atom_keep = block_keep[atom_block_id]
    selected_edge_mask = np.asarray(
        result.get("edge_mask", np.zeros(atom_keep.shape[0], dtype=np.int64))
    ).reshape(-1)
    if selected_edge_mask.shape[0] != atom_keep.shape[0]:
        raise ValueError("edge_mask does not match the atom count")
    if np.any(selected_edge_mask[atom_keep] != 0):
        raise ValueError("selected protein blocks have a nonzero edge_mask")

    old_atom_indices = np.flatnonzero(atom_keep)
    atom_remap = -np.ones(atom_keep.shape[0], dtype=np.int64)
    atom_remap[old_atom_indices] = np.arange(old_atom_indices.shape[0], dtype=np.int64)

    for key in _ATOM_FIELDS + _OPTIONAL_COORDINATE_FIELDS:
        if key in result:
            _select_atom_field(result, key, atom_keep)

    result["edge_mask"] = np.zeros(old_atom_indices.shape[0], dtype=np.int64).tolist()
    if result.get("mask") is None:
        result["mask"] = np.ones(old_atom_indices.shape[0], dtype=np.int64).tolist()

    bond_index = np.asarray(result.get("bond_index", [[], []]), dtype=np.int64)
    if bond_index.size == 0:
        result["bond_index"] = [[], []]
    else:
        if bond_index.ndim != 2 or bond_index.shape[0] != 2:
            raise ValueError(f"bond_index must have shape [2, E], got {bond_index.shape}")
        if np.any(bond_index < 0) or np.any(bond_index >= atom_keep.shape[0]):
            raise ValueError("bond_index contains an out-of-range atom index")
        keep_bonds = atom_keep[bond_index[0]] & atom_keep[bond_index[1]]
        result["bond_index"] = atom_remap[bond_index[:, keep_bonds]].tolist()

    result["atom_block_id"] = atom_remap[atom_block_id[atom_keep]].tolist()
    result["block_type"] = block_type[block_keep].tolist()
    result["block_lengths"] = block_lengths[block_keep].tolist()
    return result
