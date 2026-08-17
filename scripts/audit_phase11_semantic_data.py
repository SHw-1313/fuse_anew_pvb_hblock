"""CPU-only PDBBind/Anew semantic-fragment data-identity audit.

The audit reads raw ligand SDF chemistry and existing PVB mmap records.  It
never rewrites the original dataset and does not infer chemistry from PVB's
binary bond_index field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from data.mmap_dataset import UniDataset
from third_party.anewomni.data.bioparse import VOCAB
from third_party.anewomni.data.bioparse.tokenizer.tokenize_3d import (
    ID2BOND,
    tokenize_3d,
)


DEFAULT_MANIFEST = "/data/pvb_cross_dataset_20260810/manifests/pdbbind_half.csv"
DEFAULT_DATASET_ROOT = "/data/pvb_cross_dataset_20260810/blocks/pdbbind"
DEFAULT_OUTPUT = "reports/t1111_semantic_data_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bond_id(bond: Chem.rdchem.Bond) -> int | None:
    try:
        return int(ID2BOND.index(bond.GetBondType()))
    except ValueError:
        return None


def _tokenize_ligand(mol: Chem.rdchem.Mol) -> dict[str, Any]:
    heavy = Chem.RemoveHs(Chem.Mol(mol))
    n_atoms = heavy.GetNumAtoms()
    fallback_reason = None
    try:
        smiles, groups = tokenize_3d(None, None, rdkit_mol=heavy)
    except Exception as exc:
        # Explicit atom-level fallback; it assigns every atom and invents no chemistry.
        fallback_reason = f"{type(exc).__name__}: {exc}"
        smiles = [f"[{atom.GetSymbol()}]" for atom in heavy.GetAtoms()]
        groups = [[atom_id] for atom_id in range(n_atoms)]
    if len(smiles) != len(groups) or not smiles:
        raise ValueError("Anew tokenizer returned no aligned fragments")
    assigned = [atom_id for group in groups for atom_id in group]
    if sorted(assigned) != list(range(n_atoms)):
        raise ValueError(
            f"tokenizer atom coverage is not exactly once: {n_atoms=} "
            f"{len(assigned)=} {len(set(assigned))=}"
        )

    atom_block_id = [-1] * n_atoms
    block_ids = []
    block_lengths = []
    unknown = 0
    for local_id, (fragment_smiles, atom_ids) in enumerate(zip(smiles, groups)):
        block_id = int(VOCAB.abrv_to_idx(fragment_smiles))
        unknown += int(block_id == VOCAB.get_block_dummy_idx())
        block_ids.append(block_id)
        block_lengths.append(len(atom_ids))
        for atom_id in atom_ids:
            atom_block_id[atom_id] = local_id
    if min(atom_block_id) < 0 or sorted(set(atom_block_id)) != list(range(len(groups))):
        raise ValueError("local semantic block IDs are not contiguous")

    bond_type_counts = Counter()
    unknown_bond_types = Counter()
    for bond in heavy.GetBonds():
        bond_id = _bond_id(bond)
        if bond_id is None:
            unknown_bond_types[str(bond.GetBondType())] += 1
        else:
            bond_type_counts[str(bond_id)] += 1
    return {
        "num_atoms": n_atoms,
        "num_blocks": len(groups),
        "atom_block_id": atom_block_id,
        "block_ids": block_ids,
        "block_lengths": block_lengths,
        "fragment_smiles": list(smiles),
        "unknown_blocks": unknown,
        "fallback_reason": fallback_reason,
        "bond_type_counts": dict(bond_type_counts),
        "unknown_bond_types": dict(unknown_bond_types),
        "aromatic_bonds": sum(int(bond.GetIsAromatic()) for bond in heavy.GetBonds()),
        "heavy_atype": [atom.GetAtomicNum() - 1 for atom in heavy.GetAtoms()],
    }


def _manifest_rows(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path)
    rows: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = f"{row['group']}:{row['pdb']}"
        if key in rows:
            raise ValueError(f"duplicate manifest key: {key}")
        rows[key] = row
    return rows


def _audit_split(
    split: str,
    dataset_root: Path,
    rows: dict[str, dict[str, Any]],
    max_records: int | None,
) -> dict[str, Any]:
    dataset = UniDataset(str(dataset_root / f"{split}_block"), name=f"audit:{split}")
    limit = len(dataset) if max_records is None else min(len(dataset), max_records)
    counters: Counter[str] = Counter()
    bond_counts: Counter[str] = Counter()
    source_ids: list[str] = []
    samples: list[dict[str, Any]] = []
    fallback_examples: list[dict[str, str]] = []
    unknown_bond_examples: list[dict[str, Any]] = []

    for index in range(limit):
        item_id = str(dataset._indexes[index][0])
        if not item_id.endswith("_holo"):
            raise ValueError(f"unexpected PVB record ID: {item_id}")
        pdb = item_id[:-len("_holo")]
        row = rows.get(f"{split}:{pdb}")
        if row is None:
            raise ValueError(f"manifest has no row for {split}:{pdb}")
        ligand_path = Path(str(row["ligandFile"]))
        if not ligand_path.is_file():
            raise FileNotFoundError(f"missing ligand SDF: {ligand_path}")
        entries = list(Chem.SDMolSupplier(str(ligand_path), removeHs=False))
        if len(entries) != 1 or entries[0] is None:
            raise ValueError(
                f"expected one valid SDF molecule for {item_id}; "
                f"entries={len(entries)}"
            )
        semantic = _tokenize_ligand(entries[0])
        item = dataset[index]
        edge_mask = np.asarray(item["edge_mask"], dtype=np.int64).reshape(-1)
        pvb_ligand_atype = np.asarray(item["atype"], dtype=np.int64)[edge_mask == 1]
        if not np.array_equal(pvb_ligand_atype, semantic["heavy_atype"]):
            raise ValueError(f"PVB/SDF ligand atom ordering mismatch: {item_id}")
        if len(pvb_ligand_atype) != semantic["num_atoms"]:
            raise ValueError(f"PVB/SDF ligand atom count mismatch: {item_id}")

        source_ids.append(item_id)
        counters["ligand_atoms"] += semantic["num_atoms"]
        counters["ligand_blocks"] += semantic["num_blocks"]
        counters["ligand_bonds"] += sum(semantic["bond_type_counts"].values()) + sum(semantic["unknown_bond_types"].values())
        counters["aromatic_bonds"] += semantic["aromatic_bonds"]
        counters["unknown_blocks"] += semantic["unknown_blocks"]
        counters["unknown_bonds"] += sum(semantic["unknown_bond_types"].values())
        if semantic["unknown_bond_types"] and len(unknown_bond_examples) < 10:
            unknown_bond_examples.append(
                {"id": item_id, "ligand_path": str(ligand_path), "types": semantic["unknown_bond_types"]}
            )
        for bond_name, count in semantic["unknown_bond_types"].items():
            bond_counts[f"unknown:{bond_name}"] += int(count)
        if semantic["fallback_reason"] is not None:
            counters["fallback_records"] += 1
            counters["fallback_atoms"] += semantic["num_atoms"]
            if len(fallback_examples) < 10:
                fallback_examples.append({"id": item_id, "reason": semantic["fallback_reason"]})
        for bond_id, count in semantic["bond_type_counts"].items():
            bond_counts[str(bond_id)] += int(count)
        if len(samples) < 3:
            samples.append(
                {
                    "id": item_id,
                    "ligand_path": str(ligand_path),
                    "num_atoms": semantic["num_atoms"],
                    "num_blocks": semantic["num_blocks"],
                    "atom_block_id": semantic["atom_block_id"],
                    "block_ids": semantic["block_ids"],
                    "block_lengths": semantic["block_lengths"],
                    "fragment_smiles": semantic["fragment_smiles"],
                }
            )
        if (index + 1) % 500 == 0:
            print(f"{split}: audited {index + 1}/{limit}", file=sys.stderr)

    return {
        "split": split,
        "records": len(source_ids),
        "dataset_records": len(dataset),
        "source_id_sha256": hashlib.sha256(
            "\n".join(source_ids).encode("utf-8")
        ).hexdigest(),
        "ligand_atoms": counters["ligand_atoms"],
        "ligand_blocks": counters["ligand_blocks"],
        "ligand_bonds": counters["ligand_bonds"],
        "bond_type_counts": dict(sorted(bond_counts.items())),
        "unknown_bond_count": counters["unknown_bonds"],
        "aromatic_bonds": counters["aromatic_bonds"],
        "unknown_fragment_blocks": counters["unknown_blocks"],
        "tokenizer_fallback_records": counters["fallback_records"],
        "tokenizer_fallback_atoms": counters["fallback_atoms"],
        "tokenizer_fallback_examples": fallback_examples,
        "unknown_bond_examples": unknown_bond_examples,
        "pvb_sdf_atom_order_exact": True,
        "complete_ligand_bond_chemistry": counters["unknown_bonds"] == 0,
        "sample_records": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows = _manifest_rows(manifest)
    dataset_root = Path(args.dataset_root)
    splits = {
        split: _audit_split(split, dataset_root, rows, args.max_records)
        for split in ("train", "valid", "test")
    }
    chemistry_complete = all(result["complete_ligand_bond_chemistry"] for result in splits.values())
    all_checks_passed = chemistry_complete and all(result["pvb_sdf_atom_order_exact"] for result in splits.values())
    payload = {
        "task": "T1111",
        "status": "passed" if all_checks_passed else "blocked",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "dataset_root": str(dataset_root),
        "vocabulary": {
            "num_block_types": VOCAB.get_num_block_type(),
            "num_atom_types": VOCAB.get_num_atom_type(),
            "unknown_block_id": VOCAB.get_block_dummy_idx(),
        },
        "scope": {
            "protein_blocks": "PVB residue IDs remain unchanged",
            "ligand_blocks": "Anew tokenizer output from raw SDF with explicit bond types",
            "pdb_bond_order": "not used for ligand fragmentation; ligand SDF is the source",
            "original_mmap_untouched": True,
            "cpu_only": True,
        },
        "splits": splits,
        "all_checks_passed": all_checks_passed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all_checks_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
