'''Audit the vendored Anew tokenizer/vocabulary namespace and parity.

The source checkout is used only by isolated subprocess probes and hash
comparison. The fused runtime imports only ``third_party.anewomni``.
'''

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = "/workspace/AnewOmni"
DEFAULT_TARGET_ROOT = "/workspace/fuse_anew_pvb_hblock"
DEFAULT_OUTPUT = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/"
    "t1110_anew_vendor_provenance.json"
)

# (source-relative path, target-relative path, import-only adaptation required)
VENDOR_MAP = (
    ("data/bioparse/__init__.py", "third_party/anewomni/data/bioparse/__init__.py", False),
    ("data/bioparse/hierarchy.py", "third_party/anewomni/data/bioparse/hierarchy.py", False),
    ("data/bioparse/utils.py", "third_party/anewomni/data/bioparse/utils.py", False),
    ("data/bioparse/const.py", "third_party/anewomni/data/bioparse/const.py", False),
    ("data/bioparse/vocab.py", "third_party/anewomni/data/bioparse/vocab.py", False),
    ("data/bioparse/tokenizer/tokenize_3d.py", "third_party/anewomni/data/bioparse/tokenizer/tokenize_3d.py", True),
    ("data/bioparse/tokenizer/mol_bpe.py", "third_party/anewomni/data/bioparse/tokenizer/mol_bpe.py", True),
    ("data/bioparse/tokenizer/mol_atom_match.py", "third_party/anewomni/data/bioparse/tokenizer/mol_atom_match.py", False),
    ("data/bioparse/tokenizer/molecule.py", "third_party/anewomni/data/bioparse/tokenizer/molecule.py", True),
    ("data/bioparse/tokenizer/vocabs/chembl_kekulize_300.txt", "third_party/anewomni/data/bioparse/tokenizer/vocabs/chembl_kekulize_300.txt", False),
    ("utils/chem_utils.py", "third_party/anewomni/utils/chem_utils.py", False),
    ("utils/logger.py", "third_party/anewomni/utils/logger.py", False),
    ("utils/singleton.py", "third_party/anewomni/utils/singleton.py", False),
)

REPRESENTATIVE_SMILES = (
    "CCO",
    "c1ccccc1",
    "NCC(=O)O",
    "CC(=O)Oc1ccccc1C(=O)O",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _probe(cwd: str | Path, prefix: str) -> dict[str, Any]:
    code = f'''
import json
from rdkit import Chem
from {prefix} import VOCAB
from {prefix}.tokenizer.tokenize_3d import TOKENIZER

blocks = [
    [symbol, abbreviation, bool(VOCAB.aa_mask[index])]
    for index, (symbol, abbreviation) in enumerate(VOCAB.idx2block)
]
atoms = list(VOCAB.idx2atom)

def tokenize(smiles):
    output = TOKENIZER(Chem.MolFromSmiles(smiles))
    groups = []
    for node_id in output.nodes:
        node = output.get_node(node_id)
        groups.append({{
            "smiles": node.smiles,
            "atom_mapping": sorted((int(key), int(value)) for key, value in node.atom_mapping.items()),
        }})
    return {{
        "groups": groups,
        "reconstructed_smiles": output.to_smiles(),
    }}

print(json.dumps({{
    "num_block_types": VOCAB.get_num_block_type(),
    "num_atom_types": VOCAB.get_num_atom_type(),
    "blocks": blocks,
    "atoms": atoms,
    "tokenizer_size": len(TOKENIZER),
    "tokenizer_kekulize": bool(TOKENIZER.kekulize),
    "tokenizer_cycle_priority": bool(TOKENIZER.tokenizer.cycle_priority),
    "tokenized": {{smiles: tokenize(smiles) for smiles in {list(REPRESENTATIVE_SMILES)!r}}},
}}))
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"probe produced no JSON; stderr={result.stderr}")
    return json.loads(lines[-1])


def _normalize_imports(data: bytes) -> bytes:
    text = data.decode("utf-8")
    for name in ("chem_utils", "logger", "singleton"):
        text = text.replace(f"from ....utils.{name}", f"from utils.{name}")
    return text.encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    entries = []
    for source_relative, target_relative, adapted in VENDOR_MAP:
        source = source_root / source_relative
        target = target_root / target_relative
        if not source.is_file() or not target.is_file():
            raise FileNotFoundError(f"missing vendor pair: {source} / {target}")
        source_bytes = source.read_bytes()
        target_bytes = target.read_bytes()
        comparable = _normalize_imports(target_bytes) if adapted else target_bytes
        if comparable != source_bytes:
            raise AssertionError(f"vendored file differs beyond import adaptation: {target}")
        entries.append({
            "source": source_relative,
            "target": target_relative,
            "import_only_adaptation": adapted,
            "source_sha256": _sha256(source),
            "target_sha256": _sha256(target),
            "normalized_source_equal": True,
        })

    source_probe = _probe(source_root, "data.bioparse")
    target_probe = _probe(target_root, "third_party.anewomni.data.bioparse")
    if source_probe != target_probe:
        raise AssertionError("source and vendored Anew tokenizer/vocabulary outputs differ")
    if source_probe["num_block_types"] != 437 or source_probe["num_atom_types"] != 119:
        raise AssertionError("unexpected Anew vocabulary cardinality")
    if source_probe["tokenizer_size"] != 300 or not source_probe["tokenizer_kekulize"]:
        raise AssertionError("unexpected pinned tokenizer configuration")

    payload = {
        "task": "T1110",
        "status": "passed",
        "source_root": str(source_root),
        "target_root": str(target_root),
        "source_commit": _revision(source_root),
        "target_revision": _revision(target_root),
        "vendor_map": entries,
        "representative_smiles": list(REPRESENTATIVE_SMILES),
        "source_probe": source_probe,
        "target_probe": target_probe,
        "parity": {
            "vocabulary_exact": source_probe == target_probe,
            "tokenization_exact": source_probe["tokenized"] == target_probe["tokenized"],
            "import_only_differences": True,
            "sibling_runtime_import": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
