'''Audit and extract the exact official Anew block embedding table.

This is a provenance tool, not a runtime model dependency. It probes the
pinned Anew checkout in a subprocess whose working directory is that checkout;
it never adds the sibling repository to ``sys.path`` and it never changes the
fused model's configuration.
'''

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import torch


DEFAULT_OFFICIAL = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/"
    "checkpoints/anew_official_model.ckpt"
)
DEFAULT_ENCODER_STATE = (
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase9/"
    "checkpoints/anew_official_encoder_state_dict.pt"
)
DEFAULT_SOURCE_ROOT = "/workspace/AnewOmni"
DEFAULT_EMBEDDING_KEY = "embedding.block_embedding.weight"
DEFAULT_FULL_KEY = "base_model.autoencoder.embedding.block_embedding.weight"


SOURCE_FILES = (
    "LICENSE",
    "models/modules/nn.py",
    "models/IterVAE/model_edge.py",
    "data/bioparse/vocab.py",
    "data/bioparse/const.py",
    "data/bioparse/tokenizer/tokenize_3d.py",
    "data/bioparse/tokenizer/mol_bpe.py",
    "data/bioparse/tokenizer/vocabs/chembl_kekulize_300.txt",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _revision(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_probe(source_root: str | Path, code: str, *args: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *map(str, args)],
        cwd=str(source_root),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"source probe produced no JSON; stderr={result.stderr}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"source probe did not end in JSON: {lines[-3:]}; stderr={result.stderr}"
        ) from exc


def _probe_official_model(source_root: str | Path, checkpoint: str | Path) -> dict[str, Any]:
    return _json_probe(
        source_root,
        r'''
import hashlib
import json
import sys
import torch

path = sys.argv[1]
model = torch.load(path, map_location="cpu", weights_only=False)
state = model.state_dict()
matches = [key for key in state if key.endswith("embedding.block_embedding.weight")]
if len(matches) != 1:
    raise RuntimeError(f"expected one official block embedding key, found {matches}")
key = matches[0]
tensor = state[key].detach().cpu().contiguous()
print(json.dumps({
    "payload_type": f"{type(model).__module__}.{type(model).__name__}",
    "state_key_count": len(state),
    "full_state_key": key,
    "shape": list(tensor.shape),
    "dtype": str(tensor.dtype),
    "numel": tensor.numel(),
    "tensor_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
}))
''',
        str(checkpoint),
    )


def _probe_vocab(source_root: str | Path) -> dict[str, Any]:
    return _json_probe(
        source_root,
        r'''
import hashlib
import importlib
import json
from pathlib import Path

from data.bioparse import VOCAB

tokenize_module = importlib.import_module("data.bioparse.tokenizer.tokenize_3d")
tokenizer_vocab = (
    Path(tokenize_module.__file__).resolve().parent
    / "vocabs"
    / "chembl_kekulize_300.txt"
)
blocks = [
    {
        "index": index,
        "symbol": symbol,
        "abbreviation": abbreviation,
        "is_amino_acid": bool(VOCAB.aa_mask[index]),
    }
    for index, (symbol, abbreviation) in enumerate(VOCAB.idx2block)
]
atoms = [
    {"index": index, "symbol": symbol}
    for index, symbol in enumerate(VOCAB.idx2atom)
]
order_bytes = json.dumps(
    blocks, ensure_ascii=True, separators=(",", ":")
).encode("utf-8")
print(json.dumps({
    "vocab_class": "data.bioparse.vocab.MoleculeVocab",
    "num_block_types": VOCAB.get_num_block_type(),
    "num_atom_types": VOCAB.get_num_atom_type(),
    "block_vocab": blocks,
    "atom_vocab": atoms,
    "amino_acid_indices": [entry["index"] for entry in blocks if entry["is_amino_acid"]],
    "tokenizer_method": "PS_kekulized_300",
    "tokenizer_vocab_path": str(tokenizer_vocab),
    "tokenizer_vocab_sha256": hashlib.sha256(tokenizer_vocab.read_bytes()).hexdigest(),
    "vocab_order_sha256": hashlib.sha256(order_bytes).hexdigest(),
}))
''',
    )


def _load_state(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a state dict at {path}, got {type(payload)!r}")
    state = payload.get("state_dict", payload.get("model_state_dict", payload))
    if not isinstance(state, dict):
        raise TypeError(f"expected state dict payload at {path}")
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", default=DEFAULT_OFFICIAL)
    parser.add_argument("--encoder-state", default=DEFAULT_ENCODER_STATE)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--embedding-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    official = Path(args.official_checkpoint)
    encoder_state_path = Path(args.encoder_state)
    embedding_output = Path(args.embedding_output)
    report_output = Path(args.output)
    source_root = Path(args.source_root)
    if not official.is_file() or not encoder_state_path.is_file():
        raise FileNotFoundError("official checkpoint and derived encoder state are required")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if not args.verify_only and (embedding_output.exists() or report_output.exists()):
        raise FileExistsError("refusing to overwrite an existing T1109 artifact")

    official_probe = _probe_official_model(source_root, official)
    vocab_probe = _probe_vocab(source_root)
    encoder_state = _load_state(encoder_state_path)
    if DEFAULT_EMBEDDING_KEY not in encoder_state:
        raise KeyError(f"missing {DEFAULT_EMBEDDING_KEY} in {encoder_state_path}")
    tensor = encoder_state[DEFAULT_EMBEDDING_KEY]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("official embedding value is not a tensor")

    shape = list(tensor.shape)
    dtype = str(tensor.dtype)
    tensor_hash = _tensor_sha256(tensor)
    if official_probe["full_state_key"] != DEFAULT_FULL_KEY:
        raise AssertionError(official_probe)
    if official_probe["shape"] != shape or official_probe["dtype"] != dtype:
        raise AssertionError("full official tensor metadata disagrees with derived state")
    if official_probe["tensor_sha256"] != tensor_hash:
        raise AssertionError("full official tensor and derived encoder tensor differ")
    if shape != [vocab_probe["num_block_types"], 512]:
        raise AssertionError("block embedding shape does not match the complete vocabulary")
    if dtype != "torch.float32":
        raise AssertionError(f"unexpected official embedding dtype: {dtype}")
    if vocab_probe["num_block_types"] != 437 or vocab_probe["num_atom_types"] != 119:
        raise AssertionError("unexpected pinned Anew vocabulary cardinality")
    if vocab_probe["amino_acid_indices"] != list(range(1, 21)):
        raise AssertionError("Anew amino-acid vocabulary order is not the pinned order")

    if not args.verify_only:
        embedding_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {DEFAULT_EMBEDDING_KEY: tensor.detach().cpu()}}, embedding_output)
    if not embedding_output.is_file():
        raise FileNotFoundError(embedding_output)
    extracted_state = _load_state(embedding_output)
    if set(extracted_state) != {DEFAULT_EMBEDDING_KEY}:
        raise AssertionError("extracted artifact contains unexpected keys")
    extracted = extracted_state[DEFAULT_EMBEDDING_KEY]
    if not torch.equal(extracted, tensor):
        raise AssertionError("extracted artifact is not bitwise equal to official table")

    source_hashes = {}
    for relative in SOURCE_FILES:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[relative] = _sha256(path)
    payload = {
        "task": "T1109",
        "status": "passed",
        "source_root": str(source_root),
        "source_commit": _revision(source_root),
        "official_checkpoint": {
            "path": str(official),
            "sha256": _sha256(official),
            **official_probe,
        },
        "derived_encoder_state": {
            "path": str(encoder_state_path),
            "sha256": _sha256(encoder_state_path),
            "key": DEFAULT_EMBEDDING_KEY,
            "key_count": len(encoder_state),
            "shape": shape,
            "dtype": dtype,
            "tensor_sha256": tensor_hash,
        },
        "extracted_artifact": {
            "path": str(embedding_output),
            "sha256": _sha256(embedding_output),
            "key": DEFAULT_EMBEDDING_KEY,
            "shape": list(extracted.shape),
            "dtype": str(extracted.dtype),
            "tensor_sha256": _tensor_sha256(extracted),
        },
        "block_embedding_contract": {
            "class": "data.bioparse.vocab.MoleculeVocab + models.modules.nn.BlockEmbedding",
            "official_full_state_key": DEFAULT_FULL_KEY,
            "derived_state_key": DEFAULT_EMBEDDING_KEY,
            "num_block_types": vocab_probe["num_block_types"],
            "embed_dim": shape[1],
            "dtype": dtype,
            "exact_full_to_derived_tensor": True,
        },
        "vocabulary": vocab_probe,
        "source_file_sha256": source_hashes,
        "verify_only": bool(args.verify_only),
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
