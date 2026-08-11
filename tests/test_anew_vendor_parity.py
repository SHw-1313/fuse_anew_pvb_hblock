"""Numerical parity checks for vendored Anew components.

The upstream and vendored implementations are executed in separate Python
processes with their respective repository roots as the working directory.
This avoids importing the read-only sibling repository into the target process
and avoids runtime path injection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch


SOURCE_ROOT = Path(os.environ.get("ANEWOMNI_SOURCE", "/workspace/AnewOmni"))
TARGET_ROOT = Path(__file__).resolve().parents[1]


def _run_payload(root: Path, source: str) -> dict:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"payload failed in {root}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return json.loads(result.stdout)


_EMBEDDING_PAYLOAD = r'''
import json
import torch
from models.modules.nn import BlockEmbedding

torch.set_num_threads(1)
torch.manual_seed(1234)
model = BlockEmbedding(num_block_type=9, num_atom_type=11, embed_size=7)
S = torch.tensor([1, 4, 2], dtype=torch.long)
A = torch.tensor([2, 3, 5, 7, 1, 4], dtype=torch.long)
block_id = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
print(json.dumps(model(S, A, block_id).detach().cpu().tolist()))
'''


_VENDORED_EMBEDDING_PAYLOAD = _EMBEDDING_PAYLOAD.replace(
    "from models.modules.nn import BlockEmbedding",
    "from third_party.anewomni.models.modules.nn import BlockEmbedding",
)


_EPT_PAYLOAD = r'''
import json
import torch
from models.modules.EPT.ept import XTransEncoderAct

torch.set_num_threads(1)
torch.manual_seed(5678)
model = XTransEncoderAct(
    hidden_size=16,
    ffn_size=16,
    n_rbf=4,
    cutoff=10.0,
    edge_size=4,
    n_layers=1,
    n_head=4,
    pre_norm=True,
    use_edge_feat=True,
    sparse_k=None,
    efficient=False,
    vector_act='layernorm',
)
H = torch.randn(4, 16)
Z = torch.tensor([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.], [4., 0., 0.]])
block_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
batch_id = torch.tensor([0, 0], dtype=torch.long)
edges = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
edge_attr = torch.randn(4, 4)
H_out, V_out = model(H, Z, block_id, batch_id, edges, edge_attr)
print(json.dumps({'H': H_out.detach().cpu().tolist(), 'V': V_out.detach().cpu().tolist()}))
'''


_VENDORED_EPT_PAYLOAD = _EPT_PAYLOAD.replace(
    "from models.modules.EPT.ept import XTransEncoderAct",
    "from third_party.anewomni.models.modules.EPT.ept import XTransEncoderAct",
)


class TestAnewVendorParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SOURCE_ROOT.exists():
            raise unittest.SkipTest(f"AnewOmni source not found at {SOURCE_ROOT}")

    def test_block_embedding_matches_source(self) -> None:
        source = _run_payload(SOURCE_ROOT, _EMBEDDING_PAYLOAD)
        vendored = _run_payload(TARGET_ROOT, _VENDORED_EMBEDDING_PAYLOAD)
        torch.testing.assert_close(torch.tensor(source), torch.tensor(vendored), rtol=0.0, atol=0.0)

    def test_ept_matches_source(self) -> None:
        source = _run_payload(SOURCE_ROOT, _EPT_PAYLOAD)
        vendored = _run_payload(TARGET_ROOT, _VENDORED_EPT_PAYLOAD)
        torch.testing.assert_close(torch.tensor(source["H"]), torch.tensor(vendored["H"]), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(torch.tensor(source["V"]), torch.tensor(vendored["V"]), rtol=1e-6, atol=1e-6)
