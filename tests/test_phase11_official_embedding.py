from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch


ARTIFACT = Path(
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/"
    "anew_official_block_embedding.pt"
)
REPORT = Path(
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/"
    "t1109_official_block_embedding_provenance.json"
)
KEY = "embedding.block_embedding.weight"


@unittest.skipUnless(ARTIFACT.is_file() and REPORT.is_file(), "T1109 artifact not materialized")
class TestPhase11OfficialEmbedding(unittest.TestCase):
    def test_exact_table_and_vocab_provenance(self):
        payload = json.loads(REPORT.read_text())
        state = torch.load(ARTIFACT, map_location="cpu", weights_only=True)["state_dict"]
        self.assertEqual(set(state), {KEY})
        table = state[KEY]
        self.assertEqual(tuple(table.shape), (437, 512))
        self.assertEqual(table.dtype, torch.float32)
        self.assertEqual(payload["block_embedding_contract"]["num_block_types"], 437)
        self.assertEqual(payload["block_embedding_contract"]["embed_dim"], 512)
        self.assertEqual(payload["official_checkpoint"]["full_state_key"],
                         "base_model.autoencoder.embedding.block_embedding.weight")
        self.assertTrue(payload["block_embedding_contract"]["exact_full_to_derived_tensor"])
        blocks = payload["vocabulary"]["block_vocab"]
        self.assertEqual(len(blocks), 437)
        self.assertEqual([entry["index"] for entry in blocks], list(range(437)))
        self.assertEqual([entry["abbreviation"] for entry in blocks[1:21]], [
            "GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "TYR",
            "ASP", "HIS", "ASN", "GLU", "LYS", "GLN", "MET", "ARG",
            "SER", "THR", "CYS", "PRO",
        ])
        self.assertEqual(payload["vocabulary"]["amino_acid_indices"], list(range(1, 21)))
        self.assertEqual(
            payload["extracted_artifact"]["tensor_sha256"],
            payload["derived_encoder_state"]["tensor_sha256"],
        )
