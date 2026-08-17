from __future__ import annotations

import json
import unittest
from pathlib import Path

from third_party.anewomni.data.bioparse import VOCAB


REPORT = Path(
    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/phase11/checkpoints/"
    "t1110_anew_vendor_provenance.json"
)


class TestAnewVendorTokenizer(unittest.TestCase):
    def test_vendored_vocab_is_namespaced_and_complete(self):
        self.assertEqual(VOCAB.get_num_block_type(), 437)
        self.assertEqual(VOCAB.get_num_atom_type(), 119)
        self.assertEqual([VOCAB.idx_to_abrv(i) for i in range(1, 21)], [
            "GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "TYR",
            "ASP", "HIS", "ASN", "GLU", "LYS", "GLN", "MET", "ARG",
            "SER", "THR", "CYS", "PRO",
        ])

    @unittest.skipUnless(REPORT.is_file(), "T1110 parity report not materialized")
    def test_source_target_parity_report(self):
        payload = json.loads(REPORT.read_text())
        self.assertTrue(payload["parity"]["vocabulary_exact"])
        self.assertTrue(payload["parity"]["tokenization_exact"])
        self.assertTrue(payload["parity"]["import_only_differences"])
        self.assertFalse(payload["parity"]["sibling_runtime_import"])
