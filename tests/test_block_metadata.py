from __future__ import annotations

import warnings
import unittest

import torch

from data.block_metadata import LegacyBlockMetadataWarning, annotate_block_metadata
from data.collate import collate_fn
from data.subgraph import graph_cut


def _item(atom_block_id=None):
    item = {
        "atype": [5, 6, 7],
        "btype": [118, 118, 119],
        "edge_mask": [0, 0, 0],
        "mask": [1, 1, 1],
        "x0": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        "b0": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        "bond_index": [[0, 1], [1, 0]],
    }
    if atom_block_id is not None:
        item["atom_block_id"] = atom_block_id
        item["block_type"] = [118, 119]
        item["block_lengths"] = [2, 1]
    return item


class TestBlockMetadata(unittest.TestCase):
    def test_multi_sample_offsets_and_repeated_residue_types(self):
        first = _item([0, 0, 1])
        second = _item([0, 0, 1])
        batch = collate_fn([[first, second]])

        torch.testing.assert_close(batch["atom_block_id"], torch.tensor([0, 0, 1, 2, 2, 3]))
        torch.testing.assert_close(batch["block_type"], torch.tensor([118, 119, 118, 119]))
        torch.testing.assert_close(batch["block_batch"], torch.tensor([0, 0, 1, 1]))
        torch.testing.assert_close(batch["block_lengths"], torch.tensor([2, 1, 2, 1]))
        torch.testing.assert_close(batch["abid"], torch.tensor([0, 0, 0, 1, 1, 1]))

    def test_legacy_fallback_is_cpu_and_warns(self):
        item = _item()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metadata = collate_fn([[item]])
        self.assertTrue(any(issubclass(w.category, LegacyBlockMetadataWarning) for w in caught))
        torch.testing.assert_close(metadata["atom_block_id"], torch.tensor([0, 0, 1]))
        self.assertEqual(item["atom_block_id"], [0, 0, 1])

    def test_preprocessing_annotation_is_explicit(self):
        item = _item()
        annotated = annotate_block_metadata(item)
        self.assertEqual(annotated["atom_block_id"], [0, 0, 1])
        self.assertEqual(annotated["block_type"], [118, 119])
        self.assertEqual(annotated["block_lengths"], [2, 1])

    def test_crop_expands_partial_outer_block(self):
        x = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.1, 0.0, 0.0]]
        ).numpy()
        block_id = torch.tensor([0, 0, 1, 1, 2, 2]).numpy()
        indices, mask = graph_cut(x, xc=x[2], radius_min=0.5, radius_max=1.5, block_id=block_id)
        self.assertIn(2, indices.tolist())
        self.assertIn(3, indices.tolist())
        for block in (0, 1):
            selected = indices[block_id[indices] == block]
            expected = (block_id == block).sum()
            self.assertEqual(len(selected), expected)
        self.assertEqual(len(mask), len(indices))


if __name__ == "__main__":
    unittest.main()
