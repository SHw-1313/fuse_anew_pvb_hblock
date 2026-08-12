import unittest

import torch

from module.graph import edge_inclusion_mask


class EmptyBondGraphTest(unittest.TestCase):
    def test_empty_bond_index_returns_zero_edge_mask(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        bond_index = torch.empty((2, 0), dtype=torch.long)
        result = edge_inclusion_mask(edge_index, bond_index)
        self.assertEqual(result.tolist(), [0, 0])
        self.assertEqual(result.dtype, torch.long)

    def test_empty_edge_index_returns_empty_mask(self):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        bond_index = torch.tensor([[0], [1]], dtype=torch.long)
        result = edge_inclusion_mask(edge_index, bond_index)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
