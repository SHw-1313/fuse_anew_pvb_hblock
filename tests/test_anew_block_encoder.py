from __future__ import annotations

import unittest

import torch
from torch_scatter import scatter_sum

from module.anew_block_encoder import AnewBlockEncoder


def _inputs():
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [13.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
        ]
    )
    atom_type = torch.tensor([5, 6, 7, 5, 6, 7, 5, 6], dtype=torch.long)
    block_type = torch.tensor([118, 119, 120, 121], dtype=torch.long)
    atom_block_id = torch.tensor([0, 0, 1, 2, 2, 3, 3, 3], dtype=torch.long)
    block_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    block_lengths = torch.tensor([2, 1, 2, 3], dtype=torch.long)
    bond_index = torch.tensor(
        [[0, 1, 3, 4, 4, 3, 5, 6], [1, 0, 4, 3, 3, 4, 6, 5]], dtype=torch.long
    )
    return x, atom_type, block_type, atom_block_id, block_batch, block_lengths, bond_index


def _model() -> AnewBlockEncoder:
    torch.manual_seed(7)
    model = AnewBlockEncoder(
        hidden_size=16,
        ffn_size=16,
        edge_size=8,
        n_rbf=4,
        n_layers=1,
        n_head=4,
        k_neighbors=2,
        sparse_k=None,
    )
    model.eval()
    return model


class TestAnewBlockEncoder(unittest.TestCase):
    def test_shapes_and_anew_pooling(self):
        model = _model()
        inputs = _inputs()
        output = model(*inputs)
        self.assertEqual(output["H_atom"].shape, (8, 16))
        self.assertEqual(output["X_atom"].shape, (8, 3))
        self.assertEqual(output["H_block"].shape, (4, 16))
        self.assertEqual(output["X_block"].shape, (4, 3))
        self.assertEqual(output["log_var_block"].shape, (4, 1))

        summed = scatter_sum(output["H_atom"], inputs[3], dim=0)
        expected_h = summed / inputs[5].float().sqrt().unsqueeze(-1)
        torch.testing.assert_close(output["H_block"], expected_h)
        expected_x = torch.stack([output["X_atom"][inputs[3] == i].mean(0) for i in range(4)])
        torch.testing.assert_close(output["X_block"], expected_x)

    def test_batch_isolation(self):
        model = _model()
        inputs = list(_inputs())
        first = model(*inputs)
        inputs[0] = inputs[0].clone()
        inputs[0][3:] += torch.tensor([0.0, 17.0, -9.0])
        second = model(*inputs)
        torch.testing.assert_close(first["H_atom"][:3], second["H_atom"][:3], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(first["X_atom"][:3], second["X_atom"][:3], rtol=1e-5, atol=1e-5)

    def test_se3_equivariance(self):
        model = _model()
        inputs = _inputs()
        original = model(*inputs)
        q, _ = torch.linalg.qr(torch.tensor([[0.3, 0.2, 0.7], [0.5, 0.8, 0.1], [0.4, 0.1, 0.6]]))
        if torch.linalg.det(q) < 0:
            q[:, -1] *= -1
        translation = torch.tensor([2.0, -3.0, 1.5])
        transformed_inputs = list(inputs)
        transformed_inputs[0] = inputs[0] @ q + translation
        transformed = model(*transformed_inputs)
        torch.testing.assert_close(original["H_atom"], transformed["H_atom"], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(
            transformed["X_atom"], original["X_atom"] @ q + translation, rtol=2e-4, atol=2e-4
        )

    def test_gradients_are_finite(self):
        model = _model().train()
        output = model(*_inputs())
        loss = output["H_block"].square().mean() + output["X_block"].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_nonprotein_blocks_fail_clearly(self):
        model = _model()
        with self.assertRaisesRegex(ValueError, "protein-only"):
            model(
                torch.zeros(1, 3),
                torch.tensor([5]),
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([1]),
            )


if __name__ == "__main__":
    unittest.main()
