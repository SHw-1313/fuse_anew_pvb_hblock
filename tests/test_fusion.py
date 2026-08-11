from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from module import dyVAE
from scripts.profile_train_step import build_synthetic_batch


def _model() -> dyVAE:
    torch.manual_seed(13)
    return dyVAE(
        hidden_dim=16,
        ffn_dim=32,
        rbf_dim=4,
        heads=4,
        layers=1,
        cutoff_lower=0.0,
        cutoff_upper=10.0,
        cutoff_H=3.5,
        k_neighbors=8,
        coord_prior_var=0.5,
        using_ode=True,
        backbone="torchmdnet",
        fusion_mode="anew_block",
        anew_encoder_config={
            "hidden_size": 16,
            "ffn_size": 16,
            "edge_size": 8,
            "n_rbf": 4,
            "n_layers": 1,
            "n_head": 4,
            "k_neighbors": 2,
            "sparse_k": None,
        },
    )


def _decode_inputs():
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_vec = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edge_weight = torch.ones(2)
    return {
        "z": torch.tensor([5, 6], dtype=torch.long),
        "b": torch.tensor([118, 118], dtype=torch.long),
        "x": x,
        "t": torch.full((2, 1), 0.3),
        "batch": torch.zeros(2, dtype=torch.long),
        "edge_index": edge_index,
        "edge_weight_0": edge_weight,
        "edge_vec_0": edge_vec,
        "edge_weight_t": edge_weight,
        "edge_vec_t": edge_vec,
        "bond_type": torch.zeros(2, dtype=torch.long),
    }


class TestFusion(unittest.TestCase):
    def test_gate_zero_decoder_parity(self):
        model = _model().eval()
        batch = build_synthetic_batch(16, 2, torch.device("cpu"), seed=4)
        with torch.no_grad():
            block_output = model.anew_block_encoder(
                x_atom=batch["x0"],
                atom_type=batch["atype"],
                block_type=batch["block_type"],
                atom_block_id=batch["atom_block_id"],
                block_batch=batch["block_batch"],
                block_lengths=batch["block_lengths"],
                bond_index=batch["bond_index"],
            )
            condition = model._project_block_condition(block_output)
            self.assertEqual(float(model.block_gate), 0.0)
            torch.testing.assert_close(condition, torch.zeros_like(condition))

            inputs = _decode_inputs()
            baseline = model.decode(**inputs)
            gated = model.decode(**inputs, block_condition=condition[:2])
        torch.testing.assert_close(gated, baseline, rtol=1e-6, atol=1e-6)

    def test_projector_and_gate_receive_finite_gradients(self):
        model = _model().train()
        with torch.no_grad():
            model.block_gate.fill_(0.2)
        batch = build_synthetic_batch(16, 2, torch.device("cpu"), seed=9)
        loss, _ = model._train(batch, mode="pretrain")
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.block_gate.grad)
        self.assertTrue(torch.isfinite(model.block_gate.grad).all())
        projection_gradients = [
            parameter.grad
            for parameter in model.block_projection.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(projection_gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in projection_gradients))

    def test_fused_source_mean_is_input_structure(self):
        model = _model().eval()
        batch = build_synthetic_batch(16, 2, torch.device("cpu"), seed=12)
        with patch("module.model.torch.randn_like", side_effect=torch.zeros_like):
            with torch.no_grad():
                x_rep, _, output = model._encode_anew_block(
                    batch["x0"], batch, torch.ones(16, dtype=torch.bool)
                )
        torch.testing.assert_close(x_rep, batch["x0"])
        with torch.no_grad():
            _, _, output = model._encode_anew_block(
                batch["x0"], batch, torch.ones(16, dtype=torch.bool)
            )
        # The returned Anew coordinates are available for diagnostics, but the
        # fused path's source mean is the input tensor by construction.
        self.assertEqual(output["X_atom"].shape, batch["x0"].shape)


if __name__ == "__main__":
    unittest.main()
