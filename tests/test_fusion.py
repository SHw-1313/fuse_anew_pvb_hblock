from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from module import dyVAE
from scripts.profile_train_step import build_synthetic_batch


def _model(fusion_mode: str = "anew_block", using_ode: bool = True) -> dyVAE:
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
        using_ode=using_ode,
        backbone="torchmdnet",
        fusion_mode=fusion_mode,
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
    def test_posterior_preserving_mode_constructs(self):
        model = _model("anew_block_pvb_posterior")
        self.assertEqual(model.fusion_mode, "anew_block_pvb_posterior")
        self.assertIsNotNone(model.anew_block_encoder)
        self.assertIsNotNone(model.block_projection)
        self.assertEqual(float(model.block_gate), 0.0)

    def test_corrected_mode_uses_pvb_posterior_not_anew_variance(self):
        first = _model("anew_block_pvb_posterior", using_ode=False).eval()
        second = _model("anew_block_pvb_posterior", using_ode=False).eval()
        with torch.no_grad():
            second_state = second.state_dict()
            for key, value in first.state_dict().items():
                if key in second_state:
                    second_state[key].copy_(value)
            second.anew_block_encoder.Wx_log_var.weight.add_(17.0)
            second.anew_block_encoder.Wx_log_var.bias.add_(11.0)
            first.block_gate.fill_(0.2)
            second.block_gate.fill_(0.2)

        batch = build_synthetic_batch(16, 2, torch.device("cpu"), seed=17)
        outputs = []
        for model in (first, second):
            torch.manual_seed(1234)
            np.random.seed(5678)
            outputs.append(model._train(batch, mode="pretrain"))

        loss_first, parts_first = outputs[0]
        loss_second, parts_second = outputs[1]
        torch.testing.assert_close(loss_first, loss_second, rtol=1e-6, atol=1e-6)
        for first_part, second_part in zip(parts_first, parts_second):
            if torch.is_tensor(first_part):
                torch.testing.assert_close(first_part, second_part, rtol=1e-6, atol=1e-6)
            else:
                self.assertEqual(first_part, second_part)

        loss_second.backward()
        self.assertIsNone(second.anew_block_encoder.Wx_log_var.weight.grad)
        self.assertIsNone(second.anew_block_encoder.Wx_log_var.bias.grad)
        self.assertIsNotNone(second.block_gate.grad)
        self.assertTrue(torch.isfinite(second.block_gate.grad).all())

    def test_complete_gate_zero_posterior_and_inference_parity(self):
        off = _model("off", using_ode=False).eval()
        corrected = _model("anew_block_pvb_posterior", using_ode=False).eval()
        with torch.no_grad():
            corrected_state = corrected.state_dict()
            for key, value in off.state_dict().items():
                if key in corrected_state:
                    corrected_state[key].copy_(value)
            corrected.block_gate.zero_()

        batch = build_synthetic_batch(16, 2, torch.device("cpu"), seed=23)

        def pvb_sample(model):
            torch.manual_seed(20260810)
            np.random.seed(20260810)
            return model._encode_pvb_path(
                batch["atype"],
                batch["btype"],
                batch["x0"],
                batch["x0"],
                batch["abid"],
                batch["mask"],
                batch["edge_mask"],
                batch["bond_index"],
            )

        off_rep, off_kl = pvb_sample(off)
        corrected_rep, corrected_kl = pvb_sample(corrected)
        torch.testing.assert_close(off_rep, corrected_rep, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(off_kl, corrected_kl, rtol=1e-6, atol=1e-6)

        def train_objective(model):
            torch.manual_seed(20260811)
            np.random.seed(20260811)
            return model._train(batch, mode="pretrain")

        off_loss, off_parts = train_objective(off)
        corrected_loss, corrected_parts = train_objective(corrected)
        torch.testing.assert_close(off_loss, corrected_loss, rtol=1e-6, atol=1e-6)
        for off_part, corrected_part in zip(off_parts, corrected_parts):
            torch.testing.assert_close(off_part, corrected_part, rtol=1e-6, atol=1e-6)

        def inference_trace(model):
            calls = []
            original_decode = model.decode

            def wrapped_decode(*args, **kwargs):
                result = original_decode(*args, **kwargs)
                calls.append((args, kwargs, result))
                return result

            with patch.object(model, "decode", side_effect=wrapped_decode):
                torch.manual_seed(20260812)
                np.random.seed(20260812)
                generated = model.inference(batch, sde_step=2)
            return generated, calls

        off_generated, off_calls = inference_trace(off)
        corrected_generated, corrected_calls = inference_trace(corrected)
        self.assertEqual(len(off_calls), len(corrected_calls))
        torch.testing.assert_close(off_generated, corrected_generated, rtol=1e-6, atol=1e-6)

        # The first decoder call receives the posterior source sample. Its
        # output covers both decoder cross-attention branches.
        torch.testing.assert_close(
            off_calls[0][0][2], corrected_calls[0][0][2], rtol=1e-6, atol=1e-6
        )
        for off_output, corrected_output in zip(off_calls[0][2], corrected_calls[0][2]):
            torch.testing.assert_close(off_output, corrected_output, rtol=1e-6, atol=1e-6)
        self.assertIsNone(off_calls[0][1]["block_condition"])
        torch.testing.assert_close(
            corrected_calls[0][1]["block_condition"],
            torch.zeros_like(corrected_calls[0][1]["block_condition"]),
            rtol=0.0,
            atol=0.0,
        )

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
