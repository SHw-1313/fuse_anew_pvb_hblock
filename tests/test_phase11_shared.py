from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

import numpy as np
import torch

from module import dyVAE
from module.shared_hblock import SharedHBlockAdapter, pool_pvb_h_atom
from scripts.profile_train_step import build_synthetic_batch
from scripts.profile_training_paths import build_model as build_real_model
from tests.test_fusion import _decode_inputs
from utils.checkpoint import load_resume_checkpoint, load_role_checkpoint
from utils.fusion_training import configure_fusion_parameters, fusion_parameter_groups


class TestPhase11SharedMode(unittest.TestCase):
    @staticmethod
    def make_model(mode: str = "pvb_shared_hblock", using_ode: bool = False) -> dyVAE:
        torch.manual_seed(101)
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
            fusion_mode=mode,
        )

    def test_mode_is_pvb_only_and_constructs_without_anew(self):
        model = self.make_model()
        self.assertEqual(model.fusion_mode, "pvb_shared_hblock")
        self.assertIsNone(model.anew_block_encoder)
        self.assertEqual(model.shared_hblock_adapter.hidden_dim, 16)
        self.assertEqual(model.shared_hblock_adapter.rank, 32)
        self.assertGreater(model.shared_hblock_adapter.projection[-1].weight.abs().sum().item(), 0.0)
        self.assertEqual(model.shared_hblock_gate.item(), 0.0)
        self.assertFalse(any(name.startswith("anew_block_encoder.") for name, _ in model.named_parameters()))

    def test_pooling_matches_anew_variance_preserving_rule(self):
        h_atom = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            requires_grad=True,
        )
        atom_block_id = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
        block_lengths = torch.tensor([3, 2, 1], dtype=torch.long)
        pooled = pool_pvb_h_atom(h_atom, atom_block_id, block_lengths)
        expected = torch.stack(
            [
                h_atom[:3].sum(dim=0) / torch.sqrt(torch.tensor(3.0)),
                h_atom[3:5].sum(dim=0) / torch.sqrt(torch.tensor(2.0)),
                h_atom[5],
            ]
        )
        torch.testing.assert_close(pooled, expected)
        self.assertFalse(pooled.requires_grad)

    def test_adapter_broadcasts_and_detaches_source_feature(self):
        adapter = SharedHBlockAdapter(hidden_dim=4, rank=2)
        h_atom = torch.randn(5, 4, requires_grad=True)
        atom_block_id = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        block_lengths = torch.tensor([2, 3], dtype=torch.long)
        output = adapter(h_atom, atom_block_id, block_lengths)
        self.assertEqual(output["H_block"].shape, (2, 4))
        self.assertEqual(output["condition_atom"].shape, (5, 4))
        torch.testing.assert_close(output["condition_atom"][:2], output["condition_block"][0].expand(2, -1))
        torch.testing.assert_close(output["condition_atom"][2:], output["condition_block"][1].expand(3, -1))
        self.assertFalse(output["H_block"].requires_grad)
        self.assertTrue(torch.equal(output["atom_block_id"], atom_block_id))

    def test_shared_adapter_stage_has_only_new_parameters(self):
        model = self.make_model()
        configure_fusion_parameters(model, "adapter")
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        expected = {
            name for name, _ in model.named_parameters()
            if name == "shared_hblock_gate" or name.startswith("shared_hblock_adapter.")
        }
        self.assertEqual(trainable, expected)
        groups = fusion_parameter_groups(model, 1e-4, 1e-5, 2e-4)
        self.assertEqual({group["name"] for group in groups}, {"projector_gate"})

    def test_shared_condition_receives_finite_gradients_after_gate_moves(self):
        model = self.make_model(using_ode=False).train()
        configure_fusion_parameters(model, "adapter")
        with torch.no_grad():
            model.shared_hblock_gate.fill_(0.2)
        batch = build_synthetic_batch(16, 2, torch.device("cpu"), seed=33)
        torch.manual_seed(20260813)
        np.random.seed(20260813)
        loss, _ = model._train(batch, mode="pretrain")
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertGreater(float(model.shared_hblock_gate.grad.abs().sum()), 0.0)
        for name, parameter in trainable.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


    def test_encode_default_contract_and_optional_shared_state(self):
        model = self.make_model().eval()
        batch = build_synthetic_batch(16, 2, torch.device("cpu"), seed=31)

        torch.manual_seed(2026)
        legacy = model._encode_pvb_path(
            batch["atype"], batch["btype"], batch["x0"], batch["x0"],
            batch["abid"], batch["mask"], batch["edge_mask"], batch["bond_index"],
        )
        torch.manual_seed(2026)
        shared = model._encode_pvb_path(
            batch["atype"], batch["btype"], batch["x0"], batch["x0"],
            batch["abid"], batch["mask"], batch["edge_mask"], batch["bond_index"],
            return_state=True,
        )

        self.assertEqual(len(legacy), 2)
        self.assertEqual(len(shared), 3)
        torch.testing.assert_close(legacy[0], shared[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(legacy[1], shared[1], rtol=0.0, atol=0.0)
        state = shared[2]
        self.assertEqual(set(state), {"h_atom", "vec_atom", "log_var_pvb"})
        self.assertEqual(state["h_atom"].shape, (16, 16))
        self.assertEqual(state["vec_atom"].shape[:2], (16, 3))
        self.assertEqual(state["log_var_pvb"].shape, (16, 3))

    def test_shared_mode_training_uses_pvb_path(self):
        off = self.make_model("off").eval()
        shared = self.make_model("pvb_shared_hblock").eval()
        with torch.no_grad():
            shared.load_state_dict({**shared.state_dict(), **off.state_dict()}, strict=True)
        batch = build_synthetic_batch(16, 2, torch.device("cpu"), seed=32)
        outputs = []
        for model in (off, shared):
            torch.manual_seed(2027)
            np.random.seed(2027)
            outputs.append(model._train(batch, mode="pretrain"))
        torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0.0, atol=0.0)
        for left, right in zip(outputs[0][1], outputs[1][1]):
            if torch.is_tensor(left):
                torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
            else:
                self.assertEqual(left, right)

    def test_gate_zero_full_objective_and_inference_parity(self):
        off = self.make_model("off", using_ode=False).eval()
        shared = self.make_model("pvb_shared_hblock", using_ode=False).eval()
        with torch.no_grad():
            shared_state = shared.state_dict()
            for key, value in off.state_dict().items():
                if key in shared_state:
                    shared_state[key].copy_(value)
            shared.shared_hblock_gate.zero_()

        batch = build_synthetic_batch(16, 2, torch.device("cpu"), seed=42)
        for model in (off, shared):
            torch.manual_seed(20260810)
            np.random.seed(20260810)
            loss, parts = model._train(batch, mode="pretrain")
            if model is off:
                off_loss, off_parts = loss, parts
            else:
                torch.testing.assert_close(loss, off_loss, rtol=0.0, atol=0.0)
                for left, right in zip(parts, off_parts):
                    if torch.is_tensor(left):
                        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
                    else:
                        self.assertEqual(left, right)

        for model in (off, shared):
            torch.manual_seed(20260811)
            np.random.seed(20260811)
            generated = model.inference(batch, sde_step=2)
            if model is off:
                off_generated = generated
            else:
                torch.testing.assert_close(generated, off_generated, rtol=0.0, atol=0.0)

    def test_gate_zero_shared_posterior_state_and_decoder_output_match(self):
        off = self.make_model("off", using_ode=True).eval()
        shared = self.make_model("pvb_shared_hblock", using_ode=True).eval()
        with torch.no_grad():
            shared_state = shared.state_dict()
            for key, value in off.state_dict().items():
                if key in shared_state:
                    shared_state[key].copy_(value)
            shared.shared_hblock_gate.zero_()

        batch = build_synthetic_batch(16, 2, torch.device("cpu"), seed=43)
        encode_args = (
            batch["atype"],
            batch["btype"],
            batch["x0"],
            batch["x0"],
            batch["abid"],
            batch["mask"],
            batch["edge_mask"],
            batch["bond_index"],
        )
        encoded = []
        for model in (off, shared):
            torch.manual_seed(20260813)
            np.random.seed(20260813)
            encoded.append(
                model._encode_pvb_path(
                    *encode_args,
                    return_state=True,
                )
            )
        off_rep, off_kl, off_state = encoded[0]
        shared_rep, shared_kl, shared_state = encoded[1]
        torch.testing.assert_close(off_rep, shared_rep, rtol=0.0, atol=0.0)
        torch.testing.assert_close(off_kl, shared_kl, rtol=0.0, atol=0.0)
        self.assertEqual(set(off_state), {"h_atom", "vec_atom", "log_var_pvb"})
        for key in off_state:
            torch.testing.assert_close(
                off_state[key], shared_state[key], rtol=0.0, atol=0.0
            )

        inputs = _decode_inputs()
        zero_condition = torch.zeros(2, 16)
        with torch.no_grad():
            off_output = off.decode(**inputs)
            shared_output = shared.decode(
                **inputs,
                post_cross_condition=zero_condition,
            )
        torch.testing.assert_close(off_output, shared_output, rtol=0.0, atol=0.0)

    def test_phase9_and_phase10_checkpoint_paths_remain_loadable(self):
        cases = (
            (
                "anew_block",
                Path(
                    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/"
                    "phase9/checkpoints/source_frozen_epoch1_best.ckpt"
                ),
                "resume",
            ),
            (
                "anew_block",
                Path(
                    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/"
                    "phase9/checkpoints/legacy_fused_state_dict.pt"
                ),
                "anew",
            ),
            (
                "anew_block_pvb_posterior",
                Path(
                    "/output/pvb_cross_dataset_20260810/hblock_adapter_v1/"
                    "phase10/checkpoints/anew_block_pvb_posterior_best.ckpt"
                ),
                "resume",
            ),
        )
        missing = [str(path) for _, path, _ in cases if not path.is_file()]
        if missing:
            self.skipTest("protected Phase 9/10 artifacts unavailable: " + ", ".join(missing))
        for mode, path, role in cases:
            with self.subTest(mode=mode, role=role, path=str(path)):
                with contextlib.redirect_stdout(io.StringIO()):
                    model = build_real_model(mode)
                    if role == "resume":
                        report, _ = load_resume_checkpoint(
                            model, path, min_coverage=1.0
                        )
                    else:
                        report = load_role_checkpoint(
                            model, path, "anew", min_coverage=1.0
                        )
                self.assertEqual(report.coverage, 1.0)
                self.assertFalse(report.missing_keys)
                self.assertFalse(report.shape_mismatches)
                if role == "resume":
                    self.assertFalse(report.unexpected_keys)

    def test_post_cross_condition_is_added_once_after_branch_merge(self):
        model = self.make_model("off", using_ode=True).eval()
        inputs = _decode_inputs()
        captured = []

        def capture(module, args):
            captured.append(args[0].detach().clone())

        handle = model.decoder.attention_layers[0].register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                model.decode(**inputs)
                condition = torch.full((2, 16), 0.125)
                model.decode(**inputs, post_cross_condition=condition)
        finally:
            handle.remove()

        self.assertEqual(len(captured), 2)
        torch.testing.assert_close(
            captured[1] - captured[0],
            torch.full_like(captured[0], 0.125),
            rtol=0.0,
            atol=0.0,
        )

    def test_post_cross_condition_preserves_decoder_se3_equivariance(self):
        model = self.make_model("off", using_ode=True).eval()
        inputs = _decode_inputs()
        condition = torch.randn(2, 16)
        q, _ = torch.linalg.qr(
            torch.tensor([[0.3, 0.2, 0.7], [0.5, 0.8, 0.1], [0.4, 0.1, 0.6]])
        )
        if torch.linalg.det(q) < 0:
            q[:, -1] *= -1
        transformed = dict(inputs)
        transformed["x"] = inputs["x"] @ q + torch.tensor([2.0, -3.0, 1.5])
        transformed["edge_vec_0"] = inputs["edge_vec_0"] @ q
        transformed["edge_vec_t"] = inputs["edge_vec_t"] @ q
        with torch.no_grad():
            original = model.decode(**inputs, post_cross_condition=condition)
            rotated = model.decode(**transformed, post_cross_condition=condition)
        torch.testing.assert_close(rotated, original @ q, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
