from __future__ import annotations

import unittest

import torch

from data.dataset_wrapper import DynamicBatchWrapper
from module import dyVAE
from scripts.profile_train_step import build_synthetic_batch
from utils.fusion_training import (
    configure_fusion_parameters,
    fusion_gradient_norms,
    fusion_parameter_groups,
)


def _model(n_layers: int = 3, fusion_mode: str = "anew_block") -> dyVAE:
    return dyVAE(
        hidden_dim=16,
        ffn_dim=32,
        rbf_dim=4,
        heads=4,
        layers=1,
        cutoff_lower=0.0,
        cutoff_upper=10.0,
        k_neighbors=8,
        using_ode=True,
        fusion_mode=fusion_mode,
        anew_encoder_config={
            "hidden_size": 16,
            "ffn_size": 16,
            "edge_size": 8,
            "n_rbf": 4,
            "n_layers": n_layers,
            "n_head": 4,
            "k_neighbors": 2,
            "sparse_k": None,
        },
    )


class TestTrainingStages(unittest.TestCase):
    def test_stage_a_freezes_anew_and_groups_rates(self):
        model = _model()
        configure_fusion_parameters(model, "A")
        self.assertTrue(model.block_gate.requires_grad)
        self.assertTrue(model.decoder.atom_embedding.weight.requires_grad)
        self.assertFalse(model.encoder.atom_embedding.weight.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in model.anew_block_encoder.parameters()))
        groups = fusion_parameter_groups(model, 1e-4, 1e-5, 2e-4)
        self.assertEqual({group["name"] for group in groups}, {"pvb", "projector_gate"})

    def test_adapter_trains_only_new_projection_and_gate(self):
        model = _model()
        configure_fusion_parameters(model, "adapter")
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(
            trainable,
            {"block_gate", "block_projection.0.weight", "block_projection.0.bias",
             "block_projection.1.weight", "block_projection.1.bias"},
        )

    def test_source_frozen_uses_checkpoint_key_union(self):
        model = _model()
        loaded = {"decoder.atom_embedding.weight", "anew_block_encoder.embedding.atom_embedding.weight"}
        configure_fusion_parameters(model, "source_frozen", source_keys=loaded)
        self.assertFalse(model.decoder.atom_embedding.weight.requires_grad)
        self.assertFalse(model.anew_block_encoder.embedding.atom_embedding.weight.requires_grad)
        self.assertTrue(model.block_gate.requires_grad)
        self.assertTrue(model.decoder.bond_embedding.weight.requires_grad)
    def test_source_frozen_optimizer_is_exact_complement(self):
        model = _model()
        loaded = {
            name for name, _ in model.named_parameters()
            if name.startswith("decoder.") or name.startswith("anew_block_encoder.")
        }
        configure_fusion_parameters(model, "source_frozen", source_keys=loaded)
        expected = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if name not in loaded
        }
        groups = fusion_parameter_groups(model, 1e-4, 1e-5, 2e-4)
        actual = {
            id(parameter)
            for group in groups
            for parameter in group["params"]
        }
        self.assertEqual(actual, expected)
        self.assertEqual(
            {
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
            },
            {
                name for name, _ in model.named_parameters()
                if name not in loaded
            },
        )

    def test_stage_b_unfreezes_only_last_ept_layers(self):
        model = _model(n_layers=3)
        configure_fusion_parameters(model, "B", unfreeze_ept_layers=2)
        for name, parameter in model.named_parameters():
            if name.startswith("anew_block_encoder.encoder.encoder.layer_0."):
                self.assertFalse(parameter.requires_grad)
            if name.startswith("anew_block_encoder.encoder.encoder.layer_1."):
                self.assertTrue(parameter.requires_grad)
            if name.startswith("anew_block_encoder.encoder.encoder.layer_2."):
                self.assertTrue(parameter.requires_grad)
        self.assertFalse(model.anew_block_encoder.embedding.block_embedding.weight.requires_grad)

    def test_corrected_source_frozen_exposes_only_new_adapter_tensors(self):
        model = _model(fusion_mode="anew_block_pvb_posterior")
        loaded = {
            name
            for name, _ in model.named_parameters()
            if name != "block_gate" and not name.startswith("block_projection.")
        }
        configure_fusion_parameters(model, "source_frozen", source_keys=loaded)
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            trainable,
            {
                "block_gate",
                "block_projection.0.weight",
                "block_projection.0.bias",
                "block_projection.1.weight",
                "block_projection.1.bias",
            },
        )
        groups = fusion_parameter_groups(model, 1e-4, 1e-5, 2e-4)
        self.assertEqual({group["name"] for group in groups}, {"projector_gate"})

    def test_gradient_diagnostics_are_finite(self):
        model = _model(n_layers=1).train()
        with torch.no_grad():
            model.block_gate.fill_(0.2)
        configure_fusion_parameters(model, "A")
        batch = build_synthetic_batch(16, 2, torch.device("cpu"), seed=21)
        loss, _ = model._train(batch)
        loss.backward()
        diagnostics = fusion_gradient_norms(model)
        self.assertIn("pvb", diagnostics)
        self.assertIn("projector_gate", diagnostics)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values()))

    def test_shared_source_frozen_has_exact_pvb_union_complement(self):
        model = _model(fusion_mode="pvb_shared_hblock")
        loaded = {
            name for name, _ in model.named_parameters()
            if name.startswith(("encoder.", "W_vec_mu.", "W_vec_log_var.",
                                "decoder.", "vel_ffn.", "drf_ffn."))
        }
        configure_fusion_parameters(model, "source_frozen", source_keys=loaded)
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        expected = {
            name for name, _ in model.named_parameters()
            if name.startswith("shared_hblock_adapter.") or name == "shared_hblock_gate"
        }
        self.assertEqual(trainable, expected)
        groups = fusion_parameter_groups(model, 1e-4, 1e-5, 1e-3)
        self.assertEqual({group["name"] for group in groups}, {"projector_gate"})
        self.assertEqual(
            {id(parameter) for group in groups for parameter in group["params"]},
            {id(parameter) for name, parameter in model.named_parameters() if name in expected},
        )

    def test_dynamic_wrapper_accepts_attention_aware_budget(self):
        class Dataset:
            collate_fn = staticmethod(lambda batch: batch)

            def __init__(self):
                self.lengths = [2, 3, 4]

            def __len__(self):
                return len(self.lengths)

            def get_index_dict(self):
                return {}

            def get_len(self, index):
                return self.lengths[index]

        wrapper = DynamicBatchWrapper(Dataset(), complexity="n*n", ubound_per_batch=13)
        for indexes in wrapper.batch_indexes:
            self.assertLessEqual(sum(Dataset().lengths[i] ** 2 for i in indexes), 13)


if __name__ == "__main__":
    unittest.main()
