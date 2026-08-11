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


def _model(n_layers: int = 3) -> dyVAE:
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
        fusion_mode="anew_block",
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
