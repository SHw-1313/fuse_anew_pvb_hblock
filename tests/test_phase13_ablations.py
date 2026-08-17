from __future__ import annotations

import unittest

import torch

from module.shared_hblock import SharedHBlockAdapter
from utils.phase13_ablation import (
    get_phase13_variant,
    phase13_adapter_variant_names,
    phase13_variant_names,
)
from scripts.profile_training_paths import build_model


class TestPhase13Ablations(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.h_atom = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        self.atom_block_id = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        self.block_lengths = torch.tensor([2, 2, 2], dtype=torch.long)
        self.block_batch = torch.tensor([0, 0, 1], dtype=torch.long)

    def _adapter(self, variant):
        adapter = SharedHBlockAdapter(
            hidden_dim=4, rank=2, variant=variant, shuffle_seed=20260810
        )
        return adapter

    def test_all_controls_keep_identical_trainable_parameter_count(self):
        counts = []
        for variant in ("real", "shuffled", "constant", "atom_no_pool"):
            adapter = self._adapter(variant)
            counts.append(sum(parameter.numel() for parameter in adapter.parameters()))
        self.assertEqual(counts, [counts[0]] * len(counts))

    def test_shuffle_is_deterministic_and_sample_local(self):
        adapter = self._adapter("shuffled")
        first = adapter(
            self.h_atom,
            self.atom_block_id,
            self.block_lengths,
            self.block_batch,
        )
        second = adapter(
            self.h_atom,
            self.atom_block_id,
            self.block_lengths,
            self.block_batch,
        )
        torch.testing.assert_close(
            first["condition_source"], second["condition_source"], rtol=0.0, atol=0.0
        )
        source = first["condition_source"]
        real = adapter(
            self.h_atom,
            self.atom_block_id,
            self.block_lengths,
            self.block_batch,
            variant="real",
        )["condition_source"]
        for sample_id in (0, 1):
            block_indices = torch.where(self.block_batch == sample_id)[0]
            for row in source[block_indices]:
                self.assertTrue(
                    any(torch.equal(row, candidate) for candidate in real[block_indices])
                )

    def test_constant_has_no_record_specific_block_content(self):
        adapter = self._adapter("constant")
        output = adapter(
            self.h_atom,
            self.atom_block_id,
            self.block_lengths,
            self.block_batch,
        )
        torch.testing.assert_close(
            output["condition_source"][0],
            output["condition_source"][1],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            output["condition_source"][1],
            output["condition_source"][2],
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(output["variant"], "constant")

    def test_atom_no_pool_conditions_each_atom_directly(self):
        adapter = self._adapter("atom_no_pool")
        output = adapter(
            self.h_atom,
            self.atom_block_id,
            self.block_lengths,
            self.block_batch,
        )
        torch.testing.assert_close(
            output["condition_source"], self.h_atom, rtol=0.0, atol=0.0
        )
        self.assertEqual(tuple(output["condition_block"].shape), (0, 4))

    def test_model_variant_is_explicit_and_gate_starts_zero(self):
        for variant in ("real", "shuffled", "constant", "atom_no_pool"):
            model = build_model(
                "pvb_shared_hblock",
                shared_hblock_variant=variant,
                shared_hblock_seed=20260810,
            )
            self.assertEqual(model.shared_hblock_variant, variant)
            self.assertEqual(float(model.shared_hblock_gate), 0.0)
            self.assertEqual(model.anew_block_encoder, None)


    def test_registry_names_and_adapter_values_are_explicit(self):
        self.assertEqual(
            phase13_variant_names(),
            (
                "pvb_shared_real",
                "pvb_shared_shuffled",
                "pvb_shared_constant",
                "pvb_atom_no_pool",
            ),
        )
        self.assertEqual(
            phase13_adapter_variant_names(),
            ("real", "shuffled", "constant", "atom_no_pool"),
        )
        self.assertEqual(get_phase13_variant("pvb_shared_real").adapter_variant, "real")
        self.assertEqual(get_phase13_variant("atom_no_pool").name, "pvb_atom_no_pool")
        with self.assertRaises(ValueError):
            get_phase13_variant("anew_block")


if __name__ == "__main__":
    unittest.main()

