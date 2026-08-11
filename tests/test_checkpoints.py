from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from module import dyVAE
from utils.checkpoint import CheckpointCoverageError, load_resume_checkpoint, load_role_checkpoint


def _model(fusion_mode: str = "anew_block") -> dyVAE:
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
            "n_layers": 1,
            "n_head": 4,
            "k_neighbors": 2,
            "sparse_k": None,
        }
        if fusion_mode == "anew_block"
        else None,
    )


class TestCheckpoints(unittest.TestCase):
    def test_pvb_module_checkpoint_loads_only_decoder_and_heads(self):
        source = _model("off")
        target = _model("anew_block")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pvb.ckpt"
            torch.save(source, path)
            report = load_role_checkpoint(target, path, "pvb", min_coverage=1.0)
        self.assertEqual(report.coverage, 1.0)
        self.assertTrue(report.matched_keys)
        self.assertTrue(any(key.startswith("encoder.") for key in report.unexpected_keys))
        self.assertIn("block_projection.0.weight", target.state_dict())

    def test_anew_source_keys_are_translated(self):
        source = _model("anew_block")
        target = _model("anew_block")
        source_state = {}
        for key, value in source.anew_block_encoder.state_dict().items():
            if key.startswith("block_edge_embedding."):
                key = "edge_embedding." + key[len("block_edge_embedding."):]
            source_state[key] = value.clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anew.ckpt"
            torch.save({"state_dict": source_state}, path)
            report = load_role_checkpoint(target, path, "anew", min_coverage=1.0)
        self.assertEqual(report.coverage, 1.0)
        self.assertFalse(report.missing_keys)

    def test_resume_restores_model_and_optimizer_state(self):
        source = _model("anew_block")
        optimizer = torch.optim.Adam(source.parameters(), lr=1e-3)
        optimizer.zero_grad()
        sum(parameter.square().sum() for parameter in source.parameters()).backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.ckpt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": 3,
                    "global_step": 17,
                },
                path,
            )
            target = _model("anew_block")
            target_optimizer = torch.optim.Adam(target.parameters(), lr=1e-3)
            report, metadata = load_resume_checkpoint(
                target, path, optimizer=target_optimizer, min_coverage=1.0
            )
        self.assertEqual(report.coverage, 1.0)
        self.assertEqual(metadata["epoch"], 3)
        self.assertEqual(metadata["global_step"], 17)
        self.assertTrue(target_optimizer.state)

    def test_low_coverage_is_not_silent(self):
        target = _model("anew_block")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.ckpt"
            torch.save({"state_dict": {"decoder.atom_embedding.weight": torch.zeros(1)}}, path)
            with self.assertRaises(CheckpointCoverageError):
                load_role_checkpoint(target, path, "pvb", min_coverage=0.95)


if __name__ == "__main__":
    unittest.main()
