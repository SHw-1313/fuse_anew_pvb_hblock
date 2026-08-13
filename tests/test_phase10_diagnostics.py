from __future__ import annotations

import unittest

import torch

from scripts.profile_train_step import build_synthetic_batch
from tests.test_fusion import _model
from utils.phase10_diagnostics import collect_phase10_diagnostics, posterior_kl, summarize_tensor


class TestPhase10Diagnostics(unittest.TestCase):
    def test_distribution_summary_and_posterior_kl(self):
        value = torch.tensor([[-1.0], [-2.0], [-3.0]])
        summary = summarize_tensor(value)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["min"], -3.0)
        self.assertEqual(summary["max"], -1.0)
        kl = posterior_kl(
            value,
            torch.ones(3, dtype=torch.bool),
            coord_prior_var=0.5,
        )
        expected = -0.5 * torch.sum(
            1.0 + value.expand(-1, 3) - torch.log(torch.tensor(0.5))
            - torch.exp(value.expand(-1, 3)) / 0.5
        ) / 3
        torch.testing.assert_close(kl, expected)

    def test_corrected_record_separates_pvb_and_anew_variances(self):
        model = _model("anew_block_pvb_posterior").eval()
        batch = build_synthetic_batch(16, 2, torch.device("cpu"), seed=31)
        with torch.no_grad():
            loss, parts = model._train(batch, mode="pretrain")
            record = collect_phase10_diagnostics(
                model,
                batch,
                loss=loss,
                parts=parts,
                epoch=2,
                global_step=7,
            )
        self.assertEqual(record["epoch"], 2)
        self.assertIn("mean", record["pvb_log_var"])
        self.assertIn("p99", record["pvb_log_var"])
        self.assertIn("p50", record["anew_log_var_block"])
        self.assertIn("p90", record["anew_log_var_atom"])
        self.assertAlmostEqual(record["pvb_kl"], float(parts[0]), places=5)
        self.assertTrue(torch.isfinite(torch.tensor(record["anew_kl_diagnostic"])))
        self.assertFalse(record["anew_log_var_used_in_loss"])
        self.assertEqual(record["block_gate"], 0.0)
        self.assertEqual(record["projected_condition_norm"], 0.0)
        self.assertAlmostEqual(
            record["rec_total"],
            float(parts[1]) + float(parts[2]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
