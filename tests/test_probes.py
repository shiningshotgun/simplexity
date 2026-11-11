"""Unit tests for probe dataset, linear regression, controls, and distance analysis."""

import unittest
import numpy as np

from simplexity.processes import get_mess3_process, get_rrxor_process
from simplexity.probes import (
    build_all_sequences,
    build_probe_dataset,
    compute_next_token_distributions,
    compute_pairwise_distance_correlation,
    fit_linear_probe,
    run_cross_validation_control,
    run_shuffle_control,
)


class TestProbes(unittest.TestCase):
    def test_rrxor_probe_dataset_counts(self):
        rrxor = get_rrxor_process()
        dataset = build_probe_dataset(rrxor, seq_len=10)

        # 1024 seqs * 10 positions = 10,240 candidate samples
        self.assertEqual(dataset.total_candidate_samples, 10_240)
        # 8,108 valid samples retained
        self.assertEqual(dataset.retained_samples, 8_108)
        # 2,132 impossible zero-probability samples excluded
        self.assertEqual(dataset.excluded_samples, 2_132)
        self.assertEqual(len(dataset.beliefs), 8_108)
        self.assertEqual(dataset.beliefs.shape[1], 5)

    def test_linear_probe_exact_fit(self):
        rng = np.random.default_rng(0)
        N, d_in, d_out = 100, 10, 3
        X = rng.normal(size=(N, d_in))
        W_true = rng.normal(size=(d_in, d_out))
        c_true = rng.normal(size=(d_out,))
        Y = X @ W_true + c_true

        probe = fit_linear_probe(X, Y)
        self.assertAlmostEqual(probe.train_mse, 0.0, places=10)
        self.assertAlmostEqual(probe.train_r2, 1.0, places=10)
        np.testing.assert_allclose(probe.W, W_true, atol=1e-8)
        np.testing.assert_allclose(probe.c, c_true, atol=1e-8)

    def test_shuffle_and_cv_controls(self):
        rng = np.random.default_rng(0)
        N, d_in, d_out = 200, 8, 3
        X = rng.normal(size=(N, d_in))
        Y = rng.normal(size=(N, d_out))

        shuf = run_shuffle_control(X, Y, n_iterations=10, rng=rng)
        self.assertIn("mean_mse", shuf)
        self.assertGreater(shuf["mean_mse"], 0.0)

        cv = run_cross_validation_control(X, Y, train_ratio=0.2, n_iterations=10, rng=rng)
        self.assertIn("mean_test_mse", cv)
        self.assertGreater(cv["mean_test_mse"], 0.0)

    def test_pairwise_distance_correlation(self):
        rng = np.random.default_rng(0)
        M, k = 36, 5
        beliefs = rng.dirichlet(np.ones(k), size=M)
        reprs = beliefs + 0.01 * rng.normal(size=(M, k))
        dists = rng.dirichlet(np.ones(2), size=M)

        res = compute_pairwise_distance_correlation(beliefs, reprs, dists)
        self.assertIn("r2_repr_vs_belief", res)
        self.assertIn("r2_next_token_vs_belief", res)
        self.assertGreater(res["r2_repr_vs_belief"], 0.90)


if __name__ == "__main__":
    unittest.main()
