"""Automated integration test for Pre-training Gates 1 through 5."""

import unittest
from pathlib import Path
import numpy as np

from simplexity.belief import enumerate_rrxor_rational_beliefs
from simplexity.entropy import compute_optimal_finite_context_loss
from simplexity.processes import get_mess3_process, get_rrxor_process
from simplexity.visualization import plot_mess3_ground_truth


class TestGates(unittest.TestCase):
    def test_gate1_stationary_distribution(self):
        """Gate 1: Stationary distribution is left eigenvector with eigenvalue 1."""
        mess3 = get_mess3_process()
        rrxor = get_rrxor_process()

        # Mess3
        m3_pi_next = mess3.stationary_dist @ mess3.full_transition_matrix
        np.testing.assert_allclose(m3_pi_next, mess3.stationary_dist, atol=1e-12)

        # RRXOR
        expected_rr_pi = np.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        np.testing.assert_allclose(rrxor.stationary_dist, expected_rr_pi, atol=1e-12)
        rr_pi_next = rrxor.stationary_dist @ rrxor.full_transition_matrix
        np.testing.assert_allclose(rr_pi_next, rrxor.stationary_dist, atol=1e-12)

    def test_gate2_stochastic_matrices(self):
        """Gate 2: Row sums sum to 1."""
        mess3 = get_mess3_process()
        rrxor = get_rrxor_process()

        np.testing.assert_allclose(
            mess3.full_transition_matrix.sum(axis=1), np.ones(3), atol=1e-12
        )
        np.testing.assert_allclose(
            rrxor.full_transition_matrix.sum(axis=1), np.ones(5), atol=1e-12
        )

    def test_gate3_rrxor_stabilization(self):
        """Gate 3: Exactly 36 distinct belief states, stabilizing before length 10."""
        beliefs, history = enumerate_rrxor_rational_beliefs(max_length=10)
        self.assertEqual(len(beliefs), 36)
        # Stabilized by length 6
        self.assertEqual(history[6]["cumulative_unique"], 36)

    def test_gate4_mess3_ground_truth_geometry(self):
        """Gate 4: Generate GT_mess3.png in 2-simplex."""
        out = plot_mess3_ground_truth("figures/GT_mess3.png", max_length=10)
        self.assertTrue(out.exists())
        self.assertGreater(out.stat().st_size, 50_000)

    def test_gate5_optimal_loss_constants(self):
        """Gate 5: Theoretical optimal finite-context loss constants."""
        mess3 = get_mess3_process()
        rrxor = get_rrxor_process()

        m3_loss = compute_optimal_finite_context_loss(mess3, num_trained_positions=9)
        rr_loss = compute_optimal_finite_context_loss(rrxor, num_trained_positions=9)

        # RRXOR pos 0 must be exactly ln(2)
        self.assertAlmostEqual(rr_loss.per_position_entropy[0], np.log(2.0), places=10)

        # Expected 9-position averages
        self.assertAlmostEqual(m3_loss.average_optimal_loss, 0.803518, places=4)
        self.assertAlmostEqual(rr_loss.average_optimal_loss, 0.586501, places=4)


if __name__ == "__main__":
    unittest.main()
