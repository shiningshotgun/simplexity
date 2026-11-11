"""Unit tests for theoretical optimal loss computation (Gate 5)."""

import unittest
import numpy as np

from simplexity.entropy import compute_optimal_finite_context_loss
from simplexity.processes import get_mess3_process, get_rrxor_process


class TestEntropy(unittest.TestCase):
    def test_gate5_mess3_optimal_loss(self):
        proc = get_mess3_process()
        res = compute_optimal_finite_context_loss(proc, num_trained_positions=9)
        self.assertEqual(len(res.per_position_entropy), 9)
        # Monotonically non-increasing conditional entropy with context
        for i in range(len(res.per_position_entropy) - 1):
            self.assertGreaterEqual(
                res.per_position_entropy[i], res.per_position_entropy[i + 1] - 1e-6
            )
        # Average is approximately 0.8035 nats
        self.assertAlmostEqual(res.average_optimal_loss, 0.803518, places=4)

    def test_gate5_rrxor_optimal_loss(self):
        proc = get_rrxor_process()
        res = compute_optimal_finite_context_loss(proc, num_trained_positions=9)
        self.assertEqual(len(res.per_position_entropy), 9)

        # Position 0 (predicts token 1 given token 0) must be exactly ln(2)
        expected_pos0 = np.log(2.0)
        self.assertAlmostEqual(res.per_position_entropy[0], expected_pos0, places=10)

        # Average is approximately 0.5865 nats
        self.assertAlmostEqual(res.average_optimal_loss, 0.586501, places=4)


if __name__ == "__main__":
    unittest.main()
