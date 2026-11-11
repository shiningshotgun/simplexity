"""Unit tests for belief tracking and rational enumeration (Gate 3)."""

from fractions import Fraction
import unittest
import numpy as np

from simplexity.belief import (
    compute_sequence_beliefs_float,
    enumerate_all_prefix_beliefs,
    enumerate_rrxor_rational_beliefs,
    update_belief_float,
    update_belief_rational,
)
from simplexity.processes import get_mess3_process, get_rrxor_process


class TestBelief(unittest.TestCase):
    def test_float_belief_update_normalization(self):
        proc = get_mess3_process()
        b0 = proc.stationary_dist
        b1 = update_belief_float(b0, proc.transition_matrices[0])
        self.assertIsNotNone(b1)
        self.assertAlmostEqual(float(b1.sum()), 1.0, places=12)
        self.assertTrue(np.all(b1 >= 0.0))

    def test_gate3_rrxor_rational_beliefs_count(self):
        """Gate 3: RRXOR reachable beliefs must stabilize at exactly 36 distinct states."""
        beliefs, history = enumerate_rrxor_rational_beliefs(max_length=10)
        self.assertEqual(len(beliefs), 36)

        # Check stabilization at length 6
        h6 = history[6]
        self.assertEqual(h6["cumulative_unique"], 36)
        for h in history[6:]:
            self.assertEqual(h["cumulative_unique"], 36)

    def test_zero_probability_sequence(self):
        """RRXOR impossible sequence returns None."""
        proc = get_rrxor_process()
        # Because initial state is drawn from stationary distribution,
        # any prefix of length <= 4 has nonzero probability under some initial phase.
        # The first impossible sequences appear at length 5, e.g. (0, 0, 1, 0, 0).
        seq_impossible = [0, 0, 1, 0, 0]
        beliefs = compute_sequence_beliefs_float(proc, seq_impossible)
        self.assertIsNotNone(beliefs[0])
        self.assertIsNotNone(beliefs[1])
        self.assertIsNotNone(beliefs[2])
        self.assertIsNotNone(beliefs[3])
        self.assertIsNone(beliefs[4])


if __name__ == "__main__":
    unittest.main()
