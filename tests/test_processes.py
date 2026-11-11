"""Unit tests for Mess3 and RRXOR data generating processes."""

import unittest
import numpy as np

from simplexity.processes import (
    get_mess3_matrices,
    get_mess3_process,
    get_rrxor_matrices,
    get_rrxor_matrices_rational,
    get_rrxor_process,
)


class TestProcesses(unittest.TestCase):
    def test_mess3_matrices_and_row_sums(self):
        proc = get_mess3_process()
        self.assertEqual(proc.num_states, 3)
        self.assertEqual(len(proc.vocab), 3)

        # Gate 2: sum over token, next-state == 1
        t_full = proc.full_transition_matrix
        row_sums = t_full.sum(axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], atol=1e-12)

    def test_mess3_stationary_distribution(self):
        # Gate 1: pi T = pi
        proc = get_mess3_process()
        pi = proc.stationary_dist
        np.testing.assert_allclose(pi, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-12)
        pi_next = pi @ proc.full_transition_matrix
        np.testing.assert_allclose(pi_next, pi, atol=1e-12)

    def test_rrxor_matrices_and_row_sums(self):
        proc = get_rrxor_process()
        self.assertEqual(proc.num_states, 5)
        self.assertEqual(len(proc.vocab), 2)

        # Gate 2: sum over token, next-state == 1
        t_full = proc.full_transition_matrix
        row_sums = t_full.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(5), atol=1e-12)

    def test_rrxor_stationary_distribution(self):
        # Gate 1: pi T = pi, with derived expectation [1/3, 1/6, 1/6, 1/6, 1/6]
        proc = get_rrxor_process()
        expected = np.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        np.testing.assert_allclose(proc.stationary_dist, expected, atol=1e-12)

        pi_next = proc.stationary_dist @ proc.full_transition_matrix
        np.testing.assert_allclose(pi_next, proc.stationary_dist, atol=1e-12)

    def test_sampling_shapes_and_values(self):
        proc = get_mess3_process()
        rng = np.random.default_rng(42)
        tokens, states = proc.sample_sequences(batch_size=16, seq_len=10, rng=rng)
        self.assertEqual(tokens.shape, (16, 10))
        self.assertEqual(states.shape, (16, 11))
        self.assertTrue(np.all(tokens >= 0) and np.all(tokens < 3))
        self.assertTrue(np.all(states >= 0) and np.all(states < 3))


if __name__ == "__main__":
    unittest.main()
