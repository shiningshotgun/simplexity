"""Data generating processes: Mess3 and RRXOR Hidden Markov Models.

Faithfully implements the transition matrices from Appendix A.3 of:
Shai, Marzen, Teixeira, Oldenziel & Riechers,
"Transformers Represent Belief State Geometry in their Residual Stream", NeurIPS 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class HMMProcess:
    """Represents a discrete-time, finite-state, emission-labeled Hidden Markov Model."""

    name: str
    num_states: int
    vocab: List[str]
    transition_matrices: Dict[int, np.ndarray]  # token_id -> (num_states, num_states)
    stationary_dist: np.ndarray  # (num_states,)

    @property
    def full_transition_matrix(self) -> np.ndarray:
        """T = sum_x T^(x). Row sums must equal 1."""
        return sum(self.transition_matrices.values())

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample batch_size sequences of length seq_len from the process.

        Initial hidden state is chosen from the stationary distribution.
        Returns:
            tokens: (batch_size, seq_len) integer token IDs.
            states: (batch_size, seq_len + 1) integer state IDs visited.
        """
        if rng is None:
            rng = np.random.default_rng()

        num_tokens = len(self.vocab)
        # Flatten joint (token, next_state) transitions for fast 2D categorical sampling:
        # shape: (num_states, num_tokens * num_states)
        joint_probs = np.zeros(
            (self.num_states, num_tokens * self.num_states), dtype=np.float64
        )
        for token_id in range(num_tokens):
            t_mat = self.transition_matrices[token_id]
            for s in range(self.num_states):
                joint_probs[s, token_id * self.num_states : (token_id + 1) * self.num_states] = (
                    t_mat[s, :]
                )

        # Precompute CDF for inverse transform sampling
        joint_cdf = np.cumsum(joint_probs, axis=1)

        tokens = np.zeros((batch_size, seq_len), dtype=np.int64)
        states = np.zeros((batch_size, seq_len + 1), dtype=np.int64)

        # Sample initial state from stationary distribution
        init_cdf = np.cumsum(self.stationary_dist)
        r_init = rng.random(batch_size)
        curr_states = np.searchsorted(init_cdf, r_init)
        curr_states = np.clip(curr_states, 0, self.num_states - 1)
        states[:, 0] = curr_states

        for t in range(seq_len):
            r = rng.random(batch_size)
            # Sample next (token, state) for each sequence
            cdfs = joint_cdf[curr_states]  # (batch_size, num_tokens * num_states)
            # Find index where cdf >= r
            # Since rows have varied CDFs, use vectorized search
            choices = np.sum(cdfs < r[:, None], axis=1)
            choices = np.clip(choices, 0, num_tokens * self.num_states - 1)

            next_tokens = choices // self.num_states
            next_states = choices % self.num_states

            tokens[:, t] = next_tokens
            states[:, t + 1] = next_states
            curr_states = next_states

        return tokens, states


def get_mess3_matrices() -> Dict[int, np.ndarray]:
    """Mess3 process transition matrices from Appendix A.3, p. 14.

    Hidden states: 3. Tokens: {A: 0, B: 1, C: 2}.
    """
    t_a = np.array(
        [
            [0.765, 0.00375, 0.00375],
            [0.0425, 0.0675, 0.00375],
            [0.0425, 0.00375, 0.0675],
        ],
        dtype=np.float64,
    )
    t_b = np.array(
        [
            [0.0675, 0.0425, 0.00375],
            [0.00375, 0.765, 0.00375],
            [0.00375, 0.0425, 0.0675],
        ],
        dtype=np.float64,
    )
    t_c = np.array(
        [
            [0.0675, 0.00375, 0.0425],
            [0.00375, 0.0675, 0.0425],
            [0.00375, 0.00375, 0.765],
        ],
        dtype=np.float64,
    )
    return {0: t_a, 1: t_b, 2: t_c}


def get_mess3_process() -> HMMProcess:
    """Instantiate the Mess3 HMM with verified matrices and stationary distribution [1/3, 1/3, 1/3]."""
    mats = get_mess3_matrices()
    pi = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    return HMMProcess(
        name="Mess3",
        num_states=3,
        vocab=["A", "B", "C"],
        transition_matrices=mats,
        stationary_dist=pi,
    )


def get_rrxor_matrices() -> Dict[int, np.ndarray]:
    """RRXOR process transition matrices from Appendix A.3, p. 15.

    Hidden states: 5. Tokens: {0: 0, 1: 1}.
    """
    t_0 = np.array(
        [
            [0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    t_1 = np.array(
        [
            [0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return {0: t_0, 1: t_1}


def get_rrxor_matrices_rational() -> Dict[int, List[List[Fraction]]]:
    """Exact rational transition matrices for RRXOR."""
    t_0 = [
        [Fraction(0), Fraction(1, 2), Fraction(0), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1, 2)],
        [Fraction(0), Fraction(0), Fraction(0), Fraction(1, 2), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0)],
        [Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)],
    ]
    t_1 = [
        [Fraction(0), Fraction(0), Fraction(1, 2), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(0), Fraction(1, 2), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1, 2)],
        [Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0)],
    ]
    return {0: t_0, 1: t_1}


def get_rrxor_process() -> HMMProcess:
    """Instantiate the RRXOR HMM with verified matrices and stationary distribution [1/3, 1/6, 1/6, 1/6, 1/6]."""
    mats = get_rrxor_matrices()
    pi = np.array(
        [1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64
    )
    return HMMProcess(
        name="RRXOR",
        num_states=5,
        vocab=["0", "1"],
        transition_matrices=mats,
        stationary_dist=pi,
    )
