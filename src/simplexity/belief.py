"""Belief state computation and reachable belief geometry.

Follows Section 2.2, Eq. (1)-(2) of Shai et al. (NeurIPS 2024):
Belief update: eta' = eta T^(x) / (eta T^(x) 1)
Initial belief: stationary distribution pi of T = sum_x T^(x).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from simplexity.processes import HMMProcess, get_rrxor_matrices_rational


def update_belief_float(
    belief: np.ndarray,
    transition_matrix: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute updated belief state in float arithmetic.

    Args:
        belief: (num_states,) probability vector.
        transition_matrix: T^(x), (num_states, num_states).

    Returns:
        Updated belief vector (num_states,), or None if emission has probability 0.
    """
    unnorm = belief @ transition_matrix
    norm = unnorm.sum()
    if norm <= 1e-15:
        return None
    return unnorm / norm


def compute_sequence_beliefs_float(
    process: HMMProcess,
    sequence: Sequence[int],
) -> List[Optional[np.ndarray]]:
    """Compute the sequence of belief states after each token in sequence.

    Returns:
        List of length len(sequence), where item p is the belief after observing
        sequence[0..p]. If a prefix becomes impossible (prob 0), subsequent
        entries are None.
    """
    beliefs: List[Optional[np.ndarray]] = []
    curr = process.stationary_dist.copy()
    for token in sequence:
        if curr is None:
            beliefs.append(None)
            continue
        curr = update_belief_float(curr, process.transition_matrices[token])
        beliefs.append(curr.copy() if curr is not None else None)
    return beliefs


def update_belief_rational(
    belief: Tuple[Fraction, ...],
    transition_matrix: List[List[Fraction]],
) -> Optional[Tuple[Fraction, ...]]:
    """Compute updated belief state in exact rational arithmetic.

    Args:
        belief: tuple of Fractions summing to 1.
        transition_matrix: 2D list of Fractions.

    Returns:
        Updated belief tuple of Fractions, or None if sum is 0.
    """
    num_states = len(belief)
    unnorm = [Fraction(0)] * num_states
    for j in range(num_states):
        col_sum = Fraction(0)
        for i in range(num_states):
            col_sum += belief[i] * transition_matrix[i][j]
        unnorm[j] = col_sum

    total = sum(unnorm)
    if total == Fraction(0):
        return None
    return tuple(x / total for x in unnorm)


def enumerate_rrxor_rational_beliefs(
    max_length: int = 10,
) -> Tuple[Set[Tuple[Fraction, ...]], List[Dict[str, int]]]:
    """Enumerate reachable belief states for RRXOR using exact rational arithmetic.

    Explores all sequence prefixes up to max_length starting from the stationary belief
    pi = (1/3, 1/6, 1/6, 1/6, 1/6).

    Returns:
        all_beliefs: set of exact distinct belief state tuples.
        history: list of per-length metrics (length, frontier_size, cumulative_unique).
    """
    t_mats = get_rrxor_matrices_rational()
    pi_0 = (
        Fraction(1, 3),
        Fraction(1, 6),
        Fraction(1, 6),
        Fraction(1, 6),
        Fraction(1, 6),
    )

    all_beliefs: Set[Tuple[Fraction, ...]] = {pi_0}
    current_frontier: Set[Tuple[Fraction, ...]] = {pi_0}
    history: List[Dict[str, int]] = [
        {"length": 0, "frontier_size": 1, "cumulative_unique": 1}
    ]

    for length in range(1, max_length + 1):
        next_frontier: Set[Tuple[Fraction, ...]] = set()
        for b in current_frontier:
            for token in (0, 1):
                b_next = update_belief_rational(b, t_mats[token])
                if b_next is not None:
                    next_frontier.add(b_next)
                    all_beliefs.add(b_next)
        current_frontier = next_frontier
        history.append(
            {
                "length": length,
                "frontier_size": len(current_frontier),
                "cumulative_unique": len(all_beliefs),
            }
        )

    return all_beliefs, history


def enumerate_all_prefix_beliefs(
    process: HMMProcess,
    max_length: int = 10,
) -> Tuple[np.ndarray, List[int]]:
    """Enumerate belief states for all prefixes of length 0..max_length.

    Used for ground-truth visualization (Gate 4) and dataset generation.

    Returns:
        beliefs: (N, num_states) array of valid belief vectors.
        prefix_lengths: list of length N with prefix length for each belief.
    """
    num_tokens = len(process.vocab)
    initial_belief = process.stationary_dist.copy()

    beliefs_list = [initial_belief]
    lengths_list = [0]

    current_level = [initial_belief]
    for length in range(1, max_length + 1):
        next_level = []
        for b in current_level:
            for token in range(num_tokens):
                b_next = update_belief_float(b, process.transition_matrices[token])
                if b_next is not None:
                    next_level.append(b_next)
                    beliefs_list.append(b_next)
                    lengths_list.append(length)
        current_level = next_level

    return np.array(beliefs_list, dtype=np.float64), lengths_list
