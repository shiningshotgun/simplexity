"""Theoretical optimal cross-entropy loss computation for finite context windows.

Gate 5 derivation:
With context window size 10 and no BOS token, the transformer receives a sequence
x_0, x_1, ..., x_9 and is trained to predict next-tokens autoregressively at
9 positions (i = 0, ..., 8) where position i predicts x_{i+1} given prefix x_0..x_i.

The optimal cross-entropy loss under the true data-generating distribution is the
expected conditional entropy of X_{i+1} given prefix X_{0..i}, averaged over the 9
trained positions:
    H* = (1 / 9) sum_{i=0}^{8} H[X_{i+1} | X_0, ..., X_i]

where:
    H[X_{i+1} | x_0..x_i] = - sum_{x in V} P(x | x_0..x_i) ln P(x | x_0..x_i)
    P(x | x_0..x_i) = b(x_0..x_i) T^(x) 1
and the expectation is weighted by the true prefix probability under the HMM:
    P(x_0..x_i) = pi_0 T^(x_0) ... T^(x_i) 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from simplexity.processes import HMMProcess


@dataclass(frozen=True)
class OptimalLossResult:
    process_name: str
    num_positions: int
    per_position_entropy: List[float]  # in nats (natural log)
    average_optimal_loss: float  # in nats
    per_position_bits: List[float]  # in bits (log2)
    average_optimal_loss_bits: float  # in bits


def compute_optimal_finite_context_loss(
    process: HMMProcess,
    num_trained_positions: int = 9,
) -> OptimalLossResult:
    """Compute the exact theoretical optimal finite-context cross-entropy loss.

    Args:
        process: HMMProcess instance with transition matrices and stationary distribution.
        num_trained_positions: Number of trained next-token prediction positions (default 9
            for context window 10 with no BOS).

    Returns:
        OptimalLossResult containing per-position and average cross-entropy in nats and bits.
    """
    num_tokens = len(process.vocab)
    pi_0 = process.stationary_dist

    # Maintain frontier of (prob, belief) for prefixes of length i+1
    # At position i=0, prefix has length 1: x_0
    current_frontier: List[Tuple[float, np.ndarray]] = []
    for x in range(num_tokens):
        t_mat = process.transition_matrices[x]
        unnorm = pi_0 @ t_mat
        p = float(unnorm.sum())
        if p > 1e-15:
            b = unnorm / p
            current_frontier.append((p, b))

    per_position_nats: List[float] = []

    for pos in range(num_trained_positions):
        # Compute H[X_{pos+1} | x_0..x_pos] averaged over frontier
        h_pos = 0.0
        next_frontier: List[Tuple[float, np.ndarray]] = []

        for p_prefix, b_prefix in current_frontier:
            # Conditional distribution over next token
            next_probs = np.array(
                [
                    float((b_prefix @ process.transition_matrices[x]).sum())
                    for x in range(num_tokens)
                ],
                dtype=np.float64,
            )
            # Filter non-zero for entropy computation
            pos_mask = next_probs > 1e-15
            ent = -np.sum(next_probs[pos_mask] * np.log(next_probs[pos_mask]))
            h_pos += p_prefix * float(ent)

            # Advance prefix by 1 token for subsequent positions
            if pos < num_trained_positions - 1:
                for x in range(num_tokens):
                    unnorm = b_prefix @ process.transition_matrices[x]
                    p_cond = float(unnorm.sum())
                    if p_cond > 1e-15:
                        b_next = unnorm / p_cond
                        next_frontier.append((p_prefix * p_cond, b_next))

        per_position_nats.append(h_pos)
        current_frontier = next_frontier

    avg_nats = float(np.mean(per_position_nats))
    per_pos_bits = [h / np.log(2.0) for h in per_position_nats]
    avg_bits = avg_nats / np.log(2.0)

    return OptimalLossResult(
        process_name=process.name,
        num_positions=num_trained_positions,
        per_position_entropy=per_position_nats,
        average_optimal_loss=avg_nats,
        per_position_bits=per_pos_bits,
        average_optimal_loss_bits=avg_bits,
    )
