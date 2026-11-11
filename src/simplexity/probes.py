"""Probe dataset construction, linear regression probes, and statistical controls.

Implements Section 2.3, Appendix A.5, and Appendix A.7 of Shai et al. (NeurIPS 2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import itertools
import numpy as np

from simplexity.belief import compute_sequence_beliefs_float
from simplexity.processes import HMMProcess


@dataclass
class ProbeDataset:
    process_name: str
    sequences: np.ndarray  # (num_seqs, 10) int
    positions: np.ndarray  # (N,) int
    beliefs: np.ndarray  # (N, num_states) float64
    sequence_indices: np.ndarray  # (N,) int
    total_candidate_samples: int
    retained_samples: int
    excluded_samples: int


def build_all_sequences(vocab_size: int, seq_len: int = 10) -> np.ndarray:
    """Generate all vocab_size^seq_len possible sequences of length seq_len."""
    all_seqs = list(itertools.product(range(vocab_size), repeat=seq_len))
    return np.array(all_seqs, dtype=np.int64)


def build_probe_dataset(process: HMMProcess, seq_len: int = 10) -> ProbeDataset:
    """Construct full probe dataset across all sequences and positions.

    - Mess3: 3^10 = 59,049 sequences x 10 positions = 590,490 samples.
    - RRXOR: 2^10 = 1,024 sequences x 10 positions = 10,240 candidate samples.
      Zero-probability prefixes (unnormalized belief == 0) are excluded per INTERPRETATION.

    Returns:
        ProbeDataset containing valid (sequence, position) samples and belief labels.
    """
    vocab_size = len(process.vocab)
    all_seqs = build_all_sequences(vocab_size, seq_len=seq_len)
    total_candidates = len(all_seqs) * seq_len

    seq_idx_list: List[int] = []
    pos_list: List[int] = []
    belief_list: List[np.ndarray] = []

    for s_idx, seq in enumerate(all_seqs):
        seq_beliefs = compute_sequence_beliefs_float(process, seq)
        for pos in range(seq_len):
            b = seq_beliefs[pos]
            if b is not None:
                seq_idx_list.append(s_idx)
                pos_list.append(pos)
                belief_list.append(b)

    retained = len(belief_list)
    excluded = total_candidates - retained

    return ProbeDataset(
        process_name=process.name,
        sequences=all_seqs,
        positions=np.array(pos_list, dtype=np.int64),
        beliefs=np.array(belief_list, dtype=np.float64),
        sequence_indices=np.array(seq_idx_list, dtype=np.int64),
        total_candidate_samples=total_candidates,
        retained_samples=retained,
        excluded_samples=excluded,
    )


@dataclass
class LinearProbe:
    W: np.ndarray  # (d_in, d_out)
    c: np.ndarray  # (d_out,)
    train_mse: float
    train_r2: float

    def predict(self, activations: np.ndarray) -> np.ndarray:
        return activations @ self.W + self.c


def fit_linear_probe(
    activations: np.ndarray,  # (N, d)
    targets: np.ndarray,  # (N, k)
) -> LinearProbe:
    """Fit affine linear regression: min_{W, c} sum_i || b_i - (W a_i + c) ||^2.

    Appendix A.5: standard linear regression with affine model b = W a + c.
    """
    N, d = activations.shape
    # Augment activations with constant column 1 for bias c
    A_aug = np.hstack([activations, np.ones((N, 1), dtype=activations.dtype)])

    # Solve least squares
    sol, residuals, rank, s = np.linalg.lstsq(A_aug, targets, rcond=None)
    W = sol[:-1, :]
    c = sol[-1, :]

    preds = activations @ W + c
    mse = float(np.mean(np.sum((targets - preds) ** 2, axis=1)))

    # Overall R^2
    var_tot = float(np.mean(np.sum((targets - np.mean(targets, axis=0)) ** 2, axis=1)))
    r2 = 1.0 - (mse / var_tot) if var_tot > 1e-12 else 0.0

    return LinearProbe(W=W, c=c, train_mse=mse, train_r2=r2)


def run_shuffle_control(
    activations: np.ndarray,
    targets: np.ndarray,
    n_iterations: int = 1000,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    """Shuffle control (Fig 6C & App A.7): permute belief labels among distinct belief states 1000 times.

    Operates on the full dataset without subsampling.
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    N, d = activations.shape
    unique_beliefs, inv_idx = np.unique(np.round(targets, 8), axis=0, return_inverse=True)
    num_distinct = len(unique_beliefs)

    A_aug = np.hstack([activations, np.ones((N, 1), dtype=activations.dtype)])
    AtA = A_aug.T @ A_aug
    AtA_inv = np.linalg.pinv(AtA)
    M = AtA_inv @ A_aug.T  # (d+1, N)

    mses: List[float] = []
    for _ in range(n_iterations):
        perm = rng.permutation(num_distinct)
        shuffled_targets = unique_beliefs[perm][inv_idx]
        W = M @ shuffled_targets
        preds = A_aug @ W
        mse = float(np.mean(np.sum((shuffled_targets - preds) ** 2, axis=1)))
        mses.append(mse)

    mses_arr = np.array(mses)
    return {
        "mean_mse": float(np.mean(mses_arr)),
        "std_mse": float(np.std(mses_arr)),
        "min_mse": float(np.min(mses_arr)),
        "max_mse": float(np.max(mses_arr)),
        "num_distinct_states": num_distinct,
        "n_samples": N,
    }


def run_cross_validation_control(
    activations: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.20,
    n_iterations: int = 1000,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    """Cross-validation control (Sec 3.1 & App A.7): 20% train / 80% test split 1000 times.

    Operates on the full dataset without subsampling.
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    N, d = activations.shape
    A_aug = np.hstack([activations, np.ones((N, 1), dtype=activations.dtype)])
    n_train = max(1, int(round(N * train_ratio)))
    test_mses: List[float] = []

    for _ in range(n_iterations):
        perm = rng.permutation(N)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        A_train = A_aug[train_idx]
        T_train = targets[train_idx]
        A_test = A_aug[test_idx]
        T_test = targets[test_idx]

        AtA_tr = A_train.T @ A_train
        AtT_tr = A_train.T @ T_train
        try:
            W = np.linalg.solve(AtA_tr, AtT_tr)
        except np.linalg.LinAlgError:
            W, _, _, _ = np.linalg.lstsq(A_train, T_train, rcond=1e-3)

        test_preds = A_test @ W
        test_mse = float(np.mean(np.sum((T_test - test_preds) ** 2, axis=1)))
        test_mses.append(test_mse)

    mses_arr = np.array(test_mses)
    return {
        "mean_test_mse": float(np.mean(mses_arr)),
        "std_test_mse": float(np.std(mses_arr)),
        "min_test_mse": float(np.min(mses_arr)),
        "max_test_mse": float(np.max(mses_arr)),
        "n_samples": N,
        "n_train": n_train,
        "n_test": N - n_train,
    }


def compute_pairwise_distance_correlation(
    true_beliefs: np.ndarray,  # (M, k)
    representations: np.ndarray,  # (M, d)
    next_token_dists: np.ndarray,  # (M, |V|)
) -> Dict[str, float]:
    """Compute R^2 between pairwise Euclidean distances (paper Fig 7D).

    Args:
        true_beliefs: distinct ground-truth belief states.
        representations: corresponding representations (or projected representations).
        next_token_dists: corresponding conditional next-token distributions.

    Returns:
        R^2 for (dist_repr vs dist_belief) and (dist_next_token vs dist_belief).
    """
    from scipy.spatial.distance import pdist

    d_true = pdist(true_beliefs, metric="euclidean")
    d_repr = pdist(representations, metric="euclidean")
    d_next = pdist(next_token_dists, metric="euclidean")

    # Fit linear regression between pairwise distances
    # R^2 = pearson_r^2 for simple linear regression
    r_repr = np.corrcoef(d_true, d_repr)[0, 1]
    r_next = np.corrcoef(d_true, d_next)[0, 1]

    return {
        "r2_repr_vs_belief": float(r_repr**2),
        "r2_next_token_vs_belief": float(r_next**2),
    }


def extract_dataset_activations(
    model: Any,
    dataset: ProbeDataset,
    batch_size: int = 1024,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Extract layer and final residual stream activations for all dataset samples.

    Returns:
        Dict mapping hookpoint names to numpy arrays of shape (retained_samples, d):
            'blocks.0.hook_resid_post': (N, 64)
            'blocks.1.hook_resid_post': (N, 64)
            'blocks.2.hook_resid_post': (N, 64)
            'blocks.3.hook_resid_post': (N, 64)
            'concat': (N, 256)
            'ln_final.hook_normalized': (N, 64)
    """
    import torch

    model.eval()
    num_seqs = len(dataset.sequences)
    seq_idx = dataset.sequence_indices
    pos_idx = dataset.positions
    N = dataset.retained_samples

    # Check whether model is HookedTransformer or SimplexityTransformer
    is_hooked = hasattr(model, "run_with_cache")

    # Arrays to accumulate sample activations
    b0_list, b1_list, b2_list, b3_list, ln_list = [], [], [], [], []

    # Map sequence index to its offset in the dataset
    # We batch across sequences:
    all_b0 = []
    all_b1 = []
    all_b2 = []
    all_b3 = []
    all_ln = []

    with torch.no_grad():
        for start_idx in range(0, num_seqs, batch_size):
            end_idx = min(start_idx + batch_size, num_seqs)
            batch_seqs = torch.from_numpy(dataset.sequences[start_idx:end_idx]).to(device)

            if is_hooked:
                _, cache = model.run_with_cache(
                    batch_seqs,
                    names_filter=lambda n: n in [f"blocks.{i}.hook_resid_post" for i in range(4)]
                    or n == "ln_final.hook_normalized",
                )
                b0 = cache["blocks.0.hook_resid_post"].detach().cpu().numpy()
                b1 = cache["blocks.1.hook_resid_post"].detach().cpu().numpy()
                b2 = cache["blocks.2.hook_resid_post"].detach().cpu().numpy()
                b3 = cache["blocks.3.hook_resid_post"].detach().cpu().numpy()
                ln = cache["ln_final.hook_normalized"].detach().cpu().numpy()
            else:
                _, cache = model(batch_seqs, return_cache=True)
                b0 = cache["blocks.0.hook_resid_post"].detach().cpu().numpy()
                b1 = cache["blocks.1.hook_resid_post"].detach().cpu().numpy()
                b2 = cache["blocks.2.hook_resid_post"].detach().cpu().numpy()
                b3 = cache["blocks.3.hook_resid_post"].detach().cpu().numpy()
                ln = cache["ln_final.hook_normalized"].detach().cpu().numpy()

            all_b0.append(b0)
            all_b1.append(b1)
            all_b2.append(b2)
            all_b3.append(b3)
            all_ln.append(ln)

    all_b0_arr = np.concatenate(all_b0, axis=0)  # (num_seqs, 10, 64)
    all_b1_arr = np.concatenate(all_b1, axis=0)
    all_b2_arr = np.concatenate(all_b2, axis=0)
    all_b3_arr = np.concatenate(all_b3, axis=0)
    all_ln_arr = np.concatenate(all_ln, axis=0)

    # Gather activations at (seq_idx, pos_idx)
    act_b0 = all_b0_arr[seq_idx, pos_idx, :]
    act_b1 = all_b1_arr[seq_idx, pos_idx, :]
    act_b2 = all_b2_arr[seq_idx, pos_idx, :]
    act_b3 = all_b3_arr[seq_idx, pos_idx, :]
    act_ln = all_bln = all_ln_arr[seq_idx, pos_idx, :]
    act_concat = np.concatenate([act_b0, act_b1, act_b2, act_b3], axis=-1)

    return {
        "blocks.0.hook_resid_post": act_b0,
        "blocks.1.hook_resid_post": act_b1,
        "blocks.2.hook_resid_post": act_b2,
        "blocks.3.hook_resid_post": act_b3,
        "concat": act_concat,
        "ln_final.hook_normalized": act_ln,
    }


def compute_next_token_distributions(
    process: HMMProcess,
    belief_states: np.ndarray,  # (M, k)
) -> np.ndarray:
    """Compute next-token emission distribution P(X = x | eta) = eta T^(x) 1 for each belief."""
    M, k = belief_states.shape
    vocab_size = len(process.vocab)
    dists = np.zeros((M, vocab_size), dtype=np.float64)

    for x_tok in range(vocab_size):
        Tx = process.transition_matrices[x_tok]  # (k, k)
        ones = np.ones(k, dtype=np.float64)
        prob_x = belief_states @ (Tx @ ones)  # (M,)
        dists[:, x_tok] = prob_x

    # Row normalize to be strictly sum to 1
    row_sums = dists.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return dists / row_sums
