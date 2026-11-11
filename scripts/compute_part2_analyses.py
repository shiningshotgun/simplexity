"""Run Part 2 new analyses for Shai et al. (NeurIPS 2024) reproduction:
1. RRXOR per-layer emergence across all 13 saved 1M snapshots.
2. Matched 20/80 CV shuffle null for RRXOR (concat and Layer 3, 1000 iterations).
3. Activation-centroid pairwise distance diagnostic (RRXOR all layers + Mess3 final layer).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np
from scipy.spatial.distance import pdist
import torch

from simplexity.model import SimplexityModelConfig, build_hooked_transformer
from simplexity.processes import get_mess3_process, get_rrxor_process
from simplexity.probes import (
    build_probe_dataset,
    compute_next_token_distributions,
    compute_pairwise_distance_correlation,
    extract_dataset_activations,
    fit_linear_probe,
)


def load_model(ckpt_path: Path, d_vocab: int):
    cfg = SimplexityModelConfig(d_vocab=d_vocab, seed=0)
    model = build_hooked_transformer(cfg, device="cpu")
    data = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model, data.get("step", 0)


def run_matched_cv_shuffle_null(
    activations: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.20,
    n_iterations: int = 1000,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate belief-state-level shuffle null under the SAME 20/80 CV protocol as real CV."""
    rng = np.random.default_rng(rng_seed)
    N, d = activations.shape
    unique_beliefs, inv_idx = np.unique(np.round(targets, 8), axis=0, return_inverse=True)
    num_distinct = len(unique_beliefs)

    A_aug = np.hstack([activations, np.ones((N, 1), dtype=activations.dtype)])
    n_train = max(1, int(round(N * train_ratio)))
    test_mses: List[float] = []

    for _ in range(n_iterations):
        # 1. Permute targets across distinct belief states
        perm_states = rng.permutation(num_distinct)
        shuffled_targets = unique_beliefs[perm_states][inv_idx]

        # 2. 20/80 train/test split on samples (matching run_cross_validation_control)
        perm_samples = rng.permutation(N)
        train_idx = perm_samples[:n_train]
        test_idx = perm_samples[n_train:]

        A_train = A_aug[train_idx]
        T_train = shuffled_targets[train_idx]
        A_test = A_aug[test_idx]
        T_test = shuffled_targets[test_idx]

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
        "num_distinct_states": num_distinct,
        "n_samples": N,
        "n_train": n_train,
        "n_test": N - n_train,
        "n_iterations": n_iterations,
    }


def compute_activation_centroid_distances(
    activations: np.ndarray,
    targets: np.ndarray,
    sample_indices: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute pairwise Euclidean distance R^2 using per-state activation centroids (Fig 7C dots)."""
    unique_beliefs, inv_idx = np.unique(np.round(targets, 8), axis=0, return_inverse=True)
    num_distinct = len(unique_beliefs)

    if sample_indices is not None:
        eval_state_indices = sample_indices
    else:
        eval_state_indices = np.arange(num_distinct)

    eval_beliefs = unique_beliefs[eval_state_indices]
    centroids = []
    for k in eval_state_indices:
        mask = inv_idx == k
        centroids.append(np.mean(activations[mask], axis=0))
    centroids = np.array(centroids)

    d_true = pdist(eval_beliefs, metric="euclidean")
    d_centroid = pdist(centroids, metric="euclidean")

    r_centroid = np.corrcoef(d_true, d_centroid)[0, 1]
    return {
        "r2_centroid_vs_belief": float(r_centroid**2),
        "pearson_r": float(r_centroid),
        "num_evaluated_states": len(eval_state_indices),
    }


def main():
    print("=" * 70)
    print("STARTING PART 2 ANALYSES (NO RETRAINING)")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. RRXOR Per-Layer Emergence Series Across 13 Checkpoints
    # -------------------------------------------------------------
    print("\n--- 1. Computing RRXOR Per-Layer Emergence (13 Snapshots) ---")
    proc_rr = get_rrxor_process()
    ds_rr = build_probe_dataset(proc_rr, seq_len=10)
    print(f"Loaded RRXOR probe dataset: {ds_rr.retained_samples} samples")

    ckpt_steps = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    rr_emergence_series = []

    # Read existing emergence to preserve normalized val loss
    with open("outputs/1M/emergence_summary.json", "r") as f:
        existing_em = json.load(f)
    val_loss_map = {x["step"]: x["val_loss_normalized"] for x in existing_em["RRXOR"]}

    for step in ckpt_steps:
        ckpt_file = Path(f"checkpoints/1M/RRXOR/step_{step}.pt")
        model, _ = load_model(ckpt_file, d_vocab=2)
        acts = extract_dataset_activations(model, ds_rr, batch_size=1024, device="cpu")

        p0 = fit_linear_probe(acts["blocks.0.hook_resid_post"], ds_rr.beliefs)
        p1 = fit_linear_probe(acts["blocks.1.hook_resid_post"], ds_rr.beliefs)
        p2 = fit_linear_probe(acts["blocks.2.hook_resid_post"], ds_rr.beliefs)
        p3 = fit_linear_probe(acts["blocks.3.hook_resid_post"], ds_rr.beliefs)
        p_cat = fit_linear_probe(acts["concat"], ds_rr.beliefs)
        p_ln = fit_linear_probe(acts["ln_final.hook_normalized"], ds_rr.beliefs)

        entry = {
            "step": step,
            "layer_0_mse": float(p0.train_mse),
            "layer_1_mse": float(p1.train_mse),
            "layer_2_mse": float(p2.train_mse),
            "layer_3_mse": float(p3.train_mse),
            "concat_mse": float(p_cat.train_mse),
            "post_ln_mse": float(p_ln.train_mse),
            "layer_3_r2": float(p3.train_r2),
            "concat_r2": float(p_cat.train_r2),
            "val_loss_normalized": val_loss_map.get(step),
        }
        rr_emergence_series.append(entry)
        print(f"  Step {step:7d} | Layer 3 MSE: {p3.train_mse:.6f} | Concat MSE: {p_cat.train_mse:.6f} | Val Norm: {entry['val_loss_normalized']:.4f}")

    with open("outputs/1M/rrxor_layer_emergence.json", "w") as f:
        json.dump(rr_emergence_series, f, indent=2)
    print("Saved RRXOR per-layer emergence to outputs/1M/rrxor_layer_emergence.json")

    # Update emergence_summary.json to include layer_3_mse in RRXOR entries
    for i, step in enumerate(ckpt_steps):
        existing_em["RRXOR"][i]["layer_3_mse"] = rr_emergence_series[i]["layer_3_mse"]
        existing_em["RRXOR"][i]["layer_3_r2"] = rr_emergence_series[i]["layer_3_r2"]
    with open("outputs/1M/emergence_summary.json", "w") as f:
        json.dump(existing_em, f, indent=2)
    print("Updated outputs/1M/emergence_summary.json with verified Layer 3 emergence")

    # -------------------------------------------------------------
    # 2. Matched 20/80 CV Shuffle Null for RRXOR (Concat & Layer 3)
    # -------------------------------------------------------------
    print("\n--- 2. Computing Matched 20/80 CV Shuffle Null (1,000 Iterations) ---")
    final_rr_ckpt = Path("checkpoints/1M/RRXOR/step_1000000.pt")
    model_rr_final, _ = load_model(final_rr_ckpt, d_vocab=2)
    final_rr_acts = extract_dataset_activations(model_rr_final, ds_rr, batch_size=1024, device="cpu")

    print("Running Matched CV Shuffle Null on Concat (256 dims)...")
    t0 = time.perf_counter()
    cv_shuffle_concat = run_matched_cv_shuffle_null(
        final_rr_acts["concat"], ds_rr.beliefs, train_ratio=0.20, n_iterations=1000, rng_seed=0
    )
    dt_cat = time.perf_counter() - t0
    print(f"  Concat Matched CV Shuffle: Mean Test MSE = {cv_shuffle_concat['mean_test_mse']:.6f} (+/- {cv_shuffle_concat['std_test_mse']:.6f}) [{dt_cat:.1f}s]")

    print("Running Matched CV Shuffle Null on Layer 3 (64 dims)...")
    t0 = time.perf_counter()
    cv_shuffle_l3 = run_matched_cv_shuffle_null(
        final_rr_acts["blocks.3.hook_resid_post"], ds_rr.beliefs, train_ratio=0.20, n_iterations=1000, rng_seed=0
    )
    dt_l3 = time.perf_counter() - t0
    print(f"  Layer 3 Matched CV Shuffle: Mean Test MSE = {cv_shuffle_l3['mean_test_mse']:.6f} (+/- {cv_shuffle_l3['std_test_mse']:.6f}) [{dt_l3:.1f}s]")

    # -------------------------------------------------------------
    # 3. Activation-Centroid Distance Diagnostic
    # -------------------------------------------------------------
    print("\n--- 3. Computing Activation-Centroid Distance Diagnostic ---")
    centroid_results: Dict[str, Any] = {"rrxor": {}, "mess3": {}}

    for layer_name in [
        "blocks.0.hook_resid_post",
        "blocks.1.hook_resid_post",
        "blocks.2.hook_resid_post",
        "blocks.3.hook_resid_post",
        "concat",
        "ln_final.hook_normalized",
    ]:
        res = compute_activation_centroid_distances(final_rr_acts[layer_name], ds_rr.beliefs)
        centroid_results["rrxor"][layer_name] = res
        print(f"  RRXOR {layer_name:26s} | Centroid R^2 vs Belief: {res['r2_centroid_vs_belief']:.4f} (r = {res['pearson_r']:.4f})")

    # Mess3 activation centroid diagnostic
    print("\nComputing Mess3 Final Layer Activation-Centroid Distance Diagnostic...")
    proc_m3 = get_mess3_process()
    ds_m3 = build_probe_dataset(proc_m3, seq_len=10)
    final_m3_ckpt = Path("checkpoints/1M/Mess3/step_1000000.pt")
    model_m3_final, _ = load_model(final_m3_ckpt, d_vocab=3)
    final_m3_acts = extract_dataset_activations(model_m3_final, ds_m3, batch_size=1024, device="cpu")

    # Sample 500 unique states matching existing probe distance sampling
    rng = np.random.default_rng(0)
    unique_beliefs_m3, inv_idx_m3 = np.unique(np.round(ds_m3.beliefs, 8), axis=0, return_inverse=True)
    sample_500 = rng.choice(len(unique_beliefs_m3), size=500, replace=False)

    m3_centroid_res = compute_activation_centroid_distances(
        final_m3_acts["blocks.3.hook_resid_post"], ds_m3.beliefs, sample_indices=sample_500
    )
    centroid_results["mess3"]["blocks.3.hook_resid_post"] = m3_centroid_res
    print(f"  Mess3 Final Layer Pre-LN | Centroid R^2 vs Belief: {m3_centroid_res['r2_centroid_vs_belief']:.4f} (r = {m3_centroid_res['pearson_r']:.4f})")

    m3_post_ln_res = compute_activation_centroid_distances(
        final_m3_acts["ln_final.hook_normalized"], ds_m3.beliefs, sample_indices=sample_500
    )
    centroid_results["mess3"]["ln_final.hook_normalized"] = m3_post_ln_res
    print(f"  Mess3 Final Layer Post-LN | Centroid R^2 vs Belief: {m3_post_ln_res['r2_centroid_vs_belief']:.4f} (r = {m3_post_ln_res['pearson_r']:.4f})")

    with open("outputs/1M/activation_centroid_distances.json", "w") as f:
        json.dump(centroid_results, f, indent=2)
    print("Saved centroid distances to outputs/1M/activation_centroid_distances.json")

    # -------------------------------------------------------------
    # 4. Update probe_results_summary.json with new controls
    # -------------------------------------------------------------
    with open("outputs/1M/probe_results_summary.json", "r") as f:
        summary_all = json.load(f)

    summary_all["rrxor"]["matched_cv_shuffle_concat"] = cv_shuffle_concat
    summary_all["rrxor"]["matched_cv_shuffle_final"] = cv_shuffle_l3
    summary_all["rrxor"]["activation_centroid_distance_r2"] = {
        k: v["r2_centroid_vs_belief"] for k, v in centroid_results["rrxor"].items()
    }
    summary_all["mess3"]["activation_centroid_distance_r2"] = {
        k: v["r2_centroid_vs_belief"] for k, v in centroid_results["mess3"].items()
    }

    with open("outputs/1M/probe_results_summary.json", "w") as f:
        json.dump(summary_all, f, indent=2)
    print("Updated outputs/1M/probe_results_summary.json with matched CV shuffle null and centroid diagnostics")

    # Also update root outputs/ copies
    with open("outputs/emergence_summary.json", "w") as f:
        json.dump(existing_em, f, indent=2)
    with open("outputs/probe_results_summary.json", "w") as f:
        json.dump(summary_all, f, indent=2)
    with open("outputs/rrxor_layer_emergence.json", "w") as f:
        json.dump(rr_emergence_series, f, indent=2)
    with open("outputs/activation_centroid_distances.json", "w") as f:
        json.dump(centroid_results, f, indent=2)
    print("Updated root outputs/ files.")


if __name__ == "__main__":
    main()
