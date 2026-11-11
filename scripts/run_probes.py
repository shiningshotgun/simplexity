"""Comprehensive probe analysis script for Shai et al. (NeurIPS 2024) reproduction.

Implements:
1. Mess3 headline representation (paper Fig 5C) on full dataset (590,490 samples).
2. RRXOR per-layer and concatenated representations (paper Fig 7C/E) on all valid samples.
3. Statistical controls on FULL datasets: 1000x shuffle control (permuting distinct belief states)
   and 1000x 20/80 cross-validation (App A.7).
4. Pairwise distance correlation analysis (paper Fig 7D) declaring probe-predicted beliefs as representations.
5. Emergence dynamics across training snapshots (paper Fig 6A/D & Fig S2).
6. LayerNorm robustness (paper Fig S1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from simplexity.belief import enumerate_all_prefix_beliefs, enumerate_rrxor_rational_beliefs
from simplexity.model import SimplexityModelConfig, build_hooked_transformer
from simplexity.processes import get_mess3_process, get_rrxor_process
from simplexity.probes import (
    build_probe_dataset,
    compute_next_token_distributions,
    compute_pairwise_distance_correlation,
    extract_dataset_activations,
    fit_linear_probe,
    run_cross_validation_control,
    run_shuffle_control,
)
from simplexity.visualization import (
    plot_distance_analysis,
    plot_emergence_and_normalized_loss,
    plot_mess3_representation,
    plot_rrxor_projections,
)


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    d_vocab: int,
    device: str = "cpu",
):
    """Load HookedTransformer weights from checkpoint file."""
    cfg = SimplexityModelConfig(d_vocab=d_vocab, seed=0)
    model = build_hooked_transformer(cfg, device=device)
    ckpt_data = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    return model, ckpt_data


def analyze_mess3(
    checkpoint_path: Path | str,
    output_fig_dir: Path | str = "figures",
    device: str = "cpu",
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """Run full probe analysis for Mess3 on full dataset without subsampling."""
    print("\n" + "=" * 60)
    print("MESS3 PROBE ANALYSIS (FULL DATASET)")
    print("=" * 60)

    output_fig_dir = Path(output_fig_dir)
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)

    process = get_mess3_process()
    model, ckpt_data = load_model_from_checkpoint(checkpoint_path, d_vocab=3, device=device)
    step = ckpt_data.get("step", "final")
    print(f"Loaded Mess3 model from step {step}")

    print("Building Mess3 probe dataset (3^10 = 59,049 sequences x 10 positions = 590,490 samples)...")
    dataset = build_probe_dataset(process, seq_len=10)
    print(f"Dataset ready: {dataset.retained_samples:,} samples (all candidates valid)")

    print("Extracting residual stream activations...")
    activations = extract_dataset_activations(model, dataset, batch_size=1024, device=device)

    # 1. Final residual stream probe (pre-final-LayerNorm: blocks.3.hook_resid_post)
    print("Fitting linear probe on final residual stream (blocks.3.hook_resid_post)...")
    final_act = activations["blocks.3.hook_resid_post"]
    probe_final = fit_linear_probe(final_act, dataset.beliefs)
    print(f"  Final layer train MSE: {probe_final.train_mse:.6f} | R^2: {probe_final.train_r2:.4f}")

    # 2. Headline Fig 5C: 2-simplex plot of predicted beliefs
    preds_final = probe_final.predict(final_act)
    fig5c_path = output_fig_dir / "fig5c_mess3_repr.png"
    plot_mess3_representation(preds_final, dataset.beliefs, output_path=fig5c_path)
    print(f"  Saved Fig 5C to {fig5c_path}")

    # 3. LayerNorm robustness (paper Fig S1)
    post_ln_act = activations["ln_final.hook_normalized"]
    probe_post_ln = fit_linear_probe(post_ln_act, dataset.beliefs)
    print(f"  Post-LayerNorm train MSE: {probe_post_ln.train_mse:.6f} | R^2: {probe_post_ln.train_r2:.4f}")

    # 4. Statistical Controls on the FULL 590,490 dataset (1000 shuffles and 1000 CV splits)
    print("Running 1,000x belief-state label shuffle control on full dataset (590,490 samples)...")
    shuffle_res = run_shuffle_control(final_act, dataset.beliefs, n_iterations=1000, rng=rng)
    print(f"  Shuffle control MSE: mean={shuffle_res['mean_mse']:.6f}, std={shuffle_res['std_mse']:.6f} (distinct states={shuffle_res['num_distinct_states']})")

    print("Running 1,000x 20/80 cross-validation on full dataset (118,098 train / 472,392 test)...")
    cv_res_final = run_cross_validation_control(final_act, dataset.beliefs, train_ratio=0.20, n_iterations=1000, rng=rng)
    print(f"  Final layer CV test MSE: mean={cv_res_final['mean_test_mse']:.6f}, std={cv_res_final['std_test_mse']:.6f}")

    cv_res_post_ln = run_cross_validation_control(post_ln_act, dataset.beliefs, train_ratio=0.20, n_iterations=1000, rng=rng)
    print(f"  Post-LN CV test MSE: mean={cv_res_post_ln['mean_test_mse']:.6f}, std={cv_res_post_ln['std_test_mse']:.6f}")

    # 5. Pairwise distance correlation (paper Fig 7D)
    # Representation defined as probe-predicted beliefs
    print("Computing pairwise distance correlation (beliefs vs representations vs next-token)...")
    unique_beliefs, u_indices = np.unique(np.round(dataset.beliefs, 4), axis=0, return_index=True)
    if len(unique_beliefs) > 500:
        sample_u = rng.choice(len(unique_beliefs), size=500, replace=False)
        sample_beliefs = unique_beliefs[sample_u]
        sample_reprs = preds_final[u_indices[sample_u]]
    else:
        sample_beliefs = unique_beliefs
        sample_reprs = preds_final[u_indices]

    next_token_dists = compute_next_token_distributions(process, sample_beliefs)
    pairwise_stats = compute_pairwise_distance_correlation(sample_beliefs, sample_reprs, next_token_dists)
    print(f"  Pairwise R^2 representation vs belief: {pairwise_stats['r2_repr_vs_belief']:.4f}")
    print(f"  Pairwise R^2 next-token vs belief: {pairwise_stats['r2_next_token_vs_belief']:.4f}")

    from scipy.spatial.distance import pdist
    pairwise_stats["d_true"] = pdist(sample_beliefs, metric="euclidean")
    pairwise_stats["d_repr"] = pdist(sample_reprs, metric="euclidean")
    pairwise_stats["d_next"] = pdist(next_token_dists, metric="euclidean")
    fig7d_m3 = output_fig_dir / "fig7d_mess3_distances.png"
    plot_distance_analysis(pairwise_stats, process_name="Mess3", output_path=fig7d_m3)
    print(f"  Saved Mess3 distance analysis to {fig7d_m3}")

    return {
        "process": "Mess3",
        "step": step,
        "n_samples": dataset.retained_samples,
        "final_layer_pre_ln_train_mse": probe_final.train_mse,
        "final_layer_pre_ln_train_r2": probe_final.train_r2,
        "final_layer_pre_ln_cv_test_mse": cv_res_final["mean_test_mse"],
        "final_layer_pre_ln_cv_test_std": cv_res_final["std_test_mse"],
        "final_layer_post_ln_train_mse": probe_post_ln.train_mse,
        "final_layer_post_ln_train_r2": probe_post_ln.train_r2,
        "final_layer_post_ln_cv_test_mse": cv_res_post_ln["mean_test_mse"],
        "final_layer_post_ln_cv_test_std": cv_res_post_ln["std_test_mse"],
        "shuffle_control": shuffle_res,
        "cross_validation_pre_ln": cv_res_final,
        "cross_validation_post_ln": cv_res_post_ln,
        "pairwise_r2_repr": pairwise_stats["r2_repr_vs_belief"],
        "pairwise_r2_next_token": pairwise_stats["r2_next_token_vs_belief"],
    }


def analyze_rrxor(
    checkpoint_path: Path | str,
    output_fig_dir: Path | str = "figures",
    device: str = "cpu",
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """Run full probe analysis for RRXOR on full valid dataset without subsampling."""
    print("\n" + "=" * 60)
    print("RRXOR PROBE ANALYSIS (FULL VALID DATASET)")
    print("=" * 60)

    output_fig_dir = Path(output_fig_dir)
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)

    process = get_rrxor_process()
    model, ckpt_data = load_model_from_checkpoint(checkpoint_path, d_vocab=2, device=device)
    step = ckpt_data.get("step", "final")
    print(f"Loaded RRXOR model from step {step}")

    print("Building RRXOR probe dataset under zero-probability prefix exclusion rule...")
    dataset = build_probe_dataset(process, seq_len=10)
    print(
        f"Observed derivation: {dataset.total_candidate_samples} total candidates (2^10 x 10), "
        f"excluded {dataset.excluded_samples} zero-probability prefixes, "
        f"retained {dataset.retained_samples} valid samples."
    )

    print("Extracting layer and concat residual stream activations...")
    activations = extract_dataset_activations(model, dataset, batch_size=512, device=device)

    # 1. Per-layer and concat probes with train MSE and 1000x 20/80 CV test MSE
    layer_train_mses: Dict[str, float] = {}
    layer_train_r2s: Dict[str, float] = {}
    layer_cv_test_mses: Dict[str, float] = {}
    layer_probes: Dict[str, Any] = {}

    keys_to_eval = [
        "blocks.0.hook_resid_post",
        "blocks.1.hook_resid_post",
        "blocks.2.hook_resid_post",
        "blocks.3.hook_resid_post",
        "concat",
        "ln_final.hook_normalized",
    ]

    print("Evaluating linear probes and 1,000x 20/80 CV across all layers on full dataset...")
    for k in keys_to_eval:
        p = fit_linear_probe(activations[k], dataset.beliefs)
        layer_train_mses[k] = p.train_mse
        layer_train_r2s[k] = p.train_r2
        layer_probes[k] = p

        cv_res = run_cross_validation_control(
            activations[k], dataset.beliefs, train_ratio=0.20, n_iterations=1000, rng=rng
        )
        layer_cv_test_mses[k] = cv_res["mean_test_mse"]
        print(
            f"  {k:26s} | Train MSE: {p.train_mse:.6f} (R^2: {p.train_r2:.4f}) | "
            f"CV Test MSE: {cv_res['mean_test_mse']:.6f} (+/- {cv_res['std_test_mse']:.6f})"
        )

    # 2. Distinct 36 belief states representations
    rational_beliefs, _ = enumerate_rrxor_rational_beliefs(max_length=10)
    distinct_beliefs_36 = np.array([[float(x) for x in b] for b in rational_beliefs])

    distinct_concat_repr = []
    distinct_final_repr = []
    for b in distinct_beliefs_36:
        dists = np.sum((dataset.beliefs - b) ** 2, axis=1)
        match_idx = int(np.argmin(dists))
        pred_concat = layer_probes["concat"].predict(activations["concat"][match_idx : match_idx + 1])[0]
        pred_final = layer_probes["blocks.3.hook_resid_post"].predict(
            activations["blocks.3.hook_resid_post"][match_idx : match_idx + 1]
        )[0]
        distinct_concat_repr.append(pred_concat)
        distinct_final_repr.append(pred_final)

    distinct_concat_repr = np.array(distinct_concat_repr)
    distinct_final_repr = np.array(distinct_final_repr)

    # Plot Headline Figures 7C and 7E
    fig7c_path = output_fig_dir / "fig7c_rrxor_layers.png"
    fig7e_path = output_fig_dir / "fig7e_rrxor_concat.png"
    plot_rrxor_projections(
        layer_mses=layer_train_mses,
        true_distinct_beliefs=distinct_beliefs_36,
        concat_predicted_beliefs=distinct_concat_repr,
        final_layer_predicted_beliefs=distinct_final_repr,
        output_path_layers=fig7c_path,
        output_path_concat=fig7e_path,
    )
    print(f"  Saved Fig 7C to {fig7c_path}")
    print(f"  Saved Fig 7E to {fig7e_path}")

    # 3. Statistical Controls on Concatenated Activations (App A.7)
    print("Running 1,000x belief-state label shuffle control on concatenated representation (36 distinct states)...")
    shuffle_res_concat = run_shuffle_control(activations["concat"], dataset.beliefs, n_iterations=1000, rng=rng)
    print(f"  Concat shuffle control MSE: mean={shuffle_res_concat['mean_mse']:.6f}, std={shuffle_res_concat['std_mse']:.6f}")

    shuffle_res_final = run_shuffle_control(activations["blocks.3.hook_resid_post"], dataset.beliefs, n_iterations=1000, rng=rng)
    print(f"  Final layer shuffle control MSE: mean={shuffle_res_final['mean_mse']:.6f}, std={shuffle_res_final['std_mse']:.6f}")

    # 4. Pairwise distance correlation (paper Fig 7D)
    # Declaring probe-predicted beliefs as representations
    print("Computing RRXOR pairwise distance correlation across 36 belief states (probe-predicted representations)...")
    next_token_dists_36 = compute_next_token_distributions(process, distinct_beliefs_36)
    pairwise_stats = compute_pairwise_distance_correlation(
        distinct_beliefs_36, distinct_concat_repr, next_token_dists_36
    )
    print(f"  RRXOR Pairwise R^2 (representation vs belief): {pairwise_stats['r2_repr_vs_belief']:.4f} (Paper reported: ~0.95)")
    print(f"  RRXOR Pairwise R^2 (next-token vs belief): {pairwise_stats['r2_next_token_vs_belief']:.4f} (Paper reported: ~0.31)")

    from scipy.spatial.distance import pdist
    pairwise_stats["d_true"] = pdist(distinct_beliefs_36, metric="euclidean")
    pairwise_stats["d_repr"] = pdist(distinct_concat_repr, metric="euclidean")
    pairwise_stats["d_next"] = pdist(next_token_dists_36, metric="euclidean")
    fig7d_rr = output_fig_dir / "fig7d_rrxor_distances.png"
    plot_distance_analysis(pairwise_stats, process_name="RRXOR", output_path=fig7d_rr)
    print(f"  Saved RRXOR distance analysis to {fig7d_rr}")

    return {
        "process": "RRXOR",
        "step": step,
        "total_candidate_samples": dataset.total_candidate_samples,
        "excluded_samples": dataset.excluded_samples,
        "retained_samples": dataset.retained_samples,
        "layer_train_mses": layer_train_mses,
        "layer_train_r2s": layer_train_r2s,
        "layer_cv_test_mses": layer_cv_test_mses,
        "final_layer_pre_ln_train_mse": layer_train_mses["blocks.3.hook_resid_post"],
        "final_layer_pre_ln_cv_test_mse": layer_cv_test_mses["blocks.3.hook_resid_post"],
        "final_layer_post_ln_train_mse": layer_train_mses["ln_final.hook_normalized"],
        "final_layer_post_ln_cv_test_mse": layer_cv_test_mses["ln_final.hook_normalized"],
        "concat_train_mse": layer_train_mses["concat"],
        "concat_train_r2": layer_train_r2s["concat"],
        "concat_cv_test_mse": layer_cv_test_mses["concat"],
        "shuffle_control_concat": shuffle_res_concat,
        "shuffle_control_final": shuffle_res_final,
        "pairwise_r2_repr": pairwise_stats["r2_repr_vs_belief"],
        "pairwise_r2_next_token": pairwise_stats["r2_next_token_vs_belief"],
    }


def analyze_emergence(
    checkpoints_base_dir: Path | str = "checkpoints/1M",
    output_metrics_dir: Path | str = "outputs/1M",
    output_fig_dir: Path | str = "figures",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate probe MSE across all saved training checkpoints (Fig 6A/D & Fig S2)."""
    print("\n" + "=" * 60)
    print("TRAINING DYNAMICS & EMERGENCE ANALYSIS")
    print("=" * 60)

    checkpoints_base_dir = Path(checkpoints_base_dir)
    output_metrics_dir = Path(output_metrics_dir)
    output_fig_dir = Path(output_fig_dir)

    emergence_summary: Dict[str, List[Dict[str, Any]]] = {}

    for proc_name in ["Mess3", "RRXOR"]:
        d_vocab = 3 if proc_name == "Mess3" else 2
        proc = get_mess3_process() if proc_name == "Mess3" else get_rrxor_process()
        ds = build_probe_dataset(proc, seq_len=10)

        # Load validation metrics curve
        metrics_file = output_metrics_dir / f"{proc_name}_metrics.json"
        val_loss_map = {}
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                for entry in json.load(f):
                    val_loss_map[entry["step"]] = entry.get("val_loss_normalized", 1.0)

        ckpt_dir = checkpoints_base_dir / proc_name
        ckpt_files = sorted(
            list(ckpt_dir.glob("step_*.pt")),
            key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0,
        )

        proc_points = []
        for ckpt_path in ckpt_files:
            step = int(ckpt_path.stem.split("_")[1])
            model, _ = load_model_from_checkpoint(ckpt_path, d_vocab=d_vocab, device=device)

            acts = extract_dataset_activations(model, ds, batch_size=1024, device=device)
            probe_key = "blocks.3.hook_resid_post" if proc_name == "Mess3" else "concat"
            probe = fit_linear_probe(acts[probe_key], ds.beliefs)

            val_norm = val_loss_map.get(step, 1.0)
            point = {
                "step": step,
                "probe_mse": probe.train_mse,
                "probe_r2": probe.train_r2,
                "val_loss_normalized": val_norm,
            }
            proc_points.append(point)
            print(f"  {proc_name} Step {step:7d}: probe MSE = {probe.train_mse:.6f} | val loss norm = {val_norm:.4f}")

        emergence_summary[proc_name] = proc_points

    fig6_path = output_fig_dir / "fig6_emergence.png"
    figs2_path = output_fig_dir / "figS2_normalized_loss.png"
    plot_emergence_and_normalized_loss(
        emergence_summary, output_path_emergence=fig6_path, output_path_s2=figs2_path
    )
    print(f"Saved emergence plots to {fig6_path} and {figs2_path}")

    with open(output_metrics_dir / "emergence_summary.json", "w") as f:
        json.dump(emergence_summary, f, indent=2)

    return emergence_summary


def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = sorted(
        list(ckpt_dir.glob("step_*.pt")),
        key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0,
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return ckpts[-1]


def main():
    parser = argparse.ArgumentParser(description="Run complete Step 3 probe analyses on full datasets")
    parser.add_argument("--checkpoints-dir", default="checkpoints/1M", help="Path to checkpoints root")
    parser.add_argument("--output-dir", default="outputs/1M", help="Path to save output JSON metrics")
    parser.add_argument("--figures-dir", default="figures", help="Path to save figures")
    parser.add_argument("--device", default="cpu", help="Compute device")
    args = parser.parse_args()

    m3_ckpt = find_latest_checkpoint(Path(args.checkpoints_dir) / "Mess3")
    rr_ckpt = find_latest_checkpoint(Path(args.checkpoints_dir) / "RRXOR")
    print(f"Using checkpoints: {m3_ckpt} and {rr_ckpt}")

    m3_res = analyze_mess3(
        checkpoint_path=m3_ckpt,
        output_fig_dir=args.figures_dir,
        device=args.device,
    )
    rr_res = analyze_rrxor(
        checkpoint_path=rr_ckpt,
        output_fig_dir=args.figures_dir,
        device=args.device,
    )
    em_res = analyze_emergence(
        checkpoints_base_dir=args.checkpoints_dir,
        output_metrics_dir=args.output_dir,
        output_fig_dir=args.figures_dir,
        device=args.device,
    )

    out_file = Path(args.output_dir) / "probe_results_summary.json"
    with open(out_file, "w") as f:
        json.dump({"mess3": m3_res, "rrxor": rr_res}, f, indent=2)
    print(f"\nAll analyses complete! Summary saved to {out_file}")


if __name__ == "__main__":
    main()
