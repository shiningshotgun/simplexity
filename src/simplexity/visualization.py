"""Visualization routines for 2-simplex belief geometries and paper figures.

Reproduces Figure 5B (ground-truth Mess3 fractal in 2-simplex) and provides
plotting utilities for representations and probes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from simplexity.belief import enumerate_all_prefix_beliefs
from simplexity.processes import get_mess3_process


def simplex_2d_projection(beliefs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map 3-state probability distributions onto 2D equilateral triangle coordinates.

    Oriented to match paper Figure 5B:
        - Vertex 0 (state 0 / token A, Red):   (0.0, 0.0) [bottom-left]
        - Vertex 1 (state 1 / token B, Green): (1.0, 0.0) [bottom-right]
        - Vertex 2 (state 2 / token C, Blue):  (0.5, sqrt(3)/2) [top]

    Args:
        beliefs: (N, 3) array of belief state vectors.

    Returns:
        x, y: 1D arrays of 2D Cartesian coordinates.
    """
    b0 = beliefs[:, 0]
    b1 = beliefs[:, 1]
    b2 = beliefs[:, 2]

    x = b1 + 0.5 * b2
    y = (np.sqrt(3.0) / 2.0) * b2
    return x, y


def plot_mess3_ground_truth(
    output_path: str | Path = "figures/GT_mess3.png",
    max_length: int = 10,
    point_size: float = 1.0,
    alpha: float = 0.6,
    dpi: int = 300,
) -> Path:
    """Generate and save the Mess3 ground truth belief state geometry plot (Gate 4).

    Computes beliefs for all prefixes of length <= max_length (88,573 points for len 10),
    projects them to the 2-simplex, and colors each point by its RGB belief distribution.

    Args:
        output_path: Destination path for the saved image.
        max_length: Maximum prefix length to enumerate.
        point_size: Scatter point size.
        alpha: Point transparency.
        dpi: Image resolution.

    Returns:
        Path to saved PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process = get_mess3_process()
    beliefs, _ = enumerate_all_prefix_beliefs(process, max_length=max_length)

    x, y = simplex_2d_projection(beliefs)
    colors = np.clip(beliefs, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 5.2), dpi=dpi)

    # Plot simplex boundary
    v0 = (0.0, 0.0)
    v1 = (1.0, 0.0)
    v2 = (0.5, np.sqrt(3.0) / 2.0)
    tri_x = [v0[0], v1[0], v2[0], v0[0]]
    tri_y = [v0[1], v1[1], v2[1], v0[1]]
    ax.plot(tri_x, tri_y, color="#444444", lw=1.2, alpha=0.7)

    # Scatter points colored by ground-truth belief RGB
    ax.scatter(x, y, c=colors, s=point_size, alpha=alpha, edgecolors="none")

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_mess3_representation(
    predicted_beliefs: np.ndarray,
    ground_truth_beliefs: np.ndarray,
    output_path: str | Path = "figures/fig5c_mess3_repr.png",
    point_size: float = 1.0,
    alpha: float = 0.6,
    dpi: int = 300,
) -> Path:
    """Generate Mess3 headline representation plot (paper Fig 5C).

    Projects probe predictions from final residual stream activations into the 2-simplex,
    colored by ground truth belief states.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize predictions if needed so they lie on simplex
    preds_norm = np.clip(predicted_beliefs, 0.0, 1.0)
    row_sums = preds_norm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    preds_norm = preds_norm / row_sums

    x, y = simplex_2d_projection(preds_norm)
    colors = np.clip(ground_truth_beliefs, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 5.2), dpi=dpi)

    # Simplex boundary
    v0 = (0.0, 0.0)
    v1 = (1.0, 0.0)
    v2 = (0.5, np.sqrt(3.0) / 2.0)
    ax.plot([v0[0], v1[0], v2[0], v0[0]], [v0[1], v1[1], v2[1], v0[1]], color="#444444", lw=1.2, alpha=0.7)

    # Scatter points colored by ground-truth RGB
    ax.scatter(x, y, c=colors, s=point_size, alpha=alpha, edgecolors="none")

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_rrxor_projections(
    layer_mses: Dict[str, float],
    true_distinct_beliefs: np.ndarray,
    concat_predicted_beliefs: np.ndarray,
    final_layer_predicted_beliefs: np.ndarray,
    output_path_layers: str | Path = "figures/fig7c_rrxor_layers.png",
    output_path_concat: str | Path = "figures/fig7e_rrxor_concat.png",
    dpi: int = 300,
) -> Tuple[Path, Path]:
    """Generate RRXOR headline layer MSE and concatenated projection figures (paper Fig 7C/E)."""
    output_path_layers = Path(output_path_layers)
    output_path_concat = Path(output_path_concat)
    output_path_layers.parent.mkdir(parents=True, exist_ok=True)

    # Fig 7C: Per-layer MSE comparison
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    layers = ["Layer 0", "Layer 1", "Layer 2", "Layer 3", "Concatenated"]
    keys = [
        "blocks.0.hook_resid_post",
        "blocks.1.hook_resid_post",
        "blocks.2.hook_resid_post",
        "blocks.3.hook_resid_post",
        "concat",
    ]
    mses = [layer_mses.get(k, 0.0) for k in keys]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172D5", "#CCB974"]

    bars = ax.bar(layers, mses, color=colors, width=0.6, edgecolor="black", alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.0005,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Linear Probe MSE", fontsize=11)
    ax.set_title("RRXOR: Probe MSE by Layer vs Concatenation", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path_layers, dpi=dpi)
    plt.close(fig)

    # Fig 7E: 2D PCA projection of distinct belief states (Ground Truth vs Concat vs Final Layer)
    # Using pure numpy SVD for PCA
    gt_mean = np.mean(true_distinct_beliefs, axis=0)
    gt_centered = true_distinct_beliefs - gt_mean
    _, _, vt = np.linalg.svd(gt_centered, full_matrices=False)
    proj_matrix = vt[:2, :].T  # (k, 2)

    gt_2d = gt_centered @ proj_matrix
    concat_2d = (concat_predicted_beliefs - gt_mean) @ proj_matrix
    final_2d = (final_layer_predicted_beliefs - gt_mean) @ proj_matrix

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=dpi)

    axes[0].scatter(gt_2d[:, 0], gt_2d[:, 1], c="#2b5c8f", s=40, edgecolors="k", alpha=0.9)
    axes[0].set_title("Ground Truth Beliefs (36 states)", fontsize=11)
    axes[0].set_xlabel("PCA Component 1", fontsize=10)
    axes[0].set_ylabel("PCA Component 2", fontsize=10)
    axes[0].set_aspect("equal")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].scatter(final_2d[:, 0], final_2d[:, 1], c="#d95f02", s=40, edgecolors="k", alpha=0.9)
    axes[1].set_title(f"Final Layer Probe (MSE={layer_mses.get('blocks.3.hook_resid_post', 0):.4f})", fontsize=11)
    axes[1].set_xlabel("PCA Component 1", fontsize=10)
    axes[1].set_ylabel("PCA Component 2", fontsize=10)
    axes[1].set_aspect("equal")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].scatter(concat_2d[:, 0], concat_2d[:, 1], c="#1b9e77", s=40, edgecolors="k", alpha=0.9)
    axes[2].set_title(f"Concatenated Probe (MSE={layer_mses.get('concat', 0):.4f})", fontsize=11)
    axes[2].set_xlabel("PCA Component 1", fontsize=10)
    axes[2].set_ylabel("PCA Component 2", fontsize=10)
    axes[2].set_aspect("equal")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path_concat, dpi=dpi)
    plt.close(fig)

    return output_path_layers, output_path_concat


def plot_distance_analysis(
    pairwise_stats: Dict[str, Any],
    process_name: str,
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Generate pairwise distance correlation plots (paper Fig 7D)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    d_true = pairwise_stats["d_true"]
    d_repr = pairwise_stats["d_repr"]
    d_next = pairwise_stats["d_next"]
    r2_repr = pairwise_stats["r2_repr_vs_belief"]
    r2_next = pairwise_stats["r2_next_token_vs_belief"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=dpi)

    # Next-token distances vs true belief distances
    ax1.scatter(d_true, d_next, color="#e74c3c", alpha=0.6, s=15, edgecolors="none")
    m_next, b_next = np.polyfit(d_true, d_next, 1)
    x_line = np.linspace(d_true.min(), d_true.max(), 100)
    ax1.plot(x_line, m_next * x_line + b_next, color="black", lw=1.5, linestyle="--")
    ax1.set_xlabel("True Belief Distance", fontsize=11)
    ax1.set_ylabel("Next-Token Distribution Distance", fontsize=11)
    ax1.set_title(f"{process_name}: Next-Token vs Belief ($R^2 = {r2_next:.2f}$)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Representation distances vs true belief distances
    ax2.scatter(d_true, d_repr, color="#2980b9", alpha=0.6, s=15, edgecolors="none")
    m_repr, b_repr = np.polyfit(d_true, d_repr, 1)
    ax2.plot(x_line, m_repr * x_line + b_repr, color="black", lw=1.5, linestyle="--")
    ax2.set_xlabel("True Belief Distance", fontsize=11)
    ax2.set_ylabel("Representation Distance", fontsize=11)
    ax2.set_title(f"{process_name}: Representation vs Belief ($R^2 = {r2_repr:.2f}$)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return output_path


def plot_emergence_and_normalized_loss(
    emergence_data: Dict[str, List[Dict[str, Any]]],
    output_path_emergence: str | Path = "figures/fig6_emergence.png",
    output_path_s2: str | Path = "figures/figS2_normalized_loss.png",
    dpi: int = 300,
) -> Tuple[Path, Path]:
    """Generate training dynamics emergence curves (Fig 6A/D and Fig S2)."""
    output_path_emergence = Path(output_path_emergence)
    output_path_s2 = Path(output_path_s2)
    output_path_emergence.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi)

    for proc_name, points in emergence_data.items():
        steps = [p["step"] for p in points if "probe_mse" in p]
        mses = [p["probe_mse"] for p in points if "probe_mse" in p]
        if proc_name == "Mess3":
            label = "Mess3: Final-Layer Probe (blocks.3.hook_resid_post)"
            color = "#1f77b4"
        else:
            label = "RRXOR: Concatenated Probe (blocks.{0,1,2,3})"
            color = "#ff7f0e"
        ax.plot(steps, mses, marker="o", label=label, color=color, lw=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training Steps (SGD Iterations)", fontsize=11)
    ax.set_ylabel("Linear Probe MSE", fontsize=11)
    ax.set_title("Belief Representation Emergence Over Training", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path_emergence, dpi=dpi)
    plt.close(fig)

    # Fig S2: Probe MSE vs Normalized Validation Loss
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi)

    for proc_name, points in emergence_data.items():
        val_losses_norm = [p["val_loss_normalized"] for p in points if "probe_mse" in p and "val_loss_normalized" in p]
        mses = [p["probe_mse"] for p in points if "probe_mse" in p and "val_loss_normalized" in p]
        color = "#1f77b4" if proc_name == "Mess3" else "#ff7f0e"
        ax.scatter(val_losses_norm, mses, label=proc_name, color=color, s=40, edgecolors="k", alpha=0.85)

    ax.axvline(1.0, color="gray", linestyle=":", label="Optimal Loss (1.0)")
    ax.set_yscale("log")
    ax.set_xlabel("Normalized Validation Loss ($H / H^*$)", fontsize=11)
    ax.set_ylabel("Linear Probe MSE", fontsize=11)
    ax.set_title("Probe MSE vs Normalized Validation Loss (Fig S2)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path_s2, dpi=dpi)
    plt.close(fig)

    return output_path_emergence, output_path_s2
