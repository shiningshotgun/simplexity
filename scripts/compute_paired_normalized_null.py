"""Compute paired normalized CV shuffle null for RRXOR (concat and layer 3).

Follows Part 3 specification:
- 1,000 indexed 20/80 train/test splits on full 8,108 valid samples.
- Real arm: test MSE, weighted test target variance, normalized score = test MSE / test variance.
- Null arm: for permutation i, permute 35 distinct state labels, evaluate on split i, record test MSE, permuted test variance, normalized score.
- Zero-variance handling: record and report degenerate count.
- Results saved to outputs/1M/paired_normalized_null.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict
import numpy as np
import torch

from simplexity.model import SimplexityModelConfig, build_hooked_transformer
from simplexity.processes import get_rrxor_process
from simplexity.probes import build_probe_dataset, extract_dataset_activations


def main():
    print("=" * 70)
    print("COMPUTING PAIRED NORMALIZED CV SHUFFLE NULL (RRXOR)")
    print("=" * 70)

    proc = get_rrxor_process()
    dataset = build_probe_dataset(proc, seq_len=10)

    cfg = SimplexityModelConfig(d_vocab=2, seed=0)
    model = build_hooked_transformer(cfg, device="cpu")
    ckpt_path = Path("checkpoints/1M/RRXOR/step_1000000.pt")
    data = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    acts = extract_dataset_activations(model, dataset, batch_size=1024, device="cpu")
    targets = dataset.beliefs

    unique_beliefs, inv_idx = np.unique(np.round(targets, 8), axis=0, return_inverse=True)
    num_distinct = len(unique_beliefs)
    N = len(targets)
    n_train = int(round(N * 0.20))
    n_test = N - n_train
    print(f"Dataset: N={N}, n_train={n_train}, n_test={n_test}, distinct belief states={num_distinct}")

    rng = np.random.default_rng(0)
    splits = [rng.permutation(N) for _ in range(1000)]
    perms = [rng.permutation(num_distinct) for _ in range(1000)]

    def evaluate_representation(feat_name: str) -> Dict[str, Any]:
        A_feat = acts[feat_name]
        A_aug = np.hstack([A_feat, np.ones((N, 1), dtype=A_feat.dtype)])

        real_mses = []
        real_vars = []
        real_norms = []

        null_mses = []
        null_vars = []
        null_norms = []

        degen_real = 0
        degen_null = 0

        t0 = time.perf_counter()
        for i in range(1000):
            perm_idx = splits[i]
            tr = perm_idx[:n_train]
            te = perm_idx[n_train:]

            A_tr = A_aug[tr]
            A_te = A_aug[te]

            AtA = A_tr.T @ A_tr
            AtA_inv = np.linalg.pinv(AtA)
            M_tr = AtA_inv @ A_tr.T

            # Real arm
            T_tr_real = targets[tr]
            T_te_real = targets[te]

            var_te_real = float(np.mean(np.sum((T_te_real - np.mean(T_te_real, axis=0)) ** 2, axis=1)))
            W_real = M_tr @ T_tr_real
            pred_te_real = A_te @ W_real
            mse_te_real = float(np.mean(np.sum((T_te_real - pred_te_real) ** 2, axis=1)))

            if var_te_real <= 1e-12:
                degen_real += 1
            else:
                real_mses.append(mse_te_real)
                real_vars.append(var_te_real)
                real_norms.append(mse_te_real / var_te_real)

            # Null arm
            perm_states = perms[i]
            shuffled_targets = unique_beliefs[perm_states][inv_idx]
            T_tr_null = shuffled_targets[tr]
            T_te_null = shuffled_targets[te]

            var_te_null = float(np.mean(np.sum((T_te_null - np.mean(T_te_null, axis=0)) ** 2, axis=1)))
            W_null = M_tr @ T_tr_null
            pred_te_null = A_te @ W_null
            mse_te_null = float(np.mean(np.sum((T_te_null - pred_te_null) ** 2, axis=1)))

            if var_te_null <= 1e-12:
                degen_null += 1
            else:
                null_mses.append(mse_te_null)
                null_vars.append(var_te_null)
                null_norms.append(mse_te_null / var_te_null)

        dt = time.perf_counter() - t0
        print(f"Evaluated {feat_name} in {dt:.1f}s | Degenerate real: {degen_real}, null: {degen_null}")

        def stats(arr: list[float]) -> Dict[str, float]:
            a = np.array(arr)
            return {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
            }

        return {
            "feature_name": feat_name,
            "feature_dim": A_feat.shape[1],
            "n_iterations": len(real_mses),
            "degenerate_real_count": degen_real,
            "degenerate_null_count": degen_null,
            "real_raw_test_mse": stats(real_mses),
            "real_test_target_variance": stats(real_vars),
            "real_normalized_score": stats(real_norms),
            "null_raw_test_mse": stats(null_mses),
            "null_test_target_variance": stats(null_vars),
            "null_normalized_score": stats(null_norms),
        }

    results = {
        "dataset_name": "RRXOR",
        "n_samples": N,
        "n_train": n_train,
        "n_test": n_test,
        "num_distinct_states": num_distinct,
        "concat": evaluate_representation("concat"),
        "layer_3": evaluate_representation("blocks.3.hook_resid_post"),
    }

    out_file = Path("outputs/1M/paired_normalized_null.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # Mirror to outputs/
    root_out = Path("outputs/paired_normalized_null.json")
    with open(root_out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {out_file} and {root_out}")


if __name__ == "__main__":
    main()
