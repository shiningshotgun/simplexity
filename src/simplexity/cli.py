"""Unified command-line interface for simplexity reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

from simplexity.belief import enumerate_rrxor_rational_beliefs
from simplexity.entropy import compute_optimal_finite_context_loss
from simplexity.processes import get_mess3_process, get_rrxor_process
from simplexity.train import run_pilot
from simplexity.visualization import plot_mess3_ground_truth


def cmd_run_gates(args: argparse.Namespace) -> int:
    """Execute and report Pre-training Gates 1 through 5."""
    print("=" * 60)
    print("SIMPLEXITY PRE-TRAINING GATES VERIFICATION (Shai et al. 2024)")
    print("=" * 60)

    all_passed = True

    # Gate 1 & 2: Row sums and Stationary distributions
    mess3 = get_mess3_process()
    rrxor = get_rrxor_process()

    m3_row_sums = mess3.full_transition_matrix.sum(axis=1)
    rr_row_sums = rrxor.full_transition_matrix.sum(axis=1)

    g2_m3_pass = bool(np.allclose(m3_row_sums, 1.0, atol=1e-12))
    g2_rr_pass = bool(np.allclose(rr_row_sums, 1.0, atol=1e-12))
    g2_pass = g2_m3_pass and g2_rr_pass

    print(f"\n[Gate 2] Row sums (sum over token, next-state = 1):")
    print(f"  Mess3: {m3_row_sums.tolist()} -> {'PASS' if g2_m3_pass else 'FAIL'}")
    print(f"  RRXOR: {rr_row_sums.tolist()} -> {'PASS' if g2_rr_pass else 'FAIL'}")
    all_passed = all_passed and g2_pass

    # Gate 1: Stationary distribution is left eigenvector with eigenvalue 1
    m3_pi_stat = mess3.stationary_dist @ mess3.full_transition_matrix
    g1_m3_diff = float(np.max(np.abs(m3_pi_stat - mess3.stationary_dist)))
    g1_m3_pass = g1_m3_diff < 1e-12

    rr_expected_pi = np.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    rr_pi_stat = rrxor.stationary_dist @ rrxor.full_transition_matrix
    g1_rr_diff = float(np.max(np.abs(rr_pi_stat - rrxor.stationary_dist)))
    g1_rr_exp_diff = float(np.max(np.abs(rrxor.stationary_dist - rr_expected_pi)))
    g1_rr_pass = (g1_rr_diff < 1e-12) and (g1_rr_exp_diff < 1e-12)
    g1_pass = g1_m3_pass and g1_rr_pass

    print(f"\n[Gate 1] Stationary distribution left eigenvector (pi T = pi):")
    print(
        f"  Mess3 pi: {mess3.stationary_dist.tolist()} (max diff: {g1_m3_diff:.2e}) -> {'PASS' if g1_m3_pass else 'FAIL'}"
    )
    print(
        f"  RRXOR pi: {rrxor.stationary_dist.tolist()} (max diff from derived [1/3, 1/6, 1/6, 1/6, 1/6]: {g1_rr_exp_diff:.2e}) -> {'PASS' if g1_rr_pass else 'FAIL'}"
    )
    all_passed = all_passed and g1_pass

    # Gate 3: RRXOR reachable belief states exact rational arithmetic = 36
    print(f"\n[Gate 3] RRXOR reachable belief states (exact rational enumeration):")
    cap = args.rrxor_cap if hasattr(args, "rrxor_cap") else 10
    beliefs_set, history = enumerate_rrxor_rational_beliefs(max_length=cap)
    for h in history:
        print(
            f"  Length {h['length']:2d}: frontier={h['frontier_size']:2d}, cumulative distinct={h['cumulative_unique']:2d}"
        )
    final_count = len(beliefs_set)
    g3_pass = final_count == 36 and history[6]["cumulative_unique"] == 36
    print(
        f"  Final distinct belief states: {final_count} (stabilized at length 6, expected 36) -> {'PASS' if g3_pass else 'FAIL'}"
    )
    all_passed = all_passed and g3_pass

    # Gate 4: Mess3 ground-truth geometry plot
    print(f"\n[Gate 4] Mess3 ground-truth geometry (2-simplex RGB plot):")
    gt_path = plot_mess3_ground_truth(
        output_path="figures/GT_mess3.png", max_length=10
    )
    g4_pass = gt_path.exists() and gt_path.stat().st_size > 10_000
    print(
        f"  Saved {gt_path} ({gt_path.stat().st_size / 1024:.1f} KB) -> {'PASS' if g4_pass else 'FAIL'}"
    )
    all_passed = all_passed and g4_pass

    # Gate 5: Optimal-loss normalization constant
    print(f"\n[Gate 5] Optimal-loss normalization constant (finite context 10, 9 positions):")
    loss_m3 = compute_optimal_finite_context_loss(mess3, num_trained_positions=9)
    loss_rr = compute_optimal_finite_context_loss(rrxor, num_trained_positions=9)

    print(f"  Mess3 optimal loss average: {loss_m3.average_optimal_loss:.6f} nats")
    for pos, h in enumerate(loss_m3.per_position_entropy):
        print(f"    pos {pos} (predicts token {pos+1}): {h:.6f} nats")

    print(f"\n  RRXOR optimal loss average: {loss_rr.average_optimal_loss:.6f} nats")
    for pos, h in enumerate(loss_rr.per_position_entropy):
        print(f"    pos {pos} (predicts token {pos+1}): {h:.6f} nats")

    # Pos 0 of RRXOR must be ln(2)
    pos0_err = abs(loss_rr.per_position_entropy[0] - np.log(2.0))
    g5_pass = pos0_err < 1e-10 and loss_m3.average_optimal_loss > 0 and loss_rr.average_optimal_loss > 0
    print(
        f"  RRXOR pos 0 == ln(2) check error: {pos0_err:.2e} -> {'PASS' if g5_pass else 'FAIL'}"
    )
    all_passed = all_passed and g5_pass

    print("\n" + "=" * 60)
    print(f"PRE-TRAINING GATES SUMMARY: {'ALL 5 GATES PASSED' if all_passed else 'SOME GATES FAILED'}")
    print("=" * 60)
    return 0 if all_passed else 1


def cmd_pilot(args: argparse.Namespace) -> int:
    """Run 1,000-iteration pilot benchmark for Mess3 and RRXOR."""
    print("Running measured 1,000-iteration pilot on local machine...")
    device = "mps" if (hasattr(args, "use_mps") and args.use_mps) else "cpu"

    for proc in ["Mess3", "RRXOR"]:
        print(f"\n--- {proc} Pilot Run ---")
        res = run_pilot(process_name=proc, pilot_steps=1000, device=device)
        print(f"  Model parameters: {res.parameter_count['total']:,}")
        print(f"  Elapsed for 1,000 steps: {res.elapsed_seconds:.2f} s")
        print(f"  Throughput: {res.steps_per_second:.1f} steps/s")
        print(f"  Peak memory: {res.peak_memory_mb:.1f} MB")
        print(f"  Extrapolated 1,000,000 steps time: {res.estimated_total_hours:.2f} hours ({res.estimated_total_seconds / 60:.1f} minutes)")


def cmd_train(args: argparse.Namespace) -> int:
    """Train a process for the specified number of steps."""
    from simplexity.train import TrainingConfig, train_model
    d_vocab = 3 if args.process == "Mess3" else 2
    cfg = TrainingConfig(
        process_name=args.process,
        d_vocab=d_vocab,
        total_steps=args.steps,
        device=args.device,
        prefer_hooked=not args.use_native_pytorch,
    )
    ckpt_dir = Path(args.checkpoint_dir) / args.process
    metrics_path = Path(args.output_dir) / f"{args.process}_metrics.json"
    train_model(cfg, ckpt_dir, metrics_path, device=args.device)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplexity replication CLI")
    subparsers = parser.add_subparsers(dest="command")

    # run-gates
    sub_gates = subparsers.add_parser("run-gates", help="Run Gates 1-5")
    sub_gates.set_defaults(func=cmd_run_gates)

    # pilot
    sub_pilot = subparsers.add_parser("pilot", help="Run 1,000-step pilot")
    sub_pilot.add_argument("--use-mps", action="store_true", help="Use Apple Metal GPU")
    sub_pilot.set_defaults(func=cmd_pilot)

    # train
    sub_train = subparsers.add_parser("train", help="Train a model")
    sub_train.add_argument(
        "--process", choices=["Mess3", "RRXOR"], required=True, help="Process to train"
    )
    sub_train.add_argument(
        "--steps", type=int, default=1_000_000, help="Total training steps (default: 1M)"
    )
    sub_train.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/1M", help="Checkpoints root directory"
    )
    sub_train.add_argument(
        "--output-dir", type=str, default="outputs/1M", help="Outputs root directory"
    )
    sub_train.add_argument(
        "--device", type=str, default="cpu", help="Device to train on (default: cpu)"
    )
    sub_train.add_argument(
        "--use-native-pytorch",
        action="store_true",
        help="Use native SimplexityTransformer instead of HookedTransformer",
    )
    sub_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
