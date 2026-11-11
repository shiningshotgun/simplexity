"""Orchestrator script to train Mess3 and RRXOR models sequentially for 1,000,000 steps on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import torch

from simplexity.train import TrainingConfig, train_model


def run_process_training(
    process_name: str,
    total_steps: int = 1_000_000,
    checkpoint_dir: Path | str = "checkpoints/1M",
    output_dir: Path | str = "outputs/1M",
    log_interval: int = 5000,
    device: str = "cpu",
):
    """Run full sequential training for a single process."""
    d_vocab = 3 if process_name == "Mess3" else 2
    ckpt_dir = Path(checkpoint_dir) / process_name
    out_dir = Path(output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"{process_name}_metrics.json"

    cfg = TrainingConfig(
        process_name=process_name,
        d_vocab=d_vocab,
        total_steps=total_steps,
        device=device,
        prefer_hooked=True,
    )

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting sequential run for {process_name} ({total_steps:,} steps)...")
    t0 = time.perf_counter()
    summary = train_model(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        output_metrics_path=metrics_path,
        device=device,
        log_interval=log_interval,
    )
    dt = time.perf_counter() - t0
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Completed {process_name} in {dt:.1f}s ({dt/3600.0:.2f}h) | "
        f"Avg throughput: {summary['average_steps_per_second']:.1f} steps/s | "
        f"Peak memory: {summary['peak_memory_mb']:.1f} MB | "
        f"Final val loss norm: {summary['final_val_loss_normalized']:.4f}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Mess3 and RRXOR models sequentially")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total steps per process (default: 1,000,000)")
    parser.add_argument("--process", type=str, choices=["Mess3", "RRXOR", "both"], default="both", help="Which process to run")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/1M", help="Root directory for checkpoints")
    parser.add_argument("--output-dir", type=str, default="outputs/1M", help="Root directory for metrics and logs")
    parser.add_argument("--log-interval", type=int, default=5000, help="Logging step interval")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu)")
    args = parser.parse_args()

    processes = ["Mess3", "RRXOR"] if args.process == "both" else [args.process]

    print("=" * 70)
    print("SIMPLEXITY SEQUENTIAL FULL TRAINING ORCHESTRATOR")
    print(f"Target steps per process: {args.steps:,}")
    print(f"Processes to run sequentially: {processes}")
    print(f"Checkpoint root: {args.checkpoint_dir}")
    print(f"Output root: {args.output_dir}")
    print("=" * 70)

    total_start = time.perf_counter()
    summaries = {}

    for proc in processes:
        summaries[proc] = run_process_training(
            process_name=proc,
            total_steps=args.steps,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            log_interval=args.log_interval,
            device=args.device,
        )

    total_dt = time.perf_counter() - total_start
    print("\n" + "=" * 70)
    print(f"ALL SEQUENTIAL RUNS COMPLETED in {total_dt:.1f}s ({total_dt/3600.0:.2f} hours)")
    for proc, s in summaries.items():
        print(f"  {proc}: {s['total_time_hours']:.2f}h, {s['average_steps_per_second']:.1f} steps/s, {s['peak_memory_mb']:.1f} MB peak, final val norm: {s['final_val_loss_normalized']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
