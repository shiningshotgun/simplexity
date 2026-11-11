"""Training loop, checkpointing, and pilot execution.

Implements Appendix A.6:
"For training, we used the Stochastic Gradient Descent (SGD) optimizer with a
batch size of 64, running for 1,000,000 epochs [training iterations], and a
learning rate of 0.01 with no weight decay. For each batch we generated 64 sequences
from the Mess3 or RRXOR HMM, choosing an initial hidden state from the stationary
distribution."
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from simplexity.entropy import compute_optimal_finite_context_loss
from simplexity.model import (
    SimplexityModelConfig,
    SimplexityTransformer,
    build_hooked_transformer,
    count_parameters,
)
from simplexity.processes import HMMProcess, get_mess3_process, get_rrxor_process


@dataclass
class TrainingConfig:
    process_name: str  # "Mess3" or "RRXOR"
    d_vocab: int
    n_ctx: int = 10
    d_model: int = 64
    d_head: int = 8
    n_heads: int = 1
    n_layers: int = 4
    d_mlp: int = 256
    act_fn: str = "relu"
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    momentum: float = 0.0
    total_steps: int = 1_000_000
    checkpoint_steps: Tuple[int, ...] = (
        100,
        200,
        500,
        1_000,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
        100_000,
        200_000,
        500_000,
        1_000_000,
    )
    val_interval: int = 5_000
    val_batches: int = 20  # 20 * 64 = 1,280 validation sequences
    seed: int = 0
    device: str = "cpu"
    prefer_hooked: bool = True


@dataclass
class PilotResult:
    pilot_steps: int
    elapsed_seconds: float
    steps_per_second: float
    estimated_total_seconds: float
    estimated_total_hours: float
    peak_memory_bytes: int
    peak_memory_mb: float
    parameter_count: Dict[str, int]
    model_type: str


def get_checkpoint_schedule(total_steps: int) -> List[int]:
    """Geometric checkpoint schedule from 100 to total_steps."""
    bases = [1, 2, 5]
    schedule = []
    p = 2
    while True:
        for b in bases:
            val = b * (10**p)
            if val <= total_steps:
                schedule.append(val)
            else:
                break
        if 10 ** (p + 1) > total_steps:
            break
        p += 1
    if total_steps not in schedule:
        schedule.append(total_steps)
    return sorted(list(set(schedule)))


def build_model_for_training(
    cfg: SimplexityModelConfig,
    device: str = "cpu",
    prefer_hooked: bool = True,
) -> Tuple[nn.Module, str]:
    """Build model, preferring HookedTransformer if available."""
    if prefer_hooked:
        try:
            model = build_hooked_transformer(cfg, device=device)
            return model, "HookedTransformer"
        except ImportError:
            pass
    model = SimplexityTransformer(cfg).to(device)
    return model, "SimplexityTransformer"


def compute_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss over the 9 autoregressive prediction positions.

    Args:
        logits: (batch, seq_len, d_vocab)
        targets: (batch, seq_len)

    Predictions at positions 0..8 target tokens at positions 1..9.
    """
    pred_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    target_tokens = targets[:, 1:].reshape(-1)
    return nn.functional.cross_entropy(pred_logits, target_tokens)


def evaluate_model(
    model: nn.Module,
    process: HMMProcess,
    num_batches: int = 20,
    batch_size: int = 64,
    seq_len: int = 10,
    device: str = "cpu",
    rng: np.random.Generator | None = None,
) -> float:
    """Compute average validation cross-entropy on fresh sequences."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            seqs, _ = process.sample_sequences(batch_size, seq_len, rng=rng)
            tokens = torch.from_numpy(seqs).to(device)
            logits = model(tokens)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = compute_cross_entropy_loss(logits, tokens)
            total_loss += float(loss.item())
    return total_loss / num_batches


def run_pilot(
    process_name: str = "Mess3",
    pilot_steps: int = 1000,
    total_target_steps: int = 1_000_000,
    device: str = "cpu",
    seed: int = 0,
    prefer_hooked: bool = True,
) -> PilotResult:
    """Run a measured 1,000-iteration pilot to measure throughput and memory."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if process_name == "Mess3":
        process = get_mess3_process()
        d_vocab = 3
    elif process_name == "RRXOR":
        process = get_rrxor_process()
        d_vocab = 2
    else:
        raise ValueError(f"Unknown process: {process_name}")

    cfg = SimplexityModelConfig(d_vocab=d_vocab, seed=seed)
    model, model_type = build_model_for_training(cfg, device=device, prefer_hooked=prefer_hooked)
    params = count_parameters(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)

    # Warm-up 10 steps
    for _ in range(10):
        seqs, _ = process.sample_sequences(64, 10, rng=rng)
        tokens = torch.from_numpy(seqs).to(device)
        logits = model(tokens)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = compute_cross_entropy_loss(logits, tokens)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()

    start_time = time.perf_counter()
    for _ in range(pilot_steps):
        seqs, _ = process.sample_sequences(64, 10, rng=rng)
        tokens = torch.from_numpy(seqs).to(device)
        logits = model(tokens)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = compute_cross_entropy_loss(logits, tokens)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start_time
    steps_per_sec = pilot_steps / elapsed
    est_total_sec = total_target_steps / steps_per_sec
    est_total_hours = est_total_sec / 3600.0

    peak_bytes = 0
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_bytes = ru.ru_maxrss if os.uname().sysname == "Darwin" else (ru.ru_maxrss * 1024)
    peak_mb = peak_bytes / (1024.0 * 1024.0)

    return PilotResult(
        pilot_steps=pilot_steps,
        elapsed_seconds=elapsed,
        steps_per_second=steps_per_sec,
        estimated_total_seconds=est_total_sec,
        estimated_total_hours=est_total_hours,
        peak_memory_bytes=peak_bytes,
        peak_memory_mb=peak_mb,
        parameter_count=params,
        model_type=model_type,
    )


def train_model(
    config: TrainingConfig,
    checkpoint_dir: Path | str,
    output_metrics_path: Path | str,
    device: str = "cpu",
    log_interval: int = 1000,
) -> Dict[str, Any]:
    """Execute full training run with geometric checkpointing and validation."""
    if device == "cpu":
        torch.set_num_threads(1)

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    val_rng = np.random.default_rng(config.seed + 1000)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_metrics_path = Path(output_metrics_path)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if config.process_name == "Mess3":
        process = get_mess3_process()
    elif config.process_name == "RRXOR":
        process = get_rrxor_process()
    else:
        raise ValueError(f"Unknown process: {config.process_name}")

    # Optimal loss constant for normalization
    optimal_loss_info = compute_optimal_finite_context_loss(process, num_trained_positions=9)
    optimal_loss = optimal_loss_info.average_optimal_loss

    model_cfg = SimplexityModelConfig(
        d_vocab=config.d_vocab,
        n_ctx=config.n_ctx,
        d_model=config.d_model,
        d_head=config.d_head,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_mlp=config.d_mlp,
        act_fn=config.act_fn,
        seed=config.seed,
    )
    model, model_type = build_model_for_training(
        model_cfg, device=device, prefer_hooked=config.prefer_hooked
    )
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    checkpoint_steps_set = set(config.checkpoint_steps)
    if config.total_steps not in checkpoint_steps_set:
        checkpoint_steps_set.add(config.total_steps)

    start_step = 1
    metrics: List[Dict[str, Any]] = []

    # Check if existing checkpoints and metrics can be resumed
    latest_step = 0
    for ckpt_file in checkpoint_dir.glob("step_*.pt"):
        try:
            s = int(ckpt_file.stem.split("_")[1])
            if s > latest_step:
                latest_step = s
        except (ValueError, IndexError):
            pass

    if latest_step > 0 and latest_step < config.total_steps:
        latest_ckpt_path = checkpoint_dir / f"step_{latest_step}.pt"
        ckpt_data = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"])
        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        if "torch_rng_state" in ckpt_data:
            torch.set_rng_state(ckpt_data["torch_rng_state"])
        if "numpy_rng_state" in ckpt_data:
            rng.bit_generator.state = ckpt_data["numpy_rng_state"]
        if "val_rng_state" in ckpt_data:
            val_rng.bit_generator.state = ckpt_data["val_rng_state"]
        start_step = latest_step + 1
        print(f"[{config.process_name}] Resumed from checkpoint step {latest_step}")
        if output_metrics_path.exists():
            with open(output_metrics_path, "r") as f:
                metrics = json.load(f)

    print(
        f"[{config.process_name}] Training {model_type} for {config.total_steps:,} steps on {device} "
        f"(optimal loss: {optimal_loss:.6f} nats)..."
    )

    t_start = time.perf_counter()
    step_loss_accum = 0.0

    for step in range(start_step, config.total_steps + 1):
        model.train()
        seqs, _ = process.sample_sequences(config.batch_size, config.n_ctx, rng=rng)
        tokens = torch.from_numpy(seqs).to(device)

        logits = model(tokens)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = compute_cross_entropy_loss(logits, tokens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        step_loss_accum += loss_val

        # Check if validation / checkpoint needed
        is_checkpoint_step = step in checkpoint_steps_set
        is_val_step = (step % config.val_interval == 0) or is_checkpoint_step

        if is_val_step:
            val_loss = evaluate_model(
                model=model,
                process=process,
                num_batches=config.val_batches,
                batch_size=config.batch_size,
                seq_len=config.n_ctx,
                device=device,
                rng=val_rng,
            )
            val_loss_norm = val_loss / optimal_loss
            elapsed = time.perf_counter() - t_start
            sps = (step - start_step + 1) / elapsed if elapsed > 0 else 0.0

            metric_entry = {
                "step": step,
                "train_loss": loss_val,
                "val_loss": val_loss,
                "val_loss_normalized": val_loss_norm,
                "optimal_loss": optimal_loss,
                "elapsed_seconds": elapsed,
                "steps_per_second": sps,
            }
            metrics.append(metric_entry)

            # Save metrics progressively
            with open(output_metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            if is_checkpoint_step:
                ckpt_save_path = checkpoint_dir / f"step_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_loss_normalized": val_loss_norm,
                        "config": asdict(config),
                        "torch_rng_state": torch.get_rng_state(),
                        "numpy_rng_state": rng.bit_generator.state,
                        "val_rng_state": val_rng.bit_generator.state,
                    },
                    ckpt_save_path,
                )
                print(
                    f"[{config.process_name}] Checkpoint saved: {ckpt_save_path.name} | "
                    f"Val Loss: {val_loss:.5f} (Norm: {val_loss_norm:.4f}) | {sps:.1f} steps/s"
                )

        if step % log_interval == 0 and not is_checkpoint_step:
            avg_train_loss = step_loss_accum / log_interval
            step_loss_accum = 0.0
            elapsed = time.perf_counter() - t_start
            sps = (step - start_step + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"[{config.process_name}] Step {step:,}/{config.total_steps:,} | "
                f"Train Loss: {avg_train_loss:.5f} | {sps:.1f} steps/s"
            )

    total_time = time.perf_counter() - t_start
    final_val_loss = evaluate_model(
        model=model,
        process=process,
        num_batches=50,
        batch_size=config.batch_size,
        seq_len=config.n_ctx,
        device=device,
        rng=val_rng,
    )
    final_val_loss_norm = final_val_loss / optimal_loss

    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_bytes = ru.ru_maxrss if os.uname().sysname == "Darwin" else (ru.ru_maxrss * 1024)
    peak_mb = peak_bytes / (1024.0 * 1024.0)
    avg_sps = config.total_steps / total_time if total_time > 0 else 0.0

    summary = {
        "process_name": config.process_name,
        "model_type": model_type,
        "total_steps": config.total_steps,
        "total_time_seconds": total_time,
        "total_time_hours": total_time / 3600.0,
        "average_steps_per_second": avg_sps,
        "peak_memory_mb": peak_mb,
        "final_val_loss": final_val_loss,
        "final_val_loss_normalized": final_val_loss_norm,
        "optimal_loss": optimal_loss,
        "metrics_file": str(output_metrics_path),
        "checkpoints_saved": sorted([f.name for f in checkpoint_dir.glob("step_*.pt")]),
    }
    with open(checkpoint_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
