"""Test faithful resumability of training with full RNG and optimizer state."""

import tempfile
from pathlib import Path
import torch
import numpy as np

from simplexity.train import TrainingConfig, train_model


def test_faithful_checkpoint_resume():
    """Verify that training with checkpoint resume matches uninterrupted training bit-for-bit."""
    # Config for 30 steps, checkpointing at 10 and 20
    cfg = TrainingConfig(
        process_name="Mess3",
        d_vocab=3,
        total_steps=30,
        checkpoint_steps=(10, 20, 30),
        val_interval=10,
        val_batches=2,
        seed=42,
        device="cpu",
        prefer_hooked=False,  # test on reference model for fast execution
    )

    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        dir1 = Path(tmp1)
        dir2 = Path(tmp2)

        # 1. Uninterrupted run for 30 steps
        metrics1_path = dir1 / "metrics.json"
        train_model(cfg, checkpoint_dir=dir1, output_metrics_path=metrics1_path, log_interval=10)

        # 2. Interrupted run: train first to 10 steps
        cfg_part1 = TrainingConfig(
            process_name="Mess3",
            d_vocab=3,
            total_steps=10,
            checkpoint_steps=(10,),
            val_interval=10,
            val_batches=2,
            seed=42,
            device="cpu",
            prefer_hooked=False,
        )
        metrics2_path = dir2 / "metrics.json"
        train_model(cfg_part1, checkpoint_dir=dir2, output_metrics_path=metrics2_path, log_interval=10)

        # Verify step 10 checkpoint has RNG states
        ckpt10 = torch.load(dir2 / "step_10.pt", map_location="cpu")
        assert "torch_rng_state" in ckpt10
        assert "numpy_rng_state" in ckpt10
        assert "val_rng_state" in ckpt10

        # Now resume from step 10 to 30
        train_model(cfg, checkpoint_dir=dir2, output_metrics_path=metrics2_path, log_interval=10)

        # Compare checkpoints at step 20 and step 30
        ckpt_uninterrupted_20 = torch.load(dir1 / "step_20.pt", map_location="cpu")
        ckpt_resumed_20 = torch.load(dir2 / "step_20.pt", map_location="cpu")
        assert np.isclose(ckpt_uninterrupted_20["val_loss"], ckpt_resumed_20["val_loss"], atol=1e-6)

        ckpt_uninterrupted_30 = torch.load(dir1 / "step_30.pt", map_location="cpu")
        ckpt_resumed_30 = torch.load(dir2 / "step_30.pt", map_location="cpu")
        assert np.isclose(ckpt_uninterrupted_30["val_loss"], ckpt_resumed_30["val_loss"], atol=1e-6)

        # Compare model weights exactly
        for k in ckpt_uninterrupted_30["model_state_dict"]:
            t1 = ckpt_uninterrupted_30["model_state_dict"][k]
            t2 = ckpt_resumed_30["model_state_dict"][k]
            if t1.dtype == torch.bool:
                assert (t1 == t2).all().item()
            else:
                diff = (t1 - t2).abs().max()
                assert diff.item() == 0.0, f"Discrepancy in parameter {k}: {diff.item()}"
