"""
Script to train the Robotic World Model (RWM) with autoregressive training.

Usage:
    python scripts/train_world_model.py --config configs/rwm_anymal.yaml --data_path /path/to/data

The script:
  1. Loads trajectory data
  2. Creates sliding window dataset
  3. Trains RWM with autoregressive training (Section 3.2)
  4. Evaluates autoregressive prediction accuracy
  5. Saves model checkpoints
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import RoboticWorldModel, MLPWorldModel, RSSMWorldModel, TransformerWorldModel
from training import WorldModelTrainer, TeacherForcingTrainer
from utils import TrajectoryDataset, create_dataloaders, generate_synthetic_trajectories


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(config: dict, device: torch.device):
    """Build world model from config."""
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "rwm")

    if model_type == "rwm":
        model = RoboticWorldModel(
            obs_size=model_cfg["obs_size"],
            action_size=model_cfg["action_size"],
            priv_size=model_cfg.get("priv_size", 0),
            hidden_size=model_cfg.get("hidden_size", 256),
            num_gru_layers=model_cfg.get("num_gru_layers", 2),
            head_hidden_size=model_cfg.get("head_hidden_size", 128),
        )
    elif model_type == "mlp":
        model = MLPWorldModel(
            obs_size=model_cfg["obs_size"],
            action_size=model_cfg["action_size"],
            priv_size=model_cfg.get("priv_size", 0),
            history_horizon=config["training"]["history_horizon"],
            hidden_size=model_cfg.get("hidden_size", 256),
        )
    elif model_type == "rssm":
        model = RSSMWorldModel(
            obs_size=model_cfg["obs_size"],
            action_size=model_cfg["action_size"],
            priv_size=model_cfg.get("priv_size", 0),
            hidden_size=model_cfg.get("hidden_size", 256),
            latent_dim=model_cfg.get("latent_dim", 64),
            num_categories=model_cfg.get("num_categories", 32),
        )
    elif model_type == "transformer":
        model = TransformerWorldModel(
            obs_size=model_cfg["obs_size"],
            action_size=model_cfg["action_size"],
            priv_size=model_cfg.get("priv_size", 0),
            d_model=model_cfg.get("d_model", 64),
            nhead=model_cfg.get("nhead", 8),
            num_layers=model_cfg.get("num_layers", 2),
            context_length=model_cfg.get("context_length", 32),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def load_data(data_path: str, config: dict):
    """Load trajectory data from disk or generate synthetic data."""
    if data_path is None or not os.path.exists(data_path):
        print("No data path provided or path does not exist. Using synthetic data for testing.")
        obs_size = config["model"]["obs_size"]
        action_size = config["model"]["action_size"]
        priv_size = config["model"].get("priv_size", 0)

        observations, actions, privileged_info = generate_synthetic_trajectories(
            n_trajectories=50,
            trajectory_length=500,
            obs_size=obs_size,
            action_size=action_size,
            priv_size=priv_size,
        )
        return observations, actions, privileged_info

    # Load from numpy files
    data = np.load(data_path, allow_pickle=True)
    observations = list(data["observations"])
    actions = list(data["actions"])
    privileged_info = list(data.get("privileged_info", [None] * len(observations)))

    return observations, actions, privileged_info


def train(args):
    config = load_config(args.config)
    train_cfg = config["training"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    observations, actions, privileged_info = load_data(args.data_path, config)
    print(f"Loaded {len(observations)} trajectories")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        observations=observations,
        actions=actions,
        history_horizon=train_cfg["history_horizon"],
        forecast_horizon=train_cfg["forecast_horizon"],
        privileged_info=privileged_info,
        batch_size=train_cfg["batch_size"],
        val_split=0.1,
        num_workers=args.num_workers,
        seed=seed,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    model = build_model(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    # Trainer
    use_teacher_forcing = args.teacher_forcing
    if use_teacher_forcing:
        trainer = TeacherForcingTrainer(
            model=model,
            optimizer=optimizer,
            history_horizon=train_cfg["history_horizon"],
            forecast_horizon=train_cfg["forecast_horizon"],
            forecast_decay=train_cfg.get("forecast_decay", 1.0),
            device=device,
        )
        print("Training with teacher forcing (RWM-TF)")
    else:
        trainer = WorldModelTrainer(
            model=model,
            optimizer=optimizer,
            history_horizon=train_cfg["history_horizon"],
            forecast_horizon=train_cfg["forecast_horizon"],
            forecast_decay=train_cfg.get("forecast_decay", 1.0),
            device=device,
        )
        print("Training with autoregressive training (RWM-AR)")

    # Output directory
    output_dir = Path(args.output_dir) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    for iteration in range(train_cfg["max_iterations"]):
        # Train epoch
        train_metrics = trainer.train_epoch(train_loader)

        # Validate periodically
        if (iteration + 1) % args.eval_interval == 0:
            val_metrics = trainer.evaluate(val_loader, eval_horizon=100)

            print(
                f"Iter {iteration + 1}/{train_cfg['max_iterations']} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Error: {val_metrics['mean_relative_error']:.4f}"
            )

            # Save best model
            if train_metrics["loss"] < best_val_loss:
                best_val_loss = train_metrics["loss"]
                torch.save(
                    {
                        "iteration": iteration,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "metrics": {**train_metrics, **val_metrics},
                    },
                    output_dir / "best_model.pt",
                )

        # Save checkpoint periodically
        if (iteration + 1) % args.save_interval == 0:
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                output_dir / f"checkpoint_{iteration + 1}.pt",
            )

    print(f"Training complete. Best loss: {best_val_loss:.4f}")
    print(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train RWM world model")
    parser.add_argument("--config", type=str, default="configs/rwm_anymal.yaml",
                        help="Path to config file")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to trajectory data (.npz file)")
    parser.add_argument("--output_dir", type=str, default="outputs/world_model",
                        help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--teacher_forcing", action="store_true",
                        help="Use teacher forcing instead of autoregressive training")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="Evaluation interval (iterations)")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Checkpoint save interval (iterations)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
