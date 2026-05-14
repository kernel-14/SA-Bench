"""
World model training loop (Algorithm 1, steps 3-4).

Trains RWM and all baselines using the self-supervised autoregressive
training scheme described in Sec. 3.2.

Training parameters (Table S10):
  max_iterations=2500, lr=1e-4, weight_decay=1e-5, batch_size=1024,
  M=32, N=8, alpha=1.0
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import ExperimentConfig, RWMTrainingConfig
from data import (
    ReplayBuffer,
    Trajectory,
    TrajectoryDataset,
    build_dataloader,
    generate_synthetic_dataset,
)
from model import (
    MLPBaseline,
    RSSMBaseline,
    RWM,
    TransformerBaseline,
    build_mlp_baseline,
    build_rssm_baseline,
    build_rwm,
    build_transformer_baseline,
)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MetricsLogger:
    def __init__(self):
        self._data: Dict[str, list] = {}

    def log(self, metrics: Dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            if k not in self._data:
                self._data[k] = []
            self._data[k].append((step, v))

    def get(self, key: str):
        return self._data.get(key, [])

    def save(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self._data, f)


# ---------------------------------------------------------------------------
# World Model Trainer
# ---------------------------------------------------------------------------

class WorldModelTrainer:
    """
    Trains a world model (RWM or baseline) using autoregressive training.

    Implements the training loop from Algorithm 1 (steps 3-4) and
    the loss from Eq. 2.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: RWMTrainingConfig,
        device: torch.device,
        autoregressive: bool = True,
        output_dir: str = "outputs",
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.autoregressive = autoregressive
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.max_iterations, eta_min=cfg.learning_rate * 0.1
        )
        self.logger = MetricsLogger()
        self.global_step = 0

    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.model.train()
        epoch_metrics: Dict[str, list] = {}

        for obs, actions, privileged in dataloader:
            obs = obs.to(self.device)
            actions = actions.to(self.device)
            privileged = privileged.to(self.device)

            self.optimizer.zero_grad()
            loss, metrics = self.model.compute_loss(
                obs, actions, privileged, autoregressive=self.autoregressive
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)

            self.global_step += 1
            if self.global_step >= self.cfg.max_iterations:
                break

        return {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

    def train(
        self,
        dataset: TrajectoryDataset,
        val_dataset: Optional[TrajectoryDataset] = None,
    ) -> MetricsLogger:
        dataloader = build_dataloader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = build_dataloader(
                val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=0,
            )

        print(f"Training {self.model.__class__.__name__} | "
              f"params={count_parameters(self.model):,} | "
              f"dataset={len(dataset)} | "
              f"autoregressive={self.autoregressive}")

        start_time = time.time()
        epoch = 0

        while self.global_step < self.cfg.max_iterations:
            train_metrics = self.train_epoch(dataloader)
            self.scheduler.step()

            if epoch % self.cfg.log_interval == 0:
                elapsed = time.time() - start_time
                log_str = (
                    f"[{self.global_step}/{self.cfg.max_iterations}] "
                    f"loss={train_metrics.get('loss', 0):.4f} "
                    f"obs_loss={train_metrics.get('obs_loss', 0):.4f} "
                    f"priv_loss={train_metrics.get('priv_loss', 0):.4f} "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e} "
                    f"elapsed={elapsed:.1f}s"
                )
                print(log_str)
                self.logger.log({f"train/{k}": v for k, v in train_metrics.items()}, self.global_step)

            if val_loader is not None and epoch % self.cfg.log_interval == 0:
                val_metrics = self.evaluate(val_loader)
                self.logger.log({f"val/{k}": v for k, v in val_metrics.items()}, self.global_step)
                print(f"  val_loss={val_metrics.get('loss', 0):.4f}")

            if epoch % self.cfg.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{self.global_step:06d}.pt")

            epoch += 1

        self.save_checkpoint("checkpoint_final.pt")
        self.logger.save(os.path.join(self.output_dir, "metrics.json"))
        return self.logger

    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        all_metrics: Dict[str, list] = {}

        with torch.no_grad():
            for obs, actions, privileged in dataloader:
                obs = obs.to(self.device)
                actions = actions.to(self.device)
                privileged = privileged.to(self.device)
                _, metrics = self.model.compute_loss(
                    obs, actions, privileged, autoregressive=True
                )
                for k, v in metrics.items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)

        return {k: float(np.mean(v)) for k, v in all_metrics.items()}

    def save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.cfg.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]


# ---------------------------------------------------------------------------
# Multi-model comparison training (for Fig. 4 experiments)
# ---------------------------------------------------------------------------

def train_all_models(
    cfg: ExperimentConfig,
    dataset: TrajectoryDataset,
    val_dataset: Optional[TrajectoryDataset] = None,
    output_dir: str = "outputs",
) -> Dict[str, MetricsLogger]:
    """
    Train all models (RWM-AR, RWM-TF, MLP, RSSM, Transformer) for comparison.
    Corresponds to the experiments in Sec. 4.3 and Fig. 4.
    """
    device = torch.device(cfg.rwm_training.device if torch.cuda.is_available() else "cpu")
    results = {}

    models_and_modes = [
        ("RWM-AR", build_rwm(cfg), True),
        ("RWM-TF", build_rwm(cfg), False),
        ("MLP", build_mlp_baseline(cfg), True),
        ("RSSM", build_rssm_baseline(cfg), False),
        ("Transformer", build_transformer_baseline(cfg), False),
    ]

    for name, model, autoregressive in models_and_modes:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        model_output_dir = os.path.join(output_dir, name.lower().replace("-", "_"))
        train_cfg = cfg.rwm_training
        # Override checkpoint dir per model
        import dataclasses
        train_cfg_copy = dataclasses.replace(
            train_cfg, checkpoint_dir=os.path.join(model_output_dir, "checkpoints")
        )
        trainer = WorldModelTrainer(
            model=model,
            cfg=train_cfg_copy,
            device=device,
            autoregressive=autoregressive,
            output_dir=model_output_dir,
        )
        logger = trainer.train(dataset, val_dataset)
        results[name] = logger

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train RWM world model")
    parser.add_argument("--robot", type=str, default="anymal_d",
                        choices=["anymal_d", "unitree_g1"])
    parser.add_argument("--model", type=str, default="rwm",
                        choices=["rwm", "mlp", "rssm", "transformer", "all"])
    parser.add_argument("--autoregressive", action="store_true", default=True)
    parser.add_argument("--teacher-forcing", dest="autoregressive", action="store_false")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-iterations", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = ExperimentConfig(robot=args.robot, seed=args.seed)
    cfg.rwm_training.device = args.device
    cfg.rwm_training.max_iterations = args.max_iterations
    cfg.rwm_training.batch_size = args.batch_size
    cfg.rwm_training.learning_rate = args.lr

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    robot = cfg.get_robot_config()

    if args.synthetic:
        print("Generating synthetic dataset...")
        trajectories = generate_synthetic_dataset(
            obs_dim=robot.obs_dim,
            action_dim=robot.action_dim,
            privileged_dim=robot.privileged_dim,
            num_trajectories=20,
            trajectory_length=500,
        )
        n_val = max(1, len(trajectories) // 5)
        train_trajs = trajectories[n_val:]
        val_trajs = trajectories[:n_val]
    else:
        train_trajs = None
        val_trajs = None

    M = cfg.rwm_training.history_horizon
    N = cfg.rwm_training.forecast_horizon

    if train_trajs is not None:
        dataset = TrajectoryDataset(train_trajs, M, N)
        val_dataset = TrajectoryDataset(val_trajs, M, N) if val_trajs else None
    else:
        dataset = TrajectoryDataset.from_directory(args.data_dir, M, N)
        val_dataset = None

    print(f"Dataset size: {len(dataset)} windows")

    if args.model == "all":
        train_all_models(cfg, dataset, val_dataset, args.output_dir)
    else:
        model_map = {
            "rwm": (build_rwm, args.autoregressive),
            "mlp": (build_mlp_baseline, True),
            "rssm": (build_rssm_baseline, False),
            "transformer": (build_transformer_baseline, False),
        }
        build_fn, autoregressive = model_map[args.model]
        model = build_fn(cfg)

        trainer = WorldModelTrainer(
            model=model,
            cfg=cfg.rwm_training,
            device=device,
            autoregressive=autoregressive,
            output_dir=args.output_dir,
        )
        trainer.train(dataset, val_dataset)


if __name__ == "__main__":
    main()
