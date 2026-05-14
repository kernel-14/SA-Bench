# main.py
"""
Entry point for training and evaluating consistency models with
generator‑augmented flows (GC) and baselines (IC, batch‑OT).

Usage:
    python main.py --config configs/ict_gc.yaml

This script reads a YAML configuration, sets up the data pipeline, model,
schedules, coupling, optimizer, and trainer. After training, it evaluates the
EMA model using FID, KID, and IS, printing the results to the console.
"""

import argparse
import yaml
import torch
import numpy as np
import random
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Project imports (all modules are expected to be in the same directory)
# -----------------------------------------------------------------------------
from data import DataModule
from model import ConsistencyModel
from schedules import Schedules
from coupling import Coupling
from trainer import ConsistencyTrainer
from evaluator import Evaluator

# Optional external optimizer (used in the paper)
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False
    import warnings
    warnings.warn("Lion optimizer not installed. Using AdamW as fallback.", RuntimeWarning)


class Main:
    """
    Orchestrator that reads experiment configuration, instantiates all
    components, runs training, and performs final evaluation.
    """

    def __init__(self, config_path: str) -> None:
        """
        Loads the YAML configuration from the given path.

        Args:
            config_path: Path to a YAML file following the schema of
                ``config.yaml``.
        """
        self.config = self.load_config(config_path)
        self.experiment = self.config["experiment"]
        self.data_cfg = self.config["dataset"]
        self.model_cfg = self.config["model"]
        self.train_cfg = self.config["training"]
        self.sched_cfg = self.config["schedules"]
        self.eval_cfg = self.config["evaluation"]

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------
    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        """
        Read a YAML configuration file and return it as a nested dictionary.

        Args:
            path: Path to the YAML file.

        Returns:
            dict: The loaded configuration.
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        # Insert any mandatory key if missing (for safety)
        if "evaluation" not in config:
            config["evaluation"] = {}
        return config

    # ------------------------------------------------------------------
    # Main execution pipeline
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute the full experiment: data loading, model creation, training,
        and final evaluation.
        """
        # ------------------------------------------------------------------
        # Misc setup
        # ------------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Reproducibility (optional seed from config, else 42)
        seed = self.config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True

        # ------------------------------------------------------------------
        # 1. Data module
        # ------------------------------------------------------------------
        data_module = DataModule(
            dataset_name=self.data_cfg["name"],
            batch_size=self.data_cfg["batch_size"],
            resolution=self.data_cfg["resolution"],
            data_dir=self.data_cfg.get("train_data_dir", "./data"),
        )
        train_loader = data_module.train_dataloader()

        # ------------------------------------------------------------------
        # 2. Model and EMA model
        # ------------------------------------------------------------------
        model = ConsistencyModel(
            img_channels=self.model_cfg["img_channels"],
            model_channels=self.model_cfg["model_channels"],
            num_blocks=self.model_cfg["num_blocks"],
            channel_mult=self.model_cfg["channel_mult"],
            attn_resolutions=self.model_cfg["attn_resolutions"],
            dropout=self.model_cfg["dropout"],
            sigma_data=self.model_cfg.get("sigma_data", 0.5),
        ).to(device)

        # EMA model is a separate instance with the same architecture
        ema_model = ConsistencyModel(
            img_channels=self.model_cfg["img_channels"],
            model_channels=self.model_cfg["model_channels"],
            num_blocks=self.model_cfg["num_blocks"],
            channel_mult=self.model_cfg["channel_mult"],
            attn_resolutions=self.model_cfg["attn_resolutions"],
            dropout=self.model_cfg["dropout"],
            sigma_data=self.model_cfg.get("sigma_data", 0.5),
        ).to(device)
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------
        # 3. Schedules
        # ------------------------------------------------------------------
        schedules = Schedules(
            sigma_min=self.sched_cfg["sigma_min"],
            sigma_max=self.sched_cfg["sigma_max"],
            rho=self.sched_cfg["rho"],
            s0=self.sched_cfg["s0"],
            s1=self.sched_cfg["s1"],
            total_steps=self.train_cfg["total_steps"],
        )

        # ------------------------------------------------------------------
        # 4. Coupling module (IC, OT, or GC)
        # ------------------------------------------------------------------
        coupling_type = self.experiment["coupling_type"]
        if coupling_type == "gc":
            use_ema_pred = self.experiment.get("use_ema_predictor", True)
            predictor = ema_model if use_ema_pred else model
        else:
            predictor = None  # IC / OT do not require a predictor

        coupling = Coupling(
            type=coupling_type,
            model=predictor,
            mu=self.experiment.get("mu", 0.0),
        )

        # ------------------------------------------------------------------
        # 5. Optimizer
        # ------------------------------------------------------------------
        lr = self.train_cfg["learning_rate"]
        optimizer_type = self.train_cfg.get("optimizer", "lion").lower()
        if optimizer_type == "lion":
            if LION_AVAILABLE:
                optimizer = Lion(
                    model.parameters(),
                    lr=lr,
                    betas=(0.9, 0.99),
                    weight_decay=0.0,
                )
            else:
                print("Lion not available; falling back to AdamW.")
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
        else:
            # Default to AdamW if something else is specified
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))

        # ------------------------------------------------------------------
        # 6. Trainer
        # ------------------------------------------------------------------
        # Pseudo‑Huber constant: 0.00054 * sqrt(d) where d = C*H*W
        c_default = (
            self.train_cfg.get("pseudo_huber_c", None)
            or self._compute_pseudo_huber_c(data_module.get_data_shape())
        )
        trainer = ConsistencyTrainer(
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            total_steps=self.train_cfg["total_steps"],
            schedules=schedules,
            coupling=coupling,
            data_loader=train_loader,
            ema_decay=self.train_cfg.get("ema_decay", 0.9999),
            pseudo_huber_c=c_default,
            eval_every=self.train_cfg.get("eval_every_steps", 10000),
            save_every=self.eval_cfg.get("save_checkpoint_every", 10000),
            checkpoint_dir=self.eval_cfg.get("checkpoint_dir", "./checkpoints"),
            gradient_clip_norm=self.train_cfg.get("gradient_clip_norm", None),
        )

        # ------------------------------------------------------------------
        # 7. Execute training
        # ------------------------------------------------------------------
        print("Starting training...")
        trainer.train()

        # ------------------------------------------------------------------
        # 8. Final evaluation
        # ------------------------------------------------------------------
        print("\nEvaluating final EMA model...")
        evaluator = Evaluator(
            model=ema_model,
            data_module=data_module,
            num_samples=self.eval_cfg.get("num_samples", 50000),
            sigma_max=self.sched_cfg["sigma_max"],
            device=device,
        )

        # Compute and print metrics
        metrics_to_compute = self.eval_cfg.get("metrics", ["fid", "kid", "is"])
        results = {}
        if "fid" in metrics_to_compute:
            fid = evaluator.compute_fid()
            results["FID"] = fid
            print(f"FID: {fid:.4f}")
        if "kid" in metrics_to_compute:
            kid = evaluator.compute_kid()
            results["KID (x100)"] = kid
            print(f"KID (x100): {kid:.4f}")
        if "is" in metrics_to_compute:
            is_score = evaluator.compute_is()
            results["IS"] = is_score
            print(f"IS: {is_score:.4f}")

        # Optionally save results to a text file
        results_path = self.eval_cfg.get("results_file", None)
        if results_path is not None:
            with open(results_path, "w") as f:
                for k, v in results.items():
                    f.write(f"{k}: {v:.4f}\n")
            print(f"Results saved to {results_path}")

        print("Training and evaluation completed.")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_pseudo_huber_c(data_shape) -> float:
        """
        Compute the default pseudo‑Huber constant as in the paper:
        c = 0.00054 * sqrt(C * H * W).
        """
        C, H, W = data_shape
        d = C * H * W
        return 0.00054 * (d ** 0.5)


# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a consistency model with generator-augmented flows."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    main = Main(args.config)
    main.run()
