## main.py

"""
Main entry point for reproducing experiments from
"Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions".

Orchestrates training and evaluation of Masked Diffusion Models (MDMs) and
Autoregressive Models (ARMs) across multiple tasks:

- Scaling laws for π‑learners
- L&O‑NAE‑SAT distribution
- Sudoku and Zebra logic puzzles
- LLaDA‑8B inference (placeholder)

Usage:
    python main.py --config config.yaml --task nae_sat_experiment --do_train --do_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Local project imports – following the design defined in the reproduction plan.
from configs import ExperimentConfig, ModelConfig, TrainingConfig
from datasets import (
    NAESATDataset,
    SudokuDataset,
    TextDataset,
    ZebraDataset,
)
from evaluation import Evaluator
from mdm_trainer import MDMTrainer
from model import ARMWrapper, MDMTransformer, Model
from samplers import (
    Sampler,
    TopMarginSampler,
    TopProbSampler,
    VanillaSampler,
)
from utils import MASK_TOKEN_ID, PAD_TOKEN_ID


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Helpers: seed, device, model overrides
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Reproducibility: fix random seeds for Python, NumPy and PyTorch."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make cuDNN deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_model_overrides(config: ExperimentConfig, overrides_str: Optional[str]) -> ExperimentConfig:
    """
    Parse JSON‑string model overrides (e.g. '{"hidden_size":384,"num_layers":6}')
    and return a new ExperimentConfig with those model parameters updated.
    """
    if overrides_str is None:
        return config

    try:
        overrides = json.loads(overrides_str)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON for model overrides: {overrides_str}")

    # Build a new ModelConfig with the overriden values
    new_model_kwargs = {
        "vocab_size": config.model.vocab_size,
        "max_seq_length": config.model.max_seq_length,
        "hidden_size": config.model.hidden_size,
        "num_layers": config.model.num_layers,
        "num_attention_heads": config.model.num_attention_heads,
        "intermediate_size": config.model.intermediate_size,
        "hidden_dropout_prob": config.model.hidden_dropout_prob,
        "attention_probs_dropout_prob": config.model.attention_probs_dropout_prob,
        "positional_embedding_type": config.model.positional_embedding_type,
        "use_pretrained": config.model.use_pretrained,
        "pretrained_model_name": config.model.pretrained_model_name,
    }
    # Add model_type if present (we inject it here even if not in original config)
    model_type = getattr(config.model, "model_type", "mdm")
    new_model_kwargs["model_type"] = model_type

    new_model_kwargs.update(overrides)

    new_model = ModelConfig(**new_model_kwargs)

    # Create a new ExperimentConfig using replace (dataclasses.replace)
    import dataclasses
    return dataclasses.replace(config, model=new_model)


def ensure_model_type(config: ExperimentConfig) -> ExperimentConfig:
    """
    If the loaded configuration's ModelConfig does not have a ``model_type``
    attribute, add it with a sensible default based on the task.
    """
    if not hasattr(config.model, "model_type"):
        # Provide a default; task‑specific overrides are handled in from_yaml
        # but the YAML may not contain it.
        default_type = "mdm" if config.task in ("nae_sat", "sudoku", "zebra", "llada") else "arm"
        new_model = ModelConfig(
            vocab_size=config.model.vocab_size,
            max_seq_length=config.model.max_seq_length,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            num_attention_heads=config.model.num_attention_heads,
            intermediate_size=config.model.intermediate_size,
            hidden_dropout_prob=config.model.hidden_dropout_prob,
            attention_probs_dropout_prob=config.model.attention_probs_dropout_prob,
            positional_embedding_type=config.model.positional_embedding_type,
            use_pretrained=config.model.use_pretrained,
            pretrained_model_name=config.model.pretrained_model_name,
            model_type=default_type,
        )
        import dataclasses
        config = dataclasses.replace(config, model=new_model)
    return config


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(config: ExperimentConfig) -> Model:
    """
    Instantiate the correct model variant (MDMTransformer or ARMWrapper)
    according to ``config.model.model_type``.
    """
    model_type = config.model.model_type
    if config.model.use_pretrained:
        # Pretrained MDM (e.g., LLaDA‑8B placeholder)
        if config.model.pretrained_model_name == "llada-8b":
            # Placeholder: in a real setup we would load from a checkpoint.
            # Here we raise an error because the official weights are not provided.
            raise NotImplementedError(
                "LLaDA‑8B weights are not publicly available at this time. "
                "Please provide a local checkpoint or use a smaller model."
            )
        else:
            # Generic pretrained load (not implemented)
            raise NotImplementedError(
                "Pretrained model loading from arbitrary name is not implemented."
            )

    model_cfg = config.model
    common_kwargs = dict(
        vocab_size=model_cfg.vocab_size,
        max_seq_length=model_cfg.max_seq_length,
        hidden_size=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_attention_heads,
        intermediate_size=model_cfg.intermediate_size,
        dropout=model_cfg.hidden_dropout_prob,
    )

    if model_type == "mdm":
        return MDMTransformer(config)
    elif model_type == "arm":
        return ARMWrapper(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# Dataset & DataLoader factory
# ---------------------------------------------------------------------------

def create_dataset_and_loader(
    config: ExperimentConfig,
    split: str,
    batch_size: Optional[int] = None,
    shuffle: bool = False,
) -> DataLoader:
    """
    Return a DataLoader for the requested split and task.
    """
    batch_size = batch_size or config.training.batch_size
    task = config.task

    if task == "scaling":
        # π‑learner experiments use TextDataset with a fixed permutation
        dataset = TextDataset(config, split=split)
    elif task == "nae_sat":
        dataset = NAESATDataset(config)
    elif task == "sudoku":
        dataset = SudokuDataset(config, split=split)
    elif task == "zebra":
        dataset = ZebraDataset(config, split=split)
    else:
        raise ValueError(f"Unsupported task '{task}' for dataset creation.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=True if split == "train" else False,
    )
    return loader


# ---------------------------------------------------------------------------
# Custom ARM Trainer (used for scaling‑law and ARM baselines)
# ---------------------------------------------------------------------------

class ARMTrainer:
    """
    Lightweight trainer for Autoregressive Models (ARMWrapper).

    Implements the standard left‑to‑right next‑token prediction loss.
    Supports the same training loop infrastructure as MDMTrainer, including
    gradient accumulation, cosine LR scheduling, and checkpointing.
    """

    def __init__(
        self,
        model: ARMWrapper,
        config: ExperimentConfig,
        train_loader: DataLoader,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader

        self.device = torch.device(config.device)
        self.model.to(self.device)

        self._batch_size = config.training.batch_size
        self._grad_accum_steps = config.training.gradient_accumulation_steps
        self._log_interval = config.training.log_interval
        self._save_interval = config.training.save_interval
        self._checkpoint_dir = Path(config.training.checkpoint_dir)

        self.total_steps = self._compute_total_steps()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            weight_decay=config.training.weight_decay,
        )

        self.scheduler = self._create_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.device == "cuda"))
        self._mixed_precision = config.device == "cuda"
        self.use_wandb = config.use_wandb

        if self.use_wandb:
            import wandb
            wandb.init(project=config.wandb_project, config=config.__dict__)

    def _compute_total_steps(self) -> int:
        if self.config.training.num_iterations is not None:
            return self.config.training.num_iterations
        elif self.config.training.epochs is not None:
            steps_per_epoch = len(self.train_loader)
            return self.config.training.epochs * steps_per_epoch // self._grad_accum_steps
        else:
            raise ValueError("Either num_iterations or epochs must be set.")

    def _create_scheduler(self) -> LambdaLR:
        warmup_steps = self.config.training.warmup_steps
        total_steps = self.total_steps
        initial_lr = self.config.training.learning_rate
        min_lr = self.config.training.min_learning_rate

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr + (initial_lr - min_lr) * cosine) / initial_lr

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self) -> None:
        """Run the full training loop."""
        self.model.train()
        global_step = 0
        progress = tqdm(total=self.total_steps, desc="ARM Training")

        loader_iter = iter(self.train_loader)

        while global_step < self.total_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch = next(loader_iter)

            x = batch["input_ids"].to(self.device)   # (B, L)
            labels = batch.get("labels", x)

            # Standard autoregressive cross‑entropy loss
            with torch.cuda.amp.autocast(enabled=self._mixed_precision):
                logits = self.model.get_logits(x)      # (B, L, V)
                # Shift so that position i predicts token i+1
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                # Model outputs over real tokens (1..vocab_size-1); shift labels
                shift_labels = shift_labels - 1
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-1,      # ignore mask/pad if present
                )

            loss = loss / self._grad_accum_steps

            if self._mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % self._grad_accum_steps == 0:
                if self._mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            global_step += 1
            progress.update(1)

            if global_step % self._log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                if self.use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": global_step})
                else:
                    logger.info(f"Step {global_step}/{self.total_steps} "
                                f"| Loss: {loss.item():.4f} | LR: {lr:.2e}")

            if global_step % self._save_interval == 0 or global_step == self.total_steps:
                self.save_checkpoint(global_step)

        progress.close()
        self.save_checkpoint(self.total_steps, final=True)

    def save_checkpoint(self, step: int, final: bool = False) -> None:
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": step,
            "config": self.config,
        }
        path = self._checkpoint_dir / ("final.pt" if final else f"checkpoint_{step}.pt")
        torch.save(checkpoint, path)
        if self.use_wandb:
            wandb.save(str(path))

    def load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["step"]


# ---------------------------------------------------------------------------
# Training & Evaluation dispatch
# ---------------------------------------------------------------------------

def run_train(
    model: Model,
    train_loader: DataLoader,
    config: ExperimentConfig,
    resume_from: Optional[str] = None,
) -> None:
    """
    Run training using the appropriate trainer for the model type.
    """
    if isinstance(model, MDMTransformer):
        trainer = MDMTrainer(model, config, train_loader)
    elif isinstance(model, ARMWrapper):
        trainer = ARMTrainer(model, config, train_loader)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    if resume_from is not None:
        step = trainer.load_checkpoint(resume_from)
        logger.info(f"Resumed from checkpoint at step {step}")

    trainer.train()


def run_eval(
    model: Model,
    eval_loader: DataLoader,
    sampler: Sampler,
    config: ExperimentConfig,
    proxy_model: Optional[Model] = None,
) -> Dict[str, Any]:
    """
    Run evaluation according to the task defined in ``config.task``.
    Returns a dictionary of metrics.
    """
    evaluator = Evaluator(model, eval_loader, sampler, config)

    task = config.task
    results = {}

    if task == "scaling":
        # For scaling‑law experiments, the main metric is perplexity.
        # (Note: this uses the π‑learner's evaluation method)
        results["perplexity"] = evaluator.evaluate_perplexity()

    elif task == "nae_sat":
        # Two evaluations: accuracy and (optionally) imbalance analysis.
        # For accuracy we need to run the sampler on fully masked sequences.
        # Evaluate accuracy exactly as in logic puzzle, but with no clue_mask.
        # The evaluator's evaluate_accuracy already handles this.
        results["accuracy"] = evaluator.evaluate_accuracy()

        # Imbalance analysis requires a proxy model.
        if proxy_model is not None:
            # Use the dataset from the eval_loader (must be NAESATDataset)
            dataset = eval_loader.dataset
            # Determine mask configurations: we replicate the paper's setup for ℓ=11
            # but allow command‑line override. For simplicity, use ℓ=11.
            imbalance_masks = [(11, 11 * dataset.P // dataset.N)]
            imbalance = evaluator.imbalance_analysis(proxy_model, dataset, imbalance_masks)
            results["imbalance"] = imbalance

    elif task in ("sudoku", "zebra"):
        results["accuracy"] = evaluator.evaluate_logic_puzzle()

    elif task == "llada":
        # For LLaDA‑8B we compute generative perplexity and entropy.
        gen_ppl, entropy = evaluator.evaluate()  # uses the internal `evaluate` method
        results["generative_perplexity"] = gen_ppl
        results["entropy"] = entropy

    else:
        logger.warning(f"No specific evaluation handling for task '{task}'. Using evaluator.evaluate().")
        results = evaluator.evaluate()

    return results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from 'Train for the Worst, Plan for the Best'."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base YAML configuration file (e.g. config.yaml).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Name of the task‑specific section in the YAML to apply over the defaults. "
             "If omitted, only the 'defaults' section is used.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Run training.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Run evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed (default from config).",
    )
    parser.add_argument(
        "--model_size_override",
        type=str,
        default=None,
        help="JSON string to override model architecture parameters, e.g. "
             "'{\"hidden_size\":384,\"num_layers\":6}'. Useful for scaling‑law sweeps.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results, logs, and checkpoints.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint from which to resume training.",
    )
    parser.add_argument(
        "--proxy_checkpoint",
        type=str,
        default=None,
        help="Path to a proxy (better‑trained) model checkpoint for imbalance analysis (NAE‑SAT).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main procedure
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Load configuration
    # -----------------------------------------------------------------------
    logger.info(f"Loading configuration from {args.config}")
    config = ExperimentConfig.from_yaml(args.config, override_task=args.task)

    # Apply any command‑line overrides to the loaded config.
    if args.seed is not None:
        import dataclasses
        config = dataclasses.replace(config, seed=args.seed)

    # Override model dimensions (e.g., for IsoFLOP sweeps)
    config = apply_model_overrides(config, args.model_size_override)

    # Ensure model_type is present (inject if missing)
    config = ensure_model_type(config)

    # -----------------------------------------------------------------------
    # 2. Setup reproducibility and device
    # -----------------------------------------------------------------------
    set_seed(config.seed)
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # 3. Create model
    # -----------------------------------------------------------------------
    model = create_model(config)
    logger.info(f"Model created: {type(model).__name__}")

    # -----------------------------------------------------------------------
    # 4. Create dataloaders
    # -----------------------------------------------------------------------
    train_loader = None
    eval_loader = None

    if args.do_train:
        train_loader = create_dataset_and_loader(config, split="train", shuffle=True)

    if args.do_eval:
        # For evaluation, we need a loader. Many tasks require a specific split.
        # For NAE‑SAT and scaling, we can use 'val' (or test).
        # We'll default to 'test' for logic puzzles.
        split = "test"
        if config.task in ("scaling", "nae_sat"):
            split = "val"
        eval_loader = create_dataset_and_loader(config, split=split, shuffle=False)

    # -----------------------------------------------------------------------
    # 5. Training (if requested)
    # -----------------------------------------------------------------------
    if args.do_train and train_loader is not None:
        logger.info("Starting training...")
        run_train(model, train_loader, config, resume_from=args.resume_from)
        # Save the final model inside the checkpoint directory
        logger.info("Training finished.")

    # -----------------------------------------------------------------------
    # 6. Evaluation (if requested)
    # -----------------------------------------------------------------------
    if args.do_eval and eval_loader is not None:
        logger.info("Starting evaluation...")

        # Create the sampler for inference
        sampler_type = config.diffusion.adaptive_sampler
        if sampler_type == "vanilla":
            sampler = VanillaSampler(model, config)
        elif sampler_type in ("top_prob", "top_probability"):
            sampler = TopProbSampler(model, config)
        elif sampler_type in ("top_margin", "top_probability_margin"):
            sampler = TopMarginSampler(model, config)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

        # Load proxy model if requested (only used for NAE‑SAT imbalance)
        proxy_model = None
        if args.proxy_checkpoint is not None:
            if config.task != "nae_sat":
                logger.warning("Proxy checkpoint supplied but task is not NAE‑SAT; ignoring.")
            else:
                # Load a separate MDMTransformer for proxy (use the same config)
                proxy_model = MDMTransformer(config)
                ckpt = torch.load(args.proxy_checkpoint, map_location=device)
                proxy_model.load_state_dict(ckpt["model_state_dict"])
                proxy_model.to(device)
                proxy_model.eval()
                logger.info("Proxy model loaded for imbalance analysis.")

        metrics = run_eval(
            model,
            eval_loader,
            sampler,
            config,
            proxy_model=proxy_model,
        )

        # Save metrics to output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        logger.info(f"Metrics: {metrics}")

        # Optionally save generated samples for text tasks? (could be added)

    # -----------------------------------------------------------------------
    # 7. Cleanup (WandB)
    # -----------------------------------------------------------------------
    if config.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
