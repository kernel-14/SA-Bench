## main.py

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
#  Attempt to import Accelerate (optional but recommended)
# ---------------------------------------------------------------------------
try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    print("accelerate not installed; falling back to single-GPU/CPU training.")

# ---------------------------------------------------------------------------
#  Local module imports
# ---------------------------------------------------------------------------
from models import BaseModels, FineTunedModel
from dataset import PromptDataset
from fine_tuner import AdjointMatchingTrainer
from evaluation import Evaluator


# ---------------------------------------------------------------------------
#  Configuration class (reads config.yaml)
# ---------------------------------------------------------------------------
class Config:
    """
    Lightweight configuration object that reads a YAML file and converts it
    into nested attributes for convenient dot‑notation access.

    Usage:
        config = Config.from_yaml("config.yaml")
        print(config.model.flow_model_checkpoint)
    """

    @staticmethod
    def from_yaml(path: str) -> "Config":
        """
        Load a YAML file and build a Config instance.

        Args:
            path: Filesystem path to the YAML file.

        Returns:
            Config object with all settings as attributes.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return Config(data)

    def __init__(self, data: dict):
        self._data = data
        self._convert_dict_to_attrs(self, data)

    def _convert_dict_to_attrs(self, obj: Any, data: dict) -> None:
        """Recursively set attributes from a dictionary."""
        for key, value in data.items():
            if isinstance(value, dict):
                # Create a sub‑Config that will hold nested settings
                sub_obj = Config.__new__(Config)
                sub_obj._data = value
                self._convert_dict_to_attrs(sub_obj, value)
                setattr(obj, key, sub_obj)
            else:
                setattr(obj, key, value)


# ---------------------------------------------------------------------------
#  Seeding utility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across libraries.

    Args:
        seed: Integer seed.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Encourage deterministic behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # ---------- 1. Command line arguments ----------
    parser = argparse.ArgumentParser(
        description="Reward fine‑tuning of Flow Matching models via Adjoint Matching."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--train_prompts",
        type=str,
        required=True,
        help="Text file containing the full pool of training prompts (one per line).",
    )
    parser.add_argument(
        "--test_prompts",
        type=str,
        required=True,
        help="Text file containing test prompts (one per line).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Base directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # ---------- 2. Load configuration ----------
    config = Config.from_yaml(args.config)
    # Inject paths that may not be in YAML (but are required by the script)
    config.training_prompts_file = args.train_prompts
    config.test_prompts_file = args.test_prompts
    config.output_dir = args.output_dir

    # ---------- 3. Device / accelerator setup ----------
    if HAS_ACCELERATE:
        accelerator = Accelerator(
            split_batches=True,
            mixed_precision="bf16",
        )
    else:
        # Fallback: use a single GPU if available
        accelerator = None
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    # ---------- 4. Reproducibility ----------
    set_seed(args.seed)

    # ---------- 5. Load pre‑trained base models ----------
    base_models = BaseModels(config)
    base_models.freeze_base()

    # Tokenizer is already part of BaseModels; extract for dataset creation
    tokenizer = base_models.tokenizer

    # ---------- 6. Prepare training prompt dataset ----------
    # Read all training prompts, shuffle, and keep 40k for this run.
    with open(args.train_prompts, "r", encoding="utf-8") as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    random.shuffle(all_prompts)
    train_prompts = all_prompts[:40000]

    train_dataset = PromptDataset(train_prompts, tokenizer, config)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size_per_gpu,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # If using Accelerate, wrap the dataloader
    if accelerator is not None:
        train_dataloader = accelerator.prepare(train_dataloader)

    # ---------- 7. Logging (TensorBoard) ----------
    writer = None
    if config.logging.use_tensorboard:
        os.makedirs(config.logging.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=config.logging.log_dir)

    # ---------- 8. Loop over λ values ----------
    for lam in config.fine_tuning.lambda_values:
        print(f"\n====== Fine‑tuning with λ = {lam} ======")

        # Dynamically attach λ to the config so the trainer can read it.
        config.fine_tuning.lambda_ = lam

        # ---------- 8.1 Build fine‑tuned model ----------
        fine_model = FineTunedModel(base_models.flow_model, config)

        # ---------- 8.2 Create trainer ----------
        # The trainer internally uses the accelerator to move models/data and
        # perform gradient scaling.  We pass the accelerator instance if available.
        trainer = AdjointMatchingTrainer(
            base_models=base_models,
            fine_model=fine_model,
            config=config,
            accelerator=accelerator,
        )

        # ---------- 8.3 Train ----------
        trainer.train(train_dataloader, num_epochs=1)

        # ---------- 8.4 Save checkpoint ----------
        ckpt_dir = os.path.join(config.logging.checkpoint_dir, f"lambda_{lam}")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "finetuned_model.pt")
        torch.save(fine_model.unet.state_dict(), ckpt_path)
        print(f"Fine‑tuned model saved to {ckpt_path}")

        # ---------- 8.5 Evaluation ----------
        evaluator = Evaluator(fine_model, base_models, config)

        # Iterate over classifier‑free guidance weights
        for w in config.evaluation.guidance_weights_to_test:
            config.evaluation.classifier_free_guidance_weight = w

            # Evaluate with both ODE (σ=0) and memoryless SDE sampling after fine‑tuning
            for use_sde in [False, True]:
                config.evaluation.sampling_sde = use_sde

                metrics = evaluator.evaluate_all(config.test_prompts_file)

                # Log to TensorBoard
                if writer is not None:
                    for metric_name, value in metrics.items():
                        tag = (
                            f"lambda_{lam}/"
                            f"guidance_{w}/"
                            f"sde_{use_sde}/"
                            f"{metric_name}"
                        )
                        writer.add_scalar(tag, value, global_step=0)

                # Print to terminal
                print(
                    f"  [λ={lam}, guidance={w}, SDE={use_sde}] "
                    + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                )

        # ---------- 8.6 Cleanup for next λ ----------
        del trainer, evaluator, fine_model
        torch.cuda.empty_cache()

    # ---------- 9. Wrap up ----------
    if writer is not None:
        writer.close()
    if accelerator is not None:
        accelerator.wait_for_everyone()

    print("All experiments finished.")


if __name__ == "__main__":
    main()
