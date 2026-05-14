# main.py

"""
NaViL reproduction entry point.

This module implements the ``Main`` class, which serves as the central
orchestrator for the three primary workflows described in the NaViL paper:

1. ``run_design_experiment``   – validates architectural choices (LLM init,
   MoE, visual encoder depth/width) via small‑scale training.
2. ``run_scaling_experiment``  – explores scaling properties of visual
   encoder and LLM independently.
3. ``run_full_training``       – executes the full three‑stage training
   recipe (S1.1 → S1.2 → S2) followed by evaluation on all reported
   benchmarks.

All hyperparameters are drawn from the configuration file (default
``config.yaml``).  Distributed training is handled via DeepSpeed with
automatic process‑group initialisation when launched with ``torchrun``.

Usage examples::

    python main.py --config config.yaml --mode design_experiment
    python main.py --config config.yaml --mode scaling
    python main.py --config config.yaml --mode full_training
"""

from __future__ import annotations

import argparse
import copy
import itertools
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

# Project‑internal imports – note the careful ordering to avoid circular deps.
from config import Config, ModelConfig, TrainingConfig, StageConfig, DataConfig, load_config
from data.dataset import MultiModalDataset
from data.preprocessing import TextTokenizer, ImageTokenizer
from models.navil_model import NaViLModel
from training.trainer import Trainer
from training.stages import apply_freeze_pattern
from evaluation.evaluator import Evaluator
from utils.logging import setup_logging as _setup_root_logger
from utils.logging import get_logger
from utils.checkpoint import get_checkpoint_dir

# ---------------------------------------------------------------------------
# Logger – will inherit root configuration set up in `main()` before use.
# ---------------------------------------------------------------------------
logger = get_logger(__name__)


# ========================================================================
# Public entry point
# ========================================================================

def main() -> None:
    """Parse CLI arguments and dispatch to the requested workflow."""
    parser = argparse.ArgumentParser(
        description="Reproduce the NaViL multimodal model experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["design_experiment", "scaling", "full_training"],
        required=True,
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Override model variant (2B or 9B).",
    )
    parser.add_argument(
        "--design_submode",
        type=str,
        choices=["init", "moe", "encoder", "all"],
        default="all",
        help="Sub‑experiment for design mode.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Automatically set by torchrun / DeepSpeed launcher.",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="Optional path to a DeepSpeed configuration JSON file.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Set up distributed process group if needed.
    # ------------------------------------------------------------------
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        logger.info("Distributed initialised: rank %d / %d", rank, world_size)
    else:
        world_size = 1
        rank = 0

    # ------------------------------------------------------------------
    # 2. Configure root logging – only rank 0 emits messages.
    # ------------------------------------------------------------------
    _setup_root_logger(
        level=logging.INFO if rank == 0 else logging.WARNING,
        log_file=None,  # logs to console only for simplicity; can be extended.
    )

    # ------------------------------------------------------------------
    # 3. Load configuration, apply variant overrides, set seed.
    # ------------------------------------------------------------------
    base_cfg = load_config(args.config)
    if args.variant is not None:
        base_cfg.model.variant = args.variant

    # The 9B variant override is already handled inside `load_config` when
    # the variant is "9B".  So the returned `Config` is fully resolved.
    config = base_cfg

    # Set global seed for reproducibility.
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    # ------------------------------------------------------------------
    # 4. Dispatch to the selected workflow.
    # ------------------------------------------------------------------
    if args.mode == "design_experiment":
        Main.run_design_experiment(config, args.design_submode)
    elif args.mode == "scaling":
        Main.run_scaling_experiment(config)
    elif args.mode == "full_training":
        Main.run_full_training(config, args.deepspeed_config)

    logger.info("Experiment finished successfully.")


# ========================================================================
# Main orchestrator class (as per design)
# ========================================================================

class Main:
    """
    Static entry points that correspond to the three reproduction workflows.

    All experiments are driven by a ``Config`` object obtained from
    ``config.yaml``, loaded and validated by ``config.py``.
    """

    # ------------------------------------------------------------------
    # Design experiment: validate architectural choices (Sec. 3.2)
    # ------------------------------------------------------------------
    @staticmethod
    def run_design_experiment(config: Config, sub_mode: str = "all") -> None:
        """
        Run small‑scale training to reproduce the three key findings
        described in Section 3.2 of the NaViL paper.

        Args:
            config: Full system configuration.
            sub_mode: One of ``"init"``, ``"moe"``, ``"encoder"``, ``"all"``.
        """
        logger.info("Starting design experiment (sub‑mode: %s)", sub_mode)
        # We use a reduced training budget – 10,000 steps with a small global
        # batch size to keep experiments fast while still revealing trends.
        # All runs share the same S1.1‑like noisy data.
        design_steps = 10000
        design_batch = 256  # global batch size per design run (scaled down)

        # Validation will be performed on a small held‑out set.
        # We assume that the S1.1 dataset has a validation split accessible via
        # a separate WebDataset path.  If not, fall back to using a fraction of
        # the training shards.
        if hasattr(config.data, "s1_1_valid") and config.data.s1_1_valid:
            valid_paths = config.data.s1_1_valid
        else:
            # Use a single shard as validation – acceptable for small‑scale design.
            if config.data.s1_1.raw_image_text:
                valid_paths = [config.data.s1_1.raw_image_text[0]]
            else:
                valid_paths = ["/tmp/fake_valid"]  # safe fallback

        # ------------------------------------------------------------------
        # Sub‑experiment 1: LLM initialization
        # ------------------------------------------------------------------
        if sub_mode in ("all", "init"):
            logger.info("=== Design: LLM initialization ===")
            # --- Model A: pre‑trained LLM (standard) ---
            model_init = Main._build_model(config.model)
            loss_init = Main._run_quick_train_and_eval(
                model_init,
                config,
                "s1_1",        # mimic S1.1 freezing
                design_steps,
                design_batch,
                valid_paths,
                tag="init_pre_trained",
            )
            logger.info("Pre‑trained init final val loss: %.4f", loss_init[-1] if loss_init else float("nan"))

            # --- Model B: randomly initialised LLM ---
            # Create a model with the same architecture but fresh linguistic weights.
            # We achieve this by temporarily replacing the base model with a
            # randomly initialised version (using the same config).
            # A quick way: build a second model with a different base model path
            # that points to a randomly initialised checkpoint.  For simplicity
            # we can create a fresh model using the same architecture but not
            # loading pre‑trained weights.  We'll reuse `_build_model` but with
            # a flag to skip loading.
            # Here we implement a dedicated builder `_build_model_random`.
            model_random = Main._build_model_random(config.model)
            loss_random = Main._run_quick_train_and_eval(
                model_random,
                config,
                "s1_1",
                design_steps,
                design_batch,
                valid_paths,
                tag="init_random",
            )
            logger.info("Random init final val loss: %.4f", loss_random[-1] if loss_random else float("nan"))

        # ------------------------------------------------------------------
        # Sub‑experiment 2: MoE effectiveness
        # ------------------------------------------------------------------
        if sub_mode in ("all", "moe"):
            logger.info("=== Design: MoE effectiveness ===")
            # --- Model C: vanilla LLM (no modality‑specific experts) ---
            # Build model but disable MoE by setting both attention_experts and
            # ffn_experts to False.
            cfg_no_moe = copy.deepcopy(config.model)
            cfg_no_moe.llm.attention_experts = False
            cfg_no_moe.llm.ffn_experts = False
            model_vanilla = Main._build_model(cfg_no_moe)
            loss_vanilla = Main._run_quick_train_and_eval(
                model_vanilla,
                config,
                "s1_1",
                design_steps,
                design_batch,
                valid_paths,
                tag="moe_vanilla",
            )
            logger.info("Vanilla LLM final val loss: %.4f", loss_vanilla[-1] if loss_vanilla else float("nan"))

            # --- Model D: MoE‑extended LLM (default) ---
            # Already built for the init experiment; do not rebuild if already available.
            model_moe = Main._build_model(config.model)
            loss_moe = Main._run_quick_train_and_eval(
                model_moe,
                config,
                "s1_1",
                design_steps,
                design_batch,
                valid_paths,
                tag="moe_moe",
            )
            logger.info("MoE LLM final val loss: %.4f", loss_moe[-1] if loss_moe else float("nan"))

        # ------------------------------------------------------------------
        # Sub‑experiment 3: Encoder depth/width trade‑off (600 M budget)
        # ------------------------------------------------------------------
        if sub_mode in ("all", "encoder"):
            logger.info("=== Design: Visual encoder depth/width ===")
            # Parameter budget (in billions, 0.6B)
            budget = 600_000_000
            depths = [3, 6, 12, 24, 48]
            results = {}
            for d in depths:
                w, mlp_w, n_heads = Main._compute_encoder_shape(d, budget)
                logger.info("Depth %d → width %d, heads %d", d, w, n_heads)
                # Override visual encoder config
                ve_cfg = copy.deepcopy(config.model.visual_encoder)
                ve_cfg.depth = d
                ve_cfg.width = w
                ve_cfg.mlp_width = mlp_w
                ve_cfg.num_attention_heads = n_heads
                model_cfg = copy.deepcopy(config.model)
                model_cfg.visual_encoder = ve_cfg
                model_variant = Main._build_model(model_cfg)
                losses = Main._run_quick_train_and_eval(
                    model_variant,
                    config,
                    "s1_1",
                    design_steps,
                    design_batch,
                    valid_paths,
                    tag=f"encoder_d{d}",
                )
                results[(d, w)] = losses[-1] if losses else float("nan")
            logger.info("Encoder design results: %s", results)

    # ------------------------------------------------------------------
    # Scaling experiment: explore scaling laws (Sec. 3.3)
    # ------------------------------------------------------------------
    @staticmethod
    def run_scaling_experiment(config: Config) -> None:
        """
        Investigate scaling properties of visual encoder and LLM
        independently, and derive the optimal encoder size for a given LLM.

        This method follows the protocol described in Section 3.3 of the
        paper, training multiple model variants on a shared noisy dataset
        and recording validation loss at the end of each run.
        """
        logger.info("Starting scaling experiment")
        # --- Shared training settings ---
        # For practical reasons we reduce the training steps compared to the
        # full scale‑up; the goal is to observe convergence trends.
        scaling_steps = 20000   # per configuration
        scaling_batch = 512     # global batch size (scaled appropriately)

        # Which LLM sizes to test.
        llm_sizes: List[float] = [0.5, 1.8, 7.0]   # in Billions
        # Path to pre‑trained checkpoints for each size (must exist).
        llm_paths: Dict[float, str] = {
            0.5: "internlm2-0.5b",   # replace with actual paths
            1.8: config.model.llm.base_model,
            7.0: "internlm2-7b",
        }
        # Visual encoder sizes to sweep (in Millions)
        encoder_sizes: List[int] = [75, 150, 300, 600, 1200, 2400]

        # Validation data – same as design, a held‑out subset.
        valid_paths = config.data.s1_1.raw_image_text[:1] if config.data.s1_1.raw_image_text else ["/tmp/fake"]

        results: Dict[str, float] = {}  # key: "llmX_encY" → final loss

        for llm_size in llm_sizes:
            for enc_size in encoder_sizes:
                key = f"llm{llm_size}B_enc{enc_size}M"
                logger.info("Scaling run: %s", key)
                # Build model configuration
                mcfg = copy.deepcopy(config.model)
                # LLM
                mcfg.llm.base_model = llm_paths[llm_size]
                # Visual encoder
                d, w, mlp_w, n_heads = Main._compute_encoder_shape_for_size(enc_size * 1e6)
                mcfg.visual_encoder.depth = d
                mcfg.visual_encoder.width = w
                mcfg.visual_encoder.mlp_width = mlp_w
                mcfg.visual_encoder.num_attention_heads = n_heads

                model = Main._build_model(mcfg)
                losses = Main._run_quick_train_and_eval(
                    model,
                    config,
                    "s1_1",          # vision‑only trainable, as in scaling paper
                    scaling_steps,
                    scaling_batch,
                    valid_paths,
                    tag=key,
                )
                final_loss = losses[-1] if losses else float("nan")
                results[key] = final_loss
                logger.info("%s final val loss: %.4f", key, final_loss)

        logger.info("Scaling experiment results: %s", results)
        # After collection, the user can plot the loss curves ; the paper’s
        # optimal encoder size criterion is applied via post‑processing.

    # ------------------------------------------------------------------
    # Full training: three‑stage pipeline + evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def run_full_training(
        config: Config,
        deepspeed_config_path: Optional[str] = None,
    ) -> None:
        """
        Execute the complete training recipe (S1.1 → S1.2 → S2) and
        evaluate on all benchmarks listed in ``config.evaluation``.

        Args:
            config: Fully resolved configuration.
            deepspeed_config_path: If provided, used instead of the built‑in
                DeepSpeed defaults.
        """
        logger.info("Starting full training for variant %s", config.model.variant)

        # Detect device – Trainer will handle distribution.
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Build the model.
        model = Main._build_model(config.model)
        logger.info("Model built. Total parameters: %.1f M",
                    sum(p.numel() for p in model.parameters()) / 1e6)

        # 2. Instantiate Trainer.
        trainer = Trainer(
            model=model,
            config=config.training,
            device=device,
            deepspeed_config_path=deepspeed_config_path,
        )

        # Determine number of available workers for DataLoader.
        # (The Trainer uses config.data.num_workers; we could override.)

        # 3. Stage S1.1 – Vision‑only trainable.
        logger.info("=== Stage S1.1 ===")
        stage_cfg = config.training.stages["s1_1"]
        # Build dataset for S1.1.
        ds_s1_1 = MultiModalDataset(
            config=config.data,
            split="train",
            stage="s1_1",
        )
        # Trainer’s train_stage will freeze parameters automatically via freeze_pattern.
        trainer.train_stage("s1_1", ds_s1_1)
        logger.info("Stage S1.1 completed.")

        # 4. Stage S1.2 – Unfreeze attention, high‑quality data.
        logger.info("=== Stage S1.2 ===")
        # Data for S1.2 – mixture of multimodal and pure language.
        ds_s1_2 = MultiModalDataset(
            config=config.data,
            split="train",
            stage="s1_2",
        )
        trainer.train_stage("s1_2", ds_s1_2)
        logger.info("Stage S1.2 completed.")

        # 5. Stage S2 – Supervised fine‑tuning.
        logger.info("=== Stage S2 ===")
        ds_s2 = MultiModalDataset(
            config=config.data,
            split="train",
            stage="s2",
        )
        trainer.train_stage("s2", ds_s2)
        logger.info("Stage S2 completed.")

        # 6. Evaluation.
        logger.info("=== Evaluation ===")
        # Only rank 0 performs evaluation in a distributed setting.
        if not dist.is_initialized() or dist.get_rank() == 0:
            evaluator = Evaluator(model=model, config=config.evaluation)
            metrics = evaluator.evaluate()
            logger.info("Final metrics: %s", metrics)
            # Save to output directory
            os.makedirs(config.evaluation.output_dir, exist_ok=True)
            output_file = os.path.join(config.evaluation.output_dir, "results.json")
            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info("Results saved to %s", output_file)
        else:
            logger.info("Rank %d skipping evaluation (handled by rank 0).", dist.get_rank())

    # ------------------------------------------------------------------
    # Private helper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model(model_config: ModelConfig, tokenizer: Optional[TextTokenizer] = None) -> NaViLModel:
        """
        Convenience wrapper that instantiates a full NaViLModel from a
        configuration.

        The tokenizer is created inside the model if not provided;
        otherwise it is reused.
        """
        if tokenizer is None:
            tokenizer = TextTokenizer(
                tokenizer_name=model_config.llm.base_model,
                special_tokens=model_config.special_tokens,
            )
        model = NaViLModel(config=model_config, tokenizer=tokenizer)
        return model

    @staticmethod
    def _build_model_random(model_config: ModelConfig) -> NaViLModel:
        """
        Build a model where the linguistic parameters are randomly
        initialised (not from a pre‑trained checkpoint).

        This is achieved by loading a base model with the same architecture
        but using the ``AutoModelForCausalLM.from_config`` method and
        overriding its state dict.  However, to keep the MoE wrapper
        consistent, we first build the model normally, then reset the
        linguistic weights to random.
        """
        model = NaViLModel(config=model_config)
        # Reset all parameters that belong to the MoE linguistic experts.
        # We can recognise them by their names: those containing "ling_"
        # and the base embed/lm_head.  A brute‑force approach:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "ling_" in name or "embed_tokens" in name or "lm_head" in name or "norm" in name:
                    if "weight" in name:
                        nn.init.normal_(param, std=0.02)
                    elif "bias" in name:
                        nn.init.zeros_(param)
        # Re‑initialise the token embeddings as well if needed.
        return model

    @staticmethod
    def _run_quick_train_and_eval(
        model: NaViLModel,
        full_config: Config,
        stage: str,
        total_steps: int,
        global_batch_size: int,
        valid_shard_paths: List[str],
        tag: str = "",
    ) -> List[float]:
        """
        Run a short training loop on a small subset of data and return the
        evolution of validation loss.

        The function creates a mini‑config by overriding the stage settings,
        constructs a temporary Trainer (without DeepSpeed if only one GPU
        is available), and runs training.  After training, it evaluates
        validation loss on a separate held‑out dataset.

        Returns:
            A list of validation losses recorded at intervals.
        """
        # Build a training config for this run.
        train_cfg = copy.deepcopy(full_config.training)
        stage_cfg = copy.deepcopy(train_cfg.stages[stage])
        stage_cfg.steps = total_steps
        stage_cfg.global_batch_size = global_batch_size
        train_cfg.stages[stage] = stage_cfg

        # Use a simple loop – for quick experiments we bypass DeepSpeed and
        # use a single GPU (or all available GPUs if needed, but Trainer
        # handles that).  We'll use the Trainer class, which initializes
        # DeepSpeed; if only one GPU we can avoid it, but to reuse code we
        # keep it.  To avoid heavy DeepSpeed overhead, we can set
        # `deepspeed_config_path` to a minimal config.
        device = next(model.parameters()).device.type
        trainer = Trainer(
            model=model,
            config=train_cfg,
            device=device,
            deepspeed_config_path=None,  # triggers built‑in minimal config
        )

        # Build dataset for training.
        ds_train = MultiModalDataset(
            config=full_config.data,
            split="train",
            stage=stage,
        )

        # Train.
        trainer.train_stage(stage, ds_train)

        # Validation loss
        val_losses = []
        if valid_shard_paths:
            ds_valid = MultiModalDataset(
                config=full_config.data,
                split="valid",
                stage=stage,
            )
            # The dataset is streaming; we need a DataLoader.
            dataloader = torch.utils.data.DataLoader(
                ds_valid,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=MultiModalDataset.collate_fn,
            )
            val_loss = Main._compute_validation_loss(model, dataloader)
            val_losses.append(val_loss)
        else:
            val_losses = [float("nan")]

        logger.info("Quick train (%s) final val loss: %.4f", tag, val_losses[-1])
        return val_losses

    @staticmethod
    def _compute_validation_loss(model: NaViLModel, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Compute teacher‑forcing cross‑entropy loss over a validation set.

        The model is switched to evaluation mode and no gradients are
        tracked.
        """
        was_training = model.training
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to model device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device=next(model.parameters()).device)
                outputs = model(
                    pixel_values=batch.get("pixel_values"),
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    image_token_mask=batch["image_token_mask"],
                    labels=batch["labels"],
                )
                loss = outputs["loss"]
                if loss is not None:
                    # loss is averaged across batch; we need to accumulate sum
                    num_tokens = (batch["labels"] != -100).sum().item()
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
        if was_training:
            model.train()
        return total_loss / total_tokens if total_tokens > 0 else 0.0

    @staticmethod
    def _compute_encoder_shape(
        depth: int,
        target_params: int,
    ) -> Tuple[int, int, int, int]:
        """
        Given a target parameter count and a depth, return a plausible width,
        mlp_width and number of heads for the visual encoder.

        The parameter count is approximately ``12 * depth * width^2``
        (ignoring patch embedding and final norm).  The mlp_width is set to
        ``4 * width``, and heads are chosen as ``width // 64`` while ensuring
        divisibility.

        Returns:
            Tuple (width, mlp_width, num_heads).
        """
        # Approximate width from formula
        width_approx = math.sqrt(target_params / (12 * depth))
        # Round to nearest integer divisible by 64
        width = int(round(width_approx / 64.0)) * 64
        if width == 0:
            width = 64  # fallback
        # Ensure width is at least 64
        width = max(width, 64)
        # Compute heads
        num_heads = width // 64
        # mlp_width: 4x width (standard)
        mlp_w = 4 * width
        return width, mlp_w, num_heads

    @staticmethod
    def _compute_encoder_shape_for_size(
        target_params: int,
    ) -> Tuple[int, int, int, int]:
        """
        Similar to ``_compute_encoder_shape`` but automatically determines
        a reasonable depth (e.g., 24 is a good default).  Falls back to the
        formula.
        """
        # We fix depth to 24 for sizes around 600M, adjust for others.
        if target_params < 100_000_000:
            depth = 12
        elif target_params < 300_000_000:
            depth = 18
        elif target_params < 1_200_000_000:
            depth = 24
        else:
            depth = 32
        return Main._compute_encoder_shape(depth, target_params)


# ---------------------------------------------------------------------------
# Run if executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
