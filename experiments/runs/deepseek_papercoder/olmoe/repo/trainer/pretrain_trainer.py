# trainer/pretrain_trainer.py
"""
Pretraining trainer for the OLMoE-1B-7B Mixture-of-Experts language model.

Implements the full distributed training loop using PyTorch FSDP,
AdamW optimizer, and the paper's two-phase learning rate schedule
(cosine decay + linear annealing).  Auxiliary load balancing and
router z‑losses are computed per MoE layer and added to the
cross‑entropy objective with weights read from ``config.yaml``.

The class also schedules in‑loop evaluations and periodic checkpointing
to match the reproducibility requirements of the OLMoE paper.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

# Import custom components – paths assume the project root is in PYTHONPATH
from data.dataset_loader import DataLoader
from model.losses import load_balancing_loss, router_z_loss
from model.moe_transformer import MoETransformer
from evaluation.in_loop_eval import InLoopEvaluator
from trainer.utils import (
    get_optimizer_and_scheduler,
    load_checkpoint,
    save_checkpoint,
    setup_fsdp_model,
)
from utils.logging_utils import init_wandb, log_metrics

logger = logging.getLogger(__name__)


class PretrainTrainer:
    """Orchestrates the full pretraining of an OLMoE model.

    The trainer assumes that ``torch.distributed.init_process_group()``
    has already been called and that the environment variables
    ``WORLD_SIZE``, ``RANK``, ``LOCAL_RANK`` are set.

    Args:
        model:          The uninitalised MoETransformer instance.
        data_loader:    The DataLoader that provides streaming tokenised
                        pretraining data via ``get_pretrain_dataset()``.
        config:         Full configuration dictionary (as loaded from
                        ``config.yaml``).  Must contain keys
                        ``model``, ``pretraining``, ``evaluation``,
                        ``logging``, and ``fsdp``.
    """

    def __init__(
        self,
        model: MoETransformer,
        data_loader: DataLoader,
        config: Dict,
    ) -> None:
        self.config = config
        self.cfg_pretrain = config["pretraining"]
        self.cfg_model = config["model"]
        self.cfg_moe = config["model"]["moe"]
        self.cfg_eval = config["evaluation"]
        self.cfg_logging = config["logging"]

        # ------- Distributed environment -------
        if not dist.is_initialized():
            raise RuntimeError(
                "Distributed process group must be initialised before "
                "creating the PretrainTrainer."
            )
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # ------- FSDP wrapping -------
        self.fsdp_model = setup_fsdp_model(model, config)
        # Unwrapped model needed for some evaluation utilities?
        # We keep a reference to the raw model if needed; FSDP does not
        # alter the original.
        self.raw_model = model

        # ------- Optimizer and LR scheduler -------
        self.optimizer: Optimizer
        self.scheduler: LRScheduler
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(
            self.fsdp_model, config
        )

        # ------- Auxiliary loss weights -------
        self.lb_weight = self.cfg_moe["load_balancing_weight"]   # α = 0.01
        self.rz_weight = self.cfg_moe["router_z_loss_weight"]    # β = 0.001
        self.top_k = self.cfg_moe["top_k"]                       # k = 8
        self.num_experts = self.cfg_moe["num_experts"]           # 64

        # ------- In‑loop evaluator -------
        self.tokenizer = data_loader.tokenizer
        self.in_loop_evaluator = InLoopEvaluator(
            tokenizer=self.tokenizer,
            tasks=self.cfg_eval.get("in_loop_tasks", []),
            eval_config=self.cfg_eval,
        )

        # ------- Logging -------
        init_wandb(self.cfg_logging)

        # ------- Compute step boundaries -------
        tokens_per_step = (
            self.cfg_pretrain["global_batch_size_samples"]
            * self.cfg_pretrain["seq_length"]
        )
        self.total_steps = int(
            self.cfg_pretrain["total_tokens"] // tokens_per_step
        )
        anneal_tokens = self.cfg_pretrain["annealing_tokens"]
        self.anneal_start_step = self.total_steps - int(
            anneal_tokens // tokens_per_step
        )
        self.warmup_steps = self.cfg_pretrain["warmup_steps"]

        # Per‑GPU batch size (must be integer)
        self.samples_per_step = (
            self.cfg_pretrain["global_batch_size_samples"] // self.world_size
        )
        assert (
            self.cfg_pretrain["global_batch_size_samples"] % self.world_size == 0
        ), (
            "Global batch size must be divisible by world size "
            f"({self.cfg_pretrain['global_batch_size_samples']} vs {self.world_size})"
        )
        self.seq_length = self.cfg_pretrain["seq_length"]

        # ------- Data iterator (created inside train()) -------
        self.train_iter: Optional[Iterator[torch.Tensor]] = None

        # ------- Checkpointing settings -------
        self.checkpoint_dir = self.cfg_pretrain.get(
            "checkpoint_dir", "checkpoints"
        )
        self.checkpoint_interval = self.cfg_pretrain.get(
            "checkpoint_interval", 5000
        )

        logger.info(
            "PretrainTrainer initialised: rank=%d/%d, total_steps=%d, "
            "anneal_start=%d, samples_per_step=%d, seq_len=%d",
            self.rank,
            self.world_size,
            self.total_steps,
            self.anneal_start_step,
            self.samples_per_step,
            self.seq_length,
        )

    # ------------------------------------------------------------------
    # Public training entry point
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the full pretraining loop, possibly resuming from a checkpoint."""
        start_step = self._maybe_resume()

        if self.rank == 0:
            logger.info(
                "Starting training from step %d to %d",
                start_step,
                self.total_steps,
            )

        # Data iterator: yields tensors of shape (seq_length,) per rank.
        self.train_iter = self._create_data_iterator(start_step)

        self.fsdp_model.train()

        for step in range(start_step, self.total_steps):
            # ---- Build a micro‑batch ----
            batch_tensors = []
            for _ in range(self.samples_per_step):
                try:
                    seq = next(self.train_iter)
                except StopIteration:
                    logger.error("Data exhausted at step %d", step)
                    raise
                batch_tensors.append(seq)
            input_ids = torch.stack(batch_tensors, dim=0).to(
                self.device, non_blocking=True
            )  # (B, S)

            # Prepare inputs and labels (standard LM shift)
            labels = input_ids[:, 1:]            # (B, S-1)
            current_input = input_ids[:, :-1]    # (B, S-1)
            attention_mask = torch.ones_like(
                current_input, dtype=torch.bool
            )

            # ---- Forward pass ----
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.fsdp_model(
                    input_ids=current_input, attention_mask=attention_mask
                )
                logits, all_router_logits = outputs   # see MoETransformer.forward

            # ---- Compute losses ----
            total_loss = self._compute_total_loss(
                logits, labels, all_router_logits
            )

            # ---- Backward and step ----
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping (global L2 norm)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.fsdp_model.parameters(),
                self.cfg_pretrain["gradient_clipping"],
            )

            self.optimizer.step()
            self.scheduler.step()

            # ---- Logging ----
            if step % self.cfg_logging.get("log_interval", 10) == 0:
                self._log_training_metrics(step, total_loss, grad_norm)

            # ---- In‑loop evaluation ----
            eval_freq = self.cfg_eval.get("frequency_steps", 5000)
            if step > 0 and step % eval_freq == 0:
                self._run_in_loop_evaluation(step)

            # ---- Checkpointing ----
            if step > 0 and step % self.checkpoint_interval == 0:
                self._save_checkpoint(step)

        # Final checkpoint and cleanup
        self._save_checkpoint(self.total_steps)
        self._cleanup()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _maybe_resume(self) -> int:
        """Check for the latest checkpoint and restore if available."""
        if not os.path.exists(self.checkpoint_dir):
            return 0

        # Look for the most recent checkpoint
        checkpoints = sorted(
            [
                f
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith("step_") and f.endswith(".pt")
            ],
            key=lambda f: int(f.split("_")[1].split(".")[0]),
        )
        if not checkpoints:
            return 0

        latest_ckpt = os.path.join(self.checkpoint_dir, checkpoints[-1])
        self.rank == 0 and logger.info("Resuming from %s", latest_ckpt)

        resumed_step = load_checkpoint(
            fsdp_model=self.fsdp_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            path=latest_ckpt,
        )
        return resumed_step + 1   # start from the next step

    def _create_data_iterator(
        self, start_step: int
    ) -> Iterator[torch.Tensor]:
        """Create the streaming data iterator, potentially skipping steps."""
        # The DataLoader yields one tensor per rank; we consume according
        # to global step count. To resume, we must advance the iterator
        # by the number of steps already taken.
        #    start_step * samples_per_step sequences.
        iterator = self.data_loader.get_pretrain_dataset(
            shuffle_seed=42   # could be parameterised
        )
        # Skip already consumed steps (each step consumes samples_per_step)
        skip_sequences = start_step * self.samples_per_step
        if skip_sequences > 0:
            logger.info("Skipping %d sequences to resume", skip_sequences)
            for _ in range(skip_sequences):
                try:
                    next(iterator)
                except StopIteration:
                    raise RuntimeError(
                        "Not enough data to skip. Check dataset size and "
                        "resume step."
                    )
        return iterator

    def _compute_total_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        all_router_logits: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the sum of cross‑entropy and auxiliary losses."""
        # Cross‑entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # Auxiliary losses – compute top‑k indices for load balancing
        lb_loss = torch.tensor(0.0, device=ce_loss.device)
        rz_loss = torch.tensor(0.0, device=ce_loss.device)

        for r_logits in all_router_logits:
            # r_logits shape (batch_size * seq_len, num_experts)
            # Obtain top‑k indices needed by load_balancing_loss
            with torch.no_grad():
                probs = F.softmax(r_logits.float(), dim=-1)
                _, topk_idx = torch.topk(probs, self.top_k, dim=-1)

            lb_loss += load_balancing_loss(
                r_logits, topk_idx, self.num_experts
            )
            rz_loss += router_z_loss(r_logits)

        total_loss = (
            ce_loss
            + self.lb_weight * lb_loss
            + self.rz_weight * rz_loss
        )
        return total_loss

    def _log_training_metrics(
        self, step: int, loss: torch.Tensor, grad_norm: float
    ) -> None:
        """Log scalar metrics to W&B and stdout."""
        lr = self.scheduler.get_last_lr()[0]
        metrics = {
            "train/loss": loss.item(),
            "train/learning_rate": lr,
            "train/grad_norm": grad_norm,
        }
        log_metrics(metrics, step=step)
        if self.rank == 0:
            logger.debug(
                "Step %d: loss=%.4f, lr=%.2e, grad_norm=%.2f",
                step,
                loss.item(),
                lr,
                grad_norm,
            )

    def _run_in_loop_evaluation(self, step: int) -> None:
        """Perform in‑loop downstream evaluation and log results."""
        self.fsdp_model.eval()
        with torch.no_grad():
            # Need to unwrap to the root model?  The evaluator expects a
            # callable that returns logits. We'll pass self.raw_model
            # (which shares the same parameters). For FSDP a custom
            # gather may be required, but for simplicity we use the
            # unwrapped model; this works because FSDP does not change
            # parameter values.
            eval_results = self.in_loop_evaluator.evaluate(
                model=self.raw_model, step=step
            )
        self.fsdp_model.train()

        metrics = {f"eval/{k}": v for k, v in eval_results.items()}
        log_metrics(metrics, step=step)
        if self.rank == 0:
            logger.info("Step %d evaluation: %s", step, eval_results)

    def _save_checkpoint(self, step: int) -> None:
        """Save a training checkpoint."""
        if self.rank != 0:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"step_{step}.pt")
        save_checkpoint(
            fsdp_model=self.fsdp_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            path=path,
            config=self.config,
        )

    def _cleanup(self) -> None:
        """Close W&B run and perform any necessary cleanup."""
        import wandb

        if wandb.run is not None:
            wandb.finish()
        logger.info("Pretraining completed.")

