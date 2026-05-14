"""
trainer.py

Implements the ``Trainer`` class for both baseline GPT and Normalized Transformer (nGPT).

Responsibilities:
- Optimizer and learning‑rate scheduler initialisation (Adam/AdamW, warmup, cosine decay).
- Mixed‑precision training with bfloat16 autocast.
- Gradient accumulation to reach the global batch size.
- Training loop with periodic logging, validation loss computation, and checkpointing.
- For nGPT: post‑optimizer weight normalisation via ``model.normalize_weights()``.
- Optional integration with ``Evaluator`` for downstream metrics.

All configuration is read from the ``Config`` object (derived from ``config.yaml``).
"""

import os
import time
import math
import logging
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast
import tqdm

from config import Config
from utils import set_seed
from model import GPTModel

# Attempt to import Weights & Biases if logging is enabled.
try:
    import wandb
except ImportError:
    wandb = None

# Attempt to import the Evaluator for downstream tasks (optional).
try:
    from evaluation import Evaluator
except ImportError:
    Evaluator = None


class Trainer:
    """
    Controls the entire training process of a ``GPTModel``.

    Parameters
    ----------
    model : GPTModel
        The language model to train (wrapped in DDP if running distributed).
    config : Config
        The global configuration.
    train_loader : torch.utils.data.DataLoader
        DataLoader yielding training batches (x, y).
    val_loader : torch.utils.data.DataLoader
        DataLoader yielding validation batches.
    evaluator : Optional[Evaluator]
        An Evaluator for downstream tasks. If ``None``, only validation loss
        is reported.
    """

    def __init__(
        self,
        model: GPTModel,
        config: Config,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        evaluator: Optional["Evaluator"] = None,
    ):
        # ------------------------------------------------------------------
        # Basic attributes
        # ------------------------------------------------------------------
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator

        # Determine the effective device of the model
        self.device = next(model.parameters()).device

        # Distributed information
        self.distributed = dist.is_initialized()
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.rank = dist.get_rank() if self.distributed else 0

        # ------------------------------------------------------------------
        # Gradient accumulation setup
        # We compute accumulation steps so that the effective batch size
        # equals ``config.training.batch_size``.
        # ------------------------------------------------------------------
        # The DataLoader yields micro‑batches of size `local_bs` per GPU.
        # For example, with 64 GPUs and global batch 512, local_bs = 8.
        # If fewer GPUs are available, the DataLoader should be configured
        # to yield smaller micro‑batches; accumulation will compensate.
        local_batch_size = self._infer_local_batch_size()
        effective_local_bs = local_batch_size * self.world_size
        self.acc_steps = max(1, config.training.batch_size // effective_local_bs)
        if config.training.batch_size % effective_local_bs != 0:
            raise ValueError(
                f"Global batch size {config.training.batch_size} must be divisible by "
                f"local batch size {local_batch_size} × world_size {self.world_size}"
            )

        # ------------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------------
        optim_cfg = config.optim
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=optim_cfg.lr,
            betas=optim_cfg.betas,
            eps=optim_cfg.eps,
            weight_decay=optim_cfg.weight_decay,
        )

        # ------------------------------------------------------------------
        # Learning rate scheduler (warmup + cosine)
        # ------------------------------------------------------------------
        self.total_iters = config.training.num_iters
        self.warmup_iters = optim_cfg.warmup_steps

        def lr_lambda(cur_iter: int) -> float:
            if cur_iter < self.warmup_iters:
                # Linear warmup from 0 to 1
                return cur_iter / max(1, self.warmup_iters)
            else:
                # Cosine annealing to lr_final / lr
                progress = (cur_iter - self.warmup_iters) / max(1, self.total_iters - self.warmup_iters)
                cos = math.cos(math.pi * progress)
                lr_final_factor = optim_cfg.lr_final / optim_cfg.lr
                return lr_final_factor + (1.0 - lr_final_factor) * (1.0 + cos) * 0.5

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # ------------------------------------------------------------------
        # Mixed‑precision settings
        # ------------------------------------------------------------------
        self.use_amp = config.training.use_amp
        self.amp_dtype = torch.bfloat16 if config.training.dtype == "bfloat16" else torch.float16
        # For bfloat16 we do NOT need a gradient scaler.
        self.scaler = None
        if self.amp_dtype == torch.float16:
            self.scaler = torch.cuda.amp.GradScaler()

        # ------------------------------------------------------------------
        # Reproducibility
        # ------------------------------------------------------------------
        self._seed = getattr(config.logging, "seed", 42)
        set_seed(self._seed)

        # ------------------------------------------------------------------
        # Logging infrastructure
        # ------------------------------------------------------------------
        self.log_dir = config.logging.log_dir
        self.checkpoint_dir = config.logging.checkpoint_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.use_wandb = config.logging.use_wandb and wandb is not None
        if self.use_wandb and self.rank == 0:
            wandb.init(
                project=config.logging.wandb_project,
                config=config.to_dict(),
                name=os.path.basename(self.checkpoint_dir),
            )

        # Internal state
        self.current_step = 0  # will be set by load_checkpoint or start at 0

        # Print summary on rank 0
        if self.rank == 0:
            n_params = sum(p.numel() for p in model.parameters())
            logging.info(
                f"Training {self.total_iters} iterations with "
                f"global batch size {config.training.batch_size} "
                f"(accumulation steps = {self.acc_steps}). "
                f"Model parameters: {n_params / 1e6:.2f}M"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Run the full training loop.

        Iterates for ``total_iters`` steps, performing forward/backward,
        optimizer/scheduler updates, logging, validation, and checkpointing.
        """
        model = self.model
        config = self.config
        scaler = self.scaler

        # Set model to training mode
        model.train()

        # Initialize tqdm progress bar (only on rank 0)
        if self.rank == 0:
            pbar = tqdm.tqdm(total=self.total_iters, desc="Training")
        else:
            pbar = None

        # Effective iteration counter (starting from checkpoint if resumed)
        step = self.current_step

        # For logging we maintain accumulated loss over a macro‑batch
        accum_loss = 0.0

        # Iterator for the training DataLoader
        data_iter = iter(self.train_loader)

        while step < self.total_iters:
            step += 1

            # ----- Fetch next batch -----
            x, y = self._next_batch(data_iter)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # ----- Forward pass -----
            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits, loss = model(x, targets=y)  # loss is averaged over tokens
                # Scale the loss by accumulation steps so that the
                # effective gradient is a mean over the macro‑batch.
                loss = loss / self.acc_steps

            # Accumulate for logging (de‑scaled later)
            accum_loss += loss.item()

            # ----- Backward pass -----
            if self.use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # ----- Optimizer step when accumulation is complete -----
            if step % self.acc_steps == 0:
                if self.use_amp and scaler is not None:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Post‑step weight normalisation for nGPT
                if config.model.use_ngpt:
                    self._get_raw_model().normalize_weights()

                # Log training metrics (loss = average over macro‑batch)
                avg_loss = accum_loss * self.acc_steps  # de‑accumulate
                perplexity = math.exp(avg_loss)
                lr = self.optimizer.param_groups[0]["lr"]

                if self.rank == 0:
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        ppl=f"{perplexity:.2f}",
                        lr=f"{lr:.2e}",
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/perplexity": perplexity,
                                "train/lr": lr,
                                "step": step,
                            }
                        )

                accum_loss = 0.0

                # ----- Evaluation and checkpointing -----
                if step % config.training.eval_interval == 0:
                    val_loss, val_ppl = self._validate()
                    if self.rank == 0:
                        logging.info(
                            f"Step {step}: val loss {val_loss:.4f}, "
                            f"val ppl {val_ppl:.2f}"
                        )
                        if self.use_wandb:
                            wandb.log(
                                {
                                    "val/loss": val_loss,
                                    "val/perplexity": val_ppl,
                                    "step": step,
                                }
                            )

                        # Run downstream evaluation if an Evaluator was provided.
                        # We do this less frequently to save time (every 10th eval).
                        if self.evaluator and step % (config.training.eval_interval * 10) == 0:
                            self._run_downstream_eval(step)

                    self.save_checkpoint(step)

            if pbar is not None:
                pbar.update(1)

        # End of training
        if pbar is not None:
            pbar.close()
        if self.use_wandb and self.rank == 0:
            wandb.finish()

        # Save final checkpoint
        self.save_checkpoint(self.total_iters)

    def save_checkpoint(self, step: int, path: Optional[str] = None) -> None:
        """
        Save a training checkpoint that includes model state, optimizer,
        scheduler, and current step.

        Parameters
        ----------
        step : int
            Global training step (iteration) number.
        path : str, optional
            Custom file path. If ``None``, a default path inside
            ``checkpoint_dir`` is used.
        """
        if self.rank != 0:
            return  # Only rank 0 saves checkpoints

        if path is None:
            path = os.path.join(self.checkpoint_dir, f"ckpt_{step:07d}.pt")

        raw_model = self._get_raw_model()
        checkpoint = {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": step,
            "config": self.config.to_dict(),
        }
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> int:
        """
        Load model, optimizer, and scheduler states from a checkpoint.
        The model is ready to continue training from the saved step.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.

        Returns
        -------
        step : int
            The training step at which the checkpoint was saved.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        raw_model = self._get_raw_model()
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_step = checkpoint["step"]
        logging.info(f"Checkpoint loaded from {path}, resuming from step {self.current_step}")
        return self.current_step

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_raw_model(self) -> GPTModel:
        """
        Return the underlying ``GPTModel``, unwrapping DDP if necessary.
        """
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def _infer_local_batch_size(self) -> int:
        """
        Attempt to deduce the micro‑batch size (per GPU) that the DataLoader
        will yield.  This is used for gradient accumulation calculation.
        """
        # DataLoader objects have a 'batch_size' attribute when created
        # without a sampler that overrides it.
        if hasattr(self.train_loader, "batch_size"):
            return self.train_loader.batch_size
        # Fallback: iterate one batch and check its shape.
        try:
            sample_x, _ = next(iter(self.train_loader))
            return sample_x.shape[0]
        except StopIteration:
            raise RuntimeError("Cannot determine batch size from an empty DataLoader.")

    def _next_batch(self, data_iter):
        """
        Fetch the next batch from the DataLoader, gracefully restarting the
        iterator when exhausted.
        """
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(self.train_loader)
            return next(data_iter)

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        """
        Compute average cross‑entropy loss and perplexity on the validation set.

        Returns
        -------
        val_loss : float
            Average loss over all validation tokens.
        val_ppl : float
            Perplexity = exp(val_loss).
        """
        raw_model = self._get_raw_model()
        raw_model.eval()

        total_loss = 0.0
        total_tokens = 0

        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits, loss = raw_model(x, targets=y)  # loss averaged by model

            # The loss is already averaged per token, so we weight by batch size
            batch_tokens = x.numel()  # number of tokens in the batch
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

        raw_model.train()
        val_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        val_ppl = math.exp(val_loss) if val_loss < 1e12 else float("inf")
        return val_loss, val_ppl

    def _run_downstream_eval(self, step: int) -> None:
        """
        Run downstream evaluation using the provided Evaluator and log the results.
        """
        raw_model = self._get_raw_model()
        # Ensure weights are normalized before evaluation (nGPT).
        if self.config.model.use_ngpt:
            raw_model.normalize_weights()

        raw_model.eval()
        metrics = self.evaluator.evaluate_downstream()
        raw_model.train()

        if self.rank == 0 and metrics:
            logging.info(f"Downstream metrics at step {step}: {metrics}")
            if self.use_wandb:
                wandb.log({"downstream": metrics, "step": step})

