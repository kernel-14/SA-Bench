"""
trainers/generator_trainer.py

GeneratorTrainer for the Next‑Frequency Prediction transformer (VARTransformer).

Implements supervised training with cross‑entropy loss, Adam optimiser,
cosine learning rate schedule with linear warmup, classifier‑free guidance
via label dropping, and gradient clipping.  All hyperparameters are read
from the project configuration dictionary (config.yaml).
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ar_transformer import VARTransformer


class GeneratorTrainer:
    """
    Trainer for the VARTransformer (Next‑Frequency Prediction).

    Orchestrates the full training loop, including:
      - Adam optimisation with betas=(0.9, 0.96).
      - Cosine learning rate schedule with linear warmup.
      - Classifier‑free guidance by randomly replacing class labels with
        a null (unconditional) index during training.
      - Gradient clipping.
      - Periodic validation and checkpointing.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        model: VARTransformer,
        config: Dict[str, Any],
    ) -> None:
        """
        Initialise the generator trainer.

        Args:
            model: VARTransformer instance to be trained.
            config: Global project configuration dictionary (loaded from config.yaml).
        """
        self.model = model
        self.config = config

        # ---- Device detection ----
        self.device = next(self.model.parameters()).device

        # ---- Unpack configuration sections ----
        gen_cfg: Dict[str, Any] = config["generator"]
        train_cfg: Dict[str, Any] = config["training_generator"]
        data_cfg: Dict[str, Any] = config["data"]
        tokenizer_cfg: Dict[str, Any] = config["tokenizer"]

        # Vocabulary and class indices
        self.vocab_size: int = gen_cfg["vocab_size"]                  # 4096
        self.num_classes: int = data_cfg["num_classes"]               # 1000
        self.null_class_idx: int = self.num_classes                  # unconditional index = 1000

        # Frequency scale dimensions (needed to split flat token sequences)
        self.scale_sizes: List[int] = tokenizer_cfg["scale_sizes"]
        self.num_scales: int = len(self.scale_sizes)
        self.total_tokens: int = sum(s * s for s in self.scale_sizes)  # 680

        # Training hyperparameters (with defaults matching config.yaml)
        self.epochs: int = train_cfg.get("epochs", 350)
        self.warmup_epochs: int = train_cfg.get("warmup_epochs", 10)
        self.base_lr: float = train_cfg.get("learning_rate", 8.0e-5)
        self.grad_clip: float = train_cfg.get("grad_clip", 1.0)
        self.cfg_drop_prob: float = train_cfg.get("cfg_drop_prob", 0.1)

        # ---- Optimiser (Adam with GAN‑style betas from reproducibility plan) ----
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.base_lr,
            betas=(0.9, 0.96),
            weight_decay=0.0,
        )

        # ---- Learning rate scheduler (cosine with linear warmup) ----
        self.scheduler = self._build_scheduler()

        # ---- Loss function ----
        self.criterion = nn.CrossEntropyLoss()

        # ---- Mixed precision (off by default, set use_amp: true in config to enable) ----
        self.use_amp: bool = train_cfg.get("use_amp", False)
        self.scaler = GradScaler(enabled=self.use_amp)

        # ---- Logging & checkpointing ----
        log_cfg: Dict[str, Any] = config.get("logging", {})
        self.checkpoints_dir: str = log_cfg.get("checkpoints_dir", "./checkpoints")
        self.log_interval: int = log_cfg.get("log_interval", 100)
        self.eval_interval: int = log_cfg.get("eval_interval", 5)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Training state
        self.current_epoch: int = 0
        self.best_val_loss: float = float("inf")

    # ------------------------------------------------------------------
    # LR scheduler builder (private)
    # ------------------------------------------------------------------
    def _build_scheduler(self) -> optim.lr_scheduler.LambdaLR:
        """
        Create a LambdaLR scheduler implementing linear warmup followed by
        cosine annealing to zero.

        Returns:
            A LambdaLR scheduler instance.
        """
        warmup = self.warmup_epochs
        total = self.epochs

        def lr_lambda(epoch: int) -> float:
            # epoch is 0‑based when passed by LambdaLR; our training loop uses
            # 1‑based epoch counter, so we handle both conventions safely.
            e = epoch + 1 if epoch == 0 and self.current_epoch <= 1 else epoch
            if e < warmup:
                # Linear warmup: factor goes from 0 to 1
                return float(e) / float(max(1, warmup))
            else:
                # Cosine annealing: from 1 to 0 over remaining epochs
                progress = float(e - warmup) / float(max(1, total - warmup))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Token data preparation (private)
    # ------------------------------------------------------------------
    def _prepare_token_input(
        self, token_data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Convert token batch data into the per‑scale list expected by
        VARTransformer.forward.

        Args:
            token_data: Either a flat tensor of shape (B, total_tokens)
                or a list of tensors, one per frequency band, each of
                shape (B, n_i).

        Returns:
            List of tensors, each of shape (B, n_i), moved to the correct device.
        """
        if isinstance(token_data, list):
            # Already in the expected format
            return [t.to(device=self.device, dtype=torch.long) for t in token_data]

        if isinstance(token_data, torch.Tensor):
            # Flat tensor: split according to scale_sizes
            token_data = token_data.to(device=self.device, dtype=torch.long)
            splits: List[torch.Tensor] = []
            offset = 0
            for s in self.scale_sizes:
                n_i = s * s
                splits.append(token_data[:, offset:offset + n_i])
                offset += n_i
            if offset != token_data.size(1):
                raise ValueError(
                    f"Flat token tensor has {token_data.size(1)} tokens, "
                    f"but expected {offset} based on scale_sizes."
                )
            return splits

        raise TypeError(
            f"Unsupported token_data type: {type(token_data)}. "
            f"Expected torch.Tensor or list of torch.Tensor."
        )

    # ------------------------------------------------------------------
    # CFG label dropping (private)
    # ------------------------------------------------------------------
    def _apply_cfg_label_drop(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Randomly replace a fraction of class labels with the unconditional
        (null) index for classifier‑free guidance training.

        Args:
            labels: Tensor of shape (B,) with class indices in [0, num_classes-1].

        Returns:
            Modified labels tensor.
        """
        if self.cfg_drop_prob <= 0.0:
            return labels

        drop_mask = torch.rand(labels.shape, device=labels.device) < self.cfg_drop_prob
        labels = labels.clone()
        labels[drop_mask] = self.null_class_idx
        return labels

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(
        self, batch: Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]
    ) -> Dict[str, float]:
        """
        Execute a single supervised training step on one batch.

        Args:
            batch: A tuple (token_data, class_labels).
                - token_data: Flat tensor (B, total_tokens) or list of
                  per‑scale tensors.
                - class_labels: Tensor of shape (B,) with class indices.

        Returns:
            Dictionary with keys:
                - 'loss': Cross‑entropy loss (float).
                - 'perplexity': exp(loss) (float).
                - 'lr': Current learning rate (float).
        """
        self.model.train()
        token_data, labels = batch
        labels = labels.to(device=self.device, dtype=torch.long)

        # Apply CFG label dropping
        labels = self._apply_cfg_label_drop(labels)

        # Prepare per‑scale token list
        token_list = self._prepare_token_input(token_data)

        # ---- Forward pass (with optional AMP) ----
        with autocast(enabled=self.use_amp):
            logits = self.model(token_list, labels)               # (B, S, vocab_size)
            B, S, V = logits.shape
            logits_flat = logits.reshape(B * S, V)

            # Build target by concatenating all per‑scale ground‑truth tokens
            target_flat = torch.cat(
                [t.reshape(-1) for t in token_list], dim=0
            )                                                    # (B * S,)

            loss = self.criterion(logits_flat, target_flat)

        # ---- Backward pass ----
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        # Compute derived metrics (detached)
        loss_val: float = loss.item()
        perplexity: float = math.exp(loss_val) if loss_val < 100 else float("inf")
        current_lr: float = self.optimizer.param_groups[0]["lr"]

        return {
            "loss": loss_val,
            "perplexity": perplexity,
            "lr": current_lr,
        }

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validation_step(
        self, batch: Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate the model on a validation batch.  No CFG dropping or
        gradient computation is performed.

        Args:
            batch: Tuple (token_data, class_labels).

        Returns:
            Dictionary with keys 'loss' and 'perplexity'.
        """
        self.model.eval()
        token_data, labels = batch
        labels = labels.to(device=self.device, dtype=torch.long)

        token_list = self._prepare_token_input(token_data)

        with autocast(enabled=self.use_amp):
            logits = self.model(token_list, labels)
            B, S, V = logits.shape
            logits_flat = logits.reshape(B * S, V)
            target_flat = torch.cat(
                [t.reshape(-1) for t in token_list], dim=0
            )
            loss = self.criterion(logits_flat, target_flat)

        loss_val: float = loss.item()
        perplexity: float = math.exp(loss_val) if loss_val < 100 else float("inf")
        return {
            "loss": loss_val,
            "perplexity": perplexity,
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(
        self,
        dataloader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Execute the full generator training procedure.

        Args:
            dataloader: DataLoader yielding training batches of
                (token_data, class_labels).
            val_loader: DataLoader yielding validation batches of the same format.
        """
        print(
            f"[GeneratorTrainer] Starting training for {self.epochs} epochs.\n"
            f"  Warmup epochs: {self.warmup_epochs}\n"
            f"  Base LR: {self.base_lr}\n"
            f"  CFG drop prob: {self.cfg_drop_prob}\n"
            f"  Gradient clipping: {self.grad_clip}\n"
            f"  AMP enabled: {self.use_amp}"
        )

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch

            # --------------------------------------------------------------
            # Training phase
            # --------------------------------------------------------------
            self.model.train()
            train_metrics: Dict[str, float] = {}
            num_train_batches: int = 0

            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch:3d}/{self.epochs} [Train]",
                leave=False,
            )
            for batch_idx, batch in enumerate(pbar):
                step_results = self.training_step(batch)

                # Accumulate running statistics
                for k, v in step_results.items():
                    train_metrics[k] = train_metrics.get(k, 0.0) + v
                num_train_batches += 1

                # Update progress bar at log intervals
                if (batch_idx + 1) % self.log_interval == 0:
                    avg = {k: v / num_train_batches for k, v in train_metrics.items()}
                    pbar.set_postfix(avg)

            # Average metrics over the epoch
            for k in train_metrics:
                train_metrics[k] /= max(num_train_batches, 1)

            train_loss = train_metrics.get("loss", float("inf"))
            train_ppl = train_metrics.get("perplexity", float("inf"))
            current_lr = train_metrics.get("lr", 0.0)

            print(
                f"Epoch {epoch:3d}  Train | loss={train_loss:7.4f}  "
                f"ppl={train_ppl:8.2f}  lr={current_lr:.2e}"
            )

            # --------------------------------------------------------------
            # Validation phase (run at specified intervals or final epoch)
            # --------------------------------------------------------------
            if epoch % self.eval_interval == 0 or epoch == self.epochs:
                self.model.eval()
                val_metrics: Dict[str, float] = {}
                num_val_batches: int = 0

                pbar_val = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch:3d}/{self.epochs} [Val]  ",
                    leave=False,
                )
                for batch in pbar_val:
                    step_results = self.validation_step(batch)
                    for k, v in step_results.items():
                        val_metrics[k] = val_metrics.get(k, 0.0) + v
                    num_val_batches += 1

                for k in val_metrics:
                    val_metrics[k] /= max(num_val_batches, 1)

                val_loss = val_metrics.get("loss", float("inf"))
                val_ppl = val_metrics.get("perplexity", float("inf"))

                print(
                    f"Epoch {epoch:3d}  Val   | loss={val_loss:7.4f}  "
                    f"ppl={val_ppl:8.2f}"
                )

                # Save checkpoint whenever validation loss improves
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"  -> New best model (val_loss={val_loss:.4f})")

                # Also save a regular epoch checkpoint
                self._save_checkpoint(epoch, is_best=False)

            # --------------------------------------------------------------
            # Step the learning rate scheduler once per epoch
            # --------------------------------------------------------------
            self.scheduler.step()

        # ------------------------------------------------------------------
        # Save final model
        # ------------------------------------------------------------------
        final_path = os.path.join(self.checkpoints_dir, "generator_final.pth")
        torch.save(
            {
                "epoch": self.epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            final_path,
        )
        print(
            f"[GeneratorTrainer] Training completed. "
            f"Final model saved to: {final_path}"
        )

    # ------------------------------------------------------------------
    # Checkpointing helper (private)
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Persist model weights, optimizer state, scheduler state, and
        training metadata to disk.

        Args:
            epoch: Current epoch number (1‑based).
            is_best: If True, the checkpoint is saved as 'generator_best.pt'.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if is_best:
            path = os.path.join(self.checkpoints_dir, "generator_best.pt")
        else:
            path = os.path.join(
                self.checkpoints_dir, f"generator_epoch_{epoch:03d}.pt"
            )
        torch.save(state, path)
