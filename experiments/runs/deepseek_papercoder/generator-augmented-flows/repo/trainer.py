## trainer.py
"""
trainer.py – Consistency model training loop with generator‑augmented flows.

This module provides the `ConsistencyTrainer` class, which implements the
full training procedure as described in the paper "Improving Consistency Models
with Generator‑Augmented Flows". It integrates the noise schedule, coupling
strategy, loss computation, EMA updates, and checkpointing.

The trainer assumes that all external configuration (model, optimizer, schedules,
coupling, data loader) is set up before being passed in, following the design
and the parameters in `config.yaml`.
"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List
import os
import time

from model import ConsistencyModel
from schedules import Schedules
from coupling import Coupling
from utils import pseudo_huber_loss, weighting, save_model


class ConsistencyTrainer:
    """
    Training loop orchestrator for consistency models.

    The trainer advances a single consistency model from scratch using the
    specified coupling (IC, OT, or GC) and the EDM‑style noise schedule
    discretisation. It maintains an EMA copy of the model parameters and
    periodically logs the loss and saves checkpoints.

    Args:
        model: The trainable consistency model (instance of .
        ema_model: An EMA copy of `model`. Usually initialised as a deep
            copy of `model` at the start of training.
        optimizer: Optimizer instance, typically Lion with the learning
            rate from configuration.
        total_steps: Total number of training iterations.
        schedules: ``Schedules`` instance providing time‑dependent
            discretisation and timestep sampling distributions.
        coupling: ``Coupling`` instance that constructs the intermediate
            points for the consistency loss.
        data_loader: PyTorch DataLoader yielding batches of real images
            (values in [-1, 1]).
        ema_decay: Decay rate for the exponential moving average. Default
            is 0.9999 as recommended by Song & Dhariwal (2024).
        pseudo_huber_c: Smoothing constant for the pseudo‑Huber loss,
            derived as 0.00054 * sqrt(d) where d = C * H * W.
        eval_every: Number of steps between loss / progress logging.
        save_every: Number of steps between model checkpoints.
        checkpoint_dir: Directory where checkpoints are stored.
        gradient_clip_norm: If not None, clips gradients to this maximum
            norm. The paper does not use gradient clipping.
    """

    def __init__(
        self,
        model: ConsistencyModel,
        ema_model: ConsistencyModel,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        schedules: Schedules,
        coupling: Coupling,
        data_loader: DataLoader,
        ema_decay: float = 0.9999,
        pseudo_huber_c: float = 0.03,
        eval_every: int = 10000,
        save_every: int = 10000,
        checkpoint_dir: str = "./checkpoints",
        gradient_clip_norm: Optional[float] = None,
    ) -> None:
        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.schedules = schedules
        self.coupling = coupling
        self.data_loader = data_loader

        self.ema_decay = ema_decay
        self.pseudo_huber_c = pseudo_huber_c
        self.eval_every_steps = eval_every
        self.save_every_steps = save_every
        self.checkpoint_dir = checkpoint_dir
        self.gradient_clip_norm = gradient_clip_norm

        # select device from the model
        self.device = next(model.parameters()).device

        # ensure both models are on the correct device
        self.model.to(self.device)
        self.ema_model.to(self.device)

        # create checkpoint directory if needed
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # logging containers
        self.loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def train(self) -> None:
        """
        Run the full training loop for ``self.total_steps`` iterations.

        The method iterates over the data loader; when the loader is exhausted
        it resets automatically (typical Python iteration). Every
        ``eval_every_steps`` a progress message is printed, and every
        ``save_every_steps`` a checkpoint is written.
        """
        self.model.train()
        global_step = 0
        data_iter = iter(self.data_loader)

        print(f"Starting training for {self.total_steps} steps...")
        start_time = time.time()

        while global_step < self.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # restart the data loader for a new epoch
                data_iter = iter(self.data_loader)
                batch = next(data_iter)

            x_star = batch.to(self.device) if not isinstance(batch, (list, tuple)) else batch[0].to(self.device)

            # ------------------------------------------------------------------
            # Perform a single training step
            # ------------------------------------------------------------------
            self._train_step(x_star, global_step)
            global_step += 1

            # ------------------------------------------------------------------
            # Periodic logging
            # ------------------------------------------------------------------
            if global_step % self.eval_every_steps == 0 or global_step == 1:
                N = self.schedules.current_N(global_step - 1)  # N used in the last step
                loss = self.loss_history[-1] if self.loss_history else 0.0
                elapsed = time.time() - start_time
                print(
                    f"Step [{global_step:6d}/{self.total_steps}] "
                    f"| Loss: {loss:.6f} "
                    f"| N: {N} "
                    f"| Elapsed: {elapsed:.1f}s"
                )

            # ------------------------------------------------------------------
            # Periodic checkpointing
            # ------------------------------------------------------------------
            if global_step % self.save_every_steps == 0:
                self._save_checkpoint(global_step)

        # Final save at the end of training
        self._save_checkpoint(self.total_steps)
        total_elapsed = time.time() - start_time
        print(f"Training finished. Total time: {total_elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _train_step(self, x_star: Tensor, step: int) -> None:
        """
        Execute one optimisation step.

        This method implements the core logic described in the paper and in
        Algorithm 1 of the appendix:

        1. Sample noise.
        2. Determine the current number of discrete timesteps N and obtain
           the sigma schedule.
        3. Sample a timestep index i according to the log‑normal distribution.
        4. Construct the pair (x_ti, x_tip1) using the configured coupling.
        5. Compute the consistency loss with weighting and pseudo‑Huber distance.
        6. Backward pass & optimizer step.
        7. Update the EMA model.

        Args:
            x_star: Batch of real images, shape (B, C, H, W), values in [-1, 1].
            step:   Global training step counter (0‑based).
        """
        B = x_star.shape[0]

        # ---- 1. Sample noise -------------------------------------------------
        noise = torch.randn_like(x_star)

        # ---- 2. Timestep discretisation -------------------------------------
        N = self.schedules.current_N(step)
        sigma_all = self.schedules.get_discrete_sigma(N).to(self.device)   # (N+1,)
        probs = self.schedules.timestep_sampling_distribution(N).to(self.device)  # (N,)

        # ---- 3. Sample indices and extract sigma values ---------------------
        indices = torch.multinomial(probs, B, replacement=True)            # (B,)
        sigma_i = sigma_all[indices]                                       # (B,)
        sigma_ip1 = sigma_all[indices + 1]                                 # (B,)

        # ---- 4. Construct intermediate points (IC / OT / GC) ---------------
        x_ti, x_tip1 = self.coupling.construct_pair(x_star, noise, sigma_i, sigma_ip1)

        # ---- 5. Compute the consistency loss --------------------------------
        # Weighting: λ(σ_i) = 1 / (σ_{i+1} - σ_i)
        w = weighting(sigma_i, sigma_ip1)            # (B,)
        w = w.view(B, 1, 1, 1)                      # broadcast to image shape

        # Model outputs: target = f_θ(x_{t_{i+1}}, σ_{i+1}) ; pred = f_θ(x_{t_i}, σ_i)
        target = self.model(x_tip1, sigma_ip1)
        pred = self.model(x_ti, sigma_i)

        # Stop‑gradient on the first argument
        pred_sg = pred.detach()

        # Pseudo‑Huber distance
        loss_per_sample = pseudo_huber_loss(pred_sg, target, self.pseudo_huber_c)  # (B,)
        loss = (w * loss_per_sample).mean()

        # ---- 6. Backward pass & optimisation --------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()

        # ---- 7. Update EMA model -------------------------------------------
        self._update_ema()

        # Store loss for logging
        self.loss_history.append(loss.item())

    def _update_ema(self) -> None:
        """
        Perform one step of exponential moving average on the model parameters.

        ema_param ← decay * ema_param + (1 - decay) * param
        """
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

    def _save_checkpoint(self, step: int) -> None:
        """
        Save model, EMA model, and optimizer state to a checkpoint file.

        Args:
            step: Current global training step.
        """
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{step:06d}.pt")
        save_model(
            path=path,
            model=self.model,
            ema_model=self.ema_model,
            optimizer=self.optimizer,
            step=step,
        )
        print(f"Checkpoint saved to {path}")

