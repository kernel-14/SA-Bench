```python
## training/trainer.py
"""Core training loop implementing Algorithm 1 from the iCT-GC paper.

This module implements the ``Trainer`` class, which orchestrates the joint
IC+GC learning procedure described in Algorithm 1 of "Improving Consistency
Models with Generator-Augmented Flows".

The key contribution implemented here is the **generator-augmented coupling**:
at each training step, the EMA model predicts a cleaner endpoint from a noisy
IC intermediate point, which is then re-coupled with the original noise vector
to construct GC training pairs. A per-sample Bernoulli mask mixes IC and GC
trajectories with probability μ.

Algorithm 1 data flow::

    x_star (real data)
        ├── z ~ N(0,I)
        ├── x_ti = x_star + σ_i · z                    [IC intermediate]
        ├── x_hat = EMA_model(x_ti, σ_i).detach()       [predicted endpoint]
        ├── m ~ Bernoulli(μ)                             [per-sample mask]
        ├── x_hat_mixed = m·x_hat + (1-m)·x_star        [mixed endpoint]
        ├── x_tilde_i   = x_hat_mixed + σ_i   · z       [GC lower point]
        ├── x_tilde_i1  = x_hat_mixed + σ_{i+1} · z     [GC upper point]
        ├── f_upper = model(x_tilde_i1, σ_{i+1})        [gradient flows]
        ├── f_lower = model(x_tilde_i,  σ_i).detach()   [stop-gradient]
        ├── λ = 1 / (σ_{i+1} - σ_i)                    [loss weight]
        └── loss = mean(λ · D(f_upper, f_lower))         [scalar]

Config values used (from config.yaml):
    mu:              0.5      (joint learning parameter, CIFAR-10 default)
    training_steps:  100000   (CIFAR-10) / 150000 (others)
    learning_rate:   0.0001   (CIFAR-10) / 0.00008 (others)
    s0:              10       (initial discretization intervals)
    s1:              1280     (final discretization intervals)
    sigma_min:       0.002
    sigma_max:       80.0
    rho:             7.0
    P_mean:         -1.1
    P_std:           2.0
    distance_fn:     pseudo_huber
    pseudo_huber_c:  0.00054
    coupling:        gc
    eval_every:      10000
    save_dir:        ./checkpoints/cifar10
    log_dir:         ./logs/cifar10
"""

import json
import math
import os
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lion_pytorch import Lion

from models.consistency_model import ConsistencyModel
from training.couplings import Coupling
from training.losses import DistanceFunction
from training.schedules import NoiseSchedule
from utils.ema import EMA


class Trainer:
    """Trains a consistency model with generator-augmented flows (Algorithm 1).

    Implements the joint IC+GC learning strategy from Section 5.2 of the
    paper. At each training step, a per-sample Bernoulli mask with probability
    μ determines whether each sample uses a GC trajectory (model-predicted
    endpoint) or an IC trajectory (real data endpoint).

    The combined loss in expectation is:
        L_{GC-μ}(θ) = μ · L_GC(θ) + (1 - μ) · L_CT(θ)

    Special cases:
    - μ=0: Pure IC training (reproduces iCT-IC baseline exactly)
    - μ=1: Pure GC training (fails due to distribution shift, see Figure 7)
    - μ=0.5: Optimal for CIFAR-10 (Table 1, Figure 5)

    The Lion optimizer (Chen et al., 2023) is used as specified in the paper
    (Appendix D, Tables 4-6). This is a sign-based momentum optimizer that
    differs fundamentally from Adam.

    Attributes:
        config: Configuration object with all hyperparameters.
        model: Online ConsistencyModel. Gradients flow through this.
        ema: EMA wrapper for stop-gradient endpoint prediction.
        train_loader: DataLoader for the training dataset.
        schedule: NoiseSchedule for sigma values and timestep sampling.
        distance_fn: Distance function D(·,·) for the consistency loss.
        coupling: Initial data-noise coupling strategy.
        optimizer: Lion optimizer for model parameter updates.
        global_step: Current training step (0-indexed).
        best_fid: Best FID score seen during training (for checkpoint saving).
        writer: TensorBoard SummaryWriter for metric logging.
    """

    def __init__(
        self,
        config: Any,
        model: ConsistencyModel,
        ema: EMA,
        train_loader: DataLoader,
    ) -> None:
        """Initialise the trainer with all required components.

        Instantiates the NoiseSchedule, DistanceFunction, Coupling, and
        Lion optimizer. Creates the save and log directories. Does not
        start training — call ``train()`` to begin.

        Args:
            config: Configuration object. Must expose the following attributes
                (all present in config.yaml):
                - device (str): 'cuda' or 'cpu'
                - mu (float): Joint learning parameter μ ∈ [0, 1]
                - training_steps (int): Total number of training steps K
                - eval_every (int): Checkpoint/log interval
                - s0 (int): Initial discretization intervals (default 10)
                - s1 (int): Final discretization intervals (default 1280)
                - sigma_min (float): Minimum noise level (default 0.002)
                - sigma_max (float): Maximum noise level (default 80.0)
                - rho (float): Noise schedule exponent (default 7.0)
                - P_mean (float): Timestep distribution mean (default -1.1)
                - P_std (float): Timestep distribution std (default 2.0)
                - learning_rate (float): Lion optimizer learning rate
                - weight_decay (float): Lion optimizer weight decay (default 0.0)
                - distance_fn (str): Distance mode ('pseudo_huber', 'l2', 'lpips')
                - pseudo_huber_c (float): Pseudo-Huber constant (default 0.00054)
                - coupling (str): Coupling mode ('ic', 'ot', 'gc')
                - ot_batch_size (int): OT solver batch size
                - save_dir (str): Directory for checkpoint files
                - log_dir (str): Directory for TensorBoard logs
            model: Instantiated ConsistencyModel on the target device.
                All learnable parameters reside in ``model.net`` (SongUNet).
            ema: EMA wrapper initialised with the model's current weights.
                Used exclusively for stop-gradient endpoint prediction.
            train_loader: DataLoader yielding ``(images, labels)`` or
                ``(images,)`` tuples. Must have ``drop_last=True``.

        Raises:
            TypeError: If ``model`` is not a ``ConsistencyModel`` instance.
            TypeError: If ``ema`` is not an ``EMA`` instance.
        """
        if not isinstance(model, ConsistencyModel):
            raise TypeError(
                f"Expected 'model' to be a ConsistencyModel instance, "
                f"got {type(model).__name__}."
            )
        if not isinstance(ema, EMA):
            raise TypeError(
                f"Expected 'ema' to be an EMA instance, "
                f"got {type(ema).__name__}."
            )

        self.config: Any = config
        self.model: ConsistencyModel = model
        self.ema: EMA = ema
        self.train_loader: DataLoader = train_loader

        # --- Device ---
        self.device: torch.device = torch.device(
            str(getattr(config, "device", "cuda"))
        )

        # --- Training hyperparameters (cached for hot-loop efficiency) ---
        self.mu: float = float(getattr(config, "mu", 0.5))
        self.training_steps: int = int(getattr(config, "training_steps", 100000))
        self.eval_every: int = int(getattr(config, "eval_every", 10000))
        self.s0: int = int(getattr(config, "s0", 10))
        self.s1: int = int(getattr(config, "s1", 1280))
        self.sigma_min: float = float(getattr(config, "sigma_min", 0.002))
        self.sigma_max: float = float(getattr(config, "sigma_max", 80.0))

        # --- Sub-components ---
        self.schedule: NoiseSchedule = NoiseSchedule(
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=float(getattr(config, "rho", 7.0)),
            P_mean=float(getattr(config, "P_mean", -1.1)),
            P_std=float(getattr(config, "P_std", 2.0)),
        )

        self.distance_fn: DistanceFunction = DistanceFunction(
            mode=str(getattr(config, "distance_fn", "pseudo_huber")),
            c=float(getattr(config, "pseudo_huber_c", 0.00054)),
        ).to(self.device)

        self.coupling: Coupling = Coupling(
            mode=str(getattr(config, "coupling", "gc")),
            ot_batch_size=int(getattr(config, "ot_batch_size", 512)),
        )

        # --- Optimizer ---
        self.optimizer: torch.optim.Optimizer = self._build_optimizer()

        # --- State tracking ---
        self.global_step: int = 0
        self.best_fid: float = float("inf")

        # --- Directories ---
        save_dir: str = str(getattr(config, "save_dir", "./checkpoints"))
        log_dir: str = str(getattr(config, "log_dir", "./logs"))
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # --- TensorBoard logging ---
        self.writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the Lion optimizer for consistency model training.

        Uses the Lion optimizer (Chen et al., 2023) as specified in the paper
        (Appendix D, Tables 4-6). Lion uses sign-based momentum updates and
        is fundamentally different from Adam — do not substitute.

        The optimizer is applied to the **online model** parameters only.
        EMA weights are updated separately via ``ema.update(model)`` and
        are never passed to the optimizer.

        Returns:
            A ``Lion`` optimizer instance configured with the learning rate
            and weight decay from config.yaml.

        Raises:
            ImportError: If ``lion_pytorch`` is not installed.
                Install with: ``pip install lion-pytorch==0.1.2``
        """
        learning_rate: float = float(
            getattr(self.config, "learning_rate", 0.0001)
        )
        weight_decay: float = float(
            getattr(self.config, "weight_decay", 0.0)
        )

        return Lion(
            params=self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop for ``config.training_steps`` steps.

        Implements the outer loop of Algorithm 1. At each step:
        1. Fetches a batch from the training DataLoader (with infinite cycling).
        2. Calls ``training_step`` to compute the loss.
        3. Backpropagates and updates the online model via Lion.
        4. Updates the EMA model weights.
        5. Logs metrics and saves checkpoints at ``eval_every`` intervals.

        The DataLoader is cycled infinitely using a try/except StopIteration
        pattern. This is necessary because the number of training steps
        (e.g. 100k) typically exceeds the number of batches per epoch.

        Progress is displayed via a ``tqdm`` progress bar showing the current
        loss and training step.

        After training completes, the TensorBoard writer is flushed and closed.
        """
        self.model.train()

        # Infinite data iterator with automatic reset on exhaustion
        data_iter: Iterator = iter(self.train_loader)

        # Progress bar spanning the full training duration
        pbar = tqdm(
            total=self.training_steps,
            initial=self.global_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

        # Running loss tracker for progress bar display
        running_loss: float = 0.0
        log_interval: int = 100  # log to TensorBoard every 100 steps

        while self.global_step < self.training_steps:
            # --- Fetch next batch (cycle infinitely) ---
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Extract images from batch (labels are not used in unconditional training)
            # DataLoader returns (images, labels) for labeled datasets
            if isinstance(batch, (list, tuple)):
                x_star: torch.Tensor = batch[0].to(self.device)
            else:
                x_star = batch.to(self.device)

            # --- Forward pass and loss computation ---
            self.optimizer.zero_grad()
            loss: torch.Tensor = self.training_step(x_star)

            # --- Backward pass ---
            loss.backward()

            # --- Optimizer step (Lion: sign-based momentum update) ---
            self.optimizer.step()

            # --- EMA update (must happen AFTER optimizer step) ---
            self.ema.update(self.model)

            # --- Step counter ---
            self.global_step += 1
            loss_val: float = loss.item()
            running_loss = 0.99 * running_loss + 0.01 * loss_val  # EMA of loss

            # --- TensorBoard logging (every log_interval steps) ---
            if self.global_step % log_interval == 0:
                self._log_metrics(self.global_step, loss_val)

            # --- Checkpoint saving (every eval_every steps) ---
            if self.global_step % self.eval_every == 0:
                self._save_checkpoint(step=self.global_step, fid=None)

            # --- Progress bar update ---
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{running_loss:.4f}",
                "step": self.global_step,
                "N": self.schedule.get_N(
                    self.global_step,
                    self.training_steps,
                    self.s0,
                    self.s1,
                ),
            })

        pbar.close()

        # --- Final checkpoint ---
        self._save_checkpoint(step=self.global_step, fid=None)

        # --- Flush and close TensorBoard writer ---
        self.writer.flush()
        self.writer.close()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, x_star: torch.Tensor) -> torch.Tensor:
        """Compute the consistency loss for one batch (Algorithm 1, one iteration).

        Handles progressive N scheduling, per-sample timestep sampling, and
        delegates to ``compute_gc_loss`` for the actual GC/IC mixing.

        Args:
            x_star: Batch of real data samples of shape ``(B, C, H, W)``
                in ``[-1, 1]``. Already on ``self.device``.

        Returns:
            Scalar loss tensor with gradient graph attached (for ``.backward()``).
        """
        batch_size: int = x_star.shape[0]

        # --- Step 1: Sample noise z ~ N(0, I) ---
        # Same shape as x_star; same device guaranteed by torch.randn_like
        z: torch.Tensor = torch.randn_like(x_star)

        # --- Step 2: Get current N (progressive discretization schedule) ---
        # N increases exponentially from s0+1 to s1+1 during training.
        # Must be recomputed at every step — do not cache across steps.
        N_current: int = self.schedule.get_N(
            k=self.global_step,
            K=self.training_steps,
            s0=self.s0,
            s1=self.s1,
        )

        # --- Step 3: Get sigma schedule for current N ---
        # sigmas: (N_current+1,) float32 tensor on CPU
        # Move to device for subsequent indexing operations
        sigmas: torch.Tensor = self.schedule.get_sigmas(N_current).to(self.device)

        # --- Step 4: Sample per-sample timestep indices ---
        # indices: (B,) LongTensor, values in {0, ..., N_current-1}
        # Each sample in the batch gets an independently sampled timestep.
        indices: torch.Tensor = self.schedule.sample_timestep_indices(
            sigmas=sigmas,
            batch_size=batch_size,
        )

        # --- Step 5: Extract per-sample sigma pairs ---
        # sigma_i:  (B,) — lower noise level (closer to data)
        # sigma_i1: (B,) — higher noise level (further from data)
        sigma_i: torch.Tensor = sigmas[indices]        # shape (B,)
        sigma_i1: torch.Tensor = sigmas[indices + 1]   # shape (B,)

        # --- Step 6: Apply initial coupling (IC/OT/GC initial pairing) ---
        # For 'gc' and 'ic' modes: returns (x_star, z) unchanged.
        # For 'ot' mode: permutes z to minimise L2 transport cost.
        # The GC augmentation (endpoint prediction + re-coupling) happens
        # inside compute_gc_loss, not here.
        x_star_coupled: torch.Tensor
        z_coupled: torch.Tensor
        x_star_coupled, z_coupled = self.coupling.get_coupled_pairs(x_star, z)

        # --- Step 7: Compute GC/IC mixed loss (Algorithm 1 core) ---
        loss: torch.Tensor = self.compute_gc_loss(
            x_star=x_star_coupled,
            z=z_coupled,
            sigma_i=sigma_i,
            sigma_i1=sigma_i1,
            mu=self.mu,
        )

        return loss

    # ------------------------------------------------------------------
    # Generator-augmented consistency loss (Algorithm 1 core)
    # ------------------------------------------------------------------

    def compute_gc_loss(
        self,
        x_star: torch.Tensor,
        z: torch.Tensor,
        sigma_i: torch.Tensor,
        sigma_i1: torch.Tensor,
        mu: float,
    ) -> torch.Tensor:
        """Compute the joint IC+GC consistency loss (Algorithm 1, Equation 16).

        Implements the core of the paper's contribution. The combined loss
        in expectation is:
            L_{GC-μ}(θ) = μ · L_GC(θ) + (1 - μ) · L_CT(θ)

        This is achieved via a per-sample Bernoulli mask that selects between
        GC trajectories (m=1, probability μ) and IC trajectories (m=0,
        probability 1-μ) for each sample independently.

        Args:
            x_star: Real data samples of shape ``(B, C, H, W)`` in ``[-1, 1]``.
                For IC trajectories (m=0), this is the endpoint used directly.
            z: Noise samples of shape ``(B, C, H, W)`` from ``N(0, I)``.
                The same noise vector is reused for both IC and GC trajectories,
                which is the key property of the generator-augmented coupling.
            sigma_i: Per-sample lower noise levels of shape ``(B,)``.
                Corresponds to the lower-noise evaluation point in the
                consistency loss (stop-gradient target).
            sigma_i1: Per-sample upper noise levels of shape ``(B,)``.
                Corresponds to the higher-noise evaluation point in the
                consistency loss (gradient flows through this).
                Must satisfy ``sigma_i1 > sigma_i`` element-wise.
            mu: Joint learning parameter μ ∈ [0, 1]. Probability that each
                sample uses a GC trajectory. Config default: 0.5 (CIFAR-10).
                μ=0 → pure IC (iCT-IC baseline).
                μ=1 → pure GC (fails, see Figure 7 and Appendix C.1).

        Returns:
            Scalar loss tensor with gradient graph attached. The gradient
            flows only through ``f_upper = model(x_tilde_i1, sigma_i1)``.
            The ``f_lower = model(x_tilde_i, sigma_i).detach()`` target
            has no gradient.

        Note:
            The EMA apply/restore cycle uses try/finally to guarantee that
            the online model weights are always restored, even if an exception
            occurs during the endpoint prediction forward pass.
        """
        batch_size: int = x_star.shape[0]

        # --- Step (a): Compute IC intermediate point ---
        # x_ti = x★ + σ_i · z  (diffusion process: x_t = x★ + σ_t · z)
        # sigma_i must be reshaped to (B, 1, 1, 1) for broadcasting with (B, C, H, W)
        sigma_i_4d: torch.Tensor = sigma_i.view(batch_size, 1, 1, 1)
        sigma_i1_4d: torch.Tensor = sigma_i1.view(batch_size, 1, 1, 1)

        x_ti: torch.Tensor = x_star + sigma_i_4d * z

        # --- Step (b): Endpoint prediction via EMA model (stop-gradient) ---
        # x_hat = sg(f_θ_ema(x_ti, σ_ti))
        # The EMA model provides a stable, high-quality endpoint predictor.
        # torch.no_grad() implements the sg() operator — no gradients through x_hat.
        # try/finally guarantees online weights are always restored.
        x_hat: torch.Tensor
        self.ema.apply_shadow(self.model)
        try:
            with torch.no_grad():
                x_hat = self.model(x_ti, sigma_i)
        finally:
            self.ema.restore(self.model)

        # x_hat is already detached (torch.no_grad context), but explicit
        # .detach() makes the stop-gradient intent unambiguous.
        x_hat = x_hat.detach()

        # --- Step (c): Per-sample Bernoulli mixing mask ---
        # m ~ Binomial(μ, size=batch_size) — per-sample, not per-batch
        # m=1 (probability μ):   use GC endpoint (model-predicted x_hat)
        # m=0 (probability 1-μ): use IC endpoint (real x_star)
        # device=x_star.device is essential to avoid device mismatch errors
        m: torch.Tensor = torch.bernoulli(
            torch.full((batch_size,), mu, device=x_star.device)
        )
        # Reshape to (B, 1, 1, 1) for broadcasting with image tensors (B, C, H, W)
        m_4d: torch.Tensor = m.view(batch_size, 1, 1, 1)

        # --- Step (d): Mix IC and GC endpoints ---
        # x_hat_mixed = m · x_hat + (1 - m) · x_star
        # When m=1: x_hat_mixed = x_hat  (GC trajectory)
        # When m=0: x_hat_mixed = x_star (IC trajectory)
        x_hat_mixed: torch.Tensor = m_4d * x_hat + (1.0 - m_4d) * x_star

        # --- Steps (e) and (f): Construct GC training pairs ---
        # x̃_{t_i}   = x̂_{t_i} + σ_{t_i}   · z  (Equation 14)
        # x̃_{t_{i+1}} = x̂_{t_i} + σ_{t_{i+1}} · z  (Equation 14)
        # Both use the SAME x_hat_mixed and the SAME z — this is what makes
        # them a valid pair on the same trajectory (differing only in noise level).
        x_tilde_i: torch.Tensor = x_hat_mixed + sigma_i_4d * z
        x_tilde_i1: torch.Tensor = x_hat_mixed + sigma_i1_4d * z

        # --- Step (g): Compute upper output (gradient flows through this) ---
        # f_θ(x̃_{t_{i+1}}, σ_{t_{i+1}}) — higher noise level
        # This is the "online" evaluation; gradients flow through it.
        f_upper: torch.Tensor = self.model(x_tilde_i1, sigma_i1)

        # --- Step (h): Compute lower output (stop-gradient target) ---
        # sg(f_θ(x̃_{t_i}, σ_{t_i})) — lower noise level, detached
        # This is the "target" in the consistency loss; no gradients.
        # IMPORTANT: detach the LOWER-noise output (not the upper).
        # The model is trained to make the higher-noise prediction consistent
        # with the lower-noise prediction (treated as a fixed target).
        f_lower: torch.Tensor = self.model(x_tilde_i, sigma_i).detach()

        # --- Step (i): Loss weighting ---
        # λ(σ_i) = 1 / (σ_{i+1} - σ_i)
        # Shape: (B,) — per-sample weights
        # Large for small σ gaps (low noise) → emphasizes consistency near data
        lam: torch.Tensor = self.schedule.get_lambda(sigma_i, sigma_i1)

        # --- Step (j): Compute final loss ---
        # dist: (B,) — per-sample distances D(f_upper, f_lower)
        # loss: scalar — weighted mean over the batch
        dist: torch.Tensor = self.distance_fn(f_upper, f_lower)
        loss: torch.Tensor = (lam * dist).mean()

        return loss

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        step: int,
        fid: Optional[float],
    ) -> None:
        """Save a complete training checkpoint to disk.

        Saves all state required for full training resumption:
        - Online model state dict
        - EMA state dict (shadow weights)
        - Optimizer state dict (Lion momentum buffers)
        - Current training step
        - Best FID seen so far
        - Config as a dict (for reproducibility)

        Always saves a step-specific checkpoint and overwrites ``latest.pt``.
        If ``fid`` is not None and improves on ``self.best_fid``, also saves
        ``best.pt`` and updates ``self.best_fid``.

        Args:
            step: Current training step (used in the checkpoint filename).
            fid: FID score at this checkpoint, or ``None`` if not yet
                evaluated. Used to track the best checkpoint.
        """
        save_dir: str = str(getattr(self.config, "save_dir", "./checkpoints"))

        # Build config dict for reproducibility logging
        config_dict: Dict[str, Any] = {}
        if hasattr(self.config, "to_dict"):
            config_dict = self.config.to_dict()
        elif hasattr(self.config, "__dict__"):
            config_dict = {
                k: v for k, v in self.config.__dict__.items()
                if not k.startswith("_")
            }

        checkpoint: Dict[str, Any] = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "fid": fid,
            "best_fid": self.best_fid,
            "config": config_dict,
        }

        # Step-specific checkpoint (e.g. checkpoint_0010000.pt)
        step_path: str = os.path.join(
            save_dir, f"checkpoint_{step:07d}.pt"
        )
        torch.save(checkpoint, step_path)

        # Always overwrite latest.pt for easy resumption
        latest_path: str = os.path.join(save_dir, "latest.pt")
        torch.save(checkpoint, latest_path)

        # Save best.pt if FID improved
        if fid is not None and fid < self.best_fid:
            self.best_fid = fid
            best_path: str = os.path.join(save_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(
                f"[Trainer] New best FID: {fid:.4f} at step {step}. "
                f"Saved to '{best_path}'."
            )

    def _load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint and restore all state.

        Restores the online model, EMA shadow weights, optimizer state,
        global step counter, and best FID. After calling this method,
        training can resume from exactly where it left off.

        Args:
            path: Path to the checkpoint file (e.g. ``'./checkpoints/latest.pt'``).

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint file not found: '{path}'. "
                "Check the --checkpoint argument."
            )

        # Load to the training device (handles CPU-saved checkpoints on GPU)
        checkpoint: Dict[str, Any] = torch.load(
            path, map_location=self.device
        )

        # Restore online model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore EMA shadow weights
        self.ema.load_state_dict(checkpoint["ema_state_dict"])

        # Restore optimizer state (Lion momentum buffers)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore training step counter
        self.global_step = int(checkpoint.get("step", 0))

        # Restore best FID tracker
        self.best_fid = float(checkpoint.get("best_fid", float("inf")))

        print(
            f"[Trainer] Loaded checkpoint from '{path}'. "
            f"Resuming from step {self.global_step}. "
            f"Best FID so far: {self.best_fid:.4f}."
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------