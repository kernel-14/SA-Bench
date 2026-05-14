```python
## diffusion/conditional_diffusion.py
"""Conditional diffusion model for Prioritized Generative Replay (PGR).

Wraps DiffusionModel and DDPMScheduler with classifier-free guidance (CFG)
training and sampling. Transitions (s, a, s', r) are the data; a scalar
relevance score F(τ) from the ICM (or other relevance function) is the
conditioning signal that guides generation toward more learning-relevant
regions of transition space.

Training objective (Eq. 2 from paper, CFG formulation):
    E[|| ε_θ(x^n, n, (1-p)·y + p·∅) ||²₂]

Sampling (CFG at inference):
    ε = ω · ε_θ(x^n, n, y) + (1-ω) · ε_θ(x^n, n, ∅)

Config references (config.yaml):
    diffusion.hidden_dim:          256    # residual MLP hidden width
    diffusion.num_layers:          4      # number of residual blocks
    diffusion.time_emb_dim:        128    # sinusoidal time embedding dimension
    diffusion.cond_emb_dim:        128    # condition embedding dimension
    diffusion.num_timesteps:       100    # DDPM denoising steps
    diffusion.beta_start:          1e-4   # linear beta schedule start
    diffusion.beta_end:            0.02   # linear beta schedule end
    diffusion.lr:                  3e-4   # Adam learning rate
    diffusion.p_uncond:            0.25   # CFG condition dropout probability
    diffusion.guidance_scale:      3.0    # CFG guidance scale ω at sampling time
"""

import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from diffusion.ddpm import DDPMScheduler
from diffusion.model import DiffusionModel
from utils.normalizer import Normalizer


class ConditionalDiffusion:
    """Conditional diffusion model with classifier-free guidance for PGR.

    Implements the generative replay buffer G(τ | F(τ)) described in
    Section 4.3 of the paper. Extends the unconditional SYNTHER baseline
    (Lu et al., 2024) by adding a CFG conditioning pathway for the scalar
    relevance score F(τ), enabling guided generation toward high-relevance
    regions of transition space.

    The model is trained end-to-end with the policy in the inner loop of
    Algorithm 1 (every inner_loop_freq=10000 environment steps). At each
    inner loop call, the diffusion model is retrained from scratch on all
    transitions in D_real, then used to fill D_syn with conditionally
    generated transitions.

    Attributes:
        input_dim: Dimension of the flattened transition tuple (s, a, s', r).
            Computed as 2 * obs_dim + action_dim + 1 by the caller.
        hidden_dim: Width of the hidden representation in all residual blocks.
        num_layers: Number of residual blocks in the denoising network.
        num_timesteps: Total number of DDPM denoising steps T.
        p_uncond: Probability of dropping the condition during training (CFG).
            Corresponds to config.diffusion.p_uncond (default 0.25, paper Sec. 5).
        guidance_scale: CFG guidance scale ω at sampling time.
            Corresponds to config.diffusion.guidance_scale (default 3.0).
        device: PyTorch device string for all tensors and model parameters.
        model: DiffusionModel — the residual MLP denoising network ε_θ.
        scheduler: DDPMScheduler — DDPM noise schedule utilities.
        optimizer: Adam optimizer over model parameters.
        normalizer: Normalizer for transition tuple standardization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_emb_dim: int = 128,
        cond_emb_dim: int = 128,
        num_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        p_uncond: float = 0.25,
        guidance_scale: float = 3.0,
        lr: float = 3e-4,
        device: str = "cuda",
    ) -> None:
        """Initialises the conditional diffusion model.

        Instantiates the DiffusionModel, DDPMScheduler, Adam optimizer, and
        Normalizer. The Normalizer is NOT fit here — it is fit lazily via
        fit_normalizer() on the first inner loop call (Shared Knowledge point 6).

        Args:
            input_dim: Dimension of the flattened transition tuple (s, a, s', r).
                Computed as 2 * obs_dim + action_dim + 1 by PGRTrainer.
                For state-based quadruped-walk: 2*67 + 12 + 1 = 147.
                For pixel tasks with DRQv2 (feature_dim=50, action_dim=6):
                2*50 + 6 + 1 = 107.
            hidden_dim: Width of the hidden representation. Corresponds to
                config.diffusion.hidden_dim (default 256).
            num_layers: Number of residual blocks. Corresponds to
                config.diffusion.num_layers (default 4).
            time_emb_dim: Sinusoidal timestep embedding dimension. Corresponds
                to config.diffusion.time_emb_dim (default 128). Must be even.
            cond_emb_dim: Condition (relevance score) embedding dimension.
                Corresponds to config.diffusion.cond_emb_dim (default 128).
            num_timesteps: Total DDPM denoising steps T. Corresponds to
                config.diffusion.num_timesteps (default 100).
            beta_start: Linear beta schedule start. Corresponds to
                config.diffusion.beta_start (default 1e-4).
            beta_end: Linear beta schedule end. Corresponds to
                config.diffusion.beta_end (default 0.02).
            p_uncond: CFG condition dropout probability during training.
                Corresponds to config.diffusion.p_uncond (default 0.25,
                paper Section 5: "we randomly discard the scalar given by
                our relevance function F with probability 0.25").
            guidance_scale: CFG guidance scale ω at sampling time. Corresponds
                to config.diffusion.guidance_scale (default 3.0; not stated
                in paper — recommended sweep: {1.5, 3.0, 5.0, 7.5}).
            lr: Adam optimizer learning rate. Corresponds to
                config.diffusion.lr (default 3e-4).
            device: PyTorch device string. Corresponds to
                config.hardware.device (default "cuda").
        """
        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.num_timesteps: int = num_timesteps
        self.p_uncond: float = p_uncond
        self.guidance_scale: float = guidance_scale
        self.device: str = device

        # ── Denoising network ε_θ ─────────────────────────────────────────────
        # Residual MLP that predicts the noise added to a noisy transition.
        # Architecture matches SYNTHER exactly, extended with CFG conditioning.
        self.model: DiffusionModel = DiffusionModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_emb_dim=time_emb_dim,
            cond_emb_dim=cond_emb_dim,
        ).to(device)

        # ── DDPM noise schedule ───────────────────────────────────────────────
        # Precomputes all schedule tensors (betas, alphas, cumulative products,
        # posterior variance) on the target device.
        self.scheduler: DDPMScheduler = DDPMScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
        )

        # ── Adam optimizer ────────────────────────────────────────────────────
        # Optimizes only the DiffusionModel parameters — the normalizer has
        # no learnable parameters, and the scheduler is parameter-free.
        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

        # ── Transition normalizer ─────────────────────────────────────────────
        # Fit lazily on the first inner loop call via fit_normalizer().
        # Per Shared Knowledge point 6: fit once, then frozen for all subsequent
        # inner loop calls to prevent distribution shift in normalized space.
        self.normalizer: Normalizer = Normalizer(device=device)

        # ── Internal state ────────────────────────────────────────────────────
        # Tracks whether fit_normalizer() has been called. Guards train_step()
        # against being called before the normalizer is ready.
        self._normalizer_fitted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_normalizer(self, data: Dict[str, torch.Tensor]) -> None:
        """Fits the transition normalizer on all data in D_real.

        Called by PGRTrainer._train_diffusion() only on the first inner loop
        call (step == inner_loop_freq). Subsequent calls are no-ops because
        Normalizer.fit() is idempotent after the first successful call.

        The concatenation order [s, a, s', r] is fixed and must be identical
        to the order used in train_step() and expected by generate()'s caller.
        This is the "shared knowledge" contract across all PGR modules.

        Args:
            data: Transition dict from ReplayBuffer.get_all_as_tensor().
                Contains float32 tensors on self.device with keys:
                    'observations':      (N, obs_dim)
                    'actions':           (N, action_dim)
                    'next_observations': (N, obs_dim)
                    'rewards':           (N, 1)
                    'dones':             (N, 1)
                N is the number of valid transitions in D_real at the time
                of the first inner loop call (up to real_buffer_size=1M).
        """
        # Concatenate transition components in fixed order: [s, a, s', r].
        # Shape: (N, input_dim) where input_dim = obs_dim + action_dim + obs_dim + 1.
        # 'dones' is excluded — the diffusion model generates (s, a, s', r) tuples,
        # not done flags. Done flags are handled separately by PGRTrainer.
        concatenated: torch.Tensor = torch.cat(
            [
                data["observations"].to(device=self.device, dtype=torch.float32),
                data["actions"].to(device=self.device, dtype=torch.float32),
                data["next_observations"].to(device=self.device, dtype=torch.float32),
                data["rewards"].to(device=self.device, dtype=torch.float32),
            ],
            dim=-1,
        )  # (N, input_dim)

        # Fit the normalizer — idempotent after first call (Normalizer.fit()
        # returns immediately if self.fitted is already True).
        self.normalizer.fit(concatenated)
        self._normalizer_fitted = True

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        relevance_scores: torch.Tensor,
    ) -> float:
        """Performs one gradient step of the CFG diffusion training objective.

        Implements the training objective from Eq. 2 of the paper:
            E[|| ε_θ(x^n, n, (1-p)·y + p·∅) ||²₂]

        where p ~ Bernoulli(p_uncond=0.25) determines per-sample whether the
        null condition ∅ or the real relevance score y is used.

        The full pipeline per step:
            1. Build x0 = cat(s, a, s', r) and normalize
            2. Sample t ~ Uniform(0, T-1) and ε ~ N(0, I)
            3. Compute x_t = add_noise(x0, ε, t)
            4. Sample CFG dropout mask p ~ Bernoulli(p_uncond)
            5. Normalize relevance scores to [0, 1]
            6. Predict noise: ε_pred = model(x_t, t, y, drop_mask)
            7. Compute MSE loss and update

        Args:
            batch: Transition dict sampled from D_real by PGRTrainer.
                Contains float32 tensors on self.device with keys:
                    'observations':      (B, obs_dim)
                    'actions':           (B, action_dim)
                    'next_observations': (B, obs_dim)
                    'rewards':           (B, 1)
                    'dones':             (B, 1)
                Batch size B corresponds to config.sampling.batch_size (default 256).
            relevance_scores: Per-transition ICM curiosity scores from
                ICMRelevance.score(). Float32 tensor of shape (B, 1) or (B,).
                Raw unnormalized values — normalized to [0, 1] inside this method
                using the batch's min/max (Shared Knowledge point 3).

        Returns:
            Scalar training loss as a Python float (via loss.item()).
            Logged by PGRTrainer as "diffusion/train_loss".

        Raises:
            RuntimeError: If fit_normalizer() has not been called before
                the first train_step() call.
        """
        if not self._normalizer_fitted:
            raise RuntimeError(
                "ConditionalDiffusion.train_step() called before fit_normalizer(). "
                "Call fit_normalizer() with D_real data before starting diffusion training."
            )

        # ── Step 1: Build and normalize x0 ───────────────────────────────────
        # Concatenate transition components in fixed order: [s, a, s', r].
        # Shape: (B, input_dim)
        obs: torch.Tensor = batch["observations"].to(
            device=self.device, dtype=torch.float32
        )
        actions: torch.Tensor = batch["actions"].to(
            device=self.device, dtype=torch.float32
        )
        next_obs: torch.Tensor = batch["next_observations"].to(
            device=self.device, dtype=torch.float32
        )
        rewards: torch.Tensor = batch["rewards"].to(
            device=self.device, dtype=torch.float32
        )

        # Ensure rewards has shape (B, 1) for consistent concatenation.
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)

        x0: torch.Tensor = torch.cat(
            [obs, actions, next_obs, rewards], dim=-1
        )  # (B, input_dim)

        # Normalize to zero mean, unit variance using fitted statistics.
        x0_norm: torch.Tensor = self.normalizer.normalize(x0)  # (B, input_dim)

        batch_size: int = x0_norm.shape[0]

        # ── Step 2: Sample noise and timesteps ────────────────────────────────
        # t ~ Uniform(0, num_timesteps-1), shape: (B,)
        t: torch.Tensor = self.scheduler.sample_timesteps(batch_size)

        # ε ~ N(0, I), same shape as x0_norm: (B, input_dim)
        noise: torch.Tensor = torch.randn_like(x0_norm)

        # ── Step 3: Forward diffusion (add noise) ─────────────────────────────
        # x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε
        x_t: torch.Tensor = self.scheduler.add_noise(x0_norm, noise, t)  # (B, input_dim)

        # ── Step 4: Normalize relevance scores to [0, 1] ─────────────────────
        # Raw ICM scores can have arbitrary magnitudes (squared prediction errors).
        # Normalize per-batch using min/max to keep the conditioning signal stable.
        # Guard against zero variance (all scores identical) with + 1e-8.
        scores: torch.Tensor = relevance_scores.to(
            device=self.device, dtype=torch.float32
        ).view(batch_size, 1)  # (B, 1)

        scores_min: torch.Tensor = scores.min()
        scores_max: torch.Tensor = scores.max()
        scores_norm: torch.Tensor = (scores - scores_min) / (
            scores_max - scores_min + 1e-8
        )  # (B, 1), values in [0, 1]

        # ── Step 5: CFG condition dropout mask ───────────────────────────────
        # Sample Bernoulli mask: True (drop condition) with probability p_uncond.
        # Shape: (B,) bool tensor. True → use null condition ∅.
        #                          False → use real relevance score y.
        drop_mask: torch.Tensor = torch.bernoulli(
            torch.full((batch_size,), self.p_uncond, device=self.device)
        ).bool()  # (B,)

        # ── Step 6: Forward pass through denoiser with per-sample CFG ─────────
        # The DiffusionModel.forward accepts use_null_cond as a bool flag.
        # For per-sample CFG dropout, we implement it by:
        #   - Running the model with use_null_cond=False (conditional pass)
        #   - Running the model with use_null_cond=True (unconditional pass)
        #   - Selecting per-sample based on drop_mask
        # This is equivalent to the paper's (1-p)·y + p·∅ formulation.
        eps_cond: torch.Tensor = self.model.forward(
            x_t, t, scores_norm, use_null_cond=False
        )  # (B, input_dim)

        eps_uncond: torch.Tensor = self.model.forward(
            x_t, t, scores_norm, use_null_cond=True
        )  # (B, input_dim)

        # Select per-sample: use eps_uncond where drop_mask is True.
        # drop_mask shape: (B,) → expand to (B, input_dim) for selection.
        drop_mask_expanded: torch.Tensor = drop_mask.unsqueeze(-1).expand_as(
            eps_cond
        )  # (B, input_dim)
        predicted_noise: torch.Tensor = torch.where(
            drop_mask_expanded, eps_uncond, eps_cond
        )  # (B, input_dim)

        # ── Step 7: Compute MSE loss and update ──────────────────────────────
        # Standard DDPM noise prediction loss: E[||ε_pred - ε||²]
        loss: torch.Tensor = F.mse_loss(predicted_noise, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def generate(
        self,
        num_samples: int,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """Generates synthetic transitions conditioned on relevance scores.

        Public API called by PGRTrainer._generate_synthetic_data() to fill
        D_syn with conditionally generated transitions. Runs the full reverse
        diffusion chain from x_T ~ N(0, I) to x_0, then denormalizes.

        The caller (PGRTrainer) is responsible for:
        1. Providing conditions sampled from the top-k D_real transitions'
           relevance scores (the "prompting strategy" from Peebles et al., 2022).
        2. Splitting the returned tensor into (obs, action, next_obs, reward)
           components using the known dimension boundaries.
        3. Adding the split components to D_syn via ReplayBuffer.add().

        Args:
            num_samples: Number of synthetic transitions to generate.
                Corresponds to config.diffusion.generation_batch_size (default 1024)
                per call from PGRTrainer, which calls generate() in batches
                until D_syn is full (syn_capacity=1M transitions).
            conditions: Relevance score conditions for generation. Float32
                tensor of shape (num_samples, 1) or (num_samples,) containing
                normalized [0, 1] relevance scores sampled from the top-k
                D_real transitions. Produced by PGRTrainer from the output of
                ReplayBuffer.sample_top_k(top_k_fraction=0.1).

        Returns:
            Float32 tensor of shape (num_samples, input_dim) in the original
            (unnormalized) transition scale. The columns correspond to:
                [:obs_dim]                    → s (current observation)
                [obs_dim:obs_dim+action_dim]  → a (action)
                [obs_dim+action_dim:-1]       → s' (next observation)
                [-1:]                         → r (reward)
            The caller splits this tensor using the known obs_dim and action_dim.
        """
        # Set model to eval mode for generation (disables dropout if any).
        self.model.eval()

        with torch.no_grad():
            # Normalize conditions to [0, 1] range.
            # conditions may come pre-normalized from PGRTrainer, but we
            # apply normalization defensively to ensure stable conditioning.
            cond: torch.Tensor = conditions.to(
                device=self.device, dtype=torch.float32
            ).view(num_samples, 1)  # (num_samples, 1)

            cond_min: torch.Tensor = cond.min()
            cond_max: torch.Tensor = cond.max()
            cond_norm: torch.Tensor = (cond - cond_min) / (
                cond_max - cond_min + 1e-8
            )  # (num_samples, 1), values in [0, 1]

            # Run full reverse diffusion chain.
            x_normalized: torch.Tensor = self._ddpm_reverse(
                num_samples, cond_norm
            )  # (num_samples, input_dim)

            # Denormalize back to original transition scale.
            x: torch.Tensor = self.normalizer.denormalize(
                x_normalized
            )  # (num_samples, input_dim)

        # Restore model to training mode.
        self.model.train()

        return x

    def save(self, path: str) -> None:
        """Saves the model, optimizer, and normalizer state to a checkpoint file.

        Called by PGRTrainer.save_checkpoint() to persist the diffusion model
        state alongside the policy and relevance function checkpoints.

        Args:
            path: Full file path for the checkpoint (e.g.
                "checkpoints/quadruped_walk_curiosity_seed0/diffusion.pt").
                Parent directories are created if they do not exist.
        """
        # Create parent directories if needed.
        parent_dir: str = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Build checkpoint dict with all state needed for full restoration.
        checkpoint: dict = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "normalizer_fitted": self._normalizer_fitted,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_timesteps": self.num_timesteps,
            "p_uncond": self.p_uncond,
            "guidance_scale": self.guidance_scale,
            "device": self.device,
        }

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Restores the model, optimizer, and normalizer state from a checkpoint.

        Called by PGRTrainer.load_checkpoint() to resume training from a
        previously saved state. The model architecture must match the saved
        checkpoint (same input_dim, hidden_dim, num_layers, etc.).

        Args:
            path: Full file path to the checkpoint saved by save().

        Raises:
            FileNotFoundError: If the checkpoint file does not exist at path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ConditionalDiffusion checkpoint not found at '{path}'. "
                "Ensure the path is correct and the checkpoint was saved."
            )

        # Load checkpoint to the target device.
        checkpoint: dict = torch.load(path, map_location=self.device)

        # Restore model weights.
        self.model.load_state_dict(checkpoint["model_state"])

        # Restore optimizer state (learning rate, momentum buffers, etc.).
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Restore normalizer state (mean, std, fitted flag).
        if "normalizer" in checkpoint:
            self.normalizer.load_state_dict(checkpoint["normalizer"])

        # Restore the normalizer fitted flag.
        self._normalizer_fitted = bool(
            checkpoint.get("normalizer_fitted", self.normalizer.fitted)
        )

    # ── Private methods ───────────────────────────────────────────────────────

    def _cfg_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Performs one CFG-guided reverse denoising step.

        Implements the paper's guidance formula (Shared Knowledge point 5):
            ε = ω · ε_θ(x^n, n, y) + (1-ω) · ε_θ(x^n, n, ∅)

        Note: This is the paper's exact formulation, NOT the more common
        alternative `ε_uncond + ω * (ε_cond - ε_uncond)`. These are
        mathematically equivalent only when ω is interpreted differently.
        We use the paper's formula directly with guidance_scale=ω.

        With guidance_scale=3.0 (config default):
            ε = 3.0 * ε_cond + (-2.0) * ε_uncond

        This is a strong guidance signal that pushes generations toward
        high-relevance regions. The negative weight on ε_uncond actively
        steers away from the unconditional (average) distribution.

        Args:
            x_t: Current noisy sample at timestep t. Float32 tensor of shape
                (B, input_dim). Must be on self.device.
            t: Diffusion timestep indices. Long tensor of shape (B,) with
                values in [0, num_timesteps-1]. All elements should be the
                same value (same timestep for all samples in the batch) during
                the reverse chain in _ddpm_reverse().
            cond: Normalized relevance score conditions. Float32 tensor of
                shape (B, 1) with values in [0, 1]. Passed to the conditional
                model pass; ignored by the unconditional pass (use_null_cond=True).

        Returns:
            Float32 tensor of shape (B, input_dim) — the denoised sample
            x_{t-1} after one reverse step using the CFG-combined noise
            prediction.
        """
        # ── Conditional noise prediction ε_θ(x_t, t, y) ─────────────────────
        # use_null_cond=False: uses the real relevance score condition y.
        eps_cond: torch.Tensor = self.model.forward(
            x_t, t, cond, use_null_cond=False
        )  # (B, input_dim)

        # ── Unconditional noise prediction ε_θ(x_t, t, ∅) ───────────────────
        # use_null_cond=True: uses the learned null condition token ∅.
        # The cond argument is passed but ignored internally by DiffusionModel
        # when use_null_cond=True (it uses null_cond_emb instead).
        eps_uncond: torch.Tensor = self.model.forward(
            x_t, t, cond, use_null_cond=True
        )  # (B, input_dim)

        # ── CFG combination (paper's exact formula) ───────────────────────────
        # ε = ω · ε_cond + (1-ω) · ε_uncond
        # With ω=3.0: ε = 3.0 * ε_cond + (-2.0) * ε_uncond
        eps: torch.Tensor = (
            self.guidance_scale * eps_cond
            + (1.0 - self.guidance_scale) * eps_uncond
        )  # (B, input_dim)

        # ── One reverse DDPM step ─────────────────────────────────────────────
        # Computes x_{t-1} from x_t and the CFG-combined noise prediction.
        x_prev: torch.Tensor = self.scheduler.step(x_t, eps, t)  # (B, input_dim)

        return x_prev

    def _ddpm_reverse(
        self,
        num_samples: int,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """Runs the full reverse diffusion chain from x_T ~ N(0, I) to x_0.

        Iterates from t = num_timesteps-1 down to t = 0, calling _cfg_sample
        at each step to progressively denoise the initial Gaussian noise into
        a sample from the learned conditional distribution p_θ(x_0 | c).

        The entire loop runs under torch.no_grad() (enforced by the calling
        context in generate()) for memory efficiency — no gradients are needed
        during generation.

        Args:
            num_samples: Number of samples to generate in parallel.
                Corresponds to config.diffusion.generation_batch_size (default 1024).
            conditions: Normalized relevance score conditions. Float32 tensor
                of shape (num_samples, 1) with values in [0, 1]. One condition
                per generated sample, sampled from the top-k D_real transitions'
                relevance scores (prompting strategy, paper Sec. 4.3).

        Returns:
            Float32 tensor of shape (num_samples, input_dim) — the generated
            normalized transition tuples x_0. Still in normalized space;
            denormalization is applied by the calling generate() method.
        """
        # Initialize x_T ~ N(0, I) — pure Gaussian noise.
        # Shape: (num_samples, input_dim)
        x: torch.Tensor = torch.randn(
            num_samples, self.input_dim, device=self.device, dtype=torch.float32
        )

        # Iterate from t = T-1 down to t = 0 (0-indexed, inclusive).
        # At each step, _cfg_sample computes x_{t-1} from x_t using CFG.
        for t_val in reversed(range(self.num_timesteps)):
            # Create a constant timestep tensor for the entire batch.
            # All samples in the batch share the same timestep during generation.
            t_batch: torch.Tensor = torch.full(
                (num_samples,),
                t_val,
                dtype=torch.long,
                device=self.device,
            