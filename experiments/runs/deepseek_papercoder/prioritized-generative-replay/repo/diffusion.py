## diffusion.py
"""
Conditional diffusion model for Prioritized Generative Replay (PGR).

Implements a residual‑MLP denoiser and DDPM‑based generation with
classifier‑free guidance. Handles both state‑based (direct) and pixel‑based
(latent) data.

Training is performed on real transitions from a replay buffer, using a
`RelevanceFunction` to supply scalar conditioning values.
Generation uses top‑k prompting and guidance to produce synthetic
transitions that are pushed into the synthetic replay buffer.
"""

from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import Config
from replay_buffer import ReplayBuffer
from relevance import RelevanceFunction
from utils import get_device


# ------------------------------------------------------------------------------
# 1. Helpers for the denoiser
# ------------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """t: (batch,)  ->  (batch, dim)"""
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    """One residual block of the MLP denoiser.

    Takes:
        - x:     (B, in_dim)
        - t_emb: (B, time_emb_dim)
        - c_emb: (B, cond_emb_dim)

    Returns:
        - out: (B, in_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_emb_dim: int,
        cond_emb_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Projections for conditioning signals
        self.time_proj = nn.Linear(time_emb_dim, out_dim)
        self.cond_proj = nn.Linear(cond_emb_dim, out_dim)

        # Main branch
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.silu = nn.SiLU()

        # Residual connection if dimensions differ
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        # Small learnable scale parameter for conditioning blend
        self.cond_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, t_emb: Tensor, c_emb: Tensor) -> Tensor:
        h = self.silu(self.linear1(x))
        # Scale and shift from time + condition
        scale = self.time_proj(t_emb) + self.cond_proj(c_emb) * self.cond_scale
        h = h * (1 + scale[:, :h.size(1)])  # FiLM‐like scaling
        h = self.linear2(h)
        return h + self.residual(x)


# ------------------------------------------------------------------------------
# 2. Noise prediction network
# ------------------------------------------------------------------------------

class ResidualMLPDenoiser(nn.Module):
    """Residual MLP that predicts the noise injected at time step t.

    Architecture:
        Sinusoidal time embedding -> Linear -> SiLU
        Condition embedding (scalar) -> Linear
        Concatenated with input, passed through residual blocks.

    Parameters
    ----------
    x_dim : int
        Dimensionality of the flattened transition tuple.
    cond_dim : int
        Dimensionality of the conditioning vector (1 for scalar).
    time_emb_dim : int
        Output dimension of the sinusoidal time embedding.
    cond_emb_dim : int
        Embedding dimension for the condition.
    hidden_dim : int
        Width of the hidden layers inside the residual blocks.
    num_blocks : int
        Number of residual blocks.
    """

    def __init__(
        self,
        x_dim: int,
        cond_dim: int,
        time_emb_dim: int = 128,
        cond_emb_dim: int = 32,
        hidden_dim: int = 256,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        self.cond_embed = nn.Linear(cond_dim, cond_emb_dim)

        # Input projection
        self.input_linear = nn.Linear(x_dim, hidden_dim)

        # Residual stack
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, time_emb_dim, cond_emb_dim, hidden_dim)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x: Tensor, t: Tensor, cond_emb: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, x_dim)
            Noisy input.
        t : Tensor, shape (B,)
            Diffusion timestep (ignored for unconditional, but used for CFG).
        cond_emb : Tensor, shape (B, cond_emb_dim)
            Condition embedding (already processed by the main class).

        Returns
        -------
        Tensor, shape (B, x_dim)
            Predicted noise.
        """
        # Time embedding
        t_emb = self.time_mlp(t)

        # Initial linear
        h = self.input_linear(x)

        # Pass through blocks
        for block in self.blocks:
            h = block(h, t_emb, cond_emb)

        return self.output_linear(h)


# ------------------------------------------------------------------------------
# 3. Conditional Diffusion Model
# ------------------------------------------------------------------------------

class ConditionalDiffusion:
    """DDPM‑based conditional diffusion model for synthetic experience replay.

    Supports state‑based and pixel‑based (latent) data.  In pixel mode a
    frozen visual encoder can be stored, but the diffusion itself always
    works on (potentially encoded) vectors.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state representation (raw state or latent).
    action_dim : int
        Dimensionality of the continuous action space.
    config : Config
        Global experiment configuration.
    visual_encoder : Optional[nn.Module], default None
        If provided, denotes pixel‑based mode.  The encoder is kept frozen
        and is *not* used inside the diffusion; it is only stored for
        potential later use by the policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Config,
        visual_encoder: Optional[nn.Module] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = get_device()

        # Transition dimension: [s, a, s', r]
        self.x_dim = state_dim + action_dim + state_dim + 1

        # Keep reference to visual encoder (if any)
        self.visual_encoder = visual_encoder
        if visual_encoder is not None:
            visual_encoder.eval()
            for p in visual_encoder.parameters():
                p.requires_grad = False

        # ----- Diffusion schedule (DDPM) -----
        T = config.diffusion.denoising_steps
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32,
                               device=self.device)
        alphas = 1. - betas
        alphabar = torch.cumprod(alphas, dim=0)
        alphabar_prev = F.pad(alphabar[:-1], [1, 0], value=1.)

        # Pre‑compute often used terms
        self.sqrt_alphabar = alphabar.sqrt()
        self.sqrt_one_minus_alphabar = (1. - alphabar).sqrt()
        self.sqrt_recip_alphas = alphas.sqrt_reciprocal()
        self.posterior_variance = betas * (1. - alphabar_prev) / (1. - alphabar)

        # Save as buffers-like attributes (no grad)
        self.register_buffers({
            'betas': betas,
            'sqrt_alphabar': self.sqrt_alphabar,
            'sqrt_one_minus_alphabar': self.sqrt_one_minus_alphabar,
            'sqrt_recip_alphas': self.sqrt_recip_alphas,
            'posterior_variance': self.posterior_variance,
        })

        # ----- Noise prediction network -----
        denoiser_cfg = dict(
            x_dim=self.x_dim,
            cond_dim=config.diffusion.condition_dim,  # 1
            time_emb_dim=128,
            cond_emb_dim=32,
            hidden_dim=256,
            num_blocks=4,
        )
        self.denoiser = ResidualMLPDenoiser(**denoiser_cfg).to(self.device)

        # ----- Condition handling -----
        self.cond_embed = nn.Linear(config.diffusion.condition_dim,
                                    denoiser_cfg['cond_emb_dim']).to(self.device)
        # A learnable null token for CFG (when condition is dropped)
        self.null_embedding = nn.Parameter(
            torch.randn(1, denoiser_cfg['cond_emb_dim'], device=self.device)
        )

        # ----- Optimizer (Adam) -----
        self.optimizer = torch.optim.Adam(
            list(self.denoiser.parameters()) +
            list(self.cond_embed.parameters()) +
            [self.null_embedding],
            lr=config.diffusion.learning_rate,
        )

        # Store other hyperparams
        self.T = T
        self.uncond_prob = config.diffusion.uncond_prob
        self.guidance_scale = config.diffusion.guidance_scale
        self.prompt_ratio = config.diffusion.prompt_ratio
        self.batch_size = config.diffusion.batch_size

    def register_buffers(self, d: Dict[str, Tensor]) -> None:
        """Store tensors as attributes without gradients."""
        for name, t in d.items():
            setattr(self, name, t.to(self.device))

    # ------------------------------------------------------------------
    # Conditional / unconditional embedding helpers
    # ------------------------------------------------------------------
    def _get_condition_embeddings(self, cond_values: Tensor,
                                  uncond_mask: Optional[Tensor] = None) -> Tensor:
        """Embeds scalar conditions and optionally applies CFG dropout.

        Parameters
        ----------
        cond_values : Tensor, shape (B,)
            Scalar relevance values.
        uncond_mask : Optional[Tensor], shape (B,)
            Boolean mask; True means use null embedding for that sample.

        Returns
        -------
        Tensor, shape (B, cond_emb_dim)
        """
        # Project scalar to embedding
        emb = self.cond_embed(cond_values.unsqueeze(-1))
        if uncond_mask is not None:
            # Expand null embedding to batch
            null_emb = self.null_embedding.expand(cond_values.size(0), -1)
            emb = torch.where(uncond_mask.unsqueeze(-1), null_emb, emb)
        return emb

    # ------------------------------------------------------------------
    # Forward diffusion (noising)
    # ------------------------------------------------------------------
    def _q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Noise a clean data point at timestep t.

        x_start: (B, x_dim)
        t:       (B,) with values in [0, T-1]
        noise:   (B, x_dim) Gaussian noise
        """
        sqrt_alpha_bar_t = self.sqrt_alphabar[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphabar[t].view(-1, 1)
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    # ------------------------------------------------------------------
    # Reverse diffusion (sampling)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _p_sample(self, x: Tensor, t: Tensor, cond_emb: Tensor,
                  uncond_mask: Optional[Tensor] = None) -> Tensor:
        """Single reverse diffusion step with CFG.

        x: (B, x_dim)
        t: (B,)
        cond_emb: (B, cond_emb_dim) – the **conditional** embedding
        uncond_mask: if given, used to also compute unconditional prediction.
        """
        # 1. Conditional prediction
        eps_cond = self.denoiser(x, t, cond_emb)

        if self.guidance_scale != 1.0:
            # 2. Unconditional prediction (always use null_embedding)
            null_emb = self.null_embedding.expand(x.size(0), -1)
            eps_uncond = self.denoiser(x, t, null_emb)
            # CFG combination
            eps = self.guidance_scale * eps_cond + (1.0 - self.guidance_scale) * eps_uncond
        else:
            eps = eps_cond

        # Compute mean for reverse step
        alpha_t = 1. - self.betas[t]
        alpha_bar_t = self.sqrt_alphabar[t] ** 2  # recover alphabar[t]
        alpha_bar_prev_t = alpha_bar_t / alpha_t  # approximate
        # Standard DDPM formulas:
        x0_pred = (x - self.sqrt_one_minus_alphabar[t].view(-1, 1) * eps) / self.sqrt_alphabar[t].view(-1, 1)
        # Clamp x0 if desired (optional)
        # x0_pred = torch.clamp(x0_pred, -1., 1.)

        # Posterior mean
        posterior_mean = (
            self.sqrt_recip_alphas[t].view(-1, 1)
            * (x - (1. - alpha_t) / self.sqrt_one_minus_alphabar[t].view(-1, 1) * eps)
        )
        # Alternative: using x0 formulation, but above is correct DDPM.

        # Sample random noise for non‑zero timesteps
        z = torch.randn_like(x) if t.min() > 0 else torch.zeros_like(x)
        posterior_variance = self.posterior_variance[t].view(-1, 1)
        return posterior_mean + posterior_variance.sqrt() * z

    @torch.no_grad()
    def _p_sample_loop(self, shape: Tuple[int, ...],
                       cond_values: Tensor,
                       guidance_scale: Optional[float] = None) -> Tensor:
        """Full reverse diffusion from random noise to generated samples.

        shape: (batch_size, x_dim)
        cond_values: (batch_size,) – scalar conditions for the batch.
        guidance_scale: if not None, overrides self.guidance_scale for this call.
        """
        if guidance_scale is not None:
            old_scale = self.guidance_scale
            self.guidance_scale = guidance_scale

        batch_size = shape[0]
        device = self.device
        x = torch.randn(shape, device=device)

        # Embed conditions once (no dropout)
        cond_emb = self._get_condition_embeddings(cond_values)

        # Iterate from T-1 down to 0
        for step in reversed(range(self.T)):
            t = torch.full((batch_size,), step, dtype=torch.long, device=device)
            x = self._p_sample(x, t, cond_emb)

        if guidance_scale is not None:
            self.guidance_scale = old_scale

        return x

    # ------------------------------------------------------------------
    # Core training and generation methods
    # ------------------------------------------------------------------
    def train_step(self, real_buffer: ReplayBuffer,
                   relevance_fn: RelevanceFunction) -> float:
        """Perform a single gradient step using a batch from the real buffer.

        The condition values are recomputed inside the step to reflect the
        current relevance function.

        Returns
        -------
        float
            The loss for this batch.
        """
        # Sample a batch from the real replay buffer.
        # We only need states, actions, next_states, rewards.
        states, actions, rewards, next_states, dones, _ = real_buffer.sample(self.batch_size)
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)

        # Build x_start: [s, a, s', r]
        x_start = torch.cat([states, actions, next_states, rewards], dim=-1)

        # Compute up‑to‑date condition values using the relevance function
        cond_values = relevance_fn.compute(states, actions, next_states, rewards)

        # Create uncond dropout mask
        uncond_mask = (torch.rand(cond_values.size(0), device=self.device) < self.uncond_prob)

        # Sample timesteps
        t = torch.randint(0, self.T, (cond_values.size(0),), device=self.device)

        # Sample noise
        noise = torch.randn_like(x_start)

        # Noised input
        x_t = self._q_sample(x_start, t, noise)

        # Get condition embeddings with CFG dropout
        cond_emb = self._get_condition_embeddings(cond_values, uncond_mask=uncond_mask)

        # Predict noise
        pred_noise = self.denoiser(x_t, t, cond_emb)

        # Loss
        loss = F.mse_loss(pred_noise, noise)

        # Backprop & update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        real_buffer: ReplayBuffer,
        relevance_fn: RelevanceFunction,
        guidance_scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate a batch of synthetic transitions guided by the real data.

        The method:
          1. Computes up‑to‑date relevance values for all real transitions.
          2. Selects the top‑k relevance values as prompts.
          3. Samples conditions uniformly from these prompts.
          4. Runs reverse diffusion with CFG to produce transitions.
          5. Splits the generated tensors into (states, actions, next_states,
             rewards) and returns them as numpy arrays on CPU.

        Parameters
        ----------
        num_samples : int
            Number of synthetic transitions to produce.
        real_buffer : ReplayBuffer
            The current real replay buffer.
        relevance_fn : RelevanceFunction
            The current relevance function (possibly learning‑based).
        guidance_scale : Optional[float]
            If provided, overrides the stored `guidance_scale`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            arrays: states, actions, next_states, rewards (each with
            `num_samples` as first dimension).
        """
        # ---- 1. Extract all real data and compute current relevance ----
        # (We rely on all_data returning tensors on CPU; we move to device)
        all_states, all_actions, all_rewards, all_next_states, _, _ = real_buffer.all_data()
        all_states = all_states.to(self.device)
        all_actions = all_actions.to(self.device)
        all_next_states = all_next_states.to(self.device)
        all_rewards = all_rewards.to(self.device)

        # Recompute relevance for every transition
        all_conds = relevance_fn.compute(
            state=all_states,
            action=all_actions,
            next_state=all_next_states,
            reward=all_rewards,
        )  # shape (N,)

        # ---- 2. Top‑k prompt selection ----
        N = all_conds.size(0)
        k = max(1, int(N * self.prompt_ratio))
        # Get indices of top‑k largest values
        topk_vals, _ = torch.topk(all_conds, k, dim=0, largest=True, sorted=False)
        # These are the relevance values we will sample from

        # ---- 3. Sample conditions for generation ----
        # Randomly choose from top‑k values
        idx = torch.randint(0, k, (num_samples,), device=self.device)
        sampled_conds = topk_vals[idx]

        # ---- 4. Run reverse diffusion in minibatches to avoid OOM ----
        gen_states = []
        gen_actions = []
        gen_next_states = []
        gen_rewards = []
        mini_batch = 1024  # configurable, safe for typical GPU memory

        for start in range(0, num_samples, mini_batch):
            end = min(start + mini_batch, num_samples)
            cond_batch = sampled_conds[start:end]
            batch_size_actual = cond_batch.size(0)

            # Generate
            x0 = self._p_sample_loop(
                shape=(batch_size_actual, self.x_dim),
                cond_values=cond_batch,
                guidance_scale=guidance_scale,
            )

            # Split
            s_end = self.state_dim
            a_end = s_end + self.action_dim
            ns_end = a_end + self.state_dim

            gen_states.append(x0[:, :s_end].cpu().numpy())
            gen_actions.append(x0[:, s_end:a_end].cpu().numpy())
            gen_next_states.append(x0[:, a_end:ns_end].cpu().numpy())
            gen_rewards.append(x0[:, ns_end:].cpu().numpy())  # (B, 1)

        # Concatenate all minibatches
        states = np.concatenate(gen_states, axis=0)
        actions = np.concatenate(gen_actions, axis=0)
        next_states = np.concatenate(gen_next_states, axis=0)
        rewards = np.concatenate(gen_rewards, axis=0).squeeze(1)  # flatten to (num_samples,)

        return states, actions, next_states, rewards

