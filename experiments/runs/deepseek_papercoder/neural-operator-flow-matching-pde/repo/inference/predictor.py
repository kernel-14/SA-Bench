## inference/predictor.py
"""
Predictor module for the Flow Marching Transformer (FMT) + P2VAE model.

This module provides the `Predictor` class that performs:
  - Deterministic next‑frame prediction.
  - Stochastic ensemble generation.
  - Long‑horizon autoregressive rollouts.

All inference hyper‑parameters are read from the `config.yaml` file
(section `inference`). The FMT and VAE models are expected to be
loaded already and set to evaluation mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from models.fmt import FMT
from models.p2vae import P2VAE
from utils.data_utils import downsample_token_grid


class Predictor:
    """
    Inference predictor for the generative PDE foundation model.

    Args:
        fmt: A trained Flow Marching Transformer instance.
        vae: A trained P2VAE instance (the encoder will be used frozen).
        config: Full configuration dictionary (e.g., loaded from config.yaml).
                The `inference` sub‑dictionary is required with keys:
                - ode_steps (int), dt (float), deterministic_k (float),
                  stochastic_k (list[float]), ensemble_size (int).
    """

    def __init__(
        self,
        fmt: FMT,
        vae: P2VAE,
        config: Dict[str, Any],
    ) -> None:
        self.fmt = fmt
        self.vae = vae

        # Set models to evaluation mode
        self.fmt.eval()
        self.vae.eval()

        # Extract inference parameters
        inf_cfg = config["inference"]
        self.ode_steps: int = inf_cfg["ode_steps"]
        self.dt: float = inf_cfg["dt"]
        self.deterministic_k: float = inf_cfg["deterministic_k"]
        self.stochastic_k: List[float] = inf_cfg["stochastic_k"]
        self.ensemble_size: int = inf_cfg["ensemble_size"]

        # Store useful dimensions from FMT
        self.latent_dim: int = self.fmt.latent_dim
        self.pyramid_factors: List[int] = self.fmt.pyramid_factors
        self.token_counts: List[int] = self.fmt.token_counts
        self.dim: int = self.fmt.dim

        # For convenience, cache the device of the FMT (assumes all params on same device)
        self.device = next(self.fmt.parameters()).device

    # ------------------------------------------------------------------
    # Private helpers for tokenisation and conditioning
    # ------------------------------------------------------------------

    def _tokenize_frame(self, y: Tensor, frame_idx: int) -> Tensor:
        """
        Downsamples a latent frame according to the pyramid factor,
        projects to transformer dimension, and adds positional embeddings.

        Args:
            y: (B, latent_dim, 16, 16) latent frame.
            frame_idx: index 0..3 determining the downsampling factor.

        Returns:
            tokens: (B, N_i, dim) token sequence.
        """
        factor = self.pyramid_factors[frame_idx]
        target_h = 16 // factor
        target_w = 16 // factor
        y_down = downsample_token_grid(y, target_h, target_w)  # (B, C, H, W)
        B, C, H, W = y_down.shape
        tokens = y_down.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)
        tokens = self.fmt.proj_in(tokens)  # (B, N, dim)
        tokens = tokens + self.fmt.pos_embeddings[frame_idx]  # broadcast
        return tokens

    def _build_condition(self, t: Tensor, h: Tensor) -> Tensor:
        """
        Builds a per‑frame conditioning vector by combining a time embedding
        and a hidden state (from diffusion forcing).

        Args:
            t: (B,) float tensor of flow times.
            h: (B, dim) hidden state.

        Returns:
            cond: (B, dim) conditioning vector.
        """
        t_emb = self.fmt.time_embed(t)  # (B, dim)
        combined = torch.cat([t_emb, h], dim=-1)  # (B, 2*dim)
        cond = self.fmt.cond_proj(combined)  # (B, dim)
        return cond

    def _expand_condition(self, cond: Tensor, n_tokens: int) -> Tensor:
        """
        Repeats a per‑frame condition vector for every token of that frame.

        Args:
            cond: (B, dim) per‑frame condition.
            n_tokens: number of tokens for this frame.

        Returns:
            tokens_cond: (B, n_tokens, dim)
        """
        return cond.unsqueeze(1).expand(-1, n_tokens, -1)

    # ------------------------------------------------------------------
    # Core ODE integration in latent space (used by prediction and rollout)
    # ------------------------------------------------------------------

    def _predict_next_latent(
        self,
        latents: List[Tensor],
        k: float,
    ) -> Tensor:
        """
        Given four consecutive clean latent frames, integrate the ODE
        for the last frame from t=0 to t=1 and return the predicted
        latent of the next time step.

        Args:
            latents: list of 4 tensors, each (B, C, 16, 16).
            k: bridge parameter (0 = fully stochastic, 1 = deterministic).

        Returns:
            y_next: (B, C, 16, 16) predicted latent.
        """
        B = latents[0].size(0)

        # --- Precompute tokens and hidden states for history frames (0,1,2) ---
        tokens_hist = []
        for i in range(3):
            tokens_hist.append(self._tokenize_frame(latents[i], i))

        # Unroll diffusion forcing to obtain hidden states after each history frame
        h_gru_list = self.fmt.diff_forcing.forward_sequence(tokens_hist)  # length 3

        # Prepare conditioning hidden states for each frame index:
        # frame 0 -> zeros, frame 1 -> h_gru_list[0], frame 2 -> h_gru_list[1], frame 3 -> h_gru_list[2]
        h_init = torch.zeros(B, self.dim, device=self.device)
        h_cond = [h_init, h_gru_list[0], h_gru_list[1], h_gru_list[2]]

        # Precompute conditioning vectors for frames 0,1,2 (t=0)
        cond_hist = []
        t_zero = torch.zeros(B, device=self.device)
        for i in range(3):
            cond_vec = self._build_condition(t_zero, h_cond[i])        # (B, dim)
            cond_tokens = self._expand_condition(cond_vec, self.token_counts[i])  # (B, N_i, dim)
            cond_hist.append(cond_tokens)

        # Token sequences for frames 0,1,2 are fixed
        # (we already have tokens_hist; they will be concatenated with frame 3 tokens later)

        # Initial latent for frame 3
        if k >= 1.0:
            y3_cur = latents[3].clone()
        else:
            # location‑scale kernel at t=0: μ = k * y3, σ = 1 - k
            sigma = 1.0 - k
            z = torch.randn_like(latents[3])
            y3_cur = k * latents[3] + sigma * z

        # --- ODE integration ---
        for step in range(self.ode_steps):
            current_t = step * self.dt
            t_tensor = torch.full((B,), current_t, device=self.device)

            # Tokenise the current noisy frame 3
            tokens3 = self._tokenize_frame(y3_cur, 3)  # (B, N_3, dim)

            # Conditioning for frame 3
            cond3_vec = self._build_condition(t_tensor, h_cond[3])   # (B, dim)
            cond3_tokens = self._expand_condition(cond3_vec, self.token_counts[3])

            # Build full sequence
            all_tokens = torch.cat(tokens_hist + [tokens3], dim=1)  # (B, total, dim)
            all_cond = torch.cat(cond_hist + [cond3_tokens], dim=1) # (B, total, dim)

            # Transformer forward
            out_tokens = self.fmt.forward_tokens(all_tokens, all_cond)  # (B, total, dim)

            # Extract frame 3 output tokens
            offset = sum(self.token_counts[:3])
            n3 = self.token_counts[3]
            vel_tokens = out_tokens[:, offset:offset+n3, :]  # (B, N_3, dim)
            vel = self.fmt.vel_head(vel_tokens)               # (B, N_3, latent_dim)

            # Reshape velocity to latent grid (C, H, W)
            target_h = 16 // self.pyramid_factors[3]  # factor 1 -> 16
            target_w = target_h
            vel_grid = vel.reshape(B, target_h, target_w, self.latent_dim)
            vel_grid = vel_grid.permute(0, 3, 1, 2)   # (B, C, 16, 16)

            # Euler step
            y3_cur = y3_cur + self.dt * vel_grid

        return y3_cur

    # ------------------------------------------------------------------
    # Public prediction / generation methods
    # ------------------------------------------------------------------

    def predict_next(self, x_seq: Tensor, k: Optional[float] = None) -> Tensor:
        """
        Predict the next physical frame from four consecutive frames.

        Args:
            x_seq: (B, 4, 3, 128, 128) physical fields.
            k: bridge parameter; if None, uses `self.deterministic_k`.
               k=1 → deterministic, k<1 → stochastic.

        Returns:
            x_next: (B, 3, 128, 128) predicted next physical field.
        """
        if k is None:
            k = self.deterministic_k

        # Encode physical frames into latent space (deterministic mean)
        B = x_seq.size(0)
        frames_flat = x_seq.reshape(-1, 3, 128, 128)  # (B*4, 3, 128, 128)
        with torch.no_grad():
            mu, _ = self.vae.encode(frames_flat)
        latents = mu.reshape(B, 4, self.latent_dim, 16, 16)  # (B, 4, C, 16, 16)

        # Separate into list of tensors for the private method
        latent_list = [latents[:, i] for i in range(4)]

        y_next = self._predict_next_latent(latent_list, k)  # (B, C, 16, 16)

        # Decode to physical field
        with torch.no_grad():
            x_next = self.vae.decode(y_next)
        return x_next

    def generate_ensemble(
        self,
        x_seq: Tensor,
        k: float,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        """
        Generate an ensemble of possible next frames for a given window.

        Args:
            x_seq: (1, 4, 3, 128, 128) single initial window.
            k: stochasticity level (e.g., 0.3).
            batch_size: number of ensemble members; if None, uses
                        `self.ensemble_size`.

        Returns:
            ensemble: (batch_size, 3, 128, 128) predicted physical fields.
        """
        if batch_size is None:
            batch_size = self.ensemble_size

        # Replicate the input along the batch dimension
        x_seq_batch = x_seq.repeat(batch_size, 1, 1, 1, 1)  # (B, 4, 3, 128, 128)

        # Use predict_next with the given k (each sample will receive independent noise)
        # Note: _predict_next_latent internally draws fresh noise for each sample when k<1,
        # so the resulting batch will be diverse.
        return self.predict_next(x_seq_batch, k)

    def autoregressive_rollout(
        self,
        x_init: Tensor,
        num_steps: int,
        k: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform an autoregressive rollout of `num_steps` future frames.

        Args:
            x_init: (B, 4, 3, 128, 128) initial four‑frame window.
            num_steps: how many future frames to generate.
            k: bridge parameter (default 1.0 = deterministic).

        Returns:
            pred_latents: (B, num_steps, latent_dim, 16, 16) latent predictions.
            pred_frames:  (B, num_steps, 3, 128, 128) decoded physical frames.
        """
        B = x_init.size(0)

        # Encode initial window to latents
        with torch.no_grad():
            init_flat = x_init.reshape(-1, 3, 128, 128)
            mu_init, _ = self.vae.encode(init_flat)
        latents_init = mu_init.reshape(B, 4, self.latent_dim, 16, 16)

        # Keep a sliding window of four latents
        latent_window = [latents_init[:, i] for i in range(4)]

        pred_latents_list = []
        for _ in range(num_steps):
            y_next = self._predict_next_latent(latent_window, k)
            pred_latents_list.append(y_next)

            # Slide window: drop oldest, append new
            latent_window = latent_window[1:] + [y_next]

        # Stack predicted latents: (B, num_steps, C, 16, 16)
        latents_stack = torch.stack(pred_latents_list, dim=1)

        # Decode all predicted latents in one batch
        latents_flat = latents_stack.reshape(B * num_steps, self.latent_dim, 16, 16)
        with torch.no_grad():
            frames_flat = self.vae.decode(latents_flat)
        frames_stack = frames_flat.reshape(B, num_steps, 3, 128, 128)

        return latents_stack, frames_stack

