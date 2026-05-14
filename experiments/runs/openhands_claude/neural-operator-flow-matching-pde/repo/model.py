"""
Top-level model definitions:
  - P2VAE: Pretrained Physics Variational Autoencoder
  - FMT:   Flow Marching Transformer
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FMTConfig, P2VAEConfig
from modules import (
    ConditionMLP,
    DiffusionForcingGRU,
    PyramidOutputHead,
    PyramidPatchEmbed,
    SiTBlock,
    VAEDecoder,
    VAEEncoder,
)


# ---------------------------------------------------------------------------
# P2VAE
# ---------------------------------------------------------------------------

class P2VAE(nn.Module):
    """Pretrained Physics Variational Autoencoder.

    Compresses 3×128×128 PDE field snapshots to a 16×16×16 latent space
    (12× compression) using an SD-VAE architecture.

    Training objective:
        L_VAE = 0.5 * E[||x - x_hat||^2] + beta * KL(q(y|x) || p(y))
    """

    def __init__(self, cfg: P2VAEConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = VAEEncoder(
            in_channels=cfg.in_channels,
            base_dim=cfg.base_dim,
            channel_mult=cfg.channel_mult,
            num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=cfg.attn_resolutions,
            z_channels=cfg.z_channels,
            dropout=cfg.dropout,
        )
        self.decoder = VAEDecoder(
            out_channels=cfg.out_channels,
            base_dim=cfg.base_dim,
            channel_mult=cfg.channel_mult,
            num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=cfg.attn_resolutions,
            z_channels=cfg.z_channels,
            dropout=cfg.dropout,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode x to (mean, logvar) of the latent distribution."""
        h = self.encoder(x)  # (B, 2*z_channels, H_lat, W_lat)
        mean, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean, logvar

    def reparameterize(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns:
            x_hat:   reconstructed field (B, C, H, W)
            mean:    latent mean (B, z_channels, H_lat, W_lat)
            logvar:  latent log-variance (B, z_channels, H_lat, W_lat)
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def loss(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_hat, mean, logvar = self.forward(x)
        recon = 0.5 * F.mse_loss(x_hat, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        total = recon + self.cfg.beta_kl * kl
        return total, {"recon": recon.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent mean (no sampling) for downstream use."""
        mean, _ = self.encode(x)
        return mean


# ---------------------------------------------------------------------------
# FMT
# ---------------------------------------------------------------------------

class FMT(nn.Module):
    """Flow Marching Transformer.

    Implements:
      - Latent temporal pyramids (4 frames at resolutions 2×2, 4×4, 8×8, 16×16)
      - Diffusion forcing via GRU causal conditioning
      - AdaLN-Zero conditioned SiT blocks (RMSNorm + SwiGLU, Llama-2 style)
      - Flow marching training objective

    Training objective (Eq. 9 / 11 in paper):
        L_CFM = 0.5 * E[sum_s ||(1-t_s)*g(x_{s,t_s}^{k_s}, t_s, h_{s-1}) - (x_{s+1} - x_{s,t_s}^{k_s})||^2]
    """

    def __init__(self, cfg: FMTConfig):
        super().__init__()
        self.cfg = cfg
        n_frames = cfg.n_frames  # 4

        # Patch embeddings for each pyramid level
        self.patch_embeds = nn.ModuleList([
            PyramidPatchEmbed(
                latent_channels=cfg.latent_channels,
                latent_spatial=cfg.latent_spatial,
                factor=cfg.pyramid_factors[i],
                embed_dim=cfg.embed_dim,
            )
            for i in range(n_frames)
        ])

        # Output heads for each pyramid level
        self.output_heads = nn.ModuleList([
            PyramidOutputHead(
                embed_dim=cfg.embed_dim,
                latent_channels=cfg.latent_channels,
                latent_spatial=cfg.latent_spatial,
                factor=cfg.pyramid_factors[i],
            )
            for i in range(n_frames)
        ])

        # Token counts per frame
        self.token_counts = [
            (cfg.latent_spatial // f) ** 2
            for f in cfg.pyramid_factors
        ]
        self.total_tokens = sum(self.token_counts)

        # Condition MLP: fuse timestep + GRU hidden state
        self.cond_mlp = ConditionMLP(
            time_embed_dim=cfg.time_embed_dim,
            gru_dim=cfg.gru_dim,
            out_dim=cfg.embed_dim,
        )

        # SiT transformer blocks
        self.blocks = nn.ModuleList([
            SiTBlock(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                cond_dim=cfg.embed_dim,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.depth)
        ])

        # Diffusion forcing GRU
        self.df_gru = DiffusionForcingGRU(
            embed_dim=cfg.embed_dim,
            latent_channels=cfg.latent_channels,
            latent_spatial=cfg.latent_spatial,
            n_cross_attn_heads=cfg.n_cross_attn_heads,
        )

        # Frame-level positional bias (learnable, to distinguish frames)
        self.frame_embed = nn.Parameter(
            torch.randn(n_frames, cfg.embed_dim) * 0.02
        )

    # ------------------------------------------------------------------
    # Flow marching interpolation kernel
    # ------------------------------------------------------------------

    @staticmethod
    def interpolate(
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Construct x_t^k from (x0, x1, t, k).

        x_t^k = mu_t + sigma_t * z
        mu_t  = t * x1 + k * (1-t) * x0
        sigma_t = (1-t) * (1-k)

        Args:
            x0, x1: (B, C, H, W) consecutive latent states
            t, k:   (B,) scalars in [0, 1]

        Returns:
            x_t_k: (B, C, H, W) interpolated noisy state
            z:     (B, C, H, W) noise sample used
        """
        # Reshape for broadcasting
        t_ = t.view(-1, 1, 1, 1)
        k_ = k.view(-1, 1, 1, 1)
        z = torch.randn_like(x0)
        mu = t_ * x1 + k_ * (1 - t_) * x0
        sigma = (1 - t_) * (1 - k_)
        x_t_k = mu + sigma * z
        return x_t_k, z

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        y_noisy: List[torch.Tensor],
        t_list: List[torch.Tensor],
        h_list: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """Predict flow marching velocities for all 4 frames.

        Args:
            y_noisy: list of 4 noisy latents, each (B, C, H, W)
            t_list:  list of 4 timestep tensors, each (B,)
            h_list:  list of 4 GRU hidden states h_{s-1}, each (B, embed_dim).
                     If None, uses zero hidden states.

        Returns:
            velocities: list of 4 predicted velocities, each (B, C, H, W)
        """
        B = y_noisy[0].shape[0]
        device = y_noisy[0].device
        n_frames = self.cfg.n_frames

        if h_list is None:
            h_list = [
                self.df_gru.init_hidden(B, device) for _ in range(n_frames)
            ]

        # 1. Embed each frame at its pyramid resolution and add frame bias
        token_list = []
        for s in range(n_frames):
            tokens = self.patch_embeds[s](y_noisy[s])  # (B, n_s, embed_dim)
            tokens = tokens + self.frame_embed[s].unsqueeze(0).unsqueeze(0)
            token_list.append(tokens)

        # 2. Concatenate all tokens
        x = torch.cat(token_list, dim=1)  # (B, total_tokens, embed_dim)

        # 3. Build per-token condition vectors from each frame's (t_s, h_{s-1})
        #    and add as an additive bias so each frame's tokens carry its own
        #    timestep and history information.
        cond_bias = torch.zeros(B, self.total_tokens, self.cfg.embed_dim, device=device)
        offset = 0
        for s in range(n_frames):
            n_s = self.token_counts[s]
            c_s = self.cond_mlp(t_list[s], h_list[s])  # (B, embed_dim)
            cond_bias[:, offset:offset + n_s, :] = c_s.unsqueeze(1)
            offset += n_s

        x = x + cond_bias  # inject per-frame conditioning into token stream

        # 4. Global AdaLN-Zero condition: use the full-resolution frame's condition
        c_global = self.cond_mlp(t_list[-1], h_list[-1])  # (B, embed_dim)
        for block in self.blocks:
            x = block(x, c_global)

        # 5. Split tokens back per frame and project to velocity at full resolution
        velocities = []
        offset = 0
        for s in range(n_frames):
            n_s = self.token_counts[s]
            tokens_s = x[:, offset:offset + n_s, :]  # (B, n_s, embed_dim)
            vel_s = self.output_heads[s](tokens_s)    # (B, C, H, W)
            velocities.append(vel_s)
            offset += n_s

        return velocities

    # ------------------------------------------------------------------
    # Training step: compute CFM loss for a sequence of 4 frames
    # ------------------------------------------------------------------

    def compute_cfm_loss(
        self,
        y_seq: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the conditional flow marching loss over 4 consecutive latents.

        With 4 consecutive states (y0, y1, y2, y3) we have 3 prediction steps:
          y0 → y1,  y1 → y2,  y2 → y3

        The temporal pyramid takes all 4 noisy frames as input simultaneously.
        The 4th frame (y3) is also given a noisy version for the pyramid context,
        but no loss is computed for it (no y4 available).

        Training objective (Eq. 11):
            L_CFM = 0.5 * E[sum_s ||(1-t_s)*g - (y_{s+1} - y_{s,t_s}^{k_s})||^2]

        Args:
            y_seq: list of 4 latent tensors, each (B, C, H, W)

        Returns:
            loss:  scalar CFM loss
            info:  dict with per-step losses
        """
        B = y_seq[0].shape[0]
        device = y_seq[0].device
        n_steps = len(y_seq) - 1  # 3 prediction steps

        h = self.df_gru.init_hidden(B, device)

        y_noisy_list = []
        t_list = []
        h_list = []

        # Build noisy inputs for steps 0, 1, 2 (source frames) causally
        for s in range(n_steps):
            x0 = y_seq[s]
            x1 = y_seq[s + 1]

            t_s = torch.rand(B, device=device)
            k_s = torch.rand(B, device=device)

            y_t_k, _ = self.interpolate(x0, x1, t_s, k_s)

            h_list.append(h)
            y_noisy_list.append(y_t_k)
            t_list.append(t_s)

            # Update GRU causally with the noisy current state
            h = self.df_gru(h, y_t_k)

        # 4th pyramid frame: noisy version of y3 (no loss computed for it).
        # Construct as y3 + (1-t3)*(1-k3)*z, which is the marginal of the
        # interpolation kernel when x0=x1=y3 (self-noising).
        t3 = torch.rand(B, device=device)
        k3 = torch.rand(B, device=device)
        z3 = torch.randn_like(y_seq[-1])
        t3_ = t3.view(-1, 1, 1, 1)
        k3_ = k3.view(-1, 1, 1, 1)
        y3_noisy = y_seq[-1] + (1 - t3_) * (1 - k3_) * z3

        h_list.append(h)
        y_noisy_list.append(y3_noisy)
        t_list.append(t3)

        # Single forward pass over all 4 pyramid frames
        velocities = self.forward(y_noisy_list, t_list, h_list)

        # Compute loss for steps 0, 1, 2 only
        total_loss = torch.zeros(1, device=device)
        info: Dict[str, torch.Tensor] = {}
        for s in range(n_steps):
            x1 = y_seq[s + 1]
            y_t_k = y_noisy_list[s]
            t_s = t_list[s]
            g_s = velocities[s]

            target = x1 - y_t_k
            t_ = t_s.view(-1, 1, 1, 1)
            loss_s = 0.5 * ((1 - t_) * g_s - target).pow(2).mean()
            total_loss = total_loss + loss_s
            info[f"loss_step_{s}"] = loss_s.detach()

        total_loss = total_loss / n_steps
        info["loss"] = total_loss.detach()
        return total_loss.squeeze(), info

    # ------------------------------------------------------------------
    # Inference: Euler ODE sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_sample(
        self,
        y_init_list: List[torch.Tensor],
        k_list: List[float],
        h_list: List[torch.Tensor],
        n_steps: int = 100,
    ) -> List[torch.Tensor]:
        """Integrate all 4 pyramid frames simultaneously from t=0 to t=1.

        Per the paper: "(t_0, t_1, t_2, t_3) are initialized to be 0, and are
        updated simultaneously during the flow marching process."

        For deterministic prediction: k_s=1 for all s → frames start clean.
        For generation: k_0=k_1=k_2=1, k_3<1 → frame 3 starts noisy.

        Args:
            y_init_list: list of 4 initial latents (B, C, H, W).
                         For k_s=1: y_s (clean).
                         For k_s<1: k_s*y_s + (1-k_s)*z (noisy).
            k_list:      list of 4 bridge parameters (used only for init, not here)
            h_list:      list of 4 GRU conditions h_{s-1}, each (B, embed_dim)
            n_steps:     Euler discretization steps (N=100 per paper)

        Returns:
            y_final: list of 4 predicted next states, each (B, C, H, W)
        """
        B = y_init_list[0].shape[0]
        device = y_init_list[0].device
        dt = 1.0 / n_steps

        y_list = [y.clone() for y in y_init_list]
        t_val = 0.0

        for _ in range(n_steps):
            t_tensors = [
                torch.full((B,), t_val, device=device, dtype=y_list[0].dtype)
                for _ in range(self.cfg.n_frames)
            ]
            velocities = self.forward(y_list, t_tensors, h_list)
            y_list = [y_list[s] + dt * velocities[s] for s in range(self.cfg.n_frames)]
            t_val = t_val + dt

        return y_list

    @torch.no_grad()
    def rollout(
        self,
        y_context: List[torch.Tensor],
        n_future: int,
        k_val: float = 1.0,
        n_euler_steps: int = 100,
    ) -> List[torch.Tensor]:
        """Autoregressive rollout for long-horizon prediction.

        At each step, the sliding window of 4 frames is:
          - Frames 0, 1, 2: clean past states (k=1)
          - Frame 3: current state, noisy if k_val < 1 (generation), clean if k_val=1

        The Euler sampler integrates all 4 frames from t=0 to t=1 simultaneously.
        The last frame's output is the predicted next state.

        Args:
            y_context: list of initial latent frames (at least 3 for full context)
            n_future:  number of future frames to generate
            k_val:     bridge parameter for frame 3 (1=deterministic, <1=stochastic)
            n_euler_steps: Euler ODE steps per frame

        Returns:
            predictions: list of n_future predicted latent frames
        """
        B = y_context[0].shape[0]
        device = y_context[0].device

        # Pad context to at least 4 frames by repeating the first frame
        context = list(y_context)
        while len(context) < self.cfg.n_frames:
            context.insert(0, context[0])

        # Build GRU state from context
        h = self.df_gru.init_hidden(B, device)
        for y_ctx in context:
            h = self.df_gru(h, y_ctx)

        predictions = []
        for _ in range(n_future):
            window = context[-self.cfg.n_frames:]  # 4 most recent frames

            # Construct initial states for Euler integration
            # Frames 0, 1, 2: clean (k=1, t=0 → y_s)
            # Frame 3: noisy if k_val < 1
            y_init_list = [window[s].clone() for s in range(self.cfg.n_frames - 1)]
            y_prev = window[-1]
            if k_val < 1.0:
                z = torch.randn_like(y_prev)
                y3_init = k_val * y_prev + (1.0 - k_val) * z
            else:
                y3_init = y_prev.clone()
            y_init_list.append(y3_init)

            # GRU conditions: h_{s-1} for each frame
            # For frames 0-2 (clean context), use the accumulated h
            h_list = [h] * self.cfg.n_frames

            # Euler integration
            y_final_list = self.euler_sample(
                y_init_list,
                k_list=[1.0] * (self.cfg.n_frames - 1) + [k_val],
                h_list=h_list,
                n_steps=n_euler_steps,
            )

            # The last frame's output is the predicted next state
            y_next = y_final_list[-1]
            predictions.append(y_next)

            # Update context and GRU
            context.append(y_next)
            h = self.df_gru(h, y_next)

        return predictions


# ---------------------------------------------------------------------------
# Combined model for end-to-end finetuning
# ---------------------------------------------------------------------------

class P2VAEWithFMT(nn.Module):
    """Joint P2VAE + FMT model for end-to-end finetuning (REPA-E style).

    The finetuning loss is:
        L(θ, φ, ω) = L_CFM(θ, φ) + λ_VAE * L_VAE(ω)

    A stop-gradient is applied after encoding to prevent CFM loss from
    deteriorating the autoencoder.
    """

    def __init__(self, vae: P2VAE, fmt: FMT, lambda_vae: float = 1.0):
        super().__init__()
        self.vae = vae
        self.fmt = fmt
        self.lambda_vae = lambda_vae

    def forward(
        self,
        x_seq: List[torch.Tensor],
        stop_grad_vae: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute joint finetuning loss.

        Args:
            x_seq:         list of 4 raw field tensors, each (B, C, H, W)
            stop_grad_vae: if True, stop gradient from CFM loss into VAE

        Returns:
            total_loss: scalar
            info:       dict of loss components
        """
        # VAE reconstruction loss on all frames
        vae_loss = torch.tensor(0.0, device=x_seq[0].device)
        y_seq = []
        for x in x_seq:
            x_hat, mean, logvar = self.vae(x)
            recon = 0.5 * F.mse_loss(x_hat, x)
            kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            vae_loss = vae_loss + recon + self.vae.cfg.beta_kl * kl

            # Encode to latent (stop gradient for CFM if requested)
            z = self.vae.reparameterize(mean, logvar)
            if stop_grad_vae:
                z = z.detach()
            y_seq.append(z)

        vae_loss = vae_loss / len(x_seq)

        # CFM loss in latent space
        cfm_loss, cfm_info = self.fmt.compute_cfm_loss(y_seq)

        total = cfm_loss + self.lambda_vae * vae_loss
        info = {**cfm_info, "vae_loss": vae_loss.detach(), "total": total.detach()}
        return total, info
