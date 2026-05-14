## diffusion/model.py
"""Conditional diffusion model architecture for Prioritized Generative Replay (PGR).

Implements the residual MLP denoising network ε_θ(x^n, n, c) that forms the
core of the conditional diffusion model. The architecture extends SYNTHER's
unconditional residual MLP by adding a conditioning pathway for the scalar
relevance score F(τ), enabling classifier-free guidance (CFG) at sampling time.

Architecture overview:
    - Input projection: input_dim → hidden_dim
    - num_layers ResidualBlocks, each fusing x, timestep embedding, and condition
    - Output projection: hidden_dim → input_dim (predicted noise)

The conditioning mechanism uses:
    - Sinusoidal timestep embedding → 2-layer MLP → time_emb_dim
    - Scalar relevance score → 2-layer MLP → cond_emb_dim
    - Learned null condition token ∅ (nn.Parameter) for CFG unconditional pass

Config references (config.yaml):
    diffusion.hidden_dim:    256   # residual MLP hidden width
    diffusion.num_layers:    4     # number of residual blocks
    diffusion.time_emb_dim:  128   # sinusoidal time embedding dimension
    diffusion.cond_emb_dim:  128   # condition (relevance score) embedding dimension
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Single denoising residual block fusing data, timestep, and condition signals.

    Implements the AdaLN-style conditioning pattern where the timestep and
    condition embeddings are injected additively before LayerNorm, modulating
    the normalization statistics. The residual connection preserves gradient
    flow through deep networks.

    Forward computation:
        h = LayerNorm(x + time_proj(t_emb) + cond_proj(c_emb))
        output = x + fc2(SiLU(fc1(h)))

    Attributes:
        fc1: First feedforward projection (hidden_dim → hidden_dim).
        fc2: Second feedforward projection (hidden_dim → hidden_dim).
        time_proj: Projects time embedding into hidden space (time_emb_dim → hidden_dim).
        cond_proj: Projects condition embedding into hidden space (cond_emb_dim → hidden_dim).
        norm1: LayerNorm applied after fusing x, time, and condition signals.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        time_emb_dim: int = 128,
        cond_emb_dim: int = 128,
    ) -> None:
        """Initialises the residual block.

        Args:
            hidden_dim: Width of the hidden representation. All internal
                projections map to/from this dimension. Corresponds to
                config.diffusion.hidden_dim (default 256).
            time_emb_dim: Dimension of the incoming timestep embedding.
                Corresponds to config.diffusion.time_emb_dim (default 128).
                The time_proj linear layer maps from this dimension to hidden_dim.
            cond_emb_dim: Dimension of the incoming condition embedding.
                Corresponds to config.diffusion.cond_emb_dim (default 128).
                The cond_proj linear layer maps from this dimension to hidden_dim.
        """
        super().__init__()

        self.hidden_dim: int = hidden_dim
        self.time_emb_dim: int = time_emb_dim
        self.cond_emb_dim: int = cond_emb_dim

        # ── Feedforward projections ───────────────────────────────────────────
        # fc1 and fc2 form the MLP within the residual block.
        # Both map hidden_dim → hidden_dim; SiLU activation between them.
        self.fc1: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)

        # ── Conditioning projections ──────────────────────────────────────────
        # time_proj maps the timestep embedding into hidden space for additive
        # injection. cond_proj does the same for the relevance score embedding.
        # Both project from their respective embedding dims to hidden_dim.
        self.time_proj: nn.Linear = nn.Linear(time_emb_dim, hidden_dim)
        self.cond_proj: nn.Linear = nn.Linear(cond_emb_dim, hidden_dim)

        # ── Normalization ─────────────────────────────────────────────────────
        # LayerNorm applied after fusing x, time, and condition signals.
        # This is the single LayerNorm specified in the design document.
        self.norm1: nn.LayerNorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        c_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Applies one residual denoising step with conditioning.

        Fuses the noisy data representation x with the timestep embedding
        t_emb and condition embedding c_emb via additive injection before
        LayerNorm, then applies a 2-layer MLP with SiLU activation and a
        residual connection.

        Args:
            x: Hidden representation from the previous block or input
                projection. Float32 tensor of shape (B, hidden_dim).
            t_emb: Timestep embedding from DiffusionModel.time_mlp.
                Float32 tensor of shape (B, time_emb_dim).
            c_emb: Condition embedding from DiffusionModel.cond_mlp or
                the expanded null_cond_emb. Float32 tensor of shape
                (B, cond_emb_dim).

        Returns:
            Float32 tensor of shape (B, hidden_dim) — updated hidden
            representation with the residual connection applied.
        """
        # ── Step 1: Fuse x, time, and condition via additive injection ────────
        # Project time and condition embeddings into hidden space and add to x.
        # All three terms have shape (B, hidden_dim) after projection.
        # This is the AdaLN-style conditioning: signals modulate the input
        # to LayerNorm rather than the output, which is more stable.
        fused: torch.Tensor = x + self.time_proj(t_emb) + self.cond_proj(c_emb)

        # ── Step 2: Apply LayerNorm ───────────────────────────────────────────
        h: torch.Tensor = self.norm1(fused)  # (B, hidden_dim)

        # ── Step 3: Apply 2-layer MLP with SiLU activation ───────────────────
        # fc1: (B, hidden_dim) → (B, hidden_dim)
        # SiLU: element-wise smooth activation (better gradient flow than ReLU)
        # fc2: (B, hidden_dim) → (B, hidden_dim)
        mlp_out: torch.Tensor = self.fc2(F.silu(self.fc1(h)))  # (B, hidden_dim)

        # ── Step 4: Residual connection ───────────────────────────────────────
        # Add the original x (before fusing) to preserve gradient flow.
        # This is the standard residual connection: output = x + F(x).
        return x + mlp_out  # (B, hidden_dim)


class DiffusionModel(nn.Module):
    """Residual MLP denoising network for conditional diffusion in PGR.

    Implements the noise predictor ε_θ(x^n, n, c) that takes a noisy
    transition tuple, a diffusion timestep, and a scalar relevance score
    condition, and predicts the noise that was added to the original
    transition. Extends SYNTHER's unconditional architecture with a
    conditioning pathway for classifier-free guidance (CFG).

    The model is called twice per CFG denoising step:
        - Conditional pass: forward(x_t, t, cond, use_null_cond=False)
        - Unconditional pass: forward(x_t, t, cond, use_null_cond=True)
    Combined: eps = guidance_scale * eps_cond + (1 - guidance_scale) * eps_uncond

    Input dimension:
        input_dim = 2 * obs_dim + action_dim + 1
        (concatenation of s, a, s', r — the full transition tuple)
        For pixel tasks: obs_dim = CNN latent dim (feature_dim=50 for DRQv2)

    Attributes:
        input_dim: Dimension of the flattened transition tuple (s, a, s', r).
        hidden_dim: Width of the hidden representation in all residual blocks.
        num_layers: Number of residual blocks.
        time_emb_dim: Dimension of the sinusoidal timestep embedding.
        cond_emb_dim: Dimension of the condition (relevance score) embedding.
        input_proj: Linear projection from input_dim to hidden_dim.
        blocks: ModuleList of num_layers ResidualBlock instances.
        output_proj: Linear projection from hidden_dim back to input_dim.
        time_mlp: 2-layer MLP refining the sinusoidal timestep embedding.
        cond_mlp: 2-layer MLP embedding the scalar relevance score.
        null_cond_emb: Learned null condition token ∅ for CFG unconditional pass.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_emb_dim: int = 128,
        cond_emb_dim: int = 128,
    ) -> None:
        """Initialises the diffusion model.

        Args:
            input_dim: Dimension of the flattened transition tuple (s, a, s', r).
                Computed as 2 * obs_dim + action_dim + 1 by the caller
                (ConditionalDiffusion.__init__). For state-based quadruped-walk:
                2 * 67 + 12 + 1 = 147. For pixel tasks with DRQv2 (feature_dim=50):
                2 * 50 + action_dim + 1.
            hidden_dim: Width of the hidden representation. All residual blocks
                use this dimension. Corresponds to config.diffusion.hidden_dim
                (default 256).
            num_layers: Number of residual blocks. Corresponds to
                config.diffusion.num_layers (default 4).
            time_emb_dim: Dimension of the sinusoidal timestep embedding and
                the time_mlp output. Must be even (required by sinusoidal
                embedding). Corresponds to config.diffusion.time_emb_dim
                (default 128).
            cond_emb_dim: Dimension of the condition embedding produced by
                cond_mlp and the null_cond_emb parameter. Corresponds to
                config.diffusion.cond_emb_dim (default 128).

        Raises:
            ValueError: If time_emb_dim is odd (sinusoidal embedding requires
                even dimension for sin/cos split).
        """
        super().__init__()

        if time_emb_dim % 2 != 0:
            raise ValueError(
                f"time_emb_dim must be even for sinusoidal embedding, "
                f"got {time_emb_dim}. Use an even value (e.g. 128)."
            )

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.time_emb_dim: int = time_emb_dim
        self.cond_emb_dim: int = cond_emb_dim

        # ── Input projection ──────────────────────────────────────────────────
        # Lifts the raw flattened transition vector from input_dim to hidden_dim.
        # This is the entry point for the noisy transition x^n.
        self.input_proj: nn.Linear = nn.Linear(input_dim, hidden_dim)

        # ── Residual blocks ───────────────────────────────────────────────────
        # num_layers=4 blocks from config.diffusion.num_layers.
        # Each block receives the same t_emb and c_emb — shared across all layers.
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                ResidualBlock(
                    hidden_dim=hidden_dim,
                    time_emb_dim=time_emb_dim,
                    cond_emb_dim=cond_emb_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # ── Output projection ─────────────────────────────────────────────────
        # Projects the final hidden representation back to input_dim to produce
        # the predicted noise ε of the same shape as the input transition.
        self.output_proj: nn.Linear = nn.Linear(hidden_dim, input_dim)

        # ── Timestep embedding MLP ────────────────────────────────────────────
        # Takes the sinusoidal embedding (already of dimension time_emb_dim)
        # and refines it through a 2-layer MLP with SiLU activation.
        # Input: (B, time_emb_dim) from _sinusoidal_embedding
        # Output: (B, time_emb_dim) — same dimension, refined representation
        self.time_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ── Condition embedding MLP ───────────────────────────────────────────
        # Embeds the scalar relevance score F(τ) into a cond_emb_dim vector.
        # Input: (B, 1) — scalar relevance score, unsqueezed before passing in
        # Output: (B, cond_emb_dim) — condition embedding for all residual blocks
        self.cond_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(1, cond_emb_dim),
            nn.SiLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
        )

        # ── Null condition token (CFG) ────────────────────────────────────────
        # Learned embedding representing the null condition ∅ for the
        # unconditional pass in classifier-free guidance.
        # Shape: (cond_emb_dim,) — expanded to (B, cond_emb_dim) in forward().
        # Initialized to zeros per the design spec; will be learned during
        # training when p_uncond=0.25 causes 25% of training samples to use
        # this token instead of the actual relevance score condition.
        self.null_cond_emb: nn.Parameter = nn.Parameter(
            torch.zeros(cond_emb_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        use_null_cond: bool = False,
    ) -> torch.Tensor:
        """Predicts the noise added to a noisy transition at timestep t.

        Implements the forward pass of ε_θ(x^n, n, c) — the denoising network
        that predicts the noise ε injected at timestep n. Called by
        ConditionalDiffusion.train_step() during training and by
        ConditionalDiffusion._cfg_sample() during generation (twice per step:
        once with use_null_cond=False and once with use_null_cond=True).

        Args:
            x: Noisy transition tuple at timestep t. Float32 tensor of shape
                (B, input_dim). Produced by DDPMScheduler.add_noise() during
                training, or initialized as Gaussian noise during generation.
            t: Diffusion timestep indices. Integer or float tensor of shape
                (B,). Values in [1, num_timesteps]. Cast to float internally
                for the sinusoidal embedding computation.
            cond: Scalar relevance scores F(τ). Float32 tensor of shape (B,)
                or (B, 1). Normalized to [0, 1] by PGRTrainer before being
                passed here. Ignored when use_null_cond=True.
            use_null_cond: If True, replaces the condition embedding with the
                learned null token null_cond_emb, implementing the unconditional
                pass of classifier-free guidance. If False (default), embeds
                the actual relevance score via cond_mlp.

        Returns:
            Float32 tensor of shape (B, input_dim) — predicted noise ε.
            This is the model's estimate of the noise that was added to the
            original clean transition x^0 to produce the noisy x^n at
            timestep t.
        """
        batch_size: int = x.shape[0]

        # ── Step 1: Compute timestep embedding ───────────────────────────────
        # _sinusoidal_embedding: (B,) → (B, time_emb_dim)
        # time_mlp: (B, time_emb_dim) → (B, time_emb_dim)
        t_sin: torch.Tensor = self._sinusoidal_embedding(t, self.time_emb_dim)
        t_emb: torch.Tensor = self.time_mlp(t_sin)  # (B, time_emb_dim)

        # ── Step 2: Compute condition embedding ──────────────────────────────
        if use_null_cond:
            # Unconditional pass: broadcast the learned null token across batch.
            # null_cond_emb shape: (cond_emb_dim,) → expand to (B, cond_emb_dim)
            c_emb: torch.Tensor = self.null_cond_emb.unsqueeze(0).expand(
                batch_size, -1
            )  # (B, cond_emb_dim)
        else:
            # Conditional pass: embed the scalar relevance score via cond_mlp.
            # Normalize cond to shape (B, 1) regardless of whether it arrives
            # as (B,) or (B, 1) — cond_mlp expects (B, 1) as input.
            cond_2d: torch.Tensor = cond.view(batch_size, 1).to(
                dtype=torch.float32, device=x.device
            )  # (B, 1)
            c_emb = self.cond_mlp(cond_2d)  # (B, cond_emb_dim)

        # ── Step 3: Project input to hidden space ─────────────────────────────
        # input_proj: (B, input_dim) → (B, hidden_dim)
        h: torch.Tensor = self.input_proj(x)  # (B, hidden_dim)

        # ── Step 4: Pass through residual blocks ──────────────────────────────
        # Each block receives the same t_emb and c_emb — they are shared
        # across all num_layers=4 blocks. The blocks progressively refine
        # the hidden representation h using the conditioning signals.
        for block in self.blocks:
            h = block(h, t_emb, c_emb)  # (B, hidden_dim)

        # ── Step 5: Project to output space ───────────────────────────────────
        # output_proj: (B, hidden_dim) → (B, input_dim)
        # Returns the predicted noise ε of the same shape as the input x.
        return self.output_proj(h)  # (B, input_dim)

    def _sinusoidal_embedding(
        self,
        t: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        """Computes sinusoidal positional embeddings for diffusion timesteps.

        Implements the standard sinusoidal embedding from "Attention Is All
        You Need" (Vaswani et al., 2017), adapted for scalar diffusion
        timesteps. Produces a fixed (non-learned) embedding that encodes
        the relative magnitude of the timestep.

        Formula:
            half_dim = dim // 2
            freq_i = exp(-log(10000) * i / (half_dim - 1))  for i in [0, half_dim)
            emb_i = t * freq_i
            output = [sin(emb_0), ..., sin(emb_{half_dim-1}),
                      cos(emb_0), ..., cos(emb_{half_dim-1})]

        This produces a (B, dim) tensor where the first half contains sine
        values and the second half contains cosine values, giving the model
        a rich representation of the timestep that varies smoothly.

        Args:
            t: Diffusion timestep indices. Integer or float tensor of shape
                (B,). Values typically in [1, num_timesteps=100]. Cast to
                float32 internally before multiplication.
            dim: Embedding dimension. Must be even (enforced in __init__).
                Corresponds to config.diffusion.time_emb_dim (default 128).

        Returns:
            Float32 tensor of shape (B, dim) on the same device as t.
            The first dim//2 columns contain sine values; the last dim//2
            columns contain cosine values.
        """
        half_dim: int = dim // 2

        # Compute frequency vector: exp(-log(10000) * i / (half_dim - 1))
        # for i in [0, half_dim). Shape: (half_dim,)
        # Using math.log for the scalar constant — avoids a torch operation.
        # The denominator (half_dim - 1) handles the edge case half_dim=1
        # by producing a single frequency of 1.0 (exp(0) = 1).
        denominator: float = max(half_dim - 1, 1)  # Avoid division by zero.
        frequencies: torch.Tensor = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, dtype=torch.float32, device=t.device)
            / denominator
        )  # (half_dim,)

        # Compute outer product: t[:, None] * frequencies[None, :]
        # t shape: (B,) → (B, 1) after unsqueeze
        # frequencies shape: (half_dim,) → (1, half_dim) after unsqueeze
        # Result: (B, half_dim)
        t_float: torch.Tensor = t.float()  # Ensure float32 for multiplication.
        embeddings: torch.Tensor = t_float[:, None] * frequencies[None, :]  # (B, half_dim)

        # Concatenate sin and cos along the feature dimension.
        # Output shape: (B, dim) where dim = 2 * half_dim.
        embeddings = torch.cat(
            [torch.sin(embeddings), torch.cos(embeddings)],
            dim=-1,
        )  # (B, dim)

        return embeddings

    def __repr__(self) -> str:
        """Returns a concise string representation of the diffusion model."""
        total_params: int = sum(p.numel() for p in self.parameters())
        trainable_params: int = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return (
            f"DiffusionModel("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"time_emb_dim={self.time_emb_dim}, "
            f"cond_emb_dim={self.cond_emb_dim}, "
            f"total_params={total_params:,}, "
            f"trainable_params={trainable_params:,})"
        )
