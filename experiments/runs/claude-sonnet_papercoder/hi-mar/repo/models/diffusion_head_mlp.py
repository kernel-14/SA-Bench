## models/diffusion_head_mlp.py
"""MLP-based diffusion head for Hi-MAR Phase 1 (low-resolution tokens).

This module implements the per-token MLP diffusion head used in Phase 1 of
Hi-MAR. Its primary purpose is to optimise the Scale-Aware Transformer backbone
to produce high-quality conditional tokens ``Z^s`` (low-resolution pivots) that
guide Phase 2 generation. It does **not** produce the final output images.

Key design constraint from the paper (Section 3.3):
    "Note that we only adopt Diffusion Transformer head in the second phase
    while the first phase still utilises MLP-based diffusion head, since the
    diffusion head on the first phase mainly aims to optimise the low-resolution
    conditional tokens instead of providing intermediary pivots for the next
    phase."

Each masked token is processed **independently** — there is no self-attention
or cross-token interaction. This is the defining characteristic that
distinguishes this head from ``DiTDiffusionHead``.

Paper reference (Section 3.1):
    L(z_i, x_i) = E_{ε,t}[||ε - ε_θ(x_i^t | t, z_i)||²]

Configuration alignment (config.yaml):
    diffusion.timesteps                      = 100
    vae.latent_channels                      = 16   → input_dim
    models.himar_b.diff_head1.n_layers       = 6
    models.himar_b.diff_head1.hidden_size    = 1024
    models.himar_b.transformer.hidden_size   = 768  → cond_dim
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.losses import DiffusionUtils, sinusoidal_embedding


# ---------------------------------------------------------------------------
# SinusoidalPositionEmbedding
# ---------------------------------------------------------------------------


class SinusoidalPositionEmbedding(nn.Module):
    """Converts integer diffusion timesteps into sinusoidal embeddings.

    This is a parameter-free module that wraps the ``sinusoidal_embedding``
    function from ``training.losses`` as an ``nn.Module`` so it can be
    composed inside ``nn.Sequential`` for the ``time_embed`` pipeline.

    The output dimension is fixed at construction time. The standard choice
    for diffusion models (DDPM, MAR, DiT) is 256 as the intermediate
    sinusoidal dimension before projection to the model's hidden size.

    Attributes:
        dim: Output embedding dimension. Must be even.
    """

    def __init__(self, dim: int = 256) -> None:
        """Initialises the sinusoidal embedding module.

        Args:
            dim: Output embedding dimension. Must be even (sinusoidal
                embedding requires pairs of sin/cos components). The design
                specifies 256 as the intermediate dimension before the
                Linear projection to ``hidden_size``.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(
                f"SinusoidalPositionEmbedding requires an even dimension, "
                f"got dim={dim}."
            )
        self.dim: int = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Converts integer timestep indices to sinusoidal embeddings.

        Args:
            t: Integer timestep tensor of shape ``[B]``. Values are in
                ``{0, …, T-1}`` where ``T = diffusion.timesteps = 100``.

        Returns:
            Float tensor of shape ``[B, dim]`` on the same device as ``t``.
        """
        return sinusoidal_embedding(t, self.dim)


# ---------------------------------------------------------------------------
# MLPBlock
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    """Single AdaLN-conditioned FFN block for the MLP diffusion head.

    Implements a residual FFN block with Adaptive Layer Normalisation (AdaLN).
    Unlike ``ScaleAwareBlock`` which uses AdaLN-**Zero** with 6 modulation
    outputs (including residual gates), this block uses standard AdaLN with
    2 outputs (shift and scale only), matching the design specification
    ``adaLN_modulation = Linear(hidden_size, 2*hidden_size)``.

    The ``(1 + scale)`` formulation initialises the block as approximately
    identity at the start of training (when ``adaLN_modulation`` is
    zero-initialised, ``scale = 0`` and ``shift = 0``, so the modulated
    norm output equals ``LayerNorm(x)``).

    Block computation:
        shift, scale = split(adaLN_modulation(c), 2, dim=-1)
        y = (1 + scale) * LayerNorm(x) + shift
        x = x + FFN(y)

    Attributes:
        norm: LayerNorm without learnable affine parameters (AdaLN provides
            scale and shift externally).
        ffn: Two-layer FFN with GELU activation and 4× expansion ratio.
        adaLN_modulation: Linear layer producing shift and scale from the
            context vector. Zero-initialised for training stability.
    """

    def __init__(self, hidden_size: int = 1024) -> None:
        """Initialises the MLPBlock.

        Args:
            hidden_size: Hidden dimension of the block. Matches the diffusion
                head's hidden size (e.g., 1024 for Hi-MAR-B Phase 1 head,
                per ``config.models.himar_b.diff_head1.hidden_size``).
        """
        super().__init__()

        self.hidden_size: int = hidden_size

        # ------------------------------------------------------------------
        # LayerNorm without learnable affine parameters.
        # AdaLN provides scale and shift externally via adaLN_modulation.
        # eps=1e-6 matches DiT convention for numerical stability.
        # ------------------------------------------------------------------
        self.norm: nn.LayerNorm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        # ------------------------------------------------------------------
        # Two-layer FFN with GELU activation.
        # Intermediate dimension: hidden_size * 4 (standard 4× MLP ratio).
        # ------------------------------------------------------------------
        ffn_hidden: int = hidden_size * 4
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size),
        )

        # ------------------------------------------------------------------
        # AdaLN modulation: context vector [B, H] → [shift, scale] each [B, H].
        # Zero-initialised so that at the start of training:
        #   shift = 0, scale = 0
        #   → modulated output = (1+0)*LayerNorm(x) + 0 = LayerNorm(x)
        # This provides stable gradient flow at initialisation.
        # ------------------------------------------------------------------
        self.adaLN_modulation: nn.Linear = nn.Linear(
            hidden_size, 2 * hidden_size, bias=True
        )
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the AdaLN-conditioned FFN block.

        Supports both per-token ``[B, hidden_size]`` and batched sequence
        ``[B, N, hidden_size]`` inputs. The context vector ``c`` is broadcast
        over the sequence dimension when ``x`` has shape ``[B, N, H]``.

        Args:
            x: Input tensor. Either ``[B, hidden_size]`` (single token) or
                ``[B, N, hidden_size]`` (token sequence). In practice, the
                ``MLPDiffusionHead`` always passes ``[B, N_masked, hidden_size]``
                tensors.
            c: Context vector of shape ``[B, hidden_size]`` or
                ``[B, N, hidden_size]``. When ``[B, H]``, it is unsqueezed
                and broadcast over the ``N`` dimension automatically.

        Returns:
            Output tensor of the same shape as ``x``.
        """
        # ------------------------------------------------------------------
        # Compute AdaLN modulation parameters from context vector c.
        # ------------------------------------------------------------------
        if c.dim() == 2:
            # c: [B, H] — broadcast over sequence dimension.
            # adaLN_modulation: [B, H] → [B, 2*H]
            modulation: torch.Tensor = self.adaLN_modulation(c)  # [B, 2*H]
            # Split into shift and scale: each [B, H].
            shift: torch.Tensor
            scale: torch.Tensor
            shift, scale = modulation.chunk(2, dim=-1)  # [B, H], [B, H]

            if x.dim() == 3:
                # x: [B, N, H] — unsqueeze for broadcasting over N.
                shift = shift.unsqueeze(1)   # [B, 1, H]
                scale = scale.unsqueeze(1)   # [B, 1, H]
        else:
            # c: [B, N, H] — per-token context.
            # adaLN_modulation: [B, N, H] → [B, N, 2*H]
            modulation = self.adaLN_modulation(c)  # [B, N, 2*H]
            shift, scale = modulation.chunk(2, dim=-1)  # [B, N, H], [B, N, H]

        # ------------------------------------------------------------------
        # Apply AdaLN: y = (1 + scale) * LayerNorm(x) + shift
        # The (1 + scale) formulation initialises as identity when scale=0.
        # ------------------------------------------------------------------
        normed: torch.Tensor = self.norm(x)                    # same shape as x
        y: torch.Tensor = (1.0 + scale) * normed + shift       # same shape as x

        # ------------------------------------------------------------------
        # Apply FFN with residual connection.
        # ------------------------------------------------------------------
        x_out: torch.Tensor = x + self.ffn(y)                  # same shape as x

        return x_out


# ---------------------------------------------------------------------------
# MLPDiffusionHead
# ---------------------------------------------------------------------------


class MLPDiffusionHead(nn.Module):
    """Per-token MLP diffusion head for Hi-MAR Phase 1 (low-resolution).

    This head processes each masked token **independently** — there is no
    self-attention or cross-token interaction. It predicts the noise
    ``ε_θ(x_i^t | t, z_i)`` for each masked token ``i`` given:
    - ``x_i^t``: the noise-corrupted version of the token at diffusion timestep ``t``
    - ``z_i``: the conditional token from the Scale-Aware Transformer backbone

    Architecture (per the design spec):
        time_embed:  SinusoidalPositionEmbedding(256) → Linear(256, H) → SiLU → Linear(H, H)
        input_proj:  Linear(input_dim=16, H)
        cond_proj:   Linear(cond_dim, H)
        blocks:      n_layers × MLPBlock(H)
        final_proj:  Linear(H, input_dim=16)  [zero-initialised]

    where ``H = hidden_size`` (e.g., 1024 for Hi-MAR-B Phase 1 head).

    Configuration alignment (config.yaml):
        diffusion.timesteps                    = 100  → diff_timesteps
        vae.latent_channels                    = 16   → input_dim
        models.himar_b.diff_head1.n_layers     = 6    → n_layers
        models.himar_b.diff_head1.hidden_size  = 1024 → hidden_size
        models.himar_b.transformer.hidden_size = 768  → cond_dim

    Attributes:
        n_layers: Number of MLPBlock layers.
        hidden_size: Internal hidden dimension of the head.
        input_dim: VAE latent token dimension (16 for KL-16).
        cond_dim: Dimension of conditional tokens from the backbone.
        diff_timesteps: Total number of diffusion timesteps.
        time_embed: Sequential module converting timestep integers to vectors.
        input_proj: Projects noisy latent tokens to hidden_size.
        cond_proj: Projects backbone conditional tokens to hidden_size.
        blocks: ModuleList of MLPBlock instances.
        final_proj: Output projection from hidden_size to input_dim.
    """

    def __init__(
        self,
        n_layers: int = 6,
        hidden_size: int = 1024,
        input_dim: int = 16,
        cond_dim: int = 768,
        diff_timesteps: int = 100,
    ) -> None:
        """Initialises the MLPDiffusionHead.

        Args:
            n_layers: Number of MLPBlock layers in the head. Config values:
                6 (Hi-MAR-B), 8 (Hi-MAR-L), 12 (Hi-MAR-H), 4 (Hi-MAR-S).
                Default: 6 (Hi-MAR-B).
            hidden_size: Internal hidden dimension of the head. Config values:
                1024 (Hi-MAR-B), 1280 (Hi-MAR-L), 1536 (Hi-MAR-H), 768 (Hi-MAR-S).
                Default: 1024 (Hi-MAR-B).
            input_dim: VAE latent token dimension. Config: ``vae.latent_channels = 16``.
                This is both the input and output dimension of the head (it
                predicts noise in the same space as the input tokens).
                Default: 16.
            cond_dim: Dimension of conditional tokens ``Z^s`` from the
                Scale-Aware Transformer backbone. Equals the backbone's
                ``hidden_size`` (e.g., 768 for Hi-MAR-B backbone).
                Default: 768.
            diff_timesteps: Total number of diffusion timesteps ``T``.
                Config: ``diffusion.timesteps = 100``. Default: 100.
        """
        super().__init__()

        self.n_layers: int = n_layers
        self.hidden_size: int = hidden_size
        self.input_dim: int = input_dim
        self.cond_dim: int = cond_dim
        self.diff_timesteps: int = diff_timesteps

        # ------------------------------------------------------------------
        # Time embedding pipeline.
        # Converts integer timestep t ∈ {0, …, T-1} to a hidden_size vector.
        # Architecture: SinusoidalPositionEmbedding(256) → Linear(256, H) →
        #               SiLU → Linear(H, H)
        # The 256-dim sinusoidal embedding provides sufficient frequency
        # coverage before projection to the model's hidden dimension.
        # ------------------------------------------------------------------
        self.time_embed: nn.Sequential = nn.Sequential(
            SinusoidalPositionEmbedding(dim=256),
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # ------------------------------------------------------------------
        # Input projection: latent_dim (16) → hidden_size.
        # Projects the noise-corrupted VAE latent tokens to the head's
        # internal hidden dimension.
        # ------------------------------------------------------------------
        self.input_proj: nn.Linear = nn.Linear(input_dim, hidden_size)

        # ------------------------------------------------------------------
        # Conditional projection: cond_dim → hidden_size.
        # Projects the backbone's conditional tokens Z^s (shape [B, N, cond_dim])
        # to the head's hidden dimension for per-token conditioning.
        # ------------------------------------------------------------------
        self.cond_proj: nn.Linear = nn.Linear(cond_dim, hidden_size)

        # ------------------------------------------------------------------
        # Stack of MLPBlock layers.
        # Each block applies AdaLN-conditioned FFN independently per token.
        # ------------------------------------------------------------------
        self.blocks: nn.ModuleList = nn.ModuleList(
            [MLPBlock(hidden_size=hidden_size) for _ in range(n_layers)]
        )

        # ------------------------------------------------------------------
        # Output projection: hidden_size → input_dim (16).
        # Zero-initialised so the head predicts zero noise at initialisation,
        # providing a stable starting point for diffusion training.
        # ------------------------------------------------------------------
        self.final_proj: nn.Linear = nn.Linear(hidden_size, input_dim)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the noise for each masked token independently.

        Processes each token at position ``i`` using only its own noisy
        representation ``x_noisy[b, i]``, the shared timestep embedding
        ``t_emb[b]``, and its corresponding conditional token ``cond[b, i]``.
        There is no cross-token attention or interaction.

        Processing pipeline:
            1. t_emb = time_embed(t)                    [B, H]
            2. x = input_proj(x_noisy)                  [B, N, H]
            3. c = cond_proj(cond)                      [B, N, H]
            4. x = x + c                                [B, N, H]  (per-token conditioning)
            5. context_vec = t_emb broadcast to [B, N, H]
            6. for block in blocks: x = block(x, context_vec)
            7. return final_proj(x)                     [B, N, input_dim]

        Args:
            x_noisy: Noise-corrupted latent tokens at masked positions,
                shape ``[B, N_masked, input_dim]`` where ``input_dim = 16``.
                Produced by ``DiffusionUtils.q_sample`` during training, or
                initialised as Gaussian noise during inference.
            t: Integer diffusion timestep indices, shape ``[B]``. Each value
                is in ``{0, …, diff_timesteps-1}``. One timestep per sample
                in the batch (all tokens in a sample share the same timestep).
            cond: Conditional tokens from the Scale-Aware Transformer backbone
                (``Z^s``), shape ``[B, N_masked, cond_dim]``. These are the
                backbone outputs at masked positions only.

        Returns:
            Predicted noise ``ε_θ(x_i^t | t, z_i)`` for each masked token,
            shape ``[B, N_masked, input_dim]``. This is the training target
            for the MSE loss ``||ε - ε_θ||²``.
        """
        batch_size: int = x_noisy.shape[0]
        n_masked: int = x_noisy.shape[1]

        # ------------------------------------------------------------------
        # Step 1: Compute time embedding.
        # t: [B] → time_embed → [B, hidden_size]
        # ------------------------------------------------------------------
        t_emb: torch.Tensor = self.time_embed(t)  # [B, H]

        # ------------------------------------------------------------------
        # Step 2: Project noisy input tokens to hidden_size.
        # x_noisy: [B, N_masked, 16] → [B, N_masked, H]
        # ------------------------------------------------------------------
        x: torch.Tensor = self.input_proj(x_noisy)  # [B, N_masked, H]

        # ------------------------------------------------------------------
        # Step 3: Project conditional tokens to hidden_size.
        # cond: [B, N_masked, cond_dim] → [B, N_masked, H]
        # ------------------------------------------------------------------
        c_proj: torch.Tensor = self.cond_proj(cond)  # [B, N_masked, H]

        # ------------------------------------------------------------------
        # Step 4: Combine projected input with per-token conditioning.
        # This is the key per-token conditioning: position i only sees cond[i].
        # No cross-token interaction occurs here.
        # ------------------------------------------------------------------
        x = x + c_proj  # [B, N_masked, H]

        # ------------------------------------------------------------------
        # Step 5: Broadcast time embedding to all token positions.
        # t_emb: [B, H] → [B, N_masked, H] for AdaLN context in each block.
        # All tokens in a sample share the same diffusion timestep.
        # ------------------------------------------------------------------
        context_vec: torch.Tensor = t_emb.unsqueeze(1).expand(
            batch_size, n_masked, self.hidden_size
        )  # [B, N_masked, H]

        # ------------------------------------------------------------------
        # Step 6: Process through all MLPBlocks.
        # Each block applies AdaLN-conditioned FFN independently per token.
        # No attention — per-token independence is preserved throughout.
        # ------------------------------------------------------------------
        for block in self.blocks:
            x = block(x, context_vec)  # [B, N_masked, H]

        # ------------------------------------------------------------------
        # Step 7: Project back to latent dimension.
        # [B, N_masked, H] → [B, N_masked, input_dim=16]
        # ------------------------------------------------------------------
        noise_pred: torch.Tensor = self.final_proj(x)  # [B, N_masked, 16]

        return noise_pred

    def compute_loss(
        self,
        cond: torch.Tensor,
        x_target: torch.Tensor,
        diff_utils: DiffusionUtils,
    ) -> torch.Tensor:
        """Computes the DDPM denoising MSE loss for Phase 1 training.

        Implements the paper's diffusion loss objective (Section 3.1):
            L(z_i, x_i) = E_{ε,t}[||ε - ε_θ(x_i^t | t, z_i)||²]

        The loss is computed only over masked token positions (the caller in
        ``HiMAR.forward_phase1`` passes only masked token data for both
        ``cond`` and ``x_target``).

        Training procedure:
            1. Sample random timestep t ~ Uniform(0, T-1) per sample
            2. Apply forward diffusion: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
            3. Predict noise: ε_θ = self.forward(x_t, t, cond)
            4. Compute MSE: ||ε - ε_θ||²

        Args:
            cond: Conditional tokens from the backbone at masked positions,
                shape ``[B, N_masked, cond_dim]``. These are the ``Z^s``
                outputs from ``ScaleAwareTransformer.forward()`` at the
                positions indicated by ``mask_lr``.
            x_target: Ground-truth low-resolution latent tokens at masked
                positions, shape ``[B, N_masked, input_dim=16]``. These are
                the VAE-encoded 128×128 image tokens at masked positions.
            diff_utils: Shared ``DiffusionUtils`` instance providing the DDPM
                noise schedule and ``q_sample`` method. Instantiated once in
                ``HiMAR.__init__`` and shared between both diffusion heads.

        Returns:
            Scalar loss tensor (mean MSE over all masked token positions and
            the latent dimension). Differentiable with respect to all model
            parameters.
        """
        batch_size: int = x_target.shape[0]
        device: torch.device = x_target.device

        # ------------------------------------------------------------------
        # Step 1: Sample random diffusion timesteps.
        # t ~ Uniform(0, T-1), one per sample in the batch.
        # Different samples in the batch train at different noise levels,
        # which is the standard DDPM training procedure.
        # ------------------------------------------------------------------
        t: torch.Tensor = torch.randint(
            low=0,
            high=diff_utils.timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )  # [B]

        # ------------------------------------------------------------------
        # Step 2: Apply forward diffusion to get noisy tokens and target noise.
        # x_noisy = √ᾱ_t · x_target + √(1-ᾱ_t) · ε
        # noise = ε ~ N(0, I)  ← this is the prediction target
        # ------------------------------------------------------------------
        x_noisy: torch.Tensor
        noise: torch.Tensor
        x_noisy, noise = diff_utils.q_sample(x_target, t)
        # x_noisy: [B, N_masked, 16]
        # noise:   [B, N_masked, 16]

        # ------------------------------------------------------------------
        # Step 3: Predict the noise using the MLP diffusion head.
        # ------------------------------------------------------------------
        noise_pred: torch.Tensor = self.forward(x_noisy, t, cond)
        # noise_pred: [B, N_masked, 16]

        # ------------------------------------------------------------------
        # Step 4: Compute MSE loss between predicted and actual noise.
        # F.mse_loss with reduction='mean' averages over all elements:
        # B × N_masked × input_dim, which is the standard DDPM objective.
        # ------------------------------------------------------------------
        loss: torch.Tensor = F.mse_loss(noise_pred, noise, reduction="mean")

        return loss
