```python
## models/diffusion_head_dit.py
"""Diffusion Transformer head for Hi-MAR Phase 2 (high-resolution tokens).

This module implements the Diffusion Transformer (DiT) head used in Phase 2 of
Hi-MAR. Unlike the MLP-based head in Phase 1 which processes each token
independently, this head uses self-attention across ALL token positions to model
inter-token dependencies during denoising. This directly addresses MAR's second
limitation described in Section 3.1 of the paper.

Key design principle from Section 3.3:
    "We design a new Diffusion Transformer head by exploiting the self-attention
    to model the interdependency among tokens. In contrast to MLP-based diffusion
    head that only takes the conditional tokens of masked tokens as conditions,
    the Diffusion Transformer head considers all the masked and unmasked
    conditional tokens."

Block computation (Figure 2e, Section 3.3):
    [α₁, β₁, γ₁, α₂, β₂, γ₂] = split(c)
    y_a = y^i + γ₁ · Attention(α₁ · LN(y^i) + β₁)
    y^{i+1} = y_a + γ₂ · FFN(α₂ · LN(y_a) + β₂)
    where c = time_step_embedding + mean(conditional_tokens)

Configuration alignment (config.yaml):
    diffusion.timesteps                      = 100  → diff_timesteps
    vae.latent_channels                      = 16   → input_dim
    models.himar_b.diff_head2.n_layers       = 6    → n_layers
    models.himar_b.diff_head2.hidden_size    = 512  → hidden_size
    models.himar_b.transformer.hidden_size   = 768  → cond_dim

Paper reference: Section 3.3, Figure 2(e).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.losses import DiffusionUtils, sinusoidal_embedding


# ---------------------------------------------------------------------------
# SinusoidalPositionEmbedding
# ---------------------------------------------------------------------------


class SinusoidalPositionEmbedding(nn.Module):
    """Converts integer diffusion timesteps into sinusoidal embeddings.

    A parameter-free module wrapping the ``sinusoidal_embedding`` function
    from ``training.losses`` as an ``nn.Module`` for use inside
    ``nn.Sequential`` in the ``time_embed`` pipeline.

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

        Raises:
            ValueError: If ``dim`` is odd.
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
# DiTBlock
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    """Single Transformer block with AdaLN-Zero conditioning for the DiT head.

    Implements the diffusion Transformer block described in Section 3.3 of the
    paper and illustrated in Figure 2(e). The block uses AdaLN-Zero conditioning
    where a global context vector ``c`` (sum of time embedding and pooled
    conditional tokens) drives six modulation parameters that control the
    LayerNorm scale/shift and residual gates for both attention and FFN
    sub-layers.

    The AdaLN-Zero trick zero-initialises the ``adaLN_modulation`` linear layer
    so that at the start of training all residual gates ``γ₁, γ₂ = 0``, making
    each block an identity function. This provides stable gradient flow at
    initialisation.

    Paper equations (Section 3.3):
        [α₁, β₁, γ₁, α₂, β₂, γ₂] = split(c)
        y_a = y^i + γ₁ · Attention(α₁ · LN(y^i) + β₁)
        y^{i+1} = y_a + γ₂ · FFN(α₂ · LN(y_a) + β₂)

    Key distinction from ``ScaleAwareBlock``: The context ``c`` here comes from
    time embedding + pooled conditional tokens (denoising context), whereas in
    ``ScaleAwareBlock`` it comes from the scale vector (resolution context).

    Attributes:
        norm1: LayerNorm without learnable affine parameters (AdaLN handles it).
        norm2: Same as norm1 for the FFN sub-layer.
        attn: Multi-head self-attention with batch_first=True.
        ffn: Two-layer FFN with GELU activation.
        adaLN_modulation: Linear layer producing 6×hidden_size modulation
            parameters from the context vector. Zero-initialised.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
    ) -> None:
        """Initialises the DiTBlock.

        Args:
            hidden_size: Transformer hidden dimension D. Must be divisible by
                ``n_heads``. Config values for Phase 2 head:
                512 (Hi-MAR-B), 512 (Hi-MAR-L), 768 (Hi-MAR-H), 384 (Hi-MAR-S).
            n_heads: Number of attention heads. Must divide ``hidden_size``
                evenly. Derived as ``hidden_size // 64`` in the parent class.
            mlp_ratio: FFN hidden dimension multiplier. The FFN intermediate
                dimension is ``int(hidden_size * mlp_ratio)``. Config: 4.0.
        """
        super().__init__()

        self.hidden_size: int = hidden_size
        self.n_heads: int = n_heads
        self.mlp_ratio: float = mlp_ratio

        # ------------------------------------------------------------------
        # LayerNorm without learnable affine parameters.
        # AdaLN provides scale and shift externally via the modulation params.
        # eps=1e-6 matches DiT convention for numerical stability.
        # ------------------------------------------------------------------
        self.norm1: nn.LayerNorm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.norm2: nn.LayerNorm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        # ------------------------------------------------------------------
        # Multi-head self-attention.
        # batch_first=True: input/output shape is [B, N, D] throughout,
        # avoiding transposes. Bidirectional (no causal mask) — the DiT head
        # attends across ALL token positions simultaneously, which is the
        # key advantage over the per-token MLP head.
        # ------------------------------------------------------------------
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )

        # ------------------------------------------------------------------
        # Two-layer FFN with GELU activation.
        # Intermediate dimension: int(hidden_size * mlp_ratio).
        # ------------------------------------------------------------------
        ffn_hidden: int = int(hidden_size * mlp_ratio)
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size),
        )

        # ------------------------------------------------------------------
        # AdaLN-Zero modulation layer.
        # Maps context vector c [B, D] → [B, 6*D] producing:
        #   [α₁, β₁, γ₁, α₂, β₂, γ₂], each of shape [B, D].
        # Zero-initialised so all residual gates γ start at 0 (identity init).
        # This is the AdaLN-Zero trick from DiT (Peebles & Xie, 2023).
        # ------------------------------------------------------------------
        self.adaLN_modulation: nn.Linear = nn.Linear(
            hidden_size, 6 * hidden_size, bias=True
        )
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(
        self,
        y: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the DiT block.

        Args:
            y: Input token sequence, shape ``[B, N, D]``. ``N`` is the total
                number of token positions (all 256 high-res tokens for Phase 2).
                Self-attention operates over all N positions simultaneously.
            c: Per-sample context vector, shape ``[B, D]``. This is the global
                conditioning signal (same for all token positions within a
                sample). Computed as ``t_emb + mean(cond_projected)`` in the
                parent ``DiTDiffusionHead``.

        Returns:
            Output token sequence, shape ``[B, N, D]``, on the same device
            as ``y``.
        """
        # ------------------------------------------------------------------
        # Step 1: Compute 6 modulation parameters from the context vector.
        # adaLN_modulation: [B, D] → [B, 6*D]
        # chunk(6, dim=-1): 6 tensors of shape [B, D]
        # unsqueeze(1): [B, D] → [B, 1, D] for broadcasting over N tokens.
        # ------------------------------------------------------------------
        modulation: torch.Tensor = self.adaLN_modulation(c)  # [B, 6*D]
        chunks = modulation.chunk(6, dim=-1)  # 6 × [B, D]
        alpha1: torch.Tensor = chunks[0].unsqueeze(1)  # [B, 1, D]
        beta1: torch.Tensor = chunks[1].unsqueeze(1)   # [B, 1, D]
        gamma1: torch.Tensor = chunks[2].unsqueeze(1)  # [B, 1, D]
        alpha2: torch.Tensor = chunks[3].unsqueeze(1)  # [B, 1, D]
        beta2: torch.Tensor = chunks[4].unsqueeze(1)   # [B, 1, D]
        gamma2: torch.Tensor = chunks[5].unsqueeze(1)  # [B, 1, D]

        # ------------------------------------------------------------------
        # Step 2: Attention sub-layer with AdaLN-Zero.
        # Paper: y_a = y^i + γ₁ · Attention(α₁ · LN(y^i) + β₁)
        # Self-attention over ALL N positions — this is the key difference
        # from the MLP head which processes each token independently.
        # ------------------------------------------------------------------
        normed1: torch.Tensor = self.norm1(y)                    # [B, N, D]
        modulated1: torch.Tensor = alpha1 * normed1 + beta1      # [B, N, D]
        # Bidirectional self-attention: Q = K = V = modulated1 (no causal mask).
        attn_out: torch.Tensor
        attn_out, _ = self.attn(modulated1, modulated1, modulated1)  # [B, N, D]
        y_a: torch.Tensor = y + gamma1 * attn_out                # [B, N, D]

        # ------------------------------------------------------------------
        # Step 3: FFN sub-layer with AdaLN-Zero.
        # Paper: y^{i+1} = y_a + γ₂ · FFN(α₂ · LN(y_a) + β₂)
        # ------------------------------------------------------------------
        normed2: torch.Tensor = self.norm2(y_a)                  # [B, N, D]
        modulated2: torch.Tensor = alpha2 * normed2 + beta2      # [B, N, D]
        ffn_out: torch.Tensor = self.ffn(modulated2)             # [B, N, D]
        y_out: torch.Tensor = y_a + gamma2 * ffn_out             # [B, N, D]

        return y_out


# ---------------------------------------------------------------------------
# DiTDiffusionHead
# ---------------------------------------------------------------------------


class DiTDiffusionHead(nn.Module):
    """Diffusion Transformer head for Hi-MAR Phase 2 (high-resolution tokens).

    This head processes ALL ``N = 256`` high-resolution token positions
    simultaneously using self-attention, enabling inter-token dependency
    modeling during denoising. This is the key architectural innovation over
    MAR's per-token MLP head.

    Critical design point (Section 3.3):
        "the Diffusion Transformer head considers all the masked and unmasked
        conditional tokens"
    → ``forward()`` takes ALL N_all=256 token positions, not just masked ones.
    → Loss is computed only on masked positions (``compute_loss``).

    Architecture:
        time_embed:  SinusoidalPositionEmbedding(256) → Linear(256, H) → SiLU → Linear(H, H)
        input_proj:  Linear(input_dim=16, H)
        cond_proj:   Linear(cond_dim, H)
        blocks:      n_layers × DiTBlock(H, n_heads)
        norm:        LayerNorm(H)
        final_proj:  Linear(H, input_dim=16)  [zero-initialised]

    where H = hidden_size (e.g., 512 for Hi-MAR-B Phase 2 head).

    Context vector construction (per paper Section 3.3):
        "c denotes the context vector obtained by summating the time step
        embedding and the conditional tokens"
        → c = t_emb + mean(cond_projected)   shape [B, H]
        The mean pooling aggregates global structural information from all
        conditional tokens into a single vector that modulates AdaLN uniformly.

    Configuration alignment (config.yaml):
        diffusion.timesteps                    = 100  → diff_timesteps
        vae.latent_channels                    = 16   → input_dim
        models.himar_b.diff_head2.n_layers     = 6    → n_layers
        models.himar_b.diff_head2.hidden_size  = 512  → hidden_size
        models.himar_b.transformer.hidden_size = 768  → cond_dim

    Inference note (Section 4.5):
        "we use much fewer steps (e.g., 4 steps) in the second phase
        considering that the Diffusion Transformer head is much heavier than
        the MLP-based diffusion head."

    Attributes:
        n_layers: Number of DiTBlock layers.
        hidden_size: Internal hidden dimension of the head.
        input_dim: VAE latent token dimension (16 for KL-16).
        cond_dim: Dimension of conditional tokens from the backbone.
        diff_timesteps: Total number of diffusion timesteps.
        n_heads: Number of attention heads (derived as hidden_size // 64).
        time_embed: Sequential module converting timestep integers to vectors.
        input_proj: Projects noisy latent tokens to hidden_size.
        cond_proj: Projects backbone conditional tokens to hidden_size.
        blocks: ModuleList of DiTBlock instances.
        norm: Final LayerNorm applied after all blocks.
        final_proj: Output projection from hidden_size to input_dim.
    """

    def __init__(
        self,
        n_layers: int = 6,
        hidden_size: int = 512,
        input_dim: int = 16,
        cond_dim: int = 768,
        diff_timesteps: int = 100,
    ) -> None:
        """Initialises the DiTDiffusionHead.

        Args:
            n_layers: Number of DiTBlock layers in the head. Config values:
                6 (Hi-MAR-B), 8 (Hi-MAR-L), 12 (Hi-MAR-H), 4 (Hi-MAR-S).
                Default: 6 (Hi-MAR-B).
            hidden_size: Internal hidden dimension of the head. Config values:
                512 (Hi-MAR-B), 512 (Hi-MAR-L), 768 (Hi-MAR-H), 384 (Hi-MAR-S).
                Default: 512 (Hi-MAR-B).
            input_dim: VAE latent token dimension. Config: ``vae.latent_channels = 16``.
                This is both the input and output dimension of the head (it
                predicts noise in the same space as the input tokens).
                Default: 16.
            cond_dim: Dimension of conditional tokens ``Z^l`` from the
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
        # Derive number of attention heads from hidden_size.
        # Standard convention (DiT/ViT): n_heads = hidden_size // 64.
        # This gives: 512 → 8 heads, 768 → 12 heads.
        # Clamp to at least 1 to handle small hidden sizes gracefully.
        # ------------------------------------------------------------------
        self.n_heads: int = max(1, hidden_size // 64)

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
        # Input projection: input_dim (16) → hidden_size.
        # Projects the noise-corrupted VAE latent tokens to the head's
        # internal hidden dimension. Applied to ALL N_all token positions.
        # ------------------------------------------------------------------
        self.input_proj: nn.Linear = nn.Linear(input_dim, hidden_size)

        # ------------------------------------------------------------------
        # Conditional projection: cond_dim → hidden_size.
        # Projects the backbone's conditional tokens Z^l (shape [B, N, cond_dim])
        # to the head's hidden dimension. These are then mean-pooled to form
        # the global context vector c.
        # ------------------------------------------------------------------
        self.cond_proj: nn.Linear = nn.Linear(cond_dim, hidden_size)

        # ------------------------------------------------------------------
        # Stack of DiTBlock layers.
        # Each block applies AdaLN-conditioned self-attention + FFN over ALL
        # N_all=256 token positions simultaneously.
        # ------------------------------------------------------------------
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    n_heads=self.n_heads,
                    mlp_ratio=4.0,
                )
                for _ in range(n_layers)
            ]
        )

        # ------------------------------------------------------------------
        # Final LayerNorm applied after all DiTBlocks.
        # Uses learnable affine parameters (standard LayerNorm, not AdaLN).
        # ------------------------------------------------------------------
        self.norm: nn.LayerNorm = nn.LayerNorm(hidden_size)

        # ------------------------------------------------------------------
        # Output projection: hidden_size → input_dim (16).
        # Zero-initialised so the head predicts zero noise at initialisation,
        # providing a stable starting point for diffusion training.
        # ------------------------------------------------------------------
        self.final_proj: nn.Linear = nn.Linear(hidden_size, input_dim)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def _build_context(
        self,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Builds the per-sample global context vector c for AdaLN conditioning.

        Implements the paper's description (Section 3.3):
            "c denotes the context vector obtained by summating the time step
            embedding and the conditional tokens"

        Since ``t_emb`` is ``[B, D]`` (not per-token), the per-token conditional
        tokens must be aggregated to the same shape. Mean pooling over the token
        dimension preserves global structural information from all conditional
        tokens without introducing additional parameters.

        Args:
            t: Integer diffusion timestep indices, shape ``[B]``. Values in
                ``{0, …, diff_timesteps-1}``.
            cond: Conditional tokens from the Scale-Aware Transformer backbone
                (``Z^l``), shape ``[B, N_all, cond_dim]``. These are the
                backbone outputs for ALL high-resolution token positions.

        Returns:
            Context vector ``c`` of shape ``[B, hidden_size]``. This is the
            global conditioning signal passed to every DiTBlock.
        """
        # Time embedding: [B] → [B, hidden_size]
        t_emb: torch.Tensor = self.time_embed(t)  # [B, H]

        # Project conditional tokens: [B, N_all, cond_dim] → [B, N_all, H]
        cond_projected: torch.Tensor = self.cond_proj(cond)  # [B, N_all, H]

        # Mean pool over token dimension to get global context: [B, N_all, H] → [B, H]
        # This aggregates structural information from all conditional tokens
        # into a single vector that uniformly modulates AdaLN across all positions.
        cond_global: torch.Tensor = cond_projected.mean(dim=1)  # [B, H]

        # Sum time embedding and pooled conditional tokens.
        # Paper: "c denotes the context vector obtained by summating the time
        # step embedding and the conditional tokens"
        c: torch.Tensor = t_emb + cond_global  # [B, H]

        return c

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the noise for ALL token positions using self-attention.

        CRITICAL: This method takes ALL N_all=256 token positions (both masked
        and unmasked), not just masked ones. This enables the self-attention in
        each DiTBlock to attend across all positions simultaneously, modeling
        inter-token dependencies during denoising.

        Processing pipeline:
            1. t_emb = time_embed(t)                    [B, H]
            2. cond_projected = cond_proj(cond)         [B, N_all, H]
            3. c = t_emb + mean(cond_projected)         [B, H]  (global context)
            4. x = input_proj(x_noisy)                  [B, N_all, H]
            5. for block in blocks: x = block(x, c)     [B, N_all, H]
            6. x = norm(x)                              [B, N_all, H]
            7. return final_proj(x)                     [B, N_all, input_dim]

        Args:
            x_noisy: Noise-corrupted latent tokens for ALL positions,
                shape ``[B, N_all, input_dim]`` where ``input_dim = 16``.
                ``N_all = 256`` for high-resolution Phase 2 tokens.
                Produced by ``DiffusionUtils.q_sample`` during training, or
                initialised as Gaussian noise during inference.
            t: Integer diffusion timestep indices, shape ``[B]``. Each value
                is in ``{0, …, diff_timesteps-1}``. One timestep per sample
                in the batch (all tokens in a sample share the same timestep).
            cond: Conditional tokens from the Scale-Aware Transformer backbone
                (``Z^l``), shape ``[B, N_all, cond_dim]``. These are the
                backbone outputs for ALL high-resolution token positions.

        Returns:
            Predicted noise ``ε_θ(y^t | t, c)`` for ALL token positions,
            shape ``[B, N_all, input_dim]``. During loss computation, only
            the masked positions are used (see ``compute_loss``).
        """
        # ------------------------------------------------------------------
        # Step 1: Build global context vector c.
        # c = t_emb + mean(cond_projected)   shape [B, H]
        # ------------------------------------------------------------------
        c: torch.Tensor = self._build_context(t, cond)  # [B, H]

        # ------------------------------------------------------------------
        # Step 2: Project noisy input tokens to hidden_size.
        # x_noisy: [B, N_all, input_dim=16] → [B, N_all, H]
        # Applied to ALL token positions (masked + unmasked).
        # ------------------------------------------------------------------
        x: torch.Tensor = self.input_proj(x_noisy)  # [B, N_all, H]

        # ------------------------------------------------------------------
        # Step 3: Pass through all DiTBlocks.
        # Each block receives the full sequence [B, N_all, H] and the global
        # context vector c [B, H]. Self-attention operates over all N_all
        # positions simultaneously — this is the core advantage over the MLP head.
        # ------------------------------------------------------------------
        for block in self.blocks:
            x = block(x, c)  # [B, N_all, H]

        # ------------------------------------------------------------------
        # Step 4: Apply final LayerNorm.
        # ------------------------------------------------------------------
        x = self.norm(x)  # [B, N_all, H]

        # ------------------------------------------------------------------
        # Step 5: Project back to latent dimension.
        # [B, N_all, H] → [B, N_all, input_dim=16]
        # ------------------------------------------------------------------
        noise_pred: torch.Tensor = self.final_proj(x)  # [B, N_all, 16]

        return noise_pred

    def compute_loss(
        self,
        cond: torch.Tensor,
        x_target: torch.Tensor,
        mask: torch.Tensor,
        diff_utils: DiffusionUtils,
    ) -> torch.Tensor:
        """Computes the DDPM denoising MSE loss for Phase 2 training.

        Implements the paper's diffusion loss objective (Section 3.1):
            L(z_i, x_i) = E_{ε,t}[||ε - ε_θ(x_i^t | t, z_i)||²]

        Key design: the forward pass runs on ALL N_all=256 token positions
        (enabling global self-attention), but the loss is computed only on
        masked positions. Unmasked tokens provide context via self-attention
        but do not contribute to the gradient.

        Training procedure:
            1. Sample random timestep t ~ Uniform(0, T-1) per sample
            2. Apply forward diffusion to ALL tokens: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
            3. Predict noise for ALL tokens: ε_θ = self.forward(x_t, t, cond)
            4. Compute MSE only on masked positions: ||ε - ε_θ||² where mask=True

        Masking convention (Shared Knowledge):
            ``True = masked`` (token is hidden / to be predicted).

        Args:
            cond: Conditional tokens from the backbone for ALL positions,
                shape ``[B, N_all, cond_dim]``. These are the ``Z^l`` outputs
                from ``ScaleAwareTransformer.forward()`` with ``scale_id=1``.
                Note: ALL positions are passed (not just masked), enabling
                the DiT head to leverage global context.
            x_target: Ground-truth high-resolution latent tokens for ALL
                positions, shape ``[B, N_all, input_dim=16]``. These are the
                VAE-encoded 256×256 image tokens.
            mask: Boolean tensor, shape ``[B, N_all]``. ``True`` at positions
                that are masked (hidden / to be predicted). Only these positions
                contribute to the loss.
            diff_utils: Shared ``DiffusionUtils`` instance providing the DDPM
                noise schedule and ``q_sample`` method. Instantiated once in
                ``HiMAR.__init__`` and shared between both diffusion heads.

        Returns:
            Scalar loss tensor (mean MSE over masked token positions and the
            latent dimension). Differentiable with respect to all model
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
        # Step 2: Apply forward diffusion to ALL token positions.
        # x_noisy = √ᾱ_t · x_target + √(1-ᾱ_t) · ε
        # noise = ε ~ N(0, I)  ← this is the prediction target
        # Even unmasked tokens get noise added — this is intentional.
        # The full noisy sequence is fed to the DiT head so self-attention
        # can leverage all positions for context.
        # ------------------------------------------------------------------
        x_noisy: torch.Tensor
        noise: torch.Tensor
        x_noisy, noise = diff_utils.q_sample(x_target, t)
        # x_noisy: [B, N_all, 16]
        # noise:   [B, N_all, 16]

        # ------------------------------------------------------------------
        # Step 3: