## Code: models/mdm_transformer.py

```python
## models/mdm_transformer.py
"""Bidirectional Masked Diffusion Model (MDM) transformer denoiser.

Implements the core denoising network p_theta(· | x_t) described in
"Train for the Worst, Plan for the Best: Understanding Token Ordering in
Masked Diffusions".

The model is a bidirectional (non-causal) transformer that takes a partially
masked token sequence x_t and outputs per-position logits over the vocabulary.
Key design properties from the paper:

1. **Bidirectional attention**: No causal mask — every position attends to
   every other position, enabling the model to use all available context
   when predicting masked tokens.

2. **Time-embedding-free**: The model receives only x_t, not the noise level
   t explicitly. The number of masked tokens in x_t implicitly encodes t.

3. **Flexible positional embeddings**: Supports both learned positional
   embeddings (puzzle experiments, scaling law experiments) and Rotary
   Position Embeddings / RoPE (NAE-SAT experiment with 19M MDM).

4. **Pre-norm (norm_first=True)**: LayerNorm applied before attention and
   FFN sublayers for training stability.

Architecture variants by experiment (from config.yaml):
  - 6M Sudoku MDM:   4 layers, 8 heads, d_model=256, d_ff=1024,  learned pos
  - 19M Zebra MDM:   6 layers, 8 heads, d_model=512, d_ff=2048,  learned pos
  - 19M NAE-SAT MDM: 6 layers, 8 heads, d_model=512, d_ff=2048,  RoPE

Integration points:
  - MDMLoss.compute()         → model.forward(x_t)       → logits [B,L,V]
  - VanillaSampler.sample()   → model.get_probs(x_t)     → probs  [B,L,V]
  - AdaptiveSampler.sample()  → model.get_probs(x_t)     → probs  [B,L,V]
  - NAESATEvaluator           → model.forward(x_masked)  → logits [B,L,V]
  - utils/metrics.py          → model.count_parameters() → int
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default mask token ID, aligned with config.yaml noise_schedule.mask_token_id
_DEFAULT_MASK_TOKEN_ID: int = 0

#: Default pad token ID, aligned with config.yaml nae_sat.data.pad_token_id
_DEFAULT_PAD_TOKEN_ID: int = 4

#: Weight initialisation standard deviation (GPT-2 style)
_INIT_STD: float = 0.02

#: RoPE base frequency (standard value from Su et al. 2021)
_ROPE_BASE: float = 10000.0


# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for MDMTransformer.

    All fields correspond directly to entries in config.yaml under the
    ``model`` / ``mdm_model`` sections of each experiment.

    Attributes:
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads.
        d_model: Hidden dimension (must be divisible by n_heads).
        d_ff: Feed-forward intermediate dimension.
        vocab_size: Vocabulary size (including mask and pad tokens).
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability applied to embeddings and attention.
        pos_emb_type: Positional embedding type — ``'learned'`` or ``'rope'``.
        time_conditioned: Always ``False`` for the time-embedding-free MDM.
        mask_token_id: Token ID of the [MASK] token.
        pad_token_id: Token ID of the [PAD] token (or None to disable
            padding mask).
    """

    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    vocab_size: int = 5
    max_seq_len: int = 512
    dropout: float = 0.1
    pos_emb_type: str = "learned"   # 'learned' or 'rope'
    time_conditioned: bool = False   # always False — time-embedding-free
    mask_token_id: int = _DEFAULT_MASK_TOKEN_ID
    pad_token_id: Optional[int] = None  # set to 4 for NAE-SAT


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for transformer attention.

    Implements the rotation-based positional encoding from Su et al. (2021),
    "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    RoPE encodes position information by rotating query and key vectors in
    the attention mechanism, rather than adding positional vectors to token
    embeddings.  This gives better length generalisation than learned
    absolute positional embeddings.

    Used when ``ModelConfig.pos_emb_type == 'rope'``, as specified for the
    19M NAE-SAT MDM in config.yaml (``nae_sat.model.pos_emb_type: rope``).

    Attributes:
        d_head: Dimension of each attention head (d_model // n_heads).
        max_seq_len: Maximum sequence length for which sin/cos tables are
            precomputed.
    """

    def __init__(self, d_head: int, max_seq_len: int = 512) -> None:
        """Initialises RoPE and precomputes sin/cos frequency tables.

        Args:
            d_head: Dimension of each attention head.  Must be even.
            max_seq_len: Maximum sequence length.  Sin/cos tables are
                precomputed up to this length and cached as buffers.

        Raises:
            ValueError: If ``d_head`` is odd (RoPE requires even head dim).
        """
        super().__init__()

        if d_head % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head dimension, got d_head={d_head}."
            )

        self.d_head: int = d_head
        self.max_seq_len: int = max_seq_len

        # Precompute inverse frequency bands: theta_i = 1 / (base^(2i/d_head))
        # Shape: [d_head // 2]
        inv_freq: torch.Tensor = 1.0 / (
            _ROPE_BASE ** (
                torch.arange(0, d_head, 2, dtype=torch.float32) / d_head
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute sin/cos tables for positions 0..max_seq_len-1.
        # Shape: [max_seq_len, d_head // 2]
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Precomputes and caches sin/cos tables up to ``seq_len``.

        Args:
            seq_len: Number of positions to precompute.
        """
        positions: torch.Tensor = torch.arange(
            seq_len, dtype=torch.float32, device=self.inv_freq.device
        )
        # Outer product: [seq_len, d_head // 2]
        freqs: torch.Tensor = torch.outer(positions, self.inv_freq)
        # Concatenate to get [seq_len, d_head] (each freq appears twice)
        emb: torch.Tensor = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates the second half of the last dimension by -90 degrees.

        For a vector [x1, x2, ..., x_{d/2}, x_{d/2+1}, ..., x_d], returns
        [-x_{d/2+1}, ..., -x_d, x1, ..., x_{d/2}].

        Args:
            x: Tensor of shape ``[..., d_head]``.

        Returns:
            Rotated tensor of the same shape.
        """
        half: int = x.shape[-1] // 2
        x1: torch.Tensor = x[..., :half]
        x2: torch.Tensor = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies RoPE rotation to query and key tensors.

        Args:
            q: Query tensor of shape ``[B, n_heads, L, d_head]``.
            k: Key tensor of shape ``[B, n_heads, L, d_head]``.
            seq_len: Actual sequence length (must be <= max_seq_len).

        Returns:
            Tuple ``(q_rotated, k_rotated)`` with the same shapes as inputs.

        Raises:
            ValueError: If ``seq_len > self.max_seq_len``.
        """
        if seq_len > self.max_seq_len:
            # Extend cache on-the-fly for longer sequences.
            logger.warning(
                "RoPE: seq_len=%d exceeds precomputed max_seq_len=%d.  "
                "Extending cache.",
                seq_len,
                self.max_seq_len,
            )
            self.max_seq_len = seq_len
            self._build_cache(seq_len)

        # Slice to actual sequence length and move to correct device.
        cos: torch.Tensor = self.cos_cache[:seq_len].to(q.device)  # [L, d_head]
        sin: torch.Tensor = self.sin_cache[:seq_len].to(q.device)  # [L, d_head]

        # Reshape for broadcasting: [1, 1, L, d_head]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rotated: torch.Tensor = q * cos + self._rotate_half(q) * sin
        k_rotated: torch.Tensor = k * cos + self._rotate_half(k) * sin

        return q_rotated, k_rotated


# ---------------------------------------------------------------------------
# Custom transformer block with RoPE support
# ---------------------------------------------------------------------------


class MDMTransformerBlock(nn.Module):
    """Single transformer encoder block with optional RoPE support.

    Implements a Pre-LN transformer block:
        h = h + Attention(LayerNorm(h))
        h = h + FFN(LayerNorm(h))

    When ``rope`` is provided, applies RoPE to Q/K inside the attention
    computation.  When ``rope`` is None, uses standard absolute positional
    embeddings (added to token embeddings before the first block).

    This custom block is used when ``pos_emb_type='rope'``.  For
    ``pos_emb_type='learned'``, the standard ``nn.TransformerEncoderLayer``
    is used instead (via ``MDMTransformer._build_standard_encoder``).

    Attributes:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        d_head: Dimension per attention head (d_model // n_heads).
        rope: Optional RotaryEmbedding module.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        rope: Optional[RotaryEmbedding] = None,
    ) -> None:
        """Initialises the transformer block.

        Args:
            d_model: Hidden dimension.
            n_heads: Number of attention heads.  Must divide ``d_model``.
            d_ff: Feed-forward intermediate dimension.
            dropout: Dropout probability.
            rope: Optional ``RotaryEmbedding`` instance.  When provided,
                RoPE is applied to Q/K before computing attention scores.

        Raises:
            ValueError: If ``d_model`` is not divisible by ``n_heads``.
        """
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )

        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.d_head: int = d_model // n_heads
        self.rope: Optional[RotaryEmbedding] = rope

        # Pre-norm LayerNorms.
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)

        # Attention projections (combined QKV for efficiency).
        self.qkv_proj: nn.Linear = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj: nn.Linear = nn.Linear(d_model, d_model, bias=True)

        # Feed-forward network.
        self.ffn: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        # Dropout layers.
        self.attn_dropout: nn.Dropout = nn.Dropout(dropout)
        self.ffn_dropout: nn.Dropout = nn.Dropout(dropout)
        self.resid_dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape ``[B, L, d_model]``.
            key_padding_mask: Optional boolean tensor of shape ``[B, L]``.
                Positions marked ``True`` are ignored in attention (used to
                mask pad tokens).

        Returns:
            Output tensor of shape ``[B, L, d_model]``.
        """
        # ---- Self-attention with Pre-LN ----
        residual: torch.Tensor = x
        x_norm: torch.Tensor = self.norm1(x)

        B, L, _ = x_norm.shape

        # Compute Q, K, V via combined projection.
        qkv: torch.Tensor = self.qkv_proj(x_norm)  # [B, L, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)              # each [B, L, d_model]

        # Reshape to [B, n_heads, L, d_head] for multi-head attention.
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K if available.
        if self.rope is not None:
            q, k = self.rope.apply_rotary(q, k, seq_len=L)

        # Scaled dot-product attention (bidirectional — no causal mask).
        scale: float = math.sqrt(self.d_head)
        attn_scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / scale
        # Shape: [B, n_heads, L, L]

        # Apply key padding mask: set masked positions to -inf before softmax.
        if key_padding_mask is not None:
            # key_padding_mask: [B, L], True = ignore
            # Expand to [B, 1, 1, L] for broadcasting over heads and queries.
            mask_expanded: torch.Tensor = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded, float("-inf"))

        attn_weights: torch.Tensor = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values.
        attn_out: torch.Tensor = torch.matmul(attn_weights, v)  # [B, n_heads, L, d_head]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        attn_out = self.out_proj(attn_out)
        attn_out = self.resid_dropout(attn_out)

        x = residual + attn_out

        # ---- Feed-forward with Pre-LN ----
        residual = x
        x_norm = self.norm2(x)
        ffn_out: torch.Tensor = self.ffn(x_norm)
        ffn_out = self.ffn_dropout(ffn_out)
        x = residual + ffn_out

        return x


# ---------------------------------------------------------------------------
# MDMTransformer
# ---------------------------------------------------------------------------


class MDMTransformer(nn.Module):
    """Bidirectional transformer denoiser for Masked Diffusion Models.

    Implements p_theta(· | x_t): given a partially masked token sequence
    x_t, outputs per-position logits over the vocabulary for all positions.
    The loss (MDMLoss) supervises only the masked positions.

    This model is time-embedding-free: it receives only x_t as input, not
    the noise level t.  The number of masked tokens in x_t implicitly
    encodes t, as noted in the paper.

    The model uses bidirectional (non-causal) attention: every position
    attends to every other position, enabling full context utilisation for
    predicting masked tokens.

    Attributes:
        config: The ModelConfig used to construct this model.
        token_emb: Token embedding layer.
        pos_emb: Positional embedding (nn.Embedding for learned, None for RoPE).
        transformer: The transformer encoder (nn.TransformerEncoder for
            learned pos, nn.ModuleList of MDMTransformerBlock for RoPE).
        output_proj: Linear projection from d_model to vocab_size.
        mask_token_id: Token ID of the [MASK] token.
        pad_token_id: Token ID of the [PAD] token (or None).
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialises the MDMTransformer from a ModelConfig.

        Args:
            config: Model configuration.  All architectural hyperparameters
                are read from this object.  See ``ModelConfig`` for field
                descriptions and their correspondence to config.yaml entries.

        Raises:
            ValueError: If ``config.d_model`` is not divisible by
                ``config.n_heads``.
            ValueError: If ``config.pos_emb_type`` is not ``'learned'`` or
                ``'rope'``.
        """
        super().__init__()

        # ------------------------------------------------------------------ #
        # Validate configuration                                               #
        # ------------------------------------------------------------------ #
        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"d_model ({config.d_model}) must be divisible by "
                f"n_heads ({config.n_heads})."
            )
        if config.pos_emb_type not in ("learned", "rope"):
            raise ValueError(
                f"pos_emb_type must be 'learned' or 'rope', "
                f"got '{config.pos_emb_type}'."
            )

        self.config: ModelConfig = config
        self.mask_token_id: int = config.mask_token_id
        self.pad_token_id: Optional[int] = config.pad_token_id

        # ------------------------------------------------------------------ #
        # Token embedding                                                       #
        # ------------------------------------------------------------------ #
        self.token_emb: nn.Embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )

        # ------------------------------------------------------------------ #
        # Positional embedding                                                  #
        # ------------------------------------------------------------------ #
        self.pos_emb: Optional[nn.Embedding] = None
        self._rope: Optional[RotaryEmbedding] = None

        if config.pos_emb_type == "learned":
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            # RoPE: positional encoding is applied inside each attention block.
            d_head: int = config.d_model // config.n_heads
            self._rope = RotaryEmbedding(
                d_head=d_head,
                max_seq_len=config.max_seq_len,
            )

        # ------------------------------------------------------------------ #
        # Embedding dropout                                                     #
        # ------------------------------------------------------------------ #
        self.emb_dropout: nn.Dropout = nn.Dropout(config.dropout)

        # ------------------------------------------------------------------ #
        # Transformer encoder                                                   #
        # ------------------------------------------------------------------ #
        if config.pos_emb_type == "learned":
            self.transformer: nn.Module = self._build_standard_encoder(config)
            self._use_custom_blocks: bool = False
        else:
            # RoPE: use custom MDMTransformerBlock with shared RotaryEmbedding.
            self.transformer = nn.ModuleList(
                [
                    MDMTransformerBlock(
                        d_model=config.d_model,
                        n_heads=config.n_heads,
                        d_ff=config.d_ff,
                        dropout=config.dropout,
                        rope=self._rope,
                    )
                    for _ in range(config.n_layers)
                ]
            )
            self._use_custom_blocks = True

        # Final LayerNorm (applied after all transformer layers).
        self.final_norm: nn.LayerNorm = nn.LayerNorm(config.d_model)

        # ------------------------------------------------------------------ #
        # Output projection                                                     #
        # ------------------------------------------------------------------ #
        self.output_proj: nn.Linear = nn.Linear(
            config.d_model, config.vocab_size, bias=True
        )

        # ------------------------------------------------------------------ #
        # Weight initialisation (GPT-2 style)                                  #
        # ------------------------------------------------------------------ #
        self._init_weights()

        # Log parameter count.
        total_params: int = sum(p.numel() for p in self.parameters())
        non_emb_params: int = self.count_parameters()
        logger.info(
            "MDMTransformer initialised: n_layers=%d, n_heads=%d, "
            "d_model=%d, d_ff=%d, vocab_size=%d, max_seq_len=%d, "
            "pos_emb_type='%s', total_params=%d, non_emb_params=%d.",
            config.n_layers,
            config.n_heads,
            config.d_model,
            config.d_ff,
            config.vocab_size,
            config.max_seq_len,
            config.pos_emb_type,
            total_params,
            non_emb_params,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes per-position logits over the vocabulary.

        Takes a partially masked token sequence x_t and returns logits for
        all positions.  The MDMLoss uses only the logits at masked positions
        (where x_t == mask_token_id) to compute the training loss.

        The model uses full bidirectional attention — no causal mask is
        applied.  All positions attend to all other positions, enabling the
        model to use all available context (both masked and unmasked tokens)
        when predicting each masked token.

        Args:
            x: Integer token tensor of shape ``[B, L]``.  Values are token
                IDs including the mask token (``mask_token_id``) at positions
                that have been masked by the forward diffusion process.
                Must satisfy ``0 <= x[b, i] < vocab_size`` for all b, i.

        Returns:
            Logits tensor of shape ``[B, L, vocab_size]``.  The logits at
            unmasked positions are also computed (and can be used for
            evaluation) but are not used in the training loss.
        """
        B: int
        L: int
        B, L = x.shape
        device: torch.device = x.device

        # ---- Token embedding ----
        h: torch.Tensor = self.token_emb(x)  # [B, L, d_model]

        # ---- Positional embedding (learned only; RoPE is applied in blocks) ----
        if self.pos_emb is not None:
            positions: torch.Tensor = torch.arange(L, device=device)
            h = h + self.pos_emb(positions)  # broadcast over batch

        # ---- Embedding dropout ----
        h = self.emb_dropout(h)

        # ---- Build padding mask (True = ignore in attention) ----
        key_padding_mask: Optional[torch.Tensor] = None
        if self.pad_token_id is not None:
            key_padding_mask = (x == self.pad_token_id)  # [B, L], bool

        # ---- Transformer forward (bidirectional — no causal mask) ----
        if self._use_custom_blocks:
            # RoPE path: iterate over MDMTransformerBlock list.
            for block in self.transformer:
                h = block(h, key_padding_mask=key_padding_mask)
        else:
            # Learned pos path: use nn.TransformerEncoder.
            # src_mask=None → full bidirectional attention.
            # src_key_padding_mask: [B, L] bool, True = ignore.
            h = self.transformer(
                h,
                src_key_padding_mask=key_padding_mask,
            )

        # ---- Final LayerNorm ----
        h = self.final_norm(h)

        # ---- Output projection ----
        logits: torch.Tensor = self.output_proj(h)  # [B, L, vocab_size]

        return logits

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-position logits over the vocabulary.

        Alias for :meth:`forward` provided for clarity when callers want to
        be explicit about receiving logits rather than probabilities.

        Args:
            x: Integer token tensor of shape ``[B, L]``.

        Returns:
            Logits tensor of shape ``[B, L, vocab_size]``.
        """
        return self.forward(x)

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-position probability distributions over the vocabulary.

        Applies softmax to the logits produced by :meth:`forward`.  Used by
        ``VanillaSampler`` and ``AdaptiveSampler`` for token sampling, and
        by the oracle classes (``TopProbabilityOracle``,
        ``TopMarginOracle``) for computing certainty scores.

        The probability at the mask token position (index ``mask_token_id``)
        is included in the output.  Well-trained models assign near-zero
        probability to the mask token at masked positions (since the model
        is trained to predict the original clean token, not the mask token).

        Args:
            x: Integer token tensor of shape ``[B, L]``.

        Returns:
            Probability tensor of shape ``[B, L, vocab_size]``.  Values are
            non-negative and sum to 1 along the last dimension.
        """
        logits: torch.Tensor = self.forward(x)
        probs: torch.Tensor = F.softmax(logits, dim=-1)
        return probs

    def count_parameters(self) -> int:
        """Returns the number of non-embedding trainable parameters.

        Excludes token embedding and positional embedding parameters from
        the count, following the IsoFLOP convention of Hoffmann et al. (2022)
        and Kaplan et al. (2020).  This count is used by
        ``utils/metrics.py:count_non_embedding_params`` and
        ``utils/metrics.py:compute_isoflop_point`` for scaling law analysis.

        The formula for training tokens given a FLOP budget C is:
            training_tokens = C / (6 * count_parameters())

        Returns:
            Total number of non-embedding trainable parameters.
        """
        _EMBEDDING_KEYWORDS: tuple = ("token_emb", "pos_emb", "_rope")

        total: int = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Exclude embedding parameters.
            is_embedding: bool = any(kw in name for kw in _EMBEDDING_KEYWORDS)
            if not is_embedding:
                total += param.numel()

        return total

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_standard_encoder(self, config: ModelConfig) -> nn.TransformerEncoder:
        """Builds a standard nn.TransformerEncoder for learned positional embeddings.

        Uses Pre-LN (norm_first=True) and GELU activation, consistent with
        modern GPT-2 style architectures referenced in the paper.

        Args:
            config: Model configuration.

        Returns:
            An ``nn.TransformerEncoder`` with ``config.n_layers`` layers.
        """
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,   # input shape [B, L, d_model]
            norm_first=True,    # Pre-LN for training stability
        )
        # Disable the built-in final norm in TransformerEncoder since we
        # apply our own self.final_norm after the encoder.
        encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.n_layers,
            norm=None,  # we apply final_norm separately
        )
        return encoder

    def _init_weights(self) -> None:
        """Initialises model weights using GPT-2 style initialisation.

        - Embedding weights: Normal(0, 0.02)
        - Linear weights: Normal(0, 0.02)
        - Linear biases: zeros
        - LayerNorm weights: ones
        - LayerNorm biases: zeros

        Residual projection layers (out_proj in attention, second linear in
        FFN) use a scaled initialisation: Normal(0, 0.02 / sqrt(2 * n_layers))
        to prevent the residual stream from growing with depth.
        """
        scaled_std: float = _INIT_STD / math.sqrt(2 * self.config.n