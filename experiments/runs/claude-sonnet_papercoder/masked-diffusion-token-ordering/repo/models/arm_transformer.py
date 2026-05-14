## Code: models/arm_transformer.py

```python
## models/arm_transformer.py
"""Causal Autoregressive Transformer (ARM) for baseline experiments.

Implements the ARMTransformer used in two distinct roles:

1. **Logic puzzle baselines (Sections 4.2–4.3, Tables 2–3):**
   A 42M causal transformer trained on Sudoku/Zebra data in two variants:
   - ARM without ordering: standard left-to-right causal LM training.
   - ARM with ordering: teacher-forced permutation order from Shah et al.
     (2024), where the model learns the correct token generation order.

2. **π-learner for scaling law experiments (Section 3.2, Fig. 2 left):**
   A causal transformer trained on permuted text data π(x₀) with learnable
   positional embeddings. The π-learner is architecturally identical to the
   standard ARM — the permutation is applied to the input before the forward
   pass, and the causal mask handles the rest correctly.

Key architectural properties:
  - Causal (lower-triangular) attention mask — position i cannot attend to
    positions j > i.
  - Pre-norm (norm_first=True) for training stability.
  - Supports both learned positional embeddings and RoPE.
  - Time-embedding-free (no noise level conditioning).

Architecture variants by experiment (from config.yaml):
  - 42M Sudoku ARM:  8 layers, 8 heads, d_model=512, d_ff=2048, learned pos
  - 42M Zebra ARM:   8 layers, 8 heads, d_model=512, d_ff=2048, learned pos
  - Scaling law π-learner: variable size, learned pos (all experiments)

Integration points:
  - ARMTrainer.train_step()     → model.forward(x)              → logits [B,L,V]
  - ARMTrainer (order-aware)    → model.forward(x_permuted)     → logits [B,L,V]
  - PuzzleEvaluator             → model.generate(x_masked, ...)  → tokens [B,L]
  - utils/metrics.py            → model.pi_learner_forward(x, pi) → logits [B,L,V]
  - utils/metrics.py            → model.count_parameters()       → int
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import ModelConfig from mdm_transformer to share the same config dataclass.
# This avoids duplication and ensures both models use the same configuration
# interface, as specified in the design.
from models.mdm_transformer import ModelConfig, RotaryEmbedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default mask token ID, aligned with config.yaml noise_schedule.mask_token_id
_DEFAULT_MASK_TOKEN_ID: int = 0

#: Weight initialisation standard deviation (GPT-2 style)
_INIT_STD: float = 0.02

#: Large negative value used as -inf in causal mask (avoids true -inf NaN issues)
_NEG_INF: float = float("-inf")


# ---------------------------------------------------------------------------
# Custom causal attention block with RoPE support
# ---------------------------------------------------------------------------


class CausalSelfAttentionBlock(nn.Module):
    """Single causal transformer block with optional RoPE support.

    Implements a Pre-LN causal transformer block:
        h = h + CausalAttention(LayerNorm(h))
        h = h + FFN(LayerNorm(h))

    The causal mask is passed in from the parent module (computed once per
    forward pass and reused across all blocks for efficiency).

    When ``rope`` is provided, applies RoPE to Q/K inside the attention
    computation.  When ``rope`` is None, relies on absolute positional
    embeddings added to token embeddings before the first block.

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
        """Initialises the causal transformer block.

        Args:
            d_model: Hidden dimension.
            n_heads: Number of attention heads.  Must divide ``d_model``.
            d_ff: Feed-forward intermediate dimension.
            dropout: Dropout probability applied to attention weights and
                residual connections.
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

        # Combined QKV projection for efficiency.
        self.qkv_proj: nn.Linear = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj: nn.Linear = nn.Linear(d_model, d_model, bias=True)

        # Feed-forward network: Linear → GELU → Linear.
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
        causal_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the causal transformer block.

        Args:
            x: Input tensor of shape ``[B, L, d_model]``.
            causal_mask: Additive causal attention mask of shape ``[L, L]``.
                Values are 0 (attend) or ``-inf`` (do not attend).
                Position i cannot attend to positions j > i.
            key_padding_mask: Optional boolean tensor of shape ``[B, L]``.
                Positions marked ``True`` are ignored in attention (used to
                mask pad tokens).  Not used in puzzle experiments but
                included for completeness.

        Returns:
            Output tensor of shape ``[B, L, d_model]``.
        """
        B: int
        L: int
        B, L, _ = x.shape

        # ---- Causal self-attention with Pre-LN ----
        residual: torch.Tensor = x
        x_norm: torch.Tensor = self.norm1(x)

        # Compute Q, K, V via combined projection.
        qkv: torch.Tensor = self.qkv_proj(x_norm)  # [B, L, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)              # each [B, L, d_model]

        # Reshape to [B, n_heads, L, d_head] for multi-head attention.
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, d_head]
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, d_head]
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, d_head]

        # Apply RoPE to Q and K if available.
        if self.rope is not None:
            q, k = self.rope.apply_rotary(q, k, seq_len=L)

        # Scaled dot-product attention with causal mask.
        scale: float = math.sqrt(self.d_head)
        attn_scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / scale
        # Shape: [B, n_heads, L, L]

        # Add causal mask: shape [L, L] broadcasts over [B, n_heads, L, L].
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply key padding mask if provided.
        if key_padding_mask is not None:
            # key_padding_mask: [B, L], True = ignore.
            # Expand to [B, 1, 1, L] for broadcasting.
            mask_expanded: torch.Tensor = (
                key_padding_mask.unsqueeze(1).unsqueeze(2)
            )
            attn_scores = attn_scores.masked_fill(mask_expanded, _NEG_INF)

        attn_weights: torch.Tensor = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values.
        attn_out: torch.Tensor = torch.matmul(attn_weights, v)  # [B, H, L, d_head]
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
# ARMTransformer
# ---------------------------------------------------------------------------


class ARMTransformer(nn.Module):
    """Causal autoregressive transformer for ARM baseline experiments.

    Implements a standard causal language model that predicts token x^i
    given all preceding tokens x^0, ..., x^{i-1} in the generation order.

    For standard left-to-right training (ARM without ordering), the input
    is the original sequence and the causal mask enforces left-to-right
    attention.

    For order-aware training (ARM with ordering, Shah et al. 2024), the
    input is permuted by ``ARMTrainer`` before calling ``forward``, so the
    model sees [x^{π(0)}, x^{π(1)}, ..., x^{π(L-1)}] and predicts each
    token from its left context in the permuted order.

    For π-learner scaling law experiments (Section 3.2), use
    ``pi_learner_forward(x, pi)`` which handles the permutation internally.

    Attributes:
        config: The ModelConfig used to construct this model.
        token_emb: Token embedding layer.
        pos_emb: Learned positional embedding (None when using RoPE).
        blocks: List of CausalSelfAttentionBlock transformer layers.
        final_norm: LayerNorm applied after all transformer layers.
        output_proj: Linear projection from d_model to vocab_size.
        mask_token_id: Token ID of the [MASK] token.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialises the ARMTransformer from a ModelConfig.

        Args:
            config: Model configuration.  All architectural hyperparameters
                are read from this object.  See ``ModelConfig`` for field
                descriptions and their correspondence to config.yaml entries.

                Key config.yaml mappings:
                  - sudoku.arm_model.n_layers = 8
                  - sudoku.arm_model.n_heads = 8
                  - sudoku.arm_model.d_model = 512
                  - sudoku.arm_model.d_ff = 2048
                  - sudoku.arm_model.dropout = 0.1
                  - sudoku.arm_model.pos_emb_type = 'learned'
                  - sudoku.arm_model.max_seq_len = 81
                  - scaling_law.pos_emb_type = 'learned'

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
            # Learned absolute positional embeddings.
            # Paper (Appendix C.1): "we employ a learnable positional embedding
            # layer for all experiments to correct this [RoPE inductive bias]"
            # for scaling law experiments.  Puzzle experiments also use learned
            # embeddings per config.yaml sudoku.arm_model.pos_emb_type='learned'.
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            # RoPE: positional encoding applied inside each attention block.
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
        # Transformer blocks (causal)                                           #
        # ------------------------------------------------------------------ #
        # Use custom CausalSelfAttentionBlock for both learned and RoPE paths.
        # This gives us full control over the causal mask and RoPE application,
        # and avoids the complexity of wrapping nn.TransformerEncoder with a
        # custom attention mechanism for RoPE.
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                CausalSelfAttentionBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    rope=self._rope,  # None for learned pos, RotaryEmbedding for RoPE
                )
                for _ in range(config.n_layers)
            ]
        )

        # ------------------------------------------------------------------ #
        # Final LayerNorm (applied after all transformer layers)               #
        # ------------------------------------------------------------------ #
        self.final_norm: nn.LayerNorm = nn.LayerNorm(config.d_model)

        # ------------------------------------------------------------------ #
        # Output projection: d_model → vocab_size                              #
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
            "ARMTransformer initialised: n_layers=%d, n_heads=%d, "
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
        """Computes per-position logits over the vocabulary (causal LM).

        Applies a causal (lower-triangular) attention mask so that position
        i can only attend to positions 0..i (inclusive).  This implements
        the standard autoregressive factorisation:

            log p_θ(x) = Σᵢ log p_θ(x^i | x^0, ..., x^{i-1})

        For order-aware ARM training, the caller (``ARMTrainer``) permutes
        the input sequence before calling this method, so the model sees
        [x^{π(0)}, ..., x^{π(L-1)}] and the causal mask enforces the
        permuted generation order.

        Args:
            x: Integer token tensor of shape ``[B, L]``.  Values are token
                IDs in ``[0, vocab_size)``.  For standard training, this is
                the original sequence.  For order-aware training, this is
                the permuted sequence ``x[:, pi]``.

        Returns:
            Logits tensor of shape ``[B, L, vocab_size]``.  The logit at
            position i predicts the token at position i given positions
            0..i-1 (in the input order, which may be permuted).

            For computing the training loss, use:
                ``F.cross_entropy(logits[:, :-1].reshape(-1, V),
                                  x[:, 1:].reshape(-1))``
        """
        B: int
        L: int
        B, L = x.shape
        device: torch.device = x.device

        # ---- Token embedding ----
        h: torch.Tensor = self.token_emb(x)  # [B, L, d_model]

        # ---- Positional embedding (learned only; RoPE applied in blocks) ----
        if self.pos_emb is not None:
            positions: torch.Tensor = torch.arange(L, device=device)
            h = h + self.pos_emb(positions)  # broadcast over batch: [B, L, d_model]

        # ---- Embedding dropout ----
        h = self.emb_dropout(h)

        # ---- Build causal mask ----
        # causal_mask[i, j] = -inf if j > i (future), 0 if j <= i (past/current)
        # Shape: [L, L], additive mask for attention logits.
        causal_mask: torch.Tensor = self._build_causal_mask(L, device=device)

        # ---- Transformer forward (causal) ----
        for block in self.blocks:
            h = block(h, causal_mask=causal_mask)

        # ---- Final LayerNorm ----
        h = self.final_norm(h)

        # ---- Output projection ----
        logits: torch.Tensor = self.output_proj(h)  # [B, L, vocab_size]

        return logits

    def generate(
        self,
        prompt: torch.Tensor,
        max_len: int,
        order: Optional[torch.Tensor] = None,
        fixed_mask: Optional[torch.Tensor] = None,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generates a complete token sequence autoregressively.

        Handles two generation modes:

        **Without ordering (``order=None``):**
        Standard left-to-right autoregressive generation.  Starts from the
        prompt (known prefix or puzzle clues), iteratively predicts the next
        token, and fills positions left-to-right until ``max_len`` tokens
        are generated.

        **With ordering (``order`` is a permutation tensor):**
        Order-aware generation following Shah et al. (2024).  Generates
        tokens in the order specified by ``order[0], order[1], ...,
        order[max_len-1]``.  At step i, the model has access to all
        previously generated tokens (at positions ``order[0..i-1]``) and
        predicts the token at position ``order[i]``.

        Fixed positions (puzzle clues) are never overwritten — they are
        treated as given context throughout generation.

        Args:
            prompt: Initial token tensor of shape ``[B, max_len]``.
                Known positions (puzzle clues) contain their digit values
                (1–9 for Sudoku); unknown positions contain
                ``mask_token_id = 0``.  The full sequence buffer is
                pre-allocated to ``max_len`` tokens.
            max_len: Total sequence length to generate.  For Sudoku: 81.
                For Zebra: 25.
            order: Optional permutation tensor of shape ``[max_len]``
                (dtype ``torch.long``) specifying the generation order.
                ``order[i]`` is the sequence position to generate at step i.
                When ``None``, generates left-to-right.
            fixed_mask: Optional boolean tensor of shape ``[B, max_len]``.
                ``True`` at positions that are fixed (puzzle clues) and
                should never be overwritten.  When ``None``, inferred from
                ``prompt != mask_token_id``.
            greedy: If ``True`` (default), uses argmax decoding.  If
                ``False``, samples from the predicted distribution with
                the given ``temperature``.
            temperature: Sampling temperature (only used when
                ``greedy=False``).  Higher values increase diversity.

        Returns:
            Completed token tensor of shape ``[B, max_len]`` with all
            positions filled.  Fixed positions retain their original values
            from ``prompt``.

        Note:
            This method calls ``model.eval()`` internally and wraps the
            generation loop in ``torch.no_grad()`` for efficiency.  The
            model is restored to its original training/eval state after
            generation.
        """
        B: int = prompt.shape[0]
        device: torch.device = prompt.device

        # Infer fixed mask from prompt if not provided.
        if fixed_mask is None:
            fixed_mask = (prompt != self.mask_token_id)  # [B, max_len], bool

        # Initialise output buffer with the prompt (clues already in place).
        output: torch.Tensor = prompt.clone()  # [B, max_len]

        # Save and set eval mode for generation.
        was_training: bool = self.training
        self.eval()

        with torch.no_grad():
            if order is None:
                output = self._generate_left_to_right(
                    output=output,
                    fixed_mask=fixed_mask,
                    max_len=max_len,
                    greedy=greedy,
                    temperature=temperature,
                    device=device,
                )
            else:
                output = self._generate_with_order(
                    output=output,
                    fixed_mask=fixed_mask,
                    order=order,
                    max_len=max_len,
                    greedy=greedy,
                    temperature=temperature,
                    device=device,
                )

        # Restore original training state.
        if was_training:
            self.train()

        return output

    def pi_learner_forward(
        self,
        x: torch.Tensor,
        pi: torch.Tensor,
    ) -> torch.Tensor:
        """Computes π-learner logits for scaling law experiments (Section 3.2).

        A π-learner is a causal ARM trained on permuted sequences π(x₀).
        This method permutes the input sequence according to ``pi`` and
        runs a standard causal forward pass, returning logits in the
        permuted order.

        The π-learner loss is:
            L_π = -Σᵢ log p_θ(x₀^{π(i)} | x₀^{π(0)}, ..., x₀^{π(i-1)})

        which equals the standard causal LM cross-entropy on the permuted
        sequence x_permuted = x[:, pi]:
            L_π = cross_entropy(logits[:, :-1], x_permuted[:, 1:])

        This equivalence holds because the causal mask enforces that position
        i in the permuted sequence attends only to positions 0..i-1 in the
        permuted sequence, which corresponds exactly to the π-learner
        conditioning set {x₀^{π(0)}, ..., x₀^{π(i-1)}}.

        Args:
            x: Clean token sequence(s) of shape ``[B, L]`` or ``[L]``.
                Values are token IDs in ``[0, vocab_size)``.
            pi: Permutation tensor of shape ``[L]``, dtype ``torch.long``.
                ``pi[i]`` is the original sequence position that appears at
                position i in the permuted sequence.

        Returns:
            Logits tensor of shape ``[B, L, vocab_size]`` in permuted order.
            The logit at position i predicts x₀^{π(i+1)} given
            x₀^{π(0)}, ..., x₀^{π(i)}.

            To compute the π-learner NLL:
                ``F.cross_entropy(logits[:, :-1].reshape(-1, V),
                                  x_permuted[:, 1:].reshape(-1))``
            where ``x_permuted = x[:, pi]``.

        Note:
            This method does NOT call ``model.eval()`` — the caller is
            responsible for setting the appropriate mode.  In
            ``utils/metrics.py:compute_pi_learner_loss``, the model is
            already in eval mode via ``model.eval()`` + ``torch.no_grad()``.
        """
        # Normalise to [B, L].
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size: int = x.shape[0]

        # Permute each sequence in the batch: x_permuted[b, i] = x[b, pi[i]]
        # pi has shape [L]; expand to [B, L] for gather.
        pi_expanded: torch.Tensor = pi.unsqueeze(0).expand(batch_size, -1)  # [B, L]
        x_permuted: torch.Tensor = torch.gather(x, dim=1, index=pi_expanded)  # [B, L]

        # Standard causal forward pass on the permuted sequence.
        logits: torch.Tensor = self.forward(x_permuted)  # [B, L, vocab_size]

        return logits

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
            is_embedding: bool = any(kw in name for kw in _EMBEDDING_KEYWORDS)
            if not is_embedding:
                total += param.numel()

        return total

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Builds an additive causal attention mask of shape ``[L, L]``.

        The mask is upper-triangular with ``-inf`` above the diagonal:
            mask[i, j] = -inf  if j > i  (future positions, blocked)
            mask[i, j] = 0     if j <= i (past/current positions, allowed)

        This is an additive mask: it is added to the raw attention logits
        before softmax, so ``-inf`` entries become 0 after softmax.

        Args:
            seq_len: Sequence length L.
            device: Target device for the mask tensor.

        Returns:
            Float tensor of shape ``[seq_len, seq_len]``.
        """
        # torch.triu with diagonal=1 sets the upper triangle (j > i) to 1,
        # lower triangle and diagonal to 0.  Multiplying by -inf gives the
        # additive causal mask.
        mask: torch.Tensor = torch.triu(
            torch.full((seq_len, seq_len), _NEG_INF, device=device),
            diagonal=1,
        )
        return mask

    def _generate_left_to_right(
        self,
        output: torch.Tensor,
        fixed_mask: torch.Tensor,
        max_len: int,
        greedy: bool,
        temperature: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Generates tokens left-to-right, skipping fixed (clue) positions.

        Iterates over positions 0..max_len-1 in order.  At each position:
        - If the position is fixed (puzzle clue), skip it.
        - Otherwise, run a forward pass on the current sequence and predict
          the token at this position from its left context.

        This is efficient because we only run one forward pass per unknown
        position, using the full current sequence as context.

        Args:
            output: Current sequence buffer, shape ``[B, max_len]``.
                Fixed positions already contain their clue values.
            fixed_mask: Boolean tensor ``[B, max_len]``, True = fixed.
            max_len: Total sequence length.
            greedy: Use argmax if True, else