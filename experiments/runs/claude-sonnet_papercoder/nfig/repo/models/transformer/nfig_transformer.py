## Code: models/transformer/nfig_transformer.py

```python
## models/transformer/nfig_transformer.py
"""NFIG Transformer: autoregressive image generator operating on FR-VAE tokens.

Implements the decoder-only transformer described in Section 3.2 of the NFIG
paper. The model autoregressively generates frequency-band token sequences from
low to high frequency, conditioned on class labels via AdaLN.

Autoregressive factorization (Section 3.2):
    p(T_1, T_2, ..., T_n) = Π_i p(T_i | T_1, T_2, ..., T_{i-1})

where T_i ∈ [K]^(h_i × w_i) is the token matrix for frequency band i.
All tokens within a band are predicted in parallel (next-frequency prediction),
not token-by-token. This is the key difference from raster-scan AR models.

Architecture:
    token_seqs → token_embed + pos_embed → x [B, T, D]
    class_labels → class_embed → cond [B, D]
    x, cond → 16 × AdaLNTransformerBlock (with block-wise causal attn_mask)
    → LayerNorm → head → logits [B, T, K]

Config values used (config.yaml nfig section):
    depth:               16     (number of transformer blocks)
    hidden_dim:          1024   (D — transformer hidden dimension)
    num_heads:           16     (H — number of attention heads)
    ffn_ratio:           4      (FFN intermediate = hidden_dim × ffn_ratio)
    codebook_size:       4096   (K — vocabulary size)
    scale_factors:       [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    total_tokens:        680    (sum of s_i^2)
    num_frequency_bands: 10     (n — number of autoregressive steps)
    num_classes:         1000   (ImageNet class count)
    null_class_id:       1000   (extra index for CFG unconditional pass)
    dropout:             0.0    (no dropout)
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from models.transformer.transformer_blocks import AdaLNTransformerBlock
from utils.config import NFIGConfig


class NFIGTransformer(nn.Module):
    """Frequency-aware autoregressive transformer for next-frequency prediction.

    Generates images by autoregressively predicting frequency-band token
    sequences from low to high frequency. Each band's tokens are predicted
    in parallel (given all prior bands), enforced by a block-wise causal
    attention mask.

    The model is class-conditional via AdaLN: a class embedding is computed
    from the class label and passed to every transformer block as the
    conditioning vector. The null class (id=1000) is used for CFG.

    Token sequence layout (flat, 680 tokens total):
        [band_0: 1 token | band_1: 4 tokens | ... | band_9: 256 tokens]
        Offsets: [0, 1, 5, 14, 30, 55, 91, 155, 255, 424, 680]

    Attention mask structure (block lower-triangular at band level):
        mask[p, q] = True  iff  band_of(q) <= band_of(p)
        - Intra-band: full bidirectional attention (all tokens in band i see each other)
        - Cross-band: tokens in band i see all tokens in bands 0..i-1
        - Future bands: masked out (causal constraint)

    Attributes:
        config: Stored NFIGConfig for downstream access by trainers/samplers.
        band_sizes: List of n token counts per band [s_i^2 for s_i in scale_factors].
            For default config: [1, 4, 9, 16, 25, 36, 64, 100, 169, 256].
        band_offsets: List of n+1 cumulative offsets into the flat token sequence.
            For default config: [0, 1, 5, 14, 30, 55, 91, 155, 255, 424, 680].
        token_embed: Shared token embedding nn.Embedding(codebook_size, hidden_dim).
            Maps discrete codebook indices [0, K-1] to dense vectors.
            Shared across all frequency bands (consistent with FR-VAE shared codebook).
        pos_embed: nn.ParameterList of n learned 2D positional embeddings.
            pos_embed[i] has shape (band_sizes[i], hidden_dim).
            Each band has its own positional embedding capturing its spatial structure.
        class_embed: Class conditioning embedding nn.Embedding(num_classes+1, hidden_dim).
            Maps class labels [0, 999] and null class [1000] to conditioning vectors.
        blocks: nn.ModuleList of `depth` AdaLNTransformerBlock instances.
        norm: Final LayerNorm applied before the output head.
        head: Output projection nn.Linear(hidden_dim, codebook_size, bias=False).
        attn_mask: Registered buffer of shape (total_tokens, total_tokens), dtype bool.
            Block-wise causal mask. Moves with model.to(device).
    """

    def __init__(self, config: NFIGConfig) -> None:
        """Initialize the NFIG Transformer.

        Constructs all components in the following order:
          1. Precompute band_sizes and band_offsets from scale_factors.
          2. Token embedding (shared codebook).
          3. Per-band positional embeddings (nn.ParameterList).
          4. Class conditioning embedding (includes null class).
          5. Stack of depth=16 AdaLNTransformerBlock modules.
          6. Final LayerNorm.
          7. Output head (Linear → codebook_size logits).
          8. Build and register block-wise causal attention mask as buffer.
          9. Apply weight initialization.

        Args:
            config: NFIGConfig dataclass populated from config.yaml nfig section.
                Key values:
                  - depth:               16
                  - hidden_dim:          1024
                  - num_heads:           16
                  - ffn_ratio:           4
                  - codebook_size:       4096
                  - scale_factors:       [1,2,3,4,5,6,8,10,13,16]
                  - total_tokens:        680
                  - num_frequency_bands: 10
                  - num_classes:         1000
                  - null_class_id:       1000

        Raises:
            ValueError: If config.total_tokens does not match
                sum(s*s for s in config.scale_factors).
            ValueError: If config.null_class_id != config.num_classes.
            ValueError: If config.hidden_dim is not divisible by config.num_heads.
        """
        super().__init__()

        # --- Validate critical config constraints ---
        expected_total: int = sum(s * s for s in config.scale_factors)
        if config.total_tokens != expected_total:
            raise ValueError(
                f"config.total_tokens={config.total_tokens} does not match "
                f"sum(s*s for s in scale_factors)={expected_total}. "
                f"scale_factors={config.scale_factors}"
            )

        if config.null_class_id != config.num_classes:
            raise ValueError(
                f"config.null_class_id={config.null_class_id} must equal "
                f"config.num_classes={config.num_classes}. "
                "The null class index must be exactly one past the last real class."
            )

        if config.hidden_dim % config.num_heads != 0:
            raise ValueError(
                f"config.hidden_dim={config.hidden_dim} must be divisible by "
                f"config.num_heads={config.num_heads}. "
                f"Got hidden_dim % num_heads = {config.hidden_dim % config.num_heads}."
            )

        # --- Store config ---
        self.config: NFIGConfig = config

        # ------------------------------------------------------------------ #
        # 1. Precompute band_sizes and band_offsets
        # ------------------------------------------------------------------ #
        # band_sizes[i] = scale_factors[i]^2 = number of tokens in band i.
        # For default config: [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
        self.band_sizes: List[int] = [s * s for s in config.scale_factors]

        # band_offsets[i] = starting index of band i in the flat token sequence.
        # band_offsets[n] = total_tokens (sentinel for slicing).
        # For default config: [0, 1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
        self.band_offsets: List[int] = [0]
        cumulative: int = 0
        for size in self.band_sizes:
            cumulative += size
            self.band_offsets.append(cumulative)
        # Verify total matches config.
        assert self.band_offsets[-1] == config.total_tokens, (
            f"band_offsets[-1]={self.band_offsets[-1]} != "
            f"config.total_tokens={config.total_tokens}"
        )

        # ------------------------------------------------------------------ #
        # 2. Token embedding: shared across all frequency bands
        # ------------------------------------------------------------------ #
        # Maps discrete codebook indices [0, K-1] to dense vectors of dim D.
        # Shared codebook is consistent with FR-VAE's shared codebook Z ∈ R^(K×C).
        # The same index has the same semantic meaning regardless of which band
        # it came from.
        self.token_embed: nn.Embedding = nn.Embedding(
            config.codebook_size,   # K = 4096
            config.hidden_dim,      # D = 1024
        )

        # ------------------------------------------------------------------ #
        # 3. Per-band positional embeddings
        # ------------------------------------------------------------------ #
        # Each frequency band has its own learned positional embedding because
        # the spatial structure differs per band (1×1, 2×2, ..., 16×16).
        # pos_embed[i] has shape (band_sizes[i], hidden_dim).
        # Initialized with small random values (trunc_normal, std=0.02).
        pos_embed_params: List[nn.Parameter] = []
        for i in range(config.num_frequency_bands):
            param: nn.Parameter = nn.Parameter(
                torch.zeros(self.band_sizes[i], config.hidden_dim)
            )
            nn.init.trunc_normal_(param, std=0.02)
            pos_embed_params.append(param)

        self.pos_embed: nn.ParameterList = nn.ParameterList(pos_embed_params)

        # ------------------------------------------------------------------ #
        # 4. Class conditioning embedding
        # ------------------------------------------------------------------ #
        # Maps class labels [0, num_classes-1] and null class [num_classes]
        # to conditioning vectors of dim D.
        # num_classes + 1 entries: 1000 real classes + 1 null class (id=1000).
        # The null class is used for CFG unconditional training/inference.
        self.class_embed: nn.Embedding = nn.Embedding(
            config.num_classes + 1,  # 1001 entries: 0..999 real + 1000 null
            config.hidden_dim,       # D = 1024
        )

        # ------------------------------------------------------------------ #
        # 5. Stack of AdaLN transformer blocks
        # ------------------------------------------------------------------ #
        # depth=16 blocks, each with:
        #   - BlockwiseCausalAttention (hidden_dim=1024, num_heads=16)
        #   - FFN (hidden_dim=1024, ffn_ratio=4, intermediate=4096)
        #   - AdaLN-Zero conditioning (zero-initialized for training stability)
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    ffn_ratio=float(config.ffn_ratio),
                )
                for _ in range(config.depth)
            ]
        )

        # ------------------------------------------------------------------ #
        # 6. Final LayerNorm
        # ------------------------------------------------------------------ #
        # Applied to transformer output before the head projection.
        # elementwise_affine=True (default): learned scale and bias.
        # This is separate from the AdaLN norms inside each block.
        self.norm: nn.LayerNorm = nn.LayerNorm(
            config.hidden_dim,
            elementwise_affine=True,
            eps=1e-6,
        )

        # ------------------------------------------------------------------ #
        # 7. Output head: hidden_dim → codebook_size logits
        # ------------------------------------------------------------------ #
        # Projects transformer output to logits over K=4096 codebook entries.
        # bias=False: standard for output heads in transformer language models.
        # Output shape: (B, total_tokens, codebook_size) = (B, 680, 4096).
        self.head: nn.Linear = nn.Linear(
            config.hidden_dim,    # D = 1024
            config.codebook_size, # K = 4096
            bias=False,
        )

        # ------------------------------------------------------------------ #
        # 8. Build and register block-wise causal attention mask as buffer
        # ------------------------------------------------------------------ #
        # The mask is built once in __init__ and stored as a non-parameter buffer.
        # It moves with model.to(device) and is included in state_dict.
        # Shape: (total_tokens, total_tokens) = (680, 680), dtype=bool.
        attn_mask: Tensor = self._build_attn_mask()
        self.register_buffer("attn_mask", attn_mask, persistent=True)

        # ------------------------------------------------------------------ #
        # 9. Weight initialization
        # ------------------------------------------------------------------ #
        self._init_weights()

    def _build_attn_mask(self) -> Tensor:
        """Build the static block-wise causal attention mask.

        Constructs a boolean tensor of shape (total_tokens, total_tokens) where
        entry [p, q] is True if token p is allowed to attend to token q.

        The mask enforces the NFIG autoregressive constraint:
            mask[p, q] = True  iff  band_of(q) <= band_of(p)

        This creates a block lower-triangular structure at the band level:
          - Diagonal blocks (same band): all True — full intra-band attention.
          - Lower-triangular blocks (earlier band): all True — attend to all
            tokens from all previous frequency bands.
          - Upper-triangular blocks (later band): all False — causal constraint.

        The mask is built on CPU and moved to the target device when the model
        is moved via model.to(device) (handled by register_buffer).

        Returns:
            Boolean tensor of shape (total_tokens, total_tokens) on CPU.
            Entry [p, q] = True means token p can attend to token q.
            total_tokens = 680 for the default config.
        """
        total_tokens: int = self.config.total_tokens
        n_bands: int = self.config.num_frequency_bands

        # Build a band_id tensor: band_id[p] = i if token p belongs to band i.
        # Shape: (total_tokens,), dtype=long.
        # This vectorized approach is more efficient than nested loops.
        band_id: Tensor = torch.zeros(total_tokens, dtype=torch.long)
        for band_idx in range(n_bands):
            start: int = self.band_offsets[band_idx]
            end: int = self.band_offsets[band_idx + 1]
            band_id[start:end] = band_idx

        # mask[p, q] = True iff band_id[q] <= band_id[p].
        # Vectorized: compare (total_tokens, 1) against (1, total_tokens).
        # band_id_row: (total_tokens, 1) — band index of each query token
        # band_id_col: (1, total_tokens) — band index of each key token
        band_id_row: Tensor = band_id.unsqueeze(1)  # (total_tokens, 1)
        band_id_col: Tensor = band_id.unsqueeze(0)  # (1, total_tokens)

        # mask[p, q] = True iff band_of(key q) <= band_of(query p)
        mask: Tensor = band_id_col <= band_id_row  # (total_tokens, total_tokens)

        return mask

    def _init_weights(self) -> None:
        """Initialize model weights following standard transformer practice.

        Initialization strategy:
          - Token embedding: trunc_normal(std=0.02) — standard for ViT/GPT
          - Class embedding: trunc_normal(std=0.02)
          - Positional embeddings: already initialized in __init__ with trunc_normal
          - Output head: zero initialization for training stability
          - Final LayerNorm: weight=1, bias=0 (PyTorch default, kept as-is)
          - AdaLNTransformerBlock weights: initialized internally by each block
            (AdaLN-Zero for modulation, standard for attention/FFN)

        The output head zero-initialization ensures logits start near zero,
        which gives a near-uniform initial distribution over the codebook.
        This is a stable starting point for cross-entropy training.
        """
        # Token embedding: trunc_normal with std=0.02 (standard for transformers).
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)

        # Class embedding: trunc_normal with std=0.02.
        # Includes the null class embedding at index num_classes=1000.
        nn.init.trunc_normal_(self.class_embed.weight, std=0.02)

        # Output head: zero initialization.
        # Ensures logits start near zero → near-uniform initial distribution.
        # The head will quickly learn to differentiate tokens via gradient descent.
        nn.init.zeros_(self.head.weight)

        # Final LayerNorm: PyTorch default (weight=1, bias=0) is already correct.
        # Explicit reset for clarity and reproducibility.
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def _get_positional_embeddings(self, scale_idx: int) -> Tensor:
        """Return the positional embedding for a specific frequency band.

        Each frequency band has its own learned positional embedding that
        captures the 2D spatial structure at that band's resolution.

        Args:
            scale_idx: Index into self.pos_embed (0-based band index).
                Must be in [0, num_frequency_bands - 1].

        Returns:
            Positional embedding tensor of shape (band_sizes[scale_idx], hidden_dim).
            For band 0 (1×1): shape (1, 1024).
            For band 9 (16×16): shape (256, 1024).

        Raises:
            IndexError: If scale_idx is out of range.
        """
        if scale_idx < 0 or scale_idx >= self.config.num_frequency_bands:
            raise IndexError(
                f"scale_idx={scale_idx} is out of range "
                f"[0, {self.config.num_frequency_bands - 1}]."
            )
        return self.pos_embed[scale_idx]

    def _prepare_input(
        self,
        token_seqs: List[Tensor],
    ) -> Tuple[Tensor, int]:
        """Convert per-band token index tensors into an embedded sequence.

        For each available band i:
          1. Look up token embeddings: token_embed(token_seqs[i]) → (B, band_size_i, D)
          2. Add positional embedding: + pos_embed[i] → (B, band_size_i, D)
        Concatenate all bands along the sequence dimension → (B, current_len, D).

        Handles variable-length input (len(token_seqs) <= num_frequency_bands)
        for inference-time autoregressive generation where bands are added one
        at a time.

        Args:
            token_seqs: List of up to n=10 integer tensors.
                token_seqs[i] has shape (B, band_sizes[i]) with values in [0, K-1].
                The list may contain fewer than n bands during inference.
                During training, all n=10 bands are present.

        Returns:
            Tuple of:
                - x: Embedded sequence tensor of shape (B, current_len, hidden_dim).
                  current_len = sum(band_sizes[0..len(token_seqs)-1]).
                  For full sequence (training): (B, 680, 1024).
                - current_len: Integer sequence length (for attention mask slicing).

        Raises:
            ValueError: If token_seqs is empty.
            ValueError: If any token_seqs[i] has incorrect shape.
        """
        if not token_seqs:
            raise ValueError(
                "token_seqs must be a non-empty list. "
                "At least one frequency band must be provided."
            )

        n_available: int = len(token_seqs)
        if n_available > self.config.num_frequency_bands:
            raise ValueError(
                f"len(token_seqs)={n_available} exceeds "
                f"num_frequency_bands={self.config.num_frequency_bands}. "
                "Cannot have more token sequences than frequency bands."
            )

        # Validate shapes and infer batch size from the first tensor.
        B: int = token_seqs[0].shape[0]
        for band_idx, tokens in enumerate(token_seqs):
            expected_len: int = self.band_sizes[band_idx]
            if tokens.shape[0] != B:
                raise ValueError(
                    f"token_seqs[{band_idx}].shape[0]={tokens.shape[0]} does not "
                    f"match batch size B={B} from token_seqs[0]."
                )
            if tokens.shape[1] != expected_len:
                raise ValueError(
                    f"token_seqs[{band_idx}].shape[1]={tokens.shape[1]} does not "
                    f"match expected band_sizes[{band_idx}]={expected_len}. "
                    f"scale_factors[{band_idx}]={self.config.scale_factors[band_idx]}, "
                    f"expected {self.config.scale_factors[band_idx]}^2={expected_len} tokens."
                )

        # Build embedded sequence by processing each band.
        band_embeddings: List[Tensor] = []
        for band_idx in range(n_available):
            tokens_i: Tensor = token_seqs[band_idx]  # (B, band_size_i), dtype=long

            # Token embedding lookup: (B, band_size_i) → (B, band_size_i, D)
            token_emb_i: Tensor = self.token_embed(tokens_i)  # (B, band_size_i, D)

            # Add positional embedding: (band_size_i, D) → broadcast to (B, band_size_i, D)
            # pos_embed[band_idx] has shape (band_size_i, D).
            # Unsqueeze(0) adds batch dimension for broadcasting.
            pos_emb_i: Tensor = self.pos_embed[band_idx].unsqueeze(0)  # (1, band_size_i, D)
            emb_i: Tensor = token_emb_i + pos_emb_i  # (B, band_size_i, D)

            band_embeddings.append(emb_i)

        # Concatenate all band embeddings along the sequence dimension.
        # (B, band_size_0, D), (B, band_size_1, D), ... → (B, current_len, D)
        x: Tensor = torch.cat(band_embeddings, dim=1)  # (B, current_len, D)

        current_len: int = x.shape[1]

        return x, current_len

    def forward(
        self,
        token_seqs: List[Tensor],
        class_labels: Tensor,
    ) -> Tensor:
        """Autoregressive forward pass through the NFIG Transformer.

        Processes a (partial) token sequence through the full transformer stack
        and returns logits over the codebook vocabulary for every token position.

        Training usage:
            - token_seqs contains all 10 bands (680 tokens total).
            - Logits at positions [band_offsets[i] : band_offsets[i+1]] predict
              band i's tokens given bands 0..i-1 (enforced by attn_mask).
            - NFIGTrainer computes cross-entropy loss across all 680 positions.

        Inference usage (called by NFIGSampler):
            - At step i, token_seqs contains bands 0..i-1 (growing sequence).
            - Only logits at positions [band_offsets[i] : band_offsets[i+1]]
              are used to sample band i's tokens.
            - The attention mask is sliced to match the current sequence length.

        CFG usage:
            - Two forward passes: one with real class labels, one with null class.
            - NFIGSampler handles CFG combination externally.
            - This method is called twice per generation step during inference.

        Args:
            token_seqs: List of up to n=10 integer tensors.
                token_seqs[i] has shape (B, band_sizes[i]) with values in [0, K-1].
                During training: all 10 bands present (680 tokens total).
                During inference: bands 0..i-1 present at step i.
                dtype: torch.long (integer token indices).
            class_labels: Class label tensor of shape (B,), dtype torch.long.
                Values in [0, num_classes-1] for conditional generation.
                Value num_classes (=1000) for unconditional (CFG null class).
                From config.nfig.num_classes=1000, config.nfig.null_class_id=1000.

        Returns:
            Logits tensor of shape (B, current_len, codebook_size).
            For full sequence (training): (B, 680, 4096).
            For partial sequence (inference at step i):
                (B, sum(band_sizes[0..i-1]), 4096).
            Raw (unnormalized) logits — apply softmax or top-k sampling externally.

        Raises:
            ValueError: If token_seqs is empty or has incorrect shapes.
            RuntimeError: If class_labels has incorrect shape or dtype.
        """
        # --- Validate class_labels ---
        if class_labels.dim() != 1:
            raise RuntimeError(
                f"class_labels must be a 1D tensor of shape (B,), "
                f"got shape {tuple(class_labels.shape)}."
            )

        B: int = class_labels.shape[0]

        # ------------------------------------------------------------------ #
        # Step 1: Class conditioning vector
        # ------------------------------------------------------------------ #
        # class_embed maps class labels [0, num_classes] to dense vectors.
        # cond: (B,) → (B, hidden_dim) = (B, 1024)
        # The conditioning vector is passed to every AdaLNTransformerBlock.
        cond: Tensor = self.class_embed(class_labels)  # (B, D)

        # ------------------------------------------------------------------ #
        # Step 2: Token + positional embedding
        # ------------------------------------------------------------------ #
        # _prepare_input handles:
        #   - Token embedding lookup per band
        #   - Positional embedding addition per band
        #   - Concatenation into a single sequence tensor
        # x: (B, current_len, D), current_len <= total_tokens
        x: Tensor
        current_len: int
        x, current_len = self._prepare_input(token_seqs)

        # Validate batch size consistency.
        if x.shape[0] != B:
            raise RuntimeError(
                f"Batch size mismatch: class_labels has B={B} but "
                f"token_seqs implies B={x.shape[0]}."
            )

        # ------------------------------------------------------------------ #
        # Step 3: Slice attention mask to current sequence length
        # ------------------------------------------------------------------ #
        # During training: current_len = total_tokens = 680, use full mask.
        # During inference: current_len < total_tokens, slice to [0:current_len, 0:current_len].
        # The sliced mask preserves the block-wise causal structure for the
        # available bands.
        # self.attn_mask: (total_tokens, total_tokens) bool, registered buffer.
        attn_mask: Tensor = self.attn_mask[:current_len, :current_len]  # type: ignore[index]
        # attn_mask: (current_len, current_len) bool

        # ------------------------------------------------------------------ #
        # Step 4: Pass through transformer blocks
        # ------------------------------------------------------------------ #
        # Each AdaLNTransformerBlock applies:
        #   - AdaLN-modulated pre-norm
        #   - BlockwiseCausalAttention with the frequency-band causal mask
        #   - AdaLN-modulated pre-norm
        #   - FFN (Linear → GELU → Linear)
        # All with gated residual connections (AdaLN-Zero initialization).
        block: AdaLNTransformerBlock
        for block in self.blocks:
            x = block.forward(x, cond, attn_mask)
        # x: (B, current_len, D)

        # ------------------------------------------------------------------ #
        # Step 5: Final layer normalization
        # ------------------------------------------------------------------ #
        # Applied to the full sequence output before the head projection.
        # This is a standard transformer output norm (separate from AdaLN norms
        # inside each block).
        x = self.norm(x)  # (B, current_len, D)

        # ------------------------------------------------------------------ #
        # Step 6: Project to codebook logits
        # ------------------------------------------------------------------ #
        # head: Linear(hidden_dim, codebook_size, bias=False)
        # Maps each token position's hidden state to K=4096 logits.
        logits: Tensor = self.head(x)  # (B, current_len, K)

        return logits

    def get_band_logits(
        self,
        logits: Tensor,
        band_idx: int,
    ) -> Tensor:
        """Extract logits for a specific frequency band from the full logit tensor.

        Convenience method for NFIGSampler to extract the logits corresponding
        to a specific frequency band from the transformer's output.

        Args:
            logits: Full logit tensor of shape (B, current_len, codebook_size).
                Output of forward(). current_len must be >= band_offsets[band_idx+1].
            band_idx: Index of the frequency band to extract (0-based).
                Must be in [0, num_frequency_bands - 1].

        Returns:
            Logit tensor for band band_idx of shape (B, band_sizes[band_idx], codebook_size).
            For band 0 (1×1): shape (B, 1, 4096).
            For band 9 (16×16): shape (B, 256, 4096).

        Raises:
            IndexError: If band_idx is out of range.
            ValueError: If logits has insufficient sequence length for band_idx.
        """
        if band_idx < 0 or band_idx >= self.config.num_frequency_bands:
            raise IndexError(
                f"band_idx={band_idx} is out of range "
                f"[0, {self.config.num_frequency_bands - 1}]."
            )

        start: int = self.