import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import AdaLN, FeedForward


def build_block_causal_mask(scale_factors: List[int], device: torch.device) -> torch.Tensor:
    """
    Build block-wise causal attention mask for frequency-ordered generation.

    Tokens within the same frequency band can attend to each other (full attention),
    and all tokens can attend to tokens from previous (lower-frequency) bands.
    This is the block-wise causal attention from VAR [19], adapted for frequency bands.

    Args:
        scale_factors: list of scale factors, e.g. [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
        device: target device

    Returns:
        (L, L) boolean mask where True means "can attend"
    """
    token_counts = [s * s for s in scale_factors]
    L = sum(token_counts)
    mask = torch.zeros(L, L, dtype=torch.bool, device=device)

    start = 0
    for count in token_counts:
        end = start + count
        # All tokens up to and including current band can be attended to
        mask[start:end, :end] = True
        start = end

    return mask


class BlockCausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with block-wise causal masking.

    Tokens in frequency band i can attend to all tokens in bands 1..i.
    Within band i, all tokens attend to each other (non-causal within band).
    """

    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = attn_dropout

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            attn_mask: (L, L) boolean mask, True = can attend

        Returns:
            (B, L, D)
        """
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, L, H, head_dim)
        q = q.transpose(1, 2)  # (B, H, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Convert boolean mask to additive mask for scaled_dot_product_attention
        if attn_mask is not None:
            # True = can attend, False = masked out
            additive_mask = torch.zeros(L, L, device=x.device, dtype=x.dtype)
            additive_mask.masked_fill_(~attn_mask, float("-inf"))
            additive_mask = additive_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        else:
            additive_mask = None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=additive_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """
    Transformer block with AdaLN conditioning and block-wise causal attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = AdaLN(dim, cond_dim)
        self.attn = BlockCausalSelfAttention(dim, num_heads, attn_dropout)
        self.ff_norm = AdaLN(dim, cond_dim)
        self.ff = FeedForward(dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x, cond), attn_mask)
        x = x + self.ff(self.ff_norm(x, cond))
        return x


class NFIGTransformer(nn.Module):
    """
    Next-Frequency Image Generation Transformer (Section 3.2).

    Decoder-only transformer that autoregressively generates token sequences
    from low to high frequency bands. Uses block-wise causal attention so
    each frequency band can attend to all previous bands.

    Generation factorization (Eq. 5):
        p(T_1, ..., T_n) = prod_i p(T_i | T_1, ..., T_{i-1})

    Architecture:
    - Class embedding + frequency-level positional embeddings
    - Token embeddings from shared codebook
    - Stack of transformer blocks with AdaLN
    - Per-token logit head
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        num_classes: int = 1000,
        depth: int = 16,
        embed_dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        scale_factors: List[int] = None,
    ):
        super().__init__()
        if scale_factors is None:
            scale_factors = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.scale_factors = scale_factors
        self.token_counts = [s * s for s in scale_factors]
        self.total_tokens = sum(self.token_counts)
        self.n_levels = len(scale_factors)

        # Class conditioning: embed class label + null class for CFG
        # null class = num_classes (index)
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)

        # Token embedding (shared codebook lookup)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Learnable start-of-sequence token for each frequency level
        # Used as the "prefix" before generating each band's tokens
        self.level_start_tokens = nn.Parameter(
            torch.randn(self.n_levels, embed_dim) * 0.02
        )

        # Positional embeddings: one per token position within each level
        # We use separate positional embeddings per level
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(1, s * s, embed_dim) * 0.02)
            for s in scale_factors
        ])

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Precompute block-causal mask (registered as buffer)
        # Will be built on first forward pass or explicitly
        self._attn_mask: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        for block in self.blocks:
            nn.init.normal_(block.attn.qkv.weight, std=0.02)
            nn.init.normal_(block.attn.proj.weight, std=0.02)
            nn.init.normal_(block.ff.net[0].weight, std=0.02)
            nn.init.normal_(block.ff.net[3].weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def get_attn_mask(self, device: torch.device) -> torch.Tensor:
        if self._attn_mask is None or self._attn_mask.device != device:
            self._attn_mask = build_block_causal_mask(self.scale_factors, device)
        return self._attn_mask

    def _build_input_sequence(
        self, token_indices_list: List[torch.Tensor], class_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the input token sequence for teacher-forced training.

        The sequence is: [class_cond, level_0_start, tokens_0, level_1_start, tokens_1, ...]
        But for prediction, we shift: input at position i predicts token at position i+1.

        For block-wise prediction, the input to predict band i's tokens is:
        [class_embed, start_0, tokens_0, start_1, tokens_1, ..., start_i]
        and the target is tokens_i.

        We implement this as: for each level i, the start token of level i
        is the "query" that predicts all tokens in level i.

        Input sequence (length = total_tokens):
        For level i, positions [sum_{j<i} s_j^2 : sum_{j<=i} s_j^2]:
          - position 0 of level i: level_start_token_i (predicts first token of level i)
          - positions 1..s_i^2-1: token_embed(tokens_i[0..s_i^2-2]) (predict next tokens)

        Args:
            token_indices_list: list of (B, h_i*w_i) index tensors
            class_labels: (B,) class indices

        Returns:
            x: (B, total_tokens, D) input embeddings
            cond: (B, D) class conditioning
        """
        B = class_labels.shape[0]
        device = class_labels.device

        cond = self.class_embed(class_labels)  # (B, D)

        segments = []
        for i, (s, indices) in enumerate(zip(self.scale_factors, token_indices_list)):
            n_i = s * s
            # Start token for this level
            start = self.level_start_tokens[i].unsqueeze(0).expand(B, -1)  # (B, D)
            # Token embeddings for this level (shifted right: use start + first n_i-1 tokens)
            tok_emb = self.token_embed(indices)  # (B, n_i, D)
            tok_emb = tok_emb + self.pos_embeds[i]  # add positional embedding
            # Input: [start, tok_0, tok_1, ..., tok_{n_i-2}]
            seg = torch.cat([start.unsqueeze(1), tok_emb[:, :-1, :]], dim=1)  # (B, n_i, D)
            segments.append(seg)

        x = torch.cat(segments, dim=1)  # (B, total_tokens, D)
        return x, cond

    def forward(
        self,
        token_indices_list: List[torch.Tensor],
        class_labels: torch.Tensor,
        cfg_drop_prob: float = 0.1,
    ) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).

        Args:
            token_indices_list: list of n tensors, each (B, h_i*w_i)
            class_labels: (B,) class indices
            cfg_drop_prob: probability of dropping class label (for CFG training)

        Returns:
            logits: (B, total_tokens, vocab_size)
        """
        B = class_labels.shape[0]
        device = class_labels.device

        # CFG: randomly replace class labels with null class
        if cfg_drop_prob > 0 and self.training:
            drop_mask = torch.rand(B, device=device) < cfg_drop_prob
            null_labels = torch.full_like(class_labels, self.num_classes)
            class_labels = torch.where(drop_mask, null_labels, class_labels)

        x, cond = self._build_input_sequence(token_indices_list, class_labels)
        attn_mask = self.get_attn_mask(device)

        for block in self.blocks:
            x = block(x, cond, attn_mask)

        x = self.norm(x)
        logits = self.head(x)  # (B, total_tokens, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        class_labels: torch.Tensor,
        cfg_scale: float = 4.5,
        top_k: int = 990,
        temperature: float = 1.0,
    ) -> List[torch.Tensor]:
        """
        Autoregressive generation with classifier-free guidance.

        Generates token sequences level by level (frequency band by band).
        Within each level, generates tokens one by one using the block-causal mask.

        Args:
            class_labels: (B,) class indices
            cfg_scale: CFG guidance scale
            top_k: top-k sampling parameter
            temperature: sampling temperature

        Returns:
            indices_list: list of n tensors, each (B, h_i*w_i)
        """
        B = class_labels.shape[0]
        device = class_labels.device

        null_labels = torch.full_like(class_labels, self.num_classes)

        # We generate level by level
        # For each level, we generate all tokens in that level sequentially
        generated_indices: List[torch.Tensor] = []
        # Keep track of all generated token embeddings so far
        all_token_embs: List[torch.Tensor] = []  # list of (B, n_i, D) per level

        cond_real = self.class_embed(class_labels)  # (B, D)
        cond_null = self.class_embed(null_labels)   # (B, D)

        for level_idx, s in enumerate(self.scale_factors):
            n_i = s * s
            level_indices = torch.zeros(B, n_i, dtype=torch.long, device=device)

            for tok_idx in range(n_i):
                # Build current sequence up to this token
                # Sequence: [prev_levels..., start_i, generated_tokens_so_far_in_level_i]
                segments = []
                for prev_i, prev_embs in enumerate(all_token_embs):
                    segments.append(prev_embs)  # (B, n_prev, D)

                # Start token for current level
                start = self.level_start_tokens[level_idx].unsqueeze(0).expand(B, -1)
                start = start.unsqueeze(1)  # (B, 1, D)

                if tok_idx == 0:
                    cur_seg = start  # (B, 1, D)
                else:
                    # Already generated tokens in this level
                    prev_tok_embs = self.token_embed(level_indices[:, :tok_idx])  # (B, tok_idx, D)
                    prev_tok_embs = prev_tok_embs + self.pos_embeds[level_idx][:, :tok_idx, :]
                    cur_seg = torch.cat([start, prev_tok_embs], dim=1)  # (B, tok_idx+1, D)

                segments.append(cur_seg)
                x = torch.cat(segments, dim=1)  # (B, L_so_far, D)

                # Current sequence length
                L_cur = x.shape[1]

                # Build partial attention mask for current length
                full_mask = self.get_attn_mask(device)
                partial_mask = full_mask[:L_cur, :L_cur]

                # Forward pass for both conditional and unconditional
                def run_forward(cond_emb):
                    h = x.clone()
                    for block in self.blocks:
                        h = block(h, cond_emb, partial_mask)
                    h = self.norm(h)
                    return self.head(h[:, -1, :])  # (B, vocab_size)

                logits_cond = run_forward(cond_real)
                logits_uncond = run_forward(cond_null)

                # CFG
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)

                # Top-k sampling
                logits = logits / temperature
                if top_k > 0:
                    top_k_val = min(top_k, logits.size(-1))
                    topk_logits, _ = torch.topk(logits, top_k_val, dim=-1)
                    threshold = topk_logits[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < threshold, float("-inf"))

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
                level_indices[:, tok_idx] = next_token

            # Store generated indices for this level
            generated_indices.append(level_indices)

            # Build full token embeddings for this level (for next level's context)
            tok_embs = self.token_embed(level_indices)  # (B, n_i, D)
            tok_embs = tok_embs + self.pos_embeds[level_idx]  # (B, n_i, D)
            # Prepend start token
            start = self.level_start_tokens[level_idx].unsqueeze(0).expand(B, -1).unsqueeze(1)
            level_embs = torch.cat([start, tok_embs[:, :-1, :]], dim=1)  # (B, n_i, D)
            all_token_embs.append(level_embs)

        return generated_indices
