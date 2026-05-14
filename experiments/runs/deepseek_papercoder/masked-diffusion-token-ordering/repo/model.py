## model.py

"""
Model architectures for masked diffusion experiments.

Defines two models that share a common interface:
- MDMTransformer : bidirectional encoder with learnable absolute positional embeddings,
                    used as the denoising network in masked diffusion.
- ARMWrapper     : causal (left‑to‑right) decoder, used for autoregressive baselines.

Both models accept token IDs and an optional attention mask, and return unnormalised
logits over the real (non‑mask) vocabulary only.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# configs.py is imported only for type annotations; no risk of circular import.
from configs import ExperimentConfig


class Model(nn.Module):
    """
    Abstract base class for all models used in the project.

    Provides the common interface expected by the trainer, samplers and evaluators.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Alias for :meth:`get_logits`. Returns logits over real tokens.
        """
        return self.get_logits(x, attention_mask)

    def get_logits(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute logits over the real vocabulary.

        Args:
            x: Long tensor of shape ``(B, L)`` with token IDs (may include mask tokens).
            attention_mask: Optional boolean tensor of shape ``(B, L)`` where
                ``True`` indicates a valid (non‑padding) token and ``False``
                indicates padding. Positions with ``False`` are ignored in self‑attention.
                If ``None``, all tokens are considered valid.

        Returns:
            Logits tensor of shape ``(B, L, num_real_tokens)``.
        """
        raise NotImplementedError


class MDMTransformer(Model):
    """
    Bidirectional transformer encoder used as the MDM denoiser.

    The network is time‑embedding free; the noise level is implicitly conveyed
    through the number of masked tokens in the input sequence.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)

        # ------------------------------------------------------------------
        # Configuration shortcuts
        # ------------------------------------------------------------------
        self.mask_token_id: int = config.diffusion.mask_token_id
        self.vocab_size: int = config.model.vocab_size
        self.hidden_size: int = config.model.hidden_size
        self.max_len: int = config.model.max_seq_length

        # The vocabulary size stored in the config is assumed to already include
        # the mask token.  The number of real (predictable) tokens is therefore
        # one less.
        self.num_real_tokens: int = self.vocab_size - 1

        # ------------------------------------------------------------------
        # Embeddings
        # ------------------------------------------------------------------
        self.token_embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.pos_embed = nn.Embedding(self.max_len, self.hidden_size)

        # ------------------------------------------------------------------
        # Transformer encoder (bidirectional)
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=config.model.num_attention_heads,
            dim_feedforward=config.model.intermediate_size,
            dropout=config.model.hidden_dropout_prob,
            activation="gelu",               # matches GPT‑style activations
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.model.num_layers
        )

        # ------------------------------------------------------------------
        # Output projection (real tokens only)
        # ------------------------------------------------------------------
        self.lm_head = nn.Linear(self.hidden_size, self.num_real_tokens, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation  (simple normal, mirroring common practice)
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        std = 0.02
        nn.init.normal_(self.token_embed.weight, std=std)
        nn.init.normal_(self.pos_embed.weight, std=std)
        nn.init.normal_(self.lm_head.weight, std=std)

    # ------------------------------------------------------------------
    # Forward / logit computation
    # ------------------------------------------------------------------
    def get_logits(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute logits over real tokens for the bidirectional encoder.

        Args:
            x: Long tensor ``(B, L)`` containing token IDs (including mask tokens).
            attention_mask: Boolean tensor ``(B, L)``, ``True`` = valid, ``False`` = pad.

        Returns:
            Logits ``(B, L, num_real_tokens)``.
        """
        B, L = x.shape
        device = x.device

        # Token embeddings  (B, L, D)
        tok_emb = self.token_embed(x)

        # Absolute positional embeddings  (1, L, D) -> broadcast over batch
        pos_ids = torch.arange(L, device=device).unsqueeze(0)
        pos_emb = self.pos_embed(pos_ids)

        # Sum embeddings
        h = tok_emb + pos_emb

        # Convert the user‑friendly attention_mask into PyTorch's
        # ``src_key_padding_mask``, where ``True`` means **ignore**.
        if attention_mask is not None:
            padding_mask = ~attention_mask   # True for padding positions
        else:
            padding_mask = None

        # Bidirectional encoder forward
        h = self.encoder(h, src_key_padding_mask=padding_mask)   # (B, L, D)

        # Project to real‑token logits
        logits = self.lm_head(h)                                  # (B, L, num_real_tokens)
        return logits


class ARMWrapper(Model):
    """
    Causal (left‑to‑right) decoder used for autoregressive baselines.

    This model uses the same embedding and projection layers as ``MDMTransformer``,
    but applies a causal attention mask so that each position can only attend to
    previous positions.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)

        # ------------------------------------------------------------------
        # Configuration shortcuts
        # ------------------------------------------------------------------
        self.mask_token_id: int = config.diffusion.mask_token_id
        self.vocab_size: int = config.model.vocab_size
        self.hidden_size: int = config.model.hidden_size
        self.max_len: int = config.model.max_seq_length
        self.num_real_tokens: int = self.vocab_size - 1

        # ------------------------------------------------------------------
        # Embeddings
        # ------------------------------------------------------------------
        self.token_embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.pos_embed = nn.Embedding(self.max_len, self.hidden_size)

        # ------------------------------------------------------------------
        # Transformer layers (applied with causal mask)
        # ------------------------------------------------------------------
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=config.model.num_attention_heads,
                dim_feedforward=config.model.intermediate_size,
                dropout=config.model.hidden_dropout_prob,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(config.model.num_layers)
        ])

        # ------------------------------------------------------------------
        # Output projection
        # ------------------------------------------------------------------
        self.lm_head = nn.Linear(self.hidden_size, self.num_real_tokens, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        std = 0.02
        nn.init.normal_(self.token_embed.weight, std=std)
        nn.init.normal_(self.pos_embed.weight, std=std)
        # TransformerEncoderLayer has its own default initialisation,
        # which is acceptable; re‑initialising would be redundant.
        nn.init.normal_(self.lm_head.weight, std=std)

    # ------------------------------------------------------------------
    # Forward / logit computation
    # ------------------------------------------------------------------
    def get_logits(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute logits over real tokens for the causal decoder.

        Args:
            x: Long tensor ``(B, L)`` containing token IDs (including mask tokens).
            attention_mask: Boolean tensor ``(B, L)``, ``True`` = valid, ``False`` = pad.

        Returns:
            Logits ``(B, L, num_real_tokens)``.
        """
        B, L = x.shape
        device = x.device

        # Token & positional embeddings
        tok_emb = self.token_embed(x)
        pos_ids = torch.arange(L, device=device).unsqueeze(0)
        pos_emb = self.pos_embed(pos_ids)
        h = tok_emb + pos_emb     # (B, L, D)

        # Causal mask: each position can only attend to itself and previous ones.
        # The mask has shape (L, L) with 0 in allowed positions and -inf in disallowed.
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=device), diagonal=1
        )

        # Padding mask (True = ignore)
        if attention_mask is not None:
            padding_mask = ~attention_mask
        else:
            padding_mask = None

        # Apply each layer sequentially with the causal mask.
        for layer in self.layers:
            h = layer(h, src_mask=causal_mask, src_key_padding_mask=padding_mask)

        # Project to real‑token logits
        logits = self.lm_head(h)                                    # (B, L, num_real_tokens)
        return logits

