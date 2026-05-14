"""
Autoregressive Model (ARM) — model wrapper and training loss.

Supports two training modes:
  1. Left-to-right (standard ARM): predict x^i given x^0, ..., x^{i-1}
  2. Order-aware (ARM with ordering): predict x^{π(i)} given x^{π(0)}, ..., x^{π(i-1)}
     where π is the ground-truth generation order for each sequence.

The π-learner (Section 3.2) is a special case of order-aware training where π is
a fixed permutation applied to all sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from model import Transformer, build_arm_transformer
from config import MODEL_CONFIGS


# ---------------------------------------------------------------------------
# ARM training loss
# ---------------------------------------------------------------------------

def arm_loss(
    model: Transformer,
    x0: torch.Tensor,
    ordering: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the ARM training loss.

    Standard (left-to-right):
      L = -Σ_i log p_θ(x_0^i | x_0^0, ..., x_0^{i-1})

    Order-aware (with permutation π per sequence):
      L = -Σ_i log p_θ(x_0^{π(i)} | x_0^{π(0)}, ..., x_0^{π(i-1)})

    For order-aware training, we permute the input sequence so that the causal
    transformer sees tokens in the order specified by π.

    Args:
        model:    causal transformer
        x0:       (B, L) clean token sequences
        ordering: (B, L) permutation indices π for each sequence, or None for left-to-right

    Returns:
        scalar cross-entropy loss
    """
    B, L = x0.shape

    if ordering is not None:
        # Permute input: x_permuted[b, i] = x0[b, ordering[b, i]]
        x_input = torch.gather(x0, 1, ordering)
    else:
        x_input = x0

    # Shift: input is x[:-1], target is x[1:]
    logits = model(x_input[:, :-1])  # (B, L-1, vocab_size)
    targets = x_input[:, 1:]         # (B, L-1)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="mean",
    )
    return loss


def pi_learner_loss(
    model: Transformer,
    x0: torch.Tensor,
    pi: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the π-learner loss (Section 3.2, Eq. 3).

    The π-learner predicts x_0^{π(i)} given x_0^{π(0)}, ..., x_0^{π(i-1)}.
    This is implemented by permuting the sequence and running standard ARM loss.

    log p_θ(x_0) = Σ_i log p_θ(x_0^{π(i)} | x_0[π{i,...,L-1}])

    Args:
        model: causal transformer with learnable positional embeddings
        x0:    (B, L) clean token sequences
        pi:    (L,) permutation of {0, ..., L-1}

    Returns:
        scalar cross-entropy loss
    """
    B, L = x0.shape
    pi_expanded = pi.unsqueeze(0).expand(B, -1)  # (B, L)
    return arm_loss(model, x0, ordering=pi_expanded)


def compute_pi_learner_likelihood(
    model: Transformer,
    x0: torch.Tensor,
    pi: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the π-learner log-likelihood for evaluation (Eq. 3 in paper).

    Returns:
        log_likelihood: (B,) per-sequence log-likelihood
    """
    B, L = x0.shape
    pi_expanded = pi.unsqueeze(0).expand(B, -1)
    x_permuted = torch.gather(x0, 1, pi_expanded)

    with torch.no_grad():
        logits = model(x_permuted[:, :-1])  # (B, L-1, vocab_size)
        targets = x_permuted[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=1)


# ---------------------------------------------------------------------------
# ARM model wrapper
# ---------------------------------------------------------------------------

class ARM(nn.Module):
    """
    Autoregressive Model wrapper.

    Supports left-to-right and order-aware training.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        model_config: dict,
        use_rope: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.transformer = build_arm_transformer(
            vocab_size=vocab_size,
            seq_len=seq_len,
            model_config=model_config,
            use_rope=use_rope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) token sequences

        Returns:
            logits: (B, T, vocab_size)
        """
        return self.transformer(x)

    def compute_loss(
        self,
        x0: torch.Tensor,
        ordering: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ARM training loss."""
        return arm_loss(self.transformer, x0, ordering)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        ordering: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            prompt:         (B, T_prompt) prompt tokens
            max_new_tokens: number of tokens to generate
            temperature:    sampling temperature
            ordering:       if provided, generate in this order and re-sort at the end

        Returns:
            generated: (B, T_prompt + max_new_tokens) token sequences
        """
        B = prompt.shape[0]
        generated = prompt.clone()

        for _ in range(max_new_tokens):
            logits = self.transformer(generated)[:, -1, :]  # (B, vocab_size)
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self) -> int:
        return self.transformer.count_parameters()


# ---------------------------------------------------------------------------
# π-learner wrapper (Section 3.2)
# ---------------------------------------------------------------------------

class PiLearner(nn.Module):
    """
    π-learner: an ARM trained with a fixed permutation π.

    Used to measure the hardness of different token orderings (Section 3.2).
    Uses learnable positional embeddings instead of RoPE to avoid inductive bias
    toward left-to-right ordering.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        model_config: dict,
        pi: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        # Use learnable positional embeddings (not RoPE) as specified in Section C.1
        self.transformer = build_arm_transformer(
            vocab_size=vocab_size,
            seq_len=seq_len,
            model_config=model_config,
            use_rope=False,  # learnable pos embeddings
        )

        if pi is not None:
            self.register_buffer("pi", pi)
        else:
            self.register_buffer("pi", torch.arange(seq_len))

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        return pi_learner_loss(self.transformer, x0, self.pi)

    def compute_likelihood(self, x0: torch.Tensor) -> torch.Tensor:
        return compute_pi_learner_likelihood(self.transformer, x0, self.pi)

    def count_parameters(self) -> int:
        return self.transformer.count_parameters()
