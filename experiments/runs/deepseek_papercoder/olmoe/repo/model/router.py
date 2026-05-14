# model/router.py
"""
Implements the Router (gating network) for the Mixture-of-Experts layer.

The Router is a simple linear projection (no bias) that maps each token's
hidden state to logits over all experts, performs a softmax, and returns
the top‑k expert indices, their associated probabilities, and the raw
logits for auxiliary loss computation.

Configurable via key parameters:
    hidden_size (2048), num_experts (64), top_k (8)
Follows the OLMoE‑1B‑7B paper exactly, including truncated normal init.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Token‑choice gating network for Mixture‑of‑Experts.

    Arguments:
        hidden_size:  Model dimension (input size).
        num_experts:  Total number of experts in the MoE layer.
        top_k:        Number of experts to activate per token.
        init_std:     Truncated normal standard deviation.
        init_trunc:   Truncation factor (truncation at ±init_trunc * init_std).
        bias:         Whether to include bias in the linear layer (paper uses False).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_experts: int = 64,
        top_k: int = 8,
        init_std: float = 0.02,
        init_trunc: int = 3,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_experts ({num_experts})"
            )

        self.num_experts = num_experts
        self.top_k = top_k

        # Linear projection from hidden_size → num_experts (no bias)
        self.weight = nn.Linear(hidden_size, num_experts, bias=bias)

        # Truncated normal initialization (Section 4.2.2)
        self._init_weights(init_std, init_trunc)

    def _init_weights(self, init_std: float, init_trunc: int) -> None:
        """Initialize weight matrix with truncated normal distribution."""
        with torch.no_grad():
            nn.init.trunc_normal_(
                self.weight.weight,
                mean=0.0,
                std=init_std,
                a=-init_trunc * init_std,
                b=init_trunc * init_std,
            )
        # No bias to initialize because bias is disabled

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
                The output of the preceding attention block (after RMSNorm).

        Returns:
            topk_indices:   LongTensor, shape (batch_size, seq_len, top_k)
                Expert indices (0 ≤ idx < num_experts) for each token.
            topk_probs:     FloatTensor, shape (batch_size, seq_len, top_k)
                Softmax probabilities corresponding to the selected experts.
            router_logits:  FloatTensor, shape (batch_size, seq_len, num_experts)
                Raw logits for all experts, used by auxiliary losses.
        """
        # 1. Compute logits for all experts
        logits = self.weight(hidden_states)           # (B, S, N)

        # 2. Apply full softmax over expert dimension
        probs = F.softmax(logits, dim=-1)             # (B, S, N)

        # 3. Select top‑k experts and their probabilities
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        return topk_indices, topk_probs, logits

