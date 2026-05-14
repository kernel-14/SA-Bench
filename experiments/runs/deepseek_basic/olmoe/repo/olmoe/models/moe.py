"""Mixture-of-Experts module for OLMoE.

Implements the dropless token choice MoE with:
- Fine-grained experts (64 experts, 8 activated)
- Load balancing loss (alpha = 0.01)
- Router z-loss (beta = 0.001)

Based on:
- Shazeer et al. (2017) "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
- Zoph et al. (2022) "ST-MoE: Designing Stable and Transferable Sparse Expert Models"
- Gale et al. (2022) "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts"
- Dai et al. (2024) "DeepSeekMoE: Towards Ultimate Expert Specialization in MoE LMs"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class OLMoEMoE(nn.Module):
    """Dropless token-choice Mixture-of-Experts layer.

    Each input token selects k experts via a learned router. The outputs of
    the selected experts are weighted by the softmax-normalized router logits
    and summed.

    Args:
        hidden_size: Model hidden dimension (d_model = 2048 for OLMoE-1B-7B)
        ffn_dim: Feedforward dimension per expert (1024 for OLMoE-1B-7B)
        num_experts: Total number of experts (64 for OLMoE-1B-7B)
        num_activated: Number of experts activated per token (8 for OLMoE-1B-7B)
        activation: Activation function in FFN (SwiGLU)
        dropout: Dropout rate
        lb_loss_weight: Load balancing loss weight α (0.01)
        rz_loss_weight: Router z-loss weight β (0.001)
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        ffn_dim: int = 1024,
        num_experts: int = 64,
        num_activated: int = 8,
        activation: nn.Module = None,
        dropout: float = 0.0,
        lb_loss_weight: float = 0.01,
        rz_loss_weight: float = 0.001,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.num_activated = num_activated
        self.lb_loss_weight = lb_loss_weight
        self.rz_loss_weight = rz_loss_weight

        # Router: linear layer mapping from hidden_size to num_experts
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # Experts: each is a SwiGLU FFN
        if activation is None:
            activation = SwiGLU()
        self.experts = nn.ModuleList([
            Expert(hidden_size, ffn_dim, activation, dropout)
            for _ in range(num_experts)
        ])

        # For tracking losses
        self._lb_loss: Optional[torch.Tensor] = None
        self._rz_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            output: Output tensor of same shape as input
            total_aux_loss: Sum of weighted auxiliary losses (α*L_LB + β*L_RZ)
        """
        bsz, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden)  # (batch*seq, hidden)

        # Router logits
        router_logits = self.router(x_flat)  # (batch*seq, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (batch*seq, num_experts)

        # Top-k selection: pick k experts with highest routing probability
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.num_activated, dim=-1
        )  # (batch*seq, k)

        # Renormalize top-k probabilities (as per paper eq 1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute router z-loss (Equation 4)
        # L_RZ(x) = 1/B * Σ_i (log Σ_j exp(x_j^(i)))^2
        logsumexp = torch.logsumexp(router_logits, dim=-1)  # (batch*seq,)
        self._rz_loss = (logsumexp ** 2).mean()

        # Compute load balancing loss (Equation 3)
        # L_LB = N_E * Σ_i f_i * P_i
        # f_i: fraction of tokens routed to expert i
        # P_i: total routing probability allocated to expert i
        with torch.no_grad():
            # Create one-hot mask for expert assignments
            mask = torch.zeros_like(router_probs)
            mask.scatter_(1, top_k_indices, 1.0)
            # f_i = fraction of tokens per expert
            f_i = mask.mean(dim=0)  # (num_experts,)

        # P_i = mean routing probability for each expert
        P_i = router_probs.mean(dim=0)  # (num_experts,)

        self._lb_loss = self.num_experts * (f_i * P_i).sum()

        # Compute expert outputs (dropless: every token gets processed)
        # For efficiency, we can batch tokens per expert
        # Here we implement a simple (less efficient) version for clarity
        output = x_flat.new_zeros(bsz * seq_len, hidden)

        for k_idx in range(self.num_activated):
            expert_ids = top_k_indices[:, k_idx]  # (batch*seq,)
            weights = top_k_probs[:, k_idx].unsqueeze(-1)  # (batch*seq, 1)

            for expert_id in range(self.num_experts):
                mask_e = (expert_ids == expert_id)
                if mask_e.any():
                    expert_out = self.experts[expert_id](x_flat[mask_e])
                    output[mask_e] += weights[mask_e] * expert_out

        output = output.reshape(bsz, seq_len, hidden)

        # Total auxiliary loss
        total_aux_loss = self.lb_loss_weight * self._lb_loss + self.rz_loss_weight * self._rz_loss

        return output, total_aux_loss

    @property
    def lb_loss(self) -> Optional[torch.Tensor]:
        """Raw load balancing loss (before scaling by weight)."""
        return self._lb_loss

    @property
    def rz_loss(self) -> Optional[torch.Tensor]:
        """Raw router z-loss (before scaling by weight)."""
        return self._rz_loss


class Expert(nn.Module):
    """A single expert FFN module using SwiGLU activation.

    Architecture: Linear_up -> SwiGLU -> Linear_down
    (Following the standard SwiGLU FFN used in OLMo, Llama, etc.)
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        activation: nn.Module,
        dropout: float = 0.0,
    ):
        super().__init__()
        # SwiGLU has 3 weight matrices: gate, up, down
        self.gate_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_size, bias=False)
        self.act = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = self.act(gate, up)
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class SwiGLU(nn.Module):
    """SwiGLU activation function as used in OLMoE.

    SwiGLU(x, W, V) = SiLU(xW) ⊙ (xV)
    where SiLU(x) = x * σ(x)
    """

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up
