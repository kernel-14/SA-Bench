import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ConvExpert, FourierHead, RouterGating


class FourierLayer(nn.Module):
    """Multi-head Fourier integral operator layer.

    Splits the channel dimension into h heads, applies a frequency-domain
    two-layer MLP per head, then concatenates results. A residual skip
    connection (1×1 conv) is added to the output.

    Output: z_0^l = Concat(z_{01}^l, ..., z_{0h}^l) + W_skip * z^l
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_modes_h: int = 16,
        num_modes_w: int = 16,
    ) -> None:
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.heads = nn.ModuleList(
            [FourierHead(self.head_dim, num_modes_h, num_modes_w) for _ in range(num_heads)]
        )

        # Residual skip connection
        self.skip = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(num_heads, channels), num_channels=channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, C, H, W)
        Returns:
            z_0: (B, C, H, W)
        """
        B, C, H, W = z.shape
        head_outputs = []
        for i, head in enumerate(self.heads):
            z_i = z[:, i * self.head_dim : (i + 1) * self.head_dim, :, :]
            head_outputs.append(head(z_i))

        z_concat = torch.cat(head_outputs, dim=1)  # (B, C, H, W)
        out = z_concat + self.skip(z)
        out = self.norm(out)
        return out


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with shared and routed experts.

    Architecture (from paper Section 4):
    - N_s shared experts: always activated, output averaged
    - N_r routed experts: top-K selected per sample via router-gating network
    - Final output: (1/N_s) * Σ shared(z) + Σ_k w_k * routed_k(z)

    Load balancing loss is computed and stored in self.balance_loss.
    """

    def __init__(
        self,
        channels: int,
        num_routed_experts: int = 16,
        num_shared_experts: int = 2,
        top_k: int = 4,
        load_balance_weight: float = 0.1,
        expert_hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        # Shared experts (always activated)
        self.shared_experts = nn.ModuleList(
            [ConvExpert(channels, expert_hidden_channels) for _ in range(num_shared_experts)]
        )

        # Routed experts (top-K selected per sample)
        self.routed_experts = nn.ModuleList(
            [ConvExpert(channels, expert_hidden_channels) for _ in range(num_routed_experts)]
        )

        # Router-gating network
        self.router = RouterGating(channels, num_routed_experts)

        # Output normalization
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)

        # Stored for loss computation
        self.balance_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, C, H, W) — output from Fourier layer
        Returns:
            z_out: (B, C, H, W)
        """
        B, C, H, W = z.shape

        # --- Shared expert outputs ---
        shared_out = torch.zeros(B, C, H, W, device=z.device, dtype=z.dtype)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(z)
        shared_out = shared_out / self.num_shared_experts

        # --- Router: compute gating weights ---
        logits = self.router(z)                          # (B, N_r)
        weights = F.softmax(logits, dim=-1)              # (B, N_r)

        # Top-K selection
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)  # (B, K)
        # Re-normalize top-K weights so they sum to 1
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # --- Routed expert outputs ---
        # Only compute experts selected by at least one sample in the batch
        selected_expert_ids = topk_indices.unique().tolist()
        expert_outputs: Dict[int, torch.Tensor] = {
            eid: self.routed_experts[eid](z) for eid in selected_expert_ids
        }

        # Stack into (num_selected, B, C, H, W) for vectorized gather
        selected_ids_list = list(expert_outputs.keys())
        stacked = torch.stack(
            [expert_outputs[eid] for eid in selected_ids_list], dim=0
        )  # (S, B, C, H, W)

        # Map global expert indices to local indices in stacked tensor
        id_to_local = {eid: i for i, eid in enumerate(selected_ids_list)}

        routed_out = torch.zeros(B, C, H, W, device=z.device, dtype=z.dtype)
        for k in range(self.top_k):
            expert_idx_batch = topk_indices[:, k]   # (B,)
            w_k = topk_weights[:, k].view(B, 1, 1, 1)  # (B, 1, 1, 1)
            # Gather the k-th selected expert output for each sample
            local_ids = torch.tensor(
                [id_to_local[eid.item()] for eid in expert_idx_batch],
                device=z.device,
            )  # (B,)
            # stacked[local_ids[b], b] → (B, C, H, W)
            expert_out_k = stacked[local_ids, torch.arange(B, device=z.device)]
            routed_out = routed_out + w_k * expert_out_k

        # --- Load balancing loss ---
        # Importance_i = Σ_b w_{i,b}
        importance = weights.sum(dim=0)  # (N_r,)
        cv_sq = self._coefficient_of_variation_squared(importance)
        self.balance_loss = self.load_balance_weight * cv_sq

        # --- Final output ---
        out = shared_out + routed_out
        out = self.norm(out)
        return out

    @staticmethod
    def _coefficient_of_variation_squared(x: torch.Tensor) -> torch.Tensor:
        """CV^2 = (std / mean)^2 of a 1-D tensor."""
        mean = x.mean()
        std = x.std()
        return (std / (mean + 1e-8)) ** 2


class MoEBlock(nn.Module):
    """One transformer block: FourierLayer → MoELayer.

    Corresponds to a single block in the MoE-POT architecture (Figure 3).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        num_routed_experts: int = 16,
        num_shared_experts: int = 2,
        top_k: int = 4,
        load_balance_weight: float = 0.1,
        num_modes_h: int = 16,
        num_modes_w: int = 16,
        expert_hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.fourier_layer = FourierLayer(channels, num_heads, num_modes_h, num_modes_w)
        self.moe_layer = MoELayer(
            channels,
            num_routed_experts,
            num_shared_experts,
            top_k,
            load_balance_weight,
            expert_hidden_channels,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z0 = self.fourier_layer(z)
        z_out = self.moe_layer(z0)
        return z_out

    @property
    def balance_loss(self) -> torch.Tensor:
        return self.moe_layer.balance_loss


class OutputProjection(nn.Module):
    """Projects patch-level features back to the original spatial resolution.

    Uses a transposed convolution (or pixel-shuffle equivalent) to upsample
    from (B, embed_dim, H/P, W/P) to (B, out_channels, H, W).
    """

    def __init__(self, embed_dim: int, out_channels: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.ConvTranspose2d(
            embed_dim, out_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)
