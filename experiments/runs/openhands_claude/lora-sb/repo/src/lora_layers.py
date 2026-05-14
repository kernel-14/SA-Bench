from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Standard LoRA: W = W_0 + s * B * A, both B and A trainable."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, base_weight)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return result + self.scaling * lora_out


class rsLoRALayer(nn.Module):
    """rsLoRA: LoRA with rank-stabilized scaling s = alpha / sqrt(r)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / math.sqrt(rank)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, base_weight)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return result + self.scaling * lora_out


class PiSSALayer(nn.Module):
    """PiSSA: LoRA initialized with principal singular vectors of W_0.

    The residual W_res = W_0 - s*B*A is frozen; B and A are trainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        weight: torch.Tensor,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        # Take top-r singular components
        U_r = U[:, :rank]       # (out_features, rank)
        S_r = S[:rank]          # (rank,)
        Vh_r = Vh[:rank, :]     # (rank, in_features)

        # B = U_r * sqrt(S_r), A = sqrt(S_r) * Vh_r
        sqrt_S = torch.sqrt(S_r)
        B_init = U_r * sqrt_S.unsqueeze(0)   # (out_features, rank)
        A_init = sqrt_S.unsqueeze(1) * Vh_r  # (rank, in_features)

        self.lora_A = nn.Parameter(A_init.to(weight.dtype))
        self.lora_B = nn.Parameter(B_init.to(weight.dtype))

        # Residual weight: W_res = W_0 - s*B*A
        W_res = weight - self.scaling * (B_init @ A_init).to(weight.dtype)
        self.register_buffer("weight_res", W_res)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        # Use residual weight instead of base_weight
        result = F.linear(x, self.weight_res)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return result + self.scaling * lora_out


class DoRALayer(nn.Module):
    """DoRA: Weight-decomposed LoRA.

    W = m * (W_0 + s*B*A) / ||W_0 + s*B*A||_col
    where m is a learnable magnitude vector.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        weight: torch.Tensor,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Magnitude vector initialized from column norms of W_0
        col_norms = weight.norm(dim=1, keepdim=True).squeeze(1)  # (out_features,)
        self.magnitude = nn.Parameter(col_norms.to(weight.dtype))

        self.register_buffer("weight_0", weight.clone())
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        # Adapted weight: W_0 + s*B*A
        adapted = self.weight_0 + self.scaling * (self.lora_B @ self.lora_A)
        # Column-wise normalization
        col_norms = adapted.norm(dim=1, keepdim=True)
        adapted_normalized = adapted / col_norms
        # Scale by magnitude
        W_dora = self.magnitude.unsqueeze(1) * adapted_normalized
        return F.linear(self.dropout(x), W_dora)


class LoRAProLayer(nn.Module):
    """LoRA-Pro: LoRA with optimal gradient approximation.

    At each step, the gradient for B and A is corrected to better approximate
    the full FT gradient. In practice, this is implemented by modifying the
    optimizer step using the closed-form correction.

    For simplicity, we implement the equivalent: the forward pass is identical
    to LoRA, but we register a backward hook that applies the correction.
    The correction: g_A_corrected = (B^T B)^{-1} B^T g, g_B_corrected = g A^T (A A^T)^{-1}
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Register gradient hooks for optimal gradient approximation
        self.lora_A.register_hook(self._correct_grad_A)
        self.lora_B.register_hook(self._correct_grad_B)

    def _correct_grad_A(self, grad: torch.Tensor) -> torch.Tensor:
        # g_A_corrected = (B^T B)^{-1} B^T g_W A^T ... simplified via chain rule
        # The raw grad w.r.t. A is: g_A = s * B^T * g_W
        # Correction: multiply by (B^T B)^{-1} on the left
        BtB = self.lora_B.detach().T @ self.lora_B.detach()  # (r, r)
        try:
            BtB_inv = torch.linalg.inv(BtB + 1e-6 * torch.eye(self.rank, device=BtB.device, dtype=BtB.dtype))
        except Exception:
            return grad
        return BtB_inv @ grad

    def _correct_grad_B(self, grad: torch.Tensor) -> torch.Tensor:
        # g_B_corrected = g_W A^T (A A^T)^{-1} ... simplified
        AAt = self.lora_A.detach() @ self.lora_A.detach().T  # (r, r)
        try:
            AAt_inv = torch.linalg.inv(AAt + 1e-6 * torch.eye(self.rank, device=AAt.device, dtype=AAt.dtype))
        except Exception:
            return grad
        return grad @ AAt_inv

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, base_weight)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return result + self.scaling * lora_out


class LoRAXSLayer(nn.Module):
    """LoRA-XS: W = W_0 + s * B * R * A, B and A frozen, R trainable.

    B and A are initialized using PiSSA-style (SVD of W_0).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        weight: torch.Tensor,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        B_init, A_init = self._init_from_weight(weight, rank)

        self.register_buffer("lora_B", B_init)  # frozen (out_features, rank)
        self.register_buffer("lora_A", A_init)  # frozen (rank, in_features)
        self.lora_R = nn.Parameter(torch.eye(rank, dtype=weight.dtype))  # trainable (rank, rank)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @staticmethod
    def _init_from_weight(weight: torch.Tensor, rank: int):
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        U_r = U[:, :rank].to(weight.dtype)    # (out_features, rank)
        Vh_r = Vh[:rank, :].to(weight.dtype)  # (rank, in_features)
        return U_r, Vh_r

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, base_weight)
        # B * R * A
        BRA = self.lora_B @ self.lora_R @ self.lora_A  # (out_features, in_features)
        lora_out = F.linear(self.dropout(x), BRA)
        return result + self.scaling * lora_out


class LoRASBLayer(nn.Module):
    """LoRA-SB: W = W_0 + B * R * A, B and A frozen (orthonormal), R trainable.

    Key properties:
    - B and A are initialized from truncated SVD of the first-step gradient approximation
    - B^T B = I, A A^T = I (orthonormal bases)
    - s = 1 (scaling factor independence)
    - Only R is trainable
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        B_init: torch.Tensor,
        R_init: torch.Tensor,
        A_init: torch.Tensor,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        # s = 1 by design (scaling factor independence)
        self.scaling = 1.0

        self.register_buffer("lora_B", B_init)  # frozen (out_features, rank)
        self.register_buffer("lora_A", A_init)  # frozen (rank, in_features)
        self.lora_R = nn.Parameter(R_init.clone())  # trainable (rank, rank)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, base_weight)
        BRA = self.lora_B @ self.lora_R @ self.lora_A  # (out_features, in_features)
        lora_out = F.linear(self.dropout(x), BRA)
        return result + lora_out  # scaling = 1


class LinearWithLoRA(nn.Module):
    """Wraps a frozen nn.Linear with a LoRA adapter."""

    def __init__(self, linear: nn.Linear, lora_layer: nn.Module) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.lora = lora_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lora(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class LinearWithLoRAXS(nn.Module):
    """Wraps a frozen nn.Linear with a LoRA-XS adapter."""

    def __init__(self, linear: nn.Linear, lora_layer: LoRAXSLayer) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.lora = lora_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lora(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class LinearWithLoRASB(nn.Module):
    """Wraps a frozen nn.Linear with a LoRA-SB adapter."""

    def __init__(self, linear: nn.Linear, lora_layer: LoRASBLayer) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.lora = lora_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lora(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
