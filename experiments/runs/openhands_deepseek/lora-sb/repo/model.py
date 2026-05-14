"""LoRA-SB and baseline model implementations.

LoRA-SB (LoRA Silver Bullet):
  W = W0 + s * B @ R @ A
  - B (m x r) and A (r x n) are FIXED after initialization
  - R (r x r) is the only trainable matrix
  - Initialized via truncated SVD of the first full-FT update
  - Uses optimal gradient approximation: g^R = 1/s^2 * g_{LoRA-XS}^R
    (simplifies to g^R = g_{LoRA-XS}^R when s=1 and B, A orthonormal)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple


class LoRALinear(nn.Module):
    """Standard LoRA: W = W0 + s * B @ A, both B and A trainable."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = False,
        init_method: str = "default",
        base_layer: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        if base_layer is None:
            self.base = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.base = base_layer

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if init_method == "default":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        elif init_method == "pissa":
            self._pissa_init()
        elif init_method == "lora_ga":
            pass

    def _pissa_init(self):
        weight = self.base.weight.data
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        r = min(self.rank, len(S))
        Ur = U[:, :r]
        Sr = S[:r]
        Vhr = Vh[:r, :]
        self.lora_B.data = (Ur * Sr.unsqueeze(0)).to(weight.dtype)
        self.lora_A.data = Vhr.to(weight.dtype)
        self.base.weight.data = weight - (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out


class rsLoRALinear(LoRALinear):
    """Rank-stabilized LoRA: scaling = alpha / sqrt(rank)."""

    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0, bias=False, base_layer=None):
        super().__init__(in_features, out_features, rank, alpha, dropout, bias, base_layer)
        self.scaling = alpha / math.sqrt(rank)


class DoRALinear(nn.Module):
    """DoRA: Weight-Decomposed Low-Rank Adaptation.

    W = m * (W0 + B @ A) / ||W0 + B @ A||_c
    Uses magnitude vector m with separate directional and magnitude updates.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = False,
        base_layer: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        if base_layer is None:
            self.base = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.base = base_layer

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.m = nn.Parameter(torch.ones(1, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        combined = base_out + lora_update
        norm = torch.linalg.norm(combined, dim=-1, keepdim=True)
        combined_normed = combined / (norm + 1e-8)
        return self.m * combined_normed


class LoRAXSLinear(nn.Module):
    """LoRA-XS: W = W0 + s * B @ R @ A.
    B and A are fixed after initialization, R is trainable.
    Initialized from SVD of pre-trained weight (PiSSA-style) by default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = False,
        base_layer: Optional[nn.Linear] = None,
        init_B: Optional[torch.Tensor] = None,
        init_A: Optional[torch.Tensor] = None,
        init_R: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        if base_layer is None:
            self.base = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.base = base_layer

        if init_B is not None and init_A is not None and init_R is not None:
            self.register_buffer("lora_B", init_B.clone().detach())
            self.register_buffer("lora_A", init_A.clone().detach())
            self.lora_R = nn.Parameter(init_R.clone().detach())
        else:
            self._default_init()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _default_init(self):
        weight = self.base.weight.data
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        r = min(self.rank, len(S))
        Ur = U[:, :r]
        Sr = S[:r]
        Vhr = Vh[:r, :]
        self.register_buffer("lora_B", (Ur * Sr.unsqueeze(0)).to(weight.dtype))
        self.register_buffer("lora_A", Vhr.to(weight.dtype))
        self.lora_R = nn.Parameter(torch.eye(r, dtype=weight.dtype) / self.scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.scaling * (self.dropout(x) @ self.lora_A.T @ self.lora_R.T @ self.lora_B.T)
        return base_out + lora_out


class LoRASBLinear(nn.Module):
    """LoRA-SB: W = W0 + B @ R @ A (scaling s=1).

    B and A are fixed orthonormal matrices initialized from truncated SVD
    of the first full-FT update approximation.
    R is the only trainable matrix (r x r).
    Uses optimal gradient approximation: g^R = g_{LoRA-XS}^R (when s=1).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        base_layer: Optional[nn.Linear] = None,
        init_B: Optional[torch.Tensor] = None,
        init_A: Optional[torch.Tensor] = None,
        init_R: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = 1.0

        if base_layer is None:
            self.base = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.base = base_layer

        if init_B is not None and init_A is not None and init_R is not None:
            self.register_buffer("lora_B", init_B.clone().detach())
            self.register_buffer("lora_A", init_A.clone().detach())
            self.lora_R = nn.Parameter(init_R.clone().detach())
        else:
            self._default_init()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _default_init(self):
        weight = self.base.weight.data
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        r = min(self.rank, len(S))
        self.register_buffer("lora_B", U[:, :r].to(weight.dtype))
        self.register_buffer("lora_A", Vh[:r, :].to(weight.dtype))
        self.lora_R = nn.Parameter(torch.diag(S[:r]).to(weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_R.T @ self.lora_B.T
        return base_out + lora_out

    def get_equivalent_gradient(self, grad_R: torch.Tensor) -> torch.Tensor:
        """Compute the equivalent full-FT gradient: g_tilde = B @ g^R @ A."""
        return self.lora_B @ grad_R @ self.lora_A

    def compute_optimal_grad_R(self, grad_R_raw: torch.Tensor) -> torch.Tensor:
        """Compute optimal gradient for R using Theorem 3.
        g^R = 1/s^2 * (B^T B)^{-1} * g_{LoRA-XS}^R * (A A^T)^{-1}
        Since s=1 and B, A are orthonormal, this simplifies to:
        g^R = g_{LoRA-XS}^R
        """
        return grad_R_raw


class LoRAProLinear(nn.Module):
    """LoRA-Pro: LoRA with optimal gradient approximation.

    At each step, the gradient of B and A is adjusted to better approximate
    the full-FT gradient in the low-rank subspace.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = False,
        base_layer: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        if base_layer is None:
            self.base = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.base = base_layer

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def compute_optimal_gradients(self):
        """Compute optimal gradients for A and B using LoRA-Pro's closed form."""
        A = self.lora_A.data
        B = self.lora_B.data
        A_grad = self.lora_A.grad
        B_grad = self.lora_B.grad
        if A_grad is None or B_grad is None:
            return
        ATA = A @ A.T
        BTB = B.T @ B
        eye_r = torch.eye(self.rank, device=A.device, dtype=A.dtype)
        reg = 1e-4
        ATA_inv = torch.linalg.inv(ATA + reg * eye_r)
        BTB_inv = torch.linalg.inv(BTB + reg * eye_r)
        self.lora_A.grad = ATA_inv @ A_grad
        self.lora_B.grad = B_grad @ BTB_inv


LORA_MODULE_MAP = {
    "lora": LoRALinear,
    "rslora": rsLoRALinear,
    "dora": DoRALinear,
    "lora_xs": LoRAXSLinear,
    "lora_sb": LoRASBLinear,
    "lora_pro": LoRAProLinear,
    "pissa": LoRALinear,
}


def get_lora_module(method: str, init_method: str = "default") -> type:
    if method == "lora":
        return LoRALinear
    elif method == "rslora":
        return rsLoRALinear
    elif method == "dora":
        return DoRALinear
    elif method == "lora_xs":
        return LoRAXSLinear
    elif method == "lora_sb":
        return LoRASBLinear
    elif method == "lora_pro":
        return LoRAProLinear
    elif method == "pissa":
        return LoRALinear
    else:
        raise ValueError(f"Unknown LoRA method: {method}")


def is_lora_sb_module(module: nn.Module) -> bool:
    return isinstance(module, LoRASBLinear)


def is_lora_pro_module(module: nn.Module) -> bool:
    return isinstance(module, LoRAProLinear)


def replace_with_lora(
    model: nn.Module,
    method: str = "lora_sb",
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    init_method: str = "default",
    lora_sb_inits: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
) -> nn.Module:
    """Recursively replace nn.Linear modules with LoRA variants.

    Args:
        model: The base model.
        method: One of 'lora', 'rslora', 'dora', 'lora_xs', 'lora_sb', 'lora_pro', 'pissa'.
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.
        dropout: Dropout rate.
        target_modules: List of module name substrings to target (e.g. ['query', 'value']).
        init_method: Initialization method for LoRA ('default' or 'pissa').
        lora_sb_inits: Dict mapping module names to (B, A, R) init tensors for LoRA-SB.

    Returns:
        The modified model.
    """
    module_cls = get_lora_module(method)

    def _replace(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                should_replace = False
                if target_modules is None:
                    should_replace = True
                else:
                    for target in target_modules:
                        if target in name.lower():
                            should_replace = True
                            break

                if should_replace:
                    kwargs = dict(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        bias=child.bias is not None,
                        base_layer=child,
                    )
                    if method == "lora_sb" and lora_sb_inits and full_name in lora_sb_inits:
                        B, A, R = lora_sb_inits[full_name]
                        kwargs["init_B"] = B
                        kwargs["init_A"] = A
                        kwargs["init_R"] = R
                    if method in ("lora", "pissa"):
                        kwargs["init_method"] = init_method
                    new_module = module_cls(**kwargs)
                    setattr(module, name, new_module)
                else:
                    _replace(child, full_name)
            else:
                _replace(child, full_name)

    _replace(model)
    return model


def get_lora_parameters(model: nn.Module, method: str = "lora_sb") -> List[nn.Parameter]:
    """Get trainable LoRA parameters from the model."""
    params = []
    for module in model.modules():
        if method == "lora_sb" and isinstance(module, LoRASBLinear):
            params.append(module.lora_R)
        elif method == "lora_xs" and isinstance(module, LoRAXSLinear):
            params.append(module.lora_R)
        elif method in ("lora", "rslora", "pissa") and isinstance(module, (LoRALinear, rsLoRALinear)):
            params.extend([module.lora_A, module.lora_B])
        elif method == "dora" and isinstance(module, DoRALinear):
            params.extend([module.lora_A, module.lora_B, module.m])
        elif method == "lora_pro" and isinstance(module, LoRAProLinear):
            params.extend([module.lora_A, module.lora_B])
    return params


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
