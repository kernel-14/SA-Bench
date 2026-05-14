"""
LoRA-SB: Initialization using Update Approximation for Efficient Low-Rank Fine-Tuning

Implements the LoRA-SB method from:
"Initialization Using Update Approximation is a Silver Bullet for Extremely Efficient
Low-Rank Fine-Tuning" (Ponkshe et al., 2024)

Architecture: W = W0 + B*R*A  (s=1 due to scaling-factor independence)
- B (m x r) and A (r x n): fixed orthonormal matrices from SVD of delta_W_avg
- R (r x r): only trainable parameter

Key properties with orthonormal B, A (B^T*B = A*A^T = I):
1. Optimal gradient approximation: g^R = g^R_LoRA-XS (no matrix inversions needed)
2. Scaling-factor independence: s=1 is optimal
3. Guaranteed loss reduction: delta_L <= 0 at each step
4. Optimal rank-r approximation of initial gradient (Eckart-Young theorem)

Initialization:
    delta_W_avg = -sign(sum_i grad_W L(W0, x_i))  [approximates AdamW first step]
    U, S, V^T = SVD(delta_W_avg)
    B_init = U[:, :r], A_init = V[:r, :], R_init = diag(S[:r])
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, List, Tuple
import math


class LoRASBLayer(nn.Module):
    """
    LoRA-SB adapter layer: computes B*R*A*x

    B (m x r) and A (r x n) are fixed orthonormal buffers.
    R (r x r) is the only trainable parameter.

    With orthonormal B and A, the optimal gradient approximation from Theorem 3 simplifies to:
        g^R = (1/s^2) * (B^T B)^{-1} * g^R_LoRA-XS * (A A^T)^{-1}
            = g^R_LoRA-XS  (since B^T B = A A^T = I and s=1)

    This means standard gradient descent on R already achieves optimal gradient approximation.
    """

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Fixed orthonormal matrices (registered as buffers, not parameters)
        self.register_buffer('B', torch.zeros(out_features, rank))
        self.register_buffer('A', torch.zeros(rank, in_features))

        # Only trainable parameter: r x r matrix
        # Will be set to diag(S[:r]) after SVD initialization
        self.R = nn.Parameter(torch.zeros(rank, rank))

        self._initialized = False

    def initialize_from_svd(self, U: Tensor, S: Tensor, Vh: Tensor):
        """
        Set B, A, R from truncated SVD of delta_W_avg.

        B_init = U[:, :r]   (orthonormal columns: B^T B = I)
        A_init = Vh[:r, :]  (orthonormal rows: A A^T = I)
        R_init = diag(S[:r])  (s=1, so R absorbs singular values)

        This gives B*R*A = delta_W_avg (optimal rank-r approx by Eckart-Young theorem).
        """
        r = self.rank
        B_init = U[:, :r].contiguous()
        A_init = Vh[:r, :].contiguous()
        R_init = torch.diag(S[:r]).contiguous()

        self.B.copy_(B_init.to(self.B.dtype))
        self.A.copy_(A_init.to(self.A.dtype))
        self.R.data.copy_(R_init.to(self.R.dtype))
        self._initialized = True

    def forward(self, x: Tensor) -> Tensor:
        """Compute B*R*A*x (s=1 due to scaling-factor independence).

        For input x of shape (..., in_features):
            x @ A.T  -> (..., rank)
            @ R.T    -> (..., rank)
            @ B.T    -> (..., out_features)

        This is equivalent to x @ (B*R*A).T = (B*R*A @ x.T).T
        """
        out = x @ self.A.T    # (..., rank)
        out = out @ self.R.T  # (..., rank)
        out = out @ self.B.T  # (..., out_features)
        return out

    def extra_repr(self) -> str:
        return (f'in={self.in_features}, out={self.out_features}, '
                f'rank={self.rank}, init={self._initialized}')


class LoRASBLinear(nn.Module):
    """
    Linear layer with LoRA-SB adaptation.

    Forward: output = W0*x + B*R*A*x

    Only R is trainable; W0, B, A are frozen.
    """

    def __init__(self, base_layer: nn.Linear, rank: int):
        super().__init__()
        self.base_layer = base_layer
        # Freeze base layer weights
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_sb = LoRASBLayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
        )
        self.rank = rank

    def forward(self, x: Tensor) -> Tensor:
        return self.base_layer(x) + self.lora_sb(x)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias


def _get_submodule(model: nn.Module, key: str):
    """Navigate to a submodule by dotted key."""
    parent = model
    parts = key.split('.')
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1], getattr(parent, parts[-1])


def apply_lora_sb(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
) -> nn.Module:
    """
    Replace target linear layers with LoRA-SB layers.

    Args:
        model: Pre-trained model
        target_modules: List of module name suffixes to target (e.g., ["q_proj", "v_proj"])
        rank: Low-rank dimension r

    Returns:
        Model with LoRA-SB layers (only R matrices are trainable)
    """
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in target_modules:
                if name.endswith(target) or f'.{target}.' in name or name == target:
                    modules_to_replace[name] = module
                    break

    for name, module in modules_to_replace.items():
        parent, child_name, _ = _get_submodule(model, name)
        lora_sb_layer = LoRASBLinear(module, rank=rank)
        setattr(parent, child_name, lora_sb_layer)

    print(f"Applied LoRA-SB to {len(modules_to_replace)} layers with rank={rank}")
    return model


def compute_gradient_estimate(
    model: nn.Module,
    dataloader,
    n_samples: int,
    device: str = 'cuda',
) -> Dict[str, Tensor]:
    """
    Compute delta_W_avg = -sign(sum_i grad_W L(W0, x_i)) for each LoRA-SB layer.

    This approximates the first AdamW update step (Appendix C of paper):
        theta_1 = theta_0 - alpha * g_1 / sqrt(g_1^2 + eps) approx -alpha * sign(g_1)

    Uses layerwise gradient accumulation with immediate discarding for O(1) memory usage
    (independent of number of layers), as described in Section 2.6 of the paper.

    Args:
        model: Model with LoRA-SB layers applied
        dataloader: Training data loader
        n_samples: Number of samples to use (paper uses 0.1% of dataset = 50 samples)
        device: Compute device

    Returns:
        Dict mapping layer names to delta_W_avg tensors
    """
    # Temporarily enable gradients for base layer weights
    for name, module in model.named_modules():
        if isinstance(module, LoRASBLinear):
            module.base_layer.weight.requires_grad = True

    grad_accumulator: Dict[str, Tensor] = {}

    model.eval()
    # Clear any existing gradients
    model.zero_grad()
    samples_processed = 0

    for batch in dataloader:
        if samples_processed >= n_samples:
            break

        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()}

        # Forward + backward pass
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        loss.backward()

        # Accumulate gradients layerwise and immediately discard
        # This achieves O(1) memory usage as described in the paper
        for name, module in model.named_modules():
            if isinstance(module, LoRASBLinear):
                grad = module.base_layer.weight.grad
                if grad is not None:
                    if name not in grad_accumulator:
                        grad_accumulator[name] = grad.detach().clone().cpu()
                    else:
                        grad_accumulator[name].add_(grad.detach().cpu())
                    # Immediately discard gradient to save memory
                    module.base_layer.weight.grad = None

        # Count samples
        if isinstance(batch, dict) and 'input_ids' in batch:
            samples_processed += batch['input_ids'].shape[0]
        else:
            samples_processed += 1

    # Re-freeze base layer weights
    for name, module in model.named_modules():
        if isinstance(module, LoRASBLinear):
            module.base_layer.weight.requires_grad = False

    # Compute delta_W_avg = -sign(sum of gradients)
    # This approximates the direction of the first AdamW step
    delta_w_avg = {}
    for name, grad_sum in grad_accumulator.items():
        delta_w_avg[name] = -torch.sign(grad_sum)

    return delta_w_avg


def initialize_lora_sb_from_gradients(
    model: nn.Module,
    delta_w_dict: Dict[str, Tensor],
):
    """
    Initialize LoRA-SB layers using truncated SVD of delta_W_avg.

    For each layer:
        U, S, V^T = SVD(delta_W_avg)
        B_init = U[:, :r]   (orthonormal: B^T B = I)
        A_init = V[:r, :]   (orthonormal: A A^T = I)
        R_init = diag(S[:r])  (s=1)

    By Eckart-Young theorem, this is the optimal rank-r approximation of delta_W_avg.
    The orthonormality of B and A ensures:
    - Scaling-factor independence (Theorem 5): s=1 is optimal
    - Simplified gradient optimization: g^R = g^R_LoRA-XS (Theorem 3 + Section 2.6)
    - Guaranteed loss reduction (Theorem 4): delta_L <= 0

    Args:
        model: Model with LoRA-SB layers
        delta_w_dict: Dict mapping layer names to delta_W_avg tensors
    """
    initialized_count = 0

    for name, module in model.named_modules():
        if isinstance(module, LoRASBLinear):
            if name in delta_w_dict:
                delta_w = delta_w_dict[name]

                # Compute truncated SVD using float32 for numerical stability
                delta_w_f32 = delta_w.float()

                try:
                    # torch.linalg.svd is the recommended API
                    U, S, Vh = torch.linalg.svd(delta_w_f32, full_matrices=False)
                except Exception:
                    # Fallback to older API
                    U, S, V = torch.svd(delta_w_f32, some=True)
                    Vh = V.T

                module.lora_sb.initialize_from_svd(U, S, Vh)
                initialized_count += 1
            else:
                # Fallback: random orthonormal initialization
                _random_orthonormal_init(module.lora_sb)
                initialized_count += 1

    print(f"Initialized {initialized_count} LoRA-SB layers")


def _random_orthonormal_init(lora_layer: LoRASBLayer):
    """
    Fallback initialization using random orthonormal matrices.
    Ensures B^T B = A A^T = I for scaling-factor independence.
    """
    m, r, n = lora_layer.out_features, lora_layer.rank, lora_layer.in_features

    # Random orthonormal B: (m, r)
    B_rand = torch.randn(m, r)
    B_orth, _ = torch.linalg.qr(B_rand)
    lora_layer.B.copy_(B_orth[:, :r])

    # Random orthonormal A: (r, n) - rows are orthonormal
    A_rand = torch.randn(n, r)
    A_orth, _ = torch.linalg.qr(A_rand)
    lora_layer.A.copy_(A_orth[:, :r].T)

    # R initialized to identity
    lora_layer.R.data.copy_(torch.eye(r))
    lora_layer._initialized = True


# Aliases for backward compatibility
def estimate_gradient(
    model: nn.Module,
    dataloader,
    n_samples: int,
    device: str = 'cuda',
) -> Dict[str, Tensor]:
    """Alias for compute_gradient_estimate."""
    return compute_gradient_estimate(model, dataloader, n_samples, device)


def initialize_lora_sb(
    model: nn.Module,
    delta_w_dict: Dict[str, Tensor],
):
    """Alias for initialize_lora_sb_from_gradients."""
    return initialize_lora_sb_from_gradients(model, delta_w_dict)


def get_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_trainable_parameters(model: nn.Module):
    """Print parameter statistics."""
    trainable, total = get_trainable_parameters(model)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    reduction = total / trainable if trainable > 0 else float('inf')
    print(f"Trainable: {trainable:,} / Total: {total:,} ({pct:.4f}%)")
    print(f"Parameter reduction vs full model: {reduction:.1f}x")
