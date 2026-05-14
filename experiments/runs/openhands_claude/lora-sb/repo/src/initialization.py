"""LoRA-SB initialization: compute first-step gradient approximation and truncated SVD.

Algorithm (from paper Section 2.4 and Appendix C):
1. Accumulate gradients over n samples (0.1% of dataset)
2. ΔW_avg = -η * sign(Σ ∇_W L(W_0, x_i))  [AdamW first-step approximation]
3. Truncated SVD: U, S, V^T = SVD(ΔW_avg)
4. B_init = U[:, :r]        (out_features, rank) — orthonormal columns
5. A_init = V[:r, :]        (rank, in_features)  — orthonormal rows
6. R_init = S[:r, :r]       (rank, rank) diagonal — with s=1
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _get_target_modules(
    model: nn.Module,
    target_module_names: List[str],
) -> Dict[str, nn.Linear]:
    """Return {full_name: module} for all Linear layers matching target names."""
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            short_name = name.split(".")[-1]
            if short_name in target_module_names:
                result[name] = module
    return result


def compute_gradient_approximation(
    model: nn.Module,
    dataloader: DataLoader,
    target_module_names: List[str],
    n_samples: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute ΔW_avg = -sign(Σ ∇_W L(W_0, x_i)) for each target weight matrix.

    Uses layerwise gradient accumulation with immediate discard to keep memory O(1)
    per layer (as described in paper Section 2.6 and reference [29, 45]).

    Args:
        model: Pre-trained model (parameters frozen).
        dataloader: DataLoader yielding batches with 'input_ids', 'attention_mask', etc.
        target_module_names: Names of Linear layers to compute gradients for.
        n_samples: Number of samples to accumulate gradients over.
        device: Compute device.

    Returns:
        Dict mapping full module name to ΔW_avg tensor of shape (out_features, in_features).
    """
    model.eval()
    target_modules = _get_target_modules(model, target_module_names)

    # Enable gradients only for target weight matrices
    for param in model.parameters():
        param.requires_grad_(False)

    for name, module in target_modules.items():
        module.weight.requires_grad_(True)

    model.to(device)

    # Initialize grad sums on the correct device (after model.to(device))
    grad_sums: Dict[str, torch.Tensor] = {}
    for name, module in target_modules.items():
        grad_sums[name] = torch.zeros_like(module.weight, dtype=torch.float32)

    samples_seen = 0

    with tqdm(total=n_samples, desc="Computing gradient approximation") as pbar:
        for batch in dataloader:
            if samples_seen >= n_samples:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch_size = batch["input_ids"].shape[0]

            # Zero gradients
            for name, module in target_modules.items():
                if module.weight.grad is not None:
                    module.weight.grad.zero_()

            outputs = model(**{k: v for k, v in batch.items() if k in (
                "input_ids", "attention_mask", "labels", "token_type_ids"
            )})
            loss = outputs.loss
            loss.backward()

            # Accumulate gradients
            for name, module in target_modules.items():
                if module.weight.grad is not None:
                    grad_sums[name] += module.weight.grad.float()

            samples_seen += batch_size
            pbar.update(batch_size)

    # Compute ΔW_avg = -sign(Σ grad)
    delta_w_avg: Dict[str, torch.Tensor] = {}
    for name in grad_sums:
        delta_w_avg[name] = -torch.sign(grad_sums[name])

    # Restore model state: disable gradients for all
    for param in model.parameters():
        param.requires_grad_(False)
    for name, module in target_modules.items():
        if module.weight.grad is not None:
            module.weight.grad = None

    return delta_w_avg


def truncated_svd_init(
    delta_w: torch.Tensor,
    rank: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute truncated SVD of ΔW and return (B_init, R_init, A_init).

    Uses torch.svd_lowrank for efficiency (as mentioned in paper Appendix F).

    B_init = U[:, :r]           shape: (out_features, rank)  — orthonormal columns
    A_init = Vh[:r, :]          shape: (rank, in_features)   — orthonormal rows
    R_init = diag(S[:r])        shape: (rank, rank)           — diagonal singular values

    With s=1, B_init @ R_init @ A_init ≈ ΔW_avg (optimal rank-r approximation).

    Args:
        delta_w: ΔW_avg tensor of shape (out_features, in_features).
        rank: Target rank r.
        dtype: Output dtype.

    Returns:
        (B_init, R_init, A_init) tensors.
    """
    # Use float32 for numerical stability
    dw_float = delta_w.float()

    # torch.svd_lowrank is efficient for large matrices
    # niter=4 provides good accuracy (default)
    U, S, Vh = torch.linalg.svd(dw_float, full_matrices=False)

    # Take top-r components
    B_init = U[:, :rank].to(dtype)    # (out_features, rank)
    S_r = S[:rank]                     # (rank,)
    A_init = Vh[:rank, :].to(dtype)   # (rank, in_features)

    # R_init = diag(S_r) — with s=1, this gives B @ R @ A ≈ ΔW
    R_init = torch.diag(S_r).to(dtype)  # (rank, rank)

    return B_init, R_init, A_init


def truncated_svd_init_lowrank(
    delta_w: torch.Tensor,
    rank: int,
    dtype: torch.dtype = torch.float32,
    niter: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memory-efficient truncated SVD using torch.svd_lowrank.

    Preferred for large weight matrices in LLMs.
    """
    dw_float = delta_w.float()
    U, S, Vh = torch.svd_lowrank(dw_float, q=rank, niter=niter)
    # U: (out_features, rank), S: (rank,), Vh: (in_features, rank)

    B_init = U.to(dtype)          # (out_features, rank)
    S_r = S                        # (rank,)
    A_init = Vh.T.to(dtype)       # (rank, in_features)
    R_init = torch.diag(S_r).to(dtype)  # (rank, rank)

    return B_init, R_init, A_init


def initialize_lora_sb(
    model: nn.Module,
    dataloader: DataLoader,
    target_module_names: List[str],
    rank: int,
    n_samples: int,
    device: torch.device,
    use_lowrank_svd: bool = True,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Full LoRA-SB initialization pipeline.

    Computes gradient approximation and returns (B, R, A) for each target layer.

    Args:
        model: Pre-trained model.
        dataloader: DataLoader for initialization samples.
        target_module_names: Names of Linear layers to adapt.
        rank: LoRA rank r.
        n_samples: Number of samples for gradient estimation (0.1% of dataset).
        device: Compute device.
        use_lowrank_svd: Use memory-efficient svd_lowrank (recommended for LLMs).

    Returns:
        Dict mapping full module name to (B_init, R_init, A_init).
    """
    delta_w_dict = compute_gradient_approximation(
        model=model,
        dataloader=dataloader,
        target_module_names=target_module_names,
        n_samples=n_samples,
        device=device,
    )

    init_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    svd_fn = truncated_svd_init_lowrank if use_lowrank_svd else truncated_svd_init

    for name, delta_w in delta_w_dict.items():
        # Get the dtype of the original weight
        module = dict(model.named_modules())[name]
        weight_dtype = module.weight.dtype

        B, R, A = svd_fn(delta_w, rank=rank, dtype=weight_dtype)
        init_dict[name] = (B, R, A)

    return init_dict
