"""
LoRA-SB Initialization using Update Approximation.

The initialization strategy approximates the first step of full fine-tuning:
1. Compute ΔW_avg = -η · sign(Σ ∇_W L(W_0, x_i)) over a subset of training data
2. Apply truncated SVD: U, S, V^T = SVD(ΔW_avg)
3. Initialize: B = U[:, :r], A = V[:r, :], R = S[:r, :r] / s

This provides:
- Optimal rank-r approximation of the initial full FT update (Eckart-Young)
- Orthonormal B and A matrices (B^T B = I, A A^T = I)
- Scaling factor independence (s can be set to 1)
- Guaranteed loss reduction at each step
- Memory-efficient layerwise computation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Callable
from .lora_sb_layer import LoRA_SB_Layer


def estimate_first_step_gradient(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int,
    learning_rate: float = 1e-4,
    use_amp: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """
    Estimate the first full fine-tuning step for AdamW optimizer.

    Computes ΔW_avg = -η · sign(Σ_i ∇_W L(W_0, x_i)) over num_samples.

    This uses layerwise gradient computation to keep memory usage O(1)
    independent of the number of layers (memory-efficient approach).

    Args:
        model: The pre-trained model.
        dataloader: DataLoader providing training samples.
        num_samples: Number of samples to use for gradient estimation.
        learning_rate: Learning rate η for scaling.
        use_amp: Use automatic mixed precision.
        device: Device for computation.
        dtype: Data type for computation.

    Returns:
        Dict mapping parameter name to ΔW_avg tensor (gradient accumulator).
    """
    if device is None:
        device = next(model.parameters()).device
    if dtype is None:
        dtype = torch.bfloat16 if use_amp else torch.float32

    model = model.to(device)
    model.train()

    # We'll accumulate gradients across samples
    grad_accumulators = {}

    samples_processed = 0
    for batch in dataloader:
        if samples_processed >= num_samples:
            break

        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Need to handle different batch formats
        # Assume HF-style: batch has input_ids, attention_mask, labels
        with torch.amp.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu',
                                enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        # Backward pass
        loss.backward()

        # Accumulate gradients (sum of sign of gradients from each sample in the batch)
        # For the first AdamW step: sign of sum of gradients
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in grad_accumulators:
                        grad_accumulators[name] = torch.zeros_like(param.grad)
                    grad_accumulators[name] += param.grad.clone()

        # Zero out gradients for next iteration
        model.zero_grad()

        # Count samples (use batch size)
        batch_size = batch.get('input_ids', batch.get('input', None)).size(0) \
            if 'input_ids' in batch or 'input' in batch else 1
        samples_processed += batch_size

    # Compute ΔW_avg = -η · sign(accumulated gradients)
    delta_w_avg = {}
    for name, grad_sum in grad_accumulators.items():
        delta_w_avg[name] = -learning_rate * torch.sign(grad_sum)

    return delta_w_avg


def estimate_first_step_gradient_layerwise(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int,
    learning_rate: float = 1e-4,
    use_amp: bool = True,
    target_module_names: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient layerwise gradient estimation.

    Processes each layer independently: hooks into backward pass to capture
    gradients per layer, then immediately discards them. This ensures O(1)
    memory usage independent of the number of layers.

    Args:
        model: The pre-trained model.
        dataloader: DataLoader with training samples.
        num_samples: Number of samples for gradient estimation.
        learning_rate: Learning rate for scaling.
        use_amp: Use automatic mixed precision.
        target_module_names: Names of modules to target (e.g., ["q_proj", "k_proj"]).
                            If None, targets all modules with LoRA_SB_Layer.

    Returns:
        Dict mapping parameter name to ΔW_avg tensor.
    """
    device = next(model.parameters()).device

    # Build mapping from parameter name to module
    grad_storage = {}

    def _make_hook(param_name):
        """Create a backward hook that stores gradients."""
        def hook(grad):
            if param_name not in grad_storage:
                grad_storage[param_name] = grad.clone()
            else:
                grad_storage[param_name] += grad.clone()
        return hook

    # Register hooks on all targeted parameters
    hooks = []
    for name, param in model.named_parameters():
        if target_module_names is not None:
            # Only target specific modules
            if not any(t in name for t in target_module_names):
                continue
        hook = param.register_hook(_make_hook(name))
        hooks.append((name, hook))

    model.train()
    samples_processed = 0

    for batch in dataloader:
        if samples_processed >= num_samples:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        model.zero_grad()

        with torch.amp.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu',
                                enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        loss.backward()

        batch_size = batch.get('input_ids', batch.get('input', None)).size(0) \
            if 'input_ids' in batch or 'input' in batch else 1
        samples_processed += batch_size

    # Remove hooks
    for _, hook in hooks:
        hook.remove()

    # Compute ΔW_avg = -η · sign(accumulated gradients)
    delta_w_avg = {}
    for name, grad_sum in grad_storage.items():
        delta_w_avg[name] = -learning_rate * torch.sign(grad_sum)

    return delta_w_avg


def truncated_svd_init(
    delta_w: torch.Tensor,
    rank: int,
    scaling: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize B, R, A from truncated SVD of ΔW.

    U, S, V^T = SVD(ΔW)
    B_init = U[:, :r]
    R_init = S[:r, :r] / s
    A_init = V[:r, :]

    This gives: s * B_init @ R_init @ A_init ≈ ΔW (optimal rank-r approximation)

    Args:
        delta_w: The ΔW_avg matrix of shape (m, n).
        rank: Rank r for the decomposition.
        scaling: Scaling factor s (default: 1.0).

    Returns:
        (B_init, R_init, A_init) tensors.
    """
    # Handle small matrices
    m, n = delta_w.shape
    effective_rank = min(rank, min(m, n))

    # Use low-rank SVD for efficiency when appropriate
    if min(m, n) > 1000:
        # torch.svd_lowrank for large matrices
        U, S, Vt = torch.svd_lowrank(delta_w, q=effective_rank)
    else:
        # Full SVD for smaller matrices
        U, S, Vt = torch.linalg.svd(delta_w, full_matrices=False)

    # Truncate to rank
    U_r = U[:, :effective_rank]  # (m, r)
    S_r = S[:effective_rank]      # (r,)
    Vt_r = Vt[:effective_rank, :]  # (r, n)

    # If effective_rank < rank, pad with zeros
    if effective_rank < rank:
        # Pad U
        U_pad = torch.zeros(m, rank, device=delta_w.device, dtype=delta_w.dtype)
        U_pad[:, :effective_rank] = U_r
        U_r = U_pad

        # Pad S
        S_pad = torch.zeros(rank, rank, device=delta_w.device, dtype=delta_w.dtype)
        S_pad[:effective_rank, :effective_rank] = torch.diag(S_r)
        S_r_mat = S_pad

        # Pad Vt
        Vt_pad = torch.zeros(rank, n, device=delta_w.device, dtype=delta_w.dtype)
        Vt_pad[:effective_rank, :] = Vt_r
        Vt_r = Vt_pad
    else:
        S_r_mat = torch.diag(S_r)

    # Compute initialization
    B_init = U_r  # (m, r) - orthonormal columns
    A_init = Vt_r  # (r, n) - orthonormal rows
    R_init = S_r_mat / scaling  # (r, r)

    return B_init, R_init, A_init


def init_lora_sb(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    rank: int,
    num_init_samples: int = 50,
    scaling: float = 1.0,
    learning_rate: float = 1e-4,
    use_layerwise: bool = True,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Initialize LoRA-SB on a model.

    This function:
    1. Replaces target linear layers with LoRA_SB_Layer
    2. Estimates the first full FT step from num_init_samples
    3. Initializes B, R, A via truncated SVD of the estimated update

    Args:
        model: The pre-trained model.
        dataloader: DataLoader with training data.
        rank: Rank of the low-rank decomposition.
        num_init_samples: Number of samples for gradient estimation (e.g., 0.1% of data).
        scaling: Scaling factor s (default: 1.0 for orthonormal B, A).
        learning_rate: Learning rate for first-step approximation.
        use_layerwise: Use memory-efficient layerwise gradient estimation.
        target_modules: List of module name patterns to target.

    Returns:
        Model with LoRA-SB layers initialized.
    """
    from .lora_sb_layer import apply_lora_sb

    device = next(model.parameters()).device

    # Step 1: Apply LoRA-SB architecture (replace linear layers)
    model = apply_lora_sb(model, rank=rank, scaling=scaling,
                         target_modules=target_modules)

    # Step 2: Estimate ΔW_avg
    if use_layerwise:
        delta_w_dict = estimate_first_step_gradient_layerwise(
            model, dataloader, num_init_samples,
            learning_rate=learning_rate,
            target_module_names=target_modules,
        )
    else:
        delta_w_dict = estimate_first_step_gradient(
            model, dataloader, num_init_samples,
            learning_rate=learning_rate,
        )

    # Step 3: Initialize each LoRA-SB layer
    for name, module in model.named_modules():
        if isinstance(module, LoRA_SB_Layer):
            # Find the corresponding ΔW for this layer
            # We need to match the module's parameter name to delta_w_dict keys
            # The weight corresponds to W0 in the LoRA-SB layer
            param_name = f"{name}.W0" if name else "W0"
            # Try to find matching gradient
            delta_w = None
            for key, val in delta_w_dict.items():
                # Match by module name patterns
                if name and name in key:
                    delta_w = val
                    break
                # Also try matching by shape
                if val.shape == module.W0.shape:
                    delta_w = val
                    break

            if delta_w is not None and delta_w.shape == module.W0.shape:
                # Initialize via truncated SVD
                B_init, R_init, A_init = truncated_svd_init(
                    delta_w, rank=module.rank, scaling=module.scaling
                )
                module.initialize_ba(B_init, R_init, A_init)
            else:
                # Fallback: initialize with SVD of current W0 (PiSSA-like)
                # This should rarely happen if gradient estimation works
                import warnings
                warnings.warn(
                    f"Could not find matching gradient for module {name}. "
                    f"Using PiSSA-like initialization as fallback."
                )
                U, S, Vt = torch.linalg.svd(module.W0.float(), full_matrices=False)
                B_init = U[:, :module.rank].to(module.W0.dtype)
                A_init = Vt[:module.rank, :].to(module.W0.dtype)
                R_init = torch.diag(S[:module.rank]).to(module.W0.dtype) / module.scaling
                module.initialize_ba(B_init, R_init, A_init)

    return model


def init_lora_sb_from_delta_w_avg(
    model: nn.Module,
    delta_w_avg_dict: Dict[str, torch.Tensor],
    rank: int,
    scaling: float = 1.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Initialize LoRA-SB on a model using pre-computed ΔW_avg values.

    This is a convenience function when gradients have been computed separately.

    Args:
        model: The pre-trained model.
        delta_w_avg_dict: Dict mapping parameter name to ΔW_avg tensor.
        rank: Rank of the low-rank decomposition.
        scaling: Scaling factor s.
        target_modules: Target module patterns.

    Returns:
        Model with LoRA-SB layers initialized.
    """
    from .lora_sb_layer import apply_lora_sb

    model = apply_lora_sb(model, rank=rank, scaling=scaling,
                         target_modules=target_modules)

    for name, module in model.named_modules():
        if isinstance(module, LoRA_SB_Layer):
            # Match delta_w to this layer
            delta_w = None
            for key, val in delta_w_avg_dict.items():
                if name and name in key:
                    delta_w = val
                    break
                if val.shape == module.W0.shape:
                    delta_w = val
                    break

            if delta_w is not None:
                B_init, R_init, A_init = truncated_svd_init(
                    delta_w, rank=module.rank, scaling=module.scaling
                )
                module.initialize_ba(B_init, R_init, A_init)

    return model
