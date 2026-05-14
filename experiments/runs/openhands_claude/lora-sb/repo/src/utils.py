"""Utility functions: ablation experiments, parameter counting, logging."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters by type (trainable vs frozen)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    return {"trainable": trainable, "frozen": frozen, "total": total}


def verify_orthonormality(model: nn.Module, tol: float = 1e-4) -> Dict[str, bool]:
    """Verify that LoRA-SB B and A matrices are orthonormal (B^T B = I, A A^T = I)."""
    results = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora") and hasattr(module.lora, "lora_B"):
            lora = module.lora
            if hasattr(lora, "lora_B") and hasattr(lora, "lora_A"):
                B = lora.lora_B  # (out_features, rank)
                A = lora.lora_A  # (rank, in_features)

                BtB = B.T @ B
                AAt = A @ A.T
                r = B.shape[1]

                B_ortho = torch.allclose(BtB, torch.eye(r, device=B.device, dtype=B.dtype), atol=tol)
                A_ortho = torch.allclose(AAt, torch.eye(r, device=A.device, dtype=A.dtype), atol=tol)
                results[name] = {"B_orthonormal": B_ortho, "A_orthonormal": A_ortho}
    return results


def compute_equivalent_gradient_norm(
    model: nn.Module,
    layer_name: str,
) -> Optional[float]:
    """Compute the Frobenius norm of the equivalent gradient g_tilde = B * g^R * A.

    This is used to verify that the equivalent gradient approximates the full FT gradient.
    """
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, "lora"):
            lora = module.lora
            if hasattr(lora, "lora_R") and lora.lora_R.grad is not None:
                B = lora.lora_B.float()
                A = lora.lora_A.float()
                g_R = lora.lora_R.grad.float()
                g_tilde = B @ g_R @ A
                return g_tilde.norm(p="fro").item()
    return None


def save_adapter_weights(model: nn.Module, path: str) -> None:
    """Save only the trainable adapter weights."""
    state = {k: v for k, v in model.state_dict().items()
             if any(tag in k for tag in ["lora_R", "lora_A", "lora_B", "magnitude"])}
    torch.save(state, path)
    print(f"Saved {len(state)} adapter tensors to {path}")


def load_adapter_weights(model: nn.Module, path: str) -> None:
    """Load adapter weights into model."""
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded adapter weights: {len(state)} tensors")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")


# ---------------------------------------------------------------------------
# Ablation: initialization strategies (Section 4, Table 4)
# ---------------------------------------------------------------------------

def init_kaiming_svd(
    weight: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ablation: initialize from Kaiming random matrix (truncated SVD).

    This corresponds to 'trunc_SVD (Kaiming)' in Table 4.
    """
    import math
    kaiming = torch.empty_like(weight)
    nn.init.kaiming_uniform_(kaiming, a=math.sqrt(5))
    U, S, Vh = torch.linalg.svd(kaiming.float(), full_matrices=False)
    B = U[:, :rank].to(weight.dtype)
    R = torch.diag(S[:rank]).to(weight.dtype)
    A = Vh[:rank, :].to(weight.dtype)
    return B, R, A


def init_noisy_delta_w_svd(
    delta_w: torch.Tensor,
    rank: int,
    noise_std: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ablation: initialize from noisy ΔW_avg (truncated SVD).

    Corresponds to 'trunc_SVD (ΔW_avg + N_{μ=noise_std})' in Table 4.
    """
    noise = torch.randn_like(delta_w) * noise_std
    noisy_dw = delta_w + noise
    U, S, Vh = torch.linalg.svd(noisy_dw.float(), full_matrices=False)
    B = U[:, :rank].to(delta_w.dtype)
    R = torch.diag(S[:rank]).to(delta_w.dtype)
    A = Vh[:rank, :].to(delta_w.dtype)
    return B, R, A


def init_non_orthogonal_svd(
    delta_w: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ablation: initialize without orthonormal B (B = U*S, A = Vh, R = I).

    This ensures B_init @ R_init @ A_init ≈ ΔW_avg but B^T B ≠ I.
    Used to isolate the effect of optimal gradient approximation (Section 4).
    """
    U, S, Vh = torch.linalg.svd(delta_w.float(), full_matrices=False)
    S_r = S[:rank]
    # B = U * S (absorb singular values into B)
    B = (U[:, :rank] * S_r.unsqueeze(0)).to(delta_w.dtype)  # (out, rank)
    A = Vh[:rank, :].to(delta_w.dtype)                       # (rank, in)
    R = torch.eye(rank, dtype=delta_w.dtype)                 # identity
    return B, R, A


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Simple training logger that writes metrics to a JSON file."""

    def __init__(self, log_dir: str, run_name: str) -> None:
        self.log_dir = log_dir
        self.run_name = run_name
        self.log_path = os.path.join(log_dir, f"{run_name}_log.json")
        os.makedirs(log_dir, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        record = {"step": step, **metrics}
        self.records.append(record)

    def save(self) -> None:
        with open(self.log_path, "w") as f:
            json.dump(self.records, f, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.save()
