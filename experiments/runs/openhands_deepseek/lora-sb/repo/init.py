"""LoRA-SB Initialization: Approximate the first step of full fine-tuning.

This module computes the initial update direction using a small subset of
training data, simulating the first AdamW step using sign(gradient_sum),
then performs truncated SVD to obtain orthonormal bases B and A for the
low-rank subspace.

Algorithm (from paper):
  1. Sample n data points from training set (0.1% of total).
  2. Compute gradient of loss w.r.t. each targeted weight matrix.
  3. Sum gradients and take element-wise sign to simulate AdamW first step.
  4. Perform truncated SVD on the resulting update matrix.
  5. Initialize B = U[:, :r], A = Vh[:r, :], R = diag(S[:r]) / s (s=1).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader


class LoRASBInitializer:
    """Computes LoRA-SB initialization from a subset of training data.

    Uses memory-efficient layer-wise gradient computation:
    hooks into backward pass, computes gradients per layer, and immediately
    discards them to keep O(1) memory usage.

    Args:
        model: The pre-trained base model.
        target_modules: List of module name substrings to target.
        rank: LoRA rank.
        num_samples: Number of samples for initialization (0.1% of dataset).
        device: Device for computation.
        dtype: Data type for computation.
    """

    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
        rank: int = 8,
        num_samples: int = 50,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.num_samples = num_samples
        self.device = device
        self.dtype = dtype

        self.original_weights = {}
        self.accumulated_grads: Dict[str, torch.Tensor] = {}
        self._gradient_hooks = []

    def _get_target_linear_layers(self) -> Dict[str, nn.Linear]:
        """Find all nn.Linear layers whose names match target_modules."""
        targets = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                for target in self.target_modules:
                    if target in name.lower():
                        targets[name] = module
                        break
        return targets

    def _register_hooks(self, layer_map: Dict[str, nn.Linear]):
        """Register backward hooks to accumulate gradients per layer."""
        self.accumulated_grads = {}
        self._gradient_hooks = []

        def make_hook(layer_name: str, original_weight: nn.Parameter):
            def hook(grad: torch.Tensor):
                if layer_name not in self.accumulated_grads:
                    self.accumulated_grads[layer_name] = grad.detach().float().clone()
                else:
                    self.accumulated_grads[layer_name] += grad.detach().float()
            return hook

        for name, layer in layer_map.items():
            if layer.weight.requires_grad:
                hook = layer.weight.register_hook(make_hook(name, layer.weight))
                self._gradient_hooks.append(hook)

    def _remove_hooks(self):
        for hook in self._gradient_hooks:
            hook.remove()
        self._gradient_hooks = []

    def compute_update_approximation(
        self,
        dataloader: DataLoader,
        loss_fn: callable,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute the LoRA-SB initialization (B, A, R) for each target layer.

        Simulates the first AdamW step:
            ΔW_avg = -η * sign(∑ ∇_W L(W0, xi))

        Args:
            dataloader: DataLoader yielding batches of (input_ids, attention_mask, labels).
            loss_fn: Function that takes (model, batch) and returns scalar loss.

        Returns:
            Dictionary mapping layer names to (B, A, R) initialization tensors.
        """
        layer_map = self._get_target_linear_layers()
        print(f"Found {len(layer_map)} target linear layers for initialization.")

        self._register_hooks(layer_map)

        processed = 0
        self.model.train()
        for batch in dataloader:
            if processed >= self.num_samples:
                break
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            self.model.zero_grad()
            loss = loss_fn(self.model, batch)
            loss.backward()
            processed += batch.get("input_ids", batch.get("labels")).size(0)
            print(f"  Processed {processed}/{self.num_samples} samples", end="\r")

        self._remove_hooks()
        print(f"\nDone. Accumulated gradients for {len(self.accumulated_grads)} layers.")

        init_dict = {}
        for name, accumulated_grad in self.accumulated_grads.items():
            delta_W_avg = -torch.sign(accumulated_grad)
            B, A, R = self._svd_initialize(delta_W_avg)
            init_dict[name] = (B, A, R)

        return init_dict

    def _svd_initialize(
        self,
        delta_W: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform truncated SVD on ΔW_avg and return (B, A, R).

        B_init = U[:, :r]                    (m x r, orthonormal columns)
        A_init = Vh[:r, :]                   (r x n, orthonormal rows)
        R_init = diag(S[:r])                 (r x r)

        This ensures B^T B = I, A A^T = I, and s=1 for scaling independence.
        """
        U, S, Vh = torch.linalg.svd(delta_W.float(), full_matrices=False)
        r = min(self.rank, len(S))

        B_init = U[:, :r].to(dtype=self.dtype)
        A_init = Vh[:r, :].to(dtype=self.dtype)
        R_init = torch.diag(S[:r]).to(dtype=self.dtype)

        return B_init, A_init, R_init


def prepare_init_loader(dataset, num_samples: int, batch_size: int = 8, seed: int = 42) -> DataLoader:
    """Create a small DataLoader for initialization with randomly sampled data.

    Args:
        dataset: The full training dataset.
        num_samples: Number of samples to select for initialization.
        batch_size: Batch size for gradient computation.
        seed: Random seed for reproducibility.

    Returns:
        DataLoader yielding batches from the sampled subset.
    """
    total = len(dataset)
    num_samples = min(num_samples, total)
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(seed))[:num_samples]
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


def compute_lora_sb_init(
    model: nn.Module,
    dataloader: DataLoader,
    target_modules: List[str],
    rank: int,
    num_samples: int = 50,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """High-level function to compute LoRA-SB initialization.

    Returns dict mapping layer names to (B, A, R) tensors.
    """
    initializer = LoRASBInitializer(
        model=model,
        target_modules=target_modules,
        rank=rank,
        num_samples=num_samples,
        device=device,
        dtype=dtype,
    )

    def default_loss_fn(model, batch):
        outputs = model(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )
        return outputs.loss

    return initializer.compute_update_approximation(dataloader, default_loss_fn)
