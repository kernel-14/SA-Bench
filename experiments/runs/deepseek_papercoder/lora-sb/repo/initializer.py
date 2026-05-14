"""
initializer.py – LoRA‑SB initialisation via first‑step full fine‑tuning approximation.

This module implements the core contribution of the paper:
computing B, A, and R by performing a truncated SVD of
    ΔW_avg = -lr * sign( Σ ∇_W L(W₀, x_i) )
over a small random subset (0.1 %) of the training data.

The resulting B and A are orthonormal, and R is diagonal,
guaranteeing optimal gradient approximation and scale independence.

Classes:
    Initializer  – performs the full initialisation pipeline.

"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Any
from collections import defaultdict

from config import ExperimentConfig  # our own configuration dataclass


class Initializer:
    """
    Initialises LoRA‑SB matrices by approximating the first AdamW step.

    Args:
        config: ExperimentConfig with task, target_modules, and rank r.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config: ExperimentConfig = config
        self.r: int = config.r
        self.target_modules: List[str] = config.target_modules
        # dtype derived from the config (stored for later casting)
        self.dtype: torch.dtype = getattr(torch, self.config.dtype)

    def compute_init(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        lr: float,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Main entry point: compute B, A, R for each selected linear layer.

        Steps:
          1. Accumulate per‑sample gradients over the provided dataloader (n samples).
          2. For every layer, derive ΔW_avg = -lr * sign(sum_grad),
             then perform truncated SVD to obtain B, A, R.

        Args:
            model: Original pre‑trained model (without LoRA‑SB injection).
            dataloader: DataLoader yielding single‑sample batches (n total).
            lr: Learning rate used for LoRA‑SB training (defines the step size).

        Returns:
            Tuple of three dictionaries, each mapping module full name to tensor:
            (B_dict, A_dict, R_dict).
            These are ready to be passed to ModelWrapper.apply_lora_sb().
        """
        # 1. Accumulate gradients
        grad_sums = self._accumulate_gradients(model, dataloader)

        # 2. Per‑layer SVD initialisation
        B_dict: Dict[str, torch.Tensor] = {}
        A_dict: Dict[str, torch.Tensor] = {}
        R_dict: Dict[str, torch.Tensor] = {}

        for name, grad_sum in grad_sums.items():
            B_init, A_init, R_init = self._svd_init(grad_sum, lr)
            B_dict[name] = B_init
            A_dict[name] = A_init
            R_dict[name] = R_init

        return B_dict, A_dict, R_dict

    def _accumulate_gradients(
        self, model: nn.Module, dataloader: DataLoader
    ) -> Dict[str, torch.Tensor]:
        """
        Sum gradients of target linear weights over all samples in `dataloader`.

        Uses backward hooks that fire once per gradient computation.
        All hooks are removed before returning.

        Args:
            model: The original model (set to train mode for gradient computation).
            dataloader: Yields batches (batch_size = 1) of tokenized data.

        Returns:
            Dictionary mapping full module name -> summed gradient tensor (same shape as weight).
        """
        device = next(model.parameters()).device
        model.train()  # enable gradient computation
        model.zero_grad()

        # Identify target layers and prepare accumulator
        accum: Dict[str, torch.Tensor] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        for name, module in model.named_modules():
            if any(name.endswith("." + tgt) for tgt in self.target_modules):
                if isinstance(module, nn.Linear):
                    # Weight shape: (out_features, in_features)
                    accum[name] = torch.zeros(
                        module.weight.shape, dtype=self.dtype, device=device
                    )

                    # Create a hook factory that captures `name` and `accum`
                    def _hook_factory(module_name: str):
                        def _hook(grad: torch.Tensor) -> None:
                            # Accumulate gradient clone to avoid graph retention
                            accum[module_name] += grad.detach().clone()
                        return _hook

                    hook_handle = module.weight.register_hook(_hook_factory(name))
                    hooks.append(hook_handle)

        # Process all samples
        for batch in dataloader:
            # Move batch to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward + backward
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            if loss is None:
                raise RuntimeError(
                    "Model output does not contain 'loss'. "
                    "Ensure labels are provided in the batch and the model supports loss computation."
                )
            loss.backward()
            model.zero_grad()

        # Remove hooks to avoid interfering with future training
        for handle in hooks:
            handle.remove()

        return accum

    def _svd_init(
        self, grad_sum: torch.Tensor, lr: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute B, A, R from a summed gradient via truncated SVD.

        The procedure:
          1. Cast grad_sum to float32 for numerical stability.
          2. Compute ΔW_avg = -lr * sign(grad_sum).
          3. Perform thin SVD: U, S, Vh = svd(ΔW_avg).
          4. Extract the top‑r singular vectors:
             B_init = U[:, :r]          (m × r)
             A_init = Vh[:r, :]         (r × n)
             R_init = diag(S[:r])       (r × r)
          5. Cast all three tensors back to the model's dtype (e.g., bfloat16).

        Args:
            grad_sum: Accumulated gradient tensor of shape (out_features, in_features).
            lr: Learning rate to scale the sign direction.

        Returns:
            (B_init, A_init, R_init) tensors ready for LoRA‑SB injection.
        """
        # Use float32 for sign and SVD to avoid precision issues
        G = grad_sum.to(torch.float32)
        delta_W = -lr * torch.sign(G)   # First AdamW step approximation

        # Thin SVD (full_matrices=False gives economy‑sized U, Vh)
        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)

        # Rank‑r truncation
        r = min(self.r, len(S))   # safety if r > available singular values
        U_r = U[:, :r]               # (m, r)
        S_r = S[:r]                  # (r,)
        Vh_r = Vh[:r, :]             # (r, n)

        R_init = torch.diag(S_r)

        # Cast back to the model dtype
        B_init = U_r.to(self.dtype)
        A_init = Vh_r.to(self.dtype)
        R_init = R_init.to(self.dtype)

        return B_init, A_init, R_init
