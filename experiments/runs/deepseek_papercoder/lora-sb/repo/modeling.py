"""
modeling.py – LoRA‑SB layer and model wrapper.

Provides:
- LoraSBLayer: a custom nn.Module implementing the low‑rank parameterisation
  W = W₀ + s·B·R·A where only R is trainable.
- ModelWrapper: loads a pre‑trained HuggingFace model, applies LoRA‑SB injection,
  freezes all parameters except the R matrices, and exposes trainable parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, PreTrainedModel

from config import ExperimentConfig  # our configuration dataclass


class LoraSBLayer(nn.Module):
    """
    Linear layer augmented with a low‑rank update following LoRA‑XS architecture.

    Effective weight: W = W₀ + s * B * R * A
      - W₀ is the original frozen weight (stored as a non‑trainable parameter).
      - B ∈ R^{m×r}, A ∈ R^{r×n} are fixed orthonormal matrices (non‑trainable).
      - R ∈ R^{r×r} is the only trainable parameter.
      - s = 1.0 (scale independence, proved in the paper).

    Args:
        linear: the original nn.Linear to be replaced.
        r: rank of the low‑rank decomposition.
        B: initial B matrix (out_features, r).
        A: initial A matrix (r, in_features).
        R_init: initial R matrix (r, r), typically a diagonal matrix of singular values.
        dtype: target dtype for stored tensors (defaults to linear.weight.dtype).
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        B: Tensor,
        A: Tensor,
        R_init: Tensor,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.s: float = 1.0  # scale independence, as proven in Theorem 5

        # Use the original weight dtype if none specified
        if dtype is None:
            dtype = linear.weight.dtype

        # ---- Store original weight and bias as non‑trainable parameters ----
        self.weight_buffer = nn.Parameter(
            linear.weight.data.clone().to(dtype), requires_grad=False
        )
        if linear.bias is not None:
            self.bias_buffer = nn.Parameter(
                linear.bias.data.clone().to(dtype), requires_grad=False
            )
        else:
            self.bias_buffer = None

        # ---- Fixed low‑rank bases (non‑trainable) ----
        self.register_buffer("B", B.clone().to(dtype))
        self.register_buffer("A", A.clone().to(dtype))

        # ---- Trainable middle matrix ----
        self.R = nn.Parameter(R_init.clone().to(dtype))

        # Validate shapes (optional, but guards against mismatched initialisation)
        if not (B.shape == (self.out_features, r)):
            raise ValueError(
                f"B shape {B.shape} does not match (out_features={self.out_features}, r={r})"
            )
        if not (A.shape == (r, self.in_features)):
            raise ValueError(
                f"A shape {A.shape} does not match (r={r}, in_features={self.in_features})"
            )
        if R_init.shape != (r, r):
            raise ValueError(f"R_init shape {R_init.shape} is not ({r}, {r})")

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the low‑rank adapted linear transformation.

        Returns:
            output = x @ (W₀ + s·B·R·A)^T + bias
        """
        # Low‑rank component
        delta_W = self.B @ self.R @ self.A  # shape (out_features, in_features)
        W_eff = self.weight_buffer + self.s * delta_W
        return F.linear(x, W_eff, self.bias_buffer)


class ModelWrapper:
    """
    Loads a pre‑trained model, replaces selected linear layers with LoraSBLayer,
    freezes the whole model except the R matrices, and provides access to
    trainable parameters.

    Args:
        model_name: HuggingFace model identifier (e.g., "mistralai/Mistral-7B-v0.1").
        config: ExperimentConfig containing task, dtype, device, and other settings.
    """

    def __init__(self, model_name: str, config: ExperimentConfig) -> None:
        self.config = config
        self.device: str = config.device
        dtype_str: str = config.dtype  # e.g., "bfloat16"
        self.dtype: torch.dtype = getattr(torch, dtype_str)

        task: str = config.task

        # ---- Model class selection depending on benchmark ----
        if task.startswith("arithmetic") or task.startswith("commonsense"):
            model_cls = AutoModelForCausalLM
            model_kwargs = {}
            self._is_classification = False
        elif task.startswith("glue_"):
            sub_task = task[5:]  # e.g., "cola", "mrpc"
            model_cls = AutoModelForSequenceClassification
            # Number of labels and problem type are defined by the GLUE task.
            num_labels_map = {
                "cola": 2,
                "mrpc": 2,
                "qnli": 2,
                "rte": 2,
                "sst2": 2,
                "stsb": 1,
            }
            if sub_task not in num_labels_map:
                raise ValueError(f"Unknown GLUE sub‑task: {sub_task}")
            num_labels = num_labels_map[sub_task]
            problem_type = (
                "regression" if sub_task == "stsb" else "single_label_classification"
            )
            model_kwargs = {
                "num_labels": num_labels,
                "problem_type": problem_type,
            }
            self._is_classification = True
        else:
            raise ValueError(f"Unsupported task: {task}")

        # ---- Load the model ----
        self.model: PreTrainedModel = model_cls.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            **model_kwargs,
        )
        self.model.to(self.device)

        # List of module names that have been replaced (for logging/debugging)
        self.replaced_modules: list[str] = []

    def apply_lora_sb(
        self,
        B_dict: dict[str, Tensor],
        A_dict: dict[str, Tensor],
        R_dict: dict[str, Tensor],
    ) -> None:
        """
        Replace selected linear layers with LoRA‑SB layers and freeze the model.

        Args:
            B_dict: mapping from full module name (e.g. "model.layers.0.self_attn.q_proj")
                    to initial B tensor.
            A_dict: same structure, initial A tensor.
            R_dict: same structure, initial R tensor.
        """
        for key in B_dict.keys():
            # Split into parent and child attribute name
            parent_name, _, child_name = key.rpartition(".")
            parent = (
                self.model.get_submodule(parent_name) if parent_name else self.model
            )
            old_module = getattr(parent, child_name)
            if not isinstance(old_module, nn.Linear):
                raise TypeError(
                    f"Module '{key}' is not nn.Linear, got {type(old_module)}"
                )
            r = B_dict[key].shape[1]  # secondary dimension is rank
            lora_layer = LoraSBLayer(
                old_module, r, B_dict[key], A_dict[key], R_dict[key], dtype=self.dtype
            )
            setattr(parent, child_name, lora_layer)
            self.replaced_modules.append(key)

        # ---- Freeze all model parameters ----
        for param in self.model.parameters():
            param.requires_grad = False

        # ---- Only R matrices inside LoraSBLayer remain trainable ----
        for module in self.model.modules():
            if isinstance(module, LoraSBLayer):
                module.R.requires_grad = True

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """
        Return the list of trainable parameters after LoRA‑SB injection.
        This will consist solely of the R matrices from every LoraSBLayer.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    # Optionally, expose the bare model for other modules (e.g. Initializer)
    def get_model(self) -> PreTrainedModel:
        return self.model
