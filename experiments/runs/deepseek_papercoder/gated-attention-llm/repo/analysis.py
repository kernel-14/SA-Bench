## analysis.py
"""
Analysis module for the Gated Attention LLM reproduction.

Provides tools to inspect the trained model's internals:
  - Gate sparsity (mean gating scores, fraction of near‑zero scores).
  - Attention sink (proportion of attention directed to the first token).
  - Massive activations (mean of per‑sequence maximum hidden state values).

These metrics correspond to Table 4 and Figures 2–6 in the paper.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import model components needed for type hints and hook attachment.
import model as _model  # avoid circular import issues
import gated_attention as _ga


class Analyzer:
    """
    Instruments a `GPTModel` to extract internal statistics without modifying the model.

    Args:
        model: The trained GPTModel (should be in eval mode, usually unwrapped).
        config: The full configuration dictionary (as loaded from config.yaml).
                Only 'model' subsection is used for model structure; the 'data' section
                may be used for analysis batch configuration.
    """

    def __init__(self, model: _model.GPTModel, config: Dict) -> None:
        self.model = model
        self.config = config
        self.num_layers = model.config["num_layers"]
        self.hooks: List[Tuple[int, torch.utils.hooks.RemovableHandle]] = []
        # Prepare per-layer accumulators for hooks
        self._gate_scores_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        self._attention_weights_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        self._hidden_states_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hooks(
        self,
        capture_gate: bool = True,
        capture_attention: bool = True,
        capture_hidden: bool = True,
    ) -> None:
        """
        Attach forward hooks to capture gate scores, attention weights, and hidden states.

        Args:
            capture_gate: If True, register hooks on each GatedAttention module
                          to collect `gate_scores`.
            capture_attention: If True, register hooks on each GatedAttention module
                               to collect `attention_weights`.
            capture_hidden: If True, register hooks on each TransformerBlock module
                            to collect its output hidden states.
        """
        self._clear_hooks()  # remove any previously registered hooks
        for layer_idx, block in enumerate(self.model.layers):
            if not isinstance(block, _model.TransformerBlock):
                raise TypeError(
                    f"Layer {layer_idx} is not a TransformerBlock; "
                    f"found {type(block)}."
                )

            # Hooks on the attention module
            attn_module = block.attn
            if capture_gate or capture_attention:
                # We need to capture gate_scores and attention_weights after the attention
                # forward. A single hook suffices.
                def attn_hook(_module, _input, _output, idx=layer_idx):
                    # Only collect if the GatedAttention module actually populated the attributes
                    if capture_gate and hasattr(_module, "gate_scores") and _module.gate_scores is not None:
                        self._gate_scores_per_layer[idx].append(_module.gate_scores.detach().cpu())
                    if capture_attention and hasattr(_module, "attention_weights") and _module.attention_weights is not None:
                        self._attention_weights_per_layer[idx].append(_module.attention_weights.detach().cpu())

                handle = attn_module.register_forward_hook(attn_hook)
                self.hooks.append((layer_idx, handle))

            # Hooks on the TransformerBlock output (hidden states)
            if capture_hidden:
                def block_hook(_module, _input, output, idx=layer_idx):
                    self._hidden_states_per_layer[idx].append(output.detach().cpu())

                handle = block.register_forward_hook(block_hook)
                self.hooks.append((layer_idx, handle))

    def _clear_hooks(self) -> None:
        """Remove all previously registered hooks and reset accumulators."""
        for _, handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        # Clear accumulators to free memory
        self._gate_scores_per_layer = [[] for _ in range(self.num_layers)]
        self._attention_weights_per_layer = [[] for _ in range(self.num_layers)]
        self._hidden_states_per_layer = [[] for _ in range(self.num_layers)]

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def analyze_gate_sparsity(
        self,
        dataloader: DataLoader,
        thresholds: List[float] = [0.01, 0.1],
    ) -> Dict[str, Union[List[float], Dict[float, List[float]]]]:
        """
        Compute layer‑wise gate score mean and sparsity (proportion of scores below given thresholds).

        Args:
            dataloader: DataLoader yielding batches of `input_ids`, `labels`, and optional
                        `attention_mask`. The model is called with `labels=None` so we
                        only use `input_ids` and `attention_mask`.
            thresholds: List of scalar thresholds for measuring sparsity (e.g., 0.01, 0.1).

        Returns:
            A dictionary containing:
                - "mean_gate_scores": list of length `num_layers`, each entry is the mean
                  gate score across the dataset.
                - "sparsity": dict mapping threshold -> list of sparsity ratios per layer.
        """
        # Register hooks only for gate scores (attention weights are not needed)
        self.register_hooks(capture_gate=True, capture_attention=False, capture_hidden=False)

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)
                # Forward pass without labels; hooks collect gate_scores
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)

        # Compute per-layer statistics
        num_layers = self.num_layers
        mean_gate_scores = []
        sparsity = {thresh: [] for thresh in thresholds}

        for layer_idx in range(num_layers):
            gate_list = self._gate_scores_per_layer[layer_idx]
            if not gate_list:
                # No gate scores collected (e.g., baseline model without gating)
                mean_gate_scores.append(0.0)
                for thresh in thresholds:
                    sparsity[thresh].append(0.0)
                continue

            # Concatenate all gate scores for this layer into one 1‑D tensor
            all_scores = torch.cat([g.flatten() for g in gate_list])  # CPU
            mean_score = all_scores.mean().item()
            mean_gate_scores.append(mean_score)

            # Compute sparsity for each threshold
            for thresh in thresholds:
                # Sparsity = fraction of values strictly below threshold
                sparse_ratio = (all_scores < thresh).float().mean().item()
                sparsity[thresh].append(sparse_ratio)

        self._clear_hooks()
        return {"mean_gate_scores": mean_gate_scores, "sparsity": sparsity}

    def analyze_attention_sink(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        Compute the proportion of attention allocated to the first token for each layer.

        This corresponds to the "F‑Attn" metric in Table 4 of the paper. Attention weights
        are averaged over all heads and all query positions within a head, then averaged
        over all sequences in the dataset.

        Args:
            dataloader: DataLoader as in `analyze_gate_sparsity`.

        Returns:
            A dictionary with key "f_attn" mapping to a list of length `num_layers`,
            each entry the average fraction of attention directed to the first token.
        """
        self.register_hooks(capture_gate=False, capture_attention=True, capture_hidden=False)

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)

        f_attn_per_layer = []
        for layer_idx in range(self.num_layers):
            attn_list = self._attention_weights_per_layer[layer_idx]
            if not attn_list:
                f_attn_per_layer.append(0.0)
                continue

            # Each tensor shape: (batch, num_heads, seq_len, seq_len)
            # Compute attention to first token (key index 0) averaged over batch, heads, and query positions
            total_first_attn = 0.0
            total_elements = 0  # to weight by number of attention heads/queries
            for attn in attn_list:
                # attn shape: (B, H, T, T)
                first_token_attn = attn[:, :, :, 0]  # (B, H, T)
                # Average over batch, heads, and query positions (dim=0,1,2)
                mean_f = first_token_attn.mean().item()
                batch_size = attn.size(0) * attn.size(1) * attn.size(2)  # total number of query positions across batch and heads
                total_first_attn += mean_f * batch_size
                total_elements += batch_size
            f_attn = total_first_attn / max(total_elements, 1)
            f_attn_per_layer.append(f_attn)

        self._clear_hooks()
        return {"f_attn": f_attn_per_layer}

    def analyze_massive_activations(
        self,
        dataloader: DataLoader,
    ) -> List[float]:
        """
        Measure the mean of the maximum absolute hidden state values per layer.

        This replicates the "M‑Act" column of Table 4.  For each sequence in a batch,
        the maximum absolute value across all tokens and the hidden dimension is taken,
        and then averaged over all batches. The returned list has one entry per layer.

        Args:
            dataloader: DataLoader as before.

        Returns:
            List of length num_layers, each the average maximal activation.
        """
        self.register_hooks(capture_gate=False, capture_attention=False, capture_hidden=True)

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)

        max_act_per_layer = []
        for layer_idx in range(self.num_layers):
            hidden_list = self._hidden_states_per_layer[layer_idx]
            if not hidden_list:
                max_act_per_layer.append(0.0)
                continue

            # Each tensor shape: (batch, seq_len, hidden_size)
            total_max = 0.0
            total_sequences = 0
            for hidden in hidden_list:
                # Compute per‑sequence maximum absolute value over all dims except batch
                flat = hidden.view(hidden.size(0), -1).abs()
                max_vals = flat.max(dim=1)[0]  # shape (batch,)
                total_max += max_vals.sum().item()
                total_sequences += hidden.size(0)

            avg_max = total_max / max(total_sequences, 1)
            max_act_per_layer.append(avg_max)

        self._clear_hooks()
        return max_act_per_layer

