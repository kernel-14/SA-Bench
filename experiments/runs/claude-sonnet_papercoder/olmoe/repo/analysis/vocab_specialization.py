```python
## analysis/vocab_specialization.py
"""Vocabulary specialization analysis for OLMoE-1B-7B.

Measures how strongly individual experts specialize on specific vocabulary
items (token IDs), as described in Section 5.4 of the OLMoE paper.

Vocabulary specialization is defined as:
    VocabSpecialization(E_i, x) = N_{x, E_i}^(k) / N_x

where:
    E_i = the i-th expert in the model
    x   = a token ID (vocabulary element)
    k   = number of top-k experts considered (k=1 for main figure, k=8 for appendix)
    N_{x, E_i}^(k) = number of times token x is routed to E_i (in top-k)
    N_x = total number of times token x appears in the evaluation data

Two modes are supported:
    'input':  x is the current input token ID at each position
    'output': x is the next (ground-truth) token ID — captures which expert
              processes context when the model is about to predict token x

Key findings from the paper (Section 5.4, Figure 23, Table 8):
    - Later layers show higher specialization than earlier layers
    - Later layers specialize more on output token IDs than input token IDs
    - Expert 27 (layer 7): ~90% specialization on non-alphabetic tokens
    - Expert 7 (layer 7): religious/spiritual terms (Jesus, God, pray, Holy)
    - Expert 37 (layer 7): temporal terms (Sunday, Christmas, Olympic, days)
    - Expert 43 (layer 7): geographic/geopolitical terms (Iraq, Iran, Saudi)
    - Experts 48 and 23 (layer 7): connector words (Then, Therefore, So)
    - Expert 4 (layer 7): measurement terms (sq, YR, GHz)

Configuration values used (from config.yaml analysis.vocab_specialization):
    analysis.vocab_specialization.top_k_values: [1, 8]
    analysis.vocab_specialization.modes: ["input", "output"]
    analysis.vocab_specialization.min_token_occurrences: 8
    analysis.vocab_specialization.top_n_tokens_per_expert: 10
    model.num_experts: 64
    model.top_k: 8
    model.num_layers: 16
    model.vocab_size: 50304
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from config import OLMoEConfig
from model.olmoe_model import OLMoEModel, OLMoEOutput
from utils.logging_utils import get_logger

logger: logging.Logger = get_logger("olmoe.analysis.vocab_specialization")

# ---------------------------------------------------------------------------
# Optional matplotlib/seaborn imports for visualization.
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE: bool = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore[assignment]
    gridspec = None  # type: ignore[assignment]
    logger.warning(
        "matplotlib not available. Plotting will be disabled. "
        "Install with: pip install matplotlib"
    )

try:
    import seaborn as sns
    SEABORN_AVAILABLE: bool = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None  # type: ignore[assignment]
    logger.warning(
        "seaborn not available. Falling back to matplotlib for heatmaps. "
        "Install with: pip install seaborn"
    )

# ---------------------------------------------------------------------------
# Optional tqdm import for progress bars.
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
    TQDM_AVAILABLE: bool = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Configuration constants from config.yaml (analysis.vocab_specialization).
# ---------------------------------------------------------------------------
DEFAULT_TOP_K_VALUES: List[int] = [1, 8]
"""Analyze both k=1 and k=8 (config.yaml: analysis.vocab_specialization.top_k_values)."""

DEFAULT_MODES: List[str] = ["input", "output"]
"""Input and output token modes (config.yaml: analysis.vocab_specialization.modes)."""

DEFAULT_MIN_TOKEN_OCCURRENCES: int = 8
"""Minimum token occurrences to include in analysis (config.yaml: analysis.vocab_specialization.min_token_occurrences)."""

DEFAULT_TOP_N_TOKENS_PER_EXPERT: int = 10
"""Top-n tokens to display per expert (config.yaml: analysis.vocab_specialization.top_n_tokens_per_expert)."""

DEFAULT_INFERENCE_BATCH_SIZE: int = 8
"""Batch size for inference during analysis."""

DEFAULT_NUM_LAYERS: int = 16
"""Number of transformer layers (config.yaml: model.num_layers)."""

DEFAULT_NUM_EXPERTS: int = 64
"""Total experts per layer (config.yaml: model.num_experts)."""

DEFAULT_TOP_K: int = 8
"""Activated experts per token (config.yaml: model.top_k)."""

DEFAULT_VOCAB_SIZE: int = 50304
"""GPT-NeoX tokenizer vocabulary size (config.yaml: model.vocab_size)."""

# Uniform routing baselines from config.yaml (analysis.vocab_specialization).
UNIFORM_BASELINE_K1: float = 1.0 / 64.0
"""Uniform baseline for k=1: 1/64 ≈ 1.5625%."""

UNIFORM_BASELINE_K8: float = 8.0 / 64.0
"""Uniform baseline for k=8: 8/64 = 12.5%."""

# Layer index used for Table 8-style expert token analysis (paper uses layer 7).
TABLE8_LAYER_IDX: int = 7
"""Layer index for Table 8-style analysis (0-indexed, paper's layer 7)."""

# Expert IDs highlighted in the paper's Table 8 (layer 7).
TABLE8_EXPERT_IDS: List[int] = [27, 58, 7, 37, 43, 4, 0, 3, 48, 23]
"""Expert IDs highlighted in Table 8 of the paper."""


class VocabSpecializationAnalyzer:
    """Analyzes vocabulary specialization of experts in OLMoE-1B-7B.

    Measures how strongly individual experts specialize on specific vocabulary
    items (token IDs) in both input and output token modes. Implements the
    analysis from Section 5.4 of the paper, reproducing Figure 23 and Table 8.

    The analysis runs inference on evaluation data (typically 0.5% of C4
    validation) and accumulates routing counts per (layer, expert, token_id).
    These counts are normalized to produce specialization scores in [0, 1].

    Key findings to reproduce:
        - Specialization increases monotonically from layer 0 to layer 15
        - Later layers specialize more on output tokens than input tokens
        - Expert 27 (layer 7): ~90% specialization on non-alphabetic tokens
        - Expert 7 (layer 7): religious/spiritual terms
        - Expert 37 (layer 7): temporal terms
        - Expert 43 (layer 7): geographic/geopolitical terms
        - Experts 48 and 23 (layer 7): connector words (~60% co-activation)
        - Expert 4 (layer 7): measurement terms (arXiv/GitHub domain)

    Attributes:
        model: The OLMoEModel loaded with the final pretraining checkpoint.
        tokenizer: GPT-NeoX tokenizer (vocab_size=50304).
        num_experts: Total experts per layer = 64 (config.yaml: model.num_experts).
        num_layers: Number of transformer layers = 16 (config.yaml: model.num_layers).
        vocab_size: Tokenizer vocabulary size = 50304 (config.yaml: model.vocab_size).
        top_k: Activated experts per token = 8 (config.yaml: model.top_k).
        device: Device inferred from model parameters.
        inference_batch_size: Batch size for inference during analysis.
        min_token_occurrences: Minimum token occurrences for valid specialization.
        top_n_tokens_per_expert: Number of top tokens to return per expert.

    Example:
        >>> config = OLMoEConfig()
        >>> model = OLMoEModel(config)
        >>> # Load final checkpoint weights into model before creating analyzer
        >>> analyzer = VocabSpecializationAnalyzer(model, tokenizer, config)
        >>> results_input = analyzer.compute_specialization(c4_val_dataset, k=1, mode='input')
        >>> results_output = analyzer.compute_specialization(c4_val_dataset, k=1, mode='output')
        >>> analyzer.plot_specialization(results_input, output_path="outputs/analysis")
    """

    def __init__(
        self,
        model: OLMoEModel,
        tokenizer: Any,
        config: OLMoEConfig,
        inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
        min_token_occurrences: int = DEFAULT_MIN_TOKEN_OCCURRENCES,
        top_n_tokens_per_expert: int = DEFAULT_TOP_N_TOKENS_PER_EXPERT,
    ) -> None:
        """Initialize VocabSpecializationAnalyzer.

        Args:
            model: The OLMoEModel loaded with the final pretraining checkpoint.
                   Must already have weights loaded and be on the correct device.
                   The analyzer calls model.eval() before inference and does not
                   modify model weights.
            tokenizer: GPT-NeoX tokenizer (EleutherAI/gpt-neox-20b).
                       Used for decoding token IDs to strings in
                       get_top_tokens_per_expert(). vocab_size must be 50304
                       (config.yaml: model.vocab_size).
            config: OLMoEConfig instance. Key fields used:
                    - num_experts (64): number of experts per layer
                    - num_layers (16): number of transformer layers
                    - vocab_size (50304): tokenizer vocabulary size
                    - top_k (8): activated experts per token
            inference_batch_size: Number of sequences per inference batch.
                                  Default: 8. Reduce if OOM errors occur during
                                  analysis of 4096-token sequences.
            min_token_occurrences: Minimum number of times a token must appear
                                   in the evaluation data to be included in
                                   specialization analysis. Default: 8.
                                   From config.yaml: analysis.vocab_specialization.min_token_occurrences.
            top_n_tokens_per_expert: Number of top tokens to return per expert
                                     in get_top_tokens_per_expert(). Default: 10.
                                     From config.yaml: analysis.vocab_specialization.top_n_tokens_per_expert.

        Raises:
            ValueError: If config has invalid values.
        """
        if config.num_experts <= 0:
            raise ValueError(
                f"config.num_experts must be positive, got {config.num_experts}."
            )
        if config.top_k <= 0 or config.top_k > config.num_experts:
            raise ValueError(
                f"config.top_k must be in [1, num_experts], "
                f"got top_k={config.top_k}, num_experts={config.num_experts}."
            )
        if config.num_layers <= 0:
            raise ValueError(
                f"config.num_layers must be positive, got {config.num_layers}."
            )
        if config.vocab_size <= 0:
            raise ValueError(
                f"config.vocab_size must be positive, got {config.vocab_size}."
            )
        if inference_batch_size <= 0:
            raise ValueError(
                f"inference_batch_size must be positive, got {inference_batch_size}."
            )
        if min_token_occurrences < 0:
            raise ValueError(
                f"min_token_occurrences must be >= 0, got {min_token_occurrences}."
            )
        if top_n_tokens_per_expert <= 0:
            raise ValueError(
                f"top_n_tokens_per_expert must be positive, got {top_n_tokens_per_expert}."
            )

        self.model: OLMoEModel = model
        """The OLMoEModel loaded with the final pretraining checkpoint."""

        self.tokenizer: Any = tokenizer
        """GPT-NeoX tokenizer for decoding token IDs to strings."""

        self.num_experts: int = config.num_experts
        """Total experts per layer = 64 (config.yaml: model.num_experts)."""

        self.num_layers: int = config.num_layers
        """Number of transformer layers = 16 (config.yaml: model.num_layers)."""

        self.vocab_size: int = config.vocab_size
        """Tokenizer vocabulary size = 50304 (config.yaml: model.vocab_size)."""

        self.top_k: int = config.top_k
        """Activated experts per token = 8 (config.yaml: model.top_k)."""

        self.inference_batch_size: int = inference_batch_size
        """Batch size for inference during analysis."""

        self.min_token_occurrences: int = min_token_occurrences
        """Minimum token occurrences for valid specialization scores."""

        self.top_n_tokens_per_expert: int = top_n_tokens_per_expert
        """Number of top tokens to return per expert."""

        # Infer device from model parameters.
        try:
            first_param: Tensor = next(model.parameters())
            self.device: torch.device = first_param.device
        except StopIteration:
            self.device = torch.device("cpu")
            logger.warning(
                "Model has no parameters. Defaulting to CPU device for analysis."
            )

        logger.info(
            f"VocabSpecializationAnalyzer initialized: "
            f"num_experts={self.num_experts}, "
            f"num_layers={self.num_layers}, "
            f"vocab_size={self.vocab_size}, "
            f"top_k={self.top_k}, "
            f"device='{self.device}', "
            f"inference_batch_size={inference_batch_size}, "
            f"min_token_occurrences={min_token_occurrences}, "
            f"top_n_tokens_per_expert={top_n_tokens_per_expert}"
        )

    def compute_specialization(
        self,
        eval_data: Any,
        k: int = 1,
        mode: str = "input",
    ) -> Dict[str, Any]:
        """Compute vocabulary specialization scores for all layers and experts.

        Runs inference on eval_data, collects routing statistics, and computes
        specialization scores implementing Equation 8 from the paper:
            VocabSpecialization(E_i, x) = N_{x, E_i}^(k) / N_x

        Args:
            eval_data: Evaluation data. Can be a PyTorch Dataset, DataLoader,
                       or any iterable yielding dicts with "input_ids" key.
                       Should be the 0.5% C4 validation sample as specified in
                       config.yaml: analysis.eval_data_fraction: 0.005.
            k: Number of top experts to consider per token. Default: 1.
               Use k=1 for Figure 23 (main paper) and k=8 for Figure 30 (appendix).
               From config.yaml: analysis.vocab_specialization.top_k_values.
               Must be in [1, self.top_k].
            mode: Token mode for specialization analysis. Default: 'input'.
                  'input':  x is the current input token ID at each position.
                  'output': x is the next (ground-truth) token ID — captures
                            which expert processes context when predicting token x.
                  From config.yaml: analysis.vocab_specialization.modes.

        Returns:
            Dict with the following structure:
                {
                    'specialization': Tensor(num_layers, num_experts, vocab_size),
                        # Float32 specialization scores in [0, 1].
                        # spec[l, i, x] = fraction of times token x is routed to
                        # expert i at layer l (among all routings of token x).
                    'routing_counts': Tensor(num_layers, num_experts, vocab_size),
                        # Int64 raw routing counts.
                        # counts[l, i, x] = number of times token x was routed
                        # to expert i at layer l.
                    'per_layer_avg': Tensor(num_layers,),
                        # Float32 average specialization per layer.
                        # Averaged over experts and valid token IDs
                        # (those appearing >= min_token_occurrences times).
                    'top_tokens_per_expert': Dict[int, List[Tuple[str, float]]],
                        # Expert ID -> list of (token_string, specialization_pct)
                        # for the top top_n_tokens_per_expert tokens.
                        # Computed for TABLE8_LAYER_IDX (layer 7).
                    'k': int,   # The k value used
                    'mode': str, # The mode used ('input' or 'output')
                    'uniform_baseline': float,  # k / num_experts
                    'total_tokens': int,  # Total tokens processed
                }

        Raises:
            ValueError: If mode is not 'input' or 'output'.
            ValueError: If k is not in [1, self.top_k].
            RuntimeError: If no tokens were processed.
        """
        # Validate mode.
        if mode not in ("input", "output"):
            raise ValueError(
                f"mode must be 'input' or 'output', got '{mode}'. "
                f"'input' uses the current input token ID; "
                f"'output' uses the next (ground-truth) token ID."
            )

        # Validate k.
        if k <= 0 or k > self.top_k:
            raise ValueError(
                f"k must be in [1, top_k={self.top_k}], got k={k}. "
                f"Use k=1 for Figure 23 (main paper) or k=8 for Figure 30 (appendix)."
            )

        logger.info(
            f"Computing vocabulary specialization: "
            f"k={k}, mode='{mode}', "
            f"uniform_baseline={k / self.num_experts:.6f} ({k / self.num_experts * 100:.4f}%)"
        )

        # ------------------------------------------------------------------
        # Step 1: Collect raw routing counts via inference.
        # routing_counts shape: (num_layers, num_experts, vocab_size)
        # dtype: torch.long (int64) for exact integer arithmetic
        # ------------------------------------------------------------------
        use_output_token: bool = (mode == "output")
        routing_counts: Tensor = self._count_vocab_routing(
            data=eval_data,
            k=k,
            use_output_token=use_output_token,
        )
        # routing_counts: (num_layers, num_experts, vocab_size) on CPU

        # ------------------------------------------------------------------
        # Step 2: Compute total counts per token across all experts.
        # total_per_token[l, x] = sum over experts of counts[l, i, x]
        # Shape: (num_layers, vocab_size)
        # ------------------------------------------------------------------
        total_per_token: Tensor = routing_counts.sum(dim=1)
        # total_per_token: (num_layers, vocab_size)

        # ------------------------------------------------------------------
        # Step 3: Compute specialization scores.
        # spec[l, i, x] = counts[l, i, x] / total_per_token[l, x]
        # Handle division by zero: tokens that never appear get score 0.0.
        # ------------------------------------------------------------------
        # Expand total_per_token for broadcasting: (num_layers, 1, vocab_size)
        total_expanded: Tensor = total_per_token.unsqueeze(1)
        # total_expanded: (num_layers, 1, vocab_size)

        # Safe division: replace 0 denominators with 1 (numerator is also 0).
        safe_total: Tensor = total_expanded.clone().float()
        safe_total[safe_total == 0.0] = 1.0

        # Compute specialization as float32.
        specialization: Tensor = routing_counts.float() / safe_total
        # specialization: (num_layers, num_experts, vocab_size), float32

        # Zero out tokens that never appear (denominator was 0).
        zero_mask: Tensor = (total_expanded == 0).expand_as(specialization)
        specialization[zero_mask] = 0.0

        # Clamp to [0, 1] to handle any floating-point edge cases.
        specialization = specialization.clamp(0.0, 1.0)

        # ------------------------------------------------------------------
        # Step 4: Compute per-layer average specialization.
        # Average over experts and valid token IDs (those appearing >= min_token_occurrences).
        # valid_mask[l, x] = True if token x appears >= min_token_occurrences times at layer l.
        # ------------------------------------------------------------------
        # Use layer 0's total counts as the reference for valid tokens
        # (token occurrence counts are the same across layers since the same
        # input data is used for all layers).
        # Use the sum across all layers for a more robust valid mask.
        total_across_layers: Tensor = total_per_token.sum(dim=0)
        # total_across_layers: (vocab_size,)

        valid_token_mask: Tensor = total_across_layers >= self.min_token_occurrences
        # valid_token_mask: (vocab_size,), dtype=bool
        num_valid_tokens: int = int(valid_token_mask.sum().item())

        logger.info(
            f"Valid tokens (>= {self.min_token_occurrences} occurrences): "
            f"{num_valid_tokens:,} / {self.vocab_size:,} "
            f"({num_valid_tokens / self.vocab_size * 100:.1f}%)"
        )

        # Compute per-layer average specialization over valid tokens and all experts.
        per_layer_avg: Tensor = torch.zeros(self.num_layers, dtype=torch.float32)

        for layer_idx in range(self.num_layers):
            # Extract specialization for this layer: (num_experts, vocab_size)
            layer_spec: Tensor = specialization[layer_idx]

            if num_valid_tokens > 0:
                # Select only valid tokens: (num_experts, num_valid_tokens)
                valid_spec: Tensor = layer_spec[:, valid_token_mask]
                # Average over all experts and valid tokens.
                per_layer_avg[layer_idx] = valid_spec.mean().item()
            else:
                per_layer_avg[layer_idx] = 0.0

        # ------------------------------------------------------------------
        # Step 5: Get top tokens per expert for Table 8-style analysis.
        # Computed for TABLE8_LAYER_IDX (layer 7) as in the paper.
        # ------------------------------------------------------------------
        top_tokens_per_expert: Dict[int, List[Tuple[str, float]]] = (
            self.get_top_tokens_per_expert(
                routing_counts=routing_counts,
                top_n=self.top_n_tokens_per_expert,
                layer_idx=TABLE8_LAYER_IDX,
            )
        )

        # ------------------------------------------------------------------
        # Step 6: Compute total tokens processed.
        # Total tokens = sum of total_per_token across all layers and tokens,
        # divided by num_layers (since each token is counted once per layer).
        # ------------------------------------------------------------------
        total_tokens: int = int(total_per_token[0].sum().item())

        # ------------------------------------------------------------------
        # Step 7: Assemble and return results dict.
        # ------------------------------------------------------------------
        uniform_baseline: float = k / self.num_experts

        results: Dict[str, Any] = {
            "specialization": specialization,
            "routing_counts": routing_counts,
            "per_layer_avg": per_layer_avg,
            "top_tokens_per_expert": top_tokens_per_expert,
            "k": k,
            "mode": mode,
            "uniform_baseline": uniform_baseline,
            "total_tokens": total_tokens,
        }

        logger.info(
            f"Vocabulary specialization computation complete: "
            f"k={k}, mode='{mode}', "
            f"total_tokens={total_tokens:,}, "
            f"per_layer_avg_range=[{per_layer_avg.min().item():.4f}, "
            f"{per_layer_avg.max().item():.4f}], "
            f"uniform_baseline={uniform_baseline:.4f} ({uniform_baseline * 100:.2f}%)"
        )

        return results

    def _count_vocab_routing(
        self,
        data: Any,
        k: int,
        use_output_token: bool,
    ) -> Tensor:
        """Run inference and accumulate routing counts per (layer, expert, token_id).

        Processes all batches in the evaluation data and counts how often each
        expert is activated for each token ID at each layer. Returns a CPU tensor
        to avoid GPU memory accumulation.

        For input mode (use_output_token=False):
            token_ids[t] = input_ids[t] for all positions t in [0, seq_len-1]

        For output mode (use_output_token=True):
            token_ids[t] = input_ids[t+1] for positions t in [0, seq_len-2]
            (the routing at position t is associated with the next token)
            Only seq_len-1 positions contribute per sequence.

        Args:
            data: Evaluation data. Can be a PyTorch Dataset, DataLoader, or
                  any iterable yielding dicts with "input_ids" key.
            k: Number of top experts to count per token. Must be in [1, self.top_k].
               For k < self.top_k: selects the top-k experts by routing probability.
               For k == self.top_k: uses all activated experts.
            use_output_token: If True, use the next token ID as the vocabulary
                              element (output mode). If False, use the current
                              token ID (input mode).

        Returns:
            Tensor of shape (num_layers, num_experts, vocab_size), dtype=torch.long.
            Entry [l, i, x] = number of times token x was routed to expert i
            at layer l in the evaluation data.
            Always on CPU (moved from GPU after each batch).

        Raises:
            RuntimeError: If the model returns no routing information.
            ValueError: If no tokens were processed.
        """
        # ------------------------------------------------------------------
        # Initialize counts tensor on CPU.
        # Shape: (num_layers, num_experts, vocab_size) = (16, 64, 50304)
        # Memory: 16 * 64 * 50304 * 8 bytes (int64) ≈ 412 MB on CPU.
        # This is large but manageable; we keep it on CPU throughout.
        # ------------------------------------------------------------------
        counts: Tensor = torch.zeros(
            (self.num_layers, self.num_experts, self.vocab_size),
            dtype=torch.long,
            device="cpu",
        )

        # ------------------------------------------------------------------
        # Prepare data loader.
        # ------------------------------------------------------------------
        data_loader: Any = self._prepare_data_loader(data)

        # Set model to eval mode for inference.
        self.model.eval()

        total_tokens: int = 0
        total_batches: int = 0

        # Wrap with tqdm for progress tracking if available.
        data_iterator: Any = (
            tqdm(
                data_loader,
                desc=f"Counting vocab routing (k={k}, "
                     f"mode={'output' if use_output_token else 'input'})",
            )
            if TQDM_AVAILABLE
            else data_loader
        )

        try:
            with torch.no_grad():
                for batch in data_iterator:
                    # ----------------------------------------------------------
                    # Extract input_ids from batch.
                    # Handles dict, list/tuple, and raw tensor formats.
                    # ----------------------------------------------------------
                    input_ids: Tensor
                    if isinstance(batch, dict):
                        if "input_ids" not in batch:
                            logger.warning(
                                f"Batch {total_batches} missing 'input_ids' key. "
                                f"Got keys: {list(batch.keys())}. Skipping."
                            )
                            total_batches += 1
                            continue
                        input_ids = batch["input_ids"]
                    elif isinstance(batch, (list, tuple)):
                        input_ids = batch[0]
                    elif isinstance(batch, Tensor):
                        input_ids = batch
                    else:
                        logger.warning(
                            f"Batch {total_batches}: unexpected type {type(batch)}. "
                            f"Expected dict with 'input_ids'. Skipping."
                        )
                        total_batches += 1
                        continue

                    # Ensure input_ids is 2D: (batch_size, seq_len).
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    elif input_ids.dim() != 2:
                        logger.warning(
                            f"Batch {total_batches}: input_ids has unexpected "
                            f"shape {input_ids.shape}. Expected 2D. Skipping."
                        )
                        total_batches += 1
                        continue

                    batch_size: int = input_ids.shape[0]
                    seq_len: int = input_ids.shape[1]

                    # Move to model device for inference.
                    input_ids_device: Tensor = input_ids.to(self.device)

                    # ----------------------------------------------------------
                    # Forward pass — collect routing metadata.
                    # We only need top_k_indices and routing_weights from OLMoEOutput.
                    # Pass labels=None to skip loss computation.
                    # ----------------------------------------------------------
                    output: OLMoEOutput = self.model(
                        input_ids=input_ids_device,
                        attention_mask=None,
                        labels=None,
                    )

                    # Validate routing metadata is present.
                    if not output.top_k_indices:
                        raise RuntimeError(
                            "Model returned empty top_k_indices. "
                            "Ensure the model is an OLMoEModel with MoE layers."
                        )

                    if len(output.top_k_indices) != self.num_layers:
                        raise RuntimeError(
                            f"Expected {self.num_layers} layers of routing indices, "
                            f"got {len(output.top_k_indices)}."
                        )

                    # ----------------------------------------------------------
                    # Determine token IDs and valid positions based on mode.
                    #
                    # Input mode: token_ids = input_ids.reshape(-1)
                    #   All batch_size * seq_len positions contribute.
                    #
                    # Output mode: token_ids = input_ids[:, 1:].reshape(-1)
                    #   Only batch_size * (seq_len - 1) positions contribute.
                    #   The routing at position t predicts token t+1.
                    #   We use positions 0..seq_len-2 of the routing.
                    # ----------------------------------------------------------
                    if not use_output_token:
                        # Input mode: use current token IDs.