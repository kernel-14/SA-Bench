```python
## analysis/domain_specialization.py
"""Domain specialization analysis for OLMoE-1B-7B.

Measures how much individual experts specialize in processing tokens from
specific data domains, as described in Section 5.3 of the OLMoE paper.

Domain specialization is defined as:
    DomainSpecialization(E_i, D) = N_{E_i, D}^(k) / N_D

where:
    E_i = the i-th expert
    D   = a domain (e.g., arXiv, GitHub, C4)
    k   = number of top-k experts considered (k=8 for OLMoE, k=2 for Mixtral)
    N_{E_i, D}^(k) = number of tokens from domain D for which E_i is in top-k
    N_D = total number of tokens from domain D processed by the MoE

Uniform baseline: k / N_E = 8/64 = 12.5% for OLMoE, 2/8 = 25% for Mixtral.

Key findings from the paper (Section 5.3, Figure 22):
    - OLMoE layer 0, expert 0: nearly 100% specialized on arXiv
    - GitHub and arXiv often activate the same experts in layer 7
    - C4 (generic web) shows balanced activation across experts
    - Mixtral shows near-uniform routing across all domains and layers

Configuration values used (from config.yaml analysis.domain_specialization):
    analysis.domain_specialization.top_k_values: [1, 8]
    analysis.domain_specialization.domains: [c4, github, arxiv, wikipedia, books]
    analysis.domain_specialization.uniform_baseline_k8: 0.125
    analysis.domain_specialization.uniform_baseline_k1: 0.015625
    analysis.domain_specialization.comparison_model: "mistralai/Mixtral-8x7B-v0.1"
    model.num_experts: 64
    model.top_k: 8
    model.num_layers: 16
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from config import OLMoEConfig
from model.olmoe_model import OLMoEModel, OLMoEOutput
from utils.logging_utils import get_logger

logger: logging.Logger = get_logger("olmoe.analysis.domain_specialization")

# ---------------------------------------------------------------------------
# Optional matplotlib/seaborn imports for visualization.
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE: bool = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore[assignment]
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
# Optional HuggingFace transformers import for Mixtral comparison.
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE: bool = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]
    logger.warning(
        "HuggingFace 'transformers' library not available. "
        "Mixtral comparison will be disabled. Install with: pip install transformers"
    )

# ---------------------------------------------------------------------------
# Configuration constants from config.yaml (analysis.domain_specialization).
# ---------------------------------------------------------------------------
DEFAULT_TOP_K_VALUES: List[int] = [1, 8]
"""Analyze both k=1 and k=8 (config.yaml: analysis.domain_specialization.top_k_values)."""

DEFAULT_DOMAINS: List[str] = ["c4", "github", "arxiv", "wikipedia", "books"]
"""Domains to analyze (config.yaml: analysis.domain_specialization.domains)."""

DEFAULT_UNIFORM_BASELINE_K8: float = 0.125
"""Uniform baseline for k=8: 8/64 = 12.5% (config.yaml: analysis.domain_specialization.uniform_baseline_k8)."""

DEFAULT_UNIFORM_BASELINE_K1: float = 0.015625
"""Uniform baseline for k=1: 1/64 ≈ 1.56% (config.yaml: analysis.domain_specialization.uniform_baseline_k1)."""

DEFAULT_COMPARISON_MODEL: str = "mistralai/Mixtral-8x7B-v0.1"
"""Comparison model for domain specialization (config.yaml: analysis.domain_specialization.comparison_model)."""

DEFAULT_INFERENCE_BATCH_SIZE: int = 4
"""Batch size for inference during analysis (small to avoid OOM with 4096-token sequences)."""

DEFAULT_NUM_LAYERS: int = 16
"""Number of transformer layers for OLMoE-1B-7B (config.yaml: model.num_layers)."""

DEFAULT_NUM_EXPERTS: int = 64
"""Total experts per layer for OLMoE-1B-7B (config.yaml: model.num_experts)."""

DEFAULT_TOP_K: int = 8
"""Activated experts per token for OLMoE-1B-7B (config.yaml: model.top_k)."""

# Mixtral architecture constants (from HuggingFace model card).
MIXTRAL_NUM_EXPERTS: int = 8
"""Total experts per layer for Mixtral-8x7B."""

MIXTRAL_TOP_K: int = 2
"""Activated experts per token for Mixtral-8x7B."""

MIXTRAL_NUM_LAYERS: int = 32
"""Number of transformer layers for Mixtral-8x7B."""

# Domain display names for plot labels.
DOMAIN_DISPLAY_NAMES: Dict[str, str] = {
    "c4": "C4 (Web)",
    "github": "GitHub (Code)",
    "arxiv": "arXiv (STEM)",
    "wikipedia": "Wikipedia",
    "books": "Books",
    "dclm": "DCLM (Web)",
    "starcoder": "StarCoder (Code)",
    "pes2o": "peS2o (STEM)",
    "openwebmath": "OpenWebMath",
}

# Color palette for domain plots (one color per domain).
DOMAIN_COLORS: List[str] = [
    "#1f77b4",  # blue — C4/web
    "#ff7f0e",  # orange — GitHub/code
    "#2ca02c",  # green — arXiv/STEM
    "#d62728",  # red — Wikipedia
    "#9467bd",  # purple — Books
    "#8c564b",  # brown — additional domains
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]


class DomainSpecializationAnalyzer:
    """Analyzes domain specialization of experts in OLMoE-1B-7B.

    Measures how much individual experts specialize in processing tokens from
    specific data domains (C4/web, GitHub/code, arXiv/STEM, Wikipedia, books).
    Compares OLMoE's strong domain specialization against Mixtral's near-uniform
    routing (attributed to Mixtral being upcycled from a dense model).

    The analysis runs inference on domain-specific datasets and counts how often
    each expert appears in the top-k routing decisions for each domain. Results
    are normalized by total domain tokens to get routing fractions.

    Attributes:
        model: The OLMoEModel loaded with the final pretraining checkpoint.
        num_experts: Total experts per layer = 64 (config.yaml: model.num_experts).
        top_k: Activated experts per token = 8 (config.yaml: model.top_k).
        num_layers: Number of transformer layers = 16 (config.yaml: model.num_layers).
        device: Device string inferred from model parameters.
        inference_batch_size: Batch size for inference during analysis.

    Example:
        >>> config = OLMoEConfig()
        >>> model = OLMoEModel(config)
        >>> # Load final checkpoint weights into model before creating analyzer
        >>> analyzer = DomainSpecializationAnalyzer(model, config)
        >>> domain_datasets = {
        ...     "arxiv": arxiv_dataset,
        ...     "github": github_dataset,
        ...     "c4": c4_dataset,
        ... }
        >>> results = analyzer.compute_specialization(domain_datasets, k=8)
        >>> analyzer.plot_specialization(results, output_path="outputs/analysis")
    """

    def __init__(
        self,
        model: OLMoEModel,
        config: OLMoEConfig,
        inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    ) -> None:
        """Initialize DomainSpecializationAnalyzer.

        Args:
            model: The OLMoEModel loaded with the final pretraining checkpoint.
                   Must already have weights loaded and be on the correct device.
                   The analyzer calls model.eval() before inference and does not
                   modify model weights.
            config: OLMoEConfig instance. Key fields used:
                    - num_experts (64): number of experts per layer
                    - top_k (8): activated experts per token
                    - num_layers (16): number of transformer layers
            inference_batch_size: Number of sequences per inference batch.
                                  Default: 4. Reduce if OOM errors occur during
                                  analysis of 4096-token sequences.

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
        if inference_batch_size <= 0:
            raise ValueError(
                f"inference_batch_size must be positive, got {inference_batch_size}."
            )

        self.model: OLMoEModel = model
        """The OLMoEModel loaded with the final pretraining checkpoint."""

        self.num_experts: int = config.num_experts
        """Total experts per layer = 64 (config.yaml: model.num_experts)."""

        self.top_k: int = config.top_k
        """Activated experts per token = 8 (config.yaml: model.top_k)."""

        self.num_layers: int = config.num_layers
        """Number of transformer layers = 16 (config.yaml: model.num_layers)."""

        self.inference_batch_size: int = inference_batch_size
        """Batch size for inference during analysis."""

        # Infer device from model parameters.
        try:
            first_param: Tensor = next(model.parameters())
            self.device: str = str(first_param.device)
        except StopIteration:
            self.device = "cpu"
            logger.warning(
                "Model has no parameters. Defaulting to CPU device for analysis."
            )

        logger.info(
            f"DomainSpecializationAnalyzer initialized: "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"num_layers={self.num_layers}, "
            f"device='{self.device}', "
            f"inference_batch_size={inference_batch_size}, "
            f"uniform_baseline_k8={DEFAULT_UNIFORM_BASELINE_K8:.4f} ({DEFAULT_UNIFORM_BASELINE_K8 * 100:.1f}%), "
            f"uniform_baseline_k1={DEFAULT_UNIFORM_BASELINE_K1:.6f} ({DEFAULT_UNIFORM_BASELINE_K1 * 100:.4f}%)"
        )

    def compute_specialization(
        self,
        domain_datasets: Dict[str, Any],
        k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """Compute domain specialization for all provided domains.

        Runs inference on each domain's dataset and computes the fraction of
        tokens from that domain that activate each expert at each layer.
        Implements Equation 7 from the paper:
            DomainSpecialization(E_i, D) = N_{E_i, D}^(k) / N_D

        Args:
            domain_datasets: Dict mapping domain name strings to datasets.
                             Each dataset should be a PyTorch Dataset or DataLoader
                             yielding dicts with "input_ids" key, or any iterable
                             of such dicts. Domain names should match the domains
                             listed in config.yaml: analysis.domain_specialization.domains
                             (e.g., "c4", "github", "arxiv", "wikipedia", "books").
            k: Number of top experts to consider per token. Default: 8.
               Use k=8 for Figure 22 (main paper) and k=1 for Figure 34 (appendix).
               From config.yaml: analysis.domain_specialization.top_k_values.

        Returns:
            Dict with the following structure:
                {
                    "routing_fractions": {
                        "arxiv": Tensor(num_layers, num_experts),  # fractions in [0, 1]
                        "github": Tensor(num_layers, num_experts),
                        "c4": Tensor(num_layers, num_experts),
                        ...
                    },
                    "uniform_baseline": float,  # k / num_experts
                    "k": int,                   # the k value used
                    "num_experts": int,         # total experts per layer
                    "num_layers": int,          # number of layers
                    "domain_token_counts": {    # total tokens per domain
                        "arxiv": int,
                        ...
                    },
                }

        Raises:
            ValueError: If domain_datasets is empty.
            RuntimeError: If the model returns no routing information.
        """
        if not domain_datasets:
            raise ValueError(
                "domain_datasets must be non-empty. "
                "Provide at least one domain dataset for analysis."
            )

        if k <= 0 or k > self.num_experts:
            raise ValueError(
                f"k must be in [1, num_experts={self.num_experts}], got k={k}."
            )

        logger.info(
            f"Computing domain specialization: "
            f"{len(domain_datasets)} domains, k={k}, "
            f"uniform_baseline={k / self.num_experts:.4f} ({k / self.num_experts * 100:.1f}%)"
        )

        # Set model to eval mode for inference.
        self.model.eval()

        routing_fractions: Dict[str, Tensor] = {}
        domain_token_counts: Dict[str, int] = {}

        try:
            with torch.no_grad():
                for domain_name, domain_data in domain_datasets.items():
                    logger.info(
                        f"Processing domain: '{domain_name}' "
                        f"(k={k}, device='{self.device}')"
                    )

                    try:
                        fractions, total_tokens = self._count_domain_routing(
                            domain=domain_name,
                            data=domain_data,
                            k=k,
                        )
                        routing_fractions[domain_name] = fractions
                        domain_token_counts[domain_name] = total_tokens

                        # Log summary statistics for this domain.
                        # Find the most specialized expert (highest fraction).
                        max_fraction: float = fractions.max().item()
                        max_layer: int = int(fractions.max(dim=1).values.argmax().item())
                        max_expert: int = int(fractions[max_layer].argmax().item())
                        uniform_baseline: float = k / self.num_experts

                        logger.info(
                            f"Domain '{domain_name}': "
                            f"total_tokens={total_tokens:,}, "
                            f"max_specialization={max_fraction:.4f} ({max_fraction * 100:.1f}%) "
                            f"at layer={max_layer}, expert={max_expert}, "
                            f"uniform_baseline={uniform_baseline:.4f} ({uniform_baseline * 100:.1f}%)"
                        )

                    except Exception as exc:
                        logger.warning(
                            f"Failed to compute specialization for domain '{domain_name}': "
                            f"{type(exc).__name__}: {exc}. Skipping this domain."
                        )

        finally:
            # Always restore train mode.
            self.model.train()

        if not routing_fractions:
            raise RuntimeError(
                "No domain specialization results were computed. "
                "All domains failed during inference. Check dataset format and model."
            )

        # Compute uniform baseline: k / num_experts.
        uniform_baseline_value: float = k / self.num_experts

        results: Dict[str, Any] = {
            "routing_fractions": routing_fractions,
            "uniform_baseline": uniform_baseline_value,
            "k": k,
            "num_experts": self.num_experts,
            "num_layers": self.num_layers,
            "domain_token_counts": domain_token_counts,
        }

        logger.info(
            f"Domain specialization computation complete: "
            f"{len(routing_fractions)} domains processed, "
            f"k={k}, "
            f"uniform_baseline={uniform_baseline_value:.4f} ({uniform_baseline_value * 100:.1f}%)"
        )

        return results

    def _count_domain_routing(
        self,
        domain: str,
        data: Any,
        k: int = DEFAULT_TOP_K,
    ) -> Tuple[Tensor, int]:
        """Count expert routing fractions for a single domain.

        Runs inference on the domain's dataset and counts how often each expert
        appears in the top-k routing decisions at each layer. Normalizes by
        total token count to get routing fractions.

        Uses vectorized bincount for efficiency — avoids Python loops over
        num_experts=64 by counting all expert occurrences simultaneously.

        Args:
            domain: Domain name string (for logging). E.g., "arxiv", "github".
            data: Domain dataset. Can be:
                  - A PyTorch DataLoader yielding dicts with "input_ids" key
                  - A PyTorch Dataset (will be wrapped in a DataLoader)
                  - Any iterable yielding dicts with "input_ids" key
                  - Any iterable yielding (input_ids, ...) tuples
                  input_ids should have shape (batch_size, seq_len).
            k: Number of top experts to count per token. Default: 8.
               For k=8: counts all 8 activated experts per token.
               For k=1: counts only the highest-probability expert per token.

        Returns:
            Tuple of:
                - routing_fractions: Tensor of shape (num_layers, num_experts),
                  dtype=float32. Entry [l, e] = fraction of domain tokens for
                  which expert e was in the top-k at layer l.
                  Values in [0, 1]. Sum over experts ≈ k (not exactly due to
                  the fraction normalization).
                - total_tokens: Total number of tokens processed from this domain.

        Raises:
            RuntimeError: If the model returns no routing information.
            ValueError: If no tokens were processed (empty dataset).
        """
        # Initialize accumulation tensors on CPU.
        # Use float32 for accumulation to avoid int64 overflow on large datasets.
        # With 100K tokens and 64 experts, max count ≈ 100K * 8 = 800K, well within float32.
        routing_counts: Tensor = torch.zeros(
            (self.num_layers, self.num_experts),
            dtype=torch.float32,
            device="cpu",
        )
        total_tokens: int = 0

        # Wrap Dataset in DataLoader if needed.
        data_loader: Any = self._prepare_data_loader(data)

        # Wrap with tqdm for progress tracking if available.
        data_iterator: Any = (
            tqdm(data_loader, desc=f"  Domain '{domain}' (k={k})", leave=False)
            if TQDM_AVAILABLE
            else data_loader
        )

        batch_count: int = 0

        for batch in data_iterator:
            # ------------------------------------------------------------------
            # Extract input_ids from batch.
            # Handles dict, list/tuple, and raw tensor formats.
            # ------------------------------------------------------------------
            input_ids: Tensor
            if isinstance(batch, dict):
                if "input_ids" not in batch:
                    logger.warning(
                        f"Domain '{domain}', batch {batch_count}: "
                        f"missing 'input_ids' key. Got keys: {list(batch.keys())}. "
                        f"Skipping batch."
                    )
                    batch_count += 1
                    continue
                input_ids = batch["input_ids"]
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0]
            elif isinstance(batch, Tensor):
                input_ids = batch
            else:
                logger.warning(
                    f"Domain '{domain}', batch {batch_count}: "
                    f"unexpected batch type {type(batch)}. Skipping."
                )
                batch_count += 1
                continue

            # Ensure input_ids is 2D: (batch_size, seq_len).
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            elif input_ids.dim() != 2:
                logger.warning(
                    f"Domain '{domain}', batch {batch_count}: "
                    f"input_ids has unexpected shape {input_ids.shape}. "
                    f"Expected 2D (batch_size, seq_len). Skipping."
                )
                batch_count += 1
                continue

            batch_size: int = input_ids.shape[0]
            seq_len: int = input_ids.shape[1]
            batch_tokens: int = batch_size * seq_len

            # Move to model device for inference.
            input_ids = input_ids.to(self.device)

            # ------------------------------------------------------------------
            # Forward pass — collect routing metadata.
            # We only need top_k_indices from OLMoEOutput.
            # Pass labels=None to skip loss computation.
            # ------------------------------------------------------------------
            output: OLMoEOutput = self.model(
                input_ids=input_ids,
                attention_mask=None,
                labels=None,
            )

            # Validate routing metadata is present.
            if not output.top_k_indices:
                raise RuntimeError(
                    f"Domain '{domain}': model returned empty top_k_indices. "
                    f"Ensure the model is an OLMoEModel with MoE layers and "
                    f"that routing metadata is collected in forward()."
                )

            if len(output.top_k_indices) != self.num_layers:
                raise RuntimeError(
                    f"Domain '{domain}': expected {self.num_layers} layers of "
                    f"routing indices, got {len(output.top_k_indices)}. "
                    f"Check model architecture: config.num_layers={self.num_layers}."
                )

            # ------------------------------------------------------------------
            # Accumulate routing counts for each layer using vectorized bincount.
            #
            # For each layer l:
            #   top_k_indices[l]: (batch*seq, model.top_k=8)
            #   If k < model.top_k: we need to select only the top-k experts.
            #   If k == model.top_k: use all activated experts.
            #
            # For k=1: use only the highest-probability expert per token.
            #   This requires routing_weights to find the true top-1.
            # For k=8: use all 8 activated experts.
            # ------------------------------------------------------------------
            for layer_idx in range(self.num_layers):
                # top_k_indices[layer_idx]: (batch*seq, model.top_k)
                layer_top_k: Tensor = output.top_k_indices[layer_idx]

                # Validate shape.
                if layer_top_k.shape[0] != batch_tokens:
                    logger.warning(
                        f"Domain '{domain}', layer {layer_idx}: "
                        f"top_k_indices has {layer_top_k.shape[0]} tokens, "
                        f"expected {batch_tokens}. Skipping layer."
                    )
                    continue

                # Select the appropriate top-k experts.
                if k == self.top_k:
                    # Use all activated experts — most common case (k=8).
                    selected_indices: Tensor = layer_top_k.cpu()
                    # selected_indices: (batch*seq, k)
                elif k == 1:
                    # Use only the highest-probability expert per token.
                    # Find the true top-1 using routing weights.
                    if output.routing_weights and len(output.routing_weights) > layer_idx:
                        routing_weights: Tensor = output.routing_weights[layer_idx]
                        # routing_weights: (batch*seq, num_experts) — full softmax probs
                        # Gather weights for the selected top-k experts.
                        # layer_top_k: (batch*seq, model.top_k)
                        selected_weights: Tensor = routing_weights.gather(
                            dim=1,
                            index=layer_top_k.long(),
                        )
                        # Find which of the top-k has the highest weight.
                        # top1_within_topk: (batch*seq,)
                        top1_within_topk: Tensor = selected_weights.argmax(dim=1)
                        # Get the actual expert ID for the top-1.
                        # top1_expert: (batch*seq, 1)
                        top1_expert: Tensor = layer_top_k.gather(
                            dim=1,
                            index=top1_within_topk.unsqueeze(1),
                        )
                        selected_indices = top1_expert.cpu()
                        # selected_indices: (batch*seq, 1)
                    else:
                        # Fallback: use first column (assumes sorted by routing prob).
                        selected_indices = layer_top_k[:, :1].cpu()
                        # selected_indices: (batch*seq, 1)
                elif k < self.top_k:
                    # Use the top-k experts by routing probability.
                    # This handles cases like k=2 for Mixtral comparison.
                    if output.routing_weights and len(output.routing_weights) > layer_idx:
                        routing_weights = output.routing_weights[layer_idx]
                        selected_weights = routing_weights.gather(
                            dim=1,
                            index=layer_top_k.long(),
                        )
                        # Get indices of top-k within the activated experts.
                        _, topk_within: Tensor = torch.topk(
                            selected_weights, k=k, dim=1, largest=True, sorted=False
                        )
                        # Gather the actual expert IDs.
                        selected_indices = layer_top_k.gather(
                            dim=1,
                            index=topk_within,
                        ).cpu()
                        # selected_indices: (batch*seq, k)
                    else:
                        # Fallback: use first k columns.
                        selected_indices = layer_top_k[:, :k].cpu()
                else:
                    # k > model.top_k: not supported, use all activated experts.
                    logger.warning(
                        f"Requested k={k} > model.top_k={self.top_k}. "
                        f"Using all {self.top_k} activated experts."
                    )
                    selected_indices = layer_top_k.cpu()

                # ------------------------------------------------------------------
                # Vectorized count using bincount.
                #
                # Flatten selected_indices to 1D and count occurrences of each
                # expert ID. This gives the number of (token, expert_slot) pairs
                # involving each expert, which equals the number of tokens for
                # which that expert was in the top-k (since each token activates
                # each expert at most once in top-k).
                #
                # flat_indices: (batch*seq * k_actual,)
                # counts: (num_experts,) — count of tokens activating each expert
                # ------------------------------------------------------------------
                flat_indices: Tensor = selected_indices.reshape(-1).long()

                # Validate index range before bincount.
                if flat_indices.numel() > 0:
                    min_idx: int = int(flat_indices.min().item())
                    max_idx: int = int(flat_indices.max().item())
                    if min_idx < 0 or max_idx >= self.num_experts:
                        logger.warning(
                            f"Domain '{domain}', layer {layer_idx}: "
                            f"expert indices out of range [{min_idx}, {max_idx}], "
                            f"expected [0, {self.num_experts - 1}]. "
                            f"Clamping to valid range."
                        )
                        flat_indices = flat_indices.clamp(0, self.num_experts - 1)

                    counts: Tensor = torch.bincount(
                        flat_indices,
                        minlength=self.num_experts,
                    ).float()
                    # counts: (num_experts,)

                    routing_counts[layer_idx] += counts

            total_tokens += batch_tokens
            batch_count += 1

            # Free GPU memory from this batch's output.
            del output
            del input_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Validate that we processed some data.
        # ------------------------------------------------------------------
        if total_tokens == 0:
            raise ValueError(
                f"No tokens were processed for domain '{domain}'. "
                f"Ensure the dataset is non-empty and contains valid 'input_ids' batches."
            )

        # ------------------------------------------------------------------
        # Normalize by total token count to get routing fractions.
        # routing_fractions[l, e] = N_{E_e, D}^(k) / N_D
        # ------------------------------------------------------------------
        routing_fractions: Tensor = routing_counts / float(total_tokens)
        # routing_fractions: (num_layers, num_experts), values in [0, 1]

        # Clamp to [0, 1] to handle any floating-point edge cases.
        routing_fractions = routing_fractions.clamp(0.0, 1.0)

        logger.debug(
            f"Domain '{domain}': "
            f"total_tokens={total_tokens:,}, "
            f"total_batches={batch_count}, "
            f"max_fraction={routing_fractions.max().item():.4f}, "
            f"mean_