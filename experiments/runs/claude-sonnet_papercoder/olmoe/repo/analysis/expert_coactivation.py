## Code: analysis/expert_coactivation.py

```python
## analysis/expert_coactivation.py
"""Expert co-activation analysis for OLMoE-1B-7B.

Measures how often pairs of experts are simultaneously activated for the same
input token, as described in Section 5.2 of the OLMoE paper.

Expert co-activation is defined as:
    CoActivation(E_i, E_j) = N_{E_i, E_j} / N_{E_i}

where:
    N_{E_i, E_j} = number of tokens where both E_i and E_j are in the top-8
    N_{E_i}      = total number of tokens where E_i is in the top-8

Key properties:
    - Asymmetric: CoActivation(E_i, E_j) ≠ CoActivation(E_j, E_i) in general
    - Diagonal: CoActivation(E_i, E_i) = 1.0 always
    - Range: [0.0, 1.0]
    - Random baseline: 7/63 ≈ 11.1% (selecting 7 more from 63 remaining experts)

Key findings from the paper (Section 5.2, Figure 21):
    - Generally low co-activation with few exceptions
    - Layers 7 and 15 show similar patterns with groups of 2-3 co-activated experts
    - Experts 48 and 23 have ~60% co-activation (both specialize in connector words)
    - Low co-activation suggests little redundancy across experts

Configuration values used (from config.yaml):
    model.num_experts: 64
    model.top_k: 8
    model.num_layers: 16
    analysis.eval_data_fraction: 0.005
    analysis.eval_dataset: "c4"
    analysis.expert_coactivation.top_k: 8
    analysis.expert_coactivation.display_top_n_experts: 32
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from config import OLMoEConfig
from model.olmoe_model import OLMoEModel, OLMoEOutput
from utils.logging_utils import get_logger

logger: logging.Logger = get_logger("olmoe.analysis.expert_coactivation")

# ---------------------------------------------------------------------------
# Optional matplotlib/seaborn imports for visualization.
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
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
# Configuration constants from config.yaml (analysis.expert_coactivation section).
# ---------------------------------------------------------------------------
DEFAULT_TOP_K: int = 8
"""Use top-8 routing for co-activation analysis (config.yaml: analysis.expert_coactivation.top_k)."""

DEFAULT_DISPLAY_TOP_N_EXPERTS: int = 32
"""Display 32 experts with highest max co-activation (config.yaml: analysis.expert_coactivation.display_top_n_experts)."""

DEFAULT_INFERENCE_BATCH_SIZE: int = 8
"""Batch size for inference during analysis."""

DEFAULT_NUM_LAYERS: int = 16
"""Number of transformer layers (config.yaml: model.num_layers)."""

DEFAULT_NUM_EXPERTS: int = 64
"""Total experts per layer (config.yaml: model.num_experts)."""

DEFAULT_TOP_K_MODEL: int = 8
"""Activated experts per token (config.yaml: model.top_k)."""

# Random baseline: probability that E_j is also selected given E_i is selected.
# With 64 experts and top-8 selection: 7 remaining slots from 63 remaining experts.
RANDOM_COACTIVATION_BASELINE: float = 7.0 / 63.0
"""Random routing baseline ≈ 11.1% (7 remaining slots / 63 remaining experts)."""


class ExpertCoactivationAnalyzer:
    """Analyzes expert co-activation patterns in OLMoE-1B-7B.

    Computes the co-activation matrix for each transformer layer, showing
    how often pairs of experts are simultaneously activated for the same
    input token. Uses the final pretraining checkpoint and a random 0.5%
    sample of C4 validation data.

    The co-activation matrix for each layer is a (64, 64) float tensor where
    entry [i, j] represents CoActivation(E_i, E_j) = fraction of times E_j
    is activated given E_i is activated.

    Key findings to reproduce (Section 5.2, Figure 21):
        - Generally low co-activation (most entries near 11.1% random baseline)
        - Layers 7 and 15 show similar patterns with groups of 2-3 co-activated experts
        - Experts 48 and 23 have ~60% co-activation (connector words)
        - Sparse structure: mostly dark heatmap with few bright spots

    Attributes:
        model: The OLMoEModel loaded with the final pretraining checkpoint.
        num_experts: Total experts per layer = 64 (config.yaml: model.num_experts).
        top_k: Activated experts per token = 8 (config.yaml: model.top_k).
        num_layers: Number of transformer layers = 16 (config.yaml: model.num_layers).
        device: Device string inferred from model parameters.
        inference_batch_size: Batch size for inference during analysis.
        display_top_n_experts: Number of experts to display in plots = 32.

    Example:
        >>> config = OLMoEConfig()
        >>> model = OLMoEModel(config)
        >>> # Load final checkpoint weights into model before creating analyzer
        >>> analyzer = ExpertCoactivationAnalyzer(model, config)
        >>> coact_matrices = analyzer.compute_coactivation(eval_dataloader)
        >>> coact_matrices[7].shape  # Layer 7 co-activation matrix
        torch.Size([64, 64])
        >>> analyzer.plot_coactivation(coact_matrices, output_path="outputs/analysis")
    """

    def __init__(
        self,
        model: OLMoEModel,
        config: OLMoEConfig,
        inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
        display_top_n_experts: int = DEFAULT_DISPLAY_TOP_N_EXPERTS,
    ) -> None:
        """Initialize ExpertCoactivationAnalyzer.

        Args:
            model: The OLMoEModel loaded with the final pretraining checkpoint.
                   Must already have weights loaded and be on the correct device.
                   The analyzer calls model.eval() before inference and does not
                   modify model weights.
            config: OLMoEConfig instance. Key fields used:
                    - num_experts (64): size of co-activation matrices
                    - top_k (8): number of activated experts per token
                    - num_layers (16): number of layers to analyze
            inference_batch_size: Number of sequences per inference batch.
                                  Default: 8. Reduce if OOM errors occur during
                                  analysis of the C4 validation data.
            display_top_n_experts: Number of experts to display in heatmap plots.
                                   Default: 32 (config.yaml: analysis.expert_coactivation.display_top_n_experts).
                                   Experts are selected by highest maximum co-activation score.

        Raises:
            ValueError: If num_experts or top_k are invalid.
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
        if display_top_n_experts <= 0 or display_top_n_experts > config.num_experts:
            raise ValueError(
                f"display_top_n_experts must be in [1, num_experts], "
                f"got {display_top_n_experts}, num_experts={config.num_experts}."
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

        self.display_top_n_experts: int = display_top_n_experts
        """Number of experts to display in heatmap plots = 32."""

        # Infer device from model parameters.
        # Use the device of the first parameter as the canonical device.
        try:
            first_param: Tensor = next(model.parameters())
            self.device: str = str(first_param.device)
        except StopIteration:
            # Model has no parameters — fall back to CPU.
            self.device = "cpu"
            logger.warning(
                "Model has no parameters. Defaulting to CPU device for analysis."
            )

        logger.info(
            f"ExpertCoactivationAnalyzer initialized: "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"num_layers={self.num_layers}, "
            f"device='{self.device}', "
            f"inference_batch_size={inference_batch_size}, "
            f"display_top_n_experts={display_top_n_experts}, "
            f"random_baseline={RANDOM_COACTIVATION_BASELINE:.4f} ({RANDOM_COACTIVATION_BASELINE * 100:.1f}%)"
        )

    def compute_coactivation(
        self,
        eval_data: Any,
    ) -> Dict[int, Tensor]:
        """Compute expert co-activation matrices for all layers.

        Runs inference on the evaluation data (0.5% C4 validation sample) and
        builds a (64, 64) co-activation matrix for each of the 16 transformer
        layers. Entry [i, j] in the matrix for layer L represents:
            CoActivation(E_i, E_j) = N_{E_i, E_j} / N_{E_i}

        The computation uses vectorized one-hot encoding and matrix multiplication
        for efficiency:
            1. Convert top_k_indices to binary activation matrix A: (num_tokens, 64)
            2. expert_counts += A.sum(dim=0)  — shape (64,)
            3. coact_counts += A.T @ A        — shape (64, 64)
            4. Normalize: coact_matrix[i, j] = coact_counts[i, j] / expert_counts[i]

        Args:
            eval_data: Evaluation data iterable. Each item should be a dict with
                       "input_ids" key containing a tensor of shape (batch, seq_len),
                       or a DataLoader yielding such dicts. Should be the 0.5%
                       random sample of C4 validation data as specified in
                       config.yaml: analysis.eval_data_fraction: 0.005.

        Returns:
            Dict mapping layer_idx (0 to num_layers-1) to a (num_experts, num_experts)
            float32 tensor representing the normalized co-activation matrix.
            Entry [i, j] is CoActivation(E_i, E_j) in [0.0, 1.0].
            Diagonal entries are 1.0 (trivially, E_i always co-activates with itself).
            Rows for dead experts (never activated) are set to 0.0.

        Raises:
            RuntimeError: If the model returns no routing information.
            ValueError: If eval_data is empty.

        Example:
            >>> matrices = analyzer.compute_coactivation(c4_val_dataloader)
            >>> matrices[7].shape
            torch.Size([64, 64])
            >>> matrices[7].diagonal()  # All 1.0 (self-coactivation)
            tensor([1., 1., ..., 1.])
            >>> matrices[7].max()  # Maximum co-activation (excluding diagonal)
            tensor(0.6123)
        """
        # ------------------------------------------------------------------
        # Initialize count accumulators for all layers.
        # Using int64 on CPU to avoid overflow on large datasets.
        # The C4 0.5% sample may contain millions of tokens.
        # ------------------------------------------------------------------
        # coact_counts[layer]: (num_experts, num_experts) — raw co-activation counts
        coact_counts: List[Tensor] = [
            torch.zeros(
                (self.num_experts, self.num_experts),
                dtype=torch.int64,
                device="cpu",
            )
            for _ in range(self.num_layers)
        ]

        # expert_counts[layer]: (num_experts,) — individual activation counts
        expert_counts: List[Tensor] = [
            torch.zeros(
                (self.num_experts,),
                dtype=torch.int64,
                device="cpu",
            )
            for _ in range(self.num_layers)
        ]

        # ------------------------------------------------------------------
        # Set model to eval mode for inference.
        # Restores train mode in the finally block.
        # ------------------------------------------------------------------
        self.model.eval()

        total_tokens: int = 0
        total_batches: int = 0

        # Wrap with tqdm if available for progress tracking.
        data_iterator = (
            tqdm(eval_data, desc="Computing co-activation")
            if TQDM_AVAILABLE
            else eval_data
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
                            continue
                        input_ids = batch["input_ids"]
                    elif isinstance(batch, (list, tuple)):
                        input_ids = batch[0]
                    elif isinstance(batch, Tensor):
                        input_ids = batch
                    else:
                        logger.warning(
                            f"Unexpected batch type: {type(batch)}. "
                            f"Expected dict with 'input_ids'. Skipping batch {total_batches}."
                        )
                        continue

                    # Move to model device for inference.
                    input_ids = input_ids.to(self.device)
                    batch_size: int = input_ids.shape[0]
                    seq_len: int = input_ids.shape[1]
                    batch_tokens: int = batch_size * seq_len

                    # ----------------------------------------------------------
                    # Forward pass — collect routing metadata.
                    # We only need top_k_indices from OLMoEOutput.
                    # Pass labels=None to skip loss computation.
                    # ----------------------------------------------------------
                    output: OLMoEOutput = self.model(
                        input_ids=input_ids,
                        attention_mask=None,
                        labels=None,
                    )

                    # Validate routing metadata is present.
                    if not output.top_k_indices:
                        raise RuntimeError(
                            "Model returned empty top_k_indices. "
                            "Ensure the model is an OLMoEModel with MoE layers "
                            "and that routing metadata is collected in forward()."
                        )

                    if len(output.top_k_indices) != self.num_layers:
                        raise RuntimeError(
                            f"Expected {self.num_layers} layers of routing indices, "
                            f"got {len(output.top_k_indices)}. "
                            f"Check model architecture: config.num_layers={self.num_layers}."
                        )

                    # ----------------------------------------------------------
                    # Accumulate co-activation counts for each layer.
                    # ----------------------------------------------------------
                    for layer_idx in range(self.num_layers):
                        # top_k_indices[layer_idx]: (batch*seq, top_k) = (batch*seq, 8)
                        layer_top_k: Tensor = output.top_k_indices[layer_idx]

                        # Validate shape.
                        if layer_top_k.shape[0] != batch_tokens:
                            logger.warning(
                                f"Layer {layer_idx} top_k_indices has {layer_top_k.shape[0]} "
                                f"tokens, expected {batch_tokens}. Skipping layer."
                            )
                            continue

                        # Accumulate counts using the vectorized approach.
                        # Move to CPU for accumulation to avoid GPU memory pressure.
                        layer_top_k_cpu: Tensor = layer_top_k.cpu()

                        batch_coact, batch_expert = self._build_coactivation_matrix(
                            layer_idx=layer_idx,
                            top_k_indices=layer_top_k_cpu,
                        )

                        coact_counts[layer_idx] += batch_coact
                        expert_counts[layer_idx] += batch_expert

                    total_tokens += batch_tokens
                    total_batches += 1

                    # Free GPU memory from this batch's output.
                    del output
                    del input_ids
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        finally:
            # Always restore train mode.
            self.model.train()

        # ------------------------------------------------------------------
        # Validate that we processed some data.
        # ------------------------------------------------------------------
        if total_tokens == 0:
            raise ValueError(
                "No tokens were processed during co-activation computation. "
                "Ensure eval_data is non-empty and contains valid 'input_ids' batches."
            )

        logger.info(
            f"Co-activation counting complete: "
            f"total_tokens={total_tokens:,}, "
            f"total_batches={total_batches:,}"
        )

        # ------------------------------------------------------------------
        # Normalize counts to get co-activation probabilities.
        # coact_matrix[i, j] = coact_counts[i, j] / expert_counts[i]
        # ------------------------------------------------------------------
        coact_matrices: Dict[int, Tensor] = {}

        for layer_idx in range(self.num_layers):
            # Convert to float32 for normalization.
            coact_float: Tensor = coact_counts[layer_idx].float()
            expert_float: Tensor = expert_counts[layer_idx].float()

            # Identify dead experts (never activated).
            dead_mask: Tensor = expert_float == 0.0
            num_dead: int = int(dead_mask.sum().item())

            if num_dead > 0:
                logger.warning(
                    f"Layer {layer_idx}: {num_dead} dead experts (never activated). "
                    f"Their co-activation rows will be set to 0.0. "
                    f"This may indicate load balancing issues."
                )

            # Normalize: divide each row i by expert_counts[i].
            # expert_float has shape (num_experts,); unsqueeze for broadcasting.
            # Replace 0 denominators with 1 to avoid NaN (rows will be 0 anyway
            # since coact_counts[i, :] = 0 when expert_counts[i] = 0).
            safe_expert_counts: Tensor = expert_float.clone()
            safe_expert_counts[dead_mask] = 1.0  # Avoid division by zero

            # coact_float: (num_experts, num_experts)
            # safe_expert_counts.unsqueeze(1): (num_experts, 1) — broadcast over columns
            normalized: Tensor = coact_float / safe_expert_counts.unsqueeze(1)

            # Set dead expert rows to 0.0 explicitly.
            normalized[dead_mask] = 0.0

            # Clamp to [0, 1] to handle any floating-point edge cases.
            normalized = normalized.clamp(0.0, 1.0)

            coact_matrices[layer_idx] = normalized

            # Log summary statistics for this layer.
            # Exclude diagonal for meaningful off-diagonal statistics.
            off_diag_mask: Tensor = ~torch.eye(
                self.num_experts, dtype=torch.bool
            )
            off_diag_values: Tensor = normalized[off_diag_mask]
            max_coact: float = off_diag_values.max().item()
            mean_coact: float = off_diag_values.mean().item()

            logger.debug(
                f"Layer {layer_idx}: "
                f"max_off_diag_coact={max_coact:.4f} ({max_coact * 100:.1f}%), "
                f"mean_off_diag_coact={mean_coact:.4f} ({mean_coact * 100:.1f}%), "
                f"random_baseline={RANDOM_COACTIVATION_BASELINE:.4f} ({RANDOM_COACTIVATION_BASELINE * 100:.1f}%), "
                f"dead_experts={num_dead}"
            )

        logger.info(
            f"Co-activation matrices computed for {self.num_layers} layers. "
            f"Matrix shape: ({self.num_experts}, {self.num_experts}). "
            f"Random baseline: {RANDOM_COACTIVATION_BASELINE:.4f} ({RANDOM_COACTIVATION_BASELINE * 100:.1f}%)"
        )

        return coact_matrices

    def _build_coactivation_matrix(
        self,
        layer_idx: int,
        top_k_indices: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute raw co-activation counts for one batch and one layer.

        Uses vectorized one-hot encoding and matrix multiplication for efficiency:
            1. Build binary activation matrix A: (num_tokens, num_experts)
               where A[t, e] = 1 if expert e is in the top-k for token t
            2. expert_counts_batch = A.sum(dim=0)  — shape (num_experts,)
            3. coact_counts_batch = A.T @ A         — shape (num_experts, num_experts)

        The matrix product A.T @ A gives exactly N_{E_i, E_j} for all pairs
        simultaneously. Diagonal entries equal expert_counts (self-co-activation).

        Args:
            layer_idx: Layer index (0 to num_layers-1). Used only for logging.
            top_k_indices: Expert assignment indices for this batch and layer.
                           Shape: (num_tokens, top_k) = (batch*seq, 8).
                           dtype: torch.long (int64).
                           Values in [0, num_experts-1].
                           Must be on CPU for accumulation.

        Returns:
            Tuple of two tensors, both on CPU:
                - coact_counts_batch: (num_experts, num_experts) int64 tensor.
                  Entry [i, j] = number of tokens where both E_i and E_j are activated.
                  Diagonal [i, i] = number of tokens where E_i is activated.
                - expert_counts_batch: (num_experts,) int64 tensor.
                  Entry [i] = number of tokens where E_i is activated.
                  Equals the diagonal of coact_counts_batch.

        Raises:
            ValueError: If top_k_indices has unexpected shape or values out of range.
        """
        num_tokens: int = top_k_indices.shape[0]
        top_k_actual: int = top_k_indices.shape[1]

        # Validate shape.
        if top_k_actual != self.top_k:
            raise ValueError(
                f"Layer {layer_idx}: top_k_indices has {top_k_actual} experts per token, "
                f"expected {self.top_k} (config.yaml: model.top_k). "
                f"Check that the model uses the correct top_k configuration."
            )

        # Validate value range.
        if top_k_indices.min().item() < 0 or top_k_indices.max().item() >= self.num_experts:
            raise ValueError(
                f"Layer {layer_idx}: top_k_indices contains values outside "
                f"[0, {self.num_experts - 1}]. "
                f"Got min={top_k_indices.min().item()}, max={top_k_indices.max().item()}. "
                f"Check that the router outputs valid expert indices."
            )

        # ------------------------------------------------------------------
        # Step 1: Build binary activation matrix A.
        # A[t, e] = 1 if expert e is in the top-k for token t, else 0.
        # Shape: (num_tokens, num_experts) = (batch*seq, 64)
        # dtype: int64 for exact integer arithmetic in the matrix multiply.
        # ------------------------------------------------------------------
        A: Tensor = torch.zeros(
            (num_tokens, self.num_experts),
            dtype=torch.int64,
            device="cpu",
        )

        # scatter_ fills A[t, top_k_indices[t, k]] = 1 for all t, k.
        # This is equivalent to one-hot encoding with multiple active positions.
        # top_k_indices must be int64 for scatter_ index argument.
        A.scatter_(
            dim=1,
            index=top_k_indices.long(),
            value=1,
        )
        # A shape: (num_tokens, num_experts) = (batch*seq, 64)
        # Each row has exactly top_k=8 ones.

        # ------------------------------------------------------------------
        # Step 2: Compute expert activation counts.
        # expert_counts_batch[e] = number of tokens where expert e is activated.
        # Shape: (num_experts,) = (64,)
        # ------------------------------------------------------------------
        expert_counts_batch: Tensor = A.sum(dim=0)
        # expert_counts_batch shape: (num_experts,) = (64,)
        # Sum of all entries = num_tokens * top_k (each token activates top_k experts)

        # ------------------------------------------------------------------
        # Step 3: Compute co-activation counts via matrix multiplication.
        # coact_counts_batch[i, j] = number of tokens where both E_i and E_j activated.
        # = (A.T @ A)[i, j] = sum over tokens t of A[t, i] * A[t, j]
        # Shape: (num_experts, num_experts) = (64, 64)
        #
        # Note: A.T @ A is symmetric, but the normalized co-activation matrix
        # CoActivation(E_i, E_j) = coact_counts[i, j] / expert_counts[i] is
        # asymmetric because expert_counts[i] ≠ expert_counts[j] in general.
        #
        # Use float32 for the matrix multiply to avoid potential int64 overflow
        # on very large datasets, then convert back to int64.
        # With num_tokens up to ~100K and values 0/1, max value is ~100K which
        # fits in int64 (max ~9.2e18), but float32 is safer for large batches.
        # ------------------------------------------------------------------
        A_float: Tensor = A.float()
        coact_counts_batch: Tensor = (A_float.T @ A_float).long()
        # coact_counts_batch shape: (num_experts, num_experts) = (64, 64)
        # Diagonal: coact_counts_batch[i, i] = expert_counts_batch[i] (self-coactivation)
        # Off-diagonal: coact_counts_batch[i, j] = N_{E_i, E_j}

        return coact_counts_batch, expert_counts_batch

    def plot_coactivation(
        self,
        coact_matrices: Dict[int, Tensor],
        output_path: str,
    ) -> None:
        """Generate and save co-activation heatmap plots.

        Reproduces Figure 21 from the paper: heatmaps showing the 32 experts
        with highest maximum co-activation score for each layer. The paper
        highlights layers 7 and 15 as having similar patterns with groups of
        2-3 co-activated experts.

        For each layer:
            1. Exclude diagonal (self-co-activation = 1.0 trivially)
            2. For each expert i, compute max_coact[i] = max over j≠i of CoActivation(E_i, j)
            3. Select top-32 experts by max_coact (descending)
            4. Extract (32, 32) submatrix using those expert indices
            5. Generate heatmap with expert IDs as axis labels

        Saves:
            - Individual layer plots: {output_path}/coactivation_layer_{layer_idx}.png
            - Combined figure with all layers: {output_path}/coactivation_all_layers.png
            - Highlighted layers 7 and 15: {output_path}/coactivation_layers_7_15.png

        Args:
            coact_matrices: Dict mapping layer_idx to (num_experts, num_experts)
                            float32 co-activation matrix. Output of compute_coactivation().
            output_path: Directory path for saving plots. Created if it doesn't exist.
                         Example: "outputs/analysis"

        Returns:
            None. Saves plots to output_path directory.
            Logs a warning if matplotlib is not available.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning(
                "matplotlib not available. Cannot generate co-activation plots. "
                "Install with: pip install matplotlib seaborn"
            )
            return

        # Create output directory if it doesn't exist.
        output_dir: Path = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Generating co-activation heatmaps for {len(coact_matrices)} layers. "
            f"Output directory: '{output_dir}'"
        )

        # ------------------------------------------------------------------
        # Generate individual layer plots.
        # ------------------------------------------------------------------
        for layer_idx, coact_matrix in sorted(coact_matrices.items()):
            try:
                self._plot_single_layer(
                    layer_idx=layer_idx,
                    coact_