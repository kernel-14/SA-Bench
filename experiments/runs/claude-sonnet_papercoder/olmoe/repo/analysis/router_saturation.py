```python
## analysis/router_saturation.py
"""Router saturation analysis for OLMoE-1B-7B.

Measures how early in pretraining the routing decisions stabilize by comparing
expert assignments at intermediate checkpoints to the final checkpoint.

Router saturation is defined in Section 5.1 of the paper:
    RouterSaturation(t) = (1/N) * Σ_{i=1}^{N} |E_i^(t) ∩ E_i^(T)| / k

where:
    N = total number of tokens in the evaluation dataset
    k = number of top-k experts activated per token (k=8 or k=1)
    E_i^(t) = set of k experts activated for token i at checkpoint t
    E_i^(T) = set of k experts activated for token i at the final checkpoint T

Key findings from the paper (Section 5.1, Figure 20):
    - At 1% of pretraining (~20B tokens): ~60% saturation for k=8
    - Layer 0 saturates significantly more slowly than other layers
    - Later layers saturate earlier during pretraining
    - At 40% of pretraining: ~80% saturation for k=8

Configuration values used (from config.yaml analysis.router_saturation):
    analysis.eval_data_fraction: 0.005       # 0.5% of C4 validation
    analysis.eval_dataset: "c4"
    analysis.eval_split: "validation"
    analysis.router_saturation.top_k_values: [1, 8]
    analysis.router_saturation.checkpoint_fractions: [0.01, 0.10, 0.20, 0.40, 1.0]
    analysis.router_saturation.random_baseline_k1: 0.015625  # 1/64
    analysis.router_saturation.random_baseline_k8: 0.125     # 8/64
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from config import OLMoEConfig
from model.olmoe_model import OLMoEModel, OLMoEOutput
from utils.checkpoint import CheckpointManager
from utils.logging_utils import get_logger

logger: logging.Logger = get_logger("olmoe.analysis.router_saturation")

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
# Configuration constants from config.yaml (analysis.router_saturation section).
# ---------------------------------------------------------------------------
DEFAULT_EVAL_DATA_FRACTION: float = 0.005
"""Random 0.5% of C4 validation data (config.yaml: analysis.eval_data_fraction)."""

DEFAULT_TOP_K_VALUES: List[int] = [1, 8]
"""Analyze both k=1 and k=8 (config.yaml: analysis.router_saturation.top_k_values)."""

DEFAULT_CHECKPOINT_FRACTIONS: List[float] = [0.01, 0.10, 0.20, 0.40, 1.0]
"""Checkpoint fractions to analyze (config.yaml: analysis.router_saturation.checkpoint_fractions)."""

DEFAULT_RANDOM_BASELINE_K1: float = 0.015625
"""Random baseline for k=1: 1/64 = 1.5625% (config.yaml: analysis.router_saturation.random_baseline_k1)."""

DEFAULT_RANDOM_BASELINE_K8: float = 0.125
"""Random baseline for k=8: 8/64 = 12.5% (config.yaml: analysis.router_saturation.random_baseline_k8)."""

DEFAULT_INFERENCE_BATCH_SIZE: int = 8
"""Batch size for inference during analysis (smaller than training to fit in memory)."""

DEFAULT_NUM_LAYERS: int = 16
"""Number of transformer layers in OLMoE-1B-7B (config.yaml: model.num_layers)."""

DEFAULT_NUM_EXPERTS: int = 64
"""Total experts per layer (config.yaml: model.num_experts)."""

DEFAULT_TOP_K: int = 8
"""Activated experts per token (config.yaml: model.top_k)."""


class RouterSaturationAnalyzer:
    """Analyzes router saturation during OLMoE pretraining.

    Measures how early in pretraining the routing decisions stabilize by
    comparing expert assignments at intermediate checkpoints to the final
    checkpoint over the same evaluation data.

    The analysis uses a random 0.5% sample of C4 validation data and compares
    routing at checkpoints corresponding to 1%, 10%, 20%, and 40% of pretraining
    against the final checkpoint (100%).

    Key findings to reproduce (Section 5.1, Figure 20):
        - ~60% saturation at 1% of pretraining for k=8
        - Layer 0 saturates significantly more slowly than other layers
        - Later layers saturate earlier during pretraining
        - ~80% saturation at 40% of pretraining for k=8

    Attributes:
        model: The OLMoEModel instance (used for architecture reference).
        tokenizer: GPT-NeoX tokenizer (vocab_size=50304).
        device: Device string for tensor operations (e.g., "cuda:0").
        num_layers: Number of transformer layers (16 for OLMoE-1B-7B).
        num_experts: Total experts per layer (64 for OLMoE-1B-7B).
        top_k: Activated experts per token (8 for OLMoE-1B-7B).
        eval_data_fraction: Fraction of C4 validation to use (0.005 = 0.5%).
        checkpoint_fractions: Fractions of pretraining to analyze.
        random_baseline_k1: Random routing baseline for k=1 (1/64 ≈ 1.56%).
        random_baseline_k8: Random routing baseline for k=8 (8/64 = 12.5%).
        inference_batch_size: Batch size for inference during analysis.

    Example:
        >>> analyzer = RouterSaturationAnalyzer(model, tokenizer, device="cuda:0")
        >>> results = analyzer.compute_saturation(
        ...     checkpoint_paths=["outputs/checkpoint-00012500",
        ...                       "outputs/checkpoint-00125000",
        ...                       "outputs/checkpoint-00250000",
        ...                       "outputs/checkpoint-00500000"],
        ...     eval_data=c4_val_dataloader,
        ...     final_checkpoint="outputs/checkpoint-01223958",
        ... )
        >>> analyzer.plot_saturation(results, output_path="outputs/router_saturation.png")
    """

    def __init__(
        self,
        model: OLMoEModel,
        tokenizer: Any,
        device: str = "cuda",
        eval_data_fraction: float = DEFAULT_EVAL_DATA_FRACTION,
        checkpoint_fractions: Optional[List[float]] = None,
        random_baseline_k1: float = DEFAULT_RANDOM_BASELINE_K1,
        random_baseline_k8: float = DEFAULT_RANDOM_BASELINE_K8,
        inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    ) -> None:
        """Initialize RouterSaturationAnalyzer.

        Args:
            model: The OLMoEModel instance. Used to read architecture config
                   (num_layers, num_experts, top_k) and as a template for
                   loading checkpoint weights. The model itself is not modified.
            tokenizer: GPT-NeoX tokenizer (EleutherAI/gpt-neox-20b).
                       Used for reference; not directly used in saturation computation.
            device: Device string for tensor operations. Default: "cuda".
                    Should match the device the model is on.
                    Examples: "cuda", "cuda:0", "cpu".
            eval_data_fraction: Fraction of C4 validation data to use for analysis.
                                Default: 0.005 (0.5%) from config.yaml.
            checkpoint_fractions: List of training fractions to analyze.
                                  Default: [0.01, 0.10, 0.20, 0.40, 1.0] from config.yaml.
                                  The last value (1.0) is the final checkpoint.
            random_baseline_k1: Random routing baseline for k=1 analysis.
                                 Default: 1/64 = 0.015625 from config.yaml.
            random_baseline_k8: Random routing baseline for k=8 analysis.
                                 Default: 8/64 = 0.125 from config.yaml.
            inference_batch_size: Number of sequences per inference batch.
                                  Default: 8. Reduce if OOM errors occur.
        """
        self.model: OLMoEModel = model
        self.tokenizer: Any = tokenizer
        self.device: str = device

        # Read architecture constants from model config.
        # These are used throughout the analysis for validation and computation.
        self.num_layers: int = model.config.num_layers
        """Number of transformer layers = 16 (config.yaml: model.num_layers)."""

        self.num_experts: int = model.config.num_experts
        """Total experts per layer = 64 (config.yaml: model.num_experts)."""

        self.top_k: int = model.config.top_k
        """Activated experts per token = 8 (config.yaml: model.top_k)."""

        # Analysis configuration from config.yaml (analysis.router_saturation section).
        self.eval_data_fraction: float = eval_data_fraction
        """Fraction of C4 validation to use = 0.005 (config.yaml: analysis.eval_data_fraction)."""

        self.checkpoint_fractions: List[float] = (
            checkpoint_fractions
            if checkpoint_fractions is not None
            else DEFAULT_CHECKPOINT_FRACTIONS
        )
        """Checkpoint fractions to analyze (config.yaml: analysis.router_saturation.checkpoint_fractions)."""

        self.random_baseline_k1: float = random_baseline_k1
        """Random baseline for k=1: 1/64 ≈ 1.56% (config.yaml: analysis.router_saturation.random_baseline_k1)."""

        self.random_baseline_k8: float = random_baseline_k8
        """Random baseline for k=8: 8/64 = 12.5% (config.yaml: analysis.router_saturation.random_baseline_k8)."""

        self.inference_batch_size: int = inference_batch_size
        """Batch size for inference during analysis."""

        # Checkpoint manager for loading model weights.
        # Uses a temporary output directory (not used for saving here).
        self._checkpoint_manager: CheckpointManager = CheckpointManager(
            output_dir="outputs_analysis_temp",
            max_checkpoints=None,
        )

        logger.info(
            f"RouterSaturationAnalyzer initialized: "
            f"num_layers={self.num_layers}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"device='{device}', "
            f"eval_data_fraction={eval_data_fraction}, "
            f"checkpoint_fractions={self.checkpoint_fractions}, "
            f"random_baseline_k1={random_baseline_k1:.6f}, "
            f"random_baseline_k8={random_baseline_k8:.4f}, "
            f"inference_batch_size={inference_batch_size}"
        )

    def compute_saturation(
        self,
        checkpoint_paths: List[str],
        eval_data: Any,
        final_checkpoint: str,
    ) -> Dict[str, Any]:
        """Compute router saturation across intermediate checkpoints.

        Loads each checkpoint, runs inference on the evaluation data, and
        computes the fraction of expert activations that match the final
        checkpoint's routing decisions.

        The computation is performed for both k=8 (top-8 experts) and k=1
        (top-1 expert by routing probability) as described in Section 5.1.

        Memory management: A single temporary model object is reused across
        all checkpoint loads to avoid repeated GPU memory allocation. After
        getting assignments from each checkpoint, the tensors are moved to
        CPU to free GPU memory.

        Args:
            checkpoint_paths: List of paths to intermediate checkpoint directories.
                              Each path should be a local directory containing
                              model.safetensors (or model.pt) from save().
                              These correspond to 1%, 10%, 20%, 40% of pretraining.
                              Example: ["outputs/checkpoint-00012500",
                                        "outputs/checkpoint-00125000",
                                        "outputs/checkpoint-00250000",
                                        "outputs/checkpoint-00500000"]
            eval_data: Evaluation data for computing routing assignments.
                       Can be a DataLoader, list of batches, or any iterable
                       yielding dicts with "input_ids" key.
                       Should be the 0.5% C4 validation sample.
            final_checkpoint: Path to the final pretraining checkpoint.
                              Used as the reference (T) for saturation computation.
                              Example: "outputs/checkpoint-01223958" or
                              "allenai/OLMoE-1B-7B-0924" (HuggingFace Hub).

        Returns:
            Dict with the following structure:
                {
                    "checkpoint_paths": List[str],  # input checkpoint paths
                    "checkpoint_fractions": List[float],  # fractions of pretraining
                    "num_tokens": int,  # total tokens in eval data
                    "saturation_k8": {  # layer_idx -> List[float] (one per checkpoint)
                        0: [0.15, 0.35, 0.50, 0.65],   # layer 0: slowest
                        1: [0.55, 0.70, 0.78, 0.82],
                        ...
                        15: [0.62, 0.75, 0.80, 0.85],
                    },
                    "saturation_k1": {  # same structure for k=1
                        0: [0.08, 0.20, 0.30, 0.42],
                        ...
                    },
                    "random_baseline_k8": 0.125,
                    "random_baseline_k1": 0.015625,
                    "per_layer_final_saturation_k8": List[float],  # at last checkpoint
                    "per_layer_final_saturation_k1": List[float],
                }

        Raises:
            ValueError: If checkpoint_paths is empty.
            FileNotFoundError: If final_checkpoint does not exist.
        """
        if not checkpoint_paths:
            raise ValueError(
                "checkpoint_paths must be non-empty. "
                "Provide paths to intermediate training checkpoints."
            )

        logger.info(
            f"Computing router saturation: "
            f"{len(checkpoint_paths)} intermediate checkpoints, "
            f"final_checkpoint='{final_checkpoint}'"
        )

        # ------------------------------------------------------------------
        # Step 1: Create a temporary model for loading checkpoints.
        # Reuse the same model object across all checkpoint loads to avoid
        # repeated GPU memory allocation/deallocation.
        # ------------------------------------------------------------------
        logger.info("Creating temporary model for checkpoint loading...")
        temp_model: OLMoEModel = OLMoEModel(self.model.config)
        temp_model = temp_model.to(self.device)
        temp_model.eval()

        # ------------------------------------------------------------------
        # Step 2: Load the final checkpoint and get its expert assignments.
        # This is the reference (T) for all saturation computations.
        # ------------------------------------------------------------------
        logger.info(f"Loading final checkpoint: '{final_checkpoint}'")
        self._checkpoint_manager.load_model_only(
            path=final_checkpoint,
            model=temp_model,
            strict=True,
        )
        temp_model.eval()

        logger.info("Computing expert assignments for final checkpoint (k=8)...")
        assignments_T_k8: List[Tensor] = self._get_expert_assignments(
            model=temp_model,
            data=eval_data,
            top_k=8,
        )

        logger.info("Computing expert assignments for final checkpoint (k=1)...")
        assignments_T_k1: List[Tensor] = self._get_expert_assignments(
            model=temp_model,
            data=eval_data,
            top_k=1,
        )

        # Get total token count from the first layer's assignments.
        num_tokens: int = assignments_T_k8[0].shape[0] if assignments_T_k8 else 0
        logger.info(f"Total evaluation tokens: {num_tokens:,}")

        # ------------------------------------------------------------------
        # Step 3: Initialize result storage.
        # saturation_k8[layer_idx] = list of saturation values, one per checkpoint
        # ------------------------------------------------------------------
        saturation_k8: Dict[int, List[float]] = {
            layer_idx: [] for layer_idx in range(self.num_layers)
        }
        saturation_k1: Dict[int, List[float]] = {
            layer_idx: [] for layer_idx in range(self.num_layers)
        }

        # ------------------------------------------------------------------
        # Step 4: Iterate over intermediate checkpoints.
        # For each checkpoint, load weights, get assignments, compute overlap.
        # ------------------------------------------------------------------
        checkpoint_iterator = (
            tqdm(checkpoint_paths, desc="Processing checkpoints")
            if TQDM_AVAILABLE
            else checkpoint_paths
        )

        for ckpt_idx, ckpt_path in enumerate(checkpoint_iterator):
            logger.info(
                f"Processing checkpoint {ckpt_idx + 1}/{len(checkpoint_paths)}: "
                f"'{ckpt_path}'"
            )

            # Load checkpoint weights into the temporary model.
            try:
                self._checkpoint_manager.load_model_only(
                    path=ckpt_path,
                    model=temp_model,
                    strict=True,
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to load checkpoint '{ckpt_path}': "
                    f"{type(exc).__name__}: {exc}. Skipping."
                )
                # Fill with zeros for this checkpoint.
                for layer_idx in range(self.num_layers):
                    saturation_k8[layer_idx].append(0.0)
                    saturation_k1[layer_idx].append(0.0)
                continue

            temp_model.eval()

            # Get expert assignments for this checkpoint.
            logger.debug(f"Getting k=8 assignments for checkpoint '{ckpt_path}'...")
            assignments_t_k8: List[Tensor] = self._get_expert_assignments(
                model=temp_model,
                data=eval_data,
                top_k=8,
            )

            logger.debug(f"Getting k=1 assignments for checkpoint '{ckpt_path}'...")
            assignments_t_k1: List[Tensor] = self._get_expert_assignments(
                model=temp_model,
                data=eval_data,
                top_k=1,
            )

            # Compute overlap with final checkpoint assignments.
            overlap_k8: Tensor = self._compute_overlap(
                assignments_t=assignments_t_k8,
                assignments_T=assignments_T_k8,
                k=8,
            )
            overlap_k1: Tensor = self._compute_overlap(
                assignments_t=assignments_t_k1,
                assignments_T=assignments_T_k1,
                k=1,
            )

            # Store per-layer saturation values.
            for layer_idx in range(self.num_layers):
                saturation_k8[layer_idx].append(float(overlap_k8[layer_idx].item()))
                saturation_k1[layer_idx].append(float(overlap_k1[layer_idx].item()))

            # Free GPU memory from intermediate assignments.
            del assignments_t_k8
            del assignments_t_k1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                f"Checkpoint '{ckpt_path}' processed: "
                f"avg_saturation_k8={overlap_k8.mean().item():.4f}, "
                f"avg_saturation_k1={overlap_k1.mean().item():.4f}"
            )

        # ------------------------------------------------------------------
        # Step 5: Clean up temporary model and GPU memory.
        # ------------------------------------------------------------------
        del temp_model
        del assignments_T_k8
        del assignments_T_k1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Step 6: Compute per-layer final saturation (at the last checkpoint).
        # ------------------------------------------------------------------
        per_layer_final_k8: List[float] = [
            saturation_k8[layer_idx][-1] if saturation_k8[layer_idx] else 0.0
            for layer_idx in range(self.num_layers)
        ]
        per_layer_final_k1: List[float] = [
            saturation_k1[layer_idx][-1] if saturation_k1[layer_idx] else 0.0
            for layer_idx in range(self.num_layers)
        ]

        # ------------------------------------------------------------------
        # Step 7: Assemble and return results dict.
        # ------------------------------------------------------------------
        # Use only the intermediate checkpoint fractions (exclude 1.0 = final).
        intermediate_fractions: List[float] = [
            f for f in self.checkpoint_fractions if f < 1.0
        ]
        # If checkpoint_paths has fewer entries than intermediate_fractions,
        # use the actual number of checkpoints processed.
        used_fractions: List[float] = intermediate_fractions[:len(checkpoint_paths)]

        results: Dict[str, Any] = {
            "checkpoint_paths": checkpoint_paths,
            "checkpoint_fractions": used_fractions,
            "num_tokens": num_tokens,
            "saturation_k8": saturation_k8,
            "saturation_k1": saturation_k1,
            "random_baseline_k8": self.random_baseline_k8,
            "random_baseline_k1": self.random_baseline_k1,
            "per_layer_final_saturation_k8": per_layer_final_k8,
            "per_layer_final_saturation_k1": per_layer_final_k1,
        }

        logger.info(
            f"Router saturation computation complete: "
            f"num_checkpoints={len(checkpoint_paths)}, "
            f"num_tokens={num_tokens:,}, "
            f"avg_final_saturation_k8={np.mean(per_layer_final_k8):.4f}, "
            f"avg_final_saturation_k1={np.mean(per_layer_final_k1):.4f}"
        )

        return results

    def _get_expert_assignments(
        self,
        model: OLMoEModel,
        data: Any,
        top_k: int = 8,
    ) -> List[Tensor]:
        """Run inference and collect expert assignments per token per layer.

        Processes all batches in the evaluation data and collects which experts
        are activated for each token at each layer. Returns CPU tensors to
        avoid GPU memory accumulation across multiple checkpoint comparisons.

        The model always computes top-8 routing internally (config.yaml: model.top_k=8).
        For top_k=1 analysis, we take only the first column of the top_k_indices
        tensor (the expert with the highest routing probability, since MoERouter
        returns indices sorted by routing probability descending).

        Args:
            model: OLMoEModel loaded with checkpoint weights, in eval mode.
                   Must be on self.device.
            data: Evaluation data iterable. Each item should be a dict with
                  "input_ids" key containing a tensor of shape (batch, seq_len).
                  Can be a DataLoader, list of dicts, or any iterable.
            top_k: Number of top experts to track per token. Must be 1 or 8.
                   For top_k=8: use all 8 activated experts.
                   For top_k=1: use only the highest-probability expert.

        Returns:
            List of num_layers=16 tensors, each of shape (total_tokens, top_k).
            Tensors are on CPU (moved from GPU after each batch to save memory).
            total_tokens = sum of (batch_size * seq_len) across all batches.

        Raises:
            ValueError: If top_k is not 1 or 8.
            RuntimeError: If the model returns no routing information.
        """
        if top_k not in (1, 8):
            raise ValueError(
                f"top_k must be 1 or 8 for router saturation analysis, "
                f"got {top_k}. "
                f"k=8 analyzes all activated experts; k=1 analyzes only the top expert."
            )

        # Initialize per-layer accumulation lists.
        # Each list will hold CPU tensors from each batch.
        layer_assignments: List[List[Tensor]] = [
            [] for _ in range(self.num_layers)
        ]

        total_batches: int = 0
        total_tokens: int = 0

        # Wrap data iterator with tqdm if available.
        data_iterator = (
            tqdm(data, desc=f"  Collecting k={top_k} assignments", leave=False)
            if TQDM_AVAILABLE
            else data
        )

        with torch.no_grad():
            for batch in data_iterator:
                # Extract input_ids from batch dict.
                if isinstance(batch, dict):
                    input_ids: Tensor = batch["input_ids"]
                elif isinstance(batch, (list, tuple)):
                    input_ids = batch[0]
                elif isinstance(batch, Tensor):
                    input_ids = batch
                else:
                    logger.warning(
                        f"Unexpected batch type: {type(batch)}. "
                        f"Expected dict with 'input_ids' key. Skipping batch."
                    )
                    continue

                # Move to device.
                input_ids = input_ids.to(self.device)
                batch_size: int = input_ids.shape[0]
                seq_len: int = input_ids.shape[1]
                batch_tokens: int = batch_size * seq_len

                # Forward pass — collect routing metadata.
                # OLMoEOutput.top_k_indices: List[Tensor], one per layer.
                # Each tensor shape: (batch_size * seq_len, model.top_k=8)
                output: OLMoEOutput = model(input_ids=input_ids)

                if not output.top_k_indices:
                    raise RuntimeError(
                        "Model returned empty top_k_indices. "
                        "Ensure the model is an OLMoEModel with MoE layers."
                    )

                if len(output.top_k_indices) != self.num_layers:
                    raise RuntimeError(
                        f"Expected {self.num_layers} layers of routing indices, "
                        f"got {len(output.top_k_indices)}. "
                        f"Check model architecture configuration."
                    )

                # Collect assignments for each layer.
                for layer_idx in range(self.num_layers):
                    # top_k_indices[layer_idx] shape: (batch*seq, model.top_k=8)
                    layer_indices: Tensor = output.top_k_indices[layer_idx]

                    # Validate shape.
                    assert layer_indices.shape[0] == batch_tokens, (
                        f"Layer {layer_idx} indices have {layer_indices.shape[0]} tokens, "
                        f"expected {batch_tokens} (batch_size={batch_size} * seq_len={seq_len})."
                    )
                    assert layer_indices.shape[1] == self.top_k, (
                        f"Layer {layer_idx} indices have {layer_indices.shape[1]} experts, "
                        f"expected model.top_k={self.top_k}."
                    )

                    # For top_k=1: take only the first column (highest routing prob expert).
                    # MoERouter.forward() uses torch.topk which returns indices in
                    # descending order of routing probability (sorted=False in our impl,
                    # but we use the first column as the highest-prob expert).
                    # Note: MoERouter uses sorted=False, so we need to find the actual
                    # top-1 by looking at routing weights.
                    if top_k == 1:
                        # Get the routing weights to find the true top-1 expert.
                        # output.routing_weights[layer_idx] shape: (batch*seq, num_experts)
                        if output.routing_weights and len(output.routing_weights) > layer_idx:
                            routing_weights: Tensor = output.routing_weights[layer_idx]
                            # For each token, find which of the top-k indices has the
                            # highest routing weight.
                            # routing_weights has shape (batch*seq, num_experts) — full softmax.
                            # We need the expert with the highest weight among the top-k selected.
                            # Gather weights for the selected experts.
                            # layer_indices: (batch*seq, 8)
                            # selected_weights: (batch*seq, 8)
                            selected_weights: Tensor = routing_weights.gather(
                                dim=1,
                                index=layer_indices.long(),
                            )
                            # Find the index within top-k that has the highest weight.
                            # top1_within_topk: (batch*seq,)
                            top1_within_topk: Tensor = selected_weights.argmax(dim=1)
                            # Get the actual expert ID for the top-1.
                            # top1_expert: (batch*seq, 1)
                            top1_expert: Tensor = layer_indices.gather(
                                dim=1,
                                index=top1_within_topk.unsqueeze(1),
                            )
                            selected_indices: Tensor = top1_expert.cpu()
                        else:
                            # Fallback: use first column if routing weights unavailable.
                            # This assumes top_k_indices are sorted by routing probability.
                            selected_indices = layer_indices[:, :1].cpu()
                    else:
                        # top_k=8: use all activated expert indices.
                        selected_indices = layer_indices.cpu()

                    layer_assignments[layer_idx].append(selected_indices)

                total_batches += 1
                total_tokens += batch_tokens

                # Free GPU memory from this batch's output.
                del output
                del input_ids

        if total_tokens == 0:
            logger.warning(
                "No tokens processed during expert assignment collection. "
                "Check that eval_data is non-empty."
            )
            # Return empty tensors.
            return [
                torch.zeros((0, top_k), dtype=torch.long)
                for _ in range(self.num_layers)
            ]

        # Concatenate all batches for each layer.
        # Result: list of (total_tokens, top_k) tensors on CPU.
        result: List[Tensor] = []
        for layer_idx in range(self.num_layers):
            if layer_assignments[layer_idx]:
                concatenated: Tensor = torch.cat(
                    layer_assignments[layer_idx], dim=0
                )
                # Validate shape.
                assert concatenated.shape == (total_tokens, top_k), (
                    f"Layer {layer_idx} concatenated shape {concatenated.shape} "
                    f"doesn't match expected ({total_tokens}, {top_k})."
                )
                result.append(concatenated)
            else: