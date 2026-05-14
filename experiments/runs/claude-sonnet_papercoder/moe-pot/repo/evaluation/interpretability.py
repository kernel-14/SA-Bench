```python
# evaluation/interpretability.py
"""Interpretability analysis for the MoE-POT router-gating network.

Implements the InterpretabilityAnalyzer class, which provides tools to
analyze the emergent dataset classification capability of the router-gating
network described in Section 5.4 and Appendix B.4 of the MoE-POT paper.

Core insight from the paper:
    "The trained router-gating network can infer the PDE type of input data
    with 98% accuracy, showing MoE-POT's ability to effectively handle
    diverse PDE datasets."

The classification algorithm (Appendix B.4):
    1. For each dataset i, compute mean routing vector:
       Y_i = (1/N_i) * sum_{j=1}^{N_i} Y_{ij}
       where Y_{ij} ∈ R^16 is the full softmax output for sample j.
    2. For a new input X with routing vector I_0:
       f(I_0, Y_i) = -sum_{k=1}^{16} I_{0,k} * log(Y_{i,k})
       i_0 = argmin_i f(I_0, Y_i)
    3. Classify X as belonging to dataset i_0.

Key results to reproduce:
    - Block 2 (0-indexed: block_idx=1) achieves 97.7% classification accuracy
      (config.yaml interpretability.analysis_block_idx: 2, paper Section 5.4)
    - Related datasets (NS 1e-5 and NS 1e-3) show similar routing patterns
    - Dissimilar datasets (SWE and DR) show distinct routing patterns
    - Router specialization emerges during training (Table 16, Appendix C.6)

From config.yaml (interpretability section):
    analysis_block_idx: 2          (1-indexed; 0-indexed: 1)
    num_experts: 16                (N_r = 16 routed experts)
    classification_method: "cross_entropy"
    target_accuracy: 0.977         (97.7% dataset classification accuracy)

From config.yaml (models section):
    models.tiny.top_k: 4           (K=4 experts selected per input)
    models.tiny.num_routed_experts: 16
"""

import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.moe_pot import MoEPOT


# ---------------------------------------------------------------------------
# Module-level constants from config.yaml
# ---------------------------------------------------------------------------

# Number of routed experts N_r (config.yaml models.*.num_routed_experts: 16).
_NUM_ROUTED_EXPERTS: int = 16

# Top-K experts selected per input (config.yaml models.*.top_k: 4).
_TOP_K: int = 4

# Epsilon for numerical stability in log computations.
# Prevents log(0) when any Y_{i,k} is exactly zero.
_LOG_EPSILON: float = 1e-8

# Ordered list of the 6 pre-training dataset names, matching the index
# convention used in MultiPDEDataset (dataset_idx 0–5).
# This ordering must be consistent with the order datasets are added to
# MultiPDEDataset in main.py.
_PRETRAIN_DATASET_NAMES: List[str] = [
    "fno_ns_1e5",
    "fno_ns_1e3",
    "pdebench_cns_0p1_0p01",
    "pdebench_swe",
    "pdebench_dr",
    "cfdbench",
]

# Mapping from dataset name string to integer index (0-indexed).
# Used to convert DataLoader keys to integer labels for classification.
_DATASET_NAME_TO_IDX: Dict[str, int] = {
    name: idx for idx, name in enumerate(_PRETRAIN_DATASET_NAMES)
}


class InterpretabilityAnalyzer:
    """Analyzes the router-gating network's emergent dataset classification ability.

    Provides four analysis capabilities corresponding to the paper's
    interpretability experiments:

    1. compute_mean_routing_vectors: Computes the average routing fingerprint
       Y_i ∈ R^16 for each dataset (Appendix B.4, Phase 1 of classification).

    2. classify_dataset: Classifies a single input to its source dataset by
       finding the nearest mean routing vector via cross-entropy distance
       (Appendix B.4, Phase 2 of classification).

    3. evaluate_classification_accuracy: Computes overall classification
       accuracy across all datasets (Section 5.4, target: 97.7% at block 2).

    4. compute_expert_usage_ratio: Computes per-expert selection frequency
       for a dataset, enabling the usage ratio visualization in Figure 2 (right).

    5. track_routing_evolution: Tracks how classification accuracy emerges
       during pre-training by evaluating at different checkpoints (Table 16,
       Appendix C.6).

    All methods operate in torch.no_grad() context and use the model's
    get_router_weights() method to extract full softmax routing distributions
    without running the full prediction pipeline.

    Note on block_idx convention:
        This class uses 0-indexed block_idx internally (Python convention).
        The paper uses 1-indexed blocks: "Block 2 achieves 97.7% accuracy"
        corresponds to block_idx=1 in this implementation.
        config.yaml interpretability.analysis_block_idx: 2 (1-indexed)
        → use block_idx=1 (0-indexed) in method calls.

    Attributes:
        model: The MoEPOT model to analyze. Set to eval mode in __init__.
            All analysis methods use torch.no_grad() for memory efficiency.
        device: Target device string for model inference (e.g., 'cuda:0').
        num_routed_experts: Number of routed experts N_r. Default 16
            (config.yaml models.*.num_routed_experts: 16).
    """

    def __init__(
        self,
        model: MoEPOT,
        device: str = "cuda",
    ) -> None:
        """Initializes the InterpretabilityAnalyzer.

        Moves the model to the target device and sets it to eval mode.
        All analysis methods are inference-only and do not modify model weights.

        Args:
            model: Initialized MoEPOT model with pre-trained weights loaded.
                Will be moved to device in-place and set to eval mode.
                The model's get_router_weights() method is used throughout
                to extract full softmax routing distributions.
            device: Target device string for model inference. Default 'cuda'.
                Use 'cuda:0', 'cuda:1', etc. for specific GPU selection.
                Falls back gracefully to 'cpu' if CUDA is unavailable.
        """
        # Resolve device: fall back to CPU if CUDA is requested but unavailable.
        if device.startswith("cuda") and not torch.cuda.is_available():
            resolved_device: str = "cpu"
        else:
            resolved_device = device

        self.device: str = resolved_device

        # Move model to target device and set to eval mode.
        # eval() disables dropout and batchnorm training behavior.
        # All analysis methods use torch.no_grad() for memory efficiency.
        self.model: MoEPOT = model.to(torch.device(self.device))
        self.model.eval()

        # Number of routed experts N_r from config.yaml models.*.num_routed_experts.
        # Default 16 matches the paper's architecture (Section 3.2).
        self.num_routed_experts: int = _NUM_ROUTED_EXPERTS

    def compute_mean_routing_vectors(
        self,
        loader: DataLoader,
        dataset_idx: int,
        block_idx: int = 1,
    ) -> torch.Tensor:
        """Computes the average expert selection distribution Y_i for a dataset.

        Implements Phase 1 of the dataset classification algorithm from
        Appendix B.4:
            Y_i = (1/N_i) * sum_{j=1}^{N_i} Y_{ij}
        where Y_{ij} ∈ R^16 is the full softmax routing output for the
        j-th sample in dataset i.

        The resulting Y_i serves as the routing "fingerprint" for dataset i.
        Related datasets (e.g., NS 1e-5 and NS 1e-3) produce similar Y_i
        vectors, while dissimilar datasets (e.g., SWE and DR) produce
        distinct Y_i vectors (Figure 2 right in the paper).

        Args:
            loader: DataLoader for the dataset to compute mean vectors for.
                Yields batches of (u_input, u_target) 2-tuples or
                (u_input, u_target, dataset_idx) 3-tuples. Shapes:
                  - u_input: (B, T, C, H, W) — T=10 input frames
                  - u_target: (B, C, H, W) — next frame (not used here)
                The dataset_idx from the loader is ignored; the caller
                provides the correct index via the dataset_idx parameter.
            dataset_idx: Integer index of the dataset (0-indexed, 0–5 for
                the 6 pre-training datasets). Used only for logging/display;
                the actual computation is agnostic to this value.
            block_idx: 0-indexed block from which to extract routing weights.
                Default 1 (= paper's Block 2, which achieves 97.7% accuracy).
                config.yaml interpretability.analysis_block_idx: 2 (1-indexed)
                → block_idx=1 (0-indexed).

        Returns:
            Mean routing vector Y_i of shape (num_routed_experts,) = (16,)
            as a CPU float32 tensor. Values are in (0, 1) and sum to
            approximately 1.0 (mean of softmax outputs). Returns a uniform
            distribution (1/16 for each expert) if the loader is empty.

        Raises:
            IndexError: If block_idx is out of range for the model's blocks.
                Propagated from MoEPOT.get_router_weights().
        """
        # Initialize accumulator on CPU to avoid GPU memory accumulation
        # across multiple datasets when called in a loop.
        # Shape: (num_routed_experts,) = (16,)
        accumulator: torch.Tensor = torch.zeros(
            self.num_routed_experts,
            dtype=torch.float32,
        )
        n_samples: int = 0

        # tqdm progress bar for monitoring progress over large datasets.
        loader_iter = tqdm(
            loader,
            desc=f"Computing mean routing vectors (dataset {dataset_idx}, block {block_idx})",
            leave=False,
        )

        with torch.no_grad():
            batch: Tuple
            for batch in loader_iter:
                # ----------------------------------------------------------
                # Step 1: Unpack batch — handle 2-tuple and 3-tuple formats
                # ----------------------------------------------------------
                u_input: torch.Tensor
                if len(batch) >= 2:
                    u_input = batch[0]
                else:
                    continue

                # ----------------------------------------------------------
                # Step 2: Move input to target device
                # ----------------------------------------------------------
                u_input = u_input.to(self.device, non_blocking=True)
                # u_input shape: (B, T, C, H, W)

                batch_size: int = u_input.shape[0]

                # ----------------------------------------------------------
                # Step 3: Extract full softmax routing weights
                # ----------------------------------------------------------
                # model.get_router_weights() runs a partial forward pass
                # up to block_idx and returns the full softmax distribution
                # over all num_routed_experts experts.
                # Shape: (B, num_routed_experts) = (B, 16)
                routing_weights: torch.Tensor = self.model.get_router_weights(
                    u_input, block_idx
                )
                # routing_weights shape: (B, 16)
                # Values are in (0, 1) and sum to 1.0 along dim=-1 per sample.

                # ----------------------------------------------------------
                # Step 4: Accumulate sum over batch dimension
                # ----------------------------------------------------------
                # Sum routing weights over the batch dimension to accumulate
                # the total routing weight per expert across all samples.
                # routing_weights.sum(dim=0) shape: (16,)
                # Move to CPU before accumulating to avoid GPU memory growth.
                accumulator = accumulator + routing_weights.sum(dim=0).cpu()
                n_samples += batch_size

                # Update progress bar with current sample count.
                loader_iter.set_postfix(n_samples=n_samples)

        # ------------------------------------------------------------------
        # Step 5: Compute mean routing vector
        # ------------------------------------------------------------------
        # Guard against empty loader: return uniform distribution.
        if n_samples == 0:
            return torch.full(
                (self.num_routed_experts,),
                fill_value=1.0 / self.num_routed_experts,
                dtype=torch.float32,
            )

        # Y_i = accumulator / N_i
        # Shape: (16,) — mean routing weight per expert over all N_i samples.
        mean_vector: torch.Tensor = accumulator / float(n_samples)

        # Return as CPU tensor (detached from computation graph).
        return mean_vector.detach().cpu()

    def classify_dataset(
        self,
        x: torch.Tensor,
        mean_vectors: Dict[int, torch.Tensor],
        block_idx: int = 1,
    ) -> int:
        """Classifies an input to its source dataset via routing fingerprint matching.

        Implements Phase 2 of the dataset classification algorithm from
        Appendix B.4:
            f(I_0, Y_i) = -sum_{k=1}^{16} I_{0,k} * log(Y_{i,k})
            i_0 = argmin_i f(I_0, Y_i)

        The cross-entropy distance measures how well the input's routing
        distribution I_0 matches each dataset's mean routing fingerprint Y_i.
        The dataset with the smallest cross-entropy distance is the predicted
        source dataset.

        For batch inputs (B > 1), the routing vectors are averaged over the
        batch before computing cross-entropy distances. This is appropriate
        when all samples in the batch come from the same dataset (as in the
        evaluation protocol where each loader contains one dataset).

        Args:
            x: Input tensor of shape (B, T, C, H, W) or (1, T, C, H, W)
                for a single sample. May be on CPU — will be moved to
                self.device internally.
            mean_vectors: Dictionary mapping dataset index (int) to mean
                routing vector Y_i of shape (num_routed_experts,) = (16,).
                Produced by compute_mean_routing_vectors() for each dataset.
                Keys are integer dataset indices (0-indexed, 0–5 for the
                6 pre-training datasets).
            block_idx: 0-indexed block from which to extract routing weights.
                Default 1 (= paper's Block 2). Must match the block_idx
                used to compute mean_vectors for consistent comparison.

        Returns:
            Integer dataset index i_0 = argmin_i f(I_0, Y_i). The predicted
            source dataset for the input x. Values are in the range of keys
            in mean_vectors (typically 0–5 for the 6 pre-training datasets).

        Raises:
            ValueError: If mean_vectors is empty (no reference distributions
                to compare against).
            IndexError: If block_idx is out of range for the model's blocks.
        """
        if not mean_vectors:
            raise ValueError(
                "mean_vectors is empty. Cannot classify without reference "
                "routing distributions. Call compute_mean_routing_vectors() "
                "for each dataset first."
            )

        # ------------------------------------------------------------------
        # Step 1: Move input to target device and extract routing vector
        # ------------------------------------------------------------------
        x_device: torch.Tensor = x.to(self.device, non_blocking=True)
        # x_device shape: (B, T, C, H, W)

        with torch.no_grad():
            # Get full softmax routing weights for the input.
            # Shape: (B, num_routed_experts) = (B, 16)
            routing_weights: torch.Tensor = self.model.get_router_weights(
                x_device, block_idx
            )

        # ------------------------------------------------------------------
        # Step 2: Compute I_0 — routing vector for this input
        # ------------------------------------------------------------------
        # Average over the batch dimension to get a single routing vector.
        # For single-sample inputs (B=1), this is equivalent to squeezing.
        # For multi-sample batches, averaging gives the batch-level fingerprint.
        # I_0 shape: (16,)
        I_0: torch.Tensor = routing_weights.mean(dim=0).cpu()
        # Values are in (0, 1) and sum to approximately 1.0.

        # ------------------------------------------------------------------
        # Step 3: Compute cross-entropy distance to each dataset's fingerprint
        # ------------------------------------------------------------------
        # f(I_0, Y_i) = -sum_{k=1}^{16} I_{0,k} * log(Y_{i,k} + epsilon)
        # The dataset with the smallest cross-entropy is the predicted source.
        best_dataset_idx: int = -1
        best_cross_entropy: float = float("inf")

        dataset_idx: int
        Y_i: torch.Tensor
        for dataset_idx, Y_i in mean_vectors.items():
            # Move Y_i to CPU for computation (I_0 is already on CPU).
            Y_i_cpu: torch.Tensor = Y_i.cpu()

            # Compute cross-entropy: f(I_0, Y_i) = -sum_k I_{0,k} * log(Y_{i,k})
            # Add epsilon to Y_i inside log to prevent log(0) = -inf.
            # I_0 is used as weights (not inside log), so no epsilon needed there.
            # Shape: scalar
            log_Y_i: torch.Tensor = torch.log(Y_i_cpu + _LOG_EPSILON)
            cross_entropy: float = float(-(I_0 * log_Y_i).sum().item())

            # Track the dataset with minimum cross-entropy distance.
            if cross_entropy < best_cross_entropy:
                best_cross_entropy = cross_entropy
                best_dataset_idx = dataset_idx

        return best_dataset_idx

    def evaluate_classification_accuracy(
        self,
        loaders: Dict[str, DataLoader],
        block_idx: int = 1,
        train_loaders: Optional[Dict[str, DataLoader]] = None,
    ) -> float:
        """Computes dataset classification accuracy using the router-gating network.

        Reproduces the interpretability experiment from Section 5.4 and
        Figure 4(c) of the paper. The target accuracy is 97.7% at block 2
        (block_idx=1 in 0-indexed convention).

        Two-phase algorithm:
          Phase 1: Compute mean routing vectors Y_i for each dataset using
                   training data (or test data if train_loaders is None).
          Phase 2: Classify each test sample by finding the nearest Y_i
                   via cross-entropy distance, then compute accuracy.

        The classification is performed sample-by-sample (not batch-level)
        to match the paper's evaluation protocol. Batch processing is used
        internally for efficiency (one forward pass per batch, then classify
        each sample in the batch independently).

        Args:
            loaders: Dictionary mapping dataset name strings to their test
                DataLoaders. Keys must be recognizable dataset names (e.g.,
                'fno_ns_1e5', 'pdebench_swe'). Used for Phase 2 classification.
                If train_loaders is None, also used for Phase 1 (computing Y_i).
            block_idx: 0-indexed block from which to extract routing weights.
                Default 1 (= paper's Block 2, which achieves 97.7% accuracy).
                config.yaml interpretability.analysis_block_idx: 2 (1-indexed)
                → block_idx=1 (0-indexed).
            train_loaders: Optional dictionary mapping dataset name strings to
                their training DataLoaders. If provided, used for Phase 1
                (computing mean routing vectors Y_i) to avoid data leakage.
                If None, loaders is used for both phases (simpler, matches
                the paper's description which does not specify train/test split
                for computing Y_i).

        Returns:
            Classification accuracy as a float in [0, 1]. The paper reports
            0.977 (97.7%) for block_idx=1 (Block 2 in 1-indexed notation).
            Returns 0.0 if no samples are processed (empty loaders).

        Note:
            Dataset names in loaders that are not in _DATASET_NAME_TO_IDX
            are assigned sequential indices starting from len(_PRETRAIN_DATASET_NAMES).
            This handles downstream task datasets (NS 1e-4, CNS 1 0.01, PDEArena)
            that are not in the pre-training set.
        """
        # ------------------------------------------------------------------
        # Build dataset name → integer index mapping
        # ------------------------------------------------------------------
        # Start with the standard pre-training dataset mapping.
        # Extend with any additional datasets in loaders that are not in
        # the pre-training set (e.g., downstream task datasets).
        name_to_idx: Dict[str, int] = dict(_DATASET_NAME_TO_IDX)
        next_idx: int = len(_PRETRAIN_DATASET_NAMES)

        dataset_name: str
        for dataset_name in loaders.keys():
            if dataset_name not in name_to_idx:
                name_to_idx[dataset_name] = next_idx
                next_idx += 1

        # ------------------------------------------------------------------
        # Phase 1: Compute mean routing vectors Y_i for each dataset
        # ------------------------------------------------------------------
        # Use train_loaders if provided (avoids data leakage), otherwise
        # use the same loaders as Phase 2 (matches paper's description).
        reference_loaders: Dict[str, DataLoader] = (
            train_loaders if train_loaders is not None else loaders
        )

        mean_vectors: Dict[int, torch.Tensor] = {}

        print(f"\n[InterpretabilityAnalyzer] Phase 1: Computing mean routing vectors "
              f"for {len(reference_loaders)} datasets at block_idx={block_idx}...")

        ref_name: str
        ref_loader: DataLoader
        for ref_name, ref_loader in reference_loaders.items():
            ref_idx: int = name_to_idx.get(ref_name, -1)
            if ref_idx < 0:
                # Assign a new index for unrecognized dataset names.
                name_to_idx[ref_name] = next_idx
                ref_idx = next_idx
                next_idx += 1

            Y_i: torch.Tensor = self.compute_mean_routing_vectors(
                loader=ref_loader,
                dataset_idx=ref_idx,
                block_idx=block_idx,
            )
            mean_vectors[ref_idx] = Y_i

        print(f"[InterpretabilityAnalyzer] Phase 1 complete. "
              f"Computed mean vectors for {len(mean_vectors)} datasets.")

        # ------------------------------------------------------------------
        # Phase 2: Classify each test sample and compute accuracy
        # ------------------------------------------------------------------
        all_true_labels: List[int] = []
        all_pred_labels: List[int] = []

        print(f"[InterpretabilityAnalyzer] Phase 2: Classifying test samples...")

        test_name: str
        test_loader: DataLoader
        for test_name, test_loader in loaders.items():
            true_idx: int = name_to_idx.get(test_name, -1)
            if true_idx < 0:
                print(f"  Warning: Dataset '{test_name}' not in name_to_idx. Skipping.")
                continue

            # tqdm progress bar for this dataset.
            dataset_iter = tqdm(
                test_loader,
                desc=f"  Classifying {test_name} (true_idx={true_idx})",
                leave=False,
            )

            with torch.no_grad():
                batch: Tuple
                for batch in dataset_iter:
                    # Unpack batch — handle 2-tuple and 3-tuple formats.
                    u_input: torch.Tensor
                    if len(batch) >= 2:
                        u_input = batch[0]
                    else:
                        continue

                    # Move to target device.
                    u_input = u_input.to(self.device, non_blocking=True)
                    # u_input shape: (B, T, C, H, W)

                    batch_size: int = u_input.shape[0]

                    # Get routing weights for the entire batch in one forward pass.
                    # Shape: (B, num_routed_experts) = (B, 16)
                    routing_batch: torch.Tensor = self.model.get_router_weights(
                        u_input, block_idx
                    )
                    # Move to CPU for cross-entropy computation.
                    routing_batch_cpu: torch.Tensor = routing_batch.cpu()

                    # Classify each sample in the batch independently.
                    sample_i: int
                    for sample_i in range(batch_size):
                        # Extract routing vector for this sample.
                        # I_0 shape: (16,)
                        I_0: torch.Tensor = routing_batch_cpu[sample_i]

                        # Find the nearest dataset fingerprint via cross-entropy.
                        best_pred_idx: int = -1
                        best_ce: float = float("inf")

                        cand_idx: int
                        Y_i_cand: torch.Tensor
                        for cand_idx, Y_i_cand in mean_vectors.items():
                            # f(I_0, Y_i) = -sum_k I_{0,k} * log(Y_{i,k} + epsilon)
                            log_Y: torch.Tensor = torch.log(
                                Y_i_cand.cpu() + _LOG_EPSILON
                            )
                            ce: float = float(-(I_0 * log_Y).sum().item())

                            if ce < best_ce:
                                best_ce = ce
                                best_pred_idx = cand_idx

                        # Record true and predicted labels.
                        all_true_labels.append(true_idx)
                        all_pred_labels.append(best_pred_idx)

        # ------------------------------------------------------------------
        # Step 3: Compute classification accuracy
        # ------------------------------------------------------------------
        if not all_true_labels:
            print("[InterpretabilityAnalyzer] Warning: No samples classified. "
                  "Returning accuracy=0.0.")
            return 0.0

        # Compute accuracy: fraction of correctly classified samples.
        # Try sklearn first for robustness; fall back to manual computation.
        try:
            from sklearn.metrics import accuracy_score  # pylint: disable=import-outside-toplevel
            accuracy: float = float(
                accuracy_score(all_true_labels, all_pred_labels)
            )
        except ImportError:
            # Manual accuracy computation as fallback.
            correct: int = sum(
                t == p
                for t, p in zip(all_true_labels, all_pred_labels)
            )
            accuracy = correct / len(all_true_labels)

        total_samples: int = len(all_true_labels)
        correct_count: int = sum(
            t == p for t, p in zip(all_true_labels, all_pred_labels)
        )
        print(
            f"[InterpretabilityAnalyzer] Classification accuracy at block_idx={block_idx}: "
            f"{accuracy:.4f} ({correct_count}/{total_samples} correct). "
            f"Target: {0.977:.4f} (97.7%)."
        )

        return accuracy

    def compute_expert_usage_ratio(
        self,
        loader: DataLoader,
        block_idx: int = 1,
    ) -> torch.Tensor:
        """Computes the average top-K expert selection frequency for a dataset.

        Produces the expert usage ratio visualization shown in Figure 2 (right)
        of the paper. For each of the 16 routed experts, computes the fraction
        of samples that selected it in their top-K routing decisions.

        The usage ratio reflects how specialized each expert is for a given
        dataset. Related datasets (NS 1e-5 and NS 1e-3) show similar usage
        patterns, while dissimilar datasets (SWE and DR) show distinct patterns.

        Sanity check: sum(usage_ratio) == top_k == 4, since each sample
        selects exactly K=4 experts. This can be verified by the caller.

        Args:
            loader: DataLoader for the dataset to analyze. Yields batches of
                (u_input, u_target) 2-tuples or (u_input, u_target, dataset_idx)
                3-tuples. Only u_input is used.
            block_idx: 0-indexed block from which to extract routing weights.
                Default 1 (= paper's Block 2). The paper's Figure 2 (right)
                shows usage ratios for "block 4" (1-indexed) = block_idx=3
                (0-indexed). Adjust as needed for different visualizations.

        Returns:
            Expert usage ratio tensor of shape (num_routed_experts,) = (16,)
            as a CPU float32 tensor. The k-th element is the fraction of
            samples that selected expert k in their top-K routing decisions.
            Values are in [0, 1] and sum to top_k=4 (not 1.0).
            Returns a zero tensor if the loader is empty.

        Note:
            The usage ratio is based on top-K selection from the full softmax
            distribution, not from the re-normalized top-K weights. This
            matches the paper's description of "usage ratio of routed experts"
            in Figure 2 (right).
        """
        # Initialize usage count accumulator on CPU.
        # Shape: (num_routed_experts,) = (16,)
        usage_counts: torch.Tensor = torch.zeros(
            self.num_routed_experts,
            dtype=torch.float32,
        )
        n_samples: int = 0

        # tqdm progress bar.
        loader_iter = tqdm(
            loader,
            desc=f"Computing expert usage ratio (block {block_idx})",
            leave=False,
        )

        with torch.no_grad():
            batch: Tuple
            for batch in loader_iter:
                # Unpack batch — handle 2-tuple and 3-tuple formats.
                u_input: torch.Tensor
                if len(batch) >= 2:
                    u_input = batch[0]
                else:
                    continue

                # Move to target device.
                u_input = u_input.to(self.device, non_blocking=True)
                # u_input shape: (B, T, C, H, W)

                batch_size: int = u_input.shape[0]

                # ----------------------------------------------------------
                # Step 1: Get full softmax routing weights
                # ----------------------------------------------------------
                # Shape: (B, num_routed_experts) = (B, 16)
                routing_weights: torch.Tensor = self.model.get_router_weights(
                    u_input, block_idx
                )
                # routing_weights shape: (B, 16)
                # Values are in (0, 1) and sum to 1.0 along dim=-1 per sample.

                # ----------------------------------------------------------
                # Step 2: Determine top-K expert indices per sample
                # ----------------------------------------------------------
                # Select the K=4 experts with highest routing weights.
                # top_k_indices shape: (B, K) = (B, 4)
                # Values are in [0, num_routed_experts) = [0, 16).
                top_k_indices: torch.Tensor = routing_weights.topk(
                    k=_