# evaluation/evaluator.py
"""Evaluator class for the MoE-POT evaluation pipeline.

Implements zero-shot evaluation, per-dataset L2RE computation, auto-regressive
rollout, and inference time benchmarking. Faithfully implements the evaluation
protocol described in the paper:

  - Table 1 "Pre-trained" rows: evaluate_zero_shot()
  - Table 1 "Fine-tuned" rows: evaluate_dataset() after Finetuner.finetune()
  - Table 2 downstream tasks: evaluate_dataset() with downstream loaders
  - Table 3 inference time: benchmark_inference_time()
  - Appendix C.3 rollout error: autoregressive_rollout() + autoregressive_l2re()

From config.yaml (evaluation section):
    metric: "L2RE"
    rollout_steps: 10
    inference_warmup_runs: 10
    inference_timed_runs: 100

From config.yaml (architecture section):
    input_timesteps: 10     (T=10 frames as input to the model)
    max_channels: 4         (all datasets padded to 4 channels)
    target_resolution: 128  (H=W=128 after preprocessing)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import (
    autoregressive_l2re,
    compute_inference_time,
    l2_relative_error,
)
from models.moe_pot import MoEPOT


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maps dataset name to the number of meaningful (non-padded) channels.
# Used in evaluate_dataset() to slice predictions before computing L2RE,
# excluding constant-padded channels (filled with 1.0) from the error metric.
#
# Channel counts from config.yaml datasets section:
#   fno_ns_1e5:            1  (vorticity)
#   fno_ns_1e3:            1  (vorticity)
#   fno_ns_1e4:            1  (vorticity, downstream task)
#   pdebench_cns_0p1_0p01: 4  (rho, vx, vy, p)
#   pdebench_cns_1_0p01:   4  (rho, vx, vy, p, downstream task)
#   pdebench_swe:          1  (water depth h)
#   pdebench_dr:           1  (density u)
#   cfdbench:              3  (vx, vy, p — mask channel excluded from L2RE)
#   pdearena:              3  (vx, vy, p, downstream task)
#
# Note: CFDBench has has_mask: true in config.yaml, so the 4th channel is
# a binary geometry mask. DATASET_CHANNELS["cfdbench"] = 3 ensures the mask
# channel is excluded from L2RE computation by slicing u_pred[:, :3, :, :].
DATASET_CHANNELS: Dict[str, int] = {
    "fno_ns_1e5": 1,
    "fno_ns_1e3": 1,
    "fno_ns_1e4": 1,
    "pdebench_cns_0p1_0p01": 4,
    "pdebench_cns_1_0p01": 4,
    "pdebench_swe": 1,
    "pdebench_dr": 1,
    "cfdbench": 3,
    "pdearena": 3,
}

# Default number of meaningful channels when dataset name is not in the map.
# Falls back to max_channels=4 (config.yaml architecture.max_channels).
_DEFAULT_CHANNELS: int = 4

# Default inference timing parameters from config.yaml evaluation section.
_DEFAULT_WARMUP_RUNS: int = 10    # config.yaml: evaluation.inference_warmup_runs
_DEFAULT_TIMED_RUNS: int = 100    # config.yaml: evaluation.inference_timed_runs


class Evaluator:
    """Evaluates MoE-POT models on PDE datasets using L2 Relative Error.

    Provides four evaluation capabilities:
      1. Zero-shot evaluation across all test datasets (Table 1 "Pre-trained")
      2. Per-dataset L2RE computation (Table 1 "Fine-tuned", Table 2)
      3. Auto-regressive rollout for multi-step prediction (Appendix C.3)
      4. Inference time benchmarking (Table 3)

    All evaluation methods operate in torch.no_grad() context and set the
    model to eval mode before inference. The model is restored to train mode
    after each evaluation call to avoid interfering with subsequent training.

    The evaluator is stateless with respect to training configuration — it
    operates purely on the model and test data, making it reusable across
    different training stages (zero-shot, fine-tuned, downstream).

    Attributes:
        model: The MoEPOT model to evaluate. Moved to self.device in __init__.
            May be a raw MoEPOT or a DistributedDataParallel-wrapped instance.
        test_loaders: Dictionary mapping dataset name strings to their
            respective test DataLoaders. Keys match DATASET_CHANNELS keys.
            Values are DataLoaders yielding (u_input, u_target) 2-tuples or
            (u_input, u_target, dataset_idx) 3-tuples.
        device: Target device string (e.g., 'cuda:0', 'cpu'). All tensors
            are moved to this device before model inference.
    """

    def __init__(
        self,
        model: MoEPOT,
        test_loaders: Dict[str, DataLoader],
        device: str = "cuda",
    ) -> None:
        """Initializes the Evaluator and moves the model to the target device.

        Args:
            model: Initialized MoEPOT model. Will be moved to device in-place.
                May be a raw MoEPOT instance or a DistributedDataParallel
                wrapper. The model is NOT set to eval mode here — that is
                done per-method to allow flexibility.
            test_loaders: Dictionary mapping dataset name strings to their
                test DataLoaders. Keys should match DATASET_CHANNELS keys
                (e.g., 'fno_ns_1e5', 'pdebench_swe'). Values are DataLoaders
                that yield batches of (u_input, u_target) or
                (u_input, u_target, dataset_idx) tuples.
                May be an empty dict if no evaluation datasets are available.
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
        self.test_loaders: Dict[str, DataLoader] = test_loaders

        # Move model to the target device.
        # .to() is idempotent if the model is already on the correct device.
        self.model: MoEPOT = model.to(torch.device(self.device))

    def evaluate_zero_shot(self) -> Dict[str, float]:
        """Evaluates the pre-trained model on all test datasets without fine-tuning.

        Corresponds to the "Pre-trained" section of Table 1 in the paper.
        Iterates over all datasets in self.test_loaders and computes the
        mean L2 Relative Error for each.

        Expected output for MoE-POT-S (90M activated params) from Table 1:
            {
                "fno_ns_1e5":            0.0552,
                "fno_ns_1e3":            0.00583,
                "pdebench_cns_0p1_0p01": 0.00959,
                "pdebench_swe":          0.00289,
                "pdebench_dr":           0.0342,
                "cfdbench":              0.00448,
            }

        Returns:
            Dictionary mapping dataset name to mean L2RE on that dataset's
            test split. Returns an empty dict if self.test_loaders is empty.
            Values are Python floats in [0, ∞) where lower is better.
        """
        results: Dict[str, float] = {}

        dataset_name: str
        loader: DataLoader
        for dataset_name, loader in self.test_loaders.items():
            l2re: float = self.evaluate_dataset(loader, dataset_name)
            results[dataset_name] = l2re

        return results

    def evaluate_dataset(
        self,
        loader: DataLoader,
        dataset_name: str = "unknown",
    ) -> float:
        """Computes the mean L2 Relative Error for a single dataset.

        Core evaluation routine used for both zero-shot and fine-tuned
        evaluation. Runs the model in eval mode over all batches in the
        loader, computes per-batch L2RE, and returns the batch-averaged mean.

        Channel slicing: Before computing L2RE, predictions and targets are
        sliced to the actual number of meaningful channels for the dataset
        (using DATASET_CHANNELS). This excludes constant-padded channels
        (filled with 1.0) and the CFDBench geometry mask channel from the
        error metric, ensuring fair comparison across datasets.

        Args:
            loader: DataLoader for the dataset to evaluate. Yields batches
                of (u_input, u_target) 2-tuples or (u_input, u_target,
                dataset_idx) 3-tuples. Shapes:
                  - u_input:  (B, T, C, H, W) — T=10 input frames
                  - u_target: (B, C, H, W) — next frame ground truth
                  - dataset_idx: (B,) — optional, discarded if present
            dataset_name: Name of the dataset being evaluated. Used to look
                up the number of meaningful channels in DATASET_CHANNELS.
                Defaults to "unknown" which uses _DEFAULT_CHANNELS=4.

        Returns:
            Mean L2 Relative Error over all batches in the loader as a
            Python float. Returns float('nan') if the loader is empty.
            Values are in [0, ∞) where lower indicates better prediction.
        """
        # Set model to eval mode: disables dropout and batchnorm training behavior.
        # This is critical for reproducible evaluation results.
        self.model.eval()

        # Determine the number of meaningful channels for this dataset.
        # Slice predictions to these channels before computing L2RE to
        # exclude constant-padded channels from the error metric.
        num_channels: int = DATASET_CHANNELS.get(dataset_name, _DEFAULT_CHANNELS)

        # Metric accumulators.
        total_l2re: float = 0.0
        num_batches: int = 0

        # tqdm progress bar for monitoring evaluation progress.
        loader_iter = tqdm(
            loader,
            desc=f"Evaluating {dataset_name}",
            leave=False,
        )

        with torch.no_grad():
            batch: Tuple
            for batch in loader_iter:
                # ----------------------------------------------------------
                # Step 1: Unpack batch — handle 2-tuple and 3-tuple formats
                # ----------------------------------------------------------
                # MultiPDEDataset yields 3-tuples (u_input, u_target, dataset_idx).
                # Single-dataset loaders typically yield 2-tuples (u_input, u_target).
                # Both formats are supported for flexibility.
                u_input: torch.Tensor
                u_target: torch.Tensor

                if len(batch) == 3:
                    u_input = batch[0]
                    u_target = batch[1]
                    # dataset_idx (batch[2]) is discarded — not needed for evaluation
                elif len(batch) == 2:
                    u_input = batch[0]
                    u_target = batch[1]
                else:
                    raise ValueError(
                        f"Unexpected batch format for dataset '{dataset_name}': "
                        f"expected 2 or 3 elements, got {len(batch)}."
                    )

                # ----------------------------------------------------------
                # Step 2: Move tensors to target device
                # ----------------------------------------------------------
                # non_blocking=True enables asynchronous CPU→GPU transfer
                # when the DataLoader uses pin_memory=True.
                u_input = u_input.to(self.device, non_blocking=True)
                u_target = u_target.to(self.device, non_blocking=True)
                # u_input shape:  (B, T, C, H, W)
                # u_target shape: (B, C, H, W)

                # ----------------------------------------------------------
                # Step 3: Forward pass — discard balance loss
                # ----------------------------------------------------------
                # MoEPOT.forward() always returns (u_pred, total_balance_loss).
                # The balance_loss is irrelevant for evaluation and is discarded.
                u_pred: torch.Tensor
                _balance_loss: torch.Tensor
                u_pred, _balance_loss = self.model(u_input)
                # u_pred shape: (B, C, H, W) where C = max_channels = 4

                # ----------------------------------------------------------
                # Step 4: Slice to meaningful channels
                # ----------------------------------------------------------
                # Exclude constant-padded channels (filled with 1.0) and the
                # CFDBench geometry mask channel from the L2RE computation.
                # This ensures fair comparison across datasets with different
                # numbers of physical variables.
                #
                # Example: For NS datasets (1 channel), slice [:, :1, :, :]
                #          For CNS datasets (4 channels), keep all [:, :4, :, :]
                #          For CFDBench (3 channels + mask), slice [:, :3, :, :]
                #
                # Guard against num_channels exceeding actual tensor channels.
                actual_channels: int = min(num_channels, u_pred.shape[1])
                u_pred_sliced: torch.Tensor = u_pred[:, :actual_channels, :, :]
                u_target_sliced: torch.Tensor = u_target[:, :actual_channels, :, :]

                # ----------------------------------------------------------
                # Step 5: Compute L2 Relative Error for this batch
                # ----------------------------------------------------------
                # Delegate to l2_relative_error() from metrics.py.
                # Returns a Python float (mean L2RE over the batch).
                batch_l2re: float = l2_relative_error(u_pred_sliced, u_target_sliced)

                # ----------------------------------------------------------
                # Step 6: Accumulate metrics
                # ----------------------------------------------------------
                total_l2re += batch_l2re
                num_batches += 1

                # Update tqdm postfix with running average.
                loader_iter.set_postfix(
                    l2re=f"{total_l2re / num_batches:.6f}"
                )

        # Restore model to train mode after evaluation.
        # This prevents eval mode from persisting into subsequent training steps.
        self.model.train()

        # Guard against empty loader.
        if num_batches == 0:
            return float("nan")

        return total_l2re / num_batches

    def autoregressive_rollout(
        self,
        u_init: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Auto-regressively predicts num_steps future frames from T initial frames.

        Implements the inference protocol from Appendix B.3:
            "if we take the first 10 steps as our input, we can predict the
            solution x_pred for the next 10 steps."

        At each step, the model predicts the next frame from the current
        T-frame sliding window. The predicted frame is appended to the window
        and the oldest frame is dropped, maintaining a constant window size of T.

        Sliding window update at step i:
            window_i = [u_{i}, u_{i+1}, ..., u_{i+T-1}]
            u_next = model(window_i)
            window_{i+1} = [u_{i+1}, ..., u_{i+T-1}, u_next]

        No noise injection is applied during rollout — this is pure inference.
        Error accumulates over steps as noted in Appendix C.3 (Table 12).

        Args:
            u_init: Initial T-frame input tensor of shape (B, T, C, H, W) where:
                - B: Batch size
                - T: Number of input timesteps = 10 (config.yaml
                  architecture.input_timesteps)
                - C: Number of channels = 4 (config.yaml architecture.max_channels)
                - H: Spatial height = 128 (config.yaml architecture.target_resolution)
                - W: Spatial width = 128
                May be on CPU — will be moved to self.device internally.
            num_steps: Number of future frames to predict. Default 10, matching
                config.yaml evaluation.rollout_steps: 10. For the SWE rollout
                analysis in Appendix C.3, use num_steps=100.

        Returns:
            Predicted rollout sequence of shape (B, num_steps, C, H, W).
            The t-th frame (0-indexed) is the prediction for timestep T+t.
            All frames are on self.device as float32 tensors.
            Returns a tensor with num_steps=0 frames if num_steps <= 0.
        """
        # Set model to eval mode for inference.
        self.model.eval()

        # Move initial frames to target device.
        # u_init may come from CPU if loaded from disk.
        u_init_device: torch.Tensor = u_init.to(self.device)
        # u_init_device shape: (B, T, C, H, W)

        # Initialize the sliding window with the T initial frames.
        # Clone to avoid modifying the original tensor.
        window: torch.Tensor = u_init_device.clone()
        # window shape: (B, T, C, H, W)

        # Collect predicted frames.
        predictions: List[torch.Tensor] = []

        with torch.no_grad():
            step: int
            for step in range(num_steps):
                # ----------------------------------------------------------
                # Step 1: Forward pass — predict next frame from current window
                # ----------------------------------------------------------
                # MoEPOT.forward() takes (B, T, C, H, W) and returns
                # (u_next, balance_loss). Discard balance_loss.
                u_next: torch.Tensor
                _balance: torch.Tensor
                u_next, _balance = self.model(window)
                # u_next shape: (B, C, H, W)

                # ----------------------------------------------------------
                # Step 2: Store prediction
                # ----------------------------------------------------------
                # Unsqueeze time dimension for later concatenation.
                # u_next.unsqueeze(1) shape: (B, 1, C, H, W)
                predictions.append(u_next.unsqueeze(1))

                # ----------------------------------------------------------
                # Step 3: Update sliding window
                # ----------------------------------------------------------
                # Drop the oldest frame (index 0) and append the new prediction.
                # window[:, 1:, :, :, :] shape: (B, T-1, C, H, W)
                # u_next.unsqueeze(1) shape:    (B, 1, C, H, W)
                # Result shape:                 (B, T, C, H, W)
                #
                # Using torch.cat (not in-place) to create a new tensor,
                # avoiding potential issues with gradient graph references
                # even under no_grad context.
                window = torch.cat(
                    [window[:, 1:, :, :, :], u_next.unsqueeze(1)],
                    dim=1,
                )
                # window shape: (B, T, C, H, W) — maintained throughout rollout

        # Restore model to train mode.
        self.model.train()

        # Handle edge case: no steps requested.
        if len(predictions) == 0:
            batch_size: int = u_init_device.shape[0]
            c: int = u_init_device.shape[2]
            h: int = u_init_device.shape[3]
            w: int = u_init_device.shape[4]
            return torch.zeros(
                batch_size, 0, c, h, w,
                dtype=u_init_device.dtype,
                device=self.device,
            )

        # Concatenate all predicted frames along the time dimension.
        # Each element in predictions has shape (B, 1, C, H, W).
        # Result shape: (B, num_steps, C, H, W)
        pred_sequence: torch.Tensor = torch.cat(predictions, dim=1)

        return pred_sequence

    def benchmark_inference_time(
        self,
        loader: DataLoader,
        num_runs: int = _DEFAULT_TIMED_RUNS,
    ) -> float:
        """Measures average single-step inference time in milliseconds.

        Reproduces Table 3 in the paper, which compares inference times across
        DPOT and MoE-POT variants on the NS(1e-5) dataset:
            DPOT-Tiny (7.5M):    5.5 ms
            DPOT-Small (30M):    6.5 ms
            DPOT-Medium (158M): 16.7 ms
            DPOT-Large (493M):  24.3 ms
            MoE-POT-Tiny (17M):  8.8 ms
            MoE-POT-Small (90M): 12.7 ms
            MoE-POT-Medium (288M): 16.6 ms

        Extracts a single batch from the loader and delegates timing to
        compute_inference_time() from metrics.py, which uses CUDA event
        timing for accurate GPU benchmarking.

        From config.yaml (evaluation section):
            inference_warmup_runs: 10   (warmup before timing)
            inference_timed_runs: 100   (number of timed runs)

        Args:
            loader: DataLoader from which a single representative batch is
                extracted for timing. Typically the NS(1e-5) test DataLoader
                as used in Table 3 of the paper. The batch size from the
                loader determines the inference batch size for timing.
            num_runs: Number of timed forward passes. Default 100, matching
                config.yaml evaluation.inference_timed_runs: 100. More runs
                give more stable timing estimates.

        Returns:
            Average inference time per forward pass in milliseconds as a
            Python float. Returns 0.0 if the loader is empty or num_runs <= 0.
        """
        # Set model to eval mode for inference timing.
        # compute_inference_time() also calls model.eval() internally,
        # but we set it here for consistency.
        self.model.eval()

        # Extract a single batch from the loader for timing.
        # next(iter(loader)) gets the first batch without iterating the full loader.
        try:
            batch: Tuple = next(iter(loader))
        except StopIteration:
            # Empty loader — cannot benchmark.
            self.model.train()
            return 0.0

        # Unpack batch — handle both 2-tuple and 3-tuple formats.
        u_input: torch.Tensor
        if len(batch) >= 2:
            u_input = batch[0]
        else:
            self.model.train()
            return 0.0

        # Move input to target device for accurate GPU timing.
        u_input = u_input.to(self.device)
        # u_input shape: (B, T, C, H, W)

        # Delegate timing to compute_inference_time() from metrics.py.
        # Uses CUDA event timing for accurate GPU benchmarking.
        # warmup=_DEFAULT_WARMUP_RUNS=10 (config.yaml: inference_warmup_runs: 10)
        avg_ms: float = compute_inference_time(
            model=self.model,
            input_tensor=u_input,
            num_runs=num_runs,
            warmup=_DEFAULT_WARMUP_RUNS,
            device=self.device,
        )

        # Restore model to train mode after benchmarking.
        self.model.train()

        return avg_ms
