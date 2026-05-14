## Code: evaluation/evaluate.py

```python
## evaluation/evaluate.py
"""Evaluation orchestrator for the generative PDE foundation model.

Reproduces all quantitative results from the paper:
  - Table 1: P2VAE reconstruction quality (L2RE, VRMSE) across 16 sub-datasets
  - Table 2: Few-shot adaptation on Kolmogorov turbulence (L2RE, VRMSE)
  - Table 3: Long-term rollout errors (L2RE at steps 1, 5, 10, last, average)
  - Figure 3: Ensemble variance as a function of bridge parameter k_3

All evaluation methods operate under torch.no_grad() and set both models to
eval mode defensively at the start of each call. No gradient computation
occurs in this file.

Configuration alignment (config.yaml):
    evaluation.metrics: ['l2_relative_error', 'vrmse']
    evaluation.rollout.report_steps: [1, 5, 10]
    evaluation.rollout.datasets: ['pa_ns', 'pb_cns_low', 'pb_cns_high']
    evaluation.rollout.k_vals: [1.0, 1.0, 1.0, 1.0]
    ensemble.batch_size: 32
    ensemble.k3_values: [0.0, 0.3, 0.6, 0.9]
    ensemble.eval_dataset: 'pa_ns'
    fmt.inference.euler_steps: 100
"""

import json
import logging
import os
import warnings
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from evaluation.metrics import Metrics
from inference.rollout import AutoregressiveRollout
from inference.sampler import EulerSampler
from models.fmt import FMT
from models.p2vae import P2VAE

logger = logging.getLogger(__name__)


class Evaluator:
    """Orchestrates all evaluation experiments for the generative PDE foundation model.

    Provides four evaluation methods corresponding to the paper's experiments:
      - evaluate_reconstruction: Table 1 (P2VAE compression quality)
      - evaluate_rollout: Table 3 (long-horizon autoregressive prediction)
      - evaluate_ensemble: Figure 3 (ensemble diversity vs. k_3)
      - evaluate_fewshot: Table 2 (Kolmogorov turbulence adaptation)

    All methods are stateless with respect to model weights — they only read
    model outputs and accumulate scalar metrics. The models must be loaded
    with the correct checkpoint weights before being passed to __init__.

    Attributes:
        fmt: FMT model for velocity prediction. Set to eval mode in __init__.
        p2vae: P2VAE model for encoding/decoding. Set to eval mode in __init__.
        config: Full configuration dictionary from config.yaml.
        device: Inferred device from p2vae parameters.
        sampler: EulerSampler with num_steps=100 (config: fmt.inference.euler_steps).
        rollout: AutoregressiveRollout wrapping fmt, p2vae, and sampler.
        metrics: Metrics instance for L2RE, VRMSE, and batch_variance.
        report_steps: Rollout steps to report (config: evaluation.rollout.report_steps).
        k3_values: Bridge parameter values for ensemble sweep (config: ensemble.k3_values).
        ensemble_batch_size: Ensemble size for Figure 3 (config: ensemble.batch_size).
    """

    def __init__(
        self,
        fmt: FMT,
        p2vae: P2VAE,
        config: Dict,
    ) -> None:
        """Initialize the Evaluator with pretrained models and configuration.

        Sets both models to eval mode immediately. Instantiates EulerSampler,
        AutoregressiveRollout, and Metrics. Extracts frequently used config
        values as instance attributes.

        Args:
            fmt: Pretrained FMT model. Must already be loaded with the correct
                checkpoint weights. Will be set to eval mode.
            p2vae: Pretrained P2VAE model. Must already be loaded with the
                correct checkpoint weights. Will be set to eval mode.
            config: Full configuration dictionary loaded from config.yaml.
                Must contain 'fmt', 'evaluation', and 'ensemble' top-level keys.
        """
        self.config: Dict = config

        # Infer device from p2vae parameters. This avoids hardcoding device
        # strings and works correctly for both single-GPU and CPU evaluation.
        try:
            self.device: torch.device = next(p2vae.parameters()).device
        except StopIteration:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.warning(
                "Could not infer device from p2vae parameters. "
                "Defaulting to %s.",
                self.device,
            )

        # Store model references and set to eval mode.
        # Defensive eval() calls here; each evaluation method also calls eval()
        # at its start in case the models were switched to train mode externally.
        self.fmt: FMT = fmt.to(self.device)
        self.p2vae: P2VAE = p2vae.to(self.device)
        self.fmt.eval()
        self.p2vae.eval()

        # Instantiate EulerSampler with num_steps from config.
        # config.yaml: fmt.inference.euler_steps = 100 → dt = 0.01
        euler_steps: int = int(
            config.get("fmt", {})
            .get("inference", {})
            .get("euler_steps", 100)
        )
        self.sampler: EulerSampler = EulerSampler(num_steps=euler_steps)

        # Instantiate AutoregressiveRollout.
        # window_size=4 matches config: data.trajectory_length = 4.
        self.rollout: AutoregressiveRollout = AutoregressiveRollout(
            fmt=self.fmt,
            p2vae=self.p2vae,
            sampler=self.sampler,
            window_size=4,
        )

        # Instantiate Metrics utility.
        self.metrics: Metrics = Metrics()

        # Extract frequently used config values as instance attributes.
        # config.yaml: evaluation.rollout.report_steps = [1, 5, 10]
        self.report_steps: List[int] = list(
            config.get("evaluation", {})
            .get("rollout", {})
            .get("report_steps", [1, 5, 10])
        )

        # config.yaml: ensemble.k3_values = [0.0, 0.3, 0.6, 0.9]
        self.k3_values: List[float] = [
            float(v)
            for v in config.get("ensemble", {}).get("k3_values", [0.0, 0.3, 0.6, 0.9])
        ]

        # config.yaml: ensemble.batch_size = 32
        self.ensemble_batch_size: int = int(
            config.get("ensemble", {}).get("batch_size", 32)
        )

        logger.info(
            "Evaluator initialized: device=%s, euler_steps=%d, "
            "report_steps=%s, k3_values=%s, ensemble_batch_size=%d",
            self.device,
            euler_steps,
            self.report_steps,
            self.k3_values,
            self.ensemble_batch_size,
        )

    # ------------------------------------------------------------------
    # Public evaluation methods
    # ------------------------------------------------------------------

    def evaluate_reconstruction(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Measure P2VAE compression quality via encode-decode reconstruction.

        Reproduces the P2VAE rows in Table 1 of the paper. Encodes each frame
        to its posterior mean (deterministic latent, no sampling) and decodes
        back to pixel space. Computes L2RE and VRMSE between the original and
        reconstructed fields.

        The 12× spatial compression (c3p128 → c16p16 → c3p128) introduces
        information loss quantified by these metrics. The paper reports these
        values to establish the reconstruction quality baseline before FMT
        training.

        Args:
            dataloader: DataLoader over a PDEUnifiedDataset (or single-dataset
                H5Dataset) for the split to evaluate. Each batch must contain
                'frames' of shape (B, T, 3, 128, 128) where T >= 1.
                The method processes all T frames per trajectory independently.

        Returns:
            Dictionary with keys:
                'l2re': Mean L2 relative error over all frames in the dataset.
                    From config: evaluation.metrics[0] = 'l2_relative_error'.
                'vrmse': Mean variance-normalized RMSE over all frames.
                    From config: evaluation.metrics[1] = 'vrmse'.
            Both values are Python floats. Returns {'l2re': 0.0, 'vrmse': 0.0}
            if the dataloader is empty.
        """
        # Defensive eval mode: ensure models are in eval state.
        self.fmt.eval()
        self.p2vae.eval()

        # Check for empty dataloader.
        if len(dataloader) == 0:
            logger.warning(
                "evaluate_reconstruction: dataloader is empty. "
                "Returning zero metrics."
            )
            return {"l2re": 0.0, "vrmse": 0.0}

        # Accumulators for weighted averaging across batches.
        # We weight by the number of samples in each batch to handle
        # variable-size last batches correctly.
        total_l2re: float = 0.0
        total_vrmse: float = 0.0
        num_samples: int = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Extract frames tensor from batch dict.
                # Shape: (B, T, 3, 128, 128), dtype float32.
                frames: Tensor = batch["frames"]  # type: ignore[assignment]
                frames = frames.to(self.device, non_blocking=True)

                b: int = frames.shape[0]
                t: int = frames.shape[1]

                # Reshape (B, T, 3, 128, 128) → (B*T, 3, 128, 128).
                # The VAE is a spatial autoencoder — processes each frame
                # independently regardless of temporal ordering.
                x_flat: Tensor = frames.reshape(b * t, 3, 128, 128)

                # Encode to posterior mean (deterministic latent, no sampling).
                # get_latent() returns mu — consistent with FMT training
                # (shared knowledge #3: "p2vae.get_latent(x) always returns mu").
                # Shape: (B*T, 16, 16, 16)
                with torch.cuda.amp.autocast(
                    enabled=torch.cuda.is_available()
                ):
                    z: Tensor = self.p2vae.get_latent(x_flat)

                    # Decode latent back to pixel space.
                    # Shape: (B*T, 3, 128, 128)
                    x_hat: Tensor = self.p2vae.decode(z)

                # Compute metrics in float32 for numerical stability.
                x_f32: Tensor = x_flat.float()
                x_hat_f32: Tensor = x_hat.float()

                # Per-batch mean metrics (reduce='mean' over B*T samples).
                batch_l2re: Tensor = self.metrics.l2_relative_error(
                    pred=x_hat_f32,
                    target=x_f32,
                    reduce="mean",
                )
                batch_vrmse: Tensor = self.metrics.vrmse(
                    pred=x_hat_f32,
                    target=x_f32,
                    reduce="mean",
                )

                # Weighted accumulation: multiply mean by sample count.
                n_batch: int = b * t
                total_l2re += float(batch_l2re.item()) * n_batch
                total_vrmse += float(batch_vrmse.item()) * n_batch
                num_samples += n_batch

                if (batch_idx + 1) % 50 == 0:
                    logger.debug(
                        "evaluate_reconstruction: processed %d batches, "
                        "%d samples so far.",
                        batch_idx + 1,
                        num_samples,
                    )

        # Guard against empty accumulation.
        if num_samples == 0:
            logger.warning(
                "evaluate_reconstruction: no samples processed. "
                "Returning zero metrics."
            )
            return {"l2re": 0.0, "vrmse": 0.0}

        mean_l2re: float = total_l2re / num_samples
        mean_vrmse: float = total_vrmse / num_samples

        logger.info(
            "evaluate_reconstruction: l2re=%.4f, vrmse=%.4f "
            "(over %d samples)",
            mean_l2re,
            mean_vrmse,
            num_samples,
        )

        return {"l2re": mean_l2re, "vrmse": mean_vrmse}

    def evaluate_rollout(
        self,
        dataloader: DataLoader,
        rollout_steps: int = 14,
        dataset_name: str = "",
    ) -> Dict[str, float]:
        """Evaluate long-horizon autoregressive prediction quality.

        Reproduces Table 3 of the paper. Runs deterministic rollout (all k=1)
        for rollout_steps autoregressive steps and computes L2RE at each step.
        Reports metrics at steps 1, 5, 10, the last step, and the average
        across all steps.

        The dataloader must provide batches with both 'frames' (initial 4-frame
        condition) and 'targets' (ground truth future frames). If 'targets' is
        not present in the batch, the method attempts to use the last T-4 frames
        of 'frames' as targets (assuming the dataloader provides long trajectories).

        Args:
            dataloader: DataLoader providing trajectory batches. Each batch
                must contain:
                    'frames': (B, 4, 3, 128, 128) — initial condition frames.
                    'targets': (B, rollout_steps, 3, 128, 128) — ground truth
                        future frames. If absent, the last rollout_steps frames
                        of a longer 'frames' tensor are used as targets.
            rollout_steps: Number of autoregressive prediction steps. From
                config: evaluation.rollout.report_steps implies at least 10
                steps. Default 14 covers PA-NS trajectory length.
            dataset_name: Optional dataset identifier for prefixing result keys
                (e.g., 'pa_ns'). If empty, keys are unprefixed.

        Returns:
            Dictionary with L2RE at each reported step and aggregate metrics.
            Key format (with dataset_name='pa_ns'):
                'pa_ns_step_1': L2RE at step 1
                'pa_ns_step_5': L2RE at step 5
                'pa_ns_step_10': L2RE at step 10
                'pa_ns_last_step': L2RE at the last step (step rollout_steps)
                'pa_ns_average': Mean L2RE across all steps
            Without dataset_name prefix:
                'step_1', 'step_5', 'step_10', 'last_step', 'average'
            Returns all-zero dict if dataloader is empty or rollout_steps=0.
        """
        # Defensive eval mode.
        self.fmt.eval()
        self.p2vae.eval()

        # Determine k_vals from config: all k=1 for deterministic prediction.
        # config.yaml: evaluation.rollout.k_vals = [1.0, 1.0, 1.0, 1.0]
        k_vals: List[float] = [
            float(v)
            for v in self.config.get("evaluation", {})
            .get("rollout", {})
            .get("k_vals", [1.0, 1.0, 1.0, 1.0])
        ]

        # Handle trivial cases.
        if rollout_steps <= 0:
            logger.warning(
                "evaluate_rollout: rollout_steps=%d <= 0. "
                "Returning zero metrics.",
                rollout_steps,
            )
            return self._build_rollout_results(
                {s: 0.0 for s in range(1, rollout_steps + 1)},
                rollout_steps,
                dataset_name,
            )

        if len(dataloader) == 0:
            logger.warning(
                "evaluate_rollout: dataloader is empty. "
                "Returning zero metrics."
            )
            return self._build_rollout_results(
                {s: 0.0 for s in range(1, rollout_steps + 1)},
                rollout_steps,
                dataset_name,
            )

        # Per-step accumulators: step_errors[s] = list of per-batch mean L2RE
        # at step s (1-indexed). We accumulate weighted sums for correct averaging.
        step_total_l2re: Dict[int, float] = {
            s: 0.0 for s in range(1, rollout_steps + 1)
        }
        step_num_samples: Dict[int, int] = {
            s: 0 for s in range(1, rollout_steps + 1)
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Extract initial condition frames.
                # Shape: (B, 4, 3, 128, 128) or (B, T, 3, 128, 128) with T >= 4.
                frames: Tensor = batch["frames"]  # type: ignore[assignment]
                frames = frames.to(self.device, non_blocking=True)

                b: int = frames.shape[0]
                t_total: int = frames.shape[1]

                # Extract initial 4-frame condition.
                if t_total < 4:
                    logger.warning(
                        "evaluate_rollout: batch has only %d frames "
                        "(need >= 4). Skipping batch %d.",
                        t_total,
                        batch_idx,
                    )
                    continue

                initial_frames: Tensor = frames[:, :4]  # (B, 4, 3, 128, 128)

                # Determine ground truth targets.
                # Priority: explicit 'targets' key > trailing frames in 'frames'.
                targets: Optional[Tensor] = None

                if "targets" in batch:
                    targets = batch["targets"]  # type: ignore[assignment]
                    targets = targets.to(self.device, non_blocking=True)
                    # Shape: (B, rollout_steps, 3, 128, 128)
                elif t_total >= 4 + rollout_steps:
                    # Use frames 4..4+rollout_steps as targets.
                    targets = frames[:, 4 : 4 + rollout_steps]
                    # Shape: (B, rollout_steps, 3, 128, 128)
                else:
                    # Not enough frames for the requested rollout_steps.
                    available_steps: int = t_total - 4
                    if available_steps <= 0:
                        logger.warning(
                            "evaluate_rollout: batch %d has %d total frames "
                            "but needs at least %d for rollout_steps=%d. "
                            "Skipping.",
                            batch_idx,
                            t_total,
                            4 + rollout_steps,
                            rollout_steps,
                        )
                        continue
                    # Truncate to available steps.
                    targets = frames[:, 4 : 4 + available_steps]
                    effective_steps: int = available_steps
                    logger.debug(
                        "evaluate_rollout: batch %d truncated to %d steps "
                        "(requested %d).",
                        batch_idx,
                        effective_steps,
                        rollout_steps,
                    )
                    # Run rollout for available steps only.
                    with torch.cuda.amp.autocast(
                        enabled=torch.cuda.is_available()
                    ):
                        pred_traj: Tensor = self.rollout.rollout(
                            initial_frames=initial_frames,
                            num_steps=effective_steps,
                            k_vals=k_vals,
                            device=self.device,
                        )
                    # Accumulate metrics for available steps.
                    self._accumulate_step_metrics(
                        pred_traj=pred_traj,
                        targets=targets,
                        effective_steps=effective_steps,
                        b=b,
                        step_total_l2re=step_total_l2re,
                        step_num_samples=step_num_samples,
                    )
                    continue

                # Run autoregressive rollout for rollout_steps.
                with torch.cuda.amp.autocast(
                    enabled=torch.cuda.is_available()
                ):
                    pred_traj = self.rollout.rollout(
                        initial_frames=initial_frames,
                        num_steps=rollout_steps,
                        k_vals=k_vals,
                        device=self.device,
                    )
                # pred_traj shape: (B, rollout_steps, 3, 128, 128)

                # Accumulate per-step metrics.
                self._accumulate_step_metrics(
                    pred_traj=pred_traj,
                    targets=targets,
                    effective_steps=rollout_steps,
                    b=b,
                    step_total_l2re=step_total_l2re,
                    step_num_samples=step_num_samples,
                )

                if (batch_idx + 1) % 20 == 0:
                    logger.debug(
                        "evaluate_rollout [%s]: processed %d batches.",
                        dataset_name or "unknown",
                        batch_idx + 1,
                    )

        # Compute mean L2RE per step.
        mean_step_l2re: Dict[int, float] = {}
        for s in range(1, rollout_steps + 1):
            n: int = step_num_samples[s]
            if n > 0:
                mean_step_l2re[s] = step_total_l2re[s] / n
            else:
                mean_step_l2re[s] = float("nan")
                logger.warning(
                    "evaluate_rollout: no samples accumulated for step %d "
                    "(dataset=%s).",
                    s,
                    dataset_name or "unknown",
                )

        # Build and return the results dict.
        results: Dict[str, float] = self._build_rollout_results(
            mean_step_l2re, rollout_steps, dataset_name
        )

        logger.info(
            "evaluate_rollout [%s]: step_1=%.4f, step_5=%.4f, "
            "step_10=%.4f, last=%.4f, avg=%.4f",
            dataset_name or "unknown",
            results.get(self._key("step_1", dataset_name), float("nan")),
            results.get(self._key("step_5", dataset_name), float("nan")),
            results.get(self._key("step_10", dataset_name), float("nan")),
            results.get(self._key("last_step", dataset_name), float("nan")),
            results.get(self._key("average", dataset_name), float("nan")),
        )

        return results

    def evaluate_ensemble(
        self,
        dataloader: DataLoader,
        k3_values: Optional[List[float]] = None,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Measure ensemble diversity as a function of bridge parameter k_3.

        Reproduces Figure 3 of the paper. Generates a 32-sample ensemble of
        next-step predictions for a single trajectory at different k_3 noise
        levels and computes the average batch-wise variance.

        The variance is a decreasing function of k_3:
            k_3=0.0: pure noise initialization → maximum diversity
            k_3=0.9: mostly clean initialization → low diversity
            k_3=1.0: deterministic → zero diversity (all members identical)

        From the paper (Section 4.4): "We sampled one trajectory from
        PDEArena-NS and tested it on the FMT-B-42M model to generate a
        32-batch size ensemble."

        Args:
            dataloader: DataLoader providing trajectory batches. Only the
                first batch (first trajectory) is used, consistent with the
                paper's single-trajectory evaluation. Each batch must contain
                'frames' of shape (B, 4, 3, 128, 128).
            k3_values: List of k_3 values to sweep. From config:
                ensemble.k3_values = [0.0, 0.3, 0.6, 0.9]. Defaults to
                self.k3_values if None.
            batch_size: Number of ensemble members per k_3 value. From config:
                ensemble.batch_size = 32.

        Returns:
            Dictionary mapping k_3 values to their ensemble variance scalars:
                'k3_0.0': variance at k_3=0.0 (maximum diversity)
                'k3_0.3': variance at k_3=0.3
                'k3_0.6': variance at k_3=0.6
                'k3_0.9': variance at k_3=0.9 (minimum diversity)
            Returns empty dict if the dataloader is empty.
        """
        # Defensive eval mode.
        self.fmt.eval()
        self.p2vae.eval()

        # Apply defaults from config.
        if k3_values is None:
            k3_values = self.k3_values

        if len(dataloader) == 0:
            logger.warning(
                "evaluate_ensemble: dataloader is empty. "
                "Returning empty results."
            )
            return {}

        # Extract a single trajectory (first batch, first sample).
        # The paper uses one trajectory from PDEArena-NS.
        initial_frames: Optional[Tensor] = None

        with torch.no_grad():
            for batch in dataloader:
                frames: Tensor = batch["frames"]  # type: ignore[assignment]
                frames = frames.to(self.device, non_blocking=True)

                # Take only the first sample from the batch.
                # Shape: (1, 4, 3, 128, 128)
                initial_frames = frames[:1]
                break  # Only need one trajectory.

        if initial_frames is None:
            logger.warning(
                "evaluate_ensemble: could not extract initial frames. "
                "Returning empty results."
            )
            return {}

        results: Dict[str, float] = {}

        with torch.no_grad():
            for k3 in k3_values:
                logger.info(
                    "evaluate_ensemble: generating ensemble for k3=%.2f "
                    "(batch_size=%d)...",
                    k3,
                    batch_size,
                )

                # Generate ensemble of next-step predictions.
                # generate_ensemble returns (batch_size, 1, 3, 128, 128).
                with torch.cuda.amp.autocast(
                    enabled=torch.cuda.is_available()
                ):
                    ensemble_traj: Tensor = self.rollout.generate_ensemble(
                        initial_frames=initial_frames,
                        k3=k3,
                        batch_size=batch_size,
                    )
                # ensemble_traj shape: (batch_size, 1, 3, 128, 128)

                # Squeeze the time dimension to get (batch_size, 3, 128, 128).
                # batch_variance expects (B, C, H, W) where B = ensemble members.
                ensemble_frames: Tensor = ensemble_traj[:, 0].float()
                # Shape: (batch_size, 3, 128, 128)

                # Compute average per-pixel variance across ensemble members.
                variance: Tensor = self.metrics.batch_variance(ensemble_frames)
                variance_val: float = float(variance.item())

                # Key format: 'k3_0.0', 'k3_0.3', etc.
                key: str = f"k3_{k3:.1f}"
                results[key] = variance_val

                logger.info(
                    "evaluate_ensemble: k3=%.2f → variance=%.6f",
                    k3,
                    variance_val,
                )

        return results

    def evaluate_fewshot(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate the finetuned model on the Kolmogorov turbulence test set.

        Reproduces Table 2 of the paper. Computes both reconstruction quality
        (P2VAE encode-decode) and one-step prediction quality (FMT rollout)
        on the Kolmogorov turbulence test set (500 trajectories at Re=222).

        The finetuned model (P2VAE + FMT-B-42M) is evaluated after 5k steps
        of joint fine-tuning on 200 training trajectories. The models passed
        to Evaluator.__init__ must be the finetuned versions.

        Args:
            dataloader: DataLoader over the Kolmogorov test set (500 trajectories
                from config: finetune.test_trajectories=500). Each batch must
                contain 'frames' of shape (B, T, 3, 128, 128) with T >= 5
                (4 initial frames + 1 target frame for one-step prediction).
                If T < 5, only reconstruction metrics are computed.

        Returns:
            Dictionary with reconstruction and prediction metrics:
                'recon_l2re': Mean L2RE for P2VAE reconstruction.
                'recon_vrmse': Mean VRMSE for P2VAE reconstruction.
                'pred_l2re': Mean L2RE for one-step FMT prediction.
                'pred_vrmse': Mean VRMSE for one-step FMT prediction.
            If prediction targets are unavailable, 'pred_l2re' and 'pred_vrmse'
            are set to float('nan').
        """
        # Defensive eval mode.
        self.fmt.eval()
        self.p2vae.eval()

        if len(dataloader) == 0:
            logger.warning(
                "evaluate_fewshot: dataloader is empty. "
                "Returning zero metrics."
            )
            return {
                "recon_l2re": 0.0,
                "recon_vrmse": 0.0,
                "pred_l2re": float("nan"),
                "pred_vrmse": float("nan"),
            }

        # Accumulators for reconstruction metrics.
        total_recon_l2re: float = 0.0
        total_recon_vrmse: float = 0.0
        num_recon_samples: int = 0

        # Accumulators for prediction metrics.
        total_pred_l2re: float = 0.0
        total_pred_vrmse: float = 0.0
        num_pred_samples: int = 0

        # k_vals for deterministic one-step prediction (all k=1).
        k_vals_det: List[float] = [1.0, 1.0, 1.0, 1.0]

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                frames: Tensor = batch["frames"]  # type: ignore[assignment]
                frames = frames.to(self.device, non_blocking=True)

                b: int = frames.shape[0]