"""
Scaling Analysis for NaViL.

This module implements the scaling experiments described in Section 3.3 of the paper:

1. Scaling up LLMs independently (Fig. 5):
   - LLM sizes: {0.5B, 1.8B, 7B} with fixed 600M visual encoder
   - Shows log-linear decrease in validation loss with LLM size

2. Scaling up Visual Encoders independently (Fig. 6):
   - Encoder sizes: {75M, 150M, 300M, 600M, 1.2B, 2.4B} with fixed LLM
   - Shows diminishing returns beyond optimal encoder size

3. Joint scaling (Fig. 7):
   - Finds optimal encoder size for each LLM size
   - Shows log-linear relationship between optimal encoder size and LLM size

Key finding (Observation 5):
   log(optimal_encoder_size) ∝ log(llm_size)
   i.e., optimal_encoder_size ∝ llm_size^α for some α

Definition of optimal encoder size (from paper):
   Smallest encoder whose loss difference compared to an encoder twice its size
   is less than λ = 1% of the loss with the 75M encoder.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ScalingExperimentResult:
    """Results from a scaling experiment."""
    llm_size_m: float          # LLM size in millions of parameters
    encoder_size_m: float      # Visual encoder size in millions
    validation_loss: float     # Validation loss
    num_training_samples: int  # Number of training samples seen
    model_name: str = ""


class ScalingAnalyzer:
    """
    Analyzes scaling properties of native MLLMs.

    Implements the analysis from Section 3.3 of the NaViL paper.
    """

    # Encoder sizes used in experiments (millions of parameters)
    ENCODER_SIZES_M = [75, 150, 300, 600, 1200, 2400]

    # LLM sizes used in experiments (millions of parameters)
    LLM_SIZES_M = [500, 1800, 7000]

    # Threshold for optimal encoder size (1% of loss with 75M encoder)
    LAMBDA = 0.01

    def find_optimal_encoder_size(
        self,
        results: List[ScalingExperimentResult],
        llm_size_m: float,
        num_training_samples: int,
    ) -> float:
        """
        Find the optimal visual encoder size for a given LLM size.

        Definition from paper:
        "The smallest encoder whose loss difference compared to an encoder
        twice its size is less than λ = 1% of the loss with the 75M encoder."

        Args:
            results: list of experiment results for this LLM size
            llm_size_m: LLM size in millions
            num_training_samples: training data size to evaluate at
        Returns:
            optimal encoder size in millions
        """
        # Filter results for this LLM size and training data size
        relevant = [
            r for r in results
            if r.llm_size_m == llm_size_m
            and r.num_training_samples == num_training_samples
        ]

        if not relevant:
            raise ValueError(f"No results for LLM size {llm_size_m}M")

        # Sort by encoder size
        relevant.sort(key=lambda r: r.encoder_size_m)

        # Get loss with 75M encoder (baseline)
        baseline = next(
            (r for r in relevant if r.encoder_size_m == 75),
            relevant[0]
        )
        baseline_loss = baseline.validation_loss

        # Threshold: λ * baseline_loss
        threshold = self.LAMBDA * baseline_loss

        # Find smallest encoder where doubling gives < threshold improvement
        for i, result in enumerate(relevant[:-1]):
            # Find result with ~2x encoder size
            next_size = result.encoder_size_m * 2
            next_result = next(
                (r for r in relevant if abs(r.encoder_size_m - next_size) / next_size < 0.1),
                None
            )
            if next_result is None:
                continue

            loss_diff = result.validation_loss - next_result.validation_loss
            if loss_diff < threshold:
                return result.encoder_size_m

        # If no threshold found, return largest encoder
        return relevant[-1].encoder_size_m

    def fit_optimal_encoder_scaling(
        self,
        llm_sizes_m: List[float],
        optimal_encoder_sizes_m: List[float],
    ) -> Tuple[float, float]:
        """
        Fit log-linear relationship between optimal encoder size and LLM size.

        From paper (Observation 5):
        log(optimal_encoder_size) = α * log(llm_size) + β

        Args:
            llm_sizes_m: LLM sizes in millions
            optimal_encoder_sizes_m: corresponding optimal encoder sizes
        Returns:
            (alpha, beta): slope and intercept of log-log fit
        """
        log_llm = np.log(llm_sizes_m)
        log_enc = np.log(optimal_encoder_sizes_m)

        # Linear regression in log-log space
        alpha, beta = np.polyfit(log_llm, log_enc, 1)
        return float(alpha), float(beta)

    def predict_optimal_encoder_size(
        self,
        llm_size_m: float,
        alpha: float,
        beta: float,
    ) -> float:
        """
        Predict optimal encoder size for a given LLM size.

        Args:
            llm_size_m: LLM size in millions
            alpha: scaling exponent
            beta: log-space intercept
        Returns:
            predicted optimal encoder size in millions
        """
        log_enc = alpha * math.log(llm_size_m) + beta
        return math.exp(log_enc)

    def analyze_llm_scaling(
        self,
        results: List[ScalingExperimentResult],
        encoder_size_m: float = 600,
    ) -> Dict:
        """
        Analyze LLM scaling law (Fig. 5 in paper).

        Shows that validation loss decreases log-linearly with LLM size.

        Args:
            results: experiment results
            encoder_size_m: fixed encoder size (600M in paper)
        Returns:
            dict with scaling analysis
        """
        # Filter for fixed encoder size
        relevant = [
            r for r in results
            if abs(r.encoder_size_m - encoder_size_m) / encoder_size_m < 0.1
        ]

        # Group by training data size
        data_sizes = sorted(set(r.num_training_samples for r in relevant))
        analysis = {}

        for data_size in data_sizes:
            size_results = [r for r in relevant if r.num_training_samples == data_size]
            size_results.sort(key=lambda r: r.llm_size_m)

            if len(size_results) < 2:
                continue

            # Fit log-linear: loss = a * log(N) + b
            log_sizes = np.log([r.llm_size_m for r in size_results])
            losses = [r.validation_loss for r in size_results]
            slope, intercept = np.polyfit(log_sizes, losses, 1)

            analysis[data_size] = {
                "llm_sizes": [r.llm_size_m for r in size_results],
                "losses": losses,
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(np.corrcoef(log_sizes, losses)[0, 1] ** 2),
            }

        return analysis

    def analyze_encoder_scaling(
        self,
        results: List[ScalingExperimentResult],
        llm_size_m: float = 1800,
    ) -> Dict:
        """
        Analyze visual encoder scaling (Fig. 6 in paper).

        Shows diminishing returns from increasing encoder size.

        Args:
            results: experiment results
            llm_size_m: fixed LLM size (1.8B in paper)
        Returns:
            dict with scaling analysis
        """
        # Filter for fixed LLM size
        relevant = [
            r for r in results
            if abs(r.llm_size_m - llm_size_m) / llm_size_m < 0.1
        ]

        # Group by training data size
        data_sizes = sorted(set(r.num_training_samples for r in relevant))
        analysis = {}

        for data_size in data_sizes:
            size_results = [r for r in relevant if r.num_training_samples == data_size]
            size_results.sort(key=lambda r: r.encoder_size_m)

            if len(size_results) < 2:
                continue

            # Compute marginal gains
            marginal_gains = []
            for i in range(1, len(size_results)):
                gain = size_results[i-1].validation_loss - size_results[i].validation_loss
                marginal_gains.append(gain)

            analysis[data_size] = {
                "encoder_sizes": [r.encoder_size_m for r in size_results],
                "losses": [r.validation_loss for r in size_results],
                "marginal_gains": marginal_gains,
            }

        return analysis

    def compute_optimal_encoder_sizes(
        self,
        results: List[ScalingExperimentResult],
        data_size: int,
    ) -> Dict[float, float]:
        """
        Compute optimal encoder size for each LLM size (Fig. 7 in paper).

        Args:
            results: all experiment results
            data_size: training data size to evaluate at
        Returns:
            dict mapping LLM size -> optimal encoder size
        """
        optimal_sizes = {}
        for llm_size in self.LLM_SIZES_M:
            try:
                opt_enc = self.find_optimal_encoder_size(results, llm_size, data_size)
                optimal_sizes[llm_size] = opt_enc
            except ValueError:
                pass
        return optimal_sizes

    def generate_scaling_report(
        self,
        results: List[ScalingExperimentResult],
    ) -> str:
        """Generate a text report of scaling analysis."""
        lines = ["=" * 60, "NaViL Scaling Analysis Report", "=" * 60, ""]

        # LLM scaling
        lines.append("1. LLM Scaling (fixed 600M encoder)")
        lines.append("-" * 40)
        llm_analysis = self.analyze_llm_scaling(results)
        for data_size, info in sorted(llm_analysis.items()):
            lines.append(f"  Data size: {data_size/1e6:.0f}M samples")
            for size, loss in zip(info["llm_sizes"], info["losses"]):
                lines.append(f"    LLM {size/1000:.1f}B: loss={loss:.4f}")
            lines.append(f"  Log-linear fit: slope={info['slope']:.4f}, R²={info['r_squared']:.4f}")
        lines.append("")

        # Encoder scaling
        lines.append("2. Visual Encoder Scaling (fixed 1.8B LLM)")
        lines.append("-" * 40)
        enc_analysis = self.analyze_encoder_scaling(results)
        for data_size, info in sorted(enc_analysis.items()):
            lines.append(f"  Data size: {data_size/1e6:.0f}M samples")
            for size, loss in zip(info["encoder_sizes"], info["losses"]):
                lines.append(f"    Encoder {size}M: loss={loss:.4f}")
        lines.append("")

        # Optimal encoder sizes
        lines.append("3. Optimal Encoder Size vs LLM Size")
        lines.append("-" * 40)
        if results:
            max_data = max(r.num_training_samples for r in results)
            optimal = self.compute_optimal_encoder_sizes(results, max_data)
            if len(optimal) >= 2:
                llm_sizes = sorted(optimal.keys())
                enc_sizes = [optimal[s] for s in llm_sizes]
                alpha, beta = self.fit_optimal_encoder_scaling(llm_sizes, enc_sizes)
                lines.append(f"  Log-log fit: log(enc) = {alpha:.3f} * log(llm) + {beta:.3f}")
                lines.append(f"  (optimal_enc ∝ llm^{alpha:.3f})")
                for llm, enc in zip(llm_sizes, enc_sizes):
                    lines.append(f"  LLM {llm/1000:.1f}B -> optimal encoder: {enc:.0f}M")

        return "\n".join(lines)


class VisualEncoderArchitectureAnalyzer:
    """
    Analyzes the optimal depth/width configuration for visual encoders.

    From Section 3.2.3 of the paper:
    - Parameter count: N ≈ 12 * d * w^2
    - Depth d ∈ {3, 6, 12, 24, 48}
    - Width w ∈ {4096, 2880, 2048, 1472, 1024} (for 600M budget)
    """

    # Configurations explored in the paper (depth, width) for 600M budget
    CONFIGS_600M = [
        (3, 4096),
        (6, 2880),
        (12, 2048),
        (24, 1472),
        (48, 1024),
    ]

    @staticmethod
    def compute_param_count(depth: int, width: int) -> int:
        """
        Approximate parameter count: N ≈ 12 * d * w^2
        (from Kaplan et al. scaling laws paper, cited in NaViL)
        """
        return 12 * depth * width * width

    @staticmethod
    def get_configs_for_budget(
        budget_m: float,
        depths: List[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Get (depth, width) configurations for a given parameter budget.

        Args:
            budget_m: parameter budget in millions
            depths: list of depths to try
        Returns:
            list of (depth, width) tuples
        """
        if depths is None:
            depths = [3, 6, 12, 24, 48]

        budget = budget_m * 1e6
        configs = []

        for d in depths:
            # N ≈ 12 * d * w^2 => w = sqrt(N / (12 * d))
            w = int(math.sqrt(budget / (12 * d)))
            # Round to nearest multiple of 64
            w = max(64, round(w / 64) * 64)
            # Adjust num_heads
            num_heads = max(1, w // 64)
            configs.append((d, w))

        return configs

    @staticmethod
    def analyze_depth_width_tradeoff(
        results: Dict[Tuple[int, int], List[Tuple[int, float]]],
    ) -> str:
        """
        Analyze depth vs width tradeoff from experiment results.

        Args:
            results: dict mapping (depth, width) -> list of (num_samples, loss)
        Returns:
            analysis report string
        """
        lines = ["Depth vs Width Analysis", "-" * 40]

        for (d, w), data in sorted(results.items()):
            n_params = VisualEncoderArchitectureAnalyzer.compute_param_count(d, w)
            lines.append(f"d={d:3d}, w={w:4d} ({n_params/1e6:.0f}M params):")
            for n_samples, loss in sorted(data):
                lines.append(f"  {n_samples/1e6:.0f}M samples: loss={loss:.4f}")

        return "\n".join(lines)
