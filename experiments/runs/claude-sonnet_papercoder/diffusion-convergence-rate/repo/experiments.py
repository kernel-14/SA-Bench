## experiments.py
"""Experiments module for reproducing the numerical experiments from
Appendix A of "Instance-dependent Convergence Theory for Diffusion Models".

This module orchestrates the full convergence sweep across all three paper
configurations (d=10/k=10, d=100/k=10, d=500/k=100) and all T values,
delegating all mathematical computation to the specialized modules.

The main entry point is Experiments.run_all(), which returns a structured
dict of KL and TV results suitable for plotting by plots.py.
"""

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from config import Config
from gaussian_tracker import GaussianTracker
from metrics import Metrics
from sampler import Sampler
from schedule import Schedule
from score import GaussianTarget


class Experiments:
    """Orchestrates convergence experiments for the randomized midpoint sampler.

    Iterates over a list of Config objects and a range of T values, running
    the full analytical Gaussian propagation pipeline for each (cfg, T) pair.
    Collects KL divergence and TV distance results for downstream plotting.

    Attributes:
        configs: List of Config objects, one per experimental configuration.
        results: Dict accumulating results from run_all(). Keys are config
            labels (e.g., 'fig2a', 'fig2b', 'fig2c'). Populated by run_all().
    """

    def __init__(self, configs: List[Config]) -> None:
        """Initialise the Experiments orchestrator.

        Args:
            configs: List of Config objects defining the experimental
                configurations to run. Must contain at least one Config.

        Raises:
            TypeError: If configs is not a list.
            ValueError: If configs is empty.
        """
        if not isinstance(configs, list):
            raise TypeError(
                f"configs must be a list, got {type(configs).__name__}"
            )
        if len(configs) == 0:
            raise ValueError("configs must contain at least one Config object.")

        self.configs: List[Config] = configs
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_all(self) -> Dict[str, Any]:
        """Run convergence experiments for all configurations.

        Iterates over self.configs, calls run_convergence_experiment for
        each, and stores results under the config label key.

        Returns:
            Dict mapping config label -> result dict. Each result dict has:
                'T_values'  : List[int]   — T values swept.
                'kl_values' : List[float] — KL divergence at each T.
                'tv_values' : List[float] — TV distance bound at each T.
                'config'    : Config      — The Config object used.

        Example:
            {
                'fig2a': {'T_values': [50, ...], 'kl_values': [...],
                          'tv_values': [...], 'config': cfg_a},
                'fig2b': {...},
                'fig2c': {...},
            }
        """
        self.results = {}

        for cfg in tqdm(self.configs, desc="Configurations", unit="cfg"):
            # Determine result key: prefer cfg.label, fall back to d/k string
            key: str = cfg.label if cfg.label else f"d{cfg.d}_k{cfg.k_active}"

            print(f"\n{'='*60}")
            print(f"Running experiment: {key}")
            print(cfg.summary())
            print(f"{'='*60}")

            result: Dict[str, Any] = self.run_convergence_experiment(cfg)
            self.results[key] = result

            # Print summary statistics for this configuration
            self._print_result_summary(key, result)

        return self.results

    def run_convergence_experiment(self, cfg: Config) -> Dict[str, Any]:
        """Run the convergence sweep for a single configuration.

        For each T in cfg.T_values, builds the schedule, target, tracker,
        and sampler, runs the analytical propagation, and computes KL/TV.

        Args:
            cfg: Configuration object specifying d, k_active, K, T_values,
                c0, c1, seed, device, and figure_dir.

        Returns:
            Dict with keys:
                'T_values'  : List[int]   — T values swept (same as cfg.T_values).
                'kl_values' : List[float] — KL(p_{Y_K} || q_K) at each T.
                'tv_values' : List[float] — TV upper bound at each T (Pinsker).
                'config'    : Config      — The Config object used.
        """
        T_values_out: List[int] = []
        kl_values_out: List[float] = []
        tv_values_out: List[float] = []

        label: str = cfg.label if cfg.label else f"d{cfg.d}_k{cfg.k_active}"

        for T in tqdm(
            cfg.T_values,
            desc=f"  T sweep [{label}]",
            unit="T",
            leave=True,
        ):
            kl_val: float
            tv_val: float
            kl_val, tv_val = self._single_T_run(T, cfg)

            T_values_out.append(T)
            kl_values_out.append(kl_val)
            tv_values_out.append(tv_val)

            # Monotonicity check: KL should decrease as T increases
            if len(kl_values_out) >= 2:
                prev_kl: float = kl_values_out[-2]
                if (
                    not math.isnan(kl_val)
                    and not math.isnan(prev_kl)
                    and kl_val > prev_kl * 2.0  # allow some tolerance
                ):
                    warnings.warn(
                        f"[{label}] KL increased from T={T_values_out[-2]} "
                        f"(KL={prev_kl:.4e}) to T={T} (KL={kl_val:.4e}). "
                        f"This may indicate numerical issues or insufficient T.",
                        UserWarning,
                        stacklevel=2,
                    )

        return {
            "T_values": T_values_out,
            "kl_values": kl_values_out,
            "tv_values": tv_values_out,
            "config": cfg,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _single_T_run(self, T: int, cfg: Config) -> Tuple[float, float]:
        """Run the full pipeline for a single (T, cfg) pair.

        Encapsulates steps 1–8 of the convergence experiment:
            1. Build Schedule(T, K, c0, c1)
            2. Build GaussianTarget(d, k_active, seed, device)
            3. Build GaussianTracker(d, device)
            4. Build Sampler(K, device)
            5. Run sampler: (mean_K, cov_K) = sampler.run(T, schedule, target, tracker)
            6. Get reference: (mu_ref, cov_ref) = sampler.get_reference_distribution(...)
            7. Compute KL = Metrics.kl_gaussian(mean_K, cov_K, mu_ref, cov_ref)
            8. Compute TV = Metrics.tv_from_kl(KL)

        Args:
            T: Total iteration count. Must satisfy 2*T % cfg.K == 0.
            cfg: Configuration object.

        Returns:
            Tuple (kl_value, tv_value) as Python floats.
            Returns (float('nan'), float('nan')) if a numerical error occurs.
        """
        label: str = cfg.label if cfg.label else f"d{cfg.d}_k{cfg.k_active}"

        # --- Step 1: Build Schedule ---
        try:
            schedule: Schedule = Schedule(T, cfg.K, cfg.c0, cfg.c1)
        except Exception as exc:
            warnings.warn(
                f"[{label}, T={T}] Failed to build Schedule: {exc}",
                UserWarning,
                stacklevel=2,
            )
            return float("nan"), float("nan")

        # Verify schedule and warn (but do not abort) if checks fail
        schedule_ok: bool = schedule.verify_schedule(
            tol=1.0e-4  # from config.yaml: numerics.schedule_verify_tol
        )
        if not schedule_ok:
            warnings.warn(
                f"[{label}, T={T}] Schedule verification failed. "
                f"Results may be inaccurate. Consider adjusting c0/c1.",
                UserWarning,
                stacklevel=2,
            )

        # --- Step 2: Build GaussianTarget ---
        # Created per T for strict reproducibility (seed is fixed, so
        # the covariance is identical across all T values for the same cfg).
        target: GaussianTarget = GaussianTarget(
            d=cfg.d,
            k_active=cfg.k_active,
            seed=cfg.seed,
            device=cfg.device,
        )

        # --- Step 3: Build GaussianTracker ---
        tracker: GaussianTracker = GaussianTracker(
            d=cfg.d,
            device=cfg.device,
        )

        # --- Step 4: Build Sampler ---
        sampler: Sampler = Sampler(
            K=cfg.K,
            device=cfg.device,
        )

        # --- Step 5: Run the sampler ---
        try:
            mean_K: torch.Tensor
            cov_K_diag: torch.Tensor
            mean_K, cov_K_diag = sampler.run(
                T=T,
                schedule=schedule,
                target=target,
                tracker=tracker,
            )
        except Exception as exc:
            warnings.warn(
                f"[{label}, T={T}] Sampler.run() failed: {exc}",
                UserWarning,
                stacklevel=2,
            )
            return float("nan"), float("nan")

        # Ensure float64 dtype (defensive cast)
        mean_K = mean_K.to(dtype=torch.float64)
        cov_K_diag = cov_K_diag.to(dtype=torch.float64)

        # Clamp covariance diagonal from below for numerical safety
        # (matches config.yaml: numerics.sigma_min_clamp: 1.0e-12)
        cov_K_diag = cov_K_diag.clamp(min=1.0e-12)

        # --- Step 6: Get reference distribution q_K ---
        try:
            mu_ref: torch.Tensor
            cov_ref_diag: torch.Tensor
            mu_ref, cov_ref_diag = sampler.get_reference_distribution(
                schedule=schedule,
                target=target,
            )
        except Exception as exc:
            warnings.warn(
                f"[{label}, T={T}] get_reference_distribution() failed: {exc}",
                UserWarning,
                stacklevel=2,
            )
            return float("nan"), float("nan")

        # Ensure float64 dtype
        mu_ref = mu_ref.to(dtype=torch.float64)
        cov_ref_diag = cov_ref_diag.to(dtype=torch.float64)

        # Clamp reference covariance diagonal from below
        cov_ref_diag = cov_ref_diag.clamp(min=1.0e-12)

        # Move all tensors to CPU for Metrics computation
        # (Metrics operates on CPU tensors; device may be 'cuda')
        mean_K_cpu: torch.Tensor = mean_K.cpu()
        cov_K_diag_cpu: torch.Tensor = cov_K_diag.cpu()
        mu_ref_cpu: torch.Tensor = mu_ref.cpu()
        cov_ref_diag_cpu: torch.Tensor = cov_ref_diag.cpu()

        # --- Step 7: Compute KL divergence ---
        try:
            kl_val: float = Metrics.kl_gaussian(
                mu1=mean_K_cpu,
                Sigma1_diag=cov_K_diag_cpu,
                mu2=mu_ref_cpu,
                Sigma2_diag=cov_ref_diag_cpu,
                sigma_min_clamp=1.0e-12,
            )
        except Exception as exc:
            warnings.warn(
                f"[{label}, T={T}] Metrics.kl_gaussian() failed: {exc}",
                UserWarning,
                stacklevel=2,
            )
            return float("nan"), float("nan")

        # Sanity check: KL should be non-negative and finite
        if math.isnan(kl_val):
            warnings.warn(
                f"[{label}, T={T}] KL divergence is NaN. "
                f"Check schedule parameters and numerical precision.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan"), float("nan")

        if math.isinf(kl_val):
            warnings.warn(
                f"[{label}, T={T}] KL divergence is infinite. "
                f"This may indicate degenerate covariance matrices.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan"), float("nan")

        # Clamp to >= 0 (handles tiny negative values from float64 rounding)
        kl_val = max(kl_val, 0.0)

        # --- Step 8: Compute TV distance bound ---
        tv_val: float = Metrics.tv_from_kl(kl_val)

        # Clear the target's inverse covariance cache to prevent memory
        # accumulation across T values (tau values are T-dependent)
        target.clear_cache()

        return kl_val, tv_val

    def _print_result_summary(
        self, key: str, result: Dict[str, Any]
    ) -> None:
        """Print a summary of results for one configuration.

        Args:
            key: Config label string (e.g., 'fig2a').
            result: Result dict from run_convergence_experiment.
        """
        T_values: List[int] = result["T_values"]
        kl_values: List[float] = result["kl_values"]
        tv_values: List[float] = result["tv_values"]
        cfg: Config = result["config"]

        # Filter out NaN values for statistics
        valid_kl: List[float] = [
            kl for kl in kl_values if not math.isnan(kl)
        ]
        valid_T: List[int] = [
            T for T, kl in zip(T_values, kl_values) if not math.isnan(kl)
        ]

        print(f"\nResults for [{key}] (d={cfg.d}, k_active={cfg.k_active}):")
        print(f"  {'T':>8}  {'KL':>14}  {'TV':>14}")
        print(f"  {'-'*8}  {'-'*14}  {'-'*14}")
        for T_val, kl_val, tv_val in zip(T_values, kl_values, tv_values):
            kl_str: str = f"{kl_val:.6e}" if not math.isnan(kl_val) else "NaN"
            tv_str: str = f"{tv_val:.6e}" if not math.isnan(tv_val) else "NaN"
            print(f"  {T_val:>8}  {kl_str:>14}  {tv_str:>14}")

        if len(valid_kl) >= 2:
            # Fit theoretical rate and report fitted constant C
            try:
                C_fit: float
                _: np.ndarray
                C_fit, _ = Metrics.fit_theoretical_rate(valid_T, valid_kl)
                print(f"\n  Fitted constant C (KL ~ C * log^4(T) / T^3): {C_fit:.4e}")
            except Exception as exc:
                print(f"\n  Could not fit theoretical rate: {exc}")

            # Report log-log slope (should approach -3 for large T)
            if len(valid_kl) >= 3:
                log_T_arr: np.ndarray = np.log(np.array(valid_T, dtype=np.float64))
                log_kl_arr: np.ndarray = np.log(
                    np.maximum(np.array(valid_kl, dtype=np.float64), 1.0e-300)
                )
                # Linear regression on log-log scale
                coeffs: np.ndarray = np.polyfit(log_T_arr, log_kl_arr, 1)
                slope: float = float(coeffs[0])
                print(f"  Log-log slope (should approach -3): {slope:.3f}")

        print()
