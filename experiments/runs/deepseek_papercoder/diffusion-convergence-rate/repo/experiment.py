## experiment.py

"""
Orchestrates the numerical validation of the convergence rate for the
randomized midpoint sampler (Appendix A of the paper).

For each (d,k) configuration and a range of total discretisation steps T,
the exact KL divergence between the forward process (target) and the
generated sample distribution is computed via analytic covariance propagation.
Optionally fits a log‑log slope to confirm the O(log^4 T / T^3) behaviour.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Union

from schedule import Schedule
from sampler import Sampler
from gaussian_utils import target_covariance, GaussianKL


class Experiment:
    """
    Sets up and runs the convergence‑rate verification experiments.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialise the experiment from a configuration dictionary that mirrors
        config.yaml.

        Args:
            config: Dictionary with keys 'schedule', 'experiment', 'visualisation'.
        """
        # ---------- Schedule constants ----------
        sch_cfg = config['schedule']
        self.K: int = sch_cfg['K']                # number of rounds
        self.c0: float = sch_cfg['c0']             # exponent for hat_alpha_T+1
        self.c1: float = sch_cfg['c1']             # multiplicative constant in the schedule
        self.schedule_seed: int = sch_cfg['schedule_seed']  # base seed for schedule randomness

        # ---------- Experiment settings ----------
        exp_cfg = config['experiment']
        self.d_k_pairs: List[Tuple[int, int]] = [tuple(pair) for pair in exp_cfg['d_k_pairs']]
        self.sigma_seed: int = exp_cfg['sigma_seed']     # base seed for sigma_diag generation
        self.T_values: List[int] = exp_cfg['T_values']   # discretisation steps (multiples of K/2)

        # ---------- Visualisation / optional ----------
        self.ref_slope: float = config.get('visualisation', {}).get('reference_slope', -3.0)

        # Validate T values
        req_div = self.K // 2  # K=10 → divisor 5
        for T in self.T_values:
            if T % req_div != 0:
                raise ValueError(
                    f"T = {T} must be a multiple of K/2 = {req_div} to "
                    f"make N = 2T/K an integer."
                )

    def run(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Execute the convergence experiment for all (d,k) pairs.

        Returns:
            results: A dictionary mapping (d,k) → dict with keys:
                     - 'T': list of T values used
                     - 'KL': list of corresponding KL divergences
                     - 'slope': fitted log‑log slope (None if insufficient points)
        """
        results: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for d, k in self.d_k_pairs:
            # Generate the diagonal covariance of the target Gaussian
            seed = self.sigma_seed + d * 1000 + k
            rng = np.random.default_rng(seed)
            sigma_diag = np.zeros(d, dtype=np.float64)
            sigma_diag[:k] = rng.uniform(0.0, 10.0, size=k)

            # Containers for this (d,k) pair
            T_list: List[int] = []
            kl_list: List[float] = []

            for T in self.T_values:
                kl_div = self._run_single(d, sigma_diag, T)
                T_list.append(T)
                kl_list.append(kl_div)

            # Fit log‑log slope if we have enough points
            slope: Union[float, None] = None
            if len(T_list) >= 2:
                logT = np.log10(np.asarray(T_list, dtype=np.float64))
                logKL = np.log10(np.maximum(np.asarray(kl_list, dtype=np.float64), 1e-30))
                slope, _ = np.polyfit(logT, logKL, 1)

            results[(d, k)] = {
                'T': T_list,
                'KL': kl_list,
                'slope': slope,
            }

        return results

    def _run_single(self, d: int, sigma_diag: np.ndarray, T: int) -> float:
        """
        Run a single instance of the sampler for a given T and return the
        KL divergence KL(q_K || p_{Y_K}).

        Args:
            d: data dimension (unused, but kept for clarity)
            sigma_diag: 1D array of target variances, length d
            T: total number of discretisation steps

        Returns:
            KL divergence (float)
        """
        # 1. Create the randomised midpoint schedule
        seed = self.schedule_seed + T
        schedule = Schedule(T, self.K, self.c0, self.c1, seed=seed)

        # 2. Retrieve τ_{K,0} (the target noise level)
        #    schedule.tau_all is a list of length K+1; index K gives the final round.
        #    tau_{K,0} is the first element of that array.
        tau_K0 = schedule.tau_all[self.K][0]

        # 3. Instantiate the sampler and propagate covariances
        sampler = Sampler(sigma_diag, schedule)
        v_K = sampler.propagate_covariance()          # 1D variance vector
        C_K = np.diag(v_K)                            # full covariance matrix (diagonal)

        # 4. Target covariance (forward process distribution)
        C_target = target_covariance(tau_K0, sigma_diag)  # 2D diagonal matrix

        # 5. Compute KL divergence KL(q_K || p_{Y_K})
        kl_value = GaussianKL.compute(C_target, C_K)

        # Ensure non‑negative (could be slightly negative due to numerical noise)
        return max(0.0, kl_value)

