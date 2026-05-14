## config.py
"""Configuration dataclass for reproducing experiments from:
'Global Convergence of Policy Gradient in Average Reward MDPs'.

All default values are taken directly from config.yaml. The Config dataclass
serves as the single source of truth for all hyperparameters and experiment
settings across the project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Config:
    """Centralized configuration for all experiments.

    All fields have defaults matching config.yaml so that Config() with no
    arguments produces a fully valid configuration. Mutable defaults use
    field(default_factory=...) to avoid shared-mutable-default issues.

    Attributes:
        exp1_sizes: List of (S, A) tuples for Experiment 1.
        exp1_iterations: Number of PPG iterations for Experiment 1.
        exp1_kernel_type: Transition kernel type for Experiment 1.
        exp1_reward_type: Reward type for Experiment 1.
        exp2_size: (S, A) tuple for Experiment 2.
        exp2_iterations: Number of PPG iterations for Experiment 2.
        exp2_kernel_type: Transition kernel type for Experiment 2.
        exp2_special_state: Index of the special state s_0 in Experiment 2.
        exp2_reward_variants: Dict mapping variant name to fraction_negative
            and label for Experiment 2.
        exp3_size: (S, A) tuple for Experiment 3.
        exp3_iterations: Number of PPG iterations for Experiment 3.
        exp3_reward_type: Reward type for Experiment 3.
        exp3_kernel_variants: Dict mapping kernel name to type and label
            for Experiment 3.
        step_size_multiplier: Multiplier for computing eta = multiplier / L2.
        step_size_fallback: Fallback eta when L2 is zero or near-zero.
        init_policy: Initial policy type ('uniform').
        power_iter_tol: Convergence tolerance for power iteration.
        power_iter_max: Maximum iterations for power iteration.
        complexity_n_samples: Number of random policies sampled to estimate
            complexity constants C_m, C_p, C_r, kappa_r.
        dirichlet_alpha: Concentration parameter for Dirichlet policy sampling.
        random_seed: Global numpy random seed for reproducibility.
        output_dir: Directory to save output figures.
        figure_dpi: DPI for saved figures.
        figure_format: File format for saved figures.
        plot_figure1a: Plotting metadata for Figure 1a.
        plot_figure1b: Plotting metadata for Figure 1b.
        plot_figure2: Plotting metadata for Figure 2.
    """

    # -------------------------------------------------------------------------
    # Experiment 1: Convergence vs State/Action Space Size (Figure 1a)
    # Paper Section 4, Appendix C.1
    # -------------------------------------------------------------------------
    exp1_sizes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(3, 3), (9, 9), (81, 81)]
    )
    exp1_iterations: int = 2000
    exp1_kernel_type: str = "nonuniform"
    exp1_reward_type: str = "max_variance"

    # -------------------------------------------------------------------------
    # Experiment 2: Convergence vs Reward Variance / C_r (Figure 1b)
    # Paper Section 4, Appendix C.2
    # -------------------------------------------------------------------------
    exp2_size: Tuple[int, int] = (16, 16)
    exp2_iterations: int = 2000
    exp2_kernel_type: str = "random_dirichlet"
    exp2_special_state: int = 0
    exp2_reward_variants: Dict[str, Dict] = field(
        default_factory=lambda: {
            "no_variance": {
                "fraction_negative": 0.0,
                "label": "No Variance",
            },
            "low_variance": {
                "fraction_negative": 0.125,  # 1/8 of actions get -1
                "label": "Low Variance",
            },
            "high_variance": {
                "fraction_negative": 0.25,  # 1/4 of actions get -1
                "label": "High Variance",
            },
            "max_variance": {
                "fraction_negative": 0.5,  # 1/2 of actions get -1
                "label": "Max Variance",
            },
        }
    )

    # -------------------------------------------------------------------------
    # Experiment 3: Convergence vs Transition Kernel / C_p (Figure 2)
    # Paper Section 4, Appendix C.3
    # -------------------------------------------------------------------------
    exp3_size: Tuple[int, int] = (16, 16)
    exp3_iterations: int = 3000
    exp3_reward_type: str = "high_variance"
    exp3_kernel_variants: Dict[str, Dict] = field(
        default_factory=lambda: {
            "uniform": {
                "type": "uniform",
                "label": "Uniform",
            },
            "nonuniform": {
                "type": "nonuniform",
                "label": "Non-uniform",
            },
            "deterministic": {
                "type": "deterministic",
                "label": "Deterministic",
            },
        }
    )

    # -------------------------------------------------------------------------
    # Policy Gradient Algorithm Settings
    # Paper Theorem 1: eta < 1 / L_2^Pi
    # -------------------------------------------------------------------------
    step_size_multiplier: float = 0.5
    step_size_fallback: float = 0.01
    init_policy: str = "uniform"

    # -------------------------------------------------------------------------
    # Value Function Computation Settings
    # Paper Lemma 1, Equation 15
    # -------------------------------------------------------------------------
    power_iter_tol: float = 1.0e-10
    power_iter_max: int = 10000

    # -------------------------------------------------------------------------
    # Complexity Metrics Estimation Settings
    # Paper Table 1/2: C_m, C_p, C_r, kappa_r, L_2^Pi
    # -------------------------------------------------------------------------
    complexity_n_samples: int = 200
    dirichlet_alpha: float = 1.0

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    random_seed: int = 42
    output_dir: str = "results/"

    # -------------------------------------------------------------------------
    # Plotting Settings
    # -------------------------------------------------------------------------
    figure_dpi: int = 150
    figure_format: str = "png"
    plot_figure1a: Dict[str, str] = field(
        default_factory=lambda: {
            "filename": "figure1a.png",
            "title": "Figure 1(a): Convergence vs State/Action Space Size",
            "xlabel": "Iteration",
            "ylabel": "Average Reward",
        }
    )
    plot_figure1b: Dict[str, str] = field(
        default_factory=lambda: {
            "filename": "figure1b.png",
            "title": "Figure 1(b): Convergence vs Reward Variance (C_r)",
            "xlabel": "Iteration",
            "ylabel": "Average Reward",
        }
    )
    plot_figure2: Dict[str, str] = field(
        default_factory=lambda: {
            "filename": "figure2.png",
            "title": "Figure 2: Convergence vs Transition Kernel (C_p)",
            "xlabel": "Iteration",
            "ylabel": "Change in Average Reward",
        }
    )

    def __post_init__(self) -> None:
        """Perform post-initialization side effects.

        1. Creates the output directory if it does not exist.
        2. Sets the global numpy random seed for reproducibility.

        This fires immediately when Config() is instantiated, ensuring all
        downstream random operations are seeded before any module runs.
        """
        # Ensure output directory exists (idempotent)
        os.makedirs(self.output_dir, exist_ok=True)

        # Set global numpy random seed for reproducibility
        np.random.seed(self.random_seed)
