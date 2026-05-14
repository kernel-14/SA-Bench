## experiments.py

"""
Experiment runner for reproducing the Average Reward Policy Gradient simulations.

This module implements the :class:`ExperimentRunner` class with static methods
``run_experiment1``, ``run_experiment2``, and ``run_experiment3`` that execute
the three experiments described in the paper:

* Experiment 1: Varying state and action space sizes.
* Experiment 2: Influence of reward variance.
* Experiment 3: Influence of transition kernel structure (C_p).

All parameters are read from ``config.yaml`` to guarantee exact reproducibility.
"""

import matplotlib.pyplot as plt
import os
import sys
from typing import List, Dict, Any

# Attempt to import yaml; if unavailable, fall back to a simple manual parse.
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False
    print("Warning: 'pyyaml' not installed. The configuration will be read using a "
          "simple hand‑written YAML parser. This may not support all YAML features. "
          "For full compatibility, install pyyaml: pip install pyyaml",
          file=sys.stderr)

from mdp import MDP
from pg_solver import PolicyGradientSolver
from utils import set_seed

# ---------------------------------------------------------------------------
# Configuration loading utilities
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    If ``pyyaml`` is not installed, a minimal YAML‑like parser is used that
    supports nested mappings and simple scalars.  This is sufficient for the
    structure of ``config.yaml`` used in this project.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.  Default is ``config.yaml``.

    Returns
    -------
    dict
        The parsed configuration as a nested dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration file cannot be parsed.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    if _HAS_YAML:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        # Minimal parser – works for the structure in config.yaml
        config = {}
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        current_section = None
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Detect a new section (top‑level key followed by ':')
            if line[0] != " " and not stripped.startswith("-"):
                if stripped.endswith(":"):
                    current_section = stripped[:-1]
                    config[current_section] = {}
                else:
                    raise ValueError(f"Expected section header ending with ':', got: {line}")
            else:
                # Within a section: key: value or list item
                if current_section is None:
                    raise ValueError("Key without a preceding section.")
                leading_spaces = len(line) - len(line.lstrip())
                # list items
                if stripped.startswith("- "):
                    # Simple sequence parsing – only works for flat lists of scalars
                    value_str = stripped[2:].strip()
                    value = _parse_scalar(value_str)
                    # Since the config sequences are always under a key, we need a way to
                    # assign them. This primitive parser does not support complex nesting,
                    # so we provide a fallback that fails gracefully.
                    raise NotImplementedError(
                        "The fallback YAML parser does not support sequences. "
                        "Please install pyyaml to proceed."
                    )
                else:
                    if ":" in stripped:
                        key, val_str = stripped.split(":", 1)
                        key = key.strip()
                        val_str = val_str.strip()
                        config[current_section][key] = _parse_scalar(val_str)
    return config


def _parse_scalar(s: str) -> Any:
    """Convert a YAML scalar string to a Python object."""
    s = s.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


# ---------------------------------------------------------------------------
# ExperimentRunner class
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Static methods to run the three simulation experiments and produce plots.
    """

    @staticmethod
    def run_experiment1(config: Dict[str, Any] = None) -> None:
        """
        Experiment 1: Varying state and action space sizes.

        MDPs with ``(S,A) = (3,3)``, ``(9,9)``, ``(81,81)`` are constructed using
        :meth:`MDP.build_exp1`.  The projected policy gradient algorithm is executed
        for the number of iterations specified in the configuration.  The resulting
        average reward curves are plotted.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary.  If ``None``, it is loaded from ``config.yaml``.
        """
        if config is None:
            config = _load_config()

        common_cfg = config.get("common", {})
        exp1_cfg = config.get("experiment1", {})
        sizes = exp1_cfg.get("state_action_sizes", [[3, 3], [9, 9], [81, 81]])
        num_iters = exp1_cfg.get("num_iterations", 2000)
        step_size = exp1_cfg.get("step_size", 0.1)
        seed = common_cfg.get("random_seed", 42)

        set_seed(seed)
        plt.figure()

        for S, A in sizes:
            mdp = MDP.build_exp1(S, A, seed=seed)
            solver = PolicyGradientSolver(mdp, eta=step_size, max_iter=num_iters)
            history = solver.run()
            plt.plot(range(len(history)), history, label=f'S={S}, A={A}')

        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.title('Experiment 1: Convergence vs. state/action space size')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def run_experiment2(config: Dict[str, Any] = None) -> None:
        """
        Experiment 2: Influence of reward variance.

        For a fixed MDP size ``(16,16)``, four different reward variance settings
        are tested (no variance, low, high, maximal).  The transition kernel is
        randomly generated **once** and reused across all reward variants.  The
        algorithm is run with the same step size and initial policy.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary.  If ``None``, it is loaded from ``config.yaml``.
        """
        if config is None:
            config = _load_config()

        exp2_cfg = config.get("experiment2", {})
        S = exp2_cfg.get("state_size", 16)
        A = exp2_cfg.get("action_size", 16)
        num_iters = exp2_cfg.get("num_iterations", 2000)
        step_size = exp2_cfg.get("step_size", 0.05)
        variants = exp2_cfg.get("reward_variants", ["no_var", "low_var", "high_var", "max_var"])
        # Use a seed that reproduces the paper's random transition kernel.
        # The Task specification states seed=123 for this experiment.
        seed = 123

        set_seed(seed)
        plt.figure()

        label_map = {
            "no_var": "No variance",
            "low_var": "Low variance",
            "high_var": "High variance",
            "max_var": "Max variance"
        }

        for variant in variants:
            mdp = MDP.build_exp2(S, A, variant, seed=seed)
            solver = PolicyGradientSolver(mdp, eta=step_size, max_iter=num_iters)
            history = solver.run()
            plt.plot(range(len(history)), history,
                     label=label_map.get(variant, variant))

        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.title('Experiment 2: Convergence for different reward variances')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def run_experiment3(config: Dict[str, Any] = None) -> None:
        """
        Experiment 3: Influence of transition kernel (C_p).

        Three MDPs of size ``(16,16)`` are built with different transition kernels:
        uniform, non‑uniform (stochastic), and deterministic.  All use the same
        high‑variance reward function.  The algorithm runs for 3000 iterations.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary.  If ``None``, it is loaded from ``config.yaml``.
        """
        if config is None:
            config = _load_config()

        exp3_cfg = config.get("experiment3", {})
        S = exp3_cfg.get("state_size", 16)
        A = exp3_cfg.get("action_size", 16)
        num_iters = exp3_cfg.get("num_iterations", 3000)
        step_size = exp3_cfg.get("step_size", 0.05)
        kernels = exp3_cfg.get("kernel_types", ["uniform", "non_uniform", "deterministic"])
        # Task specification: seed=456 for this experiment.
        seed = 456

        set_seed(seed)
        plt.figure()

        label_map = {
            "uniform": "Uniform",
            "non_uniform": "Non‑uniform",
            "deterministic": "Deterministic"
        }

        for kernel in kernels:
            mdp = MDP.build_exp3(S, A, kernel, seed=seed)
            solver = PolicyGradientSolver(mdp, eta=step_size, max_iter=num_iters)
            history = solver.run()
            plt.plot(range(len(history)), history,
                     label=label_map.get(kernel, kernel))

        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.title('Experiment 3: Convergence vs. transition kernel type')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

