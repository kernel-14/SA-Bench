## main.py

"""
Entry point for the numerical validation of the diffusion sampler convergence.

Reads configuration from 'config.yaml' (expected in the same directory),
instantiates the experiment, runs it, and visualises the results as a
log‑log plot of KL divergence versus the total number of iterations T.
"""

import os
import sys
import yaml   # PyYAML library (pip install pyyaml)
from typing import Dict, Any

from experiment import Experiment              # our custom experiment harness
from visualization import Visualizer          # custom plotting utilities


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load the experimental configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the parsed configuration.

    Raises:
        FileNotFoundError: if the file does not exist.
        yaml.YAMLError: if the file cannot be parsed.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    """
    Main execution routine:

    1. Load configuration.
    2. Instantiate the Experiment.
    3. Run the experiment for all (d,k) pairs and T values.
    4. Visualise the KL divergence results and save the figure.
    """
    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    config_path = "config.yaml"          # default; can be changed via sys.argv
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Instantiate and run the experiment
    # ------------------------------------------------------------------
    try:
        exp = Experiment(config)
    except Exception as e:
        print(f"Error initializing experiment: {e}", file=sys.stderr)
        sys.exit(1)

    print("Running experiment on (d,k) pairs:", config["experiment"]["d_k_pairs"])
    results = exp.run()

    # Print a summary of the obtained slopes (if any)
    for (d, k), data in results.items():
        slope = data.get("slope")
        if slope is not None:
            print(f"d={d}, k={k}: fitted log‑log slope = {slope:.3f}")

    # ------------------------------------------------------------------
    # 3. Visualise the results
    # ------------------------------------------------------------------
    vis_cfg = config.get("visualisation", {})
    # Choose a default output filename based on the config, or use a fixed name
    figure_name = f"kl_convergence.{vis_cfg.get('figure_format', 'png')}"
    dpi = vis_cfg.get("dpi", 150)
    ref_slope = vis_cfg.get("reference_slope", -3.0)

    # Ensure the output directory exists (save in current dir)
    output_path = os.path.join(os.getcwd(), figure_name)

    # Generate the plot (matplotlib figure)
    Visualizer.plot_kl_vs_T(
        results,
        save_path=output_path,
        reference_slope=ref_slope,
        dpi=dpi,
        fmt=vis_cfg.get("figure_format", "png"),
    )

    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
