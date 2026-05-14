## main.py

"""
Entry point for reproducing the experiments from

    "Global Convergence of Policy Gradient in Average Reward MDPs"
    by Yashaswini Murthy et al.

This script sequentially executes the three simulation studies described in
the paper and displays the resulting convergence plots.

Usage:
    python main.py

All configuration parameters are read from ``config.yaml``.
"""

from experiments import ExperimentRunner


if __name__ == "__main__":
    print("Running Experiment 1: Varying state/action space sizes ...")
    ExperimentRunner.run_experiment1()

    print("\nRunning Experiment 2: Influence of reward variance ...")
    ExperimentRunner.run_experiment2()

    print("\nRunning Experiment 3: Influence of transition kernel ...")
    ExperimentRunner.run_experiment3()

    print("\nAll experiments completed. Close the plot windows to exit.")
