import os
import logging
from typing import Dict, List, Tuple, Any

# Import global configuration variables from config.py
import config

# Import classes from other modules
from simulator import Simulator
from plotter import Plotter

# Configure logging for the main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Main:
    """
    The main entry point of the application. It orchestrates the execution and
    visualization of all defined experiments.
    """

    def __init__(self):
        """
        Initializes the Main orchestrator.
        """
        # Ensure the plot directory exists before initializing the Plotter
        os.makedirs(config.PLOT_DIR, exist_ok=True)
        logging.info("Plot directory '%s' ensured to exist.", config.PLOT_DIR)

        # Initialize the Simulator with the full configuration dictionary
        # The config.config variable holds the dictionary loaded from config.yaml
        self.simulator: Simulator = Simulator(config_dict=config.config)
        logging.info("Simulator initialized.")

        # Initialize the Plotter with the configured plot directory
        self.plotter: Plotter = Plotter(plot_dir=config.PLOT_DIR)
        logging.info("Plotter initialized.")

    def run_all_experiments(self) -> None:
        """
        Executes all experiments defined in the configuration and generates their plots.
        """
        logging.info("Starting all experiments...")

        # --- Experiment 1: Convergence with Different State and Action Space Sizes ---
        logging.info("Running Experiment 1...")
        exp1_results, exp1_mdps = self.simulator.run_experiment1()
        exp1_optimal_rewards: Dict[Tuple[int, int], float] = {
            k: mdp.optimal_avg_reward for k, mdp in exp1_mdps.items()
        }
        self.plotter.plot_experiment_results(
            exp_name="Experiment 1",
            results=exp1_results,
            optimal_rewards=exp1_optimal_rewards,
            title="Experiment 1: Convergence with Different State and Action Space Sizes",
            xlabel="Iterations",
            ylabel="Average Reward", # Figure 1(a) plots Average Reward
            filename="exp1_convergence.png",
            metric_key="average_rewards"
        )
        logging.info("Experiment 1 completed and plot generated.")

        # --- Experiment 2: Convergence with Different Reward Functions (Variance) ---
        logging.info("Running Experiment 2...")
        exp2_results, exp2_mdps = self.simulator.run_experiment2()
        exp2_optimal_rewards: Dict[str, float] = {
            k: mdp.optimal_avg_reward for k, mdp in exp2_mdps.items()
        }
        self.plotter.plot_experiment_results(
            exp_name="Experiment 2",
            results=exp2_results,
            optimal_rewards=exp2_optimal_rewards,
            title="Experiment 2: Convergence with Different Reward Functions (Variance)",
            xlabel="Iterations",
            ylabel="Average Reward", # Figure 1(b) plots Average Reward
            filename="exp2_convergence_reward_variance.png",
            metric_key="average_rewards"
        )
        logging.info("Experiment 2 completed and plot generated.")

        # --- Experiment 3: Convergence with Different Transition Kernels (Cp) ---
        logging.info("Running Experiment 3...")
        exp3_results, exp3_mdps = self.simulator.run_experiment3()
        exp3_optimal_rewards: Dict[str, float] = {
            k: mdp.optimal_avg_reward for k, mdp in exp3_mdps.items()
        }
        self.plotter.plot_experiment_results(
            exp_name="Experiment 3",
            results=exp3_results,
            optimal_rewards=exp3_optimal_rewards,
            title="Experiment 3: Convergence with Different Transition Kernels",
            xlabel="Iterations",
            ylabel="Optimality Gap", # Figure 2 implies optimality gap or change in avg reward
            filename="exp3_convergence_transition_kernel.png",
            metric_key="optimality_gaps" # Plotting optimality gaps for this experiment
        )
        logging.info("Experiment 3 completed and plot generated.")

        logging.info("All experiments finished.")


if __name__ == '__main__':
    # Create an instance of the Main orchestrator and run all experiments
    main_orchestrator = Main()
    main_orchestrator.run_all_experiments()

