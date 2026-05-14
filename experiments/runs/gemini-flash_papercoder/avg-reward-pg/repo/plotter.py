import matplotlib.pyplot as plt
import os
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Plotter:
    """
    This class handles the creation and saving of plots for experimental results,
    visualizing the convergence behavior of the PPG algorithm.
    """

    def __init__(self, plot_dir: str):
        """
        Initializes the Plotter instance.

        Args:
            plot_dir (str): The path to the directory where plots will be saved.
        """
        if not isinstance(plot_dir, str) or not plot_dir:
            raise ValueError("plot_dir must be a non-empty string.")
        
        self.plot_dir: str = plot_dir
        
        # Create the plot directory if it doesn't exist
        os.makedirs(self.plot_dir, exist_ok=True)
        logging.info("Plots will be saved to: %s", os.path.abspath(self.plot_dir))

    def plot_experiment_results(
        self,
        exp_name: str,
        results: Dict[Any, Dict[str, List[float]]],
        optimal_rewards: Dict[Any, float],  # Not directly used for plotting lines, but provided for context.
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
        metric_key: str = "average_rewards"
    ) -> None:
        """
        Generates and saves a line plot displaying the results of a specific experiment.

        Args:
            exp_name (str): A descriptive name for the experiment (e.g., "Experiment 1").
            results (Dict[Any, Dict[str, List[float]]]): A dictionary where keys identify
                                                           different scenarios within an experiment.
                                                           Each value is another dictionary containing lists
                                                           for different metrics (e.g., 'average_rewards', 'optimality_gaps')
                                                           over iterations.
            optimal_rewards (Dict[Any, float]): A dictionary mapping scenario identifiers to their
                                                corresponding optimal average rewards. Provided for context.
            title (str): The title for the plot.
            xlabel (str): The label for the x-axis (typically "Iterations").
            ylabel (str): The label for the y-axis (e.g., "Average Reward", "Optimality Gap").
            filename (str): The name of the file (including extension, e.g., "exp1_plot.png")
                            to which the plot will be saved.
            metric_key (str): Specifies which metric from the scenario_data to plot on the y-axis.
                              Expected values are "average_rewards" or "optimality_gaps".
        """
        logging.info("Generating plot for %s: %s", exp_name, title)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort results keys for consistent plotting order
        sorted_scenario_ids = sorted(results.keys(), key=str)

        for scenario_id in sorted_scenario_ids:
            scenario_data = results[scenario_id]
            
            if metric_key not in scenario_data:
                logging.warning("Metric key '%s' not found for scenario '%s' in %s. Skipping.",
                                metric_key, scenario_id, exp_name)
                continue
            
            y_values: List[float] = scenario_data[metric_key]
            x_values: List[int] = list(range(len(y_values)))
            
            # Convert scenario_id to string for legend, especially for tuple keys
            label_str: str = str(scenario_id)
            if isinstance(scenario_id, Tuple):
                # Custom labels for (S, A) in Experiment 1 for better readability
                label_str = f"S={scenario_id[0]}, A={scenario_id[1]}"
            elif isinstance(scenario_id, str):
                # Capitalize first letter for single string labels
                label_str = scenario_id.replace(" variance", " Var.").replace("non-uniform", "Non-Uniform")
                label_str = label_str.replace("uniform", "Uniform").replace("deterministic", "Deterministic")
                label_str = label_str.title()

            ax.plot(x_values, y_values, label=label_str)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(title="Scenario")
        ax.grid(True)

        # Handle y-axis for optimality gap plots to start from 0 if possible
        if metric_key == "optimality_gaps":
            ax.set_yscale('log') # Log scale is common for optimality gaps
            ax.set_ylabel("Optimality Gap (log scale)")
            ax.set_ylim(bottom=1e-12) # Prevent log(0) and show small gaps clearly

        file_path: str = os.path.join(self.plot_dir, filename)
        fig.savefig(file_path, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        logging.info("Plot saved successfully to: %s", file_path)

