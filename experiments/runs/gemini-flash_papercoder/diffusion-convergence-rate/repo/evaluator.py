import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, List, Tuple

# Assuming Config is available from its respective module
from config import Config

class Evaluator:
    """
    Handles processing and plotting the experimental results for the diffusion model.

    This class is responsible for collecting the KL divergence values from
    multiple experimental runs and visualizing them against the total number
    of iterations (T_total), along with an overlay of the theoretical convergence rate.
    """

    def __init__(self, config: Config):
        """
        Initializes the Evaluator with a configuration object.

        Args:
            config: An instance of the Config class, providing global experiment parameters.
        """
        self.config: Config = config

    def process_and_store_results(self, exp_params: Dict[str, Any], kl_result: float, current_results: Dict[str, Any]) -> None:
        """
        Organizes and stores the KL divergence result from a single experimental
        run into a structured dictionary.

        The structure of `current_results` will be:
        {
            d_val_1: {
                k_val_1: {'T_total': [t1, t2, ...], 'KL_div': [kl1, kl2, ...]},
                k_val_2: {'T_total': [...], 'KL_div': [...]}
            },
            d_val_2: { ... }
        }

        Args:
            exp_params: A dictionary containing the specific parameters for the
                        completed experiment run, including 'd', 'k', and 'T_total'.
            kl_result: The calculated KL divergence value for this specific run.
            current_results: A dictionary that accumulates results across all
                             experimental runs. This dictionary is modified in place.
        """
        d_val: int = exp_params['d']
        k_val: int = exp_params['k']
        T_total_val: int = exp_params['T_total']

        if d_val not in current_results:
            current_results[d_val] = {}
        
        if k_val not in current_results[d_val]:
            current_results[d_val][k_val] = {'T_total': [], 'KL_div': []}
        
        current_results[d_val][k_val]['T_total'].append(T_total_val)
        current_results[d_val][k_val]['KL_div'].append(kl_result)

    def plot_results(self, results: Dict[str, Any], plot_filename: str) -> None:
        """
        Generates and saves a plot comparing the empirical KL divergence results
        with the theoretical convergence rate, similar to Figure 2 in the paper.

        Each (d, k) combination will have its own subplot.

        Args:
            results: The comprehensive dictionary containing all collected
                     experimental results, structured by 'd' and 'k'.
            plot_filename: The desired filename for the output plot (e.g., 'convergence_plot.png').
        """
        # Determine the number of subplots needed
        # The structure is results[d][k], and d_values and k_values are paired,
        # so the number of combinations is len(self.config.d_values)
        num_combinations = len(self.config.d_values)
        
        # Arrange subplots in a single row as in Figure 2 of the paper
        num_rows = 1
        num_cols = num_combinations
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5.5 * num_cols, 4.5 * num_rows), squeeze=False)
        axes_flat = axes.flatten()
        
        subplot_idx = 0
        
        # Sort d_values and iterate through the original pairs (d,k) as defined in config
        for d_val, k_val in zip(self.config.d_values, self.config.k_values):
            # Ensure this (d_val, k_val) combination actually has results
            if d_val not in results or k_val not in results[d_val]:
                print(f"Warning: No results found for d={d_val}, k={k_val}. Skipping subplot.")
                continue

            if subplot_idx >= len(axes_flat):
                print(f"Warning: Exceeded number of pre-allocated subplots. Skipping d={d_val}, k={k_val}.")
                continue

            ax = axes_flat[subplot_idx]
            exp_data = results[d_val][k_val]
            
            # Sort empirical data by T_total for correct plotting order
            sorted_indices = np.argsort(exp_data['T_total'])
            T_total_empirical = np.array(exp_data['T_total'])[sorted_indices]
            KL_div_empirical = np.array(exp_data['KL_div'])[sorted_indices]

            # Plot empirical results
            ax.plot(T_total_empirical, KL_div_empirical, 'bo-', label='Empirical Result', linewidth=1.5, markersize=5)

            # Plot theoretical rate O(log^4 T / T^3)
            if len(T_total_empirical) > 0:
                # Create a dense range of T values for a smooth theoretical curve
                T_min_plot = T_total_empirical[0]
                T_max_plot = T_total_empirical[-1]
                # np.logspace creates logarithmically spaced numbers
                T_smooth_range = np.logspace(np.log10(T_min_plot), np.log10(T_max_plot), 100)
                
                # Ensure T values are > 1 to avoid log(1)=0 or log(0) issues, as T_total are iterations.
                # The minimum T in T_total_range should typically be >1.
                T_smooth_range_safe = np.maximum(T_smooth_range, 1.1) 
                
                # Fit scaling constant C' using the last (largest T) empirical point
                # Check for valid values to prevent division by zero or log(0)
                if T_total_empirical[-1] > 1 and KL_div_empirical[-1] > 0:
                    kl_val_at_max_t = KL_div_empirical[-1]
                    t_val_at_max_t = T_total_empirical[-1]
                    
                    # Calculate theoretical factor at this point
                    theoretical_factor_at_max_t = (np.log(t_val_at_max_t)**4) / (t_val_at_max_t**3)
                    
                    # Prevent division by numbers that are numerically zero
                    if theoretical_factor_at_max_t > np.finfo(float).eps: # Use machine epsilon for robust check
                        C_prime_fit = kl_val_at_max_t / theoretical_factor_at_max_t
                    else:
                        C_prime_fit = 1.0 # Default to 1 if numerical issues (factor is too small)
                        print(f"Warning: Theoretical factor for fitting C' is extremely small for d={d_val}, k={k_val}. Defaulting C' to 1.0.")
                else:
                    C_prime_fit = 1.0 # Default to 1 if no valid data point for fitting
                    print(f"Warning: No valid empirical data point for fitting C' for d={d_val}, k={k_val}. Defaulting C' to 1.0.")

                theoretical_kl_values = C_prime_fit * (np.log(T_smooth_range_safe)**4) / (T_smooth_range_safe**3)
                ax.plot(T_smooth_range, theoretical_kl_values, 'k--', label=r'Theoretical Rate $O(\log^4 T / T^3)$', linewidth=1.5)
            else:
                print(f"No empirical data to plot theoretical curve for d={d_val}, k={k_val}.")


            # Customization
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Total Iterations (T)')
            ax.set_ylabel('KL Divergence')
            ax.set_title(f'd={d_val}, k={k_val}')
            ax.legend(fontsize='small')
            ax.grid(True, which="both", ls="-", alpha=0.6)
            
            subplot_idx += 1
        
        # Remove any unused subplots if num_combinations was less than num_cols * num_rows (e.g. if num_combinations = 2 and grid was 1x3)
        for i in range(subplot_idx, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.suptitle('KL Divergence vs. Total Iterations (T)', fontsize=16, y=1.02) # Add a main title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
        
        output_path = os.path.join(self.config.output_dir, plot_filename)
        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free memory
        print(f"Plot saved to {output_path}")

# Example usage for testing purposes (not part of the final module logic)
if __name__ == '__main__':
    import json # For pretty printing dictionary

    # Mock Config for local testing
    class MockConfig:
        def __init__(self):
            self.output_dir = "test_results_evaluator"
            self.T_total_range = [100, 200, 400, 800, 1600] # Example range for T
            self.d_values = [10, 100, 500]
            self.k_values = [10, 10, 100]

    # Create dummy results data for plotting
    dummy_results: Dict[int, Dict[int, Dict[str, List[float]]]] = {
        10: {
            10: {
                'T_total': [100, 200, 400],
                'KL_div': [0.1, 0.05, 0.025]
            }
        },
        100: {
            10: {
                'T_total': [100, 200, 400],
                'KL_div': [0.5, 0.15, 0.075]
            }
        },
        500: {
            100: {
                'T_total': [100, 200, 400],
                'KL_div': [1.0, 0.3, 0.15]
            }
        }
    }

    mock_config_instance = MockConfig()
    evaluator = Evaluator(mock_config_instance)

    # Test process_and_store_results by adding more data
    new_results_dict = {}
    # Copy initial dummy results for modification
    for d_val, d_data in dummy_results.items():
        new_results_dict[d_val] = {}
        for k_val, k_data in d_data.items():
            new_results_dict[d_val][k_val] = {'T_total': list(k_data['T_total']), 'KL_div': list(k_data['KL_div'])}

    # Add new results for d=10, k=10
    evaluator.process_and_store_results({'T_total': 800, 'd': 10, 'k': 10}, 0.012, new_results_dict)
    evaluator.process_and_store_results({'T_total': 1600, 'd': 10, 'k': 10}, 0.006, new_results_dict)
    # Add new results for d=100, k=10
    evaluator.process_and_store_results({'T_total': 800, 'd': 100, 'k': 10}, 0.04, new_results_dict)
    evaluator.process_and_store_results({'T_total': 1600, 'd': 100, 'k': 10}, 0.015, new_results_dict)
    # Add new results for d=500, k=100
    evaluator.process_and_store_results({'T_total': 800, 'd': 500, 'k': 100}, 0.08, new_results_dict)
    evaluator.process_and_store_results({'T_total': 1600, 'd': 500, 'k': 100}, 0.04, new_results_dict)


    print("Aggregated results structure:")
    print(json.dumps(new_results_dict, indent=2))

    # Ensure output directory exists for plot saving
    if not os.path.exists(mock_config_instance.output_dir):
        os.makedirs(mock_config_instance.output_dir)

    print(f"\nGenerating plot in {mock_config_instance.output_dir}...")
    evaluator.plot_results(new_results_dict, "test_convergence_plot.png")
    print("Evaluator test completed. Check 'test_results_evaluator/test_convergence_plot.png'.")

