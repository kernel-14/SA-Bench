import yaml
import numpy as np
import json
import os
from tqdm import tqdm
from typing import Dict, Any, List

# Local imports
from config import Config
from diffusion_process import GaussianDiffusionProcess
from sampler import Sampler
from evaluator import Evaluator
import utils

class Main:
    """
    The main orchestrator for running the diffusion model convergence experiments.
    Manages the lifecycle from configuration loading to result visualization.
    """

    def __init__(self, config_file_path: str = 'config.yaml'):
        """
        Initializes the main experiment runner.

        Args:
            config_file_path: Path to the YAML configuration file.
        """
        self.config_file_path: str = config_file_path
        self.config: Config = None  # type: ignore
        self.all_results: Dict[str, Any] = {}

    def load_configuration(self) -> None:
        """
        Reads the experiment parameters from the YAML configuration file and
        instantiates the Config object.
        """
        print(f"Loading configuration from {self.config_file_path}...")
        try:
            with open(self.config_file_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            exp_config = raw_config.get('experiment', {})
            
            # Extract parameters with default empty lists if not found
            T_total_range: List[int] = exp_config.get('T_total_range', [])
            d_values: List[int] = exp_config.get('d_values', [])
            k_values: List[int] = exp_config.get('k_values', [])
            num_samples_per_run: int = exp_config.get('num_samples_per_run', 1000)
            K_rounds: int = exp_config.get('K_rounds', 10)
            c0: float = exp_config.get('c0', 12.0)
            c1: float = exp_config.get('c1', 61.0)
            output_dir: str = exp_config.get('output_dir', 'results')
            rng_seed: int = exp_config.get('rng_seed', 42)

            self.config = Config(
                T_total_range=T_total_range,
                d_values=d_values,
                k_values=k_values,
                num_samples_per_run=num_samples_per_run,
                K_rounds=K_rounds,
                c0=c0,
                c1=c1,
                output_dir=output_dir,
                rng_seed=rng_seed
            )
            print("Configuration loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {self.config_file_path}")
            raise
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during configuration loading: {e}")
            raise

    def setup_experiment_environment(self) -> None:
        """
        Prepares the runtime environment by setting the random seed and
        creating the output directory.
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_configuration() first.")
        
        print(f"Setting up environment with RNG seed {self.config.rng_seed}...")
        utils.setup_rng(self.config.rng_seed)
        utils.create_output_directory(self.config.output_dir)
        print(f"Output directory '{self.config.output_dir}' ensured.")

    def run_experiments(self) -> None:
        """
        Orchestrates the main experimental loop, running simulations for each
        parameter combination, collecting samples, calculating metrics, and
        storing intermediate results.
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_configuration() first.")

        print("Starting experiments...")
        self.all_results = {} # Clear any previous results

        # Create Evaluator instance here, as its methods are used within the loop
        evaluator = Evaluator(config=self.config)

        # Total number of individual (d, k, T_total) experiment sets
        total_experiment_sets = len(self.config.T_total_range) * len(self.config.d_values)

        with tqdm(total=total_experiment_sets, desc="Overall Experiments") as pbar_overall:
            for exp_params in self.config.get_experiment_parameters():
                T_total: int = exp_params['T_total']
                d: int = exp_params['d']
                k: int = exp_params['k']
                
                print(f"\n--- Running experiment: T_total={T_total}, d={d}, k={k} ---")

                # Validate N_steps_per_round calculation for current T_total
                # N = 2 * T / K must be an integer
                if (2 * T_total) % self.config.K_rounds != 0:
                    print(f"Skipping experiment (T_total={T_total}, d={d}, k={k}) because "
                          f"2 * T_total ({2 * T_total}) is not divisible by K_rounds ({self.config.K_rounds}). "
                          f"N_steps_per_round would not be an integer.")
                    pbar_overall.update(1)
                    continue

                diffusion_process = GaussianDiffusionProcess(d=d, k=k, rng_seed=self.config.rng_seed)
                sampler = Sampler(
                    config=self.config,
                    diffusion_process=diffusion_process,
                    T_total=T_total,
                    d=d
                )

                y_k_samples_list: List[np.ndarray] = []
                # Use tqdm for the sampling loop as well
                print(f"  Generating {self.config.num_samples_per_run} Y_K samples...")
                for _ in tqdm(range(self.config.num_samples_per_run), desc=f"  Sampling Y_K (T={T_total}, d={d}, k={k})"):
                    final_y_k = sampler.run_sampler_single_pass()
                    y_k_samples_list.append(final_y_k)

                y_k_samples_array = np.array(y_k_samples_list, dtype=np.float64)

                # Calculate empirical mean and covariance for Y_K
                mu_y_k_est = np.mean(y_k_samples_array, axis=0)
                # np.cov treats rows as variables by default, we need columns as variables (num_samples x d)
                sigma_y_k_est = np.cov(y_k_samples_array, rowvar=False)

                # Retrieve q_K distribution parameters
                last_tau_k0 = sampler.get_last_tau_k0()
                q_k_covariance = diffusion_process.get_qK_covariance(last_tau_k0)
                q_k_mu = np.zeros(d, dtype=np.float64) # Target is zero-mean Gaussian

                # Compute KL divergence
                try:
                    kl_result = utils.kl_divergence_gaussian(
                        mu1=mu_y_k_est,
                        sigma1=sigma_y_k_est,
                        mu2=q_k_mu,
                        sigma2=q_k_covariance
                    )
                    print(f"  KL Divergence: {kl_result:.6f}")
                except ValueError as e:
                    print(f"  Error computing KL divergence for T={T_total}, d={d}, k={k}: {e}")
                    kl_result = float('nan') # Store NaN for failed computations

                # Store results using Evaluator's method
                evaluator.process_and_store_results(exp_params, kl_result, self.all_results)
                pbar_overall.update(1)

        print("\nAll experiments completed.")

    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """
        Persists the collected experimental results to a JSON file.

        Args:
            results: The dictionary containing all collected experimental results.
            filename: The name of the file to save the results to.
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Cannot save results.")

        output_path = os.path.join(self.config.output_dir, filename)
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Raw results saved to {output_path}")
        except IOError as e:
            print(f"Error saving results to {output_path}: {e}")
            raise

    def run(self) -> None:
        """
        The main entry point to execute the entire reproduction pipeline.
        """
        self.load_configuration()
        self.setup_experiment_environment()
        self.run_experiments()
        self.save_results(self.all_results, 'raw_results.json')

        # Generate and save plots
        evaluator = Evaluator(config=self.config)
        evaluator.plot_results(self.all_results, 'convergence_plot.png')
        print("Reproduction pipeline finished.")

if __name__ == '__main__':
    # Ensure PyYAML is installed for configuration parsing
    try:
        import yaml
    except ImportError:
        print("PyYAML not found. Please install it using 'pip install PyYAML'")
        exit(1)
        
    main_runner = Main()
    main_runner.run()

