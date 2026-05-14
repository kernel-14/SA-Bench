from typing import Iterator, Dict, Any, List

class Config:
    """
    Manages and provides access to all experimental parameters.

    This class serves as a single source of truth for configuration throughout
    the reproduction pipeline, ensuring consistency and ease of modification.
    It loads parameters from a specified source (intended to be config.yaml
    parsed by the main script) and provides a structured way to iterate
    through experimental configurations.
    """

    def __init__(self,
                 T_total_range: List[int],
                 d_values: List[int],
                 k_values: List[int],
                 num_samples_per_run: int,
                 K_rounds: int,
                 c0: float,
                 c1: float,
                 output_dir: str,
                 rng_seed: int = 42):
        """
        Initializes the Config object with experimental parameters.

        Args:
            T_total_range: A list of total iteration (T) values to experiment with.
            d_values: A list of data dimension (d) values to experiment with.
            k_values: A list of active dimensions (k) values to experiment with,
                      corresponding one-to-one with d_values.
            num_samples_per_run: The number of samples Y_K to generate for each
                                 experimental run to estimate its distribution.
            K_rounds: The fixed number of rounds (K) for the sampler.
            c0: Constant c0 used in the randomized schedule definition.
            c1: Constant c1 used in the randomized schedule definition.
            output_dir: Directory path where results and plots will be saved.
            rng_seed: Seed for the random number generator to ensure reproducibility.
        """
        if len(d_values) != len(k_values):
            raise ValueError("d_values and k_values must have the same length as they are paired.")

        self.T_total_range: List[int] = T_total_range
        self.d_values: List[int] = d_values
        self.k_values: List[int] = k_values
        self.num_samples_per_run: int = num_samples_per_run
        self.K_rounds: int = K_rounds
        self.c0: float = c0
        self.c1: float = c1
        self.output_dir: str = output_dir
        self.rng_seed: int = rng_seed

    def get_experiment_parameters(self) -> Iterator[Dict[str, Any]]:
        """
        Generates a sequence of dictionaries, each representing a unique
        combination of T_total, d, and k for an experimental run.

        The method iterates through all specified T_total values and
        paired (d, k) values, yielding a dictionary for each combination.

        Yields:
            A dictionary containing 'T_total', 'd', and 'k' for a single
            experimental configuration.
        """
        for T_total_val in self.T_total_range:
            for d_val, k_val in zip(self.d_values, self.k_values):
                yield {
                    'T_total': T_total_val,
                    'd': d_val,
                    'k': k_val
                }

# Example usage (for testing purposes, not part of the final module logic)
if __name__ == '__main__':
    # This block would typically be in main.py, parsing config.yaml
    # For demonstration, we use hardcoded values here.
    test_config_params = {
        'T_total_range': [100, 200],
        'd_values': [10, 100],
        'k_values': [10, 10],
        'num_samples_per_run': 1000,
        'K_rounds': 10,
        'c0': 12.0,
        'c1': 61.0,
        'output_dir': "test_results",
        'rng_seed': 42
    }

    config = Config(**test_config_params)

    print("--- Experiment Parameters ---")
    print(f"K_rounds: {config.K_rounds}")
    print(f"c0: {config.c0}")
    print(f"c1: {config.c1}")
    print(f"Output directory: {config.output_dir}")
    print(f"RNG Seed: {config.rng_seed}")
    print(f"Num samples per run: {config.num_samples_per_run}")

    print("\n--- Iterating through experiment configurations ---")
    for i, exp_params in enumerate(config.get_experiment_parameters()):
        print(f"Experiment {i+1}: {exp_params}")

    # Example of invalid config for validation check
    try:
        invalid_config = Config(
            T_total_range=[100],
            d_values=[10, 20],
            k_values=[10],
            num_samples_per_run=1000,
            K_rounds=10,
            c0=12.0,
            c1=61.0,
            output_dir="test_results",
            rng_seed=42
        )
    except ValueError as e:
        print(f"\nCaught expected error for invalid config: {e}")
