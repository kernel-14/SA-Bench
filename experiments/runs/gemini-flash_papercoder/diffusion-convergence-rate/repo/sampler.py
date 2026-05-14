import numpy as np
import random
from typing import Dict, Any, List, Tuple

# Assuming Config and GaussianDiffusionProcess are available from their respective modules
# To avoid circular imports for type hinting, we can use forward references or import them directly if no cycles are formed.
# For this specific file, it's fine to import them directly as Sampler won't be imported by them.
from config import Config
from diffusion_process import GaussianDiffusionProcess

class Sampler:
    """
    Implements the core randomized midpoint diffusion model sampling algorithm
    described in the paper.

    It orchestrates the forward and reverse processes, handling the randomized
    schedule, iterative updates, and noise injection to generate samples.
    """

    def __init__(self, config: Config, diffusion_process: GaussianDiffusionProcess, T_total: int, d: int):
        """
        Initializes the Sampler with configurations and dependencies.

        Args:
            config: An instance of the Config class, providing experimental parameters.
            diffusion_process: An instance of GaussianDiffusionProcess for score evaluations.
            T_total: The total number of iterations (T) for the current experiment run.
            d: The dimensionality of the data.
        """
        self.config: Config = config
        self.diffusion_process: GaussianDiffusionProcess = diffusion_process
        self.T_total: int = T_total
        self.d: int = d

        self.K_rounds: int = self.config.K_rounds
        self.c0: float = self.config.c0
        self.c1: float = self.config.c1

        # N = 2 * T / K as per paper (Appendix A)
        if (2 * self.T_total) % self.K_rounds != 0:
            raise ValueError(f"2 * T_total ({2 * self.T_total}) must be divisible by K_rounds ({self.K_rounds}) "
                             "to get an integer N_steps_per_round.")
        self.N_steps_per_round: int = (2 * self.T_total) // self.K_rounds

        self.alpha_grid: Dict[int, float] = {}  # Stores pre-computed hat_alpha_t values
        self._build_alpha_grid()

        # Cache for sampled tau and hat_tau values for each round (k)
        self.sampled_tau_cache: Dict[int, np.ndarray] = {}
        self.sampled_hat_tau_cache: Dict[int, np.ndarray] = {}

        self.last_tau_k0: float = 0.0 # To store the tau_K,0 for the last round for evaluation

    def _build_alpha_grid(self) -> None:
        """
        Pre-computes the sequence of hat_alpha_t values as defined in Equation 9.
        The indices 't' for hat_alpha_t can be negative, as implied by
        hat_tau_k,n = 1 - hat_alpha_{T - kN/2 - n}.
        """
        # Determine the range of indices 't' for alpha_grid.
        # Max index: T+1 (when k=0, n=-1 => T - 0 - (-1) = T+1)
        max_alpha_grid_idx = self.T_total + 1

        # Min index: T - (K-1)N/2 - N (when k=K-1, n=N)
        min_alpha_grid_idx = int(self.T_total - (self.K_rounds - 1) * self.N_steps_per_round / 2 - self.N_steps_per_round)

        # Initialize the smallest hat_alpha (largest index t)
        self.alpha_grid[max_alpha_grid_idx] = 1.0 / (self.T_total ** self.c0)

        # Iterate backwards to compute other hat_alpha values using Equation 9
        # hat_alpha_{t-1} = hat_alpha_t + c1 * hat_alpha_t * (1 - hat_alpha_t) * log(T) / T
        # So, we compute hat_alpha_{t_current-1} from hat_alpha_{t_current}.
        # The loop should go from max_alpha_grid_idx down to min_alpha_grid_idx + 1.
        for t_current in range(max_alpha_grid_idx, min_alpha_grid_idx, -1):
            alpha_t = self.alpha_grid[t_current]
            alpha_t_minus_1 = alpha_t + (self.c1 * alpha_t * (1.0 - alpha_t) * np.log(self.T_total)) / self.T_total
            self.alpha_grid[t_current - 1] = alpha_t_minus_1
        
        # Verify if minimum index was actually reached and handled
        if min_alpha_grid_idx not in self.alpha_grid:
            # Fallback if the loop logic wasn't fully inclusive to the min_alpha_grid_idx,
            # or if min_alpha_grid_idx happened to be max_alpha_grid_idx
            # For robustness, ensure all required indices are generated.
            # If min_alpha_grid_idx is smaller than max_alpha_grid_idx, the loop above should have covered it.
            # If min_alpha_grid_idx is even lower, we extend it.
            if min_alpha_grid_idx < min(self.alpha_grid.keys()):
                for t_current in range(min(self.alpha_grid.keys()), min_alpha_grid_idx, -1):
                    alpha_t = self.alpha_grid[t_current]
                    alpha_t_minus_1 = alpha_t + (self.c1 * alpha_t * (1.0 - alpha_t) * np.log(self.T_total)) / self.T_total
                    self.alpha_grid[t_current - 1] = alpha_t_minus_1

    def _sample_randomized_taus_for_round(self, k_round: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the randomized tau_k,n and corresponding hat_tau_k,n values
        for a specific round k_round. These are stored in internal caches.

        Args:
            k_round: The current round index (0 to K_rounds).

        Returns:
            A tuple containing:
                - np.ndarray: An array of sampled tau_k,n values for n=0 to N.
                - np.ndarray: An array of hat_tau_k,n values for n=-1 to N.
        """
        if k_round in self.sampled_tau_cache:
            return self.sampled_tau_cache[k_round], self.sampled_hat_tau_cache[k_round]

        # hat_tau_values array needs space for n=-1 up to n=N
        # So, it will have N_steps_per_round + 2 elements.
        # hat_tau_values[0] will store hat_tau_k,-1
        # hat_tau_values[n+1] will store hat_tau_k,n
        hat_tau_values = np.zeros(self.N_steps_per_round + 2, dtype=np.float64)

        # tau_values array needs space for n=0 up to n=N
        # So, it will have N_steps_per_round + 1 elements.
        # tau_values[n] will store tau_k,n
        tau_values = np.zeros(self.N_steps_per_round + 1, dtype=np.float64)

        base_alpha_idx_for_round_k = self.T_total - k_round * self.N_steps_per_round / 2

        # Calculate hat_tau_k,n for n = -1 to N_steps_per_round
        for n_idx in range(-1, self.N_steps_per_round + 1):
            alpha_grid_key = int(base_alpha_idx_for_round_k - n_idx)
            if alpha_grid_key not in self.alpha_grid:
                # This should ideally not happen if _build_alpha_grid is correct
                # and min/max indices are well-defined.
                raise KeyError(f"Alpha grid key {alpha_grid_key} not found for T_total={self.T_total}, K={self.K_rounds}, N={self.N_steps_per_round}. "
                               f"k_round={k_round}, n_idx={n_idx}. Max alpha key:{max(self.alpha_grid.keys())}, min alpha key:{min(self.alpha_grid.keys())}")
            
            hat_tau_val = 1.0 - self.alpha_grid[alpha_grid_key]
            hat_tau_values[n_idx + 1] = hat_tau_val # Shift index for 0-based array

        # Sample tau_k,n for n = 0 to N_steps_per_round
        for n_idx in range(self.N_steps_per_round + 1):
            # tau_k,n ~ Unif(hat_tau_k,n, hat_tau_k,n-1)
            # In our 0-based hat_tau_values array:
            # hat_tau_k,n is hat_tau_values[n_idx + 1]
            # hat_tau_k,n-1 is hat_tau_values[n_idx]
            lower_bound = hat_tau_values[n_idx + 1]
            upper_bound = hat_tau_values[n_idx]
            
            # Bounds check to prevent numerical issues or invalid intervals
            if lower_bound > upper_bound:
                # Swap if bounds are inverted due to precision or specific schedule choices
                lower_bound, upper_bound = upper_bound, lower_bound
            
            # If the interval is too small or effectively zero, just take one of the bounds
            if np.isclose(lower_bound, upper_bound):
                 tau_val = lower_bound
            else:
                 tau_val = np.random.uniform(lower_bound, upper_bound)
            
            tau_values[n_idx] = tau_val

        self.sampled_tau_cache[k_round] = tau_values
        self.sampled_hat_tau_cache[k_round] = hat_tau_values
        return tau_values, hat_tau_values

    def run_sampler_single_pass(self) -> np.ndarray:
        """
        Executes a single full sampling pass (K rounds) to generate one sample Y_K.

        Returns:
            np.ndarray: The final generated sample Y_K.
        """
        # 1. Initialization: Y_0 ~ N(0, I_d)
        current_y = np.random.normal(0, 1, self.d)

        # Pre-sample all randomized taus for all rounds (including K_rounds for last_tau_k0)
        # This is because tau_{k+1,0} is needed for noise injection in round k.
        for k_round_precompute in range(self.K_rounds + 1):
            self._sample_randomized_taus_for_round(k_round_precompute)

        for k_round in range(self.K_rounds):
            tau_values_curr_round = self.sampled_tau_cache[k_round]
            hat_tau_values_curr_round = self.sampled_hat_tau_cache[k_round]

            # Y_k,0 is current_y from previous round or initial N(0,I)
            y_k_0 = current_y
            
            # y_internal_history stores Y_k,0, Y_k,1, ..., Y_k,N
            # y_internal_history[0] = Y_k,0
            # y_internal_history[n] = Y_k,n
            y_internal_history = [np.zeros(self.d, dtype=np.float64)] * (self.N_steps_per_round + 1)
            y_internal_history[0] = y_k_0

            # Iterate for n = 1 to N_steps_per_round (Equation 10)
            for n in range(1, self.N_steps_per_round + 1):
                # Common denominator (1 - tau_{k,n}) from LHS
                sqrt_1_minus_tau_kn = np.sqrt(1.0 - tau_values_curr_round[n])
                
                # Term 1: Y_k,0 / sqrt(1 - tau_k,0)
                term1_val = y_k_0 / np.sqrt(1.0 - tau_values_curr_round[0])

                # Term 2: s_{T - kN/2 + 1}(Y_k,0) / (2 * (1 - tau_k,0)^1.5) * (tau_k,0 - hat_tau_k,0)
                s_arg_t_idx_y_k0 = int(self.T_total - k_round * self.N_steps_per_round / 2 + 1)
                score_tau_for_y_k0 = 1.0 - self.alpha_grid[s_arg_t_idx_y_k0]
                score_y_k0 = self.diffusion_process.get_exact_score(y_k_0, score_tau_for_y_k0)
                
                # hat_tau_values_curr_round[1] corresponds to hat_tau_k,0
                term2_val = score_y_k0 / (2.0 * (1.0 - tau_values_curr_round[0])**1.5) * \
                            (tau_values_curr_round[0] - hat_tau_values_curr_round[1])

                # Summation Term: sum_{i=1}^{n-1} ...
                sum_term_val = np.zeros(self.d, dtype=np.float64)
                for i_sum in range(1, n):
                    s_arg_t_idx_y_ki = int(self.T_total - k_round * self.N_steps_per_round / 2 - i_sum + 1)
                    score_tau_for_y_ki = 1.0 - self.alpha_grid[s_arg_t_idx_y_ki]
                    score_y_ki = self.diffusion_process.get_exact_score(y_internal_history[i_sum], score_tau_for_y_ki)
                    
                    # hat_tau_values_curr_round[i_sum] corresponds to hat_tau_k,i-1
                    # hat_tau_values_curr_round[i_sum+1] corresponds to hat_tau_k,i
                    sum_term_val += score_y_ki / (2.0 * (1.0 - tau_values_curr_round[i_sum])**1.5) * \
                                    (hat_tau_values_curr_round[i_sum] - hat_tau_values_curr_round[i_sum+1])

                # Last Term: s_{T - kN/2 - n + 2}(Y_k,n-1) / (2 * (1 - tau_k,n-1)^1.5) * (hat_tau_k,n-1 - tau_k,n)
                s_arg_t_idx_y_kn1 = int(self.T_total - k_round * self.N_steps_per_round / 2 - n + 2)
                score_tau_for_y_kn1 = 1.0 - self.alpha_grid[s_arg_t_idx_y_kn1]
                score_y_kn1 = self.diffusion_process.get_exact_score(y_internal_history[n-1], score_tau_for_y_kn1)

                # hat_tau_values_curr_round[n] corresponds to hat_tau_k,n-1
                last_term_val = score_y_kn1 / (2.0 * (1.0 - tau_values_curr_round[n-1])**1.5) * \
                                (hat_tau_values_curr_round[n] - tau_values_curr_round[n])
                
                # Sum all terms for RHS of Equation 10
                rhs_eq10 = term1_val + term2_val + sum_term_val + last_term_val
                
                # Compute Y_k,n
                y_internal_history[n] = rhs_eq10 * sqrt_1_minus_tau_kn
            
            # After inner loop, y_internal_history[self.N_steps_per_round] is Y_k,N
            y_k_N = y_internal_history[self.N_steps_per_round]

            # 3. Noise Injection (Equation 11) to get Y_{k+1}
            # For the next round's starting tau: tau_k+1,0
            tau_next_round_k0 = self.sampled_tau_cache[k_round + 1][0]
            
            z_k = np.random.normal(0, 1, self.d)

            # Denominator for square roots in Eq. 11: 1 - tau_k,N
            denominator_sqrt = 1.0 - tau_values_curr_round[self.N_steps_per_round]
            
            # The paper writes Y_{k+1} = sqrt(...) * Y_{k,N} + sqrt(...) * Z_k
            sqrt_factor_y_kn = np.sqrt((1.0 - tau_next_round_k0) / denominator_sqrt)
            sqrt_factor_z_k = np.sqrt((tau_next_round_k0 - tau_values_curr_round[self.N_steps_per_round]) / denominator_sqrt)

            current_y = sqrt_factor_y_kn * y_k_N + sqrt_factor_z_k * z_k
            
            # Store the tau_K,0 for the very last round for evaluation
            if k_round == self.K_rounds - 1:
                self.last_tau_k0 = tau_next_round_k0

        return current_y

    def get_last_tau_k0(self) -> float:
        """
        Returns the tau_K,0 value that was sampled for the final round (K_rounds).
        This is needed for the reference distribution q_K in evaluation.

        Returns:
            float: The tau_K,0 value for the last round.
        """
        return self.last_tau_k0

# Example usage for testing purposes (not part of the final module logic)
if __name__ == '__main__':
    from utils import setup_rng

    # Mock Config and GaussianDiffusionProcess for local testing
    class MockConfig:
        def __init__(self):
            self.K_rounds = 10
            self.c0 = 12.0
            self.c1 = 61.0
            self.rng_seed = 42
            self.num_samples_per_run = 100

    class MockGaussianDiffusionProcess:
        def __init__(self, d, k, rng_seed):
            self.d = d
            self.rng = np.random.default_rng(rng_seed)
            self.sigma0 = np.eye(d) # For simplicity, identity covariance
        def get_exact_score(self, x: np.ndarray, current_tau: float) -> np.ndarray:
            # Simple mock: score = -x / current_tau
            return -x / current_tau
        def get_qK_covariance(self, last_tau_k0: float) -> np.ndarray:
            return np.eye(self.d) # Mock
        def get_sigma0(self):
            return self.sigma0

    # Setup RNG
    setup_rng(42)

    # Experiment parameters for this test
    test_d = 5
    test_k = 5
    test_T_total = 1000 # Example T_total. N = 2*1000/10 = 200

    mock_config = MockConfig()
    mock_diffusion_process = MockGaussianDiffusionProcess(d=test_d, k=test_k, rng_seed=mock_config.rng_seed)

    sampler = Sampler(mock_config, mock_diffusion_process, test_T_total, test_d)

    print(f"N_steps_per_round: {sampler.N_steps_per_round}")
    print(f"Alpha grid size: {len(sampler.alpha_grid)}")
    # print(f"Alpha grid keys: {sorted(sampler.alpha_grid.keys())}")
    print(f"Alpha grid min value (at max_idx): {sampler.alpha_grid[max(sampler.alpha_grid.keys())]}")
    print(f"Alpha grid max value (at min_idx): {sampler.alpha_grid[min(sampler.alpha_grid.keys())]}")


    # Test sampling a single pass
    print("\nRunning a single sampler pass...")
    final_sample = sampler.run_sampler_single_pass()
    print(f"Generated Y_K sample shape: {final_sample.shape}")
    print(f"Generated Y_K sample (first 5 elements): {final_sample[:5]}")
    print(f"Last tau_K,0 value: {sampler.get_last_tau_k0()}")

    # Test cache
    _ = sampler._sample_randomized_taus_for_round(0)
    print(f"Cached tau values for round 0 (first 5): {sampler.sampled_tau_cache[0][:5]}")
    print(f"Cached hat_tau values for round 0 (first 5): {sampler.sampled_hat_tau_cache[0][:5]}")

    # Check bounds of tau values (should be sorted descending within interval)
    hat_tau_0_0 = sampler.sampled_hat_tau_cache[0][1] # hat_tau_k,0
    hat_tau_0_neg1 = sampler.sampled_hat_tau_cache[0][0] # hat_tau_k,-1
    tau_0_0 = sampler.sampled_tau_cache[0][0]
    print(f"hat_tau_0,0: {hat_tau_0_0}, hat_tau_0,-1: {hat_tau_0_neg1}, tau_0,0: {tau_0_0}")
    assert hat_tau_0_0 <= tau_0_0 <= hat_tau_0_neg1, "tau_0,0 not within expected range"
    print("Sampler test completed successfully (basic checks).")

