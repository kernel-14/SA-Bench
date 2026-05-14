import numpy as np
from .score_function import ScoreFunction

class DiffusionSampler:
    """
    Implements the randomized midpoint sampling technique for diffusion models.
    """

    def __init__(self, d: int, T_diffusion: int, K: int, score_function: ScoreFunction,
                 c0: float = 1.0, c1: float = 1.0, seed: int = None):
        """
        Initializes the DiffusionSampler.

        Args:
            d (int): Data dimension.
            T_diffusion (int): Total number of diffusion steps (T in the paper's equations like hat_alpha_T+1).
            K (int): Number of rounds in the sampling procedure.
            score_function (ScoreFunction): An instance of a ScoreFunction.
            c0 (float): Constant c0 from equation (7).
            c1 (float): Constant c1 from equation (7).
            seed (int): Random seed for reproducibility.
        """
        self.d = d
        self.T_diffusion = T_diffusion
        self.K = K
        # N = 2 * T_diffusion / K from "total iteration complexity of the sampler is KN = 2T"
        # Assuming T_diffusion is the "T" in "KN = 2T"
        self.N = int(2 * T_diffusion / K)
        if self.N * self.K != 2 * self.T_diffusion:
             # If T_diffusion doesn't allow for an integer N, there's a mismatch.
             # For a reproduction, we proceed with integer N, which might slightly deviate if T_diffusion isn't a multiple of K/2.
             # In a strict implementation, one might want to adjust T_diffusion or K to ensure this. For now, silently accept.
             pass
        self.score_function = score_function
        self.c0 = c0
        self.c1 = c1
        self.rng = np.random.default_rng(seed)

        self.hat_alpha_schedule = {}
        self.bar_alpha_schedule = {}
        self.hat_tau_schedule = {} # Dictionary: (k, n) -> float
        self.tau_schedule = {} # Dictionary: (k, n) -> float

        self._initialize_schedules()

    def _initialize_schedules(self):
        """
        Initializes the randomized schedule parameters: hat_alpha_t, bar_alpha_t,
        hat_tau_k_n, and tau_k_n based on equations (7), (8), and (9).
        """
        log_T_diffusion = np.log(self.T_diffusion) if self.T_diffusion > 1 else 0.0

        # Determine the full range of indices required for hat_alpha and bar_alpha.
        # Max index for bar_alpha is T_diffusion + 2 (from k=0, n=-1 in tau_k,n definition)
        # Min index for bar_alpha is 1 - N/2 (from k=K-1, n=N in tau_k,n definition)
        # For bar_alpha[t_idx] = rng.uniform(hat_alpha[t_idx], hat_alpha[t_idx-1]),
        # hat_alpha needs to be defined from  to .
        
        min_bar_alpha_idx = 1 - self.N // 2
        max_bar_alpha_idx = self.T_diffusion + 2

        min_hat_alpha_idx = min_bar_alpha_idx - 1
        max_hat_alpha_idx = max_bar_alpha_idx

        # --- Calculate hat_alpha_t (Equation 7) ---
        # hat_alpha_T+1 = 1 / (T^c0)
        # hat_alpha_t-1 = hat_alpha_t + (c1 * hat_alpha_t * (1 - hat_alpha_t) * log(T)) / T

        # Initialize the largest required hat_alpha value
        self.hat_alpha_schedule[max_hat_alpha_idx] = 1.0 / (self.T_diffusion ** self.c0)

        # Iterate backward to compute other hat_alpha values
        for t_idx in range(max_hat_alpha_idx, min_hat_alpha_idx, -1):
            current_hat_alpha = self.hat_alpha_schedule[t_idx]
            
            # The paper defines alpha_t in (0,1). The recursive definition (7) may lead to values outside this range
            # if c1 is too large or T is too small. For faithful reproduction, we follow the formula.
            # A practical implementation might clip or re-parameterize.
            next_hat_alpha = current_hat_alpha +                                                   (self.c1 * current_hat_alpha * (1 - current_hat_alpha) * log_T_diffusion) / self.T_diffusion
            self.hat_alpha_schedule[t_idx - 1] = next_hat_alpha
            
        # --- Calculate bar_alpha_t (Equation 8) ---
        # bar_alpha_t ~ Unif(hat_alpha_t, hat_alpha_t-1)
        for t_idx in range(min_bar_alpha_idx, max_bar_alpha_idx + 1):
            lower_bound = self.hat_alpha_schedule[t_idx]
            upper_bound = self.hat_alpha_schedule[t_idx - 1]
            
            # Ensure lower_bound <= upper_bound for uniform sampling.
            # Based on equation (7), hat_alpha_t-1 should be greater than hat_alpha_t if c1 > 0 and 0 < hat_alpha_t < 1.
            # This ensures upper_bound > lower_bound. If not, there's a problem with schedule parameters.
            if lower_bound > upper_bound:
                # This indicates an issue. For this reproduction, we'll swap them to avoid errors,
                # but in a real setting, this would warrant investigation of parameters c0, c1, T_diffusion.
                temp = lower_bound
                lower_bound = upper_bound
                upper_bound = temp

            self.bar_alpha_schedule[t_idx] = self.rng.uniform(lower_bound, upper_bound)

        # --- Calculate hat_tau_k_n and tau_k_n (Equation 9) ---
        # hat_tau_k_n := 1 - hat_alpha_{T_diffusion - kN/2 - n}
        # tau_k_n := 1 - bar_alpha_{T_diffusion - kN/2 - n + 1}
        
        for k in range(self.K):
            for n in range(-1, self.N + 1): # n from -1 to N, inclusive
                hat_alpha_lookup_idx = self.T_diffusion - k * (self.N // 2) - n
                bar_alpha_lookup_idx = self.T_diffusion - k * (self.N // 2) - n + 1

                # Use .get with a default value (e.g., 0.0 or raise error) if indices are somehow out of range.
                # Given the careful range calculations, they should be in range.
                self.hat_tau_schedule[(k, n)] = 1.0 - self.hat_alpha_schedule.get(hat_alpha_lookup_idx, 0.0)
                self.tau_schedule[(k, n)] = 1.0 - self.bar_alpha_schedule.get(bar_alpha_lookup_idx, 0.0)


    def run_sampler(self, initial_sample: np.ndarray = None) -> np.ndarray:
        """
        Runs the diffusion sampling procedure based on equations (10) and (11).

        Args:
            initial_sample (np.ndarray, optional): Initial sample Y_0. If None,
                                                    it will be initialized from N(0, I_d).
        Returns:
            np.ndarray: The generated sample Y_K.
        """
        if initial_sample is None:
            Y_current = self.rng.normal(0, 1, size=self.d) # Y_0 ~ N(0, I_d)
        else:
            Y_current = initial_sample.copy()

        # Y_k in the paper corresponds to Y_current in this code before starting round k
        # Y_k,0 in the paper is Y_k

        for k in range(self.K):
            Y_k_n_values = {0: Y_current.copy()} # Store Y_k,n for n = 0, ..., N. Y_k,0 = Y_k

            tau_k_0 = self.tau_schedule[(k, 0)]
            hat_tau_k_0 = self.hat_tau_schedule[(k, 0)]
            
            # Iterate for n from 1 to N (Equation 10)
            for n in range(1, self.N + 1):
                tau_k_n = self.tau_schedule[(k, n)]
                tau_k_n_minus_1 = self.tau_schedule[(k, n-1)]
                
                hat_tau_k_n = self.hat_tau_schedule[(k, n)]
                hat_tau_k_n_minus_1 = self.hat_tau_schedule[(k, n-1)]

                term_A = Y_k_n_values[0] / np.sqrt(1.0 - tau_k_0)

                # Score index for the first term outside the sum in Equation 10
                # The score function s_{T - kN/2 + 1} is evaluated at Y_k,0
                score_idx_term_B = self.T_diffusion - k * (self.N // 2) + 1
                s_val_term_B = self.score_function(Y_k_n_values[0], score_idx_term_B)
                term_B = (s_val_term_B / (2.0 * (1.0 - tau_k_0)**1.5)) * (tau_k_0 - hat_tau_k_0)

                term_C_sum = np.zeros(self.d)
                # The sum is from i=1 to n-1. If n=1, this loop doesn't run, sum is 0.
                for i in range(1, n):
                    # Score function s_{T - kN/2 - i + 1} is evaluated at Y_k,i
                    score_idx_C_i = self.T_diffusion - k * (self.N // 2) - i + 1
                    s_val_C_i = self.score_function(Y_k_n_values[i], score_idx_C_i)
                    tau_k_i = self.tau_schedule[(k, i)] # This is actually 1 - bar_alpha_{T - kN/2 - i + 1}
                    hat_tau_k_i = self.hat_tau_schedule[(k, i)]
                    hat_tau_k_i_minus_1 = self.hat_tau_schedule[(k, i-1)]
                    term_C_sum += (s_val_C_i / (2.0 * (1.0 - tau_k_i)**1.5)) * (hat_tau_k_i_minus_1 - hat_tau_k_i)
                
                # Score index for the last term in Equation 10
                # The score function s_{T - kN/2 - n + 2} is evaluated at Y_k,n-1
                score_idx_term_D = self.T_diffusion - k * (self.N // 2) - n + 2
                s_val_term_D = self.score_function(Y_k_n_values[n-1], score_idx_term_D)
                term_D = (s_val_term_D / (2.0 * (1.0 - tau_k_n_minus_1)**1.5)) * (hat_tau_k_n_minus_1 - tau_k_n)

                # Combine terms and re-normalize
                Y_k_n_normalized_val = term_A + term_B + term_C_sum + term_D
                Y_k_n = Y_k_n_normalized_val * np.sqrt(1.0 - tau_k_n)
                Y_k_n_values[n] = Y_k_n

            Y_k_N = Y_k_n_values[self.N] # Final Y_k,N after n-loop

            # --- Noise injection (Equation 11) ---
            if k < self.K - 1: # If not the last round (k from 0 to K-1)
                tau_k_plus_1_0 = self.tau_schedule[(k + 1, 0)]
                tau_k_N = self.tau_schedule[(k, self.N)]
                
                sqrt_term1_coeff = np.sqrt((1.0 - tau_k_plus_1_0) / (1.0 - tau_k_N))
                
                term_inside_sqrt2 = (tau_k_plus_1_0 - tau_k_N) / (1.0 - tau_k_N)
                
                # It's crucial that term_inside_sqrt2 >= 0. As analyzed earlier, tau_k_plus_1_0 should be >= tau_k_N.
                # If this condition is violated, it suggests an issue in the schedule ordering or interpretation of the paper's equations.
                if term_inside_sqrt2 < 0:
                    # Raising a ValueError to indicate a potential problem with the theoretical model or parameters.
                    raise ValueError(f"Term inside sqrt is negative: {term_inside_sqrt2}. This indicates tau_k+1,0 < tau_k,N. Check schedule logic or parameters.")

                sqrt_term2_coeff = np.sqrt(term_inside_sqrt2)
                
                Z_k = self.rng.normal(0, 1, size=self.d) # Z_k ~ N(0, I_d)
                Y_current = sqrt_term1_coeff * Y_k_N + sqrt_term2_coeff * Z_k
            else:
                Y_current = Y_k_N # Last round, Y_K is Y_K,N, no further noise injection for the final output.

        return Y_current
