
import numpy as np

class Config:
    # Simulation parameters
    D_DIM = 10  # Data dimension, d
    T_TOTAL = 1000  # Total iterations for the forward process (often denoted as T in diffusion models literature)
                    # This T is used in schedule definitions (Eq 8, 9) and iteration complexity.
    K_ROUNDS = 10  # Number of rounds, K, for the sampler (Section 2.2)
    # N_STEPS is calculated from T_TOTAL and K_ROUNDS, N = 2 * T_TOTAL / K_ROUNDS
    
    # Schedule constants (from Eq 8)
    C0 = 1.0  # c_0, sufficiently large constant
    C1 = 5.0  # c_1, ratio c1/c0 assumed to be sufficiently large (e.g., c1 > 5*c0)

    # Gaussian target distribution parameters (for numerical experiments in Appendix A)
    # The first K_GAUSSIAN_COMPONENTS diagonal entries of Sigma_0 are uniformly distributed in DIAG_VAR_RANGE.
    # The remaining D_DIM - K_GAUSSIAN_COMPONENTS entries are set to 0 (as per example description).
    K_GAUSSIAN_COMPONENTS = 5  # 'k' in Appendix A's description, e.g., 5 means first 5 entries vary
    DIAG_VAR_RANGE = [0.1, 10.0] # Range for first k_components diagonal entries, [0, 10] mentioned in paper
    
    # Theoretical constants (not directly used in simulation but defined in paper)
    EPSILON = 1e-3  # Target output accuracy in Total Variation (TV) distance
    THETA = 10.0 # Sufficiently large constant for typical sets (Section 4.1)
    C_R = 1.0 # Constant for bounded second-order moment (Assumption 1)

    # General settings
    SEED = 42 # Random seed for reproducibility
    
    @staticmethod
    def get_N_steps(T_total: int, K_rounds: int) -> int:
        """
        Calculates the number of steps per round, N.
        The paper states "each consisting of N = 2*I/K steps" (where I is T_total).
        And "total iteration complexity of the sampler is K*N = 2*T".
        So N_steps = 2 * T_total / K_rounds.
        """
        return int(2 * T_total / K_rounds)

    @staticmethod
    def get_tau_k_n_index(T_total: int, K_rounds: int, k: int, n: int) -> int:
        """
        Calculates the index for hat_alpha (and overline_alpha) from the definition of hat_tau_k,n (Eq 9).
        hat_tau_k,n := 1 - hat_alpha_{T - kN/2 - n}
        So the index is T - kN/2 - n.
        """
        N_steps = Config.get_N_steps(T_total, K_rounds)
        return T_total - (k * N_steps // 2) - n

    @staticmethod
    def get_alpha_bar_t_index(T_total: int, K_rounds: int, k: int, n: int) -> int:
        """
        Calculates the index for overline_alpha from the definition of tau_k,n (Eq 9).
        tau_k,n := 1 - overline_alpha_{T - kN/2 - n + 1}
        So the index is T - kN/2 - n + 1.
        """
        N_steps = Config.get_N_steps(T_total, K_rounds)
        return T_total - (k * N_steps // 2) - n + 1
