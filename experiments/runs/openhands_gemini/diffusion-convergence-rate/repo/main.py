
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from config import Config
from data import GaussianTargetDistribution
from models import ScoreFunction
from sampler import DiffusionSampler
from metrics import Metrics

def get_gaussian_moments(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the mean and covariance matrix of a set of samples.
    """
    mean = torch.mean(samples, dim=0)
    cov = torch.cov(samples.T) # samples.T because torch.cov expects (features, samples)
    return mean, cov

def run_numerical_experiment():
    """
    Runs the numerical experiment as described in Appendix A.
    - Selects a Gaussian target distribution.
    - Implements the proposed sampler with K=10 and N=2T/K.
    - Computes KL divergence between output Y_K and target q_K for different T.
    - Plots the empirical results against the theoretical rate.
    """
    print("Starting numerical experiment...")

    # Configuration
    d_dim = Config.D_DIM
    k_gaussian_components = Config.K_GAUSSIAN_COMPONENTS
    diag_var_range = Config.DIAG_VAR_RANGE
    c0 = Config.C0
    c1 = Config.C1
    seed = Config.SEED

    # Setup target distribution (p_0)
    target_distribution_p0 = GaussianTargetDistribution(d_dim, k_gaussian_components, diag_var_range, seed)
    
    # Extract true sigma_0 from the target distribution for the score function.
    # Note: target_distribution_p0.sigma_0 are the diagonal variances of X_0.
    score_function = ScoreFunction(d_dim, target_distribution_p0.sigma_0)

    # Vary T_total to observe convergence rate, as in Figure 2.
    # The paper uses T as "number of iterations", and plots against T.
    # Let's use a range of T_total values. Example T values from Figure 2:
    # (a) d=10, k=10
    # (b) d=100, k=10
    # (c) d=500, k=100
    # The x-axis is "Number of iterations T". This T refers to the T in `T_total`.
    
    # Let's choose a range of T_total values similar to a log scale for better plotting
    T_total_values = [100, 200, 400, 800, 1600, 3200, 6400, 12800] # Example range
    
    # In Appendix A, K=10 is fixed for the sampler.
    fixed_K_rounds = 10 
    
    # Number of samples for Monte Carlo estimation of final distribution statistics (Y_K, q_K)
    num_eval_samples = 10000

    kl_divergences = []
    tv_distances = []

    print(f"Running for d_dim={d_dim}, k_gaussian_components={k_gaussian_components}, K_rounds={fixed_K_rounds}")

    for T in tqdm(T_total_values, desc="Running simulations for T_total"):
        # N_steps for this T_total and K_rounds
        N_steps = Config.get_N_steps(T, fixed_K_rounds)
        
        # Initialize sampler for current T
        sampler = DiffusionSampler(d_dim, T, fixed_K_rounds, score_function, c0, c1, seed)
        
        # Generate samples Y_K from the sampler
        y_k_samples = sampler.sample(num_eval_samples)

        # Get moments for Y_K (empirical mean and covariance)
        mean_y_k, cov_y_k = get_gaussian_moments(y_k_samples)

        # Determine the target distribution for KL divergence (q_K)
        # q_K is the distribution of X_tau_K,0 which is approx X_1 (the starting point of forward process)
        # From Lemma 1: X_tau = sqrt(1-tau) * X_0 + sqrt(tau) * Z
        # q_K corresponds to X_tau_K,0.
        # tau_K,0 = 1 - overline_alpha_1 (approx, from paper text where q_K is approximately X_1)
        # Let's get the *true* X_tau_K,0 distribution moments.
        # From Lemma 1, `X_tau_k,0` is distributed as `sqrt(1 - tau_k,0) X_0 + sqrt(tau_k,0) Z`.
        # For `q_K`, it is `X_tau_K,0`.
        # Its mean is `sqrt(1 - tau_K,0) * mean_X_0 = 0`.
        # Its covariance is `(1 - tau_K,0) * Cov_X_0 + tau_K,0 * I_d`.
        
        # To get `tau_K,0`, we need `overline_alpha_1` from the schedule.
        # The true `tau_K,0` (which is 1 - `overline_alpha_1`) is random,
        # but for theoretical comparison, we need a deterministic `q_K`.
        # Appendix A: "KL divergence between Y_k,0 and X_1 has a closed-form expression."
        # This implies `X_1` is a fixed reference.
        # `X_1 = sqrt(alpha_1) X_0 + sqrt(1-alpha_1) W_1`.
        # If `alpha_1` is fixed, `overline_alpha_1 = alpha_1`.
        # The paper's definition of `tau_k,n := 1 - overline_alpha_{T - kN/2 - n + 1}`
        # For `q_K`, it refers to the target distribution `p_{X_tau_K,0}`.
        # The `tau_K,0` value is randomized. For plotting against a theoretical rate,
        # we might need to use the expected/average `tau_K,0` or a specific deterministic value.
        
        # Let's use the deterministic `hat_tau_K,0` as the reference for `q_K`.
        # hat_tau_K,0 = 1 - hat_alpha_{T - KN/2 - 0} = 1 - hat_alpha_T_minus_KN_div_2
        # T - KN/2 is T - T = 0. So hat_tau_K,0 = 1 - hat_alpha_0.
        # This seems to be the `tau` for the *initial* step of the reverse process (Y_0).
        # Let's verify `q_K` again: "performance of the sampler is evaluated using the total variation (TV) distance between p_Y_K and q_K, defined as TV(q_K, p_Y_K)".
        # And "X_tau_K,0, which is nearly the starting point of the forward process"
        
        # Let's assume q_K is Gaussian with mean 0.
        # Covariance of q_K: `(1 - hat_tau_K,0) * Cov_X_0 + hat_tau_K,0 * I_d`
        # We need `hat_tau_K,0` to be used.
        # From sampler init, `_hat_tau_k_n` stores `(k, n_val)`.
        # So `hat_tau_K_0 = sampler._hat_tau_k_n[(fixed_K_rounds-1, 0)]` No, this is for the last round k=K-1.
        # q_K is based on index K for rounds, but rounds go from 0 to K-1.
        # The definition in Lemma 1 uses `hat_X_0 = X_tau_0,0` and `hat_X_k+1 = X_tau_k+1,0`.
        # So `q_K` is the distribution of `hat_X_K`, which is `X_tau_K,0`.
        
        # For `tau_K,0` (which is `1 - overline_alpha_j` where `j = T - K*N/2 - 0 + 1 = T - T + 1 = 1`).
        # So `tau_K,0 = 1 - overline_alpha_1`.
        # The paper says `overline_alpha_t` is sampled from `Unif(hat_alpha_t, hat_alpha_{t-1})`.
        # So `overline_alpha_1` is sampled from `Unif(hat_alpha_1, hat_alpha_0)`.
        # To get a deterministic `q_K` for theoretical rate plotting, we should use a deterministic `tau_K,0`.
        # Let's use `1 - hat_alpha_1` as a deterministic value for `tau_K,0`.
        
        # Index for hat_alpha_1 is 1. Index for hat_alpha_0 is 0.
        # Let's get these from sampler's precomputed dicts.
        hat_alpha_1 = sampler._hat_alpha_dict.get(1)
        hat_alpha_0 = sampler._hat_alpha_dict.get(0)
        
        if hat_alpha_1 is None or hat_alpha_0 is None:
            # Fallback if range wasn't sufficient or indices are truly undefined
            print("Warning: hat_alpha_1 or hat_alpha_0 not found for deterministic q_K. Using approximation.")
            # If not defined, means schedule did not extend that far.
            # For simplicity, if we cannot get the exact hat_alpha values for t=1,0,
            # we can approximate overline_alpha_1 with a small constant.
            # But the schedule generation logic should cover it.
            # For `min_alpha_idx_for_unif = 1 - N_steps // 2` which is `-99` for T=1000.
            # `max_alpha_idx_for_unif = T_total + 2 = 1002`.
            # So `hat_alpha_0` and `hat_alpha_1` should be present.
            pass
        
        # Use average of bounds for deterministic overline_alpha_1
        deterministic_overline_alpha_1 = (hat_alpha_1 + hat_alpha_0) / 2.0
        deterministic_tau_K_0 = 1.0 - deterministic_overline_alpha_1

        mean_q_k = torch.zeros(d_dim)
        cov_x_0 = target_distribution_p0.covariance_matrix
        cov_q_k = (1.0 - deterministic_tau_K_0) * cov_x_0 + deterministic_tau_K_0 * torch.eye(d_dim)

        # Calculate KL divergence and TV distance
        kl_div = Metrics.kl_divergence_gaussian(mean_q_k, cov_q_k, mean_y_k, cov_y_k)
        tv_dist = Metrics.tv_distance_from_kl(kl_div)

        kl_divergences.append(kl_div.item())
        tv_distances.append(tv_dist.item())
        
        # print(f"T={T}, KL={kl_div.item():.6e}, TV={tv_dist.item():.6e}")

    # Plotting results (Figure 2 reproduction)
    plt.figure(figsize=(10, 6))
    
    # Blue line: Empirical results
    plt.plot(T_total_values, kl_divergences, 'b-o', label='Empirical KL Divergence')

    # Black line: Theoretical rate O(log^4 T / T^3)
    # The paper says: "Our theoretical analysis predicts a convergence rate of O(poly(log T) / T^3)
    # in terms of KL divergence, which is consistent with empirical observations."
    # "This further confirms that our sampler achieves a KL divergence convergence rate of O(log^4 T / T^3)
    # in terms of KL divergence, implying a total variation(TV) distance convergence rate of O(log^2 T / T^3/2)."
    # So plot theoretical rate for KL.
    
    # We need to scale the theoretical curve to match the empirical data's magnitude for visualization.
    # Pick a scaling factor from one of the empirical points.
    if len(T_total_values) > 1 and len(kl_divergences) > 1:
        # Use a scaling factor based on the first data point
        # A simple linear scaling will do for visualization purposes.
        T_ref = T_total_values[1] # Choose second point to avoid very small T values that might be unstable
        kl_ref = kl_divergences[1]
        
        theoretical_rate_unscaled = (np.log(T_total_values[1])**4) / (T_total_values[1]**3)
        scaling_factor = kl_ref / theoretical_rate_unscaled if theoretical_rate_unscaled > 1e-12 else 1.0
        
        theoretical_kl_rate = [scaling_factor * (np.log(T)**4) / (T**3) for T in T_total_values]
        
        plt.plot(T_total_values, theoretical_kl_rate, 'k--', label=r'Theoretical Rate $O(\log^4 T / T^3)$')
    else:
        print("Not enough data points to plot theoretical rate.")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of iterations T (log scale)")
    plt.ylabel("KL Divergence (log scale)")
    plt.title(f"KL Divergence vs. T (d={d_dim}, k={k_gaussian_components})")
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    # Save the plot
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"kl_divergence_d{d_dim}_k{k_gaussian_components}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    print("\nExperiment finished.")

if __name__ == "__main__":
    run_numerical_experiment()
