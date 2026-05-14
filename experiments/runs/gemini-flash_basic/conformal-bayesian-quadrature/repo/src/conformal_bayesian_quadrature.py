import numpy as np
from dirichlet_sampling import sample_dirichlet_1

def calculate_L_plus(losses: np.ndarray, B: float, num_dirichlet_samples: int) -> np.ndarray:
    """
    Calculates samples of the random variable L^+ (Equation 27).

    Args:
        losses: A 1D numpy array of observed individual losses, \ell_1, ..., \ell_n.
        B: The maximum possible value of the loss, \ell_{n+1}.
        num_dirichlet_samples: The number of Dirichlet samples to use for Monte Carlo.

    Returns:
        A 1D numpy array of num_dirichlet_samples values of L^+.
    """
    n = len(losses)
    # Sort the observed losses to get \ell_{(1)}, ..., \ell_{(n)}
    sorted_losses = np.sort(losses)

    # Combine with B to get \ell_{(1)}, ..., \ell_{(n)}, \ell_{(n+1)} = B
    ell_ordered = np.append(sorted_losses, B)

    # Sample U_1, ..., U_{n+1} from Dir(1, ..., 1)
    # The dirichlet_sampling function takes 'n' as the number of observed losses,
    # which corresponds to n+1 components for the Dirichlet distribution.
    U_samples = sample_dirichlet_1(n, num_dirichlet_samples)

    # Calculate L^+ = sum_{i=1}^{n+1} U_i * ell_{(i)}
    # This is a matrix multiplication: (num_dirichlet_samples, n+1) @ (n+1,)
    L_plus_samples = U_samples @ ell_ordered
\    return L_plus_samples

def find_b_star_beta(L_plus_samples: np.ndarray, beta: float) -> float:
    """
    Finds the critical value b_beta^* (Corollary 4.4, Equation 29).

    Args:
        L_plus_samples: A 1D numpy array of samples of L^+.
        beta: The desired confidence level (e.g., 0.95).

    Returns:
        The critical value b_beta^*.
    """
    # b_beta^* is the beta-quantile of the L_plus_samples distribution
    return np.quantile(L_plus_samples, beta)

def find_lambda_hpd_beta(loss_function, lambda_values: np.ndarray, calibration_data, B: float, alpha: float, beta: float, num_dirichlet_samples: int) -> float:
    """
    Finds the decision rule lambda_hpd^beta (Equation 31).
    This function requires iterating over a range of lambda_values and finding
    the infimum lambda such that Pr(L^+ <= alpha | losses) >= beta.

    Args:
        loss_function: A callable that takes (data_point, lambda_val) and returns an individual loss.
                       For synthetic binomial data, this will be \ell(z_i, lambda).
        lambda_values: A 1D numpy array of lambda values to search over.
        calibration_data: The calibration dataset z_1, ..., z_n.
        B: The maximum possible value of the loss.
        alpha: The target risk threshold.
        beta: The desired confidence level for the HPD interval.
        num_dirichlet_samples: Number of samples for L^+ Monte Carlo simulation.

    Returns:
        The chosen lambda_hpd^beta.
    """
    # We need to find the smallest lambda such that Pr(L^+ <= alpha) >= beta
    # This means we are looking for the smallest lambda whose b_beta^* is <= alpha.
    # More precisely, we want infimum_lambda {lambda : b_beta^*(lambda) <= alpha}
    # Note that b_beta^*(lambda) is a non-increasing function of lambda (as a larger lambda generally leads to smaller losses, hence smaller L^+ values).

    min_lambda_satisfying_condition = float('inf')

    # Sort lambda_values to ensure we find the infimum efficiently
    sorted_lambda_values = np.sort(lambda_values)

    for current_lambda in sorted_lambda_values:
        # 1. Calculate individual losses for the current lambda
        individual_losses = np.array([loss_function(z_i, current_lambda) for z_i in calibration_data])

        # 2. Calculate L^+ samples for these losses
        L_plus_samples = calculate_L_plus(individual_losses, B, num_dirichlet_samples)

        # 3. Find b_beta^* for these L^+ samples
        b_star = find_b_star_beta(L_plus_samples, beta)

        # 4. Check if the condition is met
        if b_star <= alpha:
            min_lambda_satisfying_condition = current_lambda
            break # Since b_star is non-increasing, we found the infimum

    if min_lambda_satisfying_condition == float('inf'):
        # If no lambda satisfies the condition, it implies alpha is too low or
        # B is not large enough, or the lambda range is too small.
        # For practical purposes, we might return the largest lambda or raise an error.
        # For now, let's return the largest lambda tried if nothing satisfies.
        return sorted_lambda_values[-1] # Or some other appropriate fallback
    else:
        return min_lambda_satisfying_condition



if __name__ == '__main__':
    # Example Usage for L^+ and b_beta^*
    print("--- Testing calculate_L_plus and find_b_star_beta ---")
    np.random.seed(42) # for reproducibility

    # Simulate some observed losses
    observed_losses = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
    max_loss_B = 1.0
    num_dirichlet_samples = 10000
    confidence_beta = 0.95

    L_plus_samps = calculate_L_plus(observed_losses, max_loss_B, num_dirichlet_samples)
    print(f"First 5 L^+ samples: {L_plus_samps[:5]}")
    print(f"Mean of L^+ samples: {np.mean(L_plus_samps):.4f}")
    print(f"Std Dev of L^+ samples: {np.std(L_plus_samps):.4f}")

    b_star_val = find_b_star_beta(L_plus_samps, confidence_beta)
    print(f"b_beta^* for beta={confidence_beta}: {b_star_val:.4f}")

    # Example Usage for find_lambda_hpd_beta (simplified for demonstration)
    print("\n--- Testing find_lambda_hpd_beta (conceptual) ---")

    # Dummy loss function for demonstration
    def dummy_loss_function(data_point, lambda_val):
        # Imagine data_point is just a number, and loss decreases with lambda_val
        return max(0, data_point - lambda_val)

    dummy_calibration_data = np.array([0.2, 0.6, 0.3, 0.9, 0.4])
    dummy_lambda_values = np.linspace(0, 1, 20) # A range of lambda values to try
    dummy_B = 1.0
    dummy_alpha = 0.4 # Target risk threshold
    dummy_beta = 0.9 # Confidence level

    chosen_lambda = find_lambda_hpd_beta(
        dummy_loss_function, dummy_lambda_values, dummy_calibration_data,
        dummy_B, dummy_alpha, dummy_beta, num_dirichlet_samples
    )
    print(f"Chosen lambda_hpd^beta: {chosen_lambda:.4f}")

    # Verify the chosen lambda
    if chosen_lambda != float('inf') and chosen_lambda is not None:
        test_losses = np.array([dummy_loss_function(z_i, chosen_lambda) for z_i in dummy_calibration_data])
        test_L_plus_samps = calculate_L_plus(test_losses, dummy_B, num_dirichlet_samples)
        test_b_star = find_b_star_beta(test_L_plus_samps, dummy_beta)
        print(f"Verifying chosen lambda: b_beta^*({chosen_lambda}) = {test_b_star:.4f} vs alpha={dummy_alpha}")


