import numpy as np
from tqdm import tqdm

from conformal_bayesian_quadrature import find_lambda_hpd_beta, calculate_L_plus
from baselines import conformal_risk_control, split_conformal_prediction
from utils import binomial_loss, generate_synthetic_binomial_data, heteroskedastic_miscoverage_loss_function, heteroskedastic_miscoverage_score_function, generate_synthetic_heteroskedastic_data

def run_synthetic_binomial_experiment(M: int = 10000, n: int = 10, K: int = 4, alpha: float = 0.4, beta: float = 0.95, B: float = 1.0, num_dirichlet_samples: int = 1000) -> dict:
    """
    Runs the synthetic binomial data experiment as described in Section 5.1.

    Args:
        M: Number of data splits/trials.
        n: Number of calibration samples.
        K: Number of Bernoulli trials for binomial loss.
        alpha: Target risk threshold.
        beta: Confidence level for HPD method.
        B: Maximum possible loss.
        num_dirichlet_samples: Number of samples for L^+ Monte Carlo simulation.

    Returns:
        A dictionary containing results for CRC and HPD methods.
    """
    print(f"Running Synthetic Binomial Experiment with M={M}, n={n}, K={K}, alpha={alpha}, beta={beta}, B={B}")

    crc_lambdas = []
    hpd_lambdas = []
    true_risks_crc = []
    true_risks_hpd = []

    # Define a range of lambda values for searching. This should cover [0, 1] as binomial loss is 0 at lambda=1.
    lambda_search_space = np.linspace(0, 1.0, 101) 

    for _ in tqdm(range(M), desc="Synthetic Binomial Experiment"):
        # Generate calibration data (V_ik samples for each z_i)
        calibration_data = generate_synthetic_binomial_data(n, K)

        # --- Conformal Risk Control (CRC) ---
        # CRC Loss function for binomial data
        def crc_binomial_loss_func(V_ik_samples, lambda_val):
            return binomial_loss(V_ik_samples, lambda_val, K)

        lambda_crc = conformal_risk_control(crc_binomial_loss_func, lambda_search_space, calibration_data, B, alpha)
        crc_lambdas.append(lambda_crc)

        # True risk for binomial loss is 1 - lambda (from paper text, for our specific setup)
        # It's actually the expectation of the loss: E[l(z_new, lambda)] = 1 - lambda_val if V_ik ~ U(0,1)
        true_risk_crc = 1.0 - lambda_crc # This is the true expected loss for chosen lambda
        true_risks_crc.append(true_risk_crc)

        # --- Our HPD Method ---
        def hpd_binomial_loss_func(V_ik_samples, lambda_val):
            return binomial_loss(V_ik_samples, lambda_val, K)

        lambda_hpd = find_lambda_hpd_beta(hpd_binomial_loss_func, lambda_search_space, calibration_data, B, alpha, beta, num_dirichlet_samples)
        hpd_lambdas.append(lambda_hpd)

        true_risk_hpd = 1.0 - lambda_hpd
        true_risks_hpd.append(true_risk_hpd)

    crc_lambdas = np.array(crc_lambdas)
    hpd_lambdas = np.array(hpd_lambdas)
    true_risks_crc = np.array(true_risks_crc)
    true_risks_hpd = np.array(true_risks_hpd)

    # Calculate relative frequency of exceeding target risk alpha
    crc_failure_rate = np.mean(true_risks_crc > alpha)
    hpd_failure_rate = np.mean(true_risks_hpd > alpha)

    results = {
        "crc_lambdas": crc_lambdas,
        "hpd_lambdas": hpd_lambdas,
        "true_risks_crc": true_risks_crc,
        "true_risks_hpd": true_risks_hpd,
        "crc_failure_rate": crc_failure_rate,
        "hpd_failure_rate": hpd_failure_rate,
        "mean_crc_lambda": np.mean(crc_lambdas),
        "mean_hpd_lambda": np.mean(hpd_lambdas),
        "std_crc_lambda": np.std(crc_lambdas),
        "std_hpd_lambda": np.std(hpd_lambdas),
    }
    return results

def run_synthetic_heteroskedastic_experiment(M: int = 10000, n: int = 200, alpha: float = 0.1, beta: float = 0.95, B: float = float('inf'), num_dirichlet_samples: int = 1000) -> dict:
    """
    Runs the synthetic heteroskedastic data experiment as described in Section 5.2.

    Args:
        M: Number of data splits/trials.
        n: Number of calibration samples.
        alpha: Target risk threshold.
        beta: Confidence level for HPD method.
        B: Maximum possible loss (can be effectively infinity for miscoverage).
        num_dirichlet_samples: Number of samples for L^+ Monte Carlo simulation.

    Returns:
        A dictionary containing results for CRC, HPD, and SCP methods.
    """
    print(f"Running Synthetic Heteroskedastic Experiment with M={M}, n={n}, alpha={alpha}, beta={beta}")

    crc_lambdas = []
    hpd_lambdas = []
    scp_lambdas = []

    # For miscoverage loss, the loss is 0 or 1. So B=1.0
    if B == float('inf'):
        current_B = 1.0
    else:
        current_B = B

    # Define a range of lambda values for searching. This lambda is the half-width of the prediction interval [-lambda, lambda].
    # We need to ensure this covers values observed in the data.
    # For Y ~ N(0, X^2) with X ~ U[0,4], Y can be quite large. Let's make the search space generous.
    lambda_search_space = np.linspace(0, 20.0, 201) # Example range, might need adjustment

    # Store mean prediction interval lengths for SCP and HPD
    crc_pi_lengths = []
    hpd_pi_lengths = []
    scp_pi_lengths = []

    for _ in tqdm(range(M), desc="Synthetic Heteroskedastic Experiment"):
        # Generate calibration data (X, Y pairs)
        X_cal, Y_cal = generate_synthetic_heteroskedastic_data(n)
        calibration_data = list(zip(X_cal, Y_cal)) # Convert to list of (x,y) tuples

        # Generate test data to calculate true risk (miscoverage rate)
        X_test, Y_test = generate_synthetic_heteroskedastic_data(1000) # Use a larger test set for accurate risk estimation
        test_data = list(zip(X_test, Y_test))

        # --- Conformal Risk Control (CRC) ---
        lambda_crc = conformal_risk_control(heteroskedastic_miscoverage_loss_function, lambda_search_space, calibration_data, current_B, alpha)
        crc_lambdas.append(lambda_crc)
        crc_pi_lengths.append(2 * lambda_crc)

        # --- Our HPD Method ---
        lambda_hpd = find_lambda_hpd_beta(heteroskedastic_miscoverage_loss_function, lambda_search_space, calibration_data, current_B, alpha, beta, num_dirichlet_samples)
        hpd_lambdas.append(lambda_hpd)
        hpd_pi_lengths.append(2 * lambda_hpd)

        # --- Split Conformal Prediction (SCP) ---
        lambda_scp = split_conformal_prediction(heteroskedastic_miscoverage_score_function, lambda_search_space, calibration_data, alpha)
        scp_lambdas.append(lambda_scp)
        scp_pi_lengths.append(2 * lambda_scp) # For SCP, lambda_scp is the threshold for |y|, so interval is [-lambda_scp, lambda_scp]

    crc_lambdas = np.array(crc_lambdas)
    hpd_lambdas = np.array(hpd_lambdas)
    scp_lambdas = np.array(scp_lambdas)

    crc_pi_lengths = np.array(crc_pi_lengths)
    hpd_pi_lengths = np.array(hpd_pi_lengths)
    scp_pi_lengths = np.array(scp_pi_lengths)

    # Calculate true miscoverage rates (risk) for each chosen lambda on a large test set
    true_miscoverage_rates_crc = []
    true_miscoverage_rates_hpd = []
    true_miscoverage_rates_scp = []

    for i in tqdm(range(M), desc="Calculating True Risks"):
        # For each lambda, evaluate its miscoverage on the large test set
        crc_miscoverage = np.mean([heteroskedastic_miscoverage_loss_function(dp, crc_lambdas[i]) for dp in test_data])
        true_miscoverage_rates_crc.append(crc_miscoverage)

        hpd_miscoverage = np.mean([heteroskedastic_miscoverage_loss_function(dp, hpd_lambdas[i]) for dp in test_data])
        true_miscoverage_rates_hpd.append(hpd_miscoverage)

        scp_miscoverage = np.mean([heteroskedastic_miscoverage_loss_function(dp, scp_lambdas[i]) for dp in test_data])
        true_miscoverage_rates_scp.append(scp_miscoverage)

    true_miscoverage_rates_crc = np.array(true_miscoverage_rates_crc)
    true_miscoverage_rates_hpd = np.array(true_miscoverage_rates_hpd)
    true_miscoverage_rates_scp = np.array(true_miscoverage_rates_scp)


    # Calculate failure rates (risk > alpha)
    crc_failure_rate = np.mean(true_miscoverage_rates_crc > alpha)
    hpd_failure_rate = np.mean(true_miscoverage_rates_hpd > alpha)
    scp_failure_rate = np.mean(true_miscoverage_rates_scp > alpha)

    results = {
        "crc_lambdas": crc_lambdas,
        "hpd_lambdas": hpd_lambdas,
        "scp_lambdas": scp_lambdas,
        "crc_failure_rate": crc_failure_rate,
        "hpd_failure_rate": hpd_failure_rate,
        "scp_failure_rate": scp_failure_rate,
        "mean_crc_pi_length": np.mean(crc_pi_lengths),
        "mean_hpd_pi_length": np.mean(hpd_pi_lengths),
        "mean_scp_pi_length": np.mean(scp_pi_lengths),
        "std_crc_pi_length": np.std(crc_pi_lengths),
        "std_hpd_pi_length": np.std(hpd_pi_lengths),
        "std_scp_pi_length": np.std(scp_pi_lengths),
    }
    return results


if __name__ == '__main__':
    # Run Synthetic Binomial Experiment
    binomial_results = run_synthetic_binomial_experiment(M=100) # Reduced M for quick testing
    print("
--- Synthetic Binomial Experiment Results ---")
    print(f"CRC Failure Rate: {binomial_results['crc_failure_rate']:.4f}")
    print(f"HPD Failure Rate: {binomial_results['hpd_failure_rate']:.4f}")
    print(f"Mean CRC Lambda: {binomial_results['mean_crc_lambda']:.4f}")
    print(f"Mean HPD Lambda: {binomial_results['mean_hpd_lambda']:.4f}")

    # Run Synthetic Heteroskedastic Experiment
    hetero_results = run_synthetic_heteroskedastic_experiment(M=100) # Reduced M for quick testing
    print("
--- Synthetic Heteroskedastic Experiment Results ---")
    print(f"CRC Failure Rate: {hetero_results['crc_failure_rate']:.4f}")
    print(f"HPD Failure Rate: {hetero_results['hpd_failure_rate']:.4f}")
    print(f"SCP Failure Rate: {hetero_results['scp_failure_rate']:.4f}")
    print(f"Mean CRC PI Length: {hetero_results['mean_crc_pi_length']:.4f}")
    print(f"Mean HPD PI Length: {hetero_results['mean_hpd_pi_length']:.4f}")
    print(f"Mean SCP PI Length: {hetero_results['mean_scp_pi_length']:.4f}")

