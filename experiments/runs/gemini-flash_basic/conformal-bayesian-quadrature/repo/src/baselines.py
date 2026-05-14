import numpy as np

def conformal_risk_control(loss_function, lambda_values: np.ndarray, calibration_data, B: float, alpha: float) -> float:
    """
    Implements the Conformal Risk Control (CRC) decision rule (Proposition 3.2, Equation 15).
    Finds the infimum lambda such that the empirical risk condition is met.

    Args:
        loss_function: A callable that takes (data_point, lambda_val) and returns an individual loss.
        lambda_values: A 1D numpy array of lambda values to search over.
        calibration_data: The calibration dataset z_1, ..., z_n.
        B: The maximum possible value of the loss.
        alpha: The target risk threshold.

    Returns:
        The chosen lambda_crc.
    """
    n = len(calibration_data)
    min_lambda_satisfying_condition = float('inf')

    sorted_lambda_values = np.sort(lambda_values)

    for current_lambda in sorted_lambda_values:
        # Calculate individual losses for the current lambda
        individual_losses = np.array([loss_function(z_i, current_lambda) for z_i in calibration_data])

        # Calculate empirical risk R_n(lambda)
        empirical_risk = np.mean(individual_losses)

        # Check the CRC condition (Equation 15)
        if (n / (n + 1)) * empirical_risk + (B / (n + 1)) <= alpha:
            min_lambda_satisfying_condition = current_lambda
            break # Since the condition is monotonic, we found the infimum

    if min_lambda_satisfying_condition == float('inf'):
        # If no lambda satisfies the condition, return the largest lambda in the range
        # This implies alpha is too low or the lambda range is insufficient.
        return sorted_lambda_values[-1]
    else:
        return min_lambda_satisfying_condition

def split_conformal_prediction(score_function, lambda_values: np.ndarray, calibration_data, alpha: float) -> float:
    """
    Implements the Split Conformal Prediction (SCP) decision rule (Proposition 3.1, Equation 12).
    Finds the (1-alpha) quantile of nonconformity scores.

    Args:
        score_function: A callable that takes (data_point) and returns a nonconformity score s(z_i).
        lambda_values: A 1D numpy array of lambda values to search over (not strictly used for quantile, but for consistency).
        calibration_data: The calibration dataset z_1, ..., z_n.
        alpha: The target risk threshold.

    Returns:
        The chosen lambda_scp.
    """
    n = len(calibration_data)

    # Calculate nonconformity scores for calibration data
    scores = np.array([score_function(z_i) for z_i in calibration_data])

    # Sort the scores
    sorted_scores = np.sort(scores)

    # Calculate the index k = ceil((n + 1) * (1 - alpha))
    k = int(np.ceil((n + 1) * (1 - alpha)))

    if k <= n:
        # The quantile is the k-th smallest score (1-indexed)
        lambda_scp = sorted_scores[k - 1] # 0-indexed
    else:
        # If k > n, the quantile is infinity, meaning any score is acceptable
        # For practical purposes, we can return the maximum possible lambda or infinity.
        # The paper suggests infinity, which means the prediction set will cover everything.
        # For our experiments, this might be handled by having a sufficiently large lambda_values range.
        lambda_scp = float('inf') # Representing infinity
        if len(lambda_values) > 0:
            lambda_scp = lambda_values[-1] # Fallback to largest lambda if inf is problematic for comparisons

    return lambda_scp

if __name__ == '__main__':
    # Example Usage for Conformal Risk Control
    print("--- Testing Conformal Risk Control ---")
    np.random.seed(42)

    # Dummy loss function for CRC: e.g., squared error capped at B
    def crc_dummy_loss(data_point_y, lambda_val):
        # For demonstration, assume data_point_y is the true value and lambda_val is a prediction interval half-width
        # Loss is 1 if prediction |y| > lambda_val, else 0, or some continuous loss.
        # Let's use a simple monotonic loss: max(0, |data_point_y| - lambda_val)
        return max(0, np.abs(data_point_y) - lambda_val)

    crc_calibration_data = np.random.rand(10) * 2 - 1 # 10 samples between -1 and 1
    crc_lambda_values = np.linspace(0, 1.5, 50) # Range of lambda values
    crc_B = 1.5 # Maximum possible loss
    crc_alpha = 0.2 # Target risk threshold

    lambda_crc_chosen = conformal_risk_control(crc_dummy_loss, crc_lambda_values, crc_calibration_data, crc_B, crc_alpha)
    print(f"Chosen lambda_crc: {lambda_crc_chosen:.4f}")

    # Example Usage for Split Conformal Prediction
    print("
--- Testing Split Conformal Prediction ---")
    np.random.seed(42)

    # Dummy score function for SCP: e.g., |y - y_pred|
    def scp_dummy_score(data_point_y):
        # For simplicity, assume y_pred is always 0, so score is |y|
        return np.abs(data_point_y)

    scp_calibration_data = np.random.rand(10) * 2 - 1 # 10 samples between -1 and 1
    scp_alpha = 0.1 # Target coverage level 1-alpha
    # lambda_values is not strictly needed for the quantile method but included for API consistency if needed later
    scp_lambda_values = np.linspace(0, 2, 50)

    lambda_scp_chosen = split_conformal_prediction(scp_dummy_score, scp_lambda_values, scp_calibration_data, scp_alpha)
    print(f"Chosen lambda_scp: {lambda_scp_chosen:.4f}")
