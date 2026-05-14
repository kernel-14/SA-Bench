import numpy as np

def binomial_loss(V_ik_samples: np.ndarray, lambda_val: float, K: int = 4) -> float:
    """
    Calculates the binomial loss as described in Equation 34.
    \ell(z_i, \lambda) = (1/K) * sum_{k=1}^K I{V_ik > \lambda}
    where V_ik ~ Uniform(0, 1).

    Args:
        V_ik_samples: A 1D numpy array of K samples from Uniform(0, 1) for a single z_i.
        lambda_val: The control parameter \lambda.
        K: The number of Bernoulli trials, as per the paper (K=4).

    Returns:
        The individual loss \ell(z_i, \lambda).
    """
    return np.mean(V_ik_samples > lambda_val)

def miscoverage_loss(prediction_set_func, y_true) -> float:
    """
    Calculates the miscoverage loss (0 or 1).

    Args:
        prediction_set_func: A callable that takes y_true and returns True if y_true is in the prediction set,
                             False otherwise.
        y_true: The true label.

    Returns:
        1.0 if y_true is not in the prediction set, 0.0 otherwise.
    """
    return 1.0 if not prediction_set_func(y_true) else 0.0

def generate_synthetic_binomial_data(n: int, K: int = 4):
    """
    Generates synthetic binomial data z_i = (V_i1, ..., V_iK) for calibration.
    Here, each z_i is effectively a set of K uniform samples.

    Args:
        n: Number of calibration samples.
        K: Number of Uniform(0,1) samples for each z_i.

    Returns:
        A list of numpy arrays, where each array is (K,) representing V_ik for a z_i.
    """
    return [np.random.uniform(0, 1, K) for _ in range(n)]

def generate_synthetic_heteroskedastic_data(n: int):
    """
    Generates synthetic heteroskedastic data as described in Section 5.2.
    X ~ U[0, 4], Y | X ~ N(0, X^2).

    Args:
        n: Number of calibration samples.

    Returns:
        A tuple of (X, Y) numpy arrays of shape (n,).
    """
    X = np.random.uniform(0, 4, n)
    Y = np.random.normal(0, X)
    return X, Y

def heteroskedastic_miscoverage_loss_function(data_point_x_y, lambda_val: float) -> float:
    """
    Calculates miscoverage loss for heteroskedastic data for a given lambda_val.
    The prediction interval is [-lambda_val, lambda_val].

    Args:
        data_point_x_y: A tuple (x, y) for a single data point.
        lambda_val: The half-width of the prediction interval [-lambda_val, lambda_val].

    Returns:
        1.0 if y is outside [-lambda_val, lambda_val], 0.0 otherwise.
    """
    _, y = data_point_x_y
    return 1.0 if not (-lambda_val <= y <= lambda_val) else 0.0


def heteroskedastic_miscoverage_score_function(data_point_x_y) -> float:
    """
    Calculates the nonconformity score for heteroskedastic data.
    For prediction interval [-lambda, lambda], a natural score is |y|.

    Args:
        data_point_x_y: A tuple (x, y) for a single data point.

    Returns:
        The nonconformity score, which is |y| in this context.
    """
    _, y = data_point_x_y
    return np.abs(y)


if __name__ == '__main__':
    print("--- Testing utils.py functions ---")
    np.random.seed(42)

    # Test binomial_loss
    v_samples = np.random.uniform(0, 1, 4)
    test_lambda = 0.5
    loss = binomial_loss(v_samples, test_lambda)
    print(f"Binomial loss for V_ik={v_samples}, lambda={test_lambda}: {loss:.2f}")

    # Test miscoverage_loss
    def test_pred_set_func(y):
        return -0.5 <= y <= 0.5
    y_true_in = 0.3
    y_true_out = 0.7
    print(f"Miscoverage loss for y_true={y_true_in} (in set): {miscoverage_loss(test_pred_set_func, y_true_in)}")
    print(f"Miscoverage loss for y_true={y_true_out} (out set): {miscoverage_loss(test_pred_set_func, y_true_out)}")

    # Test generate_synthetic_binomial_data
    n_binomial = 5
    K_binomial = 3
    binomial_data = generate_synthetic_binomial_data(n_binomial, K_binomial)
    print(f"Generated binomial data (first sample): {binomial_data[0]}")
    assert len(binomial_data) == n_binomial
    assert len(binomial_data[0]) == K_binomial

    # Test generate_synthetic_heteroskedastic_data
    n_hetero = 10
    X_hetero, Y_hetero = generate_synthetic_heteroskedastic_data(n_hetero)
    print(f"Generated heteroskedastic data (first X, Y): ({X_hetero[0]:.2f}, {Y_hetero[0]:.2f})")
    assert len(X_hetero) == n_hetero
    assert len(Y_hetero) == n_hetero

    # Test heteroskedastic_miscoverage_loss_function
    data_point = (X_hetero[0], Y_hetero[0])
    test_lambda_h = 0.5 # Example lambda
    loss_h = heteroskedastic_miscoverage_loss_function(data_point, test_lambda_h)
    print(f"Heteroskedastic miscoverage loss for data {data_point} with lambda {test_lambda_h}: {loss_h}")

    # Test heteroskedastic_miscoverage_score_function
    score_h = heteroskedastic_miscoverage_score_function(data_point)
    print(f"Heteroskedastic nonconformity score for data {data_point}: {score_h:.2f}")
