import numpy as np
from typing import Tuple

class DataGenerator:
    """
    Handles generation of synthetic datasets and simulation of real-world data characteristics.
    """

    @staticmethod
    def generate_synthetic_binomial_losses(
        n: int, K: int, lambda_val: float
    ) -> np.ndarray:
        """
        Generates synthetic binomial losses as described in Section 5.1.
        L(z_i, lambda) = (1/K) * sum_{k=1 to K} 1{V_ik > lambda}
        where V_ik ~ Uniform(0,1).

        Args:
            n (int): Number of calibration samples.
            K (int): Number of Bernoulli trials per sample.
            lambda_val (float): The control parameter lambda.

        Returns:
            np.ndarray: Array of shape (n,) containing individual losses.
        """
        # V_ik ~ Uniform(0,1) for i=1..n, k=1..K
        V = np.random.uniform(0, 1, size=(n, K))
        # 1{V_ik > lambda}
        indicators = (V > lambda_val).astype(float)
        # sum_{k=1 to K} 1{V_ik > lambda}
        sum_indicators = np.sum(indicators, axis=1)
        # (1/K) * sum_indicators
        losses = sum_indicators / K
        return losses

    @staticmethod
    def generate_synthetic_heteroskedastic_data(
        n: int, X_range: Tuple[float, float], mu: float, sigma_multiplier: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates synthetic heteroskedastic data (X, Y) as described in Section 5.2.
        X ~ U[X_range[0], X_range[1]]
        Y | X ~ N(mu, (X * sigma_multiplier)^2)

        Args:
            n (int): Number of calibration samples.
            X_range (Tuple[float, float]): Range for Uniform distribution of X.
            mu (float): Mean for Normal distribution of Y|X.
            sigma_multiplier (float): Multiplier for X to determine standard deviation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (X, Y) arrays, each of shape (n,).
        """
        X = np.random.uniform(X_range[0], X_range[1], size=n)
        sigma = X * sigma_multiplier
        Y = np.random.normal(loc=mu, scale=sigma, size=n)
        return X, Y

    @staticmethod
    def calculate_heteroskedastic_miscoverage_loss(
        Y_true: np.ndarray, Y_pred_lower: np.ndarray, Y_pred_upper: np.ndarray
    ) -> np.ndarray:
        """
        Calculates miscoverage loss for heteroskedastic data.
        Loss is 1 if Y_true is outside [Y_pred_lower, Y_pred_upper], 0 otherwise.

        Args:
            Y_true (np.ndarray): Array of true Y values.
            Y_pred_lower (np.ndarray): Array of lower bounds of prediction intervals.
            Y_pred_upper (np.ndarray): Array of upper bounds of prediction intervals.

        Returns:
            np.ndarray: Array of miscoverage losses, shape (n,).
        """
        losses = ((Y_true < Y_pred_lower) | (Y_true > Y_pred_upper)).astype(float)
        return losses

    @staticmethod
    def simulate_coco_data(n_calibration: int, n_test: int, dummy_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates data for the MS-COCO experiment. The paper describes the setup
        as mirroring Angelopoulos & Bates (2023, Section 5.1), which implies
        we don't need to load actual images but rather generate data points
        that would correspond to the losses for false negative rate control.
        For simplicity, we return dummy arrays of appropriate size.
        Realistically, this would involve a pre-trained model and actual data.

        Args:
            n_calibration (int): Number of calibration examples.
            n_test (int): Number of test examples.
            dummy_value (float): A placeholder value for the simulated data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of (calibration_data, test_data).
                                           Each is an array of dummy values.
        """
        calibration_data = np.full(n_calibration, dummy_value)
        test_data = np.full(n_test, dummy_value)
        return calibration_data, test_data

    @staticmethod
    def calculate_coco_false_negative_loss(
        predictions: np.ndarray, ground_truths: np.ndarray, lambda_val: float
    ) -> np.ndarray:
        """
        Calculates false negative rate loss for MS-COCO experiment.
        The actual implementation of this would depend on the specific model's output
        and how false negatives are defined in Angelopoulos & Bates (2023, Section 5.1).
        For this reproduction, we assume `predictions` and `ground_truths` are scores
        and binary labels, respectively, and `lambda_val` is a threshold.
        Loss is 1 if ground_truth is positive but prediction is below threshold (false negative).

        Args:
            predictions (np.ndarray): Model predictions (e.g., probabilities or scores).
            ground_truths (np.ndarray): True labels (e.g., binary: 0 or 1).
            lambda_val (float): Threshold for classification.

        Returns:
            np.ndarray: Array of false negative losses.
        """
        # This is a simplified interpretation. Actual FNR would be more complex.
        # Assuming ground_truths are binary (0/1) and predictions are scores.
        false_negatives = (ground_truths == 1) & (predictions < lambda_val)
        losses = false_negatives.astype(float)
        return losses
