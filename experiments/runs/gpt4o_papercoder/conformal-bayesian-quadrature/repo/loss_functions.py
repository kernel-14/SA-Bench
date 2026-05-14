# loss_functions.py

import numpy as np
from typing import Union

class LossFunctions:
    """
    Provides loss functions used across experiments:
    - Binomial loss for synthetic data (Section 5.1).
    - Miscoverage loss for heteroskedastic and MS-COCO experiments (Sections 5.2 & 5.3).
    """

    def __init__(self) -> None:
        """
        Initialize LossFunctions utility class.
        No specific configurations are required at initialization.
        """
        pass

    def binomial_loss(self, V: np.ndarray, lambda_: float) -> np.ndarray:
        """
        Compute binomial loss for synthetic binomial data experiments.

        Args:
            V (np.ndarray): A 2D array of size (n, K) where `n` is the number
                            of calibration samples and `K` is the number of trials per sample.
            lambda_ (float): The threshold value to compute the loss.

        Returns:
            np.ndarray: A 1D array of size `n`, representing the binomial losses for each calibration sample.
        """

        # Validate inputs
        if not isinstance(V, np.ndarray) or V.ndim != 2:
            raise ValueError("Input V must be a 2D NumPy array.")
        if not isinstance(lambda_, float):
            raise ValueError("Parameter 'lambda_' must be a float.")

        # Calculate the binomial loss
        # Indicator function: 1 if V_ik > lambda_, 0 otherwise
        indicator = (V > lambda_).astype(np.float64)

        # Binomial loss as the mean of indicators across axis=1
        binomial_losses = np.mean(indicator, axis=1)

        return binomial_losses

    def miscoverage_loss(self, y: Union[np.ndarray, float], lower: float, upper: float) -> float:
        """
        Compute miscoverage loss for prediction intervals.

        Args:
            y (Union[np.ndarray, float]): Ground truth values; scalar or 1D NumPy array.
            lower (float): Lower bound of the prediction interval.
            upper (float): Upper bound of the prediction interval.

        Returns:
            float: The overall miscoverage loss (proportion of `y` values outside bounds).
        """

        # Ensure y is a NumPy array if it's a scalar for uniform processing
        if isinstance(y, (int, float)):
            y = np.array([y], dtype=np.float64)
        elif isinstance(y, np.ndarray) and y.ndim == 1:
            y = y.astype(np.float64)
        else:
            raise ValueError("Input 'y' must be a float or a 1D NumPy array.")

        # Validate bounds
        if not isinstance(lower, float) or not isinstance(upper, float):
            raise ValueError("Bounds 'lower' and 'upper' must be floats.")
        if lower > upper:
            raise ValueError("Invalid bounds: 'lower' must be ≤ 'upper'.")

        # Indicator function: 1 if y is within [lower, upper], 0 otherwise
        within_interval = (y >= lower) & (y <= upper)

        # Miscoverage loss = 1 - proportion of values within interval
        miscoverage = 1.0 - np.mean(within_interval)

        return miscoverage
