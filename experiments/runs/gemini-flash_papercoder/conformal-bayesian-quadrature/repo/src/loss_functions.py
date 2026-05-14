import abc
import numpy as np
from typing import Any, Callable, List, Tuple


class LossFunction(abc.ABC):
    """
    Abstract base class for all loss functions.

    Defines the interface for calculating individual loss values ℓ(z, λ)
    for a single observation z and a control parameter λ.
    """

    @abc.abstractmethod
    def calculate_individual_loss(self, z: Any, lambda_val: float) -> float:
        """
        Calculates the individual loss for a given observation and lambda_val.

        Args:
            z: The observation (data point, e.g., (x, y)).
               Its structure depends on the specific loss function.
            lambda_val: The control parameter.

        Returns:
            The scalar loss value ℓ(z, λ).
        """
        pass


class BinomialLoss(LossFunction):
    """
    Implements the individual loss function for the Synthetic Binomial Data
    experiment (Section 5.1, Eq. 32).

    The loss is defined as ℓ(z_i, λ) = (1/K) * Σ_{k=1}^K 𝟙{V_ik > λ}.
    """

    def __init__(self, K: int):
        """
        Initializes the BinomialLoss with the number of Bernoulli trials K.

        Args:
            K: The number of Bernoulli trials (from config.synthetic_binomial.K).
        """
        if not isinstance(K, int) or K <= 0:
            raise ValueError("K must be a positive integer.")
        self.K = K

    def calculate_individual_loss(self, z: Tuple[float, ...], lambda_val: float) -> float:
        """
        Calculates the individual loss for a single calibration point.

        Args:
            z: A tuple of floats (V_i1, ..., V_iK) representing the K uniform
               random variables for a single calibration sample.
            lambda_val: The control parameter.

        Returns:
            The scalar loss value ℓ(z, λ).
        """
        v_ik_array = np.array(z)
        # The indicator function 𝟙{V_ik > λ}
        indicators = (v_ik_array > lambda_val).astype(float)
        return np.sum(indicators) / self.K


class MiscoverageLoss(LossFunction):
    """
    Implements the individual miscoverage loss for the Synthetic Heteroskedastic Data
    experiment (Section 5.2).

    The prediction interval is [-λ, λ], and the loss is 𝟙{y_new < -λ or y_new > λ}.
    """

    def __init__(self):
        """
        Initializes the MiscoverageLoss.
        """
        pass  # No specific parameters needed for initialization

    def calculate_individual_loss(self, z: Tuple[float, float], lambda_val: float) -> float:
        """
        Calculates the individual miscoverage loss for an observation.

        Args:
            z: A tuple (x_new, y_new), where y_new is the ground truth.
            lambda_val: The control parameter, defining the prediction interval [-λ, λ].

        Returns:
            1.0 if y_new is outside [-λ, λ], 0.0 otherwise.
        """
        # For miscoverage loss, only y_new is relevant.
        # z is (x_new, y_new)
        y_new = z[1]
        abs_y = abs(y_new)
        # Loss is 1 if y_new is outside the interval [-lambda_val, lambda_val]
        # which means |y_new| > lambda_val
        return float(abs_y > lambda_val)


class FalseNegativeLoss(LossFunction):
    """
    Implements the individual false negative rate loss for the MS-COCO experiment (Section 5.3).

    The loss is defined based on non-conformity scores and a control parameter λ.
    Specifically, ℓ(z=(x, y_true), λ) = (1 / |y_true|) * Σ_{j_true ∈ y_true} 𝟙{s_{j_true}(x) > λ}.
    """

    def __init__(self, model_output_func: Callable[[Any], np.ndarray]):
        """
        Initializes the FalseNegativeLoss.

        Args:
            model_output_func: A callable that takes an input `x` (e.g., image data)
                               and returns a NumPy array of non-conformity scores `s_j(x)`
                               for all possible classes `j`.
        """
        if not callable(model_output_func):
            raise ValueError("model_output_func must be a callable function or method.")
        self.model_output_func = model_output_func

    def calculate_individual_loss(self, z: Tuple[Any, List[int]], lambda_val: float) -> float:
        """
        Calculates the individual false negative rate loss for an observation.

        Args:
            z: A tuple (x, y_true), where x is the input data (e.g., preprocessed image)
               and y_true is a list of true label indices for that input.
            lambda_val: The control parameter, acting as a threshold on non-conformity scores.

        Returns:
            The false negative rate for the given observation.
        """
        x, y_true = z
        s_scores = self.model_output_func(x)

        false_negatives_count = 0
        total_true_positives = len(y_true)

        if total_true_positives == 0:
            # If there are no true labels, there can be no false negatives.
            return 0.0

        for j_true in y_true:
            # A false negative occurs if a true label's non-conformity score
            # is above the threshold, meaning it's not included in the prediction set.
            if s_scores[j_true] > lambda_val:
                false_negatives_count += 1

        return false_negatives_count / total_true_positives

