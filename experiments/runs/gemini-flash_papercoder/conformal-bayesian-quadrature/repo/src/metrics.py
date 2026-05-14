import numpy as np
from scipy.stats import beta
from typing import List, Tuple, TYPE_CHECKING

# To avoid circular dependency, use TYPE_CHECKING and string literal for type hints
if TYPE_CHECKING:
    from src.loss_functions import FalseNegativeLoss


class EvaluationMetrics:
    """
    Provides methods for calculating and reporting evaluation metrics
    such as exceedance frequency, prediction interval length, and prediction set size.
    """

    def __init__(self, target_alpha: float, target_failure_rate: float):
        """
        Initializes the EvaluationMetrics calculator.

        Args:
            target_alpha: The target risk threshold (e.g., alpha from config).
            target_failure_rate: The maximum allowable failure rate (1 - beta from config).
        """
        if not (0 <= target_alpha <= 1):
            raise ValueError("target_alpha must be between 0 and 1.")
        if not (0 <= target_failure_rate <= 1):
            raise ValueError("target_failure_rate must be between 0 and 1.")

        self.target_alpha = target_alpha
        self.target_failure_rate = target_failure_rate

    def calculate_exceedance_frequency(self, true_risks: List[float]) -> Tuple[float, Tuple[float, float]]:
        """
        Calculates the relative frequency of trials where the true risk exceeded
        the target_alpha, along with its 95% Clopper-Pearson confidence interval.

        Args:
            true_risks: A list of true risk values (one for each trial).

        Returns:
            A tuple containing:
                - The exceedance frequency (float, as a percentage).
                - A tuple (lower_bound, upper_bound) for the 95% Clopper-Pearson CI (as percentages).
        """
        num_trials = len(true_risks)
        if num_trials == 0:
            return 0.0, (0.0, 0.0)

        # Count trials where true_risk > target_alpha
        num_exceeding = np.sum(np.array(true_risks) > self.target_alpha)

        frequency = (num_exceeding / num_trials) * 100.0

        # Calculate 95% Clopper-Pearson confidence intervals for binomial proportions
        # For a proportion p, estimated by k/n, the 100*(1-confidence_level)% CI is:
        # Lower bound: beta.ppf(alpha/2, k, n - k + 1)
        # Upper bound: beta.ppf(1 - alpha/2, k + 1, n - k)
        # Here, alpha/2 is 0.025 for a 95% CI.
        ci_lower = beta.ppf(0.025, num_exceeding, num_trials - num_exceeding + 1)
        ci_upper = beta.ppf(0.975, num_exceeding + 1, num_trials - num_exceeding)

        # Handle cases where ppf might return NaN (e.g., for num_exceeding=0 or num_trials)
        if np.isnan(ci_lower):
            ci_lower = 0.0
        if np.isnan(ci_upper):
            ci_upper = 1.0 # Max possible upper bound

        return frequency, (ci_lower * 100.0, ci_upper * 100.0)

    def calculate_mean_prediction_interval_length(self, lambdas: List[float]) -> float:
        """
        Calculates the mean prediction interval length (2 * lambda) across all trials.
        This metric is relevant for experiments like synthetic heteroskedastic data.

        Args:
            lambdas: A list of lambda values chosen in each trial.

        Returns:
            The mean prediction interval length. Returns 0.0 if the list is empty.
        """
        if not lambdas:
            return 0.0
        return float(np.mean(2.0 * np.array(lambdas)))

    def calculate_mean_prediction_set_size(
        self, lambda_vals: List[float], test_data_per_trial: List[List[Tuple]], loss_fn: "FalseNegativeLoss"
    ) -> float:
        """
        Calculates the average size of prediction sets generated across all trials.
        This metric is relevant for experiments like MS-COCO.

        Args:
            lambda_vals: A list of lambda values chosen in each trial.
            test_data_per_trial: A list of lists, where each inner list contains
                                 the test data points for a specific trial.
                                 For MS-COCO, each data point z_item is expected to be
                                 (image_data, true_labels, model_scores_array).
            loss_fn: An instance of FalseNegativeLoss. Its `model_output_func` is used
                     to extract the pre-computed model scores from a test data item.

        Returns:
            The average prediction set size. Returns 0.0 if no test examples were processed.
        """
        if not lambda_vals or not test_data_per_trial:
            return 0.0
        if len(lambda_vals) != len(test_data_per_trial):
            raise ValueError("Lengths of lambda_vals and test_data_per_trial must match.")

        total_set_size = 0.0
        total_examples = 0

        for trial_idx, current_lambda in enumerate(lambda_vals):
            current_test_data = test_data_per_trial[trial_idx]

            for z_item in current_test_data:
                # z_item is expected to be (image_data, true_labels, model_scores) from CocoDataLoader
                # Use the model_output_func from FalseNegativeLoss to extract scores
                # assuming it was set up to extract the pre-computed scores (e.g., lambda z: z[2])
                model_scores = loss_fn.model_output_func(z_item)

                # A prediction set C_lambda(x) = {j : f_j(x) > lambda}
                # So, count how many scores are greater than the current lambda
                pred_set_size = np.sum(model_scores > current_lambda)

                total_set_size += pred_set_size
                total_examples += 1

        if total_examples == 0:
            return 0.0
        return total_set_size / total_examples

