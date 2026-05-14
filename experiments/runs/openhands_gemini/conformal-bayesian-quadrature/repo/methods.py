import numpy as np
from typing import Tuple
from utils import Utils # Assuming Utils class is in utils.py
import pdb

class ConformalMethods:
    """
    Implements Conformal Risk Control (CRC), Risk-controlling Prediction Sets (RCPS),
    and our proposed Bayesian Quadrature (BQ) method.
    """

    @staticmethod
    def conformal_risk_control(
        losses: np.ndarray, alpha: float, B: float, lambda_range: np.ndarray = None
    ) -> float:
        """
        Implements Conformal Risk Control (CRC) as described in Section 2.1 and 3.2.
        Finds the smallest lambda such that E(L_crc) <= alpha.
        Eq. 15: lambda_crc = inf {lambda : (1/(n+1)) * (sum(l_i(lambda)) + B) <= alpha}

        Args:
            losses (np.ndarray): Array of individual losses for each lambda in lambda_range.
                                 Shape (n, len(lambda_range)).
            alpha (float): Target risk level.
            B (float): Maximum possible loss.
            lambda_range (np.ndarray): Optional array of lambda values to search over.
                                       If not provided, it implies a direct loss array.
                                       This is crucial for the infimum search.

        Returns:
            float: The chosen lambda_crc.
        """
        n = losses.shape[0]
        # The paper describes L_crc as a function of lambda.
        # We assume `losses` here are already a function of some lambda values,
        # where `losses[i, j]` is l(z_i, lambda_range[j]).

        if lambda_range is None:
            # If no lambda_range is given, we assume losses are already calculated
            # for a specific lambda and we are checking if it meets the criterion.
            # This interpretation is likely incorrect for finding infimum.
            raise ValueError("lambda_range must be provided for CRC to find the infimum.")

        # Calculate empirical risk for each lambda in the range
        # losses has shape (n, num_lambda_steps)
        # sum_losses_per_lambda has shape (num_lambda_steps,)
        sum_losses_per_lambda = np.sum(losses, axis=0)

        # Calculate the criterion for each lambda
        criterion = (1 / (n + 1)) * (sum_losses_per_lambda + B)

        # Find the smallest lambda that satisfies the criterion
        # We need to find the *first* lambda where criterion <= alpha
        satisfied_indices = np.where(criterion <= alpha)[0]

        if len(satisfied_indices) > 0:
            # The infimum would be the smallest lambda satisfying the condition
            lambda_crc = lambda_range[satisfied_indices[0]]
        else:
            # If no lambda satisfies the condition, CRC might select the largest possible lambda (or infinity)
            # This indicates that even the largest lambda cannot achieve the target risk.
            # Depending on the application, one might return the maximum lambda or indicate failure.
            lambda_crc = lambda_range[-1] # Default to largest lambda if none satisfy

        return lambda_crc

    @staticmethod
    def split_conformal_prediction(
        scores: np.ndarray, alpha: float
    ) -> float:
        """
        Implements Split Conformal Prediction (SCP) as described in Section 2.1 and 3.1.
        Calculates the quantile q_hat.
        Eq. 12: lambda_scp = s_(\lceil (n+1)(1-alpha) \rceil)

        Args:
            scores (np.ndarray): Array of nonconformity scores s(z_i).
            alpha (float): Target miscoverage level.

        Returns:
            float: The chosen lambda_scp (quantile).
        """
        n = len(scores)
        sorted_scores = np.sort(scores)

        k_index = int(np.ceil((n + 1) * (1 - alpha))) - 1 # Adjust for 0-based indexing

        if k_index < n:
            lambda_scp = sorted_scores[k_index]
        else:
            # If k_index is out of bounds, means we need a very large quantile
            lambda_scp = np.inf # Or the maximum possible score if bounded

        return lambda_scp

    @staticmethod
    def bayesian_quadrature_hpd(
        losses: np.ndarray, alpha: float, B: float, beta: float,
        dirichlet_samples: int, lambda_range: np.ndarray = None
    ) -> float:
        """
        Implements our Bayesian Quadrature (BQ) method using the one-sided HPD interval
        as described in Section 4.5 and 5.
        Finds lambda_hpd_beta, the infimum lambda such that Pr(L+ <= alpha | losses) >= beta.
        Eq. 29: b*_beta = inf {b : Pr(L+ <= b | losses) >= beta}

        Args:
            losses (np.ndarray): Array of individual losses for each lambda in lambda_range.
                                 Shape (n, len(lambda_range)).
            alpha (float): Target risk level (b in Eq. 29).
            B (float): Maximum possible loss.
            beta (float): Desired confidence level for the HPD interval.
            dirichlet_samples (int): Number of Monte Carlo samples for L+.
            lambda_range (np.ndarray): Array of lambda values to search over.

        Returns:
            float: The chosen lambda_hpd_beta.
        """
        n = losses.shape[0]

        if lambda_range is None:
            raise ValueError("lambda_range must be provided for BQ-HPD to find the infimum.")

        lambda_hpd_beta = None
        min_lambda_satisfied = np.inf

        for j, current_lambda in enumerate(lambda_range):
            # Sort losses for the current lambda
            sorted_losses = np.sort(losses[:, j])

            # Augment with B for the (n+1)-th loss
            l_ordered_plus_B = np.append(sorted_losses, B)

            # Sample Dirichlet U_i (alpha_params = [1, ..., 1])
            alpha_params = np.ones(n + 1)
            U_samples = Utils.sample_dirichlet(dirichlet_samples, alpha_params)

            # Calculate L+ for each sample (Eq. 27)
            L_plus_samples = np.sum(U_samples * l_ordered_plus_B, axis=1)

            # Check the condition: Pr(L+ <= alpha) >= beta
            # This means finding the (1-beta)th quantile of L+ samples and checking if it's <= alpha.
            # Or, more directly, count how many samples are <= alpha.
            proportion_le_alpha = np.mean(L_plus_samples <= alpha)

            if proportion_le_alpha >= beta:
                if current_lambda < min_lambda_satisfied:
                    min_lambda_satisfied = current_lambda
                    lambda_hpd_beta = current_lambda
                # We want the infimum lambda, so once a condition is met, we continue to see
                # if a smaller lambda also meets it. But usually, lambda_range is sorted,
                # so the first one we find is the smallest.
                # However, L+ is a stochastic variable, so the condition might not be monotonic.
                # Thus, we need to iterate through all lambdas to ensure we find the infimum.
            
        if lambda_hpd_beta is None:
             # If no lambda satisfies the condition, return largest possible lambda or indicate failure.
             lambda_hpd_beta = lambda_range[-1] # Default to largest lambda if none satisfy
        
        return lambda_hpd_beta

    @staticmethod
    def rcps_hoeffding(
        losses_at_lambda: np.ndarray, alpha: float, B: float, delta: float, lambda_range: np.ndarray = None
    ) -> float:
        """
        Implements Risk-Controlling Prediction Sets (RCPS) with Hoeffding upper confidence bound.
        This is a placeholder as the paper only mentions it as a baseline
        and does not provide the explicit formula for lambda_rcps.
        It usually involves finding lambda that satisfies an upper bound on risk.
        For simplicity, we'll implement a conceptual placeholder based on common RCPS approaches.
        The exact formula for RCPS in Bates et al. (2021) is typically more complex.

        A common bound for RCPS would be something like:
        lambda_rcps = inf {lambda : empirical_risk(lambda) + hoeffding_bound <= alpha}
        empirical_risk(lambda) = (1/n) * sum(l_i(lambda))

        Args:
            losses_at_lambda (np.ndarray): Array of individual losses for each lambda in lambda_range.
                                           Shape (n, len(lambda_range)).
            alpha (float): Target risk level.
            B (float): Maximum possible loss (bound on individual loss values).
            delta (float): Confidence parameter for Hoeffding bound (e.g., 1-beta).
                           If beta is confidence level, then delta is 1 - beta.
            lambda_range (np.ndarray): Array of lambda values to search over.

        Returns:
            float: The chosen lambda_rcps.
        """
        n = losses_at_lambda.shape[0]

        if lambda_range is None:
            raise ValueError("lambda_range must be provided for RCPS to find the infimum.")

        lambda_rcps = None
        min_lambda_satisfied = np.inf

        for j, current_lambda in enumerate(lambda_range):
            current_losses = losses_at_lambda[:, j]
            empirical_risk = np.mean(current_losses)

            # Hoeffding bound: sqrt( (B^2 * log(1/delta)) / (2 * n) )
            # Here, we assume B is the bound for individual losses, not for the mean.
            # A tighter bound for the mean would be needed, or ensure that individual losses are scaled within [0, B]
            hoeffding_bound = B * np.sqrt(np.log(1/delta) / (2 * n))
            
            # The paper's RCPS uses a slightly different form:
            # lambda_hat = inf {lambda : (n / (n+1)) * R_hat_n(lambda) + B / (n+1) <= alpha}
            # which is essentially the conformal risk control formula with R_hat_n = empirical_risk.
            # The RCPS paper (Bates et al., 2021) uses Hoeffding on the supremum of the empirical risk process.
            # For reproduction, we use the formulation from Angelopoulos et al. (2024) which is essentially CRC.
            # The mention of Hoeffding in this paper's RCPS description seems to be a slightly different variant
            # than the standard RCPS implementation, or an older version.
            # Given the phrasing "RCPS with Hoeffding upper confidence bound as an additional baseline",
            # and then comparing it to CRC, it's likely referring to a variant that incorporates Hoeffding.
            # Let's re-align with the formula in Angelopoulos et al. (2024) Theorem 1, which is cited,
            # and then add a Hoeffding correction on top of it.
            # However, the paper's main text for RCPS (Eq 3) directly relates to CRC.
            # The description in Section 5 seems to suggest a different RCPS.
            # To be faithful to the text, we should use the one referenced in the paper for RCPS (Bates et al., 2021).
            # The expression for RCPS is (n / (n+1)) * R_hat_n(lambda) + B / (n+1) <= alpha
            # This is the same as CRC's criterion, so for now, we'll use that.
            
            # Let's use the explicit Hoeffding bound if the problem refers to a different RCPS.
            # The problem description for RCPS: "Risk-controlling Prediction Sets (RCPS) (Bates et al., 2021) with Hoeffding upper confidence bound"
            # This implies a Hoeffding correction on the empirical risk, not just the conformal risk control formula.
            # The goal is to control P(Risk > alpha) <= delta.
            # A simple application of Hoeffding for a mean of bounded random variables is:
            # P(mean(X) - E[mean(X)] > epsilon) <= exp(-2 * n * epsilon^2 / (B-A)^2)
            # So, to ensure P(E[L] > alpha) <= delta, we need mean(losses) + epsilon <= alpha.
            # epsilon = B * sqrt(log(1/delta) / (2 * n))
            # So, criterion = empirical_risk + hoeffding_bound

            criterion = empirical_risk + hoeffding_bound

            if criterion <= alpha:
                if current_lambda < min_lambda_satisfied:
                    min_lambda_satisfied = current_lambda
                    lambda_rcps = current_lambda
        
        if lambda_rcps is None:
            lambda_rcps = lambda_range[-1] # Default to largest lambda if none satisfy

        return lambda_rcps

