import logging
import math
import random
import numpy as np
from scipy.stats import chi2 # Not explicitly used for p-value calculation in design, but good to have for stat tests.
from typing import Dict, List, Any, Optional, Tuple, Union

from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MitigationAnalyzer:
    """
    Assesses the cost model of an adversarial attack and evaluates the effectiveness
    of various mitigation strategies described in the paper.
    """

    def __init__(self, config: Config):
        """
        Initializes the MitigationAnalyzer with the global configuration object.

        Args:
            config (Config): The global configuration object.
        """
        self.config = config
        self.bradley_terry_scale_factor: float = self.config.SIMULATOR_BRADLEY_TERRY_SCALE_FACTOR

        # Set random seed for reproducibility
        np.random.seed(self.config.RANDOM_SEED)
        random.seed(self.config.RANDOM_SEED)

    def calculate_attack_cost(self, num_actions: int, m_per_account: Union[int, float], c_account: float, c_action: float) -> float:
        """
        Computes the total cost of an attack based on the formula provided in Section 4.1.

        Args:
            num_actions (int): Total number of actions (interactions/votes) required for the attack.
            m_per_account (Union[int, float]): Maximum number of actions permitted per user account.
                                                Use float('inf') for no rate limiting.
            c_account (float): Cost of obtaining a single user account.
            c_action (float): Cost per individual action (e.g., per query, per CAPTCHA).

        Returns:
            float: The total estimated cost of the attack.
        """
        c_detector = self.config.MITIGATION_DETECTOR_COST

        # Account maintenance cost
        if m_per_account == 0 or math.isinf(m_per_account):
            num_accounts = 1
        else:
            num_accounts = math.ceil(num_actions / m_per_account)
        account_maintenance_cost = num_accounts * c_account

        # Action cost
        action_cost = num_actions * c_action

        total_cost = account_maintenance_cost + action_cost + c_detector
        return total_cost

    def _calculate_win_probability_bt(self, rating_a: float, rating_b: float) -> float:
        """
        Helper function to calculate the probability of model_a winning against model_b
        based on their Bradley-Terry (Elo-like) ratings.

        Args:
            rating_a (float): The current rating of model A.
            rating_b (float): The current rating of model B.

        Returns:
            float: The probability of model A winning against model B (between 0 and 1).
        """
        return 1 / (1 + math.exp(-(rating_a - rating_b) / self.bradley_terry_scale_factor))

    def estimate_benign_vote_distribution(
        self, models: List[Dict], current_ratings: Dict[str, float]
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Calculates the probability distribution of benign user votes for all possible
        pairwise model comparisons, based on their Bradley-Terry ratings.

        Args:
            models (List[Dict]): A list of model dictionaries (e.g., from config.MODEL_LIST).
            current_ratings (Dict[str, float]): Current Bradley-Terry ratings for all models.

        Returns:
            Dict[Tuple[str, str], Dict[str, float]]: A dictionary where keys are sorted tuples
                                                      of model IDs (model_id1, model_id2) and
                                                      values are dictionaries containing probabilities
                                                      for 'model_id1_wins', 'model_id2_wins', and 'tie'.
        """
        benign_vote_dist: Dict[Tuple[str, str], Dict[str, float]] = {}
        model_ids = [m['model_id'] for m in models]
        initial_rating = self.config.SIMULATOR_BRADLEY_TERRY_INITIAL_RATING

        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                model_a_id = model_ids[i]
                model_b_id = model_ids[j]

                q_a = current_ratings.get(model_a_id, initial_rating)
                q_b = current_ratings.get(model_b_id, initial_rating)

                p_a_wins_b = self._calculate_win_probability_bt(q_a, q_b)
                p_b_wins_a = self._calculate_win_probability_bt(q_b, q_a)

                # Assume ties account for the remaining probability
                p_tie = 1.0 - (p_a_wins_b + p_b_wins_a)
                # Ensure probability is not negative due to floating point inaccuracies
                p_tie = max(0.0, p_tie)

                # Normalize to ensure probabilities sum to 1 in case of clamping
                total_p = p_a_wins_b + p_b_wins_a + p_tie
                if total_p > 0:
                    p_a_wins_b /= total_p
                    p_b_wins_a /= total_p
                    p_tie /= total_p
                else: # Should not happen unless all ratings are identical and scale factor is weird
                    p_a_wins_b = p_b_wins_a = p_tie = 1/3 # Default to equal probabilities

                benign_vote_dist[(model_a_id, model_b_id)] = {
                    f'{model_a_id}_wins': p_a_wins_b,
                    f'{model_b_id}_wins': p_b_wins_a,
                    'tie': p_tie
                }
                # Also store for reversed order for easy lookup
                benign_vote_dist[(model_b_id, model_a_id)] = {
                    f'{model_b_id}_wins': p_b_wins_a,
                    f'{model_a_id}_wins': p_a_wins_b,
                    'tie': p_tie
                }

        return benign_vote_dist

    def _get_outcome_probability(self, prob_map: Dict[str, float], m1_id: str, m2_id: str, outcome: str) -> float:
        """Helper to retrieve the correct probability from a prob_map based on outcome."""
        if outcome == f'{m1_id}_wins':
            return prob_map.get(f'{m1_id}_wins', 0.0)
        elif outcome == f'{m2_id}_wins':
            return prob_map.get(f'{m2_id}_wins', 0.0)
        elif outcome == 'tie':
            return prob_map.get('tie', 0.0)
        else:
            logger.warning(f"Unknown outcome type '{outcome}'. Returning 0.0 probability.")
            return 0.0

    def simulate_malicious_user_detection(
        self,
        benign_vote_dist: Dict[Tuple[str, str], Dict[str, float]],
        adversary_vote_sequence: List[Tuple[str, str, str]],
        all_model_ids: List[str]
    ) -> Tuple[float, List[float]]:
        """
        Implements the likelihood test (Scenario 1) to detect malicious users by comparing
        their voting patterns against a known benign distribution (Section 4.2.3).

        Args:
            benign_vote_dist (Dict[Tuple[str, str], Dict[str, float]]):
                The probability distribution for benign user votes for all pairwise comparisons.
            adversary_vote_sequence (List[Tuple[str, str, str]]):
                A list of tuples (model1_id, model2_id, outcome_voted_by_adversary).
                Outcome should be one of '{model_id1}_wins', '{model_id2}_wins', or 'tie'.
            all_model_ids (List[str]): A list of all available model IDs, used for random model
                                       selection during benign sequence simulation.

        Returns:
            Tuple[float, List[float]]: A tuple containing:
                - The test statistic T for the adversary's sequence.
                - A list of test statistics T for simulated benign sequences.
        """
        # Small epsilon to avoid log(0)
        epsilon = 1e-9

        # 1. Calculate Likelihood of Adversary's Sequence under Benign Hypothesis (L(x | H_benign))
        log_L_adversary_benign = 0.0
        for m1_id, m2_id, outcome in adversary_vote_sequence:
            # Ensure consistent key order for benign_vote_dist lookup
            pair_key = tuple(sorted((m1_id, m2_id)))
            
            # Map m1_id and m2_id to internal "model_A_id" and "model_B_id" for the stored distribution
            # The way estimate_benign_vote_distribution stores them, model_id1 is always the first in the tuple
            # and model_id2 is the second, then probabilities are model_id1_wins and model_id2_wins.
            # So, we check which order the key is stored in.
            
            prob_map: Dict[str, float]
            if (m1_id, m2_id) in benign_vote_dist:
                prob_map = benign_vote_dist[(m1_id, m2_id)]
            elif (m2_id, m1_id) in benign_vote_dist:
                prob_map = benign_vote_dist[(m2_id, m1_id)]
            else:
                logger.warning(f"Benign distribution not found for models {m1_id}, {m2_id}. Assuming 1/3 probability.")
                prob_map = {f'{m1_id}_wins': 1/3, f'{m2_id}_wins': 1/3, 'tie': 1/3}


            prob_for_outcome = self._get_outcome_probability(prob_map, m1_id, m2_id, outcome)
            log_L_adversary_benign += np.log(max(epsilon, prob_for_outcome))
        
        T_adversary = -2 * log_L_adversary_benign

        # 2. Simulate Benign Sequences and Calculate Test Statistics
        T_sim_values: List[float] = []
        num_sim_sequences = self.config.MITIGATION_MALICIOUS_DETECTION_SIM_SEQUENCES
        num_votes_in_sequence = len(adversary_vote_sequence)

        if len(all_model_ids) < 2:
            logger.error("Not enough models to simulate benign user voting. Returning empty simulated T values.")
            return T_adversary, []

        for _ in range(num_sim_sequences):
            log_L_sim_benign = 0.0
            for _ in range(num_votes_in_sequence):
                # Randomly select two distinct models
                sim_m1_id, sim_m2_id = np.random.choice(all_model_ids, 2, replace=False)
                
                # Retrieve benign probabilities for this pair
                sim_pair_key = tuple(sorted((sim_m1_id, sim_m2_id)))
                sim_prob_map = benign_vote_dist.get(sim_pair_key)
                if sim_prob_map is None:
                    # If pair not in benign_vote_dist (e.g., due to filtering), assume uniform distribution for simplicity
                    sim_prob_map = {f'{sim_m1_id}_wins': 1/3, f'{sim_m2_id}_wins': 1/3, 'tie': 1/3}
                    logger.debug(f"Simulated pair {sim_pair_key} not in benign_vote_dist. Using uniform probs.")

                # Randomly choose an outcome based on these probabilities
                outcomes = [f'{sim_m1_id}_wins', f'{sim_m2_id}_wins', 'tie']
                probabilities = [
                    self._get_outcome_probability(sim_prob_map, sim_m1_id, sim_m2_id, f'{sim_m1_id}_wins'),
                    self._get_outcome_probability(sim_prob_map, sim_m1_id, sim_m2_id, f'{sim_m2_id}_wins'),
                    self._get_outcome_probability(sim_prob_map, sim_m1_id, sim_m2_id, 'tie')
                ]
                
                # Normalize probabilities in case of floating point issues from _get_outcome_probability
                sum_probs = sum(probabilities)
                if sum_probs == 0:
                    probabilities = [1/3, 1/3, 1/3] # Fallback to uniform if probabilities are all zero
                    sum_probs = 1
                probabilities = [p / sum_probs for p in probabilities]

                # Ensure that probabilities are valid inputs for np.random.choice
                if not np.isclose(np.sum(probabilities), 1.0):
                    logger.warning(f"Simulated probabilities for {sim_m1_id} vs {sim_m2_id} do not sum to 1: {probabilities}. Re-normalizing.")
                    probabilities = np.array(probabilities) / np.sum(probabilities)
                    
                simulated_outcome = np.random.choice(outcomes, p=probabilities)
                
                sim_prob_for_outcome = self._get_outcome_probability(sim_prob_map, sim_m1_id, sim_m2_id, simulated_outcome)
                log_L_sim_benign += np.log(max(epsilon, sim_prob_for_outcome))
            
            T_sim_values.append(-2 * log_L_sim_benign)
        
        return T_adversary, T_sim_values

    def perturb_leaderboard(self, true_ratings: Dict[str, float], noise_scale: float) -> Dict[str, float]:
        """
        Adds Gaussian noise to the true Bradley-Terry ratings to simulate a perturbed
        leaderboard as a mitigation strategy (Section 4.3).

        Args:
            true_ratings (Dict[str, float]): The true Bradley-Terry ratings for all models.
            noise_scale (float): The standard deviation of the Gaussian noise to be added.

        Returns:
            Dict[str, float]: A dictionary containing the perturbed ratings for all models.
        """
        perturbed_ratings: Dict[str, float] = {}
        for model_id, rating in true_ratings.items():
            noise = np.random.normal(loc=0, scale=noise_scale)
            perturbed_ratings[model_id] = rating + noise
        return perturbed_ratings

    def simulate_likelihood_ratio_detection(
        self,
        benign_vote_dist: Dict[Tuple[str, str], Dict[str, float]],
        malicious_vote_dist: Dict[Tuple[str, str], Dict[str, float]],
        adversary_vote_sequence: List[Tuple[str, str, str]]
    ) -> float:
        """
        Calculates the likelihood ratio for an adversary's vote sequence under a malicious
        hypothesis vs. a benign hypothesis (Scenario 2, Section 4.2.3).

        Args:
            benign_vote_dist (Dict[Tuple[str, str], Dict[str, float]]):
                The true benign probability distribution.
            malicious_vote_dist (Dict[Tuple[str, str], Dict[str, float]]):
                The probability distribution representing the adversary's voting strategy
                (e.g., derived from perturbed ratings).
            adversary_vote_sequence (List[Tuple[str, str, str]]):
                The sequence of votes cast by the adversary.

        Returns:
            float: The likelihood ratio Lambda(x) = Pr_M(x) / Pr_B(x).
                   Returns 0.0 if both Pr_M(x) and Pr_B(x) are 0 to avoid division by zero.
        """
        epsilon = 1e-9 # Small epsilon to avoid log(0)

        log_L_malicious = 0.0
        log_L_benign = 0.0

        for m1_id, m2_id, outcome in adversary_vote_sequence:
            pair_key = tuple(sorted((m1_id, m2_id)))

            # Get malicious probabilities
            mal_prob_map = malicious_vote_dist.get(pair_key)
            if mal_prob_map is None:
                logger.warning(f"Malicious distribution not found for models {m1_id}, {m2_id}. Assuming 1/3 probability.")
                mal_prob_map = {f'{m1_id}_wins': 1/3, f'{m2_id}_wins': 1/3, 'tie': 1/3}

            prob_malicious = self._get_outcome_probability(mal_prob_map, m1_id, m2_id, outcome)
            log_L_malicious += np.log(max(epsilon, prob_malicious))

            # Get benign probabilities
            ben_prob_map = benign_vote_dist.get(pair_key)
            if ben_prob_map is None:
                logger.warning(f"Benign distribution not found for models {m1_id}, {m2_id}. Assuming 1/3 probability.")
                ben_prob_map = {f'{m1_id}_wins': 1/3, f'{m2_id}_wins': 1/3, 'tie': 1/3}

            prob_benign = self._get_outcome_probability(ben_prob_map, m1_id, m2_id, outcome)
            log_L_benign += np.log(max(epsilon, prob_benign))

        # Calculate likelihood ratio. Handle case where both logs are very small
        log_likelihood_ratio = log_L_malicious - log_L_benign
        
        # If log_L_benign is extremely low (meaning Pr_B(x) was near zero), and log_L_malicious is not,
        # the ratio can be very large. If both are very low, division could lead to inf/0.
        # Here we return exp(difference) which is equivalent to ratio.
        # If log_L_benign is -inf and log_L_malicious is also -inf (very small numbers for both),
        # np.exp(-inf - (-inf)) would be 0, which is correct (0/0 indeterminate handled as 0).
        likelihood_ratio = np.exp(log_likelihood_ratio)
        
        return float(likelihood_ratio)

