
import numpy as np
import random
from typing import Dict, List, Tuple
from scipy.stats import norm
from config import Config

class BradleyTerryModel:
    """
    Implements the Bradley-Terry model for updating model ratings based on pairwise comparisons.
    The paper mentions using Bradley-Terry coefficients, which are similar to Elo ratings.
    """
    def __init__(self, initial_ratings: Dict[str, float], k_factor: float = Config.ELO_K_FACTOR, scale_factor: float = Config.BRADLEY_TERRY_SCALE_FACTOR):
        self.ratings = initial_ratings
        self.k_factor = k_factor
        self.scale_factor = scale_factor
        self.model_ids = list(initial_ratings.keys())

    def _expected_outcome(self, rating1: float, rating2: float) -> float:
        """
        Calculates the expected probability that model 1 wins against model 2.
        Using a logistic function as described in the paper (Section 4.2.3).
        """
        return 1 / (1 + np.exp(-(rating1 - rating2) / self.scale_factor))

    def update_ratings(self, winner: str, loser: str, is_tie: bool = False):
        """
        Updates the ratings of two models based on the outcome of a comparison.
        Args:
            winner: Name of the winning model.
            loser: Name of the losing model.
            is_tie: True if the outcome was a tie, False otherwise.
        """
        rating_winner = self.ratings[winner]
        rating_loser = self.ratings[loser]

        expected_winner = self._expected_outcome(rating_winner, rating_loser)
        expected_loser = self._expected_outcome(rating_loser, rating_winner)

        if is_tie:
            # For a tie, both models get a small update towards the expected draw
            self.ratings[winner] += self.k_factor * (0.5 - expected_winner)
            self.ratings[loser] += self.k_factor * (0.5 - expected_loser)
        else:
            # Winner's rating increases, loser's decreases
            self.ratings[winner] += self.k_factor * (1 - expected_winner)
            self.ratings[loser] += self.k_factor * (0 - expected_loser)

    def get_rankings(self) -> List[Tuple[str, float]]:
        """Returns models ranked by their current ratings."""
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)

    def get_perturbed_rankings(self, noise_scale: float) -> List[Tuple[str, float]]:
        """
        Returns models ranked by their ratings with added Gaussian noise (for mitigation scenario 2).
        Args:
            noise_scale: Standard deviation of the Gaussian noise.
        """
        if noise_scale == 0:
            return self.get_rankings()
        
        perturbed_ratings = {
            model: rating + np.random.normal(0, noise_scale) for model, rating in self.ratings.items()
        }
        return sorted(perturbed_ratings.items(), key=lambda item: item[1], reverse=True)

    def get_model_preference_probabilities(self, model_ratings: Dict[str, float]) -> Dict[str, float]:
        """
        Calculates the probability distribution Pr(i) that a user votes for model i
        over all other models, assuming models are chosen pairwise and independently.
        This is a simplification; the paper describes it as a product over pairwise preferences.
        For simulation, we approximate this by averaging pairwise win probabilities.
        """
        pr_i = {}
        for model_i in self.model_ids:
            total_prob = 0.0
            num_comparisons = 0
            for model_j in self.model_ids:
                if model_i != model_j:
                    total_prob += self._expected_outcome(model_ratings[model_i], model_ratings[model_j])
                    num_comparisons += 1
            pr_i[model_i] = total_prob / num_comparisons if num_comparisons > 0 else 0.0
        
        # Normalize probabilities
        sum_pr = sum(pr_i.values())
        if sum_pr > 0:
            return {model: prob / sum_pr for model, prob in pr_i.items()}
        return {model: 1.0 / len(self.model_ids) for model in self.model_ids}


class ChatbotArenaSimulator:
    """
    Simulates the Chatbot Arena environment and adversarial attacks.
    """
    def __init__(self,
                 models: List[str],
                 initial_ratings: Dict[str, float],
                 target_model: str,
                 detector_accuracy: float = Config.DETECTOR_ACCURACY,
                 false_positive_rate: float = Config.FALSE_POSITIVE_RATE,
                 false_negative_rate: float = Config.FALSE_NEGATIVE_RATE,
                 adversary_non_detection_strategy: str = Config.ATTACKER_NON_DETECTION_STRATEGY,
                 perturbed_leaderboard_noise_scale: float = Config.NOISE_SCALE,
                 random_state: int = Config.RANDOM_STATE):

        self.bt_model = BradleyTerryModel(initial_ratings)
        self.models = models
        self.target_model = target_model
        self.detector_accuracy = detector_accuracy
        self.false_positive_rate = false_positive_rate # P(detect target | not target)
        self.false_negative_rate = false_negative_rate # P(not detect target | target)
        self.adversary_non_detection_strategy = adversary_non_detection_strategy
        self.perturbed_leaderboard_noise_scale = perturbed_leaderboard_noise_scale
        self.random_state = random_state
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # Ensure detector_accuracy, FPR, FNR are consistent:
        # P(detect target | target) = detector_accuracy = 1 - FNR
        # P(detect target | not target) = FPR
        # For simplicity, we assume the detector_accuracy given is P(detect target | target).
        # We need to make sure 1 - FNR is actually detector_accuracy if FNR is given.
        # Or, we interpret detector_accuracy as overall accuracy, and assume FPR=FNR for simplicity.
        # Paper says: "detection accuracy of 95%, with symmetric false positive and false negative rates of 5%"
        # So: P(detect target | target) = 0.95, P(not detect target | target) = 0.05 (FNR)
        # And: P(detect target | not target) = 0.05 (FPR)

    def _simulate_detection(self, model_a: str, model_b: str) -> Tuple[bool, str]:
        """
        Simulates the attacker's de-anonymization process for two models.
        Returns: (True if target model is identified, identified_model_name)
        """
        identified_model = None
        
        # Check if model_a is the target model
        if model_a == self.target_model:
            if random.random() < (1 - self.false_negative_rate): # Successfully detect target
                identified_model = model_a
        else: # model_a is not the target model
            if random.random() < self.false_positive_rate: # False positive detection
                identified_model = model_a
        
        # Check if model_b is the target model (if not already identified)
        if identified_model is None:
            if model_b == self.target_model:
                if random.random() < (1 - self.false_negative_rate): # Successfully detect target
                    identified_model = model_b
            else: # model_b is not the target model
                if random.random() < self.false_positive_rate: # False positive detection
                    identified_model = model_b

        return identified_model is not None, identified_model

    def _adversarial_vote(self, model_a: str, model_b: str, identified_model: str, attack_type: str = "upvote") -> Tuple[str, str, bool]:
        """
        Simulates an adversarial vote given the identified model.
        Args:
            model_a, model_b: The two models presented.
            identified_model: The model the attacker believes is the target, or None.
            attack_type: "upvote" or "downvote"
        Returns: (winner, loser, is_tie)
        """
        if identified_model == self.target_model:
            # Attacker successfully identified target model
            if attack_type == "upvote":
                return self.target_model, (model_b if model_a == self.target_model else model_a), False # Target wins
            else: # downvote
                return (model_b if model_a == self.target_model else model_a), self.target_model, False # Target loses
        else:
            # Attacker did not identify target model, or had a false positive detection
            # Implement non-detection strategy
            if self.adversary_non_detection_strategy == "do_nothing":
                return None, None, False # No vote
            elif self.adversary_non_detection_strategy == "random_upvote":
                # Randomly upvote one of the two presented models
                winner = random.choice([model_a, model_b])
                loser = model_b if winner == model_a else model_a
                return winner, loser, False
            elif self.adversary_non_detection_strategy == "vote_tie":
                return model_a, model_b, True # Force a tie
            elif self.adversary_non_detection_strategy == "vote_tie_both_bad":
                return model_a, model_b, True # Similar to vote_tie for BT model
            else:
                raise ValueError(f"Unknown non-detection strategy: {self.adversary_non_detection_strategy}")

    def _benign_vote(self, model_a: str, model_b: str) -> Tuple[str, str, bool]:
        """
        Simulates a benign vote based on the current Bradley-Terry model probabilities.
        """
        prob_a_wins = self.bt_model._expected_outcome(self.bt_model.ratings[model_a], self.bt_model.ratings[model_b])
        if random.random() < prob_a_wins:
            return model_a, model_b, False
        else:
            return model_b, model_a, False

    def simulate_attack(self,
                        num_interactions: int,
                        attack_type: str = "upvote", # "upvote" or "downvote"
                        is_adversary: bool = True,
                        benign_user_voting_ratio: float = 0.5, # Ratio of benign votes in the system
                        track_ranks_every: int = Config.SIMULATION_ITERATIONS
                        ) -> Tuple[List[Dict[str, float]], List[int], List[int]]:
        """
        Runs a simulation of adversarial or benign voting on the Chatbot Arena.
        Args:
            num_interactions: Total number of interactions (queries submitted).
            attack_type: "upvote" or "downvote" for adversarial users.
            is_adversary: If True, simulate an adversary; otherwise, simulate a benign user.
            benign_user_voting_ratio: The proportion of votes that are benign.
            track_ranks_every: How often to record the current rankings.
        Returns:
            A tuple: (list of rating snapshots, list of votes cast, list of interactions)
        """
        all_rankings_snapshots = []
        cumulative_votes_cast = []
        cumulative_interactions_made = []

        interactions_count = 0
        votes_count = 0
        
        # Initial rankings
        all_rankings_snapshots.append({model: rating for model, rating in self.bt_model.get_rankings()})
        cumulative_votes_cast.append(0)
        cumulative_interactions_made.append(0)

        for i in range(num_interactions):
            interactions_count += 1
            
            # Select two random models for comparison
            model_a, model_b = random.sample(self.models, 2)

            winner, loser, is_tie = None, None, False

            if is_adversary and random.random() > benign_user_voting_ratio: # Simulate adversarial interaction
                # Attacker attempts to de-anonymize
                detected, identified_model = self._simulate_detection(model_a, model_b)

                if detected and identified_model is not None:
                    # Attacker identified a model (could be target or false positive)
                    if identified_model == self.target_model:
                        winner, loser, is_tie = self._adversarial_vote(model_a, model_b, identified_model, attack_type)
                    else: # False positive detection - target was not in the pair, but another model was identified
                        # Attacker still votes for/against the falsely identified model
                        # For simplicity, we assume if it's not the actual target, they abstain for now
                        # Or they might still vote for the falsely identified model if their goal is just to cause chaos
                        # Paper implies attacker only votes if *their* target was selected
                        winner, loser, is_tie = None, None, False # Attacker abstains on false positive of non-target
                else: # No model detected or false positive detection of a non-target model
                    # Attacker follows non-detection strategy
                    winner, loser, is_tie = self._adversarial_vote(model_a, model_b, None, attack_type="upvote") # Strategy handles abstention

            else: # Simulate benign interaction
                winner, loser, is_tie = self._benign_vote(model_a, model_b)

            if winner and loser: # A vote was cast
                self.bt_model.update_ratings(winner, loser, is_tie)
                votes_count += 1

            if (i + 1) % track_ranks_every == 0:
                all_rankings_snapshots.append({model: rating for model, rating in self.bt_model.get_rankings()})
                cumulative_votes_cast.append(votes_count)
                cumulative_interactions_made.append(interactions_count)
        
        # Add final snapshot
        if num_interactions % track_ranks_every != 0:
            all_rankings_snapshots.append({model: rating for model, rating in self.bt_model.get_rankings()})
            cumulative_votes_cast.append(votes_count)
            cumulative_interactions_made.append(interactions_count)


        return all_rankings_snapshots, cumulative_votes_cast, cumulative_interactions_made


class MaliciousUserDetector:
    """
    Implements malicious user identification based on voting patterns (Section 4.2.3).
    Scenario 1: Known Benign Distribution
    Scenario 2: Known Benign and Malicious Distributions (with perturbed leaderboard)
    """
    def __init__(self,
                 bt_model: BradleyTerryModel,
                 all_models: List[str],
                 significance_level_alpha: float = Config.SIGNIFICANCE_LEVEL_ALPHA,
                 num_simulated_sequences_for_p_value: int = Config.NUM_SIMULATED_SEQUENCES_FOR_P_VALUE,
                 random_state: int = Config.RANDOM_STATE):
        self.bt_model = bt_model
        self.all_models = all_models
        self.alpha = significance_level_alpha
        self.num_sim_seq = num_simulated_sequences_for_p_value
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)

    def _calculate_pairwise_prob(self, rating1: float, rating2: float) -> float:
        """Helper to calculate Pr(model1 preferred over model2)."""
        return 1 / (1 + np.exp(-(rating1 - rating2) / Config.BRADLEY_TERRY_SCALE_FACTOR))

    def _get_pr_model_wins_pair(self, model_a: str, model_b: str, ratings: Dict[str, float]) -> Dict[str, float]:
        """
        Returns {model_a: Pr(a wins), model_b: Pr(b wins)} based on given ratings.
        """
        prob_a_wins = self._calculate_pairwise_prob(ratings[model_a], ratings[model_b])
        return {model_a: prob_a_wins, model_b: 1 - prob_a_wins}

    def _get_vote_probabilities(self, models_in_comparison: Tuple[str, str], current_ratings: Dict[str, float],
                                adversarial_target: str = None, attack_type: str = "upvote") -> Dict[str, float]:
        """
        Calculates probabilities of voting for each model in a pair, considering benign or adversarial behavior.
        If adversarial_target is None, assume benign.
        If adversarial_target is not None, assume attacker identified target and votes accordingly.
        Returns: {model_name: probability_of_winning} for the given pair.
        """
        model_a, model_b = models_in_comparison
        
        # Benign behavior: vote based on Bradley-Terry probabilities
        prob_a_wins_benign = self._calculate_pairwise_prob(current_ratings[model_a], current_ratings[model_b])
        prob_b_wins_benign = 1 - prob_a_wins_benign

        # Adversarial behavior: assumes perfect detection and voting for/against target
        if adversarial_target:
            if model_a == adversarial_target:
                if attack_type == "upvote":
                    return {model_a: 1.0, model_b: 0.0}
                else: # downvote
                    return {model_a: 0.0, model_b: 1.0}
            elif model_b == adversarial_target:
                if attack_type == "upvote":
                    return {model_a: 0.0, model_b: 1.0}
                else: # downvote
                    return {model_a: 1.0, model_b: 0.0}
        
        # If no target or target not in pair (for adversary), fallback to benign or other non-detection strategy
        # For the purpose of malicious user detection, we need the *expected* voting pattern,
        # so if target is not in pair, the adversary will vote according to benign probabilities or abstain.
        # Here we model the probability of voting a certain way if a vote *is* cast.
        return {model_a: prob_a_wins_benign, model_b: prob_b_wins_benign} # Assuming benign for non-target pairs.


    def detect_malicious_user_scenario1(self,
                                        user_vote_history: List[Tuple[str, str, str]], # [(model_a, model_b, winner), ...]
                                        benign_ratings: Dict[str, float]
                                        ) -> Tuple[bool, float]:
        """
        Detects malicious users assuming only benign distribution is known (Scenario 1).
        Args:
            user_vote_history: List of (model_a, model_b, winner) tuples for the user.
            benign_ratings: The known ratings reflecting benign user preferences.
        Returns:
            Tuple: (is_malicious: bool, p_value: float)
        """
        if not user_vote_history:
            return False, 1.0 # Cannot detect with no history

        # Calculate likelihood of observed sequence under benign hypothesis
        log_likelihood_observed = 0.0
        for model_a, model_b, winner in user_vote_history:
            probs = self._get_vote_probabilities((model_a, model_b), benign_ratings, adversarial_target=None)
            if winner == model_a:
                log_likelihood_observed += np.log(probs.get(model_a, 1e-9))
            elif winner == model_b:
                log_likelihood_observed += np.log(probs.get(model_b, 1e-9))
            # Ties are not explicitly handled by _get_vote_probabilities, assuming winner/loser pairs for simplicity
            # For a tie, the contribution to likelihood is for 0.5 probability for each, but we simplify to winner for now.

        test_statistic_observed = -2 * log_likelihood_observed

        # Simulate sequences under the benign hypothesis
        simulated_test_statistics = []
        for _ in range(self.num_sim_seq):
            log_likelihood_simulated = 0.0
            for model_a, model_b, _ in user_vote_history: # Use the same pairs as observed history
                probs = self._get_vote_probabilities((model_a, model_b), benign_ratings, adversarial_target=None)
                # Simulate a vote based on benign probabilities
                simulated_winner = model_a if random.random() < probs.get(model_a, 0.5) else model_b
                log_likelihood_simulated += np.log(probs.get(simulated_winner, 1e-9))
            simulated_test_statistics.append(-2 * log_likelihood_simulated)

        p_value = np.mean([1 for ts in simulated_test_statistics if ts >= test_statistic_observed])
        return p_value < self.alpha, p_value

    def detect_malicious_user_scenario2(self,
                                        user_vote_history: List[Tuple[str, str, str]],
                                        benign_ratings: Dict[str, float],
                                        adversarial_ratings: Dict[str, float], # Perturbed ratings known by adversary
                                        target_model_of_adversary: str,
                                        attack_type: str = "upvote"
                                        ) -> Tuple[bool, float]:
        """
        Detects malicious users using Neyman-Pearson Lemma (likelihood ratio test)
        assuming benign and (mimicked) adversarial distributions are known. (Scenario 2).
        Args:
            user_vote_history: List of (model_a, model_b, winner) tuples for the user.
            benign_ratings: The true benign ratings.
            adversarial_ratings: The (perturbed) ratings the adversary is trying to mimic.
            target_model_of_adversary: The model the adversary is targeting.
            attack_type: "upvote" or "downvote"
        Returns:
            Tuple: (is_malicious: bool, likelihood_ratio: float)
        """
        if not user_vote_history:
            return False, 1.0 # Cannot detect with no history

        log_likelihood_ratio = 0.0
        for model_a, model_b, winner in user_vote_history:
            # P_M(x): Probability of observation under malicious (mimicked) hypothesis
            # If target model is in pair, adversary votes strategically.
            # Otherwise, adversary votes based on perturbed leaderboard.
            probs_malicious = self._get_vote_probabilities(
                (model_a, model_b), adversarial_ratings, target_model_of_adversary, attack_type
            )
            # P_B(x): Probability of observation under benign hypothesis
            probs_benign = self._get_vote_probabilities((model_a, model_b), benign_ratings)

            prob_mal_observed = 1e-9
            prob_ben_observed = 1e-9

            if winner == model_a:
                prob_mal_observed = probs_malicious.get(model_a, 1e-9)
                prob_ben_observed = probs_benign.get(model_a, 1e-9)
            elif winner == model_b:
                prob_mal_observed = probs_malicious.get(model_b, 1e-9)
                prob_ben_observed = probs_benign.get(model_b, 1e-9)
            
            log_likelihood_ratio += np.log(prob_mal_observed / prob_ben_observed)

        # For Neyman-Pearson, we reject if likelihood ratio is above a threshold.
        # The threshold depends on alpha and the distribution of the likelihood ratio.
        # A simpler approach is to use the p-value as in Scenario 1, or just report LR.
        # Here, we'll just return the log-likelihood ratio for now, assuming a threshold would be set.
        # A high positive value indicates malicious behavior.
        # We don't have enough info in the paper to derive the threshold for NP test,
        # so we'll just indicate "malicious" if LR is significantly positive.
        # For simplicity, we can set a heuristic threshold for now or rely on external analysis.
        # The paper says: "We can use the Neyman-Pearson Lemma to construct the hypothesis test."
        # but doesn't provide the threshold derivation.
        
        # Let's use a heuristic for now, or assume external calibration would provide a threshold.
        # For simulation, we can consider if log_likelihood_ratio is positive.
        is_malicious = log_likelihood_ratio > 0.0 # Heuristic
        return is_malicious, log_likelihood_ratio


class AttackCostModel:
    """
    Calculates the total cost of an attack as described in Section 4.1.
    """
    def __init__(self,
                 detector_training_cost: float,
                 cost_account_maintenance: float = Config.COST_ACCOUNT_MAINTENANCE,
                 cost_action: float = Config.COST_ACTION,
                 max_actions_per_account: int = Config.MAX_ACTIONS_PER_ACCOUNT):
        self.c_detector = detector_training_cost
        self.c_account = cost_account_maintenance
        self.c_action = cost_action
        self.m = max_actions_per_account

    def calculate_total_cost(self, num_actions: int) -> float:
        """
        Calculates the total attack cost.
        Args:
            num_actions: Total number of actions (interactions or votes) required for the attack.
        Returns:
            Total attack cost.
        """
        if self.m == 0: # Avoid division by zero if max_actions_per_account is 0 (shouldn't happen)
            num_accounts = num_actions if num_actions > 0 else 0
        else:
            num_accounts = np.ceil(num_actions / self.m) if num_actions > 0 else 0
        
        account_maintenance_cost = num_accounts * self.c_account
        action_cost = num_actions * self.c_action
        
        total_cost = self.c_detector + account_maintenance_cost + action_cost
        return total_cost
