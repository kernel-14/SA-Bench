import logging
import random
import numpy as np
import math
import json # For loading historical votes

from typing import Dict, List, Any, Optional, Tuple, Union
from config import Config
# Assuming Detector class is imported from detector.py.
# This assumes no circular dependency issue, as per design.
# However, the design for `_determine_detection_outcome` and `simulate_adversarial_attack`
# does not actually use an instance of the `Detector` class for the `_determine_detection_outcome`
# logic itself, but rather parameters like `detector_accuracy`, `fp_rate`, `fn_rate`.
# The `Detector` object is passed into ablation methods in the design, but not directly used
# in the core simulation for detection logic.
# I will keep the `detector: Detector` parameter in `simulate_adversarial_attack` and ablation
# methods as per design, but note it's not strictly used for the detection *logic* inside
# `_determine_detection_outcome`, as that logic is statistical.
from detector import Detector 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LeaderboardSimulator:
    """
    Simulates the Chatbot Arena leaderboard using Bradley-Terry coefficients and adversarial
    voting behavior. It includes functionalities for initializing ratings, updating them
    based on votes, and simulating adversarial attacks to change model rankings.
    """

    def __init__(self, config: Config, models: List[Dict], historical_votes_path: str):
        """
        Initializes the simulator with configuration, available models, and historical voting data.

        Args:
            config (Config): The configuration object containing experiment parameters.
            models (List[Dict]): A list of model dictionaries (e.g., from config.MODEL_LIST).
            historical_votes_path (str): Path to a JSON file containing historical voting data.
                                         Each entry should be a dict with 'model_a_id', 'model_b_id', 'outcome'.
                                         Outcome can be 'A_wins', 'B_wins', 'tie'.
        """
        self._config = config
        self._models_list = models
        self._model_ids = [model['model_id'] for model in models]

        # Set numpy random seed for reproducibility of model selection and probabilities
        np.random.seed(self._config.RANDOM_SEED)
        random.seed(self._config.RANDOM_SEED)

        # K-factor for Elo-like rating updates. A common value for K-factor in Chess Elo is 32.
        # This is not directly in config.yaml so defining a default here.
        # If config.yaml is updated later, this can be moved there.
        self._k_factor: float = 32.0 

        # Load historical votes
        self._historical_votes: List[Dict] = []
        try:
            with open(historical_votes_path, 'r', encoding='utf-8') as f:
                self._historical_votes = json.load(f)
            logger.info(f"Loaded {len(self._historical_votes)} historical votes from {historical_votes_path}.")
        except FileNotFoundError:
            logger.error(f"Historical votes file not found at {historical_votes_path}. Initial ratings will be default based on SIMULATOR_BRADLEY_TERRY_INITIAL_RATING.")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {historical_votes_path}. Initial ratings will be default.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading historical votes from {historical_votes_path}: {e}. Initial ratings will be default.")

    def _calculate_win_probability(self, rating_a: float, rating_b: float) -> float:
        """
        Helper function to calculate the probability of model_a winning against model_b
        based on their Bradley-Terry (Elo-like) ratings.

        Args:
            rating_a (float): The current rating of model A.
            rating_b (float): The current rating of model B.

        Returns:
            float: The probability of model A winning against model B (between 0 and 1).
        """
        # s is the scaling factor for the logistic function
        s: float = self._config.SIMULATOR_BRADLEY_TERRY_SCALE_FACTOR
        return 1 / (1 + math.exp(-(rating_a - rating_b) / s))

    def initialize_ratings(self) -> Dict[str, float]:
        """
        Establishes the initial Bradley-Terry (Elo-like) ratings for all models
        by running a pre-simulation over the provided historical votes.
        If no historical votes, initializes all models to a default rating.

        Returns:
            Dict[str, float]: A dictionary mapping model IDs to their initial ratings.
        """
        ratings: Dict[str, float] = {
            model_id: self._config.SIMULATOR_BRADLEY_TERRY_INITIAL_RATING
            for model_id in self._model_ids
        }

        if not self._historical_votes:
            logger.warning("No historical votes loaded. All models initialized with default ratings.")
            return ratings

        logger.info(f"Initializing ratings by processing {len(self._historical_votes)} historical votes...")
        for vote in self._historical_votes:
            model_a_id = vote.get('model_a_id')
            model_b_id = vote.get('model_b_id')
            outcome = vote.get('outcome')

            # Ensure both models exist in our list and the outcome is valid
            if model_a_id not in self._model_ids or model_b_id not in self._model_ids:
                logger.debug(f"Skipping historical vote with unknown model_id: {model_a_id} vs {model_b_id}")
                continue
            if outcome not in ["A_wins", "B_wins", "tie"]: # Only process known outcomes for historical votes
                logger.debug(f"Skipping historical vote with invalid outcome: {outcome}")
                continue

            self.update_bradley_terry(ratings, model_a_id, model_b_id, outcome, self._k_factor)
        
        logger.info("Historical votes processed for initial rating setup.")
        return ratings

    def update_bradley_terry(
        self,
        ratings: Dict[str, float],
        model_a: str,
        model_b: str,
        outcome: str,
        k_factor: float
    ) -> Dict[str, float]:
        """
        Adjusts the Bradley-Terry (Elo-like) ratings of two models based on a single comparison outcome.

        Args:
            ratings (Dict[str, float]): Current ratings of all models.
            model_a (str): ID of the first model in the comparison.
            model_b (str): ID of the second model in the comparison.
            outcome (str): The result of the comparison ('A_wins', 'B_wins', 'tie', 'vote_bad_both').
            k_factor (float): The K-factor for rating updates, determining sensitivity to changes.

        Returns:
            Dict[str, float]: The updated ratings dictionary.
        """
        # Use .get() with a default in case a model_id is unexpectedly not in ratings
        rating_a: float = ratings.get(model_a, self._config.SIMULATOR_BRADLEY_TERRY_INITIAL_RATING)
        rating_b: float = ratings.get(model_b, self._config.SIMULATOR_BRADLEY_TERRY_INITIAL_RATING)

        expected_a: float = self._calculate_win_probability(rating_a, rating_b)
        expected_b: float = self._calculate_win_probability(rating_b, rating_a)

        actual_a: float
        actual_b: float

        if outcome == "A_wins":
            actual_a = 1.0
            actual_b = 0.0
        elif outcome == "B_wins":
            actual_a = 0.0
            actual_b = 1.0
        elif outcome in ["tie", "vote_bad_both"]: # "vote_bad_both" treated as a tie for rating updates, as per "Anything UNCLEAR".
            actual_a = 0.5
            actual_b = 0.5
        else:
            logger.warning(f"Unknown outcome '{outcome}'. No rating update for {model_a} vs {model_b}.")
            return ratings # No change for unknown outcomes

        new_rating_a: float = rating_a + k_factor * (actual_a - expected_a)
        new_rating_b: float = rating_b + k_factor * (actual_b - expected_b)

        ratings[model_a] = new_rating_a
        ratings[model_b] = new_rating_b
        return ratings

    def get_current_rankings(self, ratings: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Ranks models based on their current Bradley-Terry ratings.

        Args:
            ratings (Dict[str, float]): A dictionary mapping model IDs to their current ratings.

        Returns:
            List[Tuple[str, float]]: A list of (model_id, rating) tuples, sorted by rating
                                     in descending order (highest rating is rank 1).
        """
        sorted_rankings: List[Tuple[str, float]] = sorted(ratings.items(), key=lambda item: item[1], reverse=True)
        return sorted_rankings
    
    def _get_model_current_rank(self, model_id: str, rankings: List[Tuple[str, float]]) -> Optional[int]:
        """
        Helper to get the current rank of a specific model from a list of sorted rankings.

        Args:
            model_id (str): The ID of the model to find the rank for.
            rankings (List[Tuple[str, float]]): The list of (model_id, rating) tuples, sorted by rating.

        Returns:
            Optional[int]: The 1-based rank of the model, or None if the model is not found.
        """
        for i, (mid, _) in enumerate(rankings):
            if mid == model_id:
                return i + 1
        return None


    def _determine_detection_outcome(
        self,
        model_a_id: str,
        model_b_id: str,
        target_model_id: str,
        detector_accuracy: float,
        fp_rate: float,
        fn_rate: float
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Simulates the adversary's detector identifying a target model (or failing to)
        for two given models, considering accuracy, false positive, and false negative rates.

        Args:
            model_a_id (str): ID of the first model presented.
            model_b_id (str): ID of the second model presented.
            target_model_id (str): The ID of the model the adversary is targeting.
            detector_accuracy (float): The probability of correctly identifying the target model (True Positive Rate).
            fp_rate (float): False positive rate (probability of incorrectly identifying a non-target model as target).
            fn_rate (float): False negative rate (probability of failing to identify the target model).
                             Note: True Negative Rate = 1 - FP_rate for non-target models.
                                   True Positive Rate = Detector Accuracy = 1 - FN_rate for target models.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple (identified_model_a, identified_model_b).
                                                  Each element is `target_model_id` if identified, else `None`.
        """
        identified_a: Optional[str] = None
        identified_b: Optional[str] = None

        # Simulate detection for model A
        if model_a_id == target_model_id:
            # Model A is the target. Simulate True Positive (correctly identified) or False Negative (missed).
            if random.random() < detector_accuracy: 
                identified_a = target_model_id
            # else: identified_a remains None (False Negative)
        else: 
            # Model A is NOT the target. Simulate False Positive (incorrectly identified) or True Negative (correctly not identified).
            if random.random() < fp_rate: 
                identified_a = target_model_id # False Positive
            # else: identified_a remains None (True Negative)

        # Simulate detection for model B
        if model_b_id == target_model_id:
            # Model B is the target. Simulate True Positive or False Negative.
            if random.random() < detector_accuracy: 
                identified_b = target_model_id
            # else: identified_b remains None (False Negative)
        else: 
            # Model B is NOT the target. Simulate False Positive or True Negative.
            if random.random() < fp_rate: 
                identified_b = target_model_id # False Positive
            # else: identified_b remains None (True Negative)
            
        return identified_a, identified_b

    def simulate_adversarial_attack(
        self,
        target_model_id: str,
        target_rank_objective: int,
        is_upvote: bool,
        detector_accuracy: float,
        false_positive_rate: float,
        false_negative_rate: float,
        non_detection_strategy: str,
        max_interactions: int = 5_000_000 # Safety break to prevent infinite loops
    ) -> Dict[str, Union[int, List[Tuple[int, int, int]]]]:
        """
        Simulates the adversarial attack to shift a target model's rank until an objective is met.

        Args:
            target_model_id (str): The ID of the model the adversary wants to manipulate.
            target_rank_objective (int): The desired rank for the target model.
                                         For upvote, achieve rank <= objective.
                                         For downvote, achieve rank >= objective.
            is_upvote (bool): True if the adversary aims to boost the rank, False to reduce it.
            detector_accuracy (float): The accuracy (True Positive Rate) of the adversary's target model detector.
            false_positive_rate (float): The false positive rate of the detector.
            false_negative_rate (float): The false negative rate of the detector.
            non_detection_strategy (str): Action when target is not detected and no clear vote is possible
                                          ('do_nothing', 'random_upvote', 'vote_tie', 'vote_bad_both').
            max_interactions (int): Maximum number of interactions before stopping the simulation.

        Returns:
            Dict[str, Union[int, List[Tuple[int, int, int]]]]: A dictionary containing:
                - 'total_interactions': Total number of interactions simulated.
                - 'total_votes_cast': Total number of votes cast by the adversary.
                - 'rank_history': A list of (interaction_count, votes_cast, current_target_rank) at intervals.
        """
        if target_model_id not in self._model_ids:
            logger.error(f"Target model '{target_model_id}' not found in the list of available models. Aborting simulation.")
            return {'total_interactions': 0, 'total_votes_cast': 0, 'rank_history': []}

        current_ratings: Dict[str, float] = self.initialize_ratings()
        total_interactions: int = 0
        total_votes_cast: int = 0
        rank_history: List[Tuple[int, int, int]] = [] # (interactions, votes, rank)

        initial_rankings: List[Tuple[str, float]] = self.get_current_rankings(current_ratings)
        initial_target_rank: Optional[int] = self._get_model_current_rank(target_model_id, initial_rankings)
        if initial_target_rank is None: # This case should ideally be caught by the first check, but for safety
            logger.error(f"Could not determine initial rank for target model '{target_model_id}'. Aborting simulation.")
            return {'total_interactions': 0, 'total_votes_cast': 0, 'rank_history': []}

        logger.info(f"Starting simulation for target model '{target_model_id}'. "
                    f"Initial rank: {initial_target_rank}, Objective rank: {target_rank_objective} ({'Up' if is_upvote else 'Down'}).")
        
        current_target_rank: int = initial_target_rank # Initialize with actual initial rank

        while total_interactions < max_interactions:
            total_interactions += 1

            # Randomly select two distinct models
            if len(self._model_ids) < 2:
                logger.error("Not enough models to perform a head-to-head comparison. Aborting simulation.")
                break
            selected_models: np.ndarray = np.random.choice(self._model_ids, 2, replace=False)
            model_a_id: str = selected_models[0]
            model_b_id: str = selected_models[1]

            # Simulate adversary's detection
            identified_a, identified_b = self._determine_detection_outcome(
                model_a_id, model_b_id, target_model_id, detector_accuracy, fp_rate, fn_rate
            )

            vote_outcome: Optional[str] = None
            vote_performed_by_adversary: bool = False

            # Adversary's Voting Decision (as per paper: "only votes if they have identified the target model in one of the two responses.")
            if identified_a == target_model_id and identified_b is None:
                # Target model A identified, Model B not identified (or false negative)
                vote_outcome = "A_wins" if is_upvote else "B_wins"
                vote_performed_by_adversary = True
            elif identified_b == target_model_id and identified_a is None:
                # Target model B identified, Model A not identified (or false negative)
                vote_outcome = "B_wins" if is_upvote else "A_wins"
                vote_performed_by_adversary = True
            # In other cases (both identified, neither identified, different models falsely identified),
            # the adversary does not make a *targeted* vote. This falls under the non_detection_strategy.

            if vote_performed_by_adversary:
                total_votes_cast += 1
                if vote_outcome is not None: # Ensure vote_outcome is not None, though logic implies it won't be here
                    current_ratings = self.update_bradley_terry(current_ratings, model_a_id, model_b_id, vote_outcome, self._k_factor)
            else:
                # Execute non-detection strategy if no targeted vote was cast
                if non_detection_strategy == "random_upvote":
                    vote_outcome = "A_wins" if random.random() < 0.5 else "B_wins"
                    total_votes_cast += 1
                    current_ratings = self.update_bradley_terry(current_ratings, model_a_id, model_b_id, vote_outcome, self._k_factor)
                elif non_detection_strategy == "vote_tie":
                    vote_outcome = "tie"
                    total_votes_cast += 1
                    current_ratings = self.update_bradley_terry(current_ratings, model_a_id, model_b_id, vote_outcome, self._k_factor)
                elif non_detection_strategy == "vote_bad_both":
                    # Treated as a tie for Bradley-Terry updates as per problem statement's "Anything UNCLEAR"
                    vote_outcome = "vote_bad_both" 
                    total_votes_cast += 1
                    current_ratings = self.update_bradley_terry(current_ratings, model_a_id, model_b_id, vote_outcome, self._k_factor)
                elif non_detection_strategy == "do_nothing":
                    pass # No vote, no rating update
                else:
                    logger.warning(f"Unknown non-detection strategy: {non_detection_strategy}. Defaulting to 'do_nothing'.")
                    pass

            # Check objective at defined intervals
            if total_interactions % self._config.SIMULATOR_INTERACTION_TRACKING_INTERVAL == 0:
                current_rankings: List[Tuple[str, float]] = self.get_current_rankings(current_ratings)
                current_target_rank = self._get_model_current_rank(target_model_id, current_rankings)
                
                if current_target_rank is None: 
                    logger.error(f"Target model '{target_model_id}' disappeared from rankings. This should not happen. Aborting.")
                    break

                rank_history.append((total_interactions, total_votes_cast, current_target_rank))
                # logger.debug(f"Interactions: {total_interactions}, Votes: {total_votes_cast}, Target Rank ({target_model_id}): {current_target_rank}")

                # Check if objective is met
                if (is_upvote and current_target_rank <= target_rank_objective) or \
                   (not is_upvote and current_target_rank >= target_rank_objective):
                    logger.info(f"Objective met: '{target_model_id}' achieved rank {current_target_rank} (target {'<=' if is_upvote else '>='} {target_rank_objective}) after {total_interactions} interactions and {total_votes_cast} votes.")
                    break
            
            if total_interactions == max_interactions:
                logger.warning(f"Maximum interactions ({max_interactions}) reached for target model '{target_model_id}' without achieving objective rank {target_rank_objective}.")
                break


        # Append final state if objective was met between tracking intervals or max_interactions was hit
        final_rankings: List[Tuple[str, float]] = self.get_current_rankings(current_ratings)
        final_target_rank: Optional[int] = self._get_model_current_rank(target_model_id, final_rankings)
        # Only append if rank_history is empty or the last entry is different from current final state
        if final_target_rank is not None and \
           (not rank_history or rank_history[-1] != (total_interactions, total_votes_cast, final_target_rank)):
            rank_history.append((total_interactions, total_votes_cast, final_target_rank))

        return {
            'total_interactions': total_interactions,
            'total_votes_cast': total_votes_cast,
            'rank_history': rank_history
        }

    def run_ablation_detector_accuracy(
        self,
        target_model_id: str,
        target_rank_objective: int,
        is_upvote: bool,
        detector_accuracies_to_test: List[float],
        non_detection_strategy: str
    ) -> Dict[str, Dict[str, Union[int, List[Tuple[int, int, int]]]]]:
        """
        Conducts an ablation study on varying detector accuracies.

        Args:
            target_model_id (str): The ID of the model to manipulate.
            target_rank_objective (int): The desired rank objective.
            is_upvote (bool): True for upvoting, False for downvoting.
            detector_accuracies_to_test (List[float]): List of detector accuracy values to test.
            non_detection_strategy (str): Strategy to use when target is not detected.

        Returns:
            Dict[str, Dict[str, Union[int, List[Tuple[int, int, int]]]]]:
                A dictionary where keys are string representations of detector accuracies,
                and values are the simulation results for that accuracy.
        """
        logger.info(f"Starting ablation study for detector accuracy on model '{target_model_id}'...")
        ablation_results: Dict[str, Dict[str, Union[int, List[Tuple[int, int, int]]]]] = {}
        for acc in detector_accuracies_to_test:
            logger.info(f"  Testing with detector accuracy: {acc * 100:.1f}%")
            # The paper assumes symmetric false positive/negative rates for ablation: FP_rate = FN_rate = 1 - accuracy
            fp_rate: float = 1.0 - acc
            fn_rate: float = 1.0 - acc
            
            results = self.simulate_adversarial_attack(
                target_model_id=target_model_id,
                target_rank_objective=target_rank_objective,
                is_upvote=is_upvote,
                detector_accuracy=acc,
                false_positive_rate=fp_rate,
                false_negative_rate=fn_rate,
                non_detection_strategy=non_detection_strategy
            )
            ablation_results[f"detector_acc_{acc:.2f}"] = results
        logger.info("Detector accuracy ablation study completed.")
        return ablation_results

    def run_ablation_non_detection_strategy(
        self,
        target_model_id: str,
        target_rank_objective: int,
        is_upvote: bool,
        detector_accuracy: float,
        strategies_to_test: List[str]
    ) -> Dict[str, Dict[str, Union[int, List[Tuple[int, int, int]]]]]:
        """
        Conducts an ablation study on different strategies when the target model is not detected.

        Args:
            target_model_id (str): The ID of the model to manipulate.
            target_rank_objective (int): The desired rank objective.
            is_upvote (bool): True for upvoting, False for downvoting.
            detector_accuracy (float): The fixed detector accuracy for this ablation.
            strategies_to_test (List[str]): List of non-detection strategies to test.

        Returns:
            Dict[str, Dict[str, Union[int, List[Tuple[int, int, int]]]]]:
                A dictionary where keys are strategy names, and values are the
                simulation results for that strategy.
        """
        logger.info(f"Starting ablation study for non-detection strategies on model '{target_model_id}'...")
        ablation_results: Dict[str, Dict[str, Union[int, List[Tuple[int, int, int]]]]] = {}
        # Use fixed FP/FN rates from config for this ablation, as detector accuracy is fixed
        fp_rate: float = self._config.SIMULATOR_FALSE_POSITIVE_RATE
        fn_rate: float = self._config.SIMULATOR_FALSE_NEGATIVE_RATE

        for strategy in strategies_to_test:
            logger.info(f"  Testing with non-detection strategy: '{strategy}'")
            results = self.simulate_adversarial_attack(
                target_model_id=target_model_id,
                target_rank_objective=target_rank_objective,
                is_upvote=is_upvote,
                detector_accuracy=detector_accuracy,
                false_positive_rate=fp_rate,
                false_negative_rate=fn_rate,
                non_detection_strategy=strategy
            )
            ablation_results[strategy] = results
        logger.info("Non-detection strategy ablation study completed.")
        return ablation_results

