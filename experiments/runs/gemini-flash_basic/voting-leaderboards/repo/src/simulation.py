import random
import json
from collections import defaultdict
from math import exp

# Assuming de_anonymization.py is in the same directory
from de_anonymization import IdentityProbingDetector, TrainingBasedDetector

class EloRatingSystem:
    """Simulates an Elo rating system for models based on pairwise comparisons.
    The k_factor determines how much ratings change per game.
    """
    def __init__(self, initial_rating=1500, k_factor=32):
        self.ratings = defaultdict(lambda: initial_rating)
        self.k_factor = k_factor

    def _expected_score(self, rating1, rating2):
        """Calculates the expected score of player 1 against player 2."""
        return 1 / (1 + exp((rating2 - rating1) / 400))

    def update_ratings(self, winner_model, loser_model, draw=False):
        """Updates Elo ratings based on a game outcome."""
        rating_winner = self.ratings[winner_model]
        rating_loser = self.ratings[loser_model]

        expected_winner = self._expected_score(rating_winner, rating_loser)
        expected_loser = self._expected_score(rating_loser, rating_winner)

        if draw:
            score_winner = 0.5
            score_loser = 0.5
        else:
            score_winner = 1
            score_loser = 0

        self.ratings[winner_model] += self.k_factor * (score_winner - expected_winner)
        self.ratings[loser_model] += self.k_factor * (score_loser - expected_loser)

    def get_rankings(self):
        """Returns models ranked by their current Elo rating."""
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)


class AdversarialVotingSimulator:
    """Simulates an attacker manipulating a leaderboard using de-anonymization.
    """
    def __init__(self, models, prompts_data, config):
        self.models = models
        self.prompts_data = prompts_data
        self.config = config
        self.simulation_params = config["simulation_params"]

        self.elo_system = EloRatingSystem()
        # Initialize all models with the base rating
        for model in self.models:
            self.elo_system.ratings[model] = 1500 # Default initial rating

        self.identity_detector = IdentityProbingDetector(
            models, config["identity_probing_prompts"], self.simulation_params)
        
        self.training_detector = TrainingBasedDetector(
            models, prompts_data, self.simulation_params)
        
        # Assume a fixed detection accuracy for the adversarial voting simulation,
        # based on the paper's reported >95% for training-based detector.
        self.detection_accuracy = 0.95 # From paper, Section 2.4
        self.false_positive_rate = 0.05 / (len(self.models) - 1) # Distribute FPs among other models
        self.false_negative_rate = 0.05

    def _simulate_detection(self, target_model, model_a, model_b, detection_method='training_based'):
        """Simulates the de-anonymization process.
        Returns the detected model (target_model or None) for each response.
        """
        # This is a simplified simulation of the detector's outcome
        detected_a = None
        detected_b = None

        if detection_method == 'training_based':
            # Simulate 95% accuracy for target, and distributed FPs for others
            # If model_a is target:
            if model_a == target_model:
                if random.random() < (1 - self.false_negative_rate): # True Positive
                    detected_a = model_a
            else: # model_a is not target
                if random.random() < self.false_positive_rate: # False Positive
                    detected_a = target_model # Incorrectly identifies non-target as target
            
            # If model_b is target:
            if model_b == target_model:
                if random.random() < (1 - self.false_negative_rate): # True Positive
                    detected_b = model_b
            else: # model_b is not target
                if random.random() < self.false_positive_rate: # False Positive
                    detected_b = target_model # Incorrectly identifies non-target as target
        
        # In a real scenario, the detector would give a score/probability
        # and we would pick the highest probability above a threshold.
        # For this simulation, we assume a binary detection outcome as per paper's description.
        return detected_a, detected_b


    def simulate_attack(self, target_model, objective='up', x_positions=1, max_interactions=100000):
        """Simulates an adversarial attack to change the ranking of a target model.
        """
        initial_rankings = self.elo_system.get_rankings()
        initial_target_rank = next((i + 1 for i, (model, _) in enumerate(initial_rankings) if model == target_model), -1)
        
        if initial_target_rank == -1:
            print(f"Error: Target model {target_model} not found in models list.")
            return 0, 0, []

        interactions = 0
        votes_cast = 0
        rank_history = []

        print(f"
Simulating {objective} attack on {target_model} by {x_positions} positions...")
        print(f"Initial rank of {target_model}: {initial_target_rank}")

        while interactions < max_interactions:
            interactions += 1
            
            # Randomly select two models for comparison
            model_a, model_b = random.sample(self.models, 2)

            # Simulate de-anonymization
            # For simplicity, we directly use _simulate_detection here based on probabilities
            # In a full integration, we would generate actual responses and run them through the detectors
            detected_a, detected_b = self._simulate_detection(target_model, model_a, model_b)

            vote_for_target = False
            vote_against_target = False

            if objective == 'up': # Promote target model
                if detected_a == target_model and detected_b != target_model: # Target vs Other
                    vote_for_target = True # Vote for A
                elif detected_b == target_model and detected_a != target_model: # Other vs Target
                    vote_for_target = True # Vote for B
                elif detected_a == target_model and detected_b == target_model: # Target vs Target - unexpected, abstains
                    pass
                elif detected_a != target_model and detected_b != target_model: # Other vs Other - abstains
                    pass
            
            elif objective == 'down': # Demote target model
                if detected_a == target_model and detected_b != target_model: # Target vs Other
                    vote_against_target = True # Vote for B
                elif detected_b == target_model and detected_a != target_model: # Other vs Target
                    vote_against_target = True # Vote for A
                elif detected_a == target_model and detected_b == target_model: # Target vs Target - unexpected, abstains
                    pass
                elif detected_a != target_model and detected_b != target_model: # Other vs Other - abstains
                    pass
            
            if vote_for_target:
                if detected_a == target_model: # Target is A, Other is B
                    self.elo_system.update_ratings(model_a, model_b)
                else: # Target is B, Other is A
                    self.elo_system.update_ratings(model_b, model_a)
                votes_cast += 1
            elif vote_against_target:
                if detected_a == target_model: # Target is A, Other is B
                    self.elo_system.update_ratings(model_b, model_a) # Vote for B, against A
                else: # Target is B, Other is A
                    self.elo_system.update_ratings(model_a, model_b) # Vote for A, against B
                votes_cast += 1
            
            current_rankings = self.elo_system.get_rankings()
            current_target_rank = next((i + 1 for i, (model, _) in enumerate(current_rankings) if model == target_model), -1)
            rank_history.append((interactions, votes_cast, current_target_rank))

            # Check if objective is met
            if objective == 'up' and current_target_rank <= (initial_target_rank - x_positions):
                print(f"Objective met! {target_model} moved up by {x_positions} positions.")
                return interactions, votes_cast, rank_history
            elif objective == 'down' and current_target_rank >= (initial_target_rank + x_positions):
                print(f"Objective met! {target_model} moved down by {x_positions} positions.")
                return interactions, votes_cast, rank_history
            
            if interactions % 1000 == 0:
                print(f"Interactions: {interactions}, Votes: {votes_cast}, Current rank of {target_model}: {current_target_rank}")

        print(f"Max interactions ({max_interactions}) reached. Objective not fully met.")
        return interactions, votes_cast, rank_history


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Load configuration and prompt data
    try:
        with open('/mnt/paper2any/pzw/proj/paperagent/hx/Research_space/SemanticAlign-Bench/experiments/runs/gemini-flash_basic/voting-leaderboards/voting-leaderboards-20260505-151946/workspace/repo/config.yaml', 'r') as f:
            config = json.load(f)
        with open('/mnt/paper2any/pzw/proj/paperagent/hx/Research_space/SemanticAlign-Bench/experiments/runs/gemini-flash_basic/voting-leaderboards/voting-leaderboards-20260505-151946/workspace/repo/data/prompts.json', 'r') as f:
            prompts_data = json.load(f)
    except FileNotFoundError:
        print("Error: config.yaml or prompts.json not found. Please ensure they are in the correct directory.")
        exit()
    
    models = config["models"]

    simulator = AdversarialVotingSimulator(models, prompts_data, config)

    # Simulate promoting 'gemma-2-27b-it' by 1 position
    target_model_up = "gemma-2-27b-it"
    interactions_up, votes_up, history_up = simulator.simulate_attack(target_model_up, objective='up', x_positions=1, max_interactions=5000)
    print(f"
Simulation complete for {target_model_up} (up): Interactions = {interactions_up}, Votes = {votes_up}")
    
    # Reset ratings for a new simulation
    simulator = AdversarialVotingSimulator(models, prompts_data, config)

    # Simulate demoting 'claude-3-5-sonnet-20240620' by 1 position
    target_model_down = "claude-3-5-sonnet-20240620"
    interactions_down, votes_down, history_down = simulator.simulate_attack(target_model_down, objective='down', x_positions=1, max_interactions=5000)
    print(f"
Simulation complete for {target_model_down} (down): Interactions = {interactions_down}, Votes = {votes_down}")
