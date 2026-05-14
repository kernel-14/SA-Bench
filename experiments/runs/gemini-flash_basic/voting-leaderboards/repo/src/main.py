import json
from de_anonymization import IdentityProbingDetector, TrainingBasedDetector
from simulation import AdversarialVotingSimulator, EloRatingSystem

def main():
    # Load configuration and prompt data
    try:
        with open('/mnt/paper2any/pzw/proj/paperagent/hx/Research_space/SemanticAlign-Bench/experiments/runs/gemini-flash_basic/voting-leaderboards/voting-leaderboards-20260505-151946/workspace/repo/config.yaml', 'r') as f:
            config = json.load(f)
        with open('/mnt/paper2any/pzw/proj/paperagent/hx/Research_space/SemanticAlign-Bench/experiments/runs/gemini-flash_basic/voting-leaderboards/voting-leaderboards-20260505-151946/workspace/repo/data/prompts.json', 'r') as f:
            prompts_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading configuration files: {e}. Make sure config.yaml and prompts.json are in their respective directories.")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}. Check syntax of config.yaml or prompts.json.")
        return

    models = config["models"]
    identity_probing_prompts = config["identity_probing_prompts"]
    simulation_params = config["simulation_params"]

    print("
--- Running De-anonymization Detector Simulations ---")
    
    # --- Identity-Probing Detector Simulation ---
    print("
*** Identity-Probing Detector ***")
    id_detector = IdentityProbingDetector(models, identity_probing_prompts, simulation_params)
    id_accuracies = {}
    for model in models:
        accuracy = id_detector.evaluate_detector(model, num_queries=100) # Reduced queries for demonstration
        id_accuracies[model] = accuracy
        print(f"Identity-Probing Detector Accuracy for {model}: {accuracy:.2f}%")
    print(f"Average Identity-Probing Detector Accuracy: {sum(id_accuracies.values()) / len(id_accuracies):.2f}%")

    # --- Training-Based Detector Simulation ---
    print("
*** Training-Based Detector ***")
    tb_detector = TrainingBasedDetector(models, prompts_data, simulation_params)
    tb_accuracies = {}
    for model in models:
        accuracy = tb_detector.evaluate_detector(model, num_prompts_per_category=5, responses_per_model=10) # Reduced for demonstration
        tb_accuracies[model] = accuracy
        print(f"Training-Based Detector Accuracy for {model}: {accuracy:.2f}%")
    print(f"Average Training-Based Detector Accuracy: {sum(tb_accuracies.values()) / len(tb_accuracies):.2f}%")

    print("
--- Running Adversarial Voting Simulations ---")

    # Initialize Elo system with models
    initial_elo_system = EloRatingSystem()
    for model in models:
        initial_elo_system.ratings[model] = 1500
    initial_rankings = initial_elo_system.get_rankings()
    print("
Initial Leaderboard Rankings:")
    for i, (model, rating) in enumerate(initial_rankings):
        print(f"{i+1}. {model}: {rating:.2f}")

    # Simulate promoting a target model
    target_model_up = "gemma-2-27b-it"
    # Create a fresh simulator for each attack to reset Elo scores
    simulator_up = AdversarialVotingSimulator(models, prompts_data, config)
    interactions_up, votes_up, history_up = simulator_up.simulate_attack(target_model_up, objective='up', x_positions=1, max_interactions=5000)
    print(f"
Simulation to promote {target_model_up} complete: Interactions = {interactions_up}, Votes = {votes_up}")
    print("Final Leaderboard (Promote):")
    for i, (model, rating) in enumerate(simulator_up.elo_system.get_rankings()):
        print(f"{i+1}. {model}: {rating:.2f}")

    # Simulate demoting a target model
    target_model_down = "claude-3-5-sonnet-20240620"
    # Create a fresh simulator for each attack to reset Elo scores
    simulator_down = AdversarialVotingSimulator(models, prompts_data, config)
    interactions_down, votes_down, history_down = simulator_down.simulate_attack(target_model_down, objective='down', x_positions=1, max_interactions=5000)
    print(f"
Simulation to demote {target_model_down} complete: Interactions = {interactions_down}, Votes = {votes_down}")
    print("Final Leaderboard (Demote):")
    for i, (model, rating) in enumerate(simulator_down.elo_system.get_rankings()):
        print(f"{i+1}. {model}: {rating:.2f}")

if __name__ == "__main__":
    main()
