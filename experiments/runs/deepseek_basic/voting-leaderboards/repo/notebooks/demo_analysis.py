"""
Demo script that runs a smaller version of the experiments
and produces key visualizations.

This serves as a self-contained demonstration of the paper's core ideas.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.deanonymization import (
    TrainingBasedDetector,
    IdentityProbingDetector,
    compute_pca_visualization,
    evaluate_prompt_for_detection,
    IDENTITY_PROBING_PROMPTS,
)
from src.simulation import (
    BradleyTerryModel,
    LeaderboardSimulation,
    AttackerConfig,
    estimate_votes_for_rank_change,
)
from src.mitigations import (
    AttackCost,
    MaliciousUserDetector,
    PerturbedLeaderboard,
    compute_vote_distribution_from_ratings,
    evaluate_malicious_detection_with_noise,
    estimate_detector_training_cost,
)
from src.models import (
    generate_synthetic_responses,
    generate_synthetic_voting_data,
    get_all_model_names,
    PROMPT_EXAMPLES,
    PCA_VISUALIZATION_PROMPTS,
)
from src.visualization import (
    plot_pca_visualization,
    plot_detection_accuracy_heatmap,
    plot_malicious_detection_likelihood,
    plot_detection_vs_noise,
    plot_detector_accuracy_comparison,
    plot_cost_analysis,
    plot_rank_trajectory,
)


def main():
    print("=" * 70)
    print("DEMO: Voting-Based Leaderboard Attack & Mitigation")
    print("=" * 70)
    
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    # Use a smaller set of models for quick demo
    model_names = [
        "claude-3-5-sonnet-20240620",
        "gemini-1.5-pro",
        "gpt-4o-mini-2024-07-18",
        "gemma-2-27b-it",
        "llama-3.1-70b-instruct",
        "mixtral-8x7b-instruct-v0.1",
        "qwen2-72b-instruct",
    ]
    
    print(f"\nUsing {len(model_names)} models for demo")
    
    # =========================================================================
    # PART 1: De-anonymization Demo
    # =========================================================================
    print("\n" + "-" * 50)
    print("PART 1: Training a De-anonymization Detector")
    print("-" * 50)
    
    prompt = PROMPT_EXAMPLES["english"][0]
    print(f"Prompt: {prompt}")
    
    # Generate synthetic responses
    n_responses = 50
    all_responses = generate_synthetic_responses(
        model_names, prompt, n_responses=n_responses, seed=42
    )
    
    # Train detector for each model as target
    print("\nDetection accuracy using different features:")
    print(f"{'Target Model':<30} {'Length_w':>8} {'Length_c':>8} {'BoW':>8} {'TF-IDF':>8}")
    print("-" * 70)
    
    for target_model in model_names:
        target_responses = all_responses[target_model]
        other_responses = []
        for m in model_names:
            if m != target_model:
                other_responses.extend(all_responses[m])
        
        row = [target_model[:28]]
        for ft in ["length_word", "length_char", "bow", "tfidf"]:
            detector = TrainingBasedDetector(
                target_model=target_model,
                feature_type=ft,
                random_state=42,
            )
            result = detector.train(target_responses, other_responses)
            row.append(f"{result['test_accuracy']*100:7.1f}")
        
        print("".join(f"{r:>10}" for r in row))
    
    # PCA Visualization
    print("\nGenerating PCA visualization...")
    projections, labels, model_names_pca = compute_pca_visualization(
        all_responses, n_components=2
    )
    
    plot_pca_visualization(
        projections, labels, model_names_pca,
        prompt_title="English prompt (demo)",
        save_path="figures/demo_pca.png",
    )
    print("-> Saved to figures/demo_pca.png")
    
    # =========================================================================
    # PART 2: Attack Simulation Demo
    # =========================================================================
    print("\n" + "-" * 50)
    print("PART 2: Attack Simulation")
    print("-" * 50)
    
    # Generate synthetic voting data
    initial_votes = generate_synthetic_voting_data(
        model_names, n_votes=5000, seed=42
    )
    
    sim = LeaderboardSimulation(model_names, initial_votes=initial_votes)
    sim.simulate_benign_votes(5000)
    
    # Show initial rankings
    ranking = sim.bt_model.get_ranking()
    print("\nInitial rankings:")
    for i, (name, score, votes) in enumerate(ranking):
        print(f"  {i+1}. {name}: {score:.1f} ({votes} votes)")
    
    # Simulate attack: push last model up
    target = model_names[-1]
    print(f"\nSimulating attack: push {target} up by 2 positions...")
    
    config = AttackerConfig(
        target_model=target,
        direction="up",
        detector_accuracy=0.95,
    )
    
    result = sim.run_attack(
        config,
        max_interactions=50000,
        target_position_change=2,
        verbose=False,
    )
    
    print(f"Result: {result['objective_achieved']}")
    print(f"  Interactions needed: {result['interactions_at_achievement']}")
    print(f"  Votes needed: {result['votes_at_achievement']}")
    print(f"  Start rank: {result['start_rank']} -> End rank: {result['end_rank']}")
    
    # Plot rank trajectory
    rank_history = sim.get_rank_history_array()
    plot_rank_trajectory(
        rank_history, model_names, target,
        title=f"Rank Trajectory During Attack (target: {target})",
        save_path="figures/demo_rank_trajectory.png",
    )
    print("-> Saved to figures/demo_rank_trajectory.png")
    
    # =========================================================================
    # PART 3: Mitigations Demo
    # =========================================================================
    print("\n" + "-" * 50)
    print("PART 3: Mitigations Analysis")
    print("-" * 50)
    
    # Cost model
    detector_cost = estimate_detector_training_cost(n_prompts=200)
    print(f"\nEstimated detector training cost: ${detector_cost:.2f}")
    
    # Cost with different defenses
    defense_configs = [
        ("No defense", float('inf'), 0.0, 0.0),
        ("Rate limiting (m=100)", 100, 0.0, 0.0),
        ("Rate limiting (m=10)", 10, 0.0, 0.0),
        ("+ Authentication", 100, 1.0, 0.0),
        ("+ CAPTCHA", 100, 1.0, 0.002),
        ("+ Prompt uniqueness", 100, 1.0, 20.0),
    ]
    
    print("\nAttack cost with different defenses (for N=3000 actions):")
    print(f"{'Defense':<30} {'Total Cost':>12}")
    print("-" * 45)
    
    cost_breakdowns = []
    for name, max_actions, cost_account, cost_action in defense_configs:
        ac = AttackCost(
            n_actions=3000,
            max_actions_per_account=max_actions,
            cost_per_account=cost_account,
            cost_per_action=cost_action,
            detector_cost=detector_cost,
        )
        bd = ac.breakdown()
        bd['defense'] = name
        cost_breakdowns.append(bd)
        print(f"  {name:<28} ${ac.total_cost():>10.2f}")
    
    # Plot cost analysis
    plot_cost_analysis(
        cost_breakdowns,
        [c['defense'] for c in cost_breakdowns],
        save_path="figures/demo_cost_analysis.png",
    )
    print("-> Saved to figures/demo_cost_analysis.png")
    
    # Malicious user detection - Scenario 1
    print("\nScenario 1: Malicious User Detection...")
    ratings = {name: 1000 + (len(model_names) - i) * 30 
               for i, name in enumerate(model_names)}
    ratings_array = np.array([ratings[m] for m in model_names])
    benign_dist = compute_vote_distribution_from_ratings(ratings_array)
    
    detector = MaliciousUserDetector(benign_dist)
    
    # Simulate benign and malicious users
    n_users = 100
    n_obs = 50
    
    benign_pvalues = []
    malicious_pvalues = []
    
    for _ in range(n_users):
        obs = np.random.choice(len(model_names), size=n_obs, p=benign_dist)
        benign_pvalues.append(detector.empirical_pvalue(obs, n_simulations=300))
        
        obs = np.ones(n_obs, dtype=int) * 0  # Always vote for top model
        malicious_pvalues.append(detector.empirical_pvalue(obs, n_simulations=300))
    
    alpha = 0.01
    benign_fpr = np.mean(np.array(benign_pvalues) < alpha)
    malicious_tpr = np.mean(np.array(malicious_pvalues) < alpha)
    
    print(f"  Benign false positive rate: {benign_fpr:.3f}")
    print(f"  Malicious detection rate: {malicious_tpr:.3f}")
    
    plot_malicious_detection_likelihood(
        benign_pvalues, malicious_pvalues, n_obs, alpha,
        save_path="figures/demo_scenario1.png",
    )
    print("-> Saved to figures/demo_scenario1.png")
    
    # Scenario 2: Perturbed leaderboard
    print("\nScenario 2: Perturbed Leaderboard...")
    noise_scales = [0, 20, 40, 60, 80, 100, 150, 200]
    noise_results = evaluate_malicious_detection_with_noise(
        true_ratings=ratings,
        attacker_target=model_names[0],
        noise_scales=noise_scales,
        n_observations=50,
        n_trials=30,
        alpha=0.01,
    )
    
    plot_detection_vs_noise(
        noise_results["noise_scale"],
        noise_results["detection_rate"],
        noise_results["utility_loss"],
        save_path="figures/demo_scenario2.png",
    )
    print("-> Saved to figures/demo_scenario2.png")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print(f"""
Key findings demonstrated:
1. BoW features achieve high detection accuracy (>90%) for model de-anonymization
2. ~{result['votes_at_achievement']} adversarial votes shifted rankings by 2 positions
3. Without defenses, attack costs ~${detector_cost:.0f} (detector training only)
4. With full defenses, attack costs increase to ${cost_breakdowns[-1]['total_cost']:.0f}
5. Malicious user detection: {malicious_tpr*100:.0f}% detection rate at {alpha} significance level
6. Perturbed leaderboards improve detection at the cost of ranking accuracy

See figures/ directory for visualizations.
""")


if __name__ == "__main__":
    main()
