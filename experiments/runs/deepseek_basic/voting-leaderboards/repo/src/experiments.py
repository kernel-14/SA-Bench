"""
Experiment orchestration module.

Coordinates running the three main experiments described in the paper:
1. De-anonymization experiments (Section 2)
2. Adversarial vote estimation (Section 3)
3. Mitigation evaluation (Section 4)
"""

import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

from .deanonymization import (
    IdentityProbingDetector,
    TrainingBasedDetector,
    TextFeatureExtractor,
    compute_pca_visualization,
    IDENTITY_PROBING_PROMPTS,
    PROMPT_CATEGORIES,
)
from .simulation import (
    BradleyTerryModel,
    LeaderboardSimulation,
    AttackerConfig,
    estimate_votes_for_rank_change,
    generate_rank_table,
)
from .mitigations import (
    AttackCost,
    MaliciousUserDetector,
    NeymanPearsonDetector,
    PerturbedLeaderboard,
    compute_vote_distribution_from_ratings,
    evaluate_malicious_detection_with_noise,
    estimate_detector_training_cost,
)
from .models import (
    get_all_model_names,
    get_high_ranked_models,
    get_low_ranked_models,
    generate_synthetic_responses,
    generate_synthetic_voting_data,
    PROMPT_EXAMPLES,
    PCA_VISUALIZATION_PROMPTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Experiment 1: De-anonymization (Section 2)
# =============================================================================

def run_deanonymization_experiment(
    model_names: List[str],
    prompts_by_category: Dict[str, List[str]],
    n_responses_per_model: int = 50,
    output_dir: str = "results/deanonymization",
    use_synthetic: bool = True,
) -> Dict:
    """
    Run the de-anonymization experiments described in Section 2.
    
    Includes:
    1. Identity-probing detector evaluation (Section 2.4.1, Table 2)
    2. Training-based detector evaluation (Section 2.4.2, Table 3, Figure 3)
    3. PCA visualization (Figure 2)
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # =========================================================================
    # 1. Identity-Probing Detector
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Running Identity-Probing Detector experiments...")
    
    identity_results = {}
    
    for prompt in IDENTITY_PROBING_PROMPTS:
        prompt_results = {}
        
        for target_model in model_names:
            if use_synthetic:
                # Generate synthetic responses
                all_responses = generate_synthetic_responses(
                    model_names, prompt, n_responses=n_responses_per_model,
                    seed=hash(target_model) % 10000
                )
                target_responses = all_responses[target_model]
                other_responses = []
                for m in model_names:
                    if m != target_model:
                        other_responses.extend(all_responses[m][:5])  # Sample from others
            else:
                # TODO: Use actual API queries
                raise NotImplementedError(
                    "Real API queries not implemented. Set use_synthetic=True for testing."
                )
            
            detector = IdentityProbingDetector(target_model)
            accuracy = detector.evaluate(target_responses, other_responses)
            prompt_results[target_model] = accuracy
        
        identity_results[prompt] = prompt_results
    
    results["identity_probing"] = identity_results
    
    # Save identity-probing results
    with open(os.path.join(output_dir, "identity_probing.json"), "w") as f:
        json.dump(identity_results, f, indent=2)
    
    logger.info("Identity-probing detector results saved.")
    
    # =========================================================================
    # 2. Training-Based Detector
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Running Training-Based Detector experiments...")
    
    training_results = {}
    feature_types = ["length_word", "length_char", "bow", "tfidf"]
    
    # Use English prompts as the default for Table 3 comparison
    english_prompts = prompts_by_category.get("english", [])
    if not english_prompts:
        english_prompts = [
            "How can identity protection services help protect me against identity theft?"
        ]
    
    # For each target model, evaluate all feature types
    for target_model in model_names[:7]:  # Use subset matching Table 3
        model_results = {}
        
        for ft in feature_types:
            ft_accuracies = []
            
            for prompt in english_prompts[:5]:  # Use up to 5 prompts
                if use_synthetic:
                    all_responses = generate_synthetic_responses(
                        model_names, prompt, n_responses=n_responses_per_model,
                        seed=42
                    )
                    target_responses = all_responses[target_model]
                    other_responses = []
                    for m in model_names:
                        if m != target_model:
                            other_responses.extend(all_responses[m])
                else:
                    raise NotImplementedError("Real API queries not implemented.")
                
                detector = TrainingBasedDetector(
                    target_model=target_model,
                    feature_type=ft,
                    random_state=42,
                )
                train_result = detector.train(target_responses, other_responses)
                ft_accuracies.append(train_result["test_accuracy"])
            
            model_results[ft] = np.mean(ft_accuracies) if ft_accuracies else 0.0
        
        training_results[target_model] = model_results
        logger.info(f"  {target_model}: BoW accuracy = {model_results.get('bow', 0):.3f}")
    
    results["training_based"] = training_results
    
    # Save training-based results
    with open(os.path.join(output_dir, "training_based.json"), "w") as f:
        json.dump(training_results, f, indent=2)
    
    logger.info("Training-based detector results saved.")
    
    # =========================================================================
    # 3. Cross-category evaluation (for Figure 3)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Running cross-category detection accuracy...")
    
    categories = list(prompts_by_category.keys())
    cross_category_results = {}
    
    for target_model in model_names[:7]:
        category_accuracies = {}
        
        for category in categories:
            prompts = prompts_by_category[category][:3]  # Up to 3 prompts per category
            cat_accuracies = []
            
            for prompt in prompts:
                if use_synthetic:
                    all_responses = generate_synthetic_responses(
                        model_names, prompt, n_responses=n_responses_per_model,
                        seed=hash(category) % 10000
                    )
                    target_responses = all_responses[target_model]
                    other_responses = []
                    for m in model_names:
                        if m != target_model:
                            other_responses.extend(all_responses[m])
                else:
                    raise NotImplementedError("Real API queries not implemented.")
                
                detector = TrainingBasedDetector(
                    target_model=target_model,
                    feature_type="bow",
                    random_state=42,
                )
                train_result = detector.train(target_responses, other_responses)
                cat_accuracies.append(train_result["test_accuracy"])
            
            category_accuracies[category] = np.mean(cat_accuracies) if cat_accuracies else 0.0
        
        cross_category_results[target_model] = category_accuracies
        logger.info(f"  {target_model}: { {k: f'{v:.3f}' for k, v in category_accuracies.items()} }")
    
    results["cross_category"] = cross_category_results
    
    with open(os.path.join(output_dir, "cross_category.json"), "w") as f:
        json.dump(cross_category_results, f, indent=2)
    
    # =========================================================================
    # 4. PCA Visualization data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Computing PCA visualization data...")
    
    pca_results = {}
    for i, prompt in enumerate(PCA_VISUALIZATION_PROMPTS):
        all_responses = generate_synthetic_responses(
            model_names[:7], prompt, n_responses=n_responses_per_model,
            seed=i * 100
        )
        projections, labels, names = compute_pca_visualization(all_responses)
        pca_results[f"prompt_{i+1}"] = {
            "projections": projections.tolist(),
            "labels": labels.tolist(),
            "model_names": names,
            "prompt": prompt[:100],
        }
    
    results["pca_visualization"] = pca_results
    
    with open(os.path.join(output_dir, "pca_visualization.json"), "w") as f:
        # Save without large arrays for readability, or use numpy
        json.dump({
            k: {kk: vv for kk, vv in v.items() if kk != 'projections'}
            for k, v in pca_results.items()
        }, f, indent=2)
    
    return results


# =============================================================================
# Experiment 2: Adversarial Vote Estimation (Section 3)
# =============================================================================

def run_vote_estimation_experiment(
    model_names: List[str],
    output_dir: str = "results/simulation",
    use_synthetic: bool = True,
) -> Dict:
    """
    Run the vote estimation simulation described in Section 3.
    
    Reproduces Tables 4 and 5.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    logger.info("=" * 60)
    logger.info("Running Adversarial Vote Estimation...")
    
    # Initialize Bradley-Terry model with synthetic historical data
    if use_synthetic:
        # Generate synthetic historical votes
        initial_votes = generate_synthetic_voting_data(
            model_names, n_votes=50000, seed=42
        )
    else:
        initial_votes = None
    
    # Create simulation
    sim = LeaderboardSimulation(model_names, initial_votes=initial_votes)
    
    # Simulate additional benign votes to stabilize ratings
    sim.simulate_benign_votes(20000)
    
    # Record initial rankings
    initial_ranking = sim.bt_model.get_ranking()
    logger.info("Initial rankings (top 10):")
    for i, (name, score, votes) in enumerate(initial_ranking[:10]):
        logger.info(f"  {i+1}. {name}: score={score:.1f}, votes={votes}")
    
    # =========================================================================
    # High-ranked models (Table 4)
    # =========================================================================
    logger.info("-" * 40)
    logger.info("Estimating votes for HIGH-ranked models...")
    
    high_ranked_results = {}
    for target_model in model_names[:5]:  # Top 5 models
        start_rank = sim.bt_model.get_rank(target_model)
        start_votes = sim.bt_model.total_votes[target_model]
        
        model_results = {
            "current_rank": start_rank,
            "current_votes": start_votes,
        }
        
        # Up-vote: move up by 1, 2, 3, 4 positions
        for pos_change in [1, 2, 3, 4]:
            if start_rank - pos_change >= 1:
                est = estimate_votes_for_rank_change(
                    sim, target_model, pos_change, "up",
                    detector_accuracy=0.95, max_interactions=300000, n_trials=1
                )
                model_results[f"up_{pos_change}"] = int(est["avg_votes_required"])
                model_results[f"up_{pos_change}_interactions"] = int(est["avg_interactions_required"])
            else:
                model_results[f"up_{pos_change}"] = "N/A"
                model_results[f"up_{pos_change}_interactions"] = "N/A"
        
        # Down-vote: move down by 1, 2, 3, 4 positions
        for pos_change in [1, 2, 3, 4]:
            if start_rank + pos_change <= len(model_names):
                est = estimate_votes_for_rank_change(
                    sim, target_model, pos_change, "down",
                    detector_accuracy=0.95, max_interactions=300000, n_trials=1
                )
                model_results[f"down_{pos_change}"] = int(est["avg_votes_required"])
                model_results[f"down_{pos_change}_interactions"] = int(est["avg_interactions_required"])
            else:
                model_results[f"down_{pos_change}"] = "N/A"
                model_results[f"down_{pos_change}_interactions"] = "N/A"
        
        high_ranked_results[target_model] = model_results
        logger.info(f"  {target_model} (rank {start_rank}): {model_results}")
    
    results["high_ranked"] = high_ranked_results
    
    # =========================================================================
    # Low-ranked models (Table 5)
    # =========================================================================
    logger.info("-" * 40)
    logger.info("Estimating votes for LOW-ranked models...")
    
    low_ranked_results = {}
    for target_model in model_names[-5:]:  # Bottom 5 models
        start_rank = sim.bt_model.get_rank(target_model)
        start_votes = sim.bt_model.total_votes[target_model]
        
        model_results = {
            "current_rank": start_rank,
            "current_votes": start_votes,
        }
        
        for pos_change in [1, 2, 3, 4, 5]:
            if start_rank - pos_change >= 1:
                est = estimate_votes_for_rank_change(
                    sim, target_model, pos_change, "up",
                    detector_accuracy=0.95, max_interactions=300000, n_trials=1
                )
                model_results[f"up_{pos_change}"] = int(est["avg_votes_required"])
                model_results[f"up_{pos_change}_interactions"] = int(est["avg_interactions_required"])
        
        low_ranked_results[target_model] = model_results
        logger.info(f"  {target_model} (rank {start_rank}): {model_results}")
    
    results["low_ranked"] = low_ranked_results
    
    # =========================================================================
    # Ablation: Varying detector accuracy (Table 8)
    # =========================================================================
    logger.info("-" * 40)
    logger.info("Running ablation: varying detector accuracy...")
    
    ablation_results = {}
    target_model = model_names[-1]  # Use last model for ablation
    
    for acc in [1.0, 0.95, 0.9]:
        acc_results = {}
        for pos_change in [1, 2, 5, 10, 20, 50]:
            if sim.bt_model.get_rank(target_model) - pos_change >= 1:
                est = estimate_votes_for_rank_change(
                    sim, target_model, pos_change, "up",
                    detector_accuracy=acc, max_interactions=300000, n_trials=1
                )
                acc_results[f"up_{pos_change}"] = int(est["avg_votes_required"])
                acc_results[f"up_{pos_change}_interactions"] = int(est["avg_interactions_required"])
        
        ablation_results[f"accuracy_{acc}"] = acc_results
        logger.info(f"  Accuracy {acc}: {acc_results}")
    
    results["ablation_accuracy"] = ablation_results
    
    # Save results
    with open(os.path.join(output_dir, "vote_estimation.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# =============================================================================
# Experiment 3: Mitigations (Section 4)
# =============================================================================

def run_mitigations_experiment(
    model_names: List[str],
    output_dir: str = "results/mitigations",
) -> Dict:
    """
    Run the mitigation analysis experiments described in Section 4.
    
    Includes:
    1. Cost model evaluation (Section 4.1)
    2. Malicious user detection - Scenario 1 (Section 4.2.3, Figure 4)
    3. Perturbed leaderboard - Scenario 2 (Section 4.2.3, Figures 5, 6)
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # =========================================================================
    # 1. Cost Model (Section 4.1)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Running Cost Model Analysis...")
    
    # Estimate detector training cost
    detector_cost = estimate_detector_training_cost()
    logger.info(f"Estimated detector training cost: ${detector_cost:.2f}")
    
    # Cost scenarios with different defenses
    cost_scenarios = []
    defense_configs = [
        ("No defense", float('inf'), 0.0, 0.0),
        ("Rate limiting (m=100)", 100, 0.0, 0.0),
        ("Rate limiting (m=10)", 10, 0.0, 0.0),
        ("Authentication", float('inf'), 1.0, 0.0),
        ("CAPTCHA", float('inf'), 0.0, 0.002),
        ("Prompt uniqueness", float('inf'), 0.0, 20.0),
        ("Full defense", 10, 1.0, 0.002),
    ]
    
    n_actions = 3000  # Typical votes needed
    
    cost_breakdowns = []
    for name, max_actions, cost_account, cost_action in defense_configs:
        ac = AttackCost(
            n_actions=n_actions,
            max_actions_per_account=max_actions,
            cost_per_account=cost_account,
            cost_per_action=cost_action,
            detector_cost=detector_cost,
        )
        breakdown = ac.breakdown()
        breakdown["defense"] = name
        cost_breakdowns.append(breakdown)
        
        logger.info(f"  {name}: ${breakdown['total_cost']:.2f}")
    
    results["cost_model"] = {
        "detector_cost": detector_cost,
        "scenarios": cost_breakdowns,
    }
    
    with open(os.path.join(output_dir, "cost_model.json"), "w") as f:
        json.dump(results["cost_model"], f, indent=2)
    
    # =========================================================================
    # 2. Malicious User Detection - Scenario 1 (Figure 4)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Running Malicious User Detection - Scenario 1...")
    
    # Create model ratings (simulate Chatbot Arena ratings)
    rng = np.random.RandomState(42)
    ratings = {
        name: 1000 + rng.normal(0, 50) * (len(model_names) - i)
        for i, name in enumerate(model_names)
    }
    
    # Compute benign vote distribution from ratings
    ratings_array = np.array([ratings[m] for m in model_names])
    benign_dist = compute_vote_distribution_from_ratings(ratings_array)
    
    # Create detector
    detector = MaliciousUserDetector(benign_dist)
    
    # Simulate benign users
    n_users = 200
    n_observations = 100
    benign_pvalues = []
    
    for _ in range(n_users):
        # Benign user: votes according to true distribution
        obs = np.random.choice(len(model_names), size=n_observations, p=benign_dist)
        pval = detector.empirical_pvalue(obs, n_simulations=500)
        benign_pvalues.append(pval)
    
    # Simulate malicious users (naive adversary: always votes for target)
    malicious_pvalues = []
    target_idx = 0  # Target the top-ranked model
    
    for _ in range(n_users):
        # Malicious user: always votes for target model
        obs = np.ones(n_observations, dtype=int) * target_idx
        pval = detector.empirical_pvalue(obs, n_simulations=500)
        malicious_pvalues.append(pval)
    
    # Calculate detection rates
    alpha = 0.01
    benign_detected = np.mean(np.array(benign_pvalues) < alpha)
    malicious_detected = np.mean(np.array(malicious_pvalues) < alpha)
    
    logger.info(f"  Benign false positive rate: {benign_detected:.3f}")
    logger.info(f"  Malicious detection rate: {malicious_detected:.3f}")
    
    results["scenario1_detection"] = {
        "benign_pvalues": benign_pvalues,
        "malicious_pvalues": malicious_pvalues,
        "alpha": alpha,
        "benign_fpr": benign_detected,
        "malicious_tpr": malicious_detected,
    }
    
    # Smart adversary: uses publicly available rankings
    smart_malicious_pvalues = []
    for _ in range(n_users):
        # Smart adversary mimics benign distribution for non-target models
        obs = []
        for _ in range(n_observations):
            if np.random.random() < 0.3:  # 30% of the time, target is involved
                obs.append(target_idx)
            else:
                obs.append(np.random.choice(len(model_names), p=benign_dist))
        obs = np.array(obs)
        pval = detector.empirical_pvalue(obs, n_simulations=500)
        smart_malicious_pvalues.append(pval)
    
    smart_malicious_detected = np.mean(np.array(smart_malicious_pvalues) < alpha)
    logger.info(f"  Smart adversary detection rate: {smart_malicious_detected:.3f}")
    
    results["scenario1_smart_adversary"] = {
        "pvalues": smart_malicious_pvalues,
        "detection_rate": smart_malicious_detected,
    }
    
    # =========================================================================
    # 3. Perturbed Leaderboard - Scenario 2 (Figures 5, 6)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Running Perturbed Leaderboard - Scenario 2...")
    
    noise_scales = [0, 20, 40, 60, 80, 100, 150, 200]
    
    noise_results = evaluate_malicious_detection_with_noise(
        true_ratings=ratings,
        attacker_target=model_names[0],
        noise_scales=noise_scales,
        n_observations=100,
        n_trials=50,
        alpha=0.01,
    )
    
    results["scenario2_noise"] = noise_results
    
    with open(os.path.join(output_dir, "scenario2_noise.json"), "w") as f:
        json.dump(noise_results, f, indent=2)
    
    return results


# =============================================================================
# Main experiment runner
# =============================================================================

def run_all_experiments(
    output_dir: str = "results",
    n_models: int = 22,
    use_synthetic: bool = True,
) -> Dict:
    """
    Run all experiments from the paper.
    
    Args:
        output_dir: Directory to save results
        n_models: Number of models to include
        use_synthetic: Whether to use synthetic data (for testing without API)
        
    Returns:
        Combined results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model names
    all_models = get_all_model_names()
    model_names = all_models[:n_models]
    
    logger.info(f"Running experiments with {len(model_names)} models")
    
    all_results = {}
    
    # Experiment 1: De-anonymization
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: DE-ANONYMIZATION (Section 2)")
    logger.info("=" * 70)
    
    exp1_results = run_deanonymization_experiment(
        model_names=model_names,
        prompts_by_category=PROMPT_EXAMPLES,
        output_dir=os.path.join(output_dir, "deanonymization"),
        use_synthetic=use_synthetic,
    )
    all_results["deanonymization"] = exp1_results
    
    # Experiment 2: Vote Estimation
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: VOTE ESTIMATION (Section 3)")
    logger.info("=" * 70)
    
    exp2_results = run_vote_estimation_experiment(
        model_names=model_names,
        output_dir=os.path.join(output_dir, "simulation"),
        use_synthetic=use_synthetic,
    )
    all_results["vote_estimation"] = exp2_results
    
    # Experiment 3: Mitigations
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: MITIGATIONS (Section 4)")
    logger.info("=" * 70)
    
    exp3_results = run_mitigations_experiment(
        model_names=model_names,
        output_dir=os.path.join(output_dir, "mitigations"),
    )
    all_results["mitigations"] = exp3_results
    
    # Save combined results
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        # Filter out large arrays for JSON serialization
        json_results = {}
        for section, data in all_results.items():
            if section == "mitigations":
                json_results[section] = {
                    k: v for k, v in data.items() 
                    if k not in ("scenario1_detection", "scenario1_smart_adversary")
                }
            else:
                json_results[section] = data
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\nAll results saved to {output_dir}/")
    
    return all_results
