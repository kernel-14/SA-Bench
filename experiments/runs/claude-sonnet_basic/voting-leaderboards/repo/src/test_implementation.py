"""
Test script to verify the implementation of core components.

This script tests the key components without requiring API access:
1. Training-based detector with synthetic data
2. Bradley-Terry model
3. Adversarial simulation with synthetic leaderboard
4. Mitigations (malicious user detection)
"""

import sys
import os
import numpy as np
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_training_based_detector():
    """Test the training-based detector with synthetic data."""
    from training_based_detector import ModelDetector, build_balanced_dataset

    logger.info("Testing training-based detector...")

    # Create synthetic responses with distinct patterns
    np.random.seed(42)

    # Target model responses: tend to use "assistant" and "helpful"
    target_responses = [
        f"I am a helpful assistant. {' '.join(np.random.choice(['helpful', 'assistant', 'AI', 'language'], 20))} response {i}"
        for i in range(50)
    ]

    # Other model responses: tend to use "model" and "system"
    other_responses = [
        f"I am a language model. {' '.join(np.random.choice(['model', 'system', 'neural', 'network'], 20))} response {i}"
        for i in range(50)
    ]

    # Test BoW detector
    detector = ModelDetector(feature_type="bow", random_state=42)
    results = detector.fit(target_responses, other_responses)

    logger.info(f"BoW detector test accuracy: {results['test_accuracy']:.1f}%")
    assert results["test_accuracy"] > 50, "BoW detector should perform better than random"

    # Test prediction
    pred = detector.predict(target_responses[0])
    logger.info(f"Prediction for target response: {pred} (expected: 1)")

    # Test TF-IDF detector
    detector_tfidf = ModelDetector(feature_type="tfidf", random_state=42)
    results_tfidf = detector_tfidf.fit(target_responses, other_responses)
    logger.info(f"TF-IDF detector test accuracy: {results_tfidf['test_accuracy']:.1f}%")

    # Test length detector
    detector_len = ModelDetector(feature_type="length_word", random_state=42)
    results_len = detector_len.fit(target_responses, other_responses)
    logger.info(f"Length detector test accuracy: {results_len['test_accuracy']:.1f}%")

    logger.info("Training-based detector tests PASSED")
    return True


def test_identity_probing_detector():
    """Test the identity-probing detector."""
    from identity_probing_detector import (
        detect_model_from_response,
        evaluate_identity_probing_detector,
        run_identity_probing_experiment,
    )

    logger.info("Testing identity-probing detector...")

    # Test detection
    response_with_claude = "I am Claude, an AI assistant made by Anthropic."
    response_without_claude = "I am an AI assistant here to help you."

    assert detect_model_from_response(
        response_with_claude, "claude-3-5-sonnet-20240620"
    ), "Should detect Claude in response"

    assert not detect_model_from_response(
        response_without_claude, "claude-3-5-sonnet-20240620"
    ), "Should not detect Claude in response without keywords"

    # Test evaluation
    responses = [
        "I am Claude, an AI assistant made by Anthropic.",
        "I am Claude, created by Anthropic.",
        "I am an AI assistant.",  # This one won't be detected
    ]
    accuracy = evaluate_identity_probing_detector(
        responses, "claude-3-5-sonnet-20240620"
    )
    logger.info(f"Identity-probing accuracy: {accuracy * 100:.1f}%")
    assert accuracy == 2 / 3, f"Expected 66.7%, got {accuracy * 100:.1f}%"

    logger.info("Identity-probing detector tests PASSED")
    return True


def test_bradley_terry():
    """Test the Bradley-Terry model."""
    from bradley_terry import (
        compute_bradley_terry_coefficients,
        get_rankings,
        win_probability,
        compute_benign_vote_distribution,
    )

    logger.info("Testing Bradley-Terry model...")

    # Create a simple wins matrix
    # Model 0 beats model 1 most of the time
    wins = np.array([
        [0, 80, 60],  # Model 0 wins
        [20, 0, 40],  # Model 1 wins
        [40, 60, 0],  # Model 2 wins
    ], dtype=float)

    ratings = compute_bradley_terry_coefficients(wins)
    logger.info(f"Bradley-Terry ratings: {ratings}")

    # Model 0 should have the highest rating
    assert ratings[0] > ratings[1], "Model 0 should be rated higher than Model 1"
    assert ratings[2] > ratings[1], "Model 2 should be rated higher than Model 1"

    # Test rankings
    model_names = ["model_0", "model_1", "model_2"]
    rankings = get_rankings(ratings, model_names)
    logger.info(f"Rankings: {rankings}")
    assert rankings[0][1] == "model_0", "Model 0 should be ranked first"

    # Test win probability
    p = win_probability(ratings[0], ratings[1])
    logger.info(f"Win probability of model_0 over model_1: {p:.3f}")
    assert p > 0.5, "Model 0 should have >50% win probability over Model 1"

    # Test benign vote distribution
    benign_dist = compute_benign_vote_distribution(ratings)
    logger.info(f"Benign vote distribution: {benign_dist}")
    assert abs(benign_dist.sum() - 1.0) < 1e-6, "Distribution should sum to 1"

    logger.info("Bradley-Terry model tests PASSED")
    return True


def test_adversarial_simulation():
    """Test the adversarial simulation with a small synthetic leaderboard."""
    from adversarial_simulation import (
        create_synthetic_leaderboard,
        AdversarialSimulator,
        SimulationConfig,
    )

    logger.info("Testing adversarial simulation...")

    # Create a small synthetic leaderboard
    leaderboard = create_synthetic_leaderboard(n_models=10, random_seed=42)
    logger.info(f"Created leaderboard with {len(leaderboard.model_names)} models")
    logger.info("Top 3 models:")
    for rank, name, rating in leaderboard.rankings[:3]:
        logger.info(f"  #{rank}: {name} (rating: {rating:.4f})")

    # Run a small simulation
    config = SimulationConfig(
        detection_accuracy=0.95,
        recalc_interval=100,
        max_interactions=5000,
        random_seed=42,
    )

    simulator = AdversarialSimulator(leaderboard, config)

    # Try to move the last-ranked model up 1 position
    last_model = leaderboard.rankings[-1][1]
    last_rank = leaderboard.rankings[-1][0]
    target_rank = last_rank - 1

    logger.info(f"Attempting to move {last_model} from rank {last_rank} to rank {target_rank}")
    result = simulator.simulate_attack(last_model, target_rank)

    logger.info(f"Attack result: {result['achieved']}")
    if result["achieved"]:
        logger.info(
            f"Achieved rank {result['final_rank']} with "
            f"{result['adversarial_votes']} adversarial votes and "
            f"{result['total_interactions']} interactions"
        )

    logger.info("Adversarial simulation tests PASSED")
    return True


def test_mitigations():
    """Test the mitigations module."""
    from mitigations import (
        compute_attack_cost,
        estimate_detector_training_cost,
        MaliciousUserDetector,
        NeymanPearsonDetector,
    )
    from adversarial_simulation import create_synthetic_leaderboard
    from bradley_terry import compute_benign_vote_distribution

    logger.info("Testing mitigations...")

    # Test cost model
    cost = compute_attack_cost(
        n_actions=1000,
        max_actions_per_account=100,
        cost_per_account=1.0,
        cost_per_action=0.01,
        detector_cost=440.0,
    )
    logger.info(f"Attack cost: ${cost:.2f}")
    assert cost > 440.0, "Cost should be at least the detector cost"

    # Test detector training cost estimation
    detector_cost = estimate_detector_training_cost()
    logger.info(f"Estimated detector training cost: ${detector_cost:.2f}")
    assert abs(detector_cost - 440.0) < 10.0, f"Expected ~$440, got ${detector_cost:.2f}"

    # Test malicious user detector
    leaderboard = create_synthetic_leaderboard(n_models=10, random_seed=42)
    benign_dist = compute_benign_vote_distribution(leaderboard.ratings)

    detector = MaliciousUserDetector(
        benign_vote_distribution=benign_dist,
        model_names=leaderboard.model_names,
        significance_level=0.01,
        n_simulations=100,  # Small for testing
        random_seed=42,
    )

    # Generate a benign sequence
    rng = np.random.RandomState(42)
    benign_seq = list(rng.choice(10, size=20, p=benign_dist))
    is_malicious, p_value = detector.is_malicious(benign_seq)
    logger.info(f"Benign sequence: is_malicious={is_malicious}, p_value={p_value:.4f}")

    # Generate a clearly adversarial sequence (always votes for model 0)
    adversarial_seq = [0] * 20
    is_malicious_adv, p_value_adv = detector.is_malicious(adversarial_seq)
    logger.info(
        f"Adversarial sequence: is_malicious={is_malicious_adv}, "
        f"p_value={p_value_adv:.4f}"
    )

    # Test Neyman-Pearson detector
    np_detector = NeymanPearsonDetector(
        true_ratings=leaderboard.ratings,
        model_names=leaderboard.model_names,
        noise_scale=0.1,
        random_seed=42,
    )

    utility = np_detector.compute_utility_impact()
    logger.info(f"Utility impact of noise: {utility:.2f} average rank changes")

    logger.info("Mitigations tests PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    tests = [
        ("Training-based detector", test_training_based_detector),
        ("Identity-probing detector", test_identity_probing_detector),
        ("Bradley-Terry model", test_bradley_terry),
        ("Adversarial simulation", test_adversarial_simulation),
        ("Mitigations", test_mitigations),
    ]

    results = {}
    for test_name, test_fn in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*60}")
        try:
            success = test_fn()
            results[test_name] = "PASSED" if success else "FAILED"
        except Exception as e:
            logger.error(f"Test {test_name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = f"FAILED: {e}"

    logger.info(f"\n{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")
    for test_name, result in results.items():
        status = "✓" if result == "PASSED" else "✗"
        logger.info(f"{status} {test_name}: {result}")

    n_passed = sum(1 for r in results.values() if r == "PASSED")
    logger.info(f"\n{n_passed}/{len(tests)} tests passed")

    return n_passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
