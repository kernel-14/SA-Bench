"""Run mitigation experiments (Section 4).

Reproduces:
  - Cost model analysis (Section 4.1)
  - Malicious user detection: Scenario 1 likelihood test (Figure 4)
  - Malicious user detection: Scenario 2 with perturbed leaderboard (Figure 5)
  - Utility vs. noise scale trade-off (Figure 6)
  - Defense cost evaluation

Usage:
    python run_mitigations.py --config config.yaml
"""
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

from config import load_config, Config
from mitigations import (
    compute_attack_cost,
    estimate_cost_without_mitigations,
    AuthenticationDefense,
    RateLimitingDefense,
    MaliciousUserDetector,
    CAPTCHADefense,
    PromptUniquenessDefense,
    evaluate_defenses,
)
from data import VotingDataSimulator


def run_cost_model_analysis(config: Config, output_dir: Path):
    """Run cost model analysis (Section 4.1).

    Computes attack cost for different numbers of actions needed,
    showing the alarmingly low cost without mitigations.
    """
    print("\n" + "=" * 60)
    print("Section 4.1: Cost Model Analysis")
    print("=" * 60)

    # Without mitigations: single account, minimal action cost
    print("\nAttack cost WITHOUT mitigations:")
    for N in [100, 500, 1000, 2000, 5000, 10000]:
        cost = estimate_cost_without_mitigations(N)
        print(f"  N={N:>6}: ${cost['total_cost']:.2f} total "
              f"(detector: ${cost['detector_cost']:.2f}, "
              f"account: ${cost['account_cost']:.2f}, "
              f"action: ${cost['action_cost']:.2f})")

    # With mitigations
    print("\nAttack cost WITH mitigations:")
    for auth_enabled in [False, True]:
        for rate_limit in [None, 100]:
            for captcha in [False, True]:
                name = (
                    f"auth={'Y' if auth_enabled else 'N'}, "
                    f"rate_limit={rate_limit or 'None'}, "
                    f"captcha={'Y' if captcha else 'N'}"
                )
                cost = evaluate_defenses(
                    N=5000,
                    c_detector=config.mitigations.cost_model.detector_cost,
                    c_action_base=config.mitigations.cost_model.action_cost,
                    c_account=config.mitigations.authentication.account_cost if auth_enabled else 0.0,
                    m=rate_limit,
                    captcha_enabled=captcha,
                    c_captcha=config.mitigations.captcha.captcha_cost,
                )
                print(f"  {name}: ${cost['total_cost']:.2f}")


def run_malicious_detection_scenario1(config: Config, output_dir: Path):
    """Run Scenario 1: Known benign distribution (Figure 4).

    Tests the likelihood-based detection:
    - Naive adversary (random untargeted): effectively detected
    - Smart adversary (uses public ranking): bypasses detection
    """
    print("\n" + "=" * 60)
    print("Section 4.2.3 - Scenario 1: Known Benign Distribution")
    print("=" * 60)

    n_models = len(config.models)
    if n_models < 5:
        print(f"Need at least 5 models, got {n_models}")
        return

    detector = MaliciousUserDetector(
        alpha=config.mitigations.malicious_detection.scenario1.significance_level,
        num_simulations=config.mitigations.malicious_detection.scenario1.num_simulations,
        bradley_terry_scale=config.simulation.bradley_terry_scale,
    )

    # Generate synthetic ratings
    rng = np.random.RandomState(42)
    ratings = rng.normal(1000, 200, n_models)
    ratings = np.sort(ratings)[::-1]

    benign_probs = detector.compute_vote_probabilities(ratings)

    # Test with synthetic observations
    print("\nBenign user detection test:")
    num_trials = 100
    false_positives = 0
    for _ in range(num_trials):
        benign_obs = list(rng.choice(n_models, size=30, p=benign_probs))
        is_malicious, p_val = detector.scenario1_detect(benign_obs, benign_probs)
        if is_malicious:
            false_positives += 1
    print(f"  False positive rate: {false_positives / num_trials:.3f} "
          f"(expected ~{config.mitigations.malicious_detection.scenario1.significance_level:.3f})")

    # Naive adversary: random uniform
    print("\nNaive adversary (random uniform):")
    detection_count = 0
    for _ in range(num_trials):
        adv_obs = list(rng.choice(n_models, size=30))
        is_malicious, p_val = detector.scenario1_detect(adv_obs, benign_probs)
        if is_malicious:
            detection_count += 1
    print(f"  Detection rate: {detection_count / num_trials:.3f}")

    # Smart adversary: uses benign probs + always upvotes target
    print("\nSmart adversary (uses public ranking):")
    target_model = 0
    smart_probs = benign_probs.copy()
    smart_probs[target_model] *= 1.5
    smart_probs /= smart_probs.sum()
    detection_count = 0
    for _ in range(num_trials):
        adv_obs = list(rng.choice(n_models, size=30, p=smart_probs))
        is_malicious, p_val = detector.scenario1_detect(adv_obs, benign_probs)
        if is_malicious:
            detection_count += 1
    print(f"  Detection rate: {detection_count / num_trials:.3f}")

    results = {
        "false_positive_rate": false_positives / num_trials,
        "naive_adversary_detection": detection_count / num_trials,
    }
    print("\nSummary: The naive adversary is easily detected (high detection rate),")
    print("but the smart adversary who mimics benign behavior can bypass detection,")
    print("motivating Scenario 2 with perturbed rankings.")


def run_malicious_detection_scenario2(config: Config, output_dir: Path):
    """Run Scenario 2: Perturbed leaderboard (Figures 5 and 6).

    Tests:
    - Detection rate vs noise scale (Figure 5):
      As noise increases, detection rate improves.
    - Utility vs noise scale (Figure 6):
      Larger noise significantly changes the rank list order.
    """
    print("\n" + "=" * 60)
    print("Section 4.2.3 - Scenario 2: Perturbed Leaderboard")
    print("=" * 60)

    n_models = len(config.models)
    if n_models < 5:
        print(f"Need at least 5 models, got {n_models}")
        return

    detector = MaliciousUserDetector(
        alpha=config.mitigations.malicious_detection.scenario1.significance_level,
        num_simulations=config.mitigations.malicious_detection.scenario1.num_simulations,
        bradley_terry_scale=config.simulation.bradley_terry_scale,
    )

    rng = np.random.RandomState(42)
    ratings = rng.normal(1000, 200, n_models)
    ratings = np.sort(ratings)[::-1]

    noise_scales = config.mitigations.malicious_detection.scenario2.noise_scales_to_test
    if not noise_scales:
        noise_scales = [0.0, 0.1, 0.5, 1.0, 2.0]

    # Detection rate analysis (Figure 5)
    print("\nDetection rate vs noise scale (Figure 5):")
    print(f"{'Noise Scale':>12} {'Detection Rate':>15}")
    print("-" * 30)

    detection_rates = {}
    for scale in noise_scales:
        detections = 0
        num_trials = 100
        for _ in range(num_trials):
            perturbed = ratings + rng.normal(0, scale * np.std(ratings), n_models)
            adv_probs = detector.compute_vote_probabilities(perturbed)
            target_model = 0
            adv_probs[target_model] *= 2.0
            adv_probs /= adv_probs.sum()
            observations = list(rng.choice(n_models, size=30, p=adv_probs))
            is_malicious, _ = detector.scenario2_detect(
                observations=observations,
                true_ratings=ratings,
                noise_scale=scale,
            )
            if is_malicious:
                detections += 1
        rate = detections / num_trials
        detection_rates[scale] = rate
        print(f"  {scale:>10.1f}  {rate:>15.3f}")

    # Utility analysis (Figure 6)
    print("\nUtility (avg abs rank change) vs noise scale (Figure 6):")
    print(f"{'Noise Scale':>12} {'Avg Abs Rank Change':>22}")
    print("-" * 38)

    utility_results = detector.evaluate_scenario2_utility(
        true_ratings=ratings,
        noise_scales=noise_scales,
    )
    for scale in noise_scales:
        print(f"  {scale:>10.1f}  {utility_results[scale]:>22.3f}")

    # Save results
    results = {
        "detection_rates": {str(k): v for k, v in detection_rates.items()},
        "utility": {str(k): v for k, v in utility_results.items()},
    }
    with open(output_dir / "scenario2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Scenario 2 results to {output_dir / 'scenario2_results.json'}")


def run_full_defense_evaluation(config: Config, output_dir: Path):
    """Evaluate all defense combinations.

    Shows how each defense increases the total attack cost.
    """
    print("\n" + "=" * 60)
    print("Full Defense Evaluation")
    print("=" * 60)

    # Estimate needed actions from simulation (or use defaults)
    N_low = 500    # ~1 position change for low-ranked models
    N_med = 2000   # ~2-3 position changes
    N_high = 5000  # ~5 position changes for high-ranked models

    print("\nCost breakdown for different attack scales:")

    for N, label in [(N_low, "small"), (N_med, "medium"), (N_high, "large")]:
        print(f"\n--- {label.upper()} attack (N={N}) ---")

        # No defenses
        cost_none = estimate_cost_without_mitigations(N)
        print(f"  No defenses:     ${cost_none['total_cost']:.2f}")

        # Auth only
        cost_auth = evaluate_defenses(
            N=N,
            c_detector=config.mitigations.cost_model.detector_cost,
            c_action_base=config.mitigations.cost_model.action_cost,
            c_account=config.mitigations.authentication.account_cost,
            m=None,
        )
        print(f"  Auth only:       ${cost_auth['total_cost']:.2f}")

        # Rate limiting
        cost_rate = evaluate_defenses(
            N=N,
            c_detector=config.mitigations.cost_model.detector_cost,
            c_action_base=config.mitigations.cost_model.action_cost,
            c_account=0.0,
            m=config.mitigations.rate_limiting.max_actions_per_account,
        )
        print(f"  Rate limit:      ${cost_rate['total_cost']:.2f}")

        # Auth + Rate limit
        cost_both = evaluate_defenses(
            N=N,
            c_detector=config.mitigations.cost_model.detector_cost,
            c_action_base=config.mitigations.cost_model.action_cost,
            c_account=config.mitigations.authentication.account_cost,
            m=config.mitigations.rate_limiting.max_actions_per_account,
        )
        print(f"  Auth + Rate:     ${cost_both['total_cost']:.2f}")

        # Full: Auth + Rate + CAPTCHA
        cost_full = evaluate_defenses(
            N=N,
            c_detector=config.mitigations.cost_model.detector_cost,
            c_action_base=config.mitigations.cost_model.action_cost,
            c_account=config.mitigations.authentication.account_cost,
            m=config.mitigations.rate_limiting.max_actions_per_account,
            captcha_enabled=True,
            c_captcha=config.mitigations.captcha.captcha_cost,
        )
        print(f"  Auth+Rate+CAPTCHA: ${cost_full['total_cost']:.2f}")

        # Full + prompt uniqueness
        cost_max = evaluate_defenses(
            N=N,
            c_detector=config.mitigations.cost_model.detector_cost,
            c_action_base=config.mitigations.cost_model.action_cost,
            c_account=config.mitigations.authentication.account_cost,
            m=config.mitigations.rate_limiting.max_actions_per_account,
            captcha_enabled=True,
            c_captcha=config.mitigations.captcha.captcha_cost,
            prompt_uniqueness_enabled=True,
            c_prompt=config.mitigations.prompt_uniqueness.cost_per_prompt,
        )
        print(f"  Max defenses:     ${cost_max['total_cost']:.2f}")

    print("\nNote: Effective mitigation should substantially increase the attack cost")
    print("compared to the ~$440 baseline without defenses.")


def main():
    parser = argparse.ArgumentParser(description="Run mitigation experiments")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs/mitigations", help="Output directory")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "cost", "scenario1", "scenario2", "defense_eval"],
                        help="Which experiment to run")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment in ("all", "cost"):
        run_cost_model_analysis(config, output_dir)

    if args.experiment in ("all", "scenario1"):
        run_malicious_detection_scenario1(config, output_dir)

    if args.experiment in ("all", "scenario2"):
        run_malicious_detection_scenario2(config, output_dir)

    if args.experiment in ("all", "defense_eval"):
        run_full_defense_evaluation(config, output_dir)

    print("\nAll mitigation experiments completed.")


if __name__ == "__main__":
    main()
