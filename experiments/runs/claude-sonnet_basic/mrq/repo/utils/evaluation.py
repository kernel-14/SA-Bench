"""
Evaluation and score normalization utilities for MR.Q.

Implements the normalization schemes used in the paper:
- Gym: TD3-normalized score
- DMC: Raw reward (max 1000)
- Atari: Human-normalized score

Also provides aggregate metrics: mean, median, IQM (interquartile mean)
with bootstrap confidence intervals.
"""

import numpy as np
from typing import Dict, List, Optional


# ============================================================================
# Gym Locomotion Normalization
# ============================================================================

# TD3 scores and random scores from paper (Table in Appendix B.3)
GYM_RANDOM_SCORES = {
    "Ant-v4": -70.288,
    "HalfCheetah-v4": -289.415,
    "Hopper-v4": 18.791,
    "Humanoid-v4": 120.423,
    "Walker2d-v4": 2.791,
}

GYM_TD3_SCORES = {
    "Ant-v4": 3942,
    "HalfCheetah-v4": 10574,
    "Hopper-v4": 3226,
    "Humanoid-v4": 5165,
    "Walker2d-v4": 3946,
}


def td3_normalize(score, env_name):
    """
    Normalize score using TD3 performance as reference.
    
    TD3-Normalized(x) = (x - random_score) / (TD3_score - random_score)
    """
    random = GYM_RANDOM_SCORES[env_name]
    td3 = GYM_TD3_SCORES[env_name]
    return (score - random) / (td3 - random)


# ============================================================================
# Atari Human Normalization
# ============================================================================

# Human and random scores from paper (Table in Appendix B.3)
ATARI_RANDOM_SCORES = {
    "Alien": 227.8, "Amidar": 5.8, "Assault": 222.4, "Asterix": 210.0,
    "Asteroids": 719.1, "Atlantis": 12850.0, "BankHeist": 14.2,
    "BattleZone": 2360.0, "BeamRider": 363.9, "Berzerk": 123.7,
    "Bowling": 23.1, "Boxing": 0.1, "Breakout": 1.7, "Centipede": 2090.9,
    "ChopperCommand": 811.0, "CrazyClimber": 10780.5, "Defender": 2874.5,
    "DemonAttack": 152.1, "DoubleDunk": -18.6, "Enduro": 0.0,
    "FishingDerby": -91.7, "Freeway": 0.0, "Frostbite": 65.2,
    "Gopher": 257.6, "Gravitar": 173.0, "Hero": 1027.0, "IceHockey": -11.2,
    "Jamesbond": 29.0, "Kangaroo": 52.0, "Krull": 1598.0,
    "KungFuMaster": 258.5, "MontezumaRevenge": 0.0, "MsPacman": 307.3,
    "NameThisGame": 2292.3, "Phoenix": 761.4, "Pitfall": -229.4,
    "Pong": -20.7, "PrivateEye": 24.9, "Qbert": 163.9, "Riverraid": 1338.5,
    "RoadRunner": 11.5, "Robotank": 2.2, "Seaquest": 68.4,
    "Skiing": -17098.1, "Solaris": 1236.3, "SpaceInvaders": 148.0,
    "StarGunner": 664.0, "Surround": -10.0, "Tennis": -23.8,
    "TimePilot": 3568.0, "Tutankham": 11.4, "UpNDown": 533.4,
    "Venture": 0.0, "VideoPinball": 16256.9, "WizardOfWor": 563.5,
    "YarsRevenge": 3092.9, "Zaxxon": 32.5,
}

ATARI_HUMAN_SCORES = {
    "Alien": 7127.7, "Amidar": 1719.5, "Assault": 742.0, "Asterix": 8503.3,
    "Asteroids": 47388.7, "Atlantis": 29028.1, "BankHeist": 753.1,
    "BattleZone": 37187.5, "BeamRider": 16926.5, "Berzerk": 2630.4,
    "Bowling": 160.7, "Boxing": 12.1, "Breakout": 30.5, "Centipede": 12017.0,
    "ChopperCommand": 7387.8, "CrazyClimber": 35829.4, "Defender": 18688.9,
    "DemonAttack": 1971.0, "DoubleDunk": -16.4, "Enduro": 860.5,
    "FishingDerby": -38.7, "Freeway": 29.6, "Frostbite": 4334.7,
    "Gopher": 2412.5, "Gravitar": 3351.4, "Hero": 30826.4, "IceHockey": 0.9,
    "Jamesbond": 302.8, "Kangaroo": 3035.0, "Krull": 2665.5,
    "KungFuMaster": 22736.3, "MontezumaRevenge": 4753.3, "MsPacman": 6951.6,
    "NameThisGame": 8049.0, "Phoenix": 7242.6, "Pitfall": 6463.7,
    "Pong": 14.6, "PrivateEye": 69571.3, "Qbert": 13455.0,
    "Riverraid": 17118.0, "RoadRunner": 7845.0, "Robotank": 11.9,
    "Seaquest": 42054.7, "Skiing": -4336.9, "Solaris": 12326.7,
    "SpaceInvaders": 1668.7, "StarGunner": 10250.0, "Surround": 6.5,
    "Tennis": -8.3, "TimePilot": 5229.2, "Tutankham": 167.6,
    "UpNDown": 11693.2, "Venture": 1187.5, "VideoPinball": 17667.9,
    "WizardOfWor": 4756.5, "YarsRevenge": 54576.9, "Zaxxon": 9173.3,
}


def human_normalize(score, game_name):
    """
    Normalize score using human performance as reference.
    
    Human-Normalized(x) = (x - random_score) / (human_score - random_score)
    """
    random = ATARI_RANDOM_SCORES[game_name]
    human = ATARI_HUMAN_SCORES[game_name]
    return (score - random) / (human - random)


# ============================================================================
# Aggregate metrics
# ============================================================================

def interquartile_mean(scores):
    """
    Compute Interquartile Mean (IQM) of scores.
    IQM = mean of scores in [25th, 75th] percentile range.
    """
    scores = np.array(scores)
    q25, q75 = np.percentile(scores, [25, 75])
    mask = (scores >= q25) & (scores <= q75)
    if mask.sum() == 0:
        return np.mean(scores)
    return np.mean(scores[mask])


def bootstrap_ci(scores, n_bootstrap=10000, ci=0.95, metric_fn=np.mean):
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        scores: Array of scores
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level
        metric_fn: Metric function (e.g., np.mean, np.median, interquartile_mean)
    
    Returns:
        (lower, upper): Confidence interval bounds
    """
    scores = np.array(scores)
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_metrics.append(metric_fn(sample))
    
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_metrics, alpha * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha) * 100)
    return lower, upper


def stratified_bootstrap_ci(scores_per_env, n_bootstrap=10000, ci=0.95, metric_fn=np.mean):
    """
    Stratified bootstrap CI: sample from each environment independently.
    
    Args:
        scores_per_env: Dict mapping env_name -> list of scores (one per seed)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level
        metric_fn: Metric function applied to normalized scores
    
    Returns:
        (mean_metric, lower_ci, upper_ci)
    """
    env_names = list(scores_per_env.keys())
    all_scores = [np.array(scores_per_env[env]) for env in env_names]
    
    # Compute point estimate
    all_flat = np.concatenate(all_scores)
    point_estimate = metric_fn(all_flat)
    
    # Bootstrap
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        # Sample one score per environment
        sampled = []
        for env_scores in all_scores:
            sampled.append(np.random.choice(env_scores))
        bootstrap_metrics.append(metric_fn(np.array(sampled)))
    
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_metrics, alpha * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha) * 100)
    
    return point_estimate, lower, upper


def compute_aggregate_stats(scores_per_env, normalize_fn=None):
    """
    Compute mean, median, and IQM with 95% stratified bootstrap CIs.
    
    Args:
        scores_per_env: Dict mapping env_name -> list of scores
        normalize_fn: Optional function to normalize scores (env_name, score) -> normalized
    
    Returns:
        Dict with mean, median, iqm and their CIs
    """
    if normalize_fn is not None:
        normalized = {
            env: [normalize_fn(env, s) for s in scores]
            for env, scores in scores_per_env.items()
        }
    else:
        normalized = scores_per_env
    
    results = {}
    for metric_name, metric_fn in [
        ("mean", np.mean),
        ("median", np.median),
        ("iqm", interquartile_mean),
    ]:
        point, lower, upper = stratified_bootstrap_ci(normalized, metric_fn=metric_fn)
        results[metric_name] = {
            "value": point,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    
    return results


def print_results_table(scores_per_env, benchmark="gym"):
    """Print results table similar to paper format."""
    print(f"\n{'='*60}")
    print(f"Results for {benchmark}")
    print(f"{'='*60}")
    
    if benchmark == "gym":
        normalize_fn = lambda env, s: td3_normalize(s, env)
        norm_label = "TD3-Normalized"
    elif benchmark == "atari":
        normalize_fn = lambda env, s: human_normalize(s, env)
        norm_label = "Human-Normalized"
    else:
        normalize_fn = lambda env, s: s / 1000.0  # DMC: divide by 1000
        norm_label = "Reward/1000"
    
    # Per-environment results
    print(f"\n{'Environment':<30} {'Mean':>10} {'Std':>10}")
    print("-" * 52)
    for env, scores in sorted(scores_per_env.items()):
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"{env:<30} {mean:>10.1f} {std:>10.1f}")
    
    # Aggregate stats
    stats = compute_aggregate_stats(scores_per_env, normalize_fn)
    print(f"\nAggregate ({norm_label}):")
    for metric, vals in stats.items():
        print(f"  {metric.capitalize():8s}: {vals['value']:.3f} "
              f"[{vals['ci_lower']:.3f}, {vals['ci_upper']:.3f}]")
