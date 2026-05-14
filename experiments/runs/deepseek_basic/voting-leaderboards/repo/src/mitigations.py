"""
Mitigations (Section 4)

This module implements the defense mechanisms described in the paper:
1. Cost model for attacks (Section 4.1)
2. Malicious user identification (Section 4.2.3)
   - Scenario 1: Known Benign Distribution
   - Scenario 2: Known Benign and Malicious Distributions
3. Perturbed leaderboard release (Section 4.2.3)
4. Authentication and rate limiting (Section 4.2.1, 4.2.2)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Cost Model (Section 4.1)
# =============================================================================

@dataclass
class AttackCost:
    """Cost model for the adversarial attack.
    
    Total cost = detector_cost + account_cost + action_cost
    
    c = detector_cost + ceil(N/m) * c_account + N * c_action
    
    where:
    - N: number of actions (interactions or votes)
    - m: max actions per account
    - c_account: cost per account
    - c_action: cost per action
    - c_detector: one-time detector training cost
    """
    
    # Number of actions required
    n_actions: int
    
    # Max actions per account before detection
    max_actions_per_account: int = float('inf')
    
    # Cost per account (e.g., cost of obtaining credentials)
    cost_per_account: float = 0.0
    
    # Cost per action
    cost_per_action: float = 0.0
    
    # One-time detector training cost (estimated at ~$440 in paper)
    detector_cost: float = 440.0
    
    def total_cost(self) -> float:
        """Compute total attack cost."""
        n_accounts = max(1, int(np.ceil(self.n_actions / self.max_actions_per_account)))
        account_cost = n_accounts * self.cost_per_account
        action_cost = self.n_actions * self.cost_per_action
        return self.detector_cost + account_cost + action_cost
    
    def breakdown(self) -> Dict:
        """Return cost breakdown."""
        n_accounts = max(1, int(np.ceil(self.n_actions / self.max_actions_per_account)))
        return {
            "detector_cost": self.detector_cost,
            "n_accounts": n_accounts,
            "account_cost": n_accounts * self.cost_per_account,
            "action_cost": self.n_actions * self.cost_per_action,
            "total_cost": self.total_cost(),
        }
    
    def cost_with_defense(
        self,
        max_actions: Optional[int] = None,
        cost_per_account: Optional[float] = None,
        cost_per_action: Optional[float] = None,
    ) -> "AttackCost":
        """Create a new cost model with defense parameters applied."""
        return AttackCost(
            n_actions=self.n_actions,
            max_actions_per_account=max_actions or self.max_actions_per_account,
            cost_per_account=cost_per_account or self.cost_per_account,
            cost_per_action=cost_per_action or self.cost_per_action,
            detector_cost=self.detector_cost,
        )


def estimate_detector_training_cost(
    n_prompts: int = 200,
    n_proprietary_models: int = 10,
    n_open_source_models: int = 20,
    responses_per_model: int = 50,
    output_tokens: int = 512,
    proprietary_cost_per_1m_tokens: float = 5.00,
    open_source_cost_per_1m_tokens: float = 1.80,
) -> float:
    """Estimate the cost of training a detector (Appendix A.3).
    
    Based on the paper's cost estimation:
    - Proprietary models: $5.00 per 1M output tokens
    - Open-source models: $1.80 per 1M output tokens
    - 50 responses per model, each 512 tokens
    - 200 prompts
    
    Returns total estimated cost in USD.
    """
    tokens_per_model = output_tokens * responses_per_model
    
    proprietary_cost = (
        proprietary_cost_per_1m_tokens * tokens_per_model * n_proprietary_models / 1e6
    )
    open_source_cost = (
        open_source_cost_per_1m_tokens * tokens_per_model * n_open_source_models / 1e6
    )
    
    cost_per_prompt = proprietary_cost + open_source_cost
    total_cost = cost_per_prompt * n_prompts
    
    return total_cost


# =============================================================================
# Malicious User Identification (Section 4.2.3)
# =============================================================================

class MaliciousUserDetector:
    """Likelihood-based malicious user detection.
    
    Based on the intuition that benign users will show similar model preferences,
    while malicious users will deviate from these patterns.
    
    Uses a likelihood test to differentiate between:
    H_benign: User's voting pattern matches known benign distribution
    H_¬benign: User is from a different source
    """
    
    def __init__(self, benign_vote_probs: np.ndarray):
        """
        Args:
            benign_vote_probs: Array of shape (n_models,) with probability
                              that a benign user votes for each model
        """
        self.benign_probs = np.array(benign_vote_probs)
        self.benign_probs = self.benign_probs / self.benign_probs.sum()
        self.n_models = len(self.benign_probs)
    
    def compute_log_likelihood(
        self, observations: np.ndarray, distribution: np.ndarray
    ) -> float:
        """Compute log-likelihood of observations under given distribution.
        
        L(x | H) = prod_i Pr(x_i | H)
        log L = sum_i log(Pr(x_i | H))
        
        Args:
            observations: Array of model indices observed
            distribution: Probability distribution over models
            
        Returns:
            log_likelihood
        """
        probs = distribution[observations]
        # Avoid log(0)
        probs = np.maximum(probs, 1e-10)
        return np.sum(np.log(probs))
    
    def compute_test_statistic(self, observations: np.ndarray) -> float:
        """Compute test statistic T(x) = -2 ln(L(x|H_benign)).
        
        As described in Section 4.2.3, Scenario 1.
        """
        ll = self.compute_log_likelihood(observations, self.benign_probs)
        return -2.0 * ll
    
    def empirical_pvalue(
        self,
        observations: np.ndarray,
        n_simulations: int = 1000,
        verbose: bool = False,
    ) -> float:
        """Compute empirical p-value for the observations.
        
        As described in Section 4.2.3:
        - Simulate m sequences under null hypothesis
        - Calculate test statistic for each
        - p = (1/m) * sum_j I(T(s^j) >= T(x))
        
        Args:
            observations: Observed sequence of model impressions
            n_simulations: Number of simulation sequences (m)
            verbose: Whether to log details
            
        Returns:
            p-value
        """
        T_obs = self.compute_test_statistic(observations)
        n = len(observations)
        
        # Simulate sequences under benign distribution
        T_sim = np.zeros(n_simulations)
        for j in range(n_simulations):
            sim_seq = np.random.choice(self.n_models, size=n, p=self.benign_probs)
            T_sim[j] = self.compute_test_statistic(sim_seq)
        
        # Compute empirical p-value
        p_value = np.mean(T_sim >= T_obs)
        
        if verbose:
            logger.info(f"Observed T: {T_obs:.2f}")
            logger.info(f"Simulated T mean: {T_sim.mean():.2f}, std: {T_sim.std():.2f}")
            logger.info(f"P-value: {p_value:.4f}")
        
        return p_value
    
    def is_malicious(
        self, 
        observations: np.ndarray, 
        alpha: float = 0.01,
        n_simulations: int = 1000,
    ) -> Tuple[bool, float]:
        """Determine if user is malicious based on likelihood test.
        
        Reject H_benign if p-value < alpha.
        As described in paper, alpha = 0.01.
        
        Returns:
            (is_malicious, p_value)
        """
        p_value = self.empirical_pvalue(observations, n_simulations)
        return (p_value < alpha, p_value)


class NeymanPearsonDetector:
    """Optimal detector using Neyman-Pearson Lemma (Scenario 2).
    
    When the defender knows both benign and malicious distributions,
    the optimal decision rule is based on the likelihood ratio.
    
    Lambda(x) = Pr_M(x) / Pr_B(x)
    
    where Pr_B is benign distribution and Pr_M is malicious distribution.
    
    As described in Section 4.2.3, Scenario 2.
    """
    
    def __init__(
        self, 
        benign_probs: np.ndarray, 
        malicious_probs: np.ndarray,
    ):
        """
        Args:
            benign_probs: Probability distribution for benign users
            malicious_probs: Probability distribution for malicious users
        """
        self.benign_probs = np.array(benign_probs)
        self.benign_probs = self.benign_probs / self.benign_probs.sum()
        
        self.malicious_probs = np.array(malicious_probs)
        self.malicious_probs = self.malicious_probs / self.malicious_probs.sum()
        
        self.n_models = len(self.benign_probs)
    
    def compute_likelihood_ratio(self, observations: np.ndarray) -> float:
        """Compute likelihood ratio Lambda(x) = Pr_M(x) / Pr_B(x).
        
        Args:
            observations: Sequence of model indices
            
        Returns:
            likelihood_ratio
        """
        log_lr = 0.0
        for obs in observations:
            prob_benign = max(self.benign_probs[obs], 1e-10)
            prob_malicious = max(self.malicious_probs[obs], 1e-10)
            log_lr += np.log(prob_malicious) - np.log(prob_benign)
        
        return np.exp(log_lr)
    
    def classify(
        self, 
        observations: np.ndarray, 
        threshold: float = 1.0,
    ) -> Tuple[bool, float]:
        """Classify user as benign or malicious.
        
        If Lambda(x) > threshold, classify as malicious.
        Threshold of 1.0 is the default (equal priors).
        
        Returns:
            (is_malicious, likelihood_ratio)
        """
        lr = self.compute_likelihood_ratio(observations)
        return (lr > threshold, lr)


# =============================================================================
# Bradley-Terry Probability Distribution (Section 4.2.3)
# =============================================================================

def compute_vote_distribution_from_ratings(
    ratings: np.ndarray,
    scale: float = 400.0,
) -> np.ndarray:
    """Compute the probability that each model wins a pairwise comparison.
    
    As described in Section 4.2.3:
    Pr_B(i) = prod_j Pr(i preferred over j | true Bradley-Terry ratings)
    
    where Pr(i preferred over j) = 1 / (1 + exp(-(Q_i - Q_j) / s))
    
    Args:
        ratings: Array of Bradley-Terry coefficients for each model
        scale: Scaling factor s
        
    Returns:
        Array of probabilities for each model
    """
    n = len(ratings)
    probs = np.ones(n)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                prob_i_over_j = 1.0 / (1.0 + np.exp(-(ratings[i] - ratings[j]) / scale))
                probs[i] *= prob_i_over_j
    
    # Normalize
    probs = probs / probs.sum()
    return probs


# =============================================================================
# Perturbed Leaderboard (Section 4.2.3, Scenario 2)
# =============================================================================

class PerturbedLeaderboard:
    """Release a perturbed version of the leaderboard.
    
    As described in Section 4.2.3:
    The defender can release perturbed rankings and counts to reduce
    an attacker's knowledge of the true values. This comes with a
    security-utility tradeoff.
    
    The perturbation adds scaled Gaussian noise to Bradley-Terry 
    coefficient ratings before releasing the rating.
    """
    
    def __init__(
        self,
        true_ratings: Dict[str, float],
        noise_scale: float = 50.0,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            true_ratings: True Bradley-Terry coefficients
            noise_scale: Standard deviation of Gaussian noise to add
            random_state: Random seed
        """
        self.true_ratings = true_ratings.copy()
        self.noise_scale = noise_scale
        self.rng = np.random.RandomState(random_state)
        
        # Generate perturbed ratings
        self.perturbed_ratings = {}
        for model, rating in true_ratings.items():
            noise = self.rng.normal(0, noise_scale)
            self.perturbed_ratings[model] = rating + noise
    
    def get_perturbed_ranking(self) -> List[Tuple[str, float]]:
        """Get perturbed ranking."""
        return sorted(
            self.perturbed_ratings.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
    
    def get_true_ranking(self) -> List[Tuple[str, float]]:
        """Get true ranking."""
        return sorted(
            self.true_ratings.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
    
    def compute_utility_loss(self) -> float:
        """Compute utility loss as the average absolute change in rankings.
        
        As described in Section 4.3 (Figure 6 caption).
        """
        true_ranked = self.get_true_ranking()
        perturbed_ranked = self.get_perturbed_ranking()
        
        true_ranks = {name: i for i, (name, _) in enumerate(true_ranked)}
        perturbed_ranks = {name: i for i, (name, _) in enumerate(perturbed_ranked)}
        
        total_change = 0.0
        for model in true_ranks:
            total_change += abs(true_ranks[model] - perturbed_ranks[model])
        
        return total_change / len(true_ranks)
    
    def get_perturbed_vote_distribution(
        self, scale: float = 400.0
    ) -> np.ndarray:
        """Compute vote distribution based on perturbed ratings.
        
        This is the distribution that an attacker would see and potentially
        use to mimic benign behavior.
        """
        sorted_models = [m for m, _ in self.get_perturbed_ranking()]
        ratings = np.array([self.perturbed_ratings[m] for m in sorted_models])
        return compute_vote_distribution_from_ratings(ratings, scale)
    
    def get_true_vote_distribution(self, scale: float = 400.0) -> np.ndarray:
        """Compute vote distribution based on true ratings."""
        sorted_models = [m for m, _ in self.get_true_ranking()]
        ratings = np.array([self.true_ratings[m] for m in sorted_models])
        return compute_vote_distribution_from_ratings(ratings, scale)


# =============================================================================
# Detection Rate vs Noise Experiment (Section 4.3, Figures 5, 6)
# =============================================================================

def evaluate_malicious_detection_with_noise(
    true_ratings: Dict[str, float],
    attacker_target: str,
    noise_scales: List[float],
    n_observations: int = 100,
    n_trials: int = 100,
    scale: float = 400.0,
    alpha: float = 0.01,
) -> Dict:
    """
    Evaluate detection rate and utility loss for different noise scales.
    
    This reproduces the analysis behind Figures 5 and 6.
    
    Args:
        true_ratings: True Bradley-Terry coefficients
        attacker_target: The model the attacker is targeting
        noise_scales: List of noise scales to test
        n_observations: Number of votes to observe per user
        n_trials: Number of trials per noise scale
        scale: Bradley-Terry scale parameter
        alpha: Significance level for detection
        
    Returns:
        Dict with detection rates and utility losses per noise scale
    """
    model_names = list(true_ratings.keys())
    true_ratings_array = np.array([true_ratings[m] for m in model_names])
    true_dist = compute_vote_distribution_from_ratings(true_ratings_array, scale)
    
    # Get target model index
    target_idx = model_names.index(attacker_target)
    
    results = {"noise_scale": [], "detection_rate": [], "utility_loss": []}
    
    for noise_scale in noise_scales:
        # Create perturbed leaderboard
        pl = PerturbedLeaderboard(true_ratings, noise_scale=noise_scale)
        perturbed_dist = pl.get_perturbed_vote_distribution(scale)
        utility_loss = pl.compute_utility_loss()
        
        # Build attacker distribution: always prefers target, uses perturbed for others
        attacker_dist = perturbed_dist.copy()
        
        detections = 0
        
        for trial in range(n_trials):
            # Generate attacker observations
            # Attacker behavior: always votes for target when detected,
            # uses perturbed distribution when target not present
            observations = []
            for _ in range(n_observations):
                # Simulate one comparison
                r = np.random.random()
                if r < 0.5:  # target is one of the two models
                    # Attacker votes for target
                    observations.append(target_idx)
                else:
                    # Attacker uses perturbed distribution
                    observations.append(np.random.choice(len(model_names), p=attacker_dist))
            
            observations = np.array(observations)
            
            # Use Scenario 1 detector (only knows benign distribution)
            detector = MaliciousUserDetector(true_dist)
            is_malicious, _ = detector.is_malicious(
                observations, alpha=alpha, n_simulations=500
            )
            
            if is_malicious:
                detections += 1
        
        detection_rate = detections / n_trials
        results["noise_scale"].append(noise_scale)
        results["detection_rate"].append(detection_rate)
        results["utility_loss"].append(utility_loss)
        
        logger.info(
            f"Noise scale={noise_scale:.1f}: "
            f"Detection rate={detection_rate:.3f}, "
            f"Utility loss={utility_loss:.2f}"
        )
    
    return results


# =============================================================================
# Rate Limiting Analysis (Section 4.2.2)
# =============================================================================

def compute_rate_limit_effect(
    benign_query_distribution: np.ndarray,
    quantile: float = 0.5,
) -> Dict:
    """
    Compute the effect of setting a rate limit at a given quantile.
    
    As described in Section 4.2.2:
    Select a quantile over user query distribution, e.g., the median.
    m should be set high enough to allow benign users as many queries 
    as possible while minimizing adversarial user queries.
    
    Args:
        benign_query_distribution: Array of query counts per benign user
        quantile: Quantile to set limit at
        
    Returns:
        Dict with analysis results
    """
    limit = np.quantile(benign_query_distribution, quantile)
    benign_users_affected = np.mean(benign_query_distribution > limit)
    benign_queries_blocked = (
        np.sum(np.maximum(benign_query_distribution - limit, 0))
        / np.sum(benign_query_distribution)
    )
    
    return {
        "rate_limit": limit,
        "quantile": quantile,
        "benign_users_affected": benign_users_affected,
        "benign_queries_blocked": benign_queries_blocked,
    }


# =============================================================================
# CAPTCHA Cost Analysis (Section 4.2.4)
# =============================================================================

def estimate_captcha_defense_cost(
    n_actions: int,
    captcha_cost_per_solve: float = 0.002,  # Typical automated CAPTCHA solving cost
) -> float:
    """Estimate additional cost from requiring CAPTCHA per action.
    
    As described in Section 4.2.4:
    c_action = N * c_CAPTCHA
    """
    return n_actions * captcha_cost_per_solve


# =============================================================================
# Prompt Uniqueness Defense (Section 4.2.4)
# =============================================================================

def estimate_prompt_uniqueness_cost(
    n_actions: int,
    cost_per_new_prompt: float = 20.0,  # ~$20 per prompt as stated in paper
) -> float:
    """Estimate additional cost from prompt uniqueness enforcement.
    
    As described in Section 4.2.4:
    Rejecting previously used prompts forces attackers to generate new
    prompts and train corresponding detectors, at ~$20 per prompt.
    """
    return n_actions * cost_per_new_prompt
