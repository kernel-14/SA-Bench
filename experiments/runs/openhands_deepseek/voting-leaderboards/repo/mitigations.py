"""Mitigations against adversarial manipulation of voting-based leaderboards (Section 4).

Implements all defense strategies discussed in Section 4:

  1. Cost model (Section 4.1): Formalizes attack cost
  2. Authentication (Section 4.2.1): Increases account maintenance cost
  3. Rate limiting (Section 4.2.2): Reduces m (max actions per account)
  4. Malicious user identification (Section 4.2.3):
     - Scenario 1: Known benign distribution (likelihood test)
     - Scenario 2: Known benign + malicious distributions (Neyman-Pearson)
  5. Increasing action cost (Section 4.2.4):
     - CAPTCHA per impression/vote
     - Prompt uniqueness enforcement

Also includes the perturbed leaderboard defense where the defender
releases perturbed Bradley-Terry ratings with Gaussian noise to
detect adversaries who use public rankings to masquerade as benign.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Section 4.1: Cost Model
# ---------------------------------------------------------------------------

def compute_attack_cost(
    N: int,
    c_detector: float = 440.0,
    c_action: float = 0.01,
    c_account: float = 1.0,
    m: Optional[int] = None,
) -> Dict[str, float]:
    """Compute total cost of the adversarial attack.

    Total cost = ceil(N/m) * c_account + N * c_action + c_detector (Section 4.1)

    Args:
        N: Total number of actions (interactions + votes)
        c_detector: One-time cost of building training-based detector ($440 from Appendix B.1)
        c_action: Cost per individual action
        c_account: Cost of obtaining a single user account
        m: Maximum actions permitted per user account (None = unlimited)

    Returns:
        Dict with cost breakdown (detector, account, action, total)
    """
    if m is None or m <= 0:
        account_cost = c_account  # Single account
    else:
        account_cost = np.ceil(N / m) * c_account

    action_cost = N * c_action
    total = c_detector + account_cost + action_cost

    return {
        "detector_cost": c_detector,
        "account_cost": account_cost,
        "action_cost": action_cost,
        "total_cost": total,
    }


def estimate_cost_without_mitigations(N: int) -> Dict[str, float]:
    """Estimate attack cost without any mitigations (Section 4.1).

    Without mitigations: single user, minimal action cost.
    Total cost dominated by detector training cost (~$440).
    """
    return compute_attack_cost(
        N=N,
        c_detector=440.0,
        c_action=0.01,
        c_account=0.0,
        m=None,
    )


# ---------------------------------------------------------------------------
# Section 4.2.1: Authentication
# ---------------------------------------------------------------------------

class AuthenticationDefense:
    """Increases account cost through authentication requirements.

    Enforces authentication through integration with existing digital
    identity providers (email, social media, phone numbers).

    The cost of each account becomes bounded by the resources required
    to obtain associated credentials.
    """

    def __init__(self, account_cost: float = 1.0):
        self.account_cost = account_cost

    def compute_cost(self, N: int, m: Optional[int] = None) -> float:
        """Compute account maintenance cost with authentication."""
        if m is None:
            return self.account_cost
        return np.ceil(N / m) * self.account_cost


# ---------------------------------------------------------------------------
# Section 4.2.2: Rate Limiting
# ---------------------------------------------------------------------------

class RateLimitingDefense:
    """Reduces m (max actions per account) through temporal rate limits.

    Forces attackers to create more accounts. m should be set high enough
    to allow benign users sufficient queries while minimizing adversarial
    actions per account.

    The paper suggests selecting a quantile over the benign user query
    distribution (e.g., the median).
    """

    def __init__(
        self,
        max_actions_per_account: int = 100,
        quantile: float = 0.5,
    ):
        self.max_actions_per_account = max_actions_per_account
        self.quantile = quantile

    def set_limit_from_distribution(self, query_counts: List[int]):
        """Set m based on quantile of benign user query distribution."""
        self.max_actions_per_account = int(np.quantile(query_counts, self.quantile))

    def get_limit(self) -> int:
        return self.max_actions_per_account


# ---------------------------------------------------------------------------
# Section 4.2.3: Malicious User Detection
# ---------------------------------------------------------------------------

class MaliciousUserDetector:
    """Detects malicious users through voting pattern analysis.

    Two scenarios as described in Section 4.2.3:

    Scenario 1: Known Benign Distribution
      - Defender estimates benign user behavior from historical data
      - Uses likelihood ratio test to detect deviations
      - Test statistic: T(x) = -2 * ln(L(x | H_benign))
      - p-value computed via Monte Carlo simulation
      - Reject H_benign if p < alpha (default alpha = 0.01)

    Scenario 2: Known Benign AND Malicious Distributions
      - Defender releases PERTURBED rankings (adds Gaussian noise)
      - Adversary uses perturbed ranks -> detectable deviation
      - Uses Neyman-Pearson Lemma for optimal decision rule
      - Likelihood ratio: Lambda(x) = Pr_M(x) / Pr_B(x)
    """

    def __init__(
        self,
        alpha: float = 0.01,
        num_simulations: int = 1000,
        bradley_terry_scale: float = 1.0,
    ):
        self.alpha = alpha
        self.num_simulations = num_simulations
        self.bradley_terry_scale = bradley_terry_scale

    def compute_bradley_terry_preference_prob(self, Q_i: float, Q_j: float) -> float:
        """Probability that model i is preferred over model j.

        Uses logistic function as described in Section 4.2.3:
        Pr(i preferred over j) = 1 / (1 + exp(-(Q_i - Q_j) / s))
        """
        s = self.bradley_terry_scale
        return 1.0 / (1.0 + np.exp(-(Q_i - Q_j) / s))

    def compute_vote_probabilities(
        self,
        ratings: np.ndarray,
    ) -> np.ndarray:
        """Compute the full probability distribution Pr_B(i) of voting for each model.

        Pr_B(i) = product over j of Pr_B(i preferred over j | true ratings)

        This represents the probability that a benign user votes for model i
        when it appears in a head-to-head comparison.
        """
        n = len(ratings)
        probs = np.ones(n)

        for i in range(n):
            for j in range(n):
                if i != j:
                    probs[i] *= self.compute_bradley_terry_preference_prob(ratings[i], ratings[j])

        probs /= probs.sum()
        return probs

    def compute_log_likelihood(
        self,
        observations: List[int],
        benign_probs: np.ndarray,
    ) -> float:
        """Compute log-likelihood of observation sequence under H_benign.

        L(x | H_benign) = prod_i Pr(x_i | H_benign)
        """
        log_lik = 0.0
        for obs in observations:
            prob = benign_probs[obs] if obs < len(benign_probs) else 0.0
            if prob > 0:
                log_lik += np.log(prob)
            else:
                log_lik += -1e10
        return log_lik

    def compute_test_statistic(
        self,
        observations: List[int],
        benign_probs: np.ndarray,
    ) -> float:
        """Compute test statistic T(x) = -2 * ln(L(x | H_benign))."""
        log_lik = self.compute_log_likelihood(observations, benign_probs)
        return -2.0 * log_lik

    def scenario1_detect(
        self,
        observations: List[int],
        benign_probs: np.ndarray,
    ) -> Tuple[bool, float]:
        """Scenario 1: Detect malicious users using known benign distribution.

        Returns (is_malicious, p_value).
        """
        T_obs = self.compute_test_statistic(observations, benign_probs)

        simulated_T = []
        rng = np.random.RandomState(42)
        for _ in range(self.num_simulations):
            sim_obs = list(rng.choice(
                len(benign_probs),
                size=len(observations),
                p=benign_probs,
            ))
            T_sim = self.compute_test_statistic(sim_obs, benign_probs)
            simulated_T.append(T_sim)

        p_value = np.mean(np.array(simulated_T) >= T_obs)
        is_malicious = p_value < self.alpha

        return is_malicious, p_value

    def scenario2_detect(
        self,
        observations: List[int],
        true_ratings: np.ndarray,
        noise_scale: float = 1.0,
    ) -> Tuple[bool, float]:
        """Scenario 2: Detect malicious users when defender releases perturbed rankings.

        Steps (Section 4.2.3):
        1. Defender adds scaled Gaussian noise to true Bradley-Terry ratings
        2. Releases perturbed ratings
        3. Adversary uses perturbed ratings to choose untargeted model votes
        4. Defender computes likelihood ratio: Lambda(x) = Pr_M(x) / Pr_B(x)
        5. Classify as malicious if Lambda(x) exceeds a threshold
        """
        n = len(true_ratings)

        # Benign probabilities from true ratings
        benign_probs = self.compute_vote_probabilities(true_ratings)

        # Defender releases perturbed ratings
        rng = np.random.RandomState(123)
        perturbed_ratings = true_ratings + rng.normal(0, noise_scale * np.std(true_ratings), n)

        # Malicious probabilities from perturbed ratings (what adversary sees)
        malicious_probs = self.compute_vote_probabilities(perturbed_ratings)

        # Compute likelihood ratio for each observation
        log_likelihood_ratio = 0.0
        for obs in observations:
            if obs < n:
                m_prob = max(malicious_probs[obs], 1e-10)
                b_prob = max(benign_probs[obs], 1e-10)
                log_likelihood_ratio += np.log(m_prob / b_prob)

        # Classify: if likelihood ratio > threshold, mark as malicious
        # The paper uses the Neyman-Pearson Lemma: optimal decision rule is based on LR
        is_malicious = log_likelihood_ratio > 0.0

        return is_malicious, log_likelihood_ratio

    def evaluate_scenario2_utility(
        self,
        true_ratings: np.ndarray,
        noise_scales: List[float],
    ) -> Dict[float, float]:
        """Evaluate utility impact of perturbed rankings.

        Utility measured as average absolute change in ranking position
        for all models (Figure 6).
        """
        n = len(true_ratings)
        true_ranking = np.argsort(-true_ratings)
        true_ranks = np.zeros(n, dtype=int)
        for i, idx in enumerate(true_ranking):
            true_ranks[idx] = i

        results = {}
        rng = np.random.RandomState(123)

        for scale in noise_scales:
            perturbed_ratings = true_ratings + rng.normal(0, scale * np.std(true_ratings), n)
            perturbed_ranking = np.argsort(-perturbed_ratings)
            perturbed_ranks = np.zeros(n, dtype=int)
            for i, idx in enumerate(perturbed_ranking):
                perturbed_ranks[idx] = i

            avg_abs_change = np.mean(np.abs(true_ranks - perturbed_ranks))
            results[scale] = avg_abs_change

        return results

    def evaluate_scenario2_detection_rate(
        self,
        true_ratings: np.ndarray,
        noise_scales: List[float],
        num_trials: int = 100,
        sequence_length: int = 50,
        adversary_prob_fn=None,
    ) -> Dict[float, float]:
        """Evaluate detection rate for Scenario 2 across different noise scales.

        Computes how often the defender correctly identifies a malicious user
        who uses the perturbed rankings to guide their voting behavior (Figure 5).
        """
        n = len(true_ratings)
        results = {}
        rng = np.random.RandomState(456)

        for scale in noise_scales:
            detections = 0
            for _ in range(num_trials):
                perturbed_ratings = true_ratings + rng.normal(0, scale * np.std(true_ratings), n)

                if adversary_prob_fn is not None:
                    adv_probs = adversary_prob_fn(perturbed_ratings)
                else:
                    # Default: adversary always upvotes a specific target model
                    # when present, otherwise uses perturbed ratings for untargeted
                    adv_probs = self.compute_vote_probabilities(perturbed_ratings)

                observations = list(rng.choice(n, size=sequence_length, p=adv_probs))

                is_malicious, _ = self.scenario2_detect(
                    observations=observations,
                    true_ratings=true_ratings,
                    noise_scale=scale,
                )
                if is_malicious:
                    detections += 1

            results[scale] = detections / num_trials

        return results


# ---------------------------------------------------------------------------
# Section 4.2.4: Increasing c_action
# ---------------------------------------------------------------------------

class CAPTCHADefense:
    """Requires CAPTCHA per impression/vote.

    Increases action cost: c_action = c_CAPTCHA.
    Automated CAPTCHA-solving services typically charge per CAPTCHA.
    """

    def __init__(self, captcha_cost: float = 0.001):
        self.captcha_cost = captcha_cost

    def compute_cost(self, N: int) -> float:
        return N * self.captcha_cost


class PromptUniquenessDefense:
    """Enforces prompt uniqueness to force new data collection.

    Rejects or downweights previously used prompts. This forces attackers
    to generate new prompts and train corresponding detectors for each action.

    Cost per prompt: approximately $20 (Appendix A.3).
    """

    def __init__(self, cost_per_prompt: float = 20.0):
        self.cost_per_prompt = cost_per_prompt
        self.seen_prompts: set = set()

    def is_new_prompt(self, prompt: str) -> bool:
        return prompt not in self.seen_prompts

    def register_prompt(self, prompt: str):
        self.seen_prompts.add(prompt)

    def compute_cost(self, N: int) -> float:
        return N * self.cost_per_prompt


# ---------------------------------------------------------------------------
# Combined defense evaluation
# ---------------------------------------------------------------------------

def evaluate_defenses(
    N: int,
    c_detector: float = 440.0,
    c_action_base: float = 0.01,
    c_account: float = 1.0,
    m: Optional[int] = None,
    captcha_enabled: bool = False,
    c_captcha: float = 0.001,
    prompt_uniqueness_enabled: bool = False,
    c_prompt: float = 20.0,
) -> Dict[str, float]:
    """Evaluate the total attack cost under various defense configurations.

    Combines all mitigation strategies into a single cost estimate,
    as described in Section 4.1.
    """
    c_action = c_action_base
    if captcha_enabled:
        c_action += c_captcha
    if prompt_uniqueness_enabled:
        c_action += c_prompt

    return compute_attack_cost(
        N=N,
        c_detector=c_detector,
        c_action=c_action,
        c_account=c_account,
        m=m,
    )


def generate_detection_rate_curve(
    true_ratings: np.ndarray,
    noise_scales: List[float],
    detector: MaliciousUserDetector,
    num_trials: int = 100,
    sequence_length: int = 50,
) -> Dict[float, float]:
    """Generate the detection rate vs noise scale curve (Figure 5)."""
    return detector.evaluate_scenario2_detection_rate(
        true_ratings=true_ratings,
        noise_scales=noise_scales,
        num_trials=num_trials,
        sequence_length=sequence_length,
    )


def generate_utility_curve(
    true_ratings: np.ndarray,
    noise_scales: List[float],
    detector: MaliciousUserDetector,
) -> Dict[float, float]:
    """Generate the utility impact vs noise scale curve (Figure 6).

    Utility = average absolute change in ranking position.
    """
    return detector.evaluate_scenario2_utility(
        true_ratings=true_ratings,
        noise_scales=noise_scales,
    )
