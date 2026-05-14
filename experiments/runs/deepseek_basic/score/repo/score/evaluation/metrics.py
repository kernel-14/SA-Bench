"""
Evaluation metrics for self-correction performance.

Implements the metrics defined in Section 3:
- Accuracy@t1: Accuracy at first attempt
- Accuracy@t2: Accuracy at second attempt
- Δ(t1, t2): Net improvement between attempts
- Δ(i→c): Fraction of problems incorrect→correct (correction rate)
- Δ(c→i): Fraction of problems correct→incorrect (degradation rate)

Also includes edit distance ratio analysis from Section 4 (Figure 4)
for detecting behavior collapse.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SelfCorrectionMetrics:
    """
    Complete self-correction evaluation metrics.
    
    Computes the five metrics described in Section 3 used throughout
    the paper (Tables 1, 2, 3, 4).
    """
    accuracy_t1: float
    accuracy_t2: float
    delta_t1_t2: float  # Accuracy@t2 - Accuracy@t1
    i_to_c: float       # incorrect→correct / total
    c_to_i: float       # correct→incorrect / total
    num_samples: int
    
    # Additional analysis metrics
    i_to_c_rate: Optional[float] = None  # i→c / num_incorrect_t1
    c_to_i_rate: Optional[float] = None  # c→i / num_correct_t1
    
    @classmethod
    def from_trajectories(
        cls,
        rewards_t1: List[float],
        rewards_t2: List[float],
    ) -> "SelfCorrectionMetrics":
        """
        Compute metrics from lists of first-turn and second-turn rewards.
        
        Args:
            rewards_t1: Binary rewards for first attempts
            rewards_t2: Binary rewards for second attempts
        """
        n = len(rewards_t1)
        
        correct_t1 = sum(r > 0.5 for r in rewards_t1)
        correct_t2 = sum(r > 0.5 for r in rewards_t2)
        
        i_to_c = 0
        c_to_i = 0
        
        for r1, r2 in zip(rewards_t1, rewards_t2):
            if r1 <= 0.5 and r2 > 0.5:
                i_to_c += 1
            elif r1 > 0.5 and r2 <= 0.5:
                c_to_i += 1
        
        num_incorrect_t1 = n - correct_t1
        
        return cls(
            accuracy_t1=correct_t1 / n if n > 0 else 0.0,
            accuracy_t2=correct_t2 / n if n > 0 else 0.0,
            delta_t1_t2=(correct_t2 - correct_t1) / n if n > 0 else 0.0,
            i_to_c=i_to_c / n if n > 0 else 0.0,
            c_to_i=c_to_i / n if n > 0 else 0.0,
            num_samples=n,
            i_to_c_rate=i_to_c / num_incorrect_t1 if num_incorrect_t1 > 0 else 0.0,
            c_to_i_rate=c_to_i / correct_t1 if correct_t1 > 0 else 0.0,
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy_t1": self.accuracy_t1,
            "accuracy_t2": self.accuracy_t2,
            "delta_t1_t2": self.delta_t1_t2,
            "i_to_c": self.i_to_c,
            "c_to_i": self.c_to_i,
            "i_to_c_rate": self.i_to_c_rate or 0.0,
            "c_to_i_rate": self.c_to_i_rate or 0.0,
            "num_samples": self.num_samples,
        }
    
    def to_table_row(self) -> str:
        """Format as a table row matching paper format."""
        return (
            f"{self.accuracy_t1:.1%} & "
            f"{self.accuracy_t2:.1%} & "
            f"{self.delta_t1_t2:.1%} & "
            f"{self.i_to_c:.1%} & "
            f"{self.c_to_i:.1%}"
        )
    
    @classmethod
    def from_responses(
        cls,
        responses_t1: List[str],
        responses_t2: List[str],
        ground_truths: List[str],
        reward_fn,
    ) -> "SelfCorrectionMetrics":
        """
        Compute metrics from raw response strings.
        
        Args:
            responses_t1: First-attempt response strings
            responses_t2: Second-attempt response strings
            ground_truths: Ground truth answers
            reward_fn: Function(response, ground_truth) -> float
        """
        rewards_t1 = [reward_fn(r, gt) for r, gt in zip(responses_t1, ground_truths)]
        rewards_t2 = [reward_fn(r, gt) for r, gt in zip(responses_t2, ground_truths)]
        return cls.from_trajectories(rewards_t1, rewards_t2)


def compute_edit_distance_ratio(
    text1: str,
    text2: str,
) -> float:
    """
    Compute normalized edit distance between two texts.
    
    Used in Section 4 (Figure 4) to analyze self-correction behavior.
    Edit distance ratio = Levenshtein distance / (len(text1) + len(text2))
    
    A low ratio indicates conservative editing (behavior collapse).
    A high ratio indicates substantial revision.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Edit distance ratio in [0, 1]
    """
    # Levenshtein distance using dynamic programming
    m, n = len(text1), len(text2)
    
    if m + n == 0:
        return 0.0
    
    # Use space-optimized DP
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, prev
    
    distance = prev[n]
    return distance / (m + n)


def compute_edit_distance_distribution(
    responses_t1: List[str],
    responses_t2: List[str],
    num_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of edit distance ratios (for Figure 4 analysis).
    
    Returns:
        bin_edges, histogram values
    """
    ratios = [
        compute_edit_distance_ratio(r1, r2)
        for r1, r2 in zip(responses_t1, responses_t2)
    ]
    
    hist, bin_edges = np.histogram(ratios, bins=num_bins, range=(0, 1))
    return bin_edges, hist


def analyze_behavior_collapse(
    responses_t1: List[str],
    responses_t2: List[str],
    rewards_t1: List[float],
    rewards_t2: List[float],
) -> Dict[str, float]:
    """
    Analyze self-correction behavior for signs of collapse.
    
    Checks for the behavior collapse patterns described in Section 4:
    - Model makes no edits (edit distance ≈ 0)
    - Model makes only superficial edits
    - Model changes correct answers to incorrect
    
    Returns dictionary of analysis metrics.
    """
    n = len(responses_t1)
    
    # Fraction of identical responses (no editing)
    identical = sum(r1 == r2 for r1, r2 in zip(responses_t1, responses_t2))
    no_edit_frac = identical / n if n > 0 else 0.0
    
    # Edit distance statistics
    edit_ratios = [
        compute_edit_distance_ratio(r1, r2)
        for r1, r2 in zip(responses_t1, responses_t2)
    ]
    
    mean_edit = np.mean(edit_ratios) if edit_ratios else 0.0
    median_edit = np.median(edit_ratios) if edit_ratios else 0.0
    
    # Fraction with very small edits (< 5% edit ratio)
    tiny_edits = sum(r < 0.05 for r in edit_ratios)
    tiny_edit_frac = tiny_edits / n if n > 0 else 0.0
    
    # How often does the model change its answer?
    # (Different final answer between turns)
    changed_answer = sum(r1 != r2 for r1, r2 in zip(responses_t1, responses_t2))
    change_frac = changed_answer / n if n > 0 else 0.0
    
    return {
        "no_edit_fraction": no_edit_frac,
        "tiny_edit_fraction": tiny_edit_frac,
        "mean_edit_ratio": mean_edit,
        "median_edit_ratio": median_edit,
        "answer_change_fraction": change_frac,
        "num_samples": n,
    }


def compute_progress_statistics(
    rewards_t1: List[float],
    rewards_t2: List[float],
) -> Dict[str, float]:
    """
    Compute progress statistics for reward shaping analysis.
    
    Progress = r̂(y₂, y*) - r̂(y₁, y*)
    
    Positive: incorrect → correct (good progress)
    Zero: no change in correctness
    Negative: correct → incorrect (bad, penalized by reward shaping)
    """
    n = len(rewards_t1)
    
    progress_values = [r2 - r1 for r1, r2 in zip(rewards_t1, rewards_t2)]
    
    positive_progress = sum(p > 0 for p in progress_values)
    zero_progress = sum(p == 0 for p in progress_values)
    negative_progress = sum(p < 0 for p in progress_values)
    
    return {
        "mean_progress": np.mean(progress_values),
        "positive_progress_frac": positive_progress / n,
        "zero_progress_frac": zero_progress / n,
        "negative_progress_frac": negative_progress / n,
    }


__all__ = [
    "SelfCorrectionMetrics",
    "compute_edit_distance_ratio",
    "compute_edit_distance_distribution",
    "analyze_behavior_collapse",
    "compute_progress_statistics",
]
