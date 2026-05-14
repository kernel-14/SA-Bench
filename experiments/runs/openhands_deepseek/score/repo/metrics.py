"""Self-correction evaluation metrics from the paper.

Metrics:
- Accuracy@t1: accuracy at first attempt
- Accuracy@t2: accuracy at second attempt  
- Δ(t1, t2): net improvement = acc@t2 - acc@t1
- Δ^{i→c}(t1, t2): fraction incorrect at t1 → correct at t2
- Δ^{c→i}(t1, t2): fraction correct at t1 → incorrect at t2
- Edit distance ratio between first and second responses
"""

import math
from typing import Dict, List


def compute_edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m

    # Use dynamic programming
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[n]


def compute_edit_distance_ratio(
    response1: str,
    response2: str,
) -> float:
    """Compute edit distance ratio as defined in the paper.

    ratio = edit_distance / (len(response1) + len(response2))
    """
    dist = compute_edit_distance(response1, response2)
    total_len = len(response1) + len(response2)
    if total_len == 0:
        return 0.0
    return dist / total_len


def compute_self_correction_metrics(
    results: List[Dict],
) -> Dict[str, float]:
    """Compute all self-correction metrics from a list of results.

    Args:
        results: List of dicts, each containing:
            - correct_t1: bool, whether first attempt was correct
            - correct_t2: bool, whether second attempt was correct

    Returns:
        Dictionary of metrics
    """
    n = len(results)
    if n == 0:
        return {
            "accuracy_t1": 0.0,
            "accuracy_t2": 0.0,
            "delta_t1_t2": 0.0,
            "incorrect_to_correct": 0.0,
            "correct_to_incorrect": 0.0,
            "num_samples": 0,
        }

    num_correct_t1 = sum(1 for r in results if r["correct_t1"])
    num_correct_t2 = sum(1 for r in results if r["correct_t2"])

    # Δ^{i→c}: incorrect at t1 → correct at t2
    incorrect_to_correct = sum(
        1 for r in results if not r["correct_t1"] and r["correct_t2"]
    )

    # Δ^{c→i}: correct at t1 → incorrect at t2
    correct_to_incorrect = sum(
        1 for r in results if r["correct_t1"] and not r["correct_t2"]
    )

    return {
        "accuracy_t1": num_correct_t1 / n,
        "accuracy_t2": num_correct_t2 / n,
        "delta_t1_t2": (num_correct_t2 - num_correct_t1) / n,
        "incorrect_to_correct": incorrect_to_correct / n,
        "correct_to_incorrect": correct_to_incorrect / n,
        "num_samples": n,
    }


def compute_inference_scaling_metrics(
    parallel_results: List[Dict],
    sequential_results: List[Dict],
) -> Dict[str, float]:
    """Compute inference-time compute scaling metrics (Section 6.2).

    Compares:
    - Parallel: 2K samples, majority vote
    - Sequential: K samples, each with one self-correction round

    Args:
        parallel_results: Results from 2K parallel samples
        sequential_results: Results from K sequential samples (each with correction)

    Returns:
        Dictionary comparing parallel vs sequential accuracy
    """
    # Parallel majority vote: accuracy of most common answer
    # This is a simplified computation
    parallel_correct = sum(1 for r in parallel_results if r["correct"])
    parallel_accuracy = parallel_correct / len(parallel_results)

    sequential_correct = sum(1 for r in sequential_results if r["correct_t2"])
    sequential_accuracy = sequential_correct / len(sequential_results)

    return {
        "parallel_accuracy": parallel_accuracy,
        "sequential_accuracy": sequential_accuracy,
        "improvement": sequential_accuracy - parallel_accuracy,
    }


def compute_edit_distance_statistics(
    responses_t1: List[str],
    responses_t2: List[str],
) -> Dict[str, float]:
    """Compute edit distance ratio statistics between turns.

    Args:
        responses_t1: List of first-turn responses
        responses_t2: List of second-turn responses

    Returns:
        Dictionary with mean, median, std of edit distance ratios
    """
    ratios = [
        compute_edit_distance_ratio(r1, r2)
        for r1, r2 in zip(responses_t1, responses_t2)
    ]

    if not ratios:
        return {"mean_edit_ratio": 0.0, "median_edit_ratio": 0.0, "std_edit_ratio": 0.0}

    ratios_sorted = sorted(ratios)
    n = len(ratios_sorted)
    median = ratios_sorted[n // 2] if n % 2 == 1 else (
        ratios_sorted[n // 2 - 1] + ratios_sorted[n // 2]
    ) / 2

    mean = sum(ratios) / n
    variance = sum((r - mean) ** 2 for r in ratios) / n
    std = math.sqrt(variance)

    return {
        "mean_edit_ratio": mean,
        "median_edit_ratio": median,
        "std_edit_ratio": std,
    }
