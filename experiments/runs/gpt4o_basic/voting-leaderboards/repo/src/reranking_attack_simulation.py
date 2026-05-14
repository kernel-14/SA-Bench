"""
Simulating reranking attacks on voting-based leaderboards as described in Section 3.
The simulation estimates the required votes to manipulate rankings based on Bradley-Terry coefficients.
"""

import numpy as np

def bradley_terry_ranking(votes):
    """
    Computes model rankings based on votes using Bradley-Terry coefficients.
    Args:
        votes (dict): Dictionary where keys are model pairs and values are pairwise vote counts.
                      Example: {("Model_A", "Model_B"): 50, ("Model_B", "Model_C"): 30}
    Returns:
        dict: Rankings with Bradley-Terry coefficients.
    """
    models = set([model for pair in votes.keys() for model in pair])
    scores = {model: 1.0 for model in models}

    for _ in range(100):  # Iterative optimization to stabilize rankings
        updated_scores = {}
        for model in models:
            total_score = 0
            for (model_a, model_b), vote_count in votes.items():
                if model == model_a:
                    total_score += vote_count / (scores[model_a] + scores[model_b])
                elif model == model_b:
                    total_score += vote_count / (scores[model_a] + scores[model_b])
            updated_scores[model] = total_score
        scores = updated_scores

    rankings = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {model: rank for rank, (model, _) in enumerate(rankings, start=1)}

# Example workflow (simulated votes):
votes = {
    ("Model_A", "Model_B"): 50,
    ("Model_B", "Model_C"): 30,
    ("Model_A", "Model_C"): 70
}
ranking = bradley_terry_ranking(votes)
print("Rankings:", ranking)
