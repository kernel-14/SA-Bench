"""Analysis tools for PETL experiments."""
from .prediction_similarity import (
    compute_prediction_similarity,
    compute_confidence_overlap,
    compute_prediction_diversity_score,
    ensemble_majority_vote,
    ensemble_average_logits,
)
from .ranking import (
    compute_rankings,
    compute_ranking_frequency,
    compute_group_ranking_frequencies,
    compute_relative_std,
    identify_task_categories,
)
from .plot_utils import (
    plot_accuracy_gain,
    plot_ranking_frequency,
    plot_prediction_similarity,
    plot_ensemble_gain,
    plot_param_size_vs_accuracy,
    plot_task_categories,
    plot_wise_curves,
)
