"""
Evaluation metrics and reporting utilities.

Computes and formats all metrics reported in the paper:
  - Detector accuracy (Table 2, Table 3, Figure 3)
  - Identity-probing accuracy (Table 2, Table 7)
  - Leaderboard rank changes from adversarial voting (Tables 4, 5, 8, 9)
  - Malicious user detection rates (Figures 4, 5)
  - Utility cost of perturbed leaderboard (Figure 6)
  - Attack cost breakdown (Section 4.1)
  - PCA visualization of BoW embeddings (Figure 2)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from bradley_terry import get_rankings, strengths_to_elo
from config import IDENTITY_PROBING_PROMPTS, MAIN_EVAL_MODELS, PROMPT_CATEGORIES
from data import VoteRecord
from features import compute_pca_embeddings


# ---------------------------------------------------------------------------
# Detector evaluation metrics
# ---------------------------------------------------------------------------

def compute_accuracy(predictions: List[int], labels: List[int]) -> float:
    """Compute binary classification accuracy."""
    if not labels:
        return 0.0
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels)


def compute_confusion_matrix(
    predictions: List[int],
    labels: List[int],
) -> Dict[str, int]:
    """
    Compute binary confusion matrix.

    Returns:
        Dict with keys: tp, fp, tn, fn.
    """
    tp = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(predictions, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_precision_recall_f1(
    predictions: List[int],
    labels: List[int],
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score."""
    cm = compute_confusion_matrix(predictions, labels)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return precision, recall, f1


def format_detector_table(
    results: Dict[str, Dict[str, float]],
    models: Optional[List[str]] = None,
    feature_types: Optional[List[str]] = None,
) -> str:
    """
    Format detector accuracy results as a table (Table 3 format).

    Args:
        results: {feature_type -> {model -> accuracy}}
        models: Model names to include (in order).
        feature_types: Feature types to include (in order).

    Returns:
        Formatted table string.
    """
    if models is None:
        models = MAIN_EVAL_MODELS
    if feature_types is None:
        feature_types = ["length_word", "length_char", "bow", "tfidf"]

    header_map = {
        "length_word": "Length(R)_word",
        "length_char": "Length(R)_char",
        "bow": "BoW(R)",
        "tfidf": "TF-IDF(R)",
    }

    col_width = 16
    model_width = 35

    header = f"{'Model':<{model_width}}"
    for ft in feature_types:
        header += f"{header_map.get(ft, ft):>{col_width}}"
    lines = [header, "-" * (model_width + col_width * len(feature_types))]

    for model in models:
        row = f"{model:<{model_width}}"
        for ft in feature_types:
            acc = results.get(ft, {}).get(model, float("nan"))
            if np.isnan(acc):
                row += f"{'N/A':>{col_width}}"
            else:
                row += f"{acc * 100:>{col_width}.1f}"
        lines.append(row)

    return "\n".join(lines)


def format_identity_probing_table(
    results: Dict[str, Dict[str, float]],
    models: Optional[List[str]] = None,
    prompts: Optional[List[str]] = None,
) -> str:
    """
    Format identity-probing detector results as a table (Table 2/7 format).

    Args:
        results: {model -> {prompt -> accuracy}}
        models: Model names to include.
        prompts: Prompts to include.

    Returns:
        Formatted table string.
    """
    if models is None:
        models = MAIN_EVAL_MODELS
    if prompts is None:
        prompts = IDENTITY_PROBING_PROMPTS

    short_prompts = [p[:20] + "..." if len(p) > 20 else p for p in prompts]
    col_width = 24
    model_width = 35

    header = f"{'Model':<{model_width}}"
    for sp in short_prompts:
        header += f"{sp:>{col_width}}"
    lines = [header, "-" * (model_width + col_width * len(prompts))]

    for model in models:
        row = f"{model:<{model_width}}"
        model_results = results.get(model, {})
        for prompt in prompts:
            acc = model_results.get(prompt, float("nan"))
            if np.isnan(acc):
                row += f"{'N/A':>{col_width}}"
            else:
                row += f"{acc * 100:>{col_width}.1f}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Leaderboard evaluation metrics
# ---------------------------------------------------------------------------

def compute_rank_correlation(
    rankings_a: Dict[str, int],
    rankings_b: Dict[str, int],
) -> float:
    """
    Compute Spearman rank correlation between two ranking systems.

    Args:
        rankings_a: {model -> rank} for system A.
        rankings_b: {model -> rank} for system B.

    Returns:
        Spearman correlation coefficient in [-1, 1].
    """
    from scipy.stats import spearmanr

    common_models = sorted(set(rankings_a.keys()) & set(rankings_b.keys()))
    if len(common_models) < 2:
        return float("nan")

    ranks_a = [rankings_a[m] for m in common_models]
    ranks_b = [rankings_b[m] for m in common_models]

    corr, _ = spearmanr(ranks_a, ranks_b)
    return float(corr)


def format_simulation_table(
    results: Dict[str, Dict[int, Tuple[float, float]]],
    metric: str = "votes",
    target_ranks: Optional[List[int]] = None,
) -> str:
    """
    Format simulation results as a table (Tables 4/5 format).

    Args:
        results: {model -> {target_rank -> (votes, interactions)}}
        metric: "votes" or "interactions".
        target_ranks: Target ranks to include.

    Returns:
        Formatted table string.
    """
    if not results:
        return "No results."

    all_target_ranks = sorted(set(
        r for model_results in results.values() for r in model_results.keys()
    ))
    if target_ranks is not None:
        all_target_ranks = [r for r in all_target_ranks if r in target_ranks]

    metric_idx = 0 if metric == "votes" else 1
    col_width = 14
    model_width = 35

    header = f"{'Target model':<{model_width}}"
    for rank in all_target_ranks:
        header += f"{'Rank ' + str(rank):>{col_width}}"
    lines = [header, "-" * (model_width + col_width * len(all_target_ranks))]

    for model, rank_results in results.items():
        row = f"{model:<{model_width}}"
        for rank in all_target_ranks:
            if rank in rank_results:
                val = rank_results[rank][metric_idx]
                if val == float("inf"):
                    row += f"{'N/A':>{col_width}}"
                else:
                    row += f"{val:>{col_width}.0f}"
            else:
                row += f"{'N/A':>{col_width}}"
        lines.append(row)

    return "\n".join(lines)


def format_ablation_table(
    results: Dict[float, Dict[int, Tuple[float, float]]],
    metric: str = "votes",
    target_ranks: Optional[List[int]] = None,
) -> str:
    """
    Format detector accuracy ablation results (Table 8 format).

    Args:
        results: {accuracy -> {target_rank -> (votes, interactions)}}
        metric: "votes" or "interactions".
        target_ranks: Target ranks to include.

    Returns:
        Formatted table string.
    """
    if not results:
        return "No results."

    all_target_ranks = sorted(set(
        r for acc_results in results.values() for r in acc_results.keys()
    ))
    if target_ranks is not None:
        all_target_ranks = [r for r in all_target_ranks if r in target_ranks]

    metric_idx = 0 if metric == "votes" else 1
    col_width = 14
    acc_width = 20

    header = f"{'Detector accuracy':<{acc_width}}"
    for rank in all_target_ranks:
        header += f"{'Rank ' + str(rank):>{col_width}}"
    lines = [header, "-" * (acc_width + col_width * len(all_target_ranks))]

    for acc in sorted(results.keys()):
        row = f"{'acc=' + str(acc):<{acc_width}}"
        for rank in all_target_ranks:
            if rank in results[acc]:
                val = results[acc][rank][metric_idx]
                if val == float("inf"):
                    row += f"{'N/A':>{col_width}}"
                else:
                    row += f"{val:>{col_width}.0f}"
            else:
                row += f"{'N/A':>{col_width}}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mitigation evaluation metrics
# ---------------------------------------------------------------------------

def compute_detection_roc(
    tpr_fpr_pairs: List[Tuple[float, float]],
) -> float:
    """
    Compute AUC-ROC from a list of (TPR, FPR) pairs.

    Args:
        tpr_fpr_pairs: List of (true_positive_rate, false_positive_rate) tuples.

    Returns:
        AUC-ROC score.
    """
    if len(tpr_fpr_pairs) < 2:
        return float("nan")

    sorted_pairs = sorted(tpr_fpr_pairs, key=lambda x: x[1])
    fprs = [p[1] for p in sorted_pairs]
    tprs = [p[0] for p in sorted_pairs]

    auc = np.trapz(tprs, fprs)
    return float(abs(auc))


def format_mitigation_table(
    noise_results: Dict[float, Dict[str, float]],
) -> str:
    """
    Format perturbed leaderboard results (Figures 5/6 format).

    Args:
        noise_results: {noise_scale -> {"fpr": float, "tpr": float, "mean_rank_change": float}}

    Returns:
        Formatted table string.
    """
    col_width = 16
    scale_width = 12

    header = (
        f"{'Noise scale':<{scale_width}}"
        f"{'TPR':>{col_width}}"
        f"{'FPR':>{col_width}}"
        f"{'Rank change':>{col_width}}"
    )
    lines = [header, "-" * (scale_width + 3 * col_width)]

    for noise_scale in sorted(noise_results.keys()):
        metrics = noise_results[noise_scale]
        row = (
            f"{noise_scale:<{scale_width}.2f}"
            f"{metrics['tpr']:>{col_width}.3f}"
            f"{metrics['fpr']:>{col_width}.3f}"
            f"{metrics['mean_rank_change']:>{col_width}.2f}"
        )
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PCA visualization (Figure 2)
# ---------------------------------------------------------------------------

def compute_bow_pca_for_visualization(
    responses_by_model: Dict[str, List[str]],
    n_components: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Compute PCA-reduced BoW embeddings for visualization (Figure 2).

    Args:
        responses_by_model: {model_name -> [responses]}
        n_components: Number of PCA components.

    Returns:
        {model_name -> array of shape (n_responses, n_components)}
    """
    all_responses = []
    model_boundaries = {}
    start = 0

    for model, responses in responses_by_model.items():
        all_responses.extend(responses)
        model_boundaries[model] = (start, start + len(responses))
        start += len(responses)

    embeddings = compute_pca_embeddings(all_responses, n_components=n_components, feature_type="bow")

    result = {}
    for model, (s, e) in model_boundaries.items():
        result[model] = embeddings[s:e]

    return result


def plot_pca_embeddings(
    embeddings_by_model: Dict[str, np.ndarray],
    prompt_label: str = "",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot PCA embeddings colored by model (Figure 2 in the paper).

    Args:
        embeddings_by_model: {model_name -> array of shape (n, 2)}
        prompt_label: Label for the plot title.
        save_path: If provided, save the figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping PCA plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(embeddings_by_model)))

    for (model, emb), color in zip(embeddings_by_model.items(), colors):
        ax.scatter(emb[:, 0], emb[:, 1], label=model, alpha=0.6, s=20, color=color)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"BoW PCA Embeddings{' - ' + prompt_label if prompt_label else ''}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"PCA plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------------
# Leaderboard display
# ---------------------------------------------------------------------------

def format_leaderboard(
    strengths: Dict[str, float],
    top_k: Optional[int] = None,
    elo_scale: float = 400.0,
) -> str:
    """
    Format a leaderboard from Bradley-Terry strength parameters.

    Args:
        strengths: {model -> BT strength}
        top_k: Show only top-k models.
        elo_scale: Scale for Elo conversion.

    Returns:
        Formatted leaderboard string.
    """
    elo_ratings = strengths_to_elo(strengths, scale=elo_scale)
    rankings = get_rankings(strengths)

    if top_k is not None:
        rankings = rankings[:top_k]

    rank_width = 6
    model_width = 40
    elo_width = 10

    header = f"{'Rank':<{rank_width}}{'Model':<{model_width}}{'Elo':>{elo_width}}"
    lines = [header, "-" * (rank_width + model_width + elo_width)]

    for rank, model, _ in rankings:
        elo = elo_ratings.get(model, 0.0)
        lines.append(f"{rank:<{rank_width}}{model:<{model_width}}{elo:>{elo_width}.1f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_summary_report(
    detector_results: Optional[Dict] = None,
    simulation_results: Optional[Dict] = None,
    mitigation_results: Optional[Dict] = None,
) -> str:
    """
    Generate a human-readable summary report of all experiment results.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 70,
        "SUMMARY REPORT: Adversarial Manipulation of Voting-Based Leaderboards",
        "=" * 70,
    ]

    if detector_results:
        lines.append("\n--- Section 2: De-anonymization Results ---")

        if "table3" in detector_results:
            lines.append("\nTable 3: Feature comparison (English prompts, BoW accuracy):")
            bow_results = detector_results["table3"].get("bow", {})
            for model, acc in bow_results.items():
                lines.append(f"  {model}: {acc * 100:.1f}%")

        if "figure3" in detector_results:
            lines.append("\nFigure 3: Best accuracy per model across categories:")
            for model in MAIN_EVAL_MODELS:
                best_acc = max(
                    (cat_results.get(model, 0.0)
                     for cat_results in detector_results["figure3"].values()),
                    default=0.0,
                )
                lines.append(f"  {model}: {best_acc * 100:.1f}%")

    if simulation_results:
        lines.append("\n--- Section 3: Adversarial Voting Simulation ---")

        if "table4" in simulation_results:
            lines.append("\nTable 4: High-ranked model attack (votes required):")
            for model, rank_results in simulation_results["table4"].items():
                min_votes = min(
                    v["votes"] for v in rank_results.values()
                    if v["votes"] != float("inf")
                ) if rank_results else float("inf")
                lines.append(f"  {model}: min {min_votes:.0f} votes to move 1 position")

        if "table5" in simulation_results:
            lines.append("\nTable 5: Low-ranked model attack (votes required):")
            for model, rank_results in simulation_results["table5"].items():
                min_votes = min(
                    v["votes"] for v in rank_results.values()
                    if v["votes"] != float("inf")
                ) if rank_results else float("inf")
                lines.append(f"  {model}: min {min_votes:.0f} votes to move 1 position")

    if mitigation_results:
        lines.append("\n--- Section 4: Mitigation Results ---")

        if "cost_model" in mitigation_results:
            cm = mitigation_results["cost_model"]
            lines.append(f"\nAttack cost without mitigations: ${cm['no_mitigation']['total']:.2f}")
            lines.append(f"Attack cost with rate limiting: ${cm['rate_limited']['total']:.2f}")
            lines.append(f"Attack cost with CAPTCHA: ${cm['captcha']['total']:.2f}")

        if "figures5_6" in mitigation_results:
            lines.append("\nPerturbed leaderboard defense:")
            for noise_scale, metrics in sorted(
                mitigation_results["figures5_6"].items(), key=lambda x: float(x[0])
            ):
                lines.append(
                    f"  Noise={noise_scale}: "
                    f"TPR={metrics['tpr']:.2f}, "
                    f"FPR={metrics['fpr']:.2f}, "
                    f"Rank change={metrics['mean_rank_change']:.1f}"
                )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
