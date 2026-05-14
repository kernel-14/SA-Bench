"""
Visualization of BoW features for model responses (Figure 2 in the paper).

This script plots the first two principal components of bag-of-words (BoW) features
for model responses to three randomly selected English prompts, demonstrating that
responses cluster distinctly by model.

The three prompts used in the paper (Appendix A.2):
1. "Beside OFAC's selective sanction..."
2. "You are the text completion model..."
3. "The sum of the perimeters of three equal squares..."
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# The three prompts used for visualization in the paper (Appendix A.2)
VISUALIZATION_PROMPTS = [
    (
        "Beside OFAC's selective sanction that target the listed individiuals and entities, "
        "please elaborate on the other types of US's sanctions, for example, comprehensive "
        "and sectoral sanctions. Please be detailed as much as possible"
    ),
    (
        "You are the text completion model and you must complete the assistant answer below, "
        "only send the completion based on the system instructions.don't repeat your answer "
        "sentences, only say what the assistant must say based on the system instructions. "
        "repeating same thing in same answer not allowed. user: descriptive answer for append "
        "many items to list python in python with proper code examples and outputs. assistant: "
    ),
    (
        "The sum of the perimeters of three equal squares is 36 cm. "
        "Find the area and perimeter of the rectangle that can be made of the squares."
    ),
]

# Colors for different models
MODEL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939",
]


def load_responses_for_prompts(
    data_dir: str,
    prompts: list,
    category: str = "english",
) -> dict:
    """
    Load responses for specific prompts from the data directory.

    Args:
        data_dir: Path to the data directory
        prompts: List of prompts to load
        category: Prompt category

    Returns:
        Dictionary mapping prompt -> {model_name -> [responses]}
    """
    data_path = Path(data_dir) / category
    results = {p: {} for p in prompts}

    for prompt_dir in sorted(data_path.iterdir()):
        if not prompt_dir.is_dir():
            continue

        prompt_file = prompt_dir / "prompt.txt"
        if not prompt_file.exists():
            continue

        prompt = prompt_file.read_text().strip()

        # Check if this prompt matches any of our target prompts
        for target_prompt in prompts:
            if prompt == target_prompt or prompt[:100] == target_prompt[:100]:
                for response_file in sorted(prompt_dir.glob("*.json")):
                    model_name = response_file.stem
                    with open(response_file) as f:
                        responses = json.load(f)
                    results[target_prompt][model_name] = responses
                break

    return results


def plot_bow_pca(
    prompt_responses: dict,
    prompt_idx: int,
    prompt: str,
    output_dir: str = None,
    ax=None,
):
    """
    Plot the first two PCA components of BoW features for a single prompt.

    Args:
        prompt_responses: Dictionary mapping model_name -> [responses]
        prompt_idx: Index of the prompt (for title)
        prompt: Prompt text
        output_dir: Directory to save the plot
        ax: Matplotlib axes to plot on (if None, creates new figure)
    """
    if not prompt_responses:
        logger.warning(f"No responses found for prompt {prompt_idx + 1}")
        return

    # Collect all responses and labels
    all_responses = []
    all_labels = []
    model_names = sorted(prompt_responses.keys())

    for model_name in model_names:
        responses = prompt_responses[model_name]
        all_responses.extend(responses)
        all_labels.extend([model_name] * len(responses))

    if not all_responses:
        return

    # Extract BoW features
    vectorizer = CountVectorizer(max_features=10000)
    X = vectorizer.fit_transform(all_responses).toarray()

    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True
    else:
        standalone = False

    for i, model_name in enumerate(model_names):
        mask = np.array(all_labels) == model_name
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            label=model_name,
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"Prompt {prompt_idx + 1}: {prompt[:60]}...")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)

    if standalone:
        plt.tight_layout()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_dir, f"bow_pca_prompt{prompt_idx + 1}.png"),
                dpi=150,
                bbox_inches="tight",
            )
        plt.close()


def plot_all_prompts(
    all_prompt_responses: dict,
    prompts: list,
    output_dir: str = None,
):
    """
    Plot BoW PCA for all three prompts in a single figure (Figure 2 in the paper).

    Args:
        all_prompt_responses: Dictionary mapping prompt -> {model_name -> [responses]}
        prompts: List of prompts
        output_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for i, (prompt, ax) in enumerate(zip(prompts, axes)):
        prompt_responses = all_prompt_responses.get(prompt, {})
        plot_bow_pca(prompt_responses, i, prompt, ax=ax)

    plt.suptitle(
        "First Two Principal Components of BoW Features\n"
        "(Responses cluster distinctly by model for each prompt)",
        fontsize=14,
    )
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "figure2_bow_pca.png")
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Figure 2 saved to {output_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BoW features for model responses (Figure 2)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/responses",
        help="Directory containing model responses",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="english",
        help="Prompt category to visualize",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        logger.error(
            f"Data directory {args.data_dir} not found. "
            "Please collect model responses first."
        )
        return

    # Load responses for visualization prompts
    logger.info("Loading responses for visualization...")
    all_prompt_responses = load_responses_for_prompts(
        data_dir=args.data_dir,
        prompts=VISUALIZATION_PROMPTS,
        category=args.category,
    )

    # Check if we have data
    has_data = any(
        bool(responses)
        for responses in all_prompt_responses.values()
    )

    if not has_data:
        logger.warning(
            "No responses found for visualization prompts. "
            "Using first available prompts instead."
        )
        # Fall back to first available prompts
        data_path = Path(args.data_dir) / args.category
        available_prompts = []
        for prompt_dir in sorted(data_path.iterdir())[:3]:
            if prompt_dir.is_dir():
                prompt_file = prompt_dir / "prompt.txt"
                if prompt_file.exists():
                    available_prompts.append(prompt_file.read_text().strip())

        if available_prompts:
            all_prompt_responses = load_responses_for_prompts(
                data_dir=args.data_dir,
                prompts=available_prompts,
                category=args.category,
            )

    # Plot
    logger.info("Generating BoW PCA visualization...")
    prompts_to_plot = [p for p in VISUALIZATION_PROMPTS if all_prompt_responses.get(p)]

    if prompts_to_plot:
        plot_all_prompts(all_prompt_responses, prompts_to_plot, args.output_dir)
    else:
        logger.error("No data available for visualization")


if __name__ == "__main__":
    main()
