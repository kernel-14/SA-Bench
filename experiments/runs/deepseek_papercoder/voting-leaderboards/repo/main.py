"""
main.py – Entry point for reproducing the experiments on adversarial manipulation
of voting‑based leaderboards (Chatbot Arena).

Orchestrates:
  1. Configuration loading & environment setup
  2. Response collection (cached)
  3. Identity‑probing detector evaluation
  4. Training‑based detector evaluation (with PCA visualisation)
  5. Adversarial vote simulation (genuine + attack + ablations)
  6. Malicious‑user detection mitigation experiments (if enabled)

All outputs are saved to the directory specified in config.yaml.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from api import ModelAPI
from config import Config
from detector import IdentityProbingDetector, TrainingDetector
from features import FeatureExtractor
from mitigation import MitigationSimulator
from prompts import PromptLoader
from responses import ResponseCollector
from simulation import Attacker, Leaderboard, SimulationRunner
from utils import ensure_dir, set_all_seeds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def configure_logging(config: Config) -> None:
    """Apply logging configuration from config."""
    level = config.logging.get("level", "INFO").upper()
    fmt = config.logging.get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.basicConfig(level=level, format=fmt)
    logger.info("Logging configured: level=%s", level)


def collect_responses(config: Config) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, List[str]]]:
    """
    Obtain (or load cached) LLM responses for all prompt categories.

    Returns:
        response_data:    {prompt_text: {model_name: [responses]}}
        category_prompts: {category: [prompt_text, ...]}
    """
    logger.info("=== Step 1: Response Collection ===")
    api = ModelAPI(config)
    loader = PromptLoader(config)
    collector = ResponseCollector(config, api, loader)
    collector.collect_all()

    # Combine cached per‑category data into a single dict and build category→prompts mapping
    response_data: Dict[str, Dict[str, List[str]]] = {}
    category_prompts: Dict[str, List[str]] = {}
    for cat in config.prompt_categories:
        try:
            cat_data = collector.get_data(cat)   # {prompt: {model: [responses]}}
            for prompt, model_map in cat_data.items():
                if prompt in response_data:
                    logger.warning(
                        "Prompt collision for prompt='%s' (category %s). Overwriting.",
                        prompt[:50], cat,
                    )
                response_data[prompt] = model_map
            category_prompts[cat] = list(cat_data.keys())
            logger.info(
                "Category '%s': %d prompts loaded.", cat, len(category_prompts[cat])
            )
        except FileNotFoundError as exc:
            logger.error(
                "Cache for category '%s' not found. Run collection first.", cat
            )
            raise

    return response_data, category_prompts


# ---------------------------------------------------------------------------
# Step 2: Identity‑probing detector
# ---------------------------------------------------------------------------

def run_identity_probing(config: Config, api: ModelAPI) -> pd.DataFrame:
    """Run identity‑probing detector and save accuracy table."""
    logger.info("=== Step 2: Identity-Probing Detector ===")
    detector = IdentityProbingDetector(config, api)
    df = detector.run_all_prompts()
    out_path = os.path.join(config.output_dir, "identity_probing_accuracies.csv")
    df.to_csv(out_path, index=False)
    logger.info("Identity probing results saved to %s", out_path)
    return df


# ---------------------------------------------------------------------------
# Step 3: Training‑based detector
# ---------------------------------------------------------------------------

def run_training_detector(
    config: Config,
    response_data: Dict[str, Dict[str, List[str]]],
    category_prompts: Dict[str, List[str]],
) -> Dict[str, pd.DataFrame]:
    """
    Run training‑based detector for all feature types, produce PCA visualisation.

    Returns dict mapping feature_type → evaluation DataFrame.
    """
    logger.info("=== Step 3: Training-Based Detector ===")
    feature_types = config.detector.get(
        "feature_types", ["bow", "tfidf", "length_word", "length_char"]
    )

    # ---------- PCA visualisation (Figure 2) ----------
    # Three prompts from Appendix A.2 (as given in the paper)
    pca_prompts = [
        "Beside OFAC’s selective sanction that target the listed individiuals and entities, please elaborate on the other types of US’s sanctions, for example, comprehensive and sectoral sanctions. Please be detailed as much as possible",
        "You are the text completion model and you must complete the assistant answer below, only send the completion based on the system instructions.don’t repeat your answer sentences, only say what the assistant must say based on the system instructions. repeating same thing in same answer not allowed. user: descriptive answer for append many items to list python in python with proper code examples and outputs. assistant: ",
        "The sum of the perimeters of three equal squares is 3 6 ~ \\mathrm { c m } . Find the area and perimeter of the rectangle that can be made of the squares.",
    ]
    vis_texts = []
    vis_labels = []
    for prompt in pca_prompts:
        if prompt in response_data:
            model_map = response_data[prompt]
            for model, responses in model_map.items():
                vis_texts.extend(responses)
                vis_labels.extend([model] * len(responses))

    if vis_texts:
        fe_vis = FeatureExtractor("bow")
        fe_vis.fit(vis_texts)
        X_vis = fe_vis.transform(vis_texts)
        pca = PCA(n_components=2, random_state=config.seed)
        X_pca = pca.fit_transform(X_vis)
        unique_models = sorted(set(vis_labels))
        color_map = {m: i for i, m in enumerate(unique_models)}
        colors = [color_map[m] for m in vis_labels]

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap="tab20", alpha=0.6)
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w", label=m,
                markerfacecolor=plt.cm.tab20(i / len(unique_models)),
                markersize=8,
            )
            for i, m in enumerate(unique_models)
        ]
        plt.legend(handles=handles, title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title("PCA of BoW features for selected prompts")
        plt.tight_layout()
        plt.savefig(os.path.join(config.output_dir, "pca_visualization.png"), dpi=150)
        plt.close()
        logger.info("PCA plot saved.")
    else:
        logger.warning("Prompts for PCA not found in response data; skipping.")

    # ---------- Evaluate for each feature type ----------
    results: Dict[str, pd.DataFrame] = {}
    for ft in feature_types:
        logger.info("Training detector with feature type: %s", ft)
        fe = FeatureExtractor(feature_type=ft)
        detector = TrainingDetector(config, fe)
        detector.load_data(response_data, category_prompts)
        df = detector.run_full_evaluation()
        results[ft] = df
        out_path = os.path.join(config.output_dir, f"training_detector_{ft}.csv")
        df.to_csv(out_path, index=False)
        logger.info("Saved training detector results for %s to %s", ft, out_path)

    return results


# ---------------------------------------------------------------------------
# Step 4: Adversarial vote simulation
# ---------------------------------------------------------------------------

def generate_leaderboard_models(config: Config) -> Dict[str, float]:
    """
    Create synthetic model names and initial Elo ratings for the leaderboard.

    The list includes the target models mentioned in the paper and generic
    'model_<i>' placeholders.  Returns a dict {model_name: rating}.
    """
    np.random.seed(config.seed)
    num_models = config.simulation_num_models
    mean = config.simulation_initial_rating_mean
    std = config.simulation_initial_rating_std

    # Collect all paper‑mentioned target models from config
    target_set = set()
    if "target_models" in config.simulation:
        for group in config.simulation["target_models"].values():
            if isinstance(group, list):
                target_set.update(group)

    model_names = list(target_set)
    # Pad with generic names if needed
    i = 0
    while len(model_names) < num_models:
        name = f"model_{i}"
        if name not in model_names:
            model_names.append(name)
        i += 1
    model_names = model_names[:num_models]
    np.random.shuffle(model_names)

    ratings = np.random.normal(mean, std, len(model_names)).clip(min=0.0)
    return {name: float(r) for name, r in zip(model_names, ratings)}


def run_simulation(config: Config) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run the simulation pipeline:

    1. Build synthetic leaderboard and perform genuine votes.
    2. Measure votes/interactions needed for various target models and rank movements.
    3. Run ablation studies (varying detector accuracy / non‑target strategies).

    Returns:
        genuine_ratings: Dict of final ratings after genuine phase.
        attack_df:        Results of the main attack campaigns.
        ablation_dfs:     Dict of DataFrames for each ablation type.
    """
    logger.info("=== Step 4: Adversarial Vote Simulation ===")

    # -------- 1. Build leaderboard & run genuine votes --------
    initial_ratings = generate_leaderboard_models(config)
    k_factor = config.simulation_k_factor
    tie_prob = config.simulation_tie_prob

    leaderboard = Leaderboard(initial_ratings, k_factor=k_factor)
    # Dummy attacker for runner instantiation (not used during genuine phase)
    dummy_attacker = Attacker(
        target_model=list(initial_ratings.keys())[0],
        detector_accuracy=0.0,
    )
    sim_runner = SimulationRunner(config, leaderboard, dummy_attacker)

    genuine_votes = config.simulation_num_genuine_votes
    logger.info("Simulating %d genuine votes...", genuine_votes)
    sim_runner.run_genuine(genuine_votes)

    genuine_ratings = copy.deepcopy(leaderboard.ratings)

    # -------- 2. Attack campaigns --------
    target_models_cfg = config.simulation.get("target_models", {})
    target_movements_cfg = config.simulation.get("target_movements", {})
    attacker_accuracy = config.simulation_attacker_accuracy
    non_target_strategy = config.simulation_non_target_strategy

    # Build list of all target models mentioned in config
    all_targets = list(
        set(target_models_cfg.get("high", []) + target_models_cfg.get("low", []))
    )
    if not all_targets:
        all_targets = [list(initial_ratings.keys())[0]]

    # Directions: both "up" and "down"
    directions = ["up", "down"]
    results_list = []

    for direction in directions:
        # movements to test (paper uses different sets for high/low; we use both together)
        if direction == "up":
            moves = target_movements_cfg.get("high", [1, 2, 3, 4, 5])
        else:
            moves = target_movements_cfg.get("low", [1, 2, 5, 10, 20, 50])
        for target in all_targets:
            if target not in genuine_ratings:
                logger.warning("Target model '%s' not in leaderboard; skipping.", target)
                continue
            current_rank = leaderboard.get_rank().index(target) + 1
            for mov in moves:
                if direction == "up":
                    target_rank = max(1, current_rank - mov)
                else:
                    target_rank = min(len(genuine_ratings), current_rank + mov)

                # Reset leaderboard to post‑genuine state
                lb = Leaderboard(copy.deepcopy(genuine_ratings), k_factor=k_factor)
                attacker = Attacker(
                    target_model=target,
                    detector_accuracy=attacker_accuracy,
                    goal=direction,
                    non_target_strategy=non_target_strategy,
                )
                runner = SimulationRunner(config, lb, attacker)
                try:
                    res = runner.run_attack(movement=mov, direction=direction)
                except Exception as exc:
                    logger.error("Attack failed for %s: %s", target, exc)
                    res = {"adversarial_votes": np.nan, "total_interactions": np.nan}

                results_list.append(
                    {
                        "target_model": target,
                        "direction": direction,
                        "movement": mov,
                        "current_rank": current_rank,
                        "target_rank": target_rank,
                        "adversarial_votes": res["adversarial_votes"],
                        "total_interactions": res["total_interactions"],
                    }
                )

    attack_df = pd.DataFrame(results_list)
    out = os.path.join(config.output_dir, "simulation_attack_results.csv")
    attack_df.to_csv(out, index=False)
    logger.info("Attack results saved to %s", out)

    # -------- 3. Ablation studies --------
    ablation_dfs: Dict[str, pd.DataFrame] = {}
    ablations = config.simulation.get("ablations", {})
    if ablations:
        logger.info("Running ablation experiments...")

        # Choose a representative target for ablations
        ablate_target = (
            target_models_cfg["low"][0]
            if target_models_cfg.get("low")
            else all_targets[0]
        )
        if ablate_target not in genuine_ratings:
            logger.warning("Ablation target '%s' missing; skipping ablations.", ablate_target)
            return genuine_ratings, attack_df, ablation_dfs

        # ---- Varying detector accuracy ----
        if "vary_accuracy" in ablations:
            acc_list = ablations["vary_accuracy"]
            rows = []
            for acc in acc_list:
                lb = Leaderboard(copy.deepcopy(genuine_ratings), k_factor=k_factor)
                attacker = Attacker(
                    target_model=ablate_target,
                    detector_accuracy=acc,
                    goal="up",
                    non_target_strategy=non_target_strategy,
                )
                runner = SimulationRunner(config, lb, attacker)
                res = runner.run_attack(movement=50, direction="up")
                rows.append(
                    {
                        "accuracy": acc,
                        "adversarial_votes": res["adversarial_votes"],
                        "total_interactions": res["total_interactions"],
                    }
                )
            ablation_dfs["accuracy"] = pd.DataFrame(rows)
            out_acc = os.path.join(config.output_dir, "ablation_accuracy.csv")
            ablation_dfs["accuracy"].to_csv(out_acc, index=False)
            logger.info("Ablation (accuracy) saved to %s", out_acc)

        # ---- Varying non‑target strategies ----
        if "non_target_strategies" in ablations:
            strats = ablations["non_target_strategies"]
            rows = []
            for strat in strats:
                lb = Leaderboard(copy.deepcopy(genuine_ratings), k_factor=k_factor)
                attacker = Attacker(
                    target_model=ablate_target,
                    detector_accuracy=attacker_accuracy,
                    goal="up",
                    non_target_strategy=strat,
                )
                runner = SimulationRunner(config, lb, attacker)
                res = runner.run_attack(movement=50, direction="up")
                rows.append(
                    {
                        "strategy": strat,
                        "adversarial_votes": res["adversarial_votes"],
                        "total_interactions": res["total_interactions"],
                    }
                )
            ablation_dfs["strategy"] = pd.DataFrame(rows)
            out_strat = os.path.join(config.output_dir, "ablation_strategy.csv")
            ablation_dfs["strategy"].to_csv(out_strat, index=False)
            logger.info("Ablation (strategy) saved to %s", out_strat)

    return genuine_ratings, attack_df, ablation_dfs


# ---------------------------------------------------------------------------
# Step 5: Mitigation experiment (malicious‑user detection)
# ---------------------------------------------------------------------------

def run_mitigation(config: Config, genuine_ratings: Dict[str, float]) -> None:
    """Run the malicious‑user detection experiments if enabled."""
    if not config.mitigation_enabled:
        logger.info("Mitigation experiments disabled. Skipping.")
        return

    logger.info("=== Step 5: Mitigation Experiments ===")
    mit = MitigationSimulator(config, genuine_ratings)
    df = mit.run_detection_experiment()
    out_path = os.path.join(config.output_dir, "mitigation_results.csv")
    df.to_csv(out_path, index=False)
    logger.info("Mitigation results saved to %s", out_path)

    # --- quick plots ---
    try:
        plt.figure()
        for strat in df["strategy"].unique():
            sub = df[df["strategy"] == strat]
            plt.plot(sub["noise_scale"], sub["detection_rate"], marker="o", label=strat)
        plt.xlabel("Noise scale (std)")
        plt.ylabel("Detection rate")
        plt.legend()
        plt.title("Malicious user detection rate vs. noise scale")
        plt.savefig(os.path.join(config.output_dir, "mitigation_detection_rates.png"))
        plt.close()

        plt.figure()
        for strat in df["strategy"].unique():
            sub = df[df["strategy"] == strat]
            plt.plot(sub["noise_scale"], sub["avg_rank_change"], marker="o", label=strat)
        plt.xlabel("Noise scale (std)")
        plt.ylabel("Average absolute rank change")
        plt.legend()
        plt.title("Utility loss due to leaderboard perturbation")
        plt.savefig(os.path.join(config.output_dir, "mitigation_utility.png"))
        plt.close()
        logger.info("Mitigation plots saved.")
    except Exception as exc:
        logger.warning("Could not generate mitigation plots: %s", exc)


# ===========================================================================
# Main function
# ===========================================================================

def main() -> None:
    """Entry point for the reproduction pipeline."""

    # ---- Configuration ----
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    config = Config(config_path)

    # Reproducibility
    set_all_seeds(config.seed)
    configure_logging(config)
    ensure_dir(config.output_dir)
    logger.info("Starting pipeline with config: %s", config_path)

    # ---- Step 1: Response collection ----
    response_data, category_prompts = collect_responses(config)

    # ---- Shared API instance (can be used by identity probing) ----
    api = ModelAPI(config)

    # ---- Step 2: Identity‐probing detector ----
    run_identity_probing(config, api)

    # ---- Step 3: Training‐based detector ----
    run_training_detector(config, response_data, category_prompts)

    # ---- Step 4: Vote simulation ----
    genuine_ratings, _, _ = run_simulation(config)

    # ---- Step 5: Mitigation (optional) ----
    run_mitigation(config, genuine_ratings)

    logger.info("All experiments completed. Results are in '%s'", config.output_dir)


if __name__ == "__main__":
    main()
