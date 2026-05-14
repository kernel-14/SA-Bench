# evaluation.py

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class Evaluation:
    """
    Evaluation class to compute metrics and generate reports for model evaluations.
    Includes mechanisms for classifier accuracy and leaderboard Elo dynamics analysis.
    """

    def __init__(self, models: List[str], predictions: np.ndarray, truth: np.ndarray, 
                 elo_scores: Optional[Dict[str, float]] = None, config: Optional[Dict] = None) -> None:
        """
        Initializes the Evaluation object with necessary inputs.

        Args:
            models (List[str]): List of model names.
            predictions (np.ndarray): Predicted labels from the classifier.
            truth (np.ndarray): Ground truth labels for evaluation.
            elo_scores (Optional[Dict[str, float]]): Final Elo scores of models after simulation.
            config (Optional[Dict]): Configuration dictionary from the config.yaml.
        """
        self.models = models
        self.predictions = predictions
        self.truth = truth
        self.elo_scores = elo_scores
        self.config = config or {}
        self.evaluation_metrics = self.config.get("evaluation", {}).get("metrics", [])
        self.plot_dir = self.config.get("evaluation", {}).get("result_plots_dir", "results/")

        # Ensure the plotting directory exists
        os.makedirs(self.plot_dir, exist_ok=True)

    def compute_accuracy(self) -> float:
        """
        Computes the accuracy of the predictions compared to the ground truth.

        Returns:
            float: Accuracy value (correct predictions / total predictions).
        """
        if self.predictions.shape != self.truth.shape:
            raise ValueError("Shape mismatch between predictions and truth arrays.")

        correct = np.sum(self.predictions == self.truth)
        total = len(self.truth)
        accuracy = correct / total
        return round(accuracy, 2)

    def compute_precision_and_recall(self, positive_class: int = 1) -> Dict[str, float]:
        """
        Computes precision and recall based on the predictions and truth.

        Args:
            positive_class (int): Label value representing the positive class.

        Returns:
            Dict[str, float]: Dictionary with keys 'precision' and 'recall'.
        """
        tp = np.sum((self.predictions == positive_class) & (self.truth == positive_class))
        fp = np.sum((self.predictions == positive_class) & (self.truth != positive_class))
        fn = np.sum((self.predictions != positive_class) & (self.truth == positive_class))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {"precision": round(precision, 2), "recall": round(recall, 2)}

    def generate_elo_report(self, prior_elos: Optional[Dict[str, float]] = None) -> None:
        """
        Generates and saves a bar plot comparing Elo scores pre- and post-simulation.

        Args:
            prior_elos (Optional[Dict[str, float]]): Elo scores before simulation, for comparison.
        """
        if not self.elo_scores:
            print("No Elo scores available. Skipping Elo report generation.")
            return

        # Sorting models by post-simulation Elo scores
        sorted_models = sorted(self.elo_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_model_names = [model[0] for model in sorted_models]
        sorted_elo_scores = [model[1] for model in sorted_models]

        prior_sorted_elo_scores = None
        if prior_elos:
            # Map prior Elo scores to the same model order
            prior_sorted_elo_scores = [prior_elos[model] for model in sorted_model_names]

        # Plot the Elo scores
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        index = np.arange(len(sorted_model_names))

        if prior_sorted_elo_scores:
            # Bar plot with two sets of bars for comparison
            plt.bar(index, prior_sorted_elo_scores, bar_width, label="Pre-Simulation")
            plt.bar(index + bar_width, sorted_elo_scores, bar_width, label="Post-Simulation")
        else:
            # Plot only post-simulation scores if no prior scores are available
            plt.bar(index, sorted_elo_scores, bar_width, label="Post-Simulation")

        plt.xlabel("Models")
        plt.ylabel("Elo Scores")
        plt.title("Elo Score Comparison")
        plt.xticks(index + (bar_width / 2 if prior_sorted_elo_scores else 0), 
                   sorted_model_names, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(self.plot_dir, "elo_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Elo comparison plot saved at: {plot_path}")

    def generate_report(self) -> None:
        """
        Consolidates evaluation metrics and Elo dynamics into a unified report.

        Saves the textual evaluation report and any associated plots.
        """
        report_content = {}

        # Compute metrics
        if "accuracy" in self.evaluation_metrics:
            report_content["accuracy"] = self.compute_accuracy()
        if "precision" in self.evaluation_metrics or "recall" in self.evaluation_metrics:
            pr_metrics = self.compute_precision_and_recall()
            report_content.update(pr_metrics)

        # Generate plots (if Elo scores are available)
        if self.elo_scores:
            prior_elos = self.config.get("simulation", {}).get("prior_elos", None)
            self.generate_elo_report(prior_elos=prior_elos)

        # Save the textual report
        report_path = os.path.join(self.plot_dir, "evaluation_report.json")
        try:
            import json
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_content, f, indent=4)
            print(f"Evaluation report successfully saved at: {report_path}")
        except Exception as e:
            print(f"Failed to save evaluation report: {e}")
