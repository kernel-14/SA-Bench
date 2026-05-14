# evaluation.py

from typing import List, Dict, Any
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from utils import log_results


class Evaluation:
    """
    Handles model evaluation on test datasets, including the computation of
    key metrics like Accuracy@t1, Accuracy@t2, delta metrics, and edit distances.
    """

    def __init__(self, model: Any, test_dataset: torch.utils.data.Dataset, config: Dict[str, Any]):
        """
        Initializes the Evaluation class with model, test data, and config.

        Args:
            model (Any): A trained model instance of the `Model` class.
            test_dataset (torch.utils.data.Dataset): The test dataset to evaluate with.
            config (Dict[str, Any]): Configuration dictionary from `config.yaml`.
        """
        self.model = model
        self.test_dataset = test_dataset
        self.config = config

        # Load evaluation-related configurations
        self.batch_size = config.get("training", {}).get("stage2", {}).get("batch_size", 128)
        self.metrics_config = config.get("evaluation", {}).get("metrics", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.model.eval()  # Set the model to evaluation mode

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the trained model on the test dataset and computes metrics.

        Returns:
            dict: Evaluation metrics, including Accuracy@t1, Accuracy@t2, Delta(t1, t2), i→c, c→i.
        """
        dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Containers for metrics calculation
        first_turn_predictions = []
        second_turn_predictions = []
        ground_truths = []
        edit_distances = []

        for batch in dataloader:
            inputs_1, attention_mask_1, inputs_2, attention_mask_2, labels = self._prepare_batch(batch)

            with torch.no_grad():
                # Forward pass for first attempt (Turn 1)
                first_turn_outputs = self.model.model(
                    input_ids=inputs_1,
                    attention_mask=attention_mask_1
                )
                first_turn_preds = torch.argmax(first_turn_outputs.logits, dim=-1)

                # Forward pass for second attempt (Turn 2)
                second_turn_outputs = self.model.model(
                    input_ids=inputs_2,
                    attention_mask=attention_mask_2
                )
                second_turn_preds = torch.argmax(second_turn_outputs.logits, dim=-1)

            # Update predictions and ground truth labels
            first_turn_predictions.extend(first_turn_preds.cpu().tolist())
            second_turn_predictions.extend(second_turn_preds.cpu().tolist())
            ground_truths.extend(labels.cpu().tolist())

            # Compute and store normalized edit distances (for qualitative analysis)
            for ft_pred, st_pred in zip(first_turn_preds.cpu().tolist(), second_turn_preds.cpu().tolist()):
                edit_distances.append(self._normalized_edit_distance(ft_pred, st_pred))

        # Compute evaluation metrics
        metrics = self.compute_metrics(
            predictions={
                "t1": first_turn_predictions,
                "t2": second_turn_predictions
            },
            ground_truth=ground_truths
        )
        metrics["edit_distances"] = np.mean(edit_distances)

        # Log results and return metrics
        log_results(metrics, self.config["logging"]["log_path"])
        return metrics

    def compute_metrics(self, predictions: Dict[str, List[int]], ground_truth: List[int]) -> Dict[str, Any]:
        """
        Computes evaluation metrics (Acc@t1, Acc@t2, Δ(t1, t2), i→c, c→i).

        Args:
            predictions (Dict[str, List[int]]): Dictionary of predictions with keys 't1' and 't2'.
            ground_truth (List[int]): Ground-truth labels.

        Returns:
            dict: Evaluation metrics calculated from the predictions and ground truth.
        """
        correct_t1 = 0
        correct_t2 = 0
        incorrect_to_correct = 0
        correct_to_incorrect = 0
        total_samples = len(ground_truth)

        for t1_pred, t2_pred, gt in zip(predictions["t1"], predictions["t2"], ground_truth):
            t1_correct = t1_pred == gt
            t2_correct = t2_pred == gt

            # Accumulate correct predictions for first and second turn
            correct_t1 += int(t1_correct)
            correct_t2 += int(t2_correct)

            # Count transition cases
            if not t1_correct and t2_correct:
                incorrect_to_correct += 1
            if t1_correct and not t2_correct:
                correct_to_incorrect += 1

        # Calculate metrics
        accuracy_t1 = correct_t1 / total_samples
        accuracy_t2 = correct_t2 / total_samples
        delta_t1_t2 = accuracy_t2 - accuracy_t1
        i_to_c = incorrect_to_correct / total_samples
        c_to_i = correct_to_incorrect / total_samples

        # Return aggregated metrics
        return {
            "Accuracy@t1": accuracy_t1,
            "Accuracy@t2": accuracy_t2,
            "Δ(t1, t2)": delta_t1_t2,
            "i→c": i_to_c,
            "c→i": c_to_i
        }

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Prepares input tensors for evaluation and moves them to the appropriate device.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of tokenized sequences.

        Returns:
            List[torch.Tensor]: Prepared tensors for first turn, second turn, and ground truth labels.
        """
        inputs_1 = batch['input_ids_1'].to(self.device)
        attention_mask_1 = batch['attention_mask_1'].to(self.device)
        inputs_2 = batch['input_ids_2'].to(self.device)
        attention_mask_2 = batch['attention_mask_2'].to(self.device)
        labels = batch['labels'].to(self.device)
        return inputs_1, attention_mask_1, inputs_2, attention_mask_2, labels

    def _normalized_edit_distance(self, seq1: List[int], seq2: List[int]) -> float:
        """
        Computes the normalized edit distance between two sequences.

        Args:
            seq1 (List[int]): First sequence (e.g., predictions from Turn 1).
            seq2 (List[int]): Second sequence (e.g., predictions from Turn 2).

        Returns:
            float: Normalized edit distance between both sequences.
        """
        len_1 = len(seq1)
        len_2 = len(seq2)
        max_len = max(len_1, len_2)
        if max_len == 0:
            return 0.0  # Avoid division by zero for empty sequences

        edit_distance = sum(x != y for x, y in zip(seq1, seq2)) + abs(len_1 - len_2)
        return edit_distance / max_len
