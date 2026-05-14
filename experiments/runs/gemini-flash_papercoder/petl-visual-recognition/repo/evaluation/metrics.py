## evaluation/metrics.py

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter


def calculate_top1_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes the Top-1 classification accuracy.

    Args:
        predictions (torch.Tensor): A tensor of logits (or probabilities) with shape
                                    (batch_size, num_classes).
        targets (torch.Tensor): A tensor of ground truth class indices with shape
                                (batch_size,).

    Returns:
        float: The calculated Top-1 accuracy as a percentage.

    Raises:
        ValueError: If predictions or targets are empty or have incompatible shapes.
    """
    if predictions.numel() == 0 or targets.numel() == 0:
        raise ValueError("Predictions or targets tensor is empty.")
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError(f"Batch sizes of predictions ({predictions.shape[0]}) and "
                         f"targets ({targets.shape[0]}) do not match.")

    # Get predicted classes by finding the index with the maximum logit/probability
    predicted_classes = predictions.argmax(dim=1)

    # Compare with ground truth and count correct predictions
    correct_predictions = (predicted_classes == targets).sum().item()

    # Calculate accuracy as a percentage
    accuracy = (correct_predictions / targets.size(0)) * 100.0

    return accuracy


def calculate_prediction_overlaps(
    predictions_dict: Dict[str, torch.Tensor],
    confidences_dict: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    topk_confident: Optional[int] = None,
    leastk_confident: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculates the percentage of samples for which pairs of models make the same prediction.
    It supports filtering for the k most confident correct predictions or k least confident
    wrong predictions.

    Args:
        predictions_dict (Dict[str, torch.Tensor]): A dictionary where keys are model names
                                                    and values are tensors of predicted class IDs
                                                    (shape: num_samples,).
        confidences_dict (Dict[str, torch.Tensor]): A dictionary where keys are model names
                                                    and values are tensors of confidence scores
                                                    (e.g., max softmax probability, shape: num_samples,).
        targets (torch.Tensor): A tensor of ground truth class indices with shape (num_samples,).
        topk_confident (Optional[int]): If provided, only considers the top K most confident
                                        correct predictions for overlap analysis.
        leastk_confident (Optional[int]): If provided, only considers the least K confident
                                          wrong predictions for overlap analysis.

    Returns:
        Dict[str, Any]: A dictionary containing:
                        - 'overall_overlap_matrix': Pairwise overlap percentages for all samples.
                        - 'topk_confident_predictions_indices': Dictionary of indices for top K confident correct predictions per model.
                        - 'leastk_confident_predictions_indices': Dictionary of indices for least K confident wrong predictions per model.

    Raises:
        ValueError: If inputs are empty, inconsistent, or `topk_confident` or `leastk_confident`
                    are non-positive.
    """
    if not predictions_dict or not confidences_dict:
        raise ValueError("predictions_dict or confidences_dict cannot be empty.")
    if not targets.numel():
        raise ValueError("Targets tensor is empty.")

    model_names = list(predictions_dict.keys())
    if not model_names:
        return {} # No models to compare
    
    num_samples = targets.size(0)
    for model_name in model_names:
        if predictions_dict[model_name].shape[0] != num_samples or confidences_dict[model_name].shape[0] != num_samples:
            raise ValueError(f"Inconsistent number of samples for model '{model_name}'. Expected {num_samples}.")

    results: Dict[str, Any] = {}

    # --- 1. Calculate Overall Overlap (Figure 3a Style) ---
    overall_overlap_matrix: Dict[str, float] = {}
    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            pred_i = predictions_dict[model_i]
            pred_j = predictions_dict[model_j]

            overlap_count = (pred_i == pred_j).sum().item()
            percentage = (overlap_count / num_samples) * 100.0
            overall_overlap_matrix[f'{model_i}_vs_{model_j}'] = percentage
    results['overall_overlap_matrix'] = overall_overlap_matrix

    # --- 2. Identify Top K Confident Correct Predictions (for Figure 1b Venn Diagram Data) ---
    topk_confident_predictions_indices: Dict[str, List[int]] = {}
    if topk_confident is not None and topk_confident > 0:
        if topk_confident > num_samples:
            topk_confident = num_samples # Cap at total samples
            
        for model_name in model_names:
            pred = predictions_dict[model_name]
            conf = confidences_dict[model_name]

            is_correct = (pred == targets)

            # Sort by confidence in descending order
            sorted_conf_indices = torch.argsort(conf, descending=True)
            
            # Filter for correct predictions among the most confident
            correct_and_confident_indices = sorted_conf_indices[is_correct[sorted_conf_indices]]
            
            # Take up to topk_confident of these
            topk_confident_predictions_indices[model_name] = correct_and_confident_indices[:topk_confident].tolist()
            
    results['topk_confident_predictions_indices'] = topk_confident_predictions_indices

    # --- 3. Identify Least K Confident Wrong Predictions (for Figure 3b Venn Diagram Data) ---
    leastk_confident_predictions_indices: Dict[str, List[int]] = {}
    if leastk_confident is not None and leastk_confident > 0:
        if leastk_confident > num_samples:
            leastk_confident = num_samples # Cap at total samples

        for model_name in model_names:
            pred = predictions_dict[model_name]
            conf = confidences_dict[model_name]

            is_wrong = (pred != targets)

            # Sort by confidence in ascending order (least confident first)
            sorted_conf_indices = torch.argsort(conf, descending=False)

            # Filter for wrong predictions among the least confident
            wrong_and_least_confident_indices = sorted_conf_indices[is_wrong[sorted_conf_indices]]

            # Take up to leastk_confident of these
            leastk_confident_predictions_indices[model_name] = wrong_and_least_confident_indices[:leastk_confident].tolist()
            
    results['leastk_confident_predictions_indices'] = leastk_confident_predictions_indices

    return results


def calculate_ensemble_predictions(
    logits_list: List[torch.Tensor],
    ensemble_type: str = 'average_logits'
) -> torch.Tensor:
    """
    Combines the predictions from multiple models into a single ensemble prediction.

    Args:
        logits_list (List[torch.Tensor]): A list of tensors, where each tensor
                                          contains logits from one model for all samples.
                                          Shape of each tensor: (num_samples, num_classes).
        ensemble_type (str): The method to use for ensembling.
                             'average_logits': Average the logits across models and then take argmax.
                             'majority_vote': Take the majority vote of predicted classes for each sample.

    Returns:
        torch.Tensor: A tensor of shape (num_samples,) representing the final ensemble
                      predictions (class IDs).

    Raises:
        ValueError: If logits_list is empty, tensors have inconsistent shapes,
                    or an unsupported ensemble_type is provided.
    """
    if not logits_list:
        raise ValueError("logits_list cannot be empty.")

    num_models = len(logits_list)
    num_samples = logits_list[0].shape[0]
    num_classes = logits_list[0].shape[1]

    for i, logits in enumerate(logits_list):
        if logits.shape[0] != num_samples or logits.shape[1] != num_classes:
            raise ValueError(f"Logits tensor at index {i} has inconsistent shape. "
                             f"Expected ({num_samples}, {num_classes}), got {logits.shape}.")

    if ensemble_type == 'average_logits':
        # Stack all logits: (num_models, num_samples, num_classes)
        stacked_logits = torch.stack(logits_list, dim=0)
        # Average across models: (num_samples, num_classes)
        averaged_logits = torch.mean(stacked_logits, dim=0)
        # Get ensemble predictions
        ensemble_predictions = averaged_logits.argmax(dim=1)
    elif ensemble_type == 'majority_vote':
        # Get predicted classes from each model: (num_samples,) for each model
        model_predictions = [logits.argmax(dim=1) for logits in logits_list]
        # Stack predictions: (num_models, num_samples)
        stacked_predictions = torch.stack(model_predictions, dim=0)

        ensemble_predictions = torch.empty(num_samples, dtype=torch.long, device=stacked_predictions.device)

        for i in range(num_samples):
            # Get votes for the current sample: (num_models,)
            sample_votes = stacked_predictions[:, i]
            
            # Count occurrences of each class ID
            # Counter works with lists, convert tensor to list.
            vote_counts = Counter(sample_votes.tolist())

            if not vote_counts: # Should not happen if num_models > 0
                raise RuntimeError(f"No votes recorded for sample {i}. This indicates an unexpected error.")

            # Find the max vote count
            max_votes = 0
            for count in vote_counts.values():
                if count > max_votes:
                    max_votes = count
            
            # Identify all classes that received the max vote count (for tie-breaking)
            majority_classes = [class_id for class_id, count in vote_counts.items() if count == max_votes]

            # Tie-breaking: choose the class with the smallest ID
            ensemble_predictions[i] = min(majority_classes)
    else:
        raise ValueError(f"Unsupported ensemble_type: {ensemble_type}. "
                         "Expected 'average_logits' or 'majority_vote'.")

    return ensemble_predictions

