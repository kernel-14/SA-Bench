"""
Evaluation utilities for PEFT methods.
Includes accuracy evaluation, ensemble methods, and prediction similarity analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


def evaluate(model, data_loader, device='cuda', return_predictions=False):
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to use
        return_predictions: If True, return predictions and confidences
    
    Returns:
        accuracy (float), optionally (predictions, confidences, labels)
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_confidences = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    
    if return_predictions:
        return accuracy, np.array(all_predictions), np.array(all_confidences), np.array(all_labels)
    return accuracy


def evaluate_ensemble(models, data_loader, device='cuda', method='avg_logits'):
    """
    Evaluate ensemble of models.
    
    Args:
        models: List of models
        data_loader: DataLoader for evaluation
        device: Device to use
        method: 'avg_logits' (average logits) or 'majority_vote'
    
    Returns:
        accuracy (float)
    """
    for model in models:
        model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if method == 'avg_logits':
                # Average logits across models
                all_logits = []
                for model in models:
                    logits = model(inputs)
                    all_logits.append(logits)
                avg_logits = torch.stack(all_logits).mean(0)
                _, predicted = avg_logits.max(1)
            
            elif method == 'majority_vote':
                # Majority vote across models
                all_preds = []
                for model in models:
                    logits = model(inputs)
                    _, preds = logits.max(1)
                    all_preds.append(preds)
                
                # Stack and take majority vote
                all_preds = torch.stack(all_preds)  # (num_models, batch_size)
                predicted = torch.mode(all_preds, dim=0).values
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / total


def compute_prediction_similarity(predictions_dict):
    """
    Compute pairwise prediction similarity between methods.
    
    Args:
        predictions_dict: Dict mapping method names to prediction arrays
    
    Returns:
        similarity_matrix: Dict of dicts with similarity percentages
    """
    methods = list(predictions_dict.keys())
    n = len(methods)
    
    similarity_matrix = {}
    for i, method_i in enumerate(methods):
        similarity_matrix[method_i] = {}
        for j, method_j in enumerate(methods):
            preds_i = predictions_dict[method_i]
            preds_j = predictions_dict[method_j]
            
            # Percentage of samples with same prediction
            similarity = 100.0 * np.mean(preds_i == preds_j)
            similarity_matrix[method_i][method_j] = similarity
    
    return similarity_matrix


def get_confident_predictions(predictions, confidences, labels, k=5000, correct_only=True):
    """
    Get the k most/least confident predictions.
    
    Args:
        predictions: Array of predictions
        confidences: Array of confidence scores
        labels: Array of true labels
        k: Number of samples to select
        correct_only: If True, select from correct predictions (most confident)
                     If False, select from wrong predictions (least confident)
    
    Returns:
        indices of selected samples
    """
    if correct_only:
        # Most confident correct predictions
        correct_mask = predictions == labels
        correct_indices = np.where(correct_mask)[0]
        correct_confidences = confidences[correct_indices]
        
        # Sort by confidence (descending)
        sorted_idx = np.argsort(-correct_confidences)
        selected = correct_indices[sorted_idx[:k]]
    else:
        # Least confident wrong predictions
        wrong_mask = predictions != labels
        wrong_indices = np.where(wrong_mask)[0]
        wrong_confidences = confidences[wrong_indices]
        
        # Sort by confidence (ascending)
        sorted_idx = np.argsort(wrong_confidences)
        selected = wrong_indices[sorted_idx[:k]]
    
    return selected


def compute_prediction_overlap(predictions_list, indices_list):
    """
    Compute prediction overlap between methods for specific samples.
    
    Args:
        predictions_list: List of prediction arrays
        indices_list: List of index arrays (one per method)
    
    Returns:
        overlap counts for Venn diagram
    """
    n_methods = len(predictions_list)
    
    # Convert to sets of indices
    index_sets = [set(idx) for idx in indices_list]
    
    # Compute overlaps
    overlaps = {}
    
    # Individual
    for i, idx_set in enumerate(index_sets):
        overlaps[f'method_{i}'] = len(idx_set)
    
    # Pairwise
    for i in range(n_methods):
        for j in range(i+1, n_methods):
            overlap = len(index_sets[i] & index_sets[j])
            overlaps[f'method_{i}_method_{j}'] = overlap
    
    # All methods
    all_overlap = len(set.intersection(*index_sets))
    overlaps['all'] = all_overlap
    
    return overlaps


def compute_ranking_frequency(results_dict, methods, datasets):
    """
    Compute ranking frequency matrix for methods across datasets.
    
    Args:
        results_dict: Dict mapping (method, dataset) to accuracy
        methods: List of method names
        datasets: List of dataset names
    
    Returns:
        ranking_matrix: numpy array of shape (n_methods, n_methods)
                       where element (i, j) is frequency method i ranks j-th
    """
    n_methods = len(methods)
    ranking_matrix = np.zeros((n_methods, n_methods), dtype=int)
    
    for dataset in datasets:
        # Get accuracies for this dataset
        accs = []
        for method in methods:
            acc = results_dict.get((method, dataset), 0.0)
            accs.append(acc)
        
        # Rank methods (higher accuracy = lower rank number)
        sorted_indices = np.argsort(-np.array(accs))
        for rank, method_idx in enumerate(sorted_indices):
            ranking_matrix[method_idx, rank] += 1
    
    return ranking_matrix
