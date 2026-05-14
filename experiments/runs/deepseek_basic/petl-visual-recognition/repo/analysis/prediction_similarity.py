"""Prediction similarity analysis between PEFT methods.

Computes prediction overlap matrices and confidence-based analyses
as described in Section 4 of the paper.
"""

import numpy as np
from scipy import stats
from collections import defaultdict


def compute_prediction_similarity(all_predictions, method_names):
    """Compute prediction similarity matrix.
    
    Element (i,j) = percentage of samples where method i and j predict the same.
    
    Args:
        all_predictions: dict mapping method_name -> numpy array of predictions
        method_names: list of method names
    
    Returns:
        similarity_matrix: [M x M] numpy array
    """
    M = len(method_names)
    similarity = np.zeros((M, M))
    
    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            if m1 in all_predictions and m2 in all_predictions:
                preds1 = all_predictions[m1]
                preds2 = all_predictions[m2]
                similarity[i, j] = np.mean(preds1 == preds2)
            else:
                similarity[i, j] = np.nan
    
    return similarity


def compute_confidence_overlap(all_logits, all_predictions, all_targets,
                                method_names, top_k=5000, mode='correct'):
    """Compute prediction overlap for most/least confident samples.
    
    As shown in Figure 1b and Figure 3b:
    - High confidence: correct predictions from top K most confident
    - Low confidence: wrong predictions from top K least confident
    
    Args:
        all_logits: dict mapping method_name -> [N, C] logits
        all_predictions: dict mapping method_name -> [N] predictions
        all_targets: [N] ground truth labels
        method_names: list of method names
        top_k: number of samples to consider
        mode: 'correct' for high confidence, 'wrong' for low confidence
    
    Returns:
        overlap_dict: For each pair of methods, number of overlapping samples
    """
    confidences = {}
    
    for method in method_names:
        if method not in all_logits:
            continue
        logits = all_logits[method]
        preds = all_predictions[method]
        
        # Compute confidence as softmax probability of predicted class
        probs = softmax(logits, axis=1)
        conf = probs[np.arange(len(preds)), preds]
        confidences[method] = conf
    
    # Find overlapping samples
    overlaps = defaultdict(dict)
    
    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            if i >= j or m1 not in confidences or m2 not in confidences:
                continue
            
            conf1 = confidences[m1]
            conf2 = confidences[m2]
            preds1 = all_predictions[m1]
            preds2 = all_predictions[m2]
            targets = all_targets
            
            if mode == 'correct':
                # Select correct predictions with highest confidence
                correct1 = preds1 == targets
                correct2 = preds2 == targets
                
                # For each method, get top K most confident among correct
                idx1 = _get_top_k(correct1, conf1, top_k, highest=True)
                idx2 = _get_top_k(correct2, conf2, top_k, highest=True)
            else:
                # Select wrong predictions with lowest confidence
                wrong1 = preds1 != targets
                wrong2 = preds2 != targets
                
                idx1 = _get_top_k(wrong1, conf1, top_k, highest=False)
                idx2 = _get_top_k(wrong2, conf2, top_k, highest=False)
            
            overlap = len(set(idx1) & set(idx2))
            overlaps[m1][m2] = overlap
            overlaps[m2][m1] = overlap
    
    return dict(overlaps)


def _get_top_k(mask, confidence, k, highest=True):
    """Get indices of top K samples by confidence within mask."""
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        return []
    
    valid_conf = confidence[valid_idx]
    if highest:
        sorted_local = np.argsort(valid_conf)[-min(k, len(valid_conf)):]
    else:
        sorted_local = np.argsort(valid_conf)[:min(k, len(valid_conf))]
    
    return valid_idx[sorted_local].tolist()


def softmax(x, axis=-1):
    """Compute softmax values."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def compute_prediction_diversity_score(all_predictions, method_names):
    """Compute a diversity score: average pairwise disagreement rate.
    
    Higher score = more diverse predictions.
    """
    similarity = compute_prediction_similarity(all_predictions, method_names)
    # Disagreement = 1 - similarity
    # Take upper triangle (excluding diagonal)
    M = len(method_names)
    upper_tri = []
    for i in range(M):
        for j in range(i+1, M):
            if not np.isnan(similarity[i, j]):
                upper_tri.append(1 - similarity[i, j])
    
    return np.mean(upper_tri) if upper_tri else 0.0


def ensemble_majority_vote(all_predictions, method_names):
    """Perform majority vote ensemble of PEFT methods.
    
    As described in Section 4 and Figure 4.
    
    Returns:
        ensemble_predictions: numpy array of ensemble predictions
    """
    if not all_predictions:
        return None
    
    # Stack predictions: [M, N]
    preds_list = []
    for method in method_names:
        if method in all_predictions:
            preds_list.append(all_predictions[method])
    
    if not preds_list:
        return None
    
    preds_stack = np.stack(preds_list, axis=0)  # [M, N]
    
    # Majority vote: for each sample, pick the most common prediction
    ensemble_preds = stats.mode(preds_stack, axis=0)[0].flatten()
    
    return ensemble_preds


def ensemble_average_logits(all_logits, method_names):
    """Ensemble by averaging logits across methods.
    
    As described in Figure 4 details.
    
    Returns:
        ensemble_predictions: numpy array of predictions from averaged logits
    """
    if not all_logits:
        return None
    
    logits_list = []
    for method in method_names:
        if method in all_logits:
            logits_list.append(all_logits[method])
    
    if not logits_list:
        return None
    
    avg_logits = np.mean(np.stack(logits_list, axis=0), axis=0)
    ensemble_preds = np.argmax(avg_logits, axis=1)
    
    return ensemble_preds
