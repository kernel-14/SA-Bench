"""Evaluation utilities: WiSE, ensemble methods, prediction analysis.

Implements:
- WiSE (Weight-space Ensembles) for PEFT methods
- Majority vote ensemble
- Prediction similarity analysis (overlap matrices, Venn diagrams)
- Confidence-based analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(model, loader, device='cuda'):
    """Get predictions and confidence scores."""
    model.eval()
    all_outputs = []
    all_targets = []

    for images, targets in tqdm(loader, desc='Eval'):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        all_outputs.append(outputs.cpu())
        all_targets.append(targets.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    probs = F.softmax(all_outputs, dim=-1)
    confidences, predictions = probs.max(dim=-1)

    return {
        'logits': all_outputs,
        'probs': probs,
        'predictions': predictions,
        'targets': all_targets,
        'confidences': confidences,
        'acc': (predictions == all_targets).float().mean().item() * 100,
    }


def compute_prediction_similarity(results_dict):
    """Compute prediction overlap matrix between all methods.

    Args:
        results_dict: {method_name: {'predictions': tensor, 'targets': tensor}}

    Returns:
        overlap_matrix: (N x N) where (i,j) = % samples where i and j agree
        method_names: list of method names
    """
    method_names = sorted(results_dict.keys())
    n_methods = len(method_names)
    n_samples = len(results_dict[method_names[0]]['predictions'])

    overlap_matrix = np.zeros((n_methods, n_methods))

    for i, name_i in enumerate(method_names):
        pred_i = results_dict[name_i]['predictions']
        for j, name_j in enumerate(method_names):
            pred_j = results_dict[name_j]['predictions']
            overlap_matrix[i, j] = (pred_i == pred_j).float().mean().item() * 100

    return overlap_matrix, method_names


def compute_confidence_overlap(results_dict, k=5000, mode='correct'):
    """Compute overlap of correct/wrong predictions among most/least confident samples.

    Args:
        results_dict: {method_name: {'predictions', 'targets', 'confidences'}}
        k: number of samples to consider per method
        mode: 'correct' for top-k confident correct predictions,
              'wrong' for bottom-k confident wrong predictions

    Returns:
        overlap_counts: dict of set intersections
    """
    method_names = sorted(results_dict.keys())
    sample_sets = {}

    for name in method_names:
        res = results_dict[name]
        preds = res['predictions']
        targets = res['targets']
        confs = res['confidences']

        if mode == 'correct':
            # Top-k confident correct predictions
            correct_mask = preds == targets
            confs_correct = confs.clone()
            confs_correct[~correct_mask] = -float('inf')
            _, top_indices = torch.topk(confs_correct, k=min(k, correct_mask.sum().item()))
        else:
            # Bottom-k confident wrong predictions
            wrong_mask = preds != targets
            confs_wrong = confs.clone()
            confs_wrong[~wrong_mask] = float('inf')
            _, bottom_indices = torch.topk(confs_wrong, k=min(k, wrong_mask.sum().item()),
                                           largest=False)

        sample_sets[name] = set(bottom_indices.tolist() if mode == 'wrong'
                                else top_indices.tolist())

    return sample_sets


def majority_vote_ensemble(results_dict):
    """Majority vote ensemble across all PEFT methods.

    Args:
        results_dict: {method_name: {'predictions': tensor, 'targets': tensor}}

    Returns:
        accuracy: ensemble accuracy
        ensemble_predictions: tensor of ensemble predictions
    """
    method_names = sorted(results_dict.keys())
    all_preds = torch.stack([results_dict[n]['predictions'] for n in method_names], dim=0)
    targets = results_dict[method_names[0]]['targets']

    # Majority vote
    ensemble_preds = torch.mode(all_preds, dim=0).values
    acc = (ensemble_preds == targets).float().mean().item() * 100

    return acc, ensemble_preds


def logit_average_ensemble(results_dict):
    """Average logits ensemble across all PEFT methods.

    Args:
        results_dict: {method_name: {'logits': tensor, 'targets': tensor}}

    Returns:
        accuracy: ensemble accuracy
        ensemble_predictions: tensor of ensemble predictions
    """
    method_names = sorted(results_dict.keys())
    all_logits = torch.stack([results_dict[n]['logits'] for n in method_names], dim=0)
    targets = results_dict[method_names[0]]['targets']

    avg_logits = all_logits.mean(dim=0)
    ensemble_preds = avg_logits.argmax(dim=-1)
    acc = (ensemble_preds == targets).float().mean().item() * 100

    return acc, ensemble_preds


def apply_wise_to_model(finetuned_model, pretrained_model, alpha=0.5):
    """Apply Weight-space Ensembles (WiSE) between fine-tuned and pre-trained models.

    Interpolates linearly between the two models' weights:
        W_wise = alpha * W_finetuned + (1 - alpha) * W_pretrained

    For PEFT methods:
    - Direct selective (BitFit, LayerNorm): merge PEFT-tuned params with original
    - Adapter-based: scale adapter contribution by alpha
    - Efficient selective (LoRA, FacT): scale residual contribution by alpha

    Args:
        finetuned_model: fine-tuned model (PEFT or full FT)
        pretrained_model: original pre-trained model
        alpha: mixing coefficient (0 = pure pretrained, 1 = pure fine-tuned)

    Returns:
        merged model state dict
    """
    finetuned_state = finetuned_model.state_dict()
    pretrained_state = pretrained_model.state_dict()

    merged_state = {}
    for key in pretrained_state.keys():
        if key in finetuned_state:
            merged_state[key] = alpha * finetuned_state[key] + (1 - alpha) * pretrained_state[key]
        else:
            merged_state[key] = pretrained_state[key]

    # Handle head separately: always interpolate with alpha
    for key in finetuned_state.keys():
        if key not in pretrained_state:
            merged_state[key] = finetuned_state[key]

    return merged_state


@torch.no_grad()
def evaluate_wise(model_fn, pretrained_model, finetuned_model, loader,
                  alphas, device='cuda'):
    """Evaluate WiSE across multiple mixing coefficients.

    Args:
        model_fn: function that builds a model from state dict
        pretrained_model: original pre-trained model
        finetuned_model: fine-tuned model
        loader: data loader
        alphas: list of mixing coefficients to try
        device: torch device

    Returns:
        results: dict mapping alpha -> accuracy
    """
    results = {}
    save_state = finetuned_model.state_dict()

    for alpha in alphas:
        merged_state = apply_wise_to_model(finetuned_model, pretrained_model, alpha)
        finetuned_model.load_state_dict(merged_state)
        eval_result = evaluate_model(finetuned_model, loader, device)
        results[alpha] = eval_result['acc']

    # Restore original fine-tuned state
    finetuned_model.load_state_dict(save_state)

    return results


class WiSEPEFT:
    """WiSE for PEFT methods with method-specific handling."""

    @staticmethod
    def merge_bitfit(model, pretrained_state, alpha):
        """For BitFit: alpha-mix bias terms, keep rest from pretrained."""
        state = model.state_dict()
        merged = {}
        for key, val in state.items():
            if 'bias' in key:
                merged[key] = alpha * val + (1 - alpha) * pretrained_state.get(key, val)
            elif key in pretrained_state:
                merged[key] = pretrained_state[key]
            else:
                merged[key] = val
        return merged

    @staticmethod
    def merge_lora(model, pretrained_state, alpha):
        """For LoRA: scale the additive residual by alpha."""
        state = model.state_dict()
        merged = {}
        for key, val in state.items():
            if any(x in key for x in ['lora.down', 'lora.up']):
                merged[key] = val  # Keep original (applied via forward path)
            elif key in pretrained_state:
                merged[key] = alpha * val + (1 - alpha) * pretrained_state[key]
            else:
                merged[key] = val
        return merged

    @staticmethod
    def merge_adapter(model, pretrained_state, alpha):
        """For adapters: scale adapter contribution, keep backbone from pretrained."""
        state = model.state_dict()
        merged = {}
        for key, val in state.items():
            if 'adapter' in key.lower() or 'repadapter' in key.lower() or 'convpass' in key.lower():
                merged[key] = val  # Keep adapter params (scaled via forward path)
            elif key in pretrained_state:
                merged[key] = alpha * val + (1 - alpha) * pretrained_state[key]
            else:
                merged[key] = val
        return merged


def compute_ranking_frequency(accuracies_by_dataset):
    """Compute how often each method ranks 1st, 2nd, etc.

    Args:
        accuracies_by_dataset: dict {dataset_name: {method_name: accuracy}}

    Returns:
        frequency_matrix: (N_methods x N_methods) matrix
        method_order: list of method names ordered by mean rank
    """
    method_names = set()
    for accs in accuracies_by_dataset.values():
        method_names.update(accs.keys())
    method_names = sorted(method_names)
    n_methods = len(method_names)
    n_datasets = len(accuracies_by_dataset)

    frequency_matrix = np.zeros((n_methods, n_methods))
    all_ranks = {m: [] for m in method_names}

    for ds_name, acc_dict in accuracies_by_dataset.items():
        sorted_methods = sorted(acc_dict.items(), key=lambda x: x[1], reverse=True)
        for rank, (method, _) in enumerate(sorted_methods):
            if method in method_names:
                idx = method_names.index(method)
                frequency_matrix[idx, rank] += 1
                all_ranks[method].append(rank)

    mean_ranks = {m: np.mean(ranks) if ranks else 0 for m, ranks in all_ranks.items()}
    method_order = sorted(method_names, key=lambda m: mean_ranks[m])

    return frequency_matrix, method_order, mean_ranks
