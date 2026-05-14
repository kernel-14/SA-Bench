"""Method ranking analysis for VTAB-1K.

Computes ranking frequency matrices as shown in Figure 2 of the paper.
"""

import numpy as np


def compute_rankings(accuracies, method_names):
    """Compute method rankings based on accuracy.
    
    Args:
        accuracies: dict mapping method_name -> accuracy value
        method_names: list of method names to rank
    
    Returns:
        rankings: dict mapping method_name -> rank (1 = best)
    """
    # Sort methods by accuracy (descending)
    sorted_methods = sorted(
        [(m, accuracies.get(m, 0)) for m in method_names],
        key=lambda x: x[1], reverse=True
    )
    
    rankings = {}
    for rank, (method, acc) in enumerate(sorted_methods, 1):
        rankings[method] = rank
    
    return rankings


def compute_ranking_frequency(all_dataset_accuracies, method_names, 
                               group_datasets=None):
    """Compute ranking frequency matrix.
    
    Args:
        all_dataset_accuracies: dict mapping dataset_name -> {method: accuracy}
        method_names: list of all method names
        group_datasets: optional dict mapping group_name -> list of dataset names
    
    Returns:
        frequency_matrix: [M x M] where element (i,j) = number of times
                         method i ranks j-th
        mean_ranks: list of mean ranks for each method
    """
    M = len(method_names)
    
    # For all datasets combined
    frequency = np.zeros((M, M))
    all_ranks = {m: [] for m in method_names}
    
    datasets = list(all_dataset_accuracies.keys())
    
    for dataset in datasets:
        accs = all_dataset_accuracies[dataset]
        rankings = compute_rankings(accs, method_names)
        
        for i, method in enumerate(method_names):
            if method in rankings:
                rank = rankings[method]
                frequency[i, rank - 1] += 1
                all_ranks[method].append(rank)
    
    # Compute mean ranks
    mean_ranks = []
    for method in method_names:
        ranks = all_ranks[method]
        mean_ranks.append(np.mean(ranks) if ranks else M)
    
    return frequency, mean_ranks


def compute_group_ranking_frequencies(all_dataset_accuracies, method_names,
                                       vtab_datasets):
    """Compute ranking frequency per group (Natural, Specialized, Structured).
    
    Args:
        all_dataset_accuracies: dict mapping dataset_name -> {method: accuracy}
        method_names: list of all method names
        vtab_datasets: dict mapping group_name -> list of dataset names
    
    Returns:
        group_frequencies: dict mapping group_name -> (frequency_matrix, mean_ranks)
    """
    group_frequencies = {}
    
    for group_name, datasets in vtab_datasets.items():
        group_accs = {d: all_dataset_accuracies.get(d, {}) for d in datasets
                      if d in all_dataset_accuracies}
        if group_accs:
            freq, mean_ranks = compute_ranking_frequency(
                group_accs, method_names
            )
            group_frequencies[group_name] = (freq, mean_ranks)
    
    return group_frequencies


def compute_relative_std(accuracies, method_names):
    """Compute relative standard deviation of accuracies across methods.
    
    Relative Std = std / mean (as percentage)
    """
    values = []
    for method in method_names:
        if method in accuracies and accuracies[method] is not None:
            values.append(accuracies[method])
    
    if not values:
        return 0.0
    
    values = np.array(values)
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    return (std_val / mean_val * 100) if mean_val > 0 else 0.0


def identify_task_categories(all_dataset_accuracies, method_names):
    """Identify two task categories as described in Section 6.
    
    Category 1: Full FT > Linear Probing (need to update backbone)
    Category 2: Linear Probing > Full FT (pre-trained features good enough)
    
    Returns:
        category1_datasets: list of datasets where full FT > linear
        category2_datasets: list of datasets where linear > full FT
    """
    category1 = []
    category2 = []
    
    for dataset, accs in all_dataset_accuracies.items():
        linear_acc = accs.get('linear', 0)
        full_acc = accs.get('full', 0)
        
        if full_acc > linear_acc:
            category1.append(dataset)
        elif linear_acc > full_acc:
            category2.append(dataset)
    
    return category1, category2
