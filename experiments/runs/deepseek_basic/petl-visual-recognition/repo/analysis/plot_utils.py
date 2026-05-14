"""Plotting utilities for PEFT analysis.

Generates the figures shown in the paper:
- Figure 1a: Accuracy gain vs linear probing
- Figure 1b: Prediction overlaps (Venn diagrams)
- Figure 2: Ranking frequency matrices
- Figure 3: Prediction similarity analysis
- Figure 4: Ensemble gain
- Figure 5: Many-shot parameter size vs accuracy
- Figure 6: Task categorization analysis
- Figure 11: Drop path rate effect
- Figure 12-14: Additional analyses
"""

import numpy as np
import json
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_accuracy_gain(accuracies_by_dataset, method_names, 
                        baseline_method='linear', output_path='fig1a.png'):
    """Figure 1a: Accuracy gain vs linear probing on VTAB-1K.
    
    Shows relative performance compared to linear probing.
    Range between highest and lowest PEFT accuracy shown as bullet range.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    datasets = sorted(accuracies_by_dataset.keys())
    
    # Compute gains relative to linear probing
    gains = {}
    for dataset in datasets:
        accs = accuracies_by_dataset[dataset]
        linear_acc = accs.get(baseline_method, 0)
        
        peft_methods = [m for m in accs if m not in ['linear', 'full']]
        peft_accs = [accs[m] for m in peft_methods if m in accs]
        
        gains[dataset] = {
            'peft_range': (min(peft_accs) - linear_acc, max(peft_accs) - linear_acc) if peft_accs else (0, 0),
            'full': accs.get('full', 0) - linear_acc,
            'linear': 0,
        }
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    x = np.arange(len(datasets))
    
    # Plot PEFT range
    for i, dataset in enumerate(datasets):
        lo, hi = gains[dataset]['peft_range']
        ax.plot([i, i], [lo, hi], 'k-', linewidth=3, alpha=0.5)
        ax.plot(i, lo, 'k.', markersize=8)
        ax.plot(i, hi, 'k.', markersize=8)
    
    # Plot full FT
    full_gains = [gains[d]['full'] for d in datasets]
    ax.scatter(x, full_gains, marker='s', s=60, c='blue', label='Full FT', zorder=5)
    
    # Plot linear (baseline)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy gain vs Linear Probing (%)')
    ax.set_title('Figure 1a: Accuracy gain vs Linear Probing on VTAB-1K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_ranking_frequency(frequency_matrix, method_names, 
                            output_path='fig2.png'):
    """Figure 2: Ranking frequency matrix.
    
    Element (i,j) = number of times method i ranks j-th.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    M = len(method_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(frequency_matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(M))
    ax.set_xticklabels([f'{i+1}' for i in range(M)])
    ax.set_yticklabels(method_names)
    
    # Add text annotations
    for i in range(M):
        for j in range(M):
            if frequency_matrix[i, j] > 0:
                text = ax.text(j, i, int(frequency_matrix[i, j]),
                              ha='center', va='center', fontsize=8)
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Method')
    ax.set_title('Ranking Frequency Matrix')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_prediction_similarity(similarity_matrix, method_names,
                                output_path='fig3a.png'):
    """Figure 3a: Prediction similarity matrix."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    M = len(method_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    
    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(M))
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(method_names, fontsize=8)
    
    # Add text annotations
    for i in range(M):
        for j in range(M):
            if not np.isnan(similarity_matrix[i, j]):
                text = ax.text(j, i, f'{similarity_matrix[i,j]:.3f}',
                              ha='center', va='center', fontsize=7)
    
    ax.set_title('Prediction Similarity Matrix')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_ensemble_gain(dataset_gains, output_path='fig4.png'):
    """Figure 4: Ensemble gain over worst method per dataset."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    datasets = sorted(dataset_gains.keys())
    gains = [dataset_gains[d] for d in datasets]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    x = np.arange(len(datasets))
    bars = ax.bar(x, gains, color='steelblue', alpha=0.8)
    
    # Highlight positive gains
    for i, (bar, gain) in enumerate(zip(bars, gains)):
        if gain > 0:
            bar.set_color('green')
        elif gain < 0:
            bar.set_color('red')
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Ensemble Gain over Worst Method (%)')
    ax.set_title('Ensemble Gain on VTAB-1K Datasets')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_param_size_vs_accuracy(param_results, output_path='fig5.png'):
    """Figure 5: Many-shot: parameter size vs accuracy."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    datasets = sorted(param_results.keys())
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        results = param_results[dataset]
        
        for method_idx, (method, points) in enumerate(sorted(results.items())):
            if not points:
                continue
            sizes = [p['params'] for p in points]
            accs = [p['accuracy'] for p in points]
            
            # Sort by param size
            sorted_pairs = sorted(zip(sizes, accs))
            sizes, accs = zip(*sorted_pairs)
            
            ax.plot(sizes, accs, 'o-', color=colors[method_idx], 
                    label=method, markersize=6, linewidth=1.5)
        
        ax.set_xlabel('Trainable Parameters (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(dataset)
        ax.grid(True, alpha=0.3)
    
    axes[0].legend(fontsize=7, loc='lower right')
    plt.suptitle('Many-shot: Parameter Size vs Accuracy')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_task_categories(cat1_results, cat2_results, output_path='fig6.png'):
    """Figure 6: Task categorization analysis.
    
    Shows accuracy trends: linear -> PEFT -> full for two task categories.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Category 1: Full FT > Linear (low-shot)
    ax = axes[0, 0]
    for dataset, accs in cat1_results.items():
        methods_ordered = ['linear', 'peft_avg', 'full']
        values = [
            accs.get('linear', 0),
            accs.get('peft_avg', 0),
            accs.get('full', 0),
        ]
        ax.plot([0, 1, 2], values, 'o-', linewidth=1, alpha=0.7, label=dataset)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Linear', 'PEFT', 'Full'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Category 1: Low-shot (Full > Linear)')
    ax.grid(True, alpha=0.3)
    
    # Category 2: Linear > Full (low-shot)
    ax = axes[0, 1]
    for dataset, accs in cat2_results.items():
        methods_ordered = ['linear', 'peft_avg', 'full']
        values = [
            accs.get('linear', 0),
            accs.get('peft_avg', 0),
            accs.get('full', 0),
        ]
        ax.plot([0, 1, 2], values, 'o-', linewidth=1, alpha=0.7, label=dataset)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Linear', 'PEFT', 'Full'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Category 2: Low-shot (Linear > Full)')
    ax.grid(True, alpha=0.3)
    
    # Many-shot versions would go in bottom row
    # (placeholder)
    axes[1, 0].text(0.5, 0.5, 'Many-shot Category 1\n(PEFT ≈ Full > Linear)',
                    ha='center', va='center', transform=axes[1,0].transAxes)
    axes[1, 1].text(0.5, 0.5, 'Many-shot Category 2\n(PEFT > Full > Linear)',
                    ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.suptitle('Task Categorization Analysis (Section 6)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_wise_curves(wise_results, output_path='fig1c.png'):
    """Figure 1c/Figure 14: WiSE accuracy curves.
    
    Target distribution vs distribution shifts for different mixing coefficients.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(wise_results)))
    
    for method_idx, (method, results) in enumerate(wise_results.items()):
        if not results:
            continue
        
        target_accs = [r['target_acc'] for r in results]
        shift_accs = [r['shift_acc'] for r in results]
        alphas = [r['alpha'] for r in results]
        
        # Sort by alpha and create curve
        sorted_pairs = sorted(zip(alphas, target_accs, shift_accs))
        alphas, target_accs, shift_accs = zip(*sorted_pairs)
        
        ax.plot(target_accs, shift_accs, 'o-', color=colors[method_idx],
                label=method, markersize=4, linewidth=1.5)
        
        # Mark α=0 and α=1
        ax.plot(target_accs[0], shift_accs[0], 's', color=colors[method_idx], markersize=8)
        ax.plot(target_accs[-1], shift_accs[-1], '*', color=colors[method_idx], markersize=10)
    
    ax.set_xlabel('Target Distribution Accuracy (%)')
    ax.set_ylabel('Distribution Shift Accuracy (%)')
    ax.set_title('WiSE: Target vs Distribution Shift Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")
