"""
Analysis script for prediction diversity and ensemble experiments.
Reproduces Figure 3 (prediction similarity) and Figure 4 (ensemble gains).

Usage:
    python analyze_predictions.py --checkpoints_dir /path/to/checkpoints --data_dir /path/to/vtab
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.evaluator import (
    evaluate, evaluate_ensemble, compute_prediction_similarity,
    get_confident_predictions, compute_prediction_overlap
)


def load_model_predictions(model, data_loader, device='cuda'):
    """Get predictions and confidences from a model."""
    model.eval()
    all_predictions = []
    all_confidences = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_logits.append(logits.cpu())
    
    return (np.array(all_predictions), np.array(all_confidences), 
            np.array(all_labels), torch.cat(all_logits))


def plot_prediction_similarity(similarity_matrix, methods, title='Prediction Similarity', 
                                save_path=None):
    """Plot prediction similarity heatmap."""
    n = len(methods)
    matrix = np.zeros((n, n))
    
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            matrix[i, j] = similarity_matrix[m1][m2]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=methods, yticklabels=methods, 
                annot=True, fmt='.1f', cmap='Blues', ax=ax,
                vmin=0, vmax=100)
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ensemble_gains(results_per_dataset, methods, save_path=None):
    """Plot ensemble performance gains (Figure 4)."""
    datasets = list(results_per_dataset.keys())
    
    # Compute gains over worst method
    gains = {}
    for dataset in datasets:
        accs = results_per_dataset[dataset]
        worst_acc = min(accs.values())
        ensemble_acc = accs.get('ensemble', worst_acc)
        
        gains[dataset] = {
            method: acc - worst_acc 
            for method, acc in accs.items()
        }
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(datasets))
    width = 0.05
    
    for i, method in enumerate(methods + ['ensemble']):
        method_gains = [gains[d].get(method, 0) for d in datasets]
        ax.bar(x + i * width, method_gains, width, label=method)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy Gain over Worst Method (%)')
    ax.set_title('Ensemble Performance Gains')
    ax.set_xticks(x + width * len(methods) / 2)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_prediction_diversity(models_dict, data_loader, device='cuda', 
                                  dataset_name='dataset', output_dir='./analysis'):
    """
    Analyze prediction diversity across PEFT methods.
    
    Args:
        models_dict: Dict mapping method names to models
        data_loader: DataLoader for evaluation
        device: Device to use
        dataset_name: Name of the dataset
        output_dir: Directory to save analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions from all models
    predictions_dict = {}
    confidences_dict = {}
    labels = None
    logits_dict = {}
    
    for method, model in models_dict.items():
        model = model.to(device)
        preds, confs, lbls, logits = load_model_predictions(model, data_loader, device)
        predictions_dict[method] = preds
        confidences_dict[method] = confs
        logits_dict[method] = logits
        if labels is None:
            labels = lbls
    
    methods = list(models_dict.keys())
    
    # Compute prediction similarity
    similarity_matrix = compute_prediction_similarity(predictions_dict)
    
    # Save similarity matrix
    sim_path = os.path.join(output_dir, f'{dataset_name}_similarity.json')
    with open(sim_path, 'w') as f:
        json.dump(similarity_matrix, f, indent=2)
    
    # Plot similarity heatmap
    plot_prediction_similarity(
        similarity_matrix, methods,
        title=f'Prediction Similarity - {dataset_name}',
        save_path=os.path.join(output_dir, f'{dataset_name}_similarity.png')
    )
    
    # Compute ensemble accuracy
    ensemble_logits = torch.stack([logits_dict[m] for m in methods]).mean(0)
    _, ensemble_preds = ensemble_logits.max(1)
    ensemble_acc = 100.0 * (ensemble_preds.numpy() == labels).mean()
    
    # Compute individual accuracies
    individual_accs = {}
    for method in methods:
        acc = 100.0 * (predictions_dict[method] == labels).mean()
        individual_accs[method] = acc
    
    print(f"\n=== {dataset_name} ===")
    print("Individual accuracies:")
    for method, acc in individual_accs.items():
        print(f"  {method}: {acc:.2f}%")
    print(f"Ensemble accuracy: {ensemble_acc:.2f}%")
    
    # Analyze confident predictions (Figure 1b / Figure 3b)
    # Select 3 representative methods (one from each category)
    selected_methods = []
    for category_methods in [
        ['lora', 'fact_tt', 'fact_tk'],  # efficient selective
        ['houl_adapter', 'pfeif_adapter', 'adaptformer', 'convpass', 'repadapter'],  # adapter
        ['ssf', 'bitfit', 'layernorm', 'difffit'],  # direct selective
    ]:
        for m in category_methods:
            if m in methods:
                selected_methods.append(m)
                break
    
    if len(selected_methods) >= 3:
        selected_methods = selected_methods[:3]
        
        # Most confident correct predictions
        correct_indices = {}
        for method in selected_methods:
            idx = get_confident_predictions(
                predictions_dict[method], confidences_dict[method], labels,
                k=5000, correct_only=True
            )
            correct_indices[method] = set(idx)
        
        # Compute overlaps
        sets = [correct_indices[m] for m in selected_methods]
        overlaps = {
            'only_0': len(sets[0] - sets[1] - sets[2]),
            'only_1': len(sets[1] - sets[0] - sets[2]),
            'only_2': len(sets[2] - sets[0] - sets[1]),
            '0_and_1': len(sets[0] & sets[1] - sets[2]),
            '0_and_2': len(sets[0] & sets[2] - sets[1]),
            '1_and_2': len(sets[1] & sets[2] - sets[0]),
            'all': len(sets[0] & sets[1] & sets[2]),
        }
        
        print(f"\nCorrect prediction overlaps (5K most confident):")
        print(f"  Methods: {selected_methods}")
        print(f"  Overlaps: {overlaps}")
    
    return {
        'similarity_matrix': similarity_matrix,
        'individual_accs': individual_accs,
        'ensemble_acc': ensemble_acc,
    }


def main():
    parser = argparse.ArgumentParser(description='Prediction Diversity Analysis')
    
    parser.add_argument('--checkpoints_dir', type=str, required=True,
                        help='Directory containing model checkpoints')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory for VTAB-1K data')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='Dataset to analyze')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models from checkpoints
    # This assumes checkpoints are saved in {checkpoints_dir}/{method}/{dataset}_best.pth
    from src.datasets.vtab import get_vtab_dataset, VTAB_NUM_CLASSES
    from train_vtab import build_model
    
    methods = ['bitfit', 'layernorm', 'difffit', 'ssf', 'vpt_shallow', 'vpt_deep',
               'pfeif_adapter', 'houl_adapter', 'adaptformer', 'convpass', 'repadapter',
               'lora', 'fact_tt', 'fact_tk']
    
    num_classes = VTAB_NUM_CLASSES.get(args.dataset, 100)
    
    models_dict = {}
    for method in methods:
        checkpoint_path = os.path.join(
            args.checkpoints_dir, method, args.dataset, 
            f'{method}_{args.dataset}_best.pth'
        )
        
        if os.path.exists(checkpoint_path):
            try:
                model = build_model(method, num_classes)
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                models_dict[method] = model
                print(f"Loaded {method} checkpoint")
            except Exception as e:
                print(f"Failed to load {method}: {e}")
    
    if not models_dict:
        print("No checkpoints found. Please run training first.")
        return
    
    # Get test data loader
    test_loader = get_vtab_dataset(
        args.data_dir, args.dataset, split='test',
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # Analyze prediction diversity
    results = analyze_prediction_diversity(
        models_dict, test_loader, device=str(device),
        dataset_name=args.dataset, output_dir=args.output_dir
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, f'{args.dataset}_analysis.json')
    with open(results_path, 'w') as f:
        json.dump({
            'individual_accs': results['individual_accs'],
            'ensemble_acc': results['ensemble_acc'],
        }, f, indent=2)
    
    print(f"\nAnalysis saved to {args.output_dir}")


if __name__ == '__main__':
    main()
