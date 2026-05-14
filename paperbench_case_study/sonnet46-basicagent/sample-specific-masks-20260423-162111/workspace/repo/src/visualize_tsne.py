"""
Feature Space Visualization using t-SNE.

Reproduces Figure 6 from the paper: t-SNE visualization of the output layer
feature space before the label mapping layer.

Following the addendum: embeddings are computed using 5000 randomly selected
samples from each training set.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision.models as models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataset, DATASET_INFO
from smm import SMMReprogramming, MaskedWatermarkReprogramming
from label_mapping import LabelMapper
from torch.utils.data import DataLoader, Subset


NUM_SAMPLES = 5000  # From addendum: 5000 randomly selected samples


def get_features(model, reprogram_model, data_loader, device, n_samples=5000):
    """
    Extract features from the output layer (before label mapping).

    Args:
        model: pre-trained model
        reprogram_model: reprogramming module (or None for no reprogramming)
        data_loader: DataLoader
        device: torch device
        n_samples: number of samples to use

    Returns:
        features: numpy array (n_samples, feature_dim)
        labels: numpy array (n_samples,)
    """
    model.eval()
    if reprogram_model is not None:
        reprogram_model.eval()

    all_features = []
    all_labels = []
    total = 0

    # Hook to extract features before the final classification layer
    features_hook = []

    def hook_fn(module, input, output):
        features_hook.append(output.detach().cpu())

    # Register hook on the layer before the final classifier
    if hasattr(model, 'fc'):
        # ResNet: hook on avgpool output
        hook = model.avgpool.register_forward_hook(hook_fn)
    elif hasattr(model, 'heads'):
        # ViT: hook on the encoder output
        hook = model.encoder.register_forward_hook(hook_fn)
    else:
        hook = model.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, labels in data_loader:
            if total >= n_samples:
                break

            images = images.to(device)
            remaining = n_samples - total
            if images.size(0) > remaining:
                images = images[:remaining]
                labels = labels[:remaining]

            if reprogram_model is not None:
                images = reprogram_model(images)

            _ = model(images)

            if features_hook:
                feat = features_hook[-1]
                if feat.dim() > 2:
                    feat = feat.flatten(1)
                all_features.append(feat.numpy())
                features_hook.clear()

            all_labels.append(labels.numpy())
            total += images.size(0)

    hook.remove()

    features = np.concatenate(all_features, axis=0)[:n_samples]
    labels = np.concatenate(all_labels, axis=0)[:n_samples]
    return features, labels


def visualize_tsne(features, labels, title, save_path, num_classes=10):
    """Create t-SNE visualization."""
    print(f"Computing t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_classes, 10)))

    for i in range(min(num_classes, 10)):
        mask = labels == i
        if mask.sum() > 0:
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                       c=[colors[i]], label=f'Class {i}', alpha=0.6, s=5)

    plt.title(title)
    plt.legend(loc='best', markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='t-SNE Feature Visualization')
    parser.add_argument('--dataset', type=str, default='SVHN',
                        choices=['SVHN', 'EuroSAT'])
    parser.add_argument('--model', type=str, default='ResNet18',
                        choices=['ResNet18', 'ResNet50', 'ViT_B32'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs/tsne')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to SMM checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data (use training set, 5000 samples)
    train_ds, _, num_classes = get_dataset(args.dataset, args.data_root, args.model)

    # Randomly select 5000 samples
    np.random.seed(42)
    indices = np.random.choice(len(train_ds), min(NUM_SAMPLES, len(train_ds)), replace=False)
    subset = Subset(train_ds, indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)

    # Load pre-trained model
    if args.model == 'ResNet18':
        pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif args.model == 'ResNet50':
        pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        pretrained = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)

    pretrained = pretrained.to(device)
    pretrained.eval()
    for p in pretrained.parameters():
        p.requires_grad = False

    # 1. No reprogramming (baseline)
    features_none, labels = get_features(pretrained, None, loader, device, NUM_SAMPLES)
    visualize_tsne(features_none, labels,
                   f'{args.dataset} - No Reprogramming',
                   os.path.join(args.output_dir, f'{args.dataset}_no_reprogram.png'),
                   num_classes)

    # 2. Watermark (Full) baseline
    watermark = MaskedWatermarkReprogramming(model_name=args.model, mask_type='full').to(device)
    watermark.eval()
    features_wm, _ = get_features(pretrained, watermark, loader, device, NUM_SAMPLES)
    visualize_tsne(features_wm, labels,
                   f'{args.dataset} - Watermark (Full)',
                   os.path.join(args.output_dir, f'{args.dataset}_watermark.png'),
                   num_classes)

    # 3. SMM (if checkpoint provided)
    if args.checkpoint and os.path.exists(args.checkpoint):
        smm = SMMReprogramming(model_name=args.model).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        smm.load_state_dict(checkpoint['reprogram_model'])
        smm.eval()
        features_smm, _ = get_features(pretrained, smm, loader, device, NUM_SAMPLES)
        visualize_tsne(features_smm, labels,
                       f'{args.dataset} - SMM (Ours)',
                       os.path.join(args.output_dir, f'{args.dataset}_smm.png'),
                       num_classes)

    print("Done!")


if __name__ == '__main__':
    main()
