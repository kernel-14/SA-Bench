"""
Training script for SMM (Sample-specific Multi-channel Masks).

Implements Algorithm 1 from the paper:
- Iterative label mapping (Ilm) updated each epoch
- Separate learning rates for delta (alpha_1) and phi (alpha_2)
- 200 epochs with milestones at 100 and 145
- Batch size 256 (64 for DTD and OxfordPets)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.models as models
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataloaders, DATASET_INFO
from smm import (SMMReprogramming, SharedPatternReprogramming,
                 MaskedWatermarkReprogramming, SampleSpecificPatternReprogramming,
                 SingleChannelSMMReprogramming, PaddingReprogramming)
from label_mapping import (random_label_mapping, iterative_label_mapping,
                            frequent_label_mapping, LabelMapper,
                            compute_frequency_distribution)


NUM_SOURCE_CLASSES = 1000  # ImageNet-1K


def get_pretrained_model(model_name, device):
    """Load pre-trained model from torchvision."""
    if model_name == 'ResNet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == 'ResNet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == 'ViT_B32':
        # ViT-B/32 with 384x384 input
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # Note: paper uses ViT-B32 which refers to patch size 32
        # torchvision has vit_b_32
        model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    model.eval()
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_reprogramming_model(method, model_name, dataset_name, patch_size=8):
    """Create the reprogramming model based on method name."""
    if method == 'smm':
        return SMMReprogramming(model_name=model_name, patch_size=patch_size)
    elif method == 'full':
        return MaskedWatermarkReprogramming(model_name=model_name, mask_type='full')
    elif method == 'narrow':
        return MaskedWatermarkReprogramming(model_name=model_name, mask_type='narrow')
    elif method == 'medium':
        return MaskedWatermarkReprogramming(model_name=model_name, mask_type='medium')
    elif method == 'pad':
        img_size = DATASET_INFO[dataset_name]['img_size']
        return PaddingReprogramming(model_name=model_name, target_img_size=img_size)
    elif method == 'only_delta':
        return SharedPatternReprogramming(model_name=model_name)
    elif method == 'only_fmask':
        return SampleSpecificPatternReprogramming(model_name=model_name, patch_size=patch_size)
    elif method == 'single_channel':
        return SingleChannelSMMReprogramming(model_name=model_name, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_optimizers(reprogram_model, method, model_name, lr_delta=0.01, lr_phi=0.01):
    """
    Create optimizers for delta and phi with separate learning rates.
    Following Chen et al. (2023): initial lr=0.01, decay=0.1
    For ViT: initial lr=0.001, decay=1 (no decay)
    """
    if method in ['smm', 'single_channel']:
        # Separate optimizers for delta and phi
        optimizer_delta = optim.SGD([reprogram_model.delta], lr=lr_delta, momentum=0.9)
        optimizer_phi = optim.SGD(reprogram_model.f_mask.parameters(), lr=lr_phi, momentum=0.9)
        return [optimizer_delta, optimizer_phi]
    elif method == 'only_fmask':
        optimizer_phi = optim.SGD(reprogram_model.f_mask.parameters(), lr=lr_phi, momentum=0.9)
        return [optimizer_phi]
    else:
        # Only delta
        optimizer_delta = optim.SGD([reprogram_model.delta], lr=lr_delta, momentum=0.9)
        return [optimizer_delta]


def train_epoch(reprogram_model, pretrained_model, train_loader, optimizers,
                label_mapper, criterion, device):
    """Train for one epoch."""
    reprogram_model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        for opt in optimizers:
            opt.zero_grad()

        # Forward pass through reprogramming
        reprogrammed = reprogram_model(images)

        # Forward pass through frozen pre-trained model
        with torch.no_grad():
            source_logits = pretrained_model(reprogrammed)

        # Get target logits via label mapping
        target_logits = label_mapper.get_target_logits(source_logits)

        # Compute loss
        loss = criterion(target_logits, labels)

        # Backward pass
        loss.backward()

        # Update parameters
        for opt in optimizers:
            opt.step()

        total_loss += loss.item() * images.size(0)
        preds = target_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate(reprogram_model, pretrained_model, test_loader, label_mapper, device):
    """Evaluate on test set."""
    reprogram_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            reprogrammed = reprogram_model(images)
            source_logits = pretrained_model(reprogrammed)
            target_logits = label_mapper.get_target_logits(source_logits)

            preds = target_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return correct / total


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine batch size (64 for DTD and OxfordPets, 256 otherwise)
    if args.dataset in ['DTD', 'OxfordPets']:
        batch_size = 64
    else:
        batch_size = 256

    # Load data
    train_loader, test_loader, num_target_classes = get_dataloaders(
        args.dataset, args.data_root, batch_size=batch_size,
        num_workers=args.num_workers, model_name=args.model
    )

    # Load pre-trained model
    pretrained_model = get_pretrained_model(args.model, device)

    # Create reprogramming model
    reprogram_model = get_reprogramming_model(
        args.method, args.model, args.dataset, patch_size=args.patch_size
    ).to(device)

    # Learning rates
    if args.model == 'ViT_B32':
        lr_delta = 0.001
        lr_phi = 0.001
        lr_decay = 1.0  # No decay
    else:
        lr_delta = 0.01
        lr_phi = 0.01
        lr_decay = 0.1

    # Optimizers
    optimizers = get_optimizers(reprogram_model, args.method, args.model,
                                lr_delta=lr_delta, lr_phi=lr_phi)

    # Learning rate schedulers with milestones at 100 and 145
    milestones = [100, 145]
    schedulers = [
        MultiStepLR(opt, milestones=milestones, gamma=lr_decay)
        for opt in optimizers
    ]

    criterion = nn.CrossEntropyLoss()

    # Initialize label mapping
    if args.label_mapping == 'rlm':
        mapping, source_subset = random_label_mapping(
            NUM_SOURCE_CLASSES, num_target_classes, seed=args.seed
        )
        label_mapper = LabelMapper(mapping, source_subset, NUM_SOURCE_CLASSES)
    elif args.label_mapping == 'flm':
        # Compute initial mapping with identity f_in
        def identity_fin(x): return x
        d = compute_frequency_distribution(
            pretrained_model, identity_fin, train_loader,
            NUM_SOURCE_CLASSES, num_target_classes, device
        )
        mapping, source_subset = frequent_label_mapping(d)
        label_mapper = LabelMapper(mapping, source_subset, NUM_SOURCE_CLASSES)
    else:  # ilm (default)
        # Initialize with identity mapping
        def identity_fin(x): return x
        d = compute_frequency_distribution(
            pretrained_model, identity_fin, train_loader,
            NUM_SOURCE_CLASSES, num_target_classes, device
        )
        mapping, source_subset = frequent_label_mapping(d)
        label_mapper = LabelMapper(mapping, source_subset, NUM_SOURCE_CLASSES)

    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Update label mapping for Ilm
        if args.label_mapping == 'ilm':
            def current_fin(x):
                with torch.no_grad():
                    return reprogram_model(x)
            mapping, source_subset = iterative_label_mapping(
                pretrained_model, current_fin, train_loader,
                NUM_SOURCE_CLASSES, num_target_classes, device
            )
            label_mapper = LabelMapper(mapping, source_subset, NUM_SOURCE_CLASSES)

        # Train one epoch
        train_loss, train_acc = train_epoch(
            reprogram_model, pretrained_model, train_loader,
            optimizers, label_mapper, criterion, device
        )

        # Evaluate
        test_acc = evaluate(reprogram_model, pretrained_model, test_loader,
                            label_mapper, device)

        # Update schedulers
        for sched in schedulers:
            sched.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'reprogram_model': reprogram_model.state_dict(),
                'label_mapping': mapping,
                'source_subset': source_subset,
                'best_acc': best_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Loss={train_loss:.4f}, Train={train_acc*100:.2f}%, "
                  f"Test={test_acc*100:.2f}%, Best={best_acc*100:.2f}%")

    print(f"\nFinal best test accuracy: {best_acc*100:.2f}%")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='SMM Visual Reprogramming')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=list(DATASET_INFO.keys()))
    parser.add_argument('--model', type=str, default='ResNet18',
                        choices=['ResNet18', 'ResNet50', 'ViT_B32'])
    parser.add_argument('--method', type=str, default='smm',
                        choices=['smm', 'full', 'narrow', 'medium', 'pad',
                                 'only_delta', 'only_fmask', 'single_channel'])
    parser.add_argument('--label_mapping', type=str, default='ilm',
                        choices=['ilm', 'flm', 'rlm'])
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Patch size for interpolation (2^l, default: 8=2^3)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
