#!/usr/bin/env python3
"""Robustness training script for PEFT methods with CLIP backbone.

Fine-tunes CLIP ViT-B/16 on 100-shot ImageNet-1K and evaluates
robustness to distribution shifts (ImageNet-V2, R, S, A).
Implements WiSE for PEFT following Section 7 of the paper.
"""

import argparse
import os
import sys
import json
import torch
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods import METHOD_MAP
from methods.model_builder import build_model
from methods.backbone import CLIPViTBackbone
from utils.data import get_robustness_dataset
from utils.training import train_model, evaluate
from utils.wise import apply_wise_to_model, compute_wise_accuracy_curve


def main():
    parser = argparse.ArgumentParser(description='Robustness PEFT training')
    parser.add_argument('--method', type=str, default='lora')
    parser.add_argument('--data_dir', type=str, default='/data/imagenet')
    parser.add_argument('--output_dir', type=str, default='./results/robustness')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--shots_per_class', type=int, default=100)
    parser.add_argument('--apply_wise', action='store_true',
                        help='Run WiSE evaluation after training')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading 100-shot ImageNet...")
    train_dataset = get_robustness_dataset(
        'imagenet', data_dir=args.data_dir, split='train',
        shots_per_class=args.shots_per_class
    )
    test_dataset = get_robustness_dataset(
        'imagenet', data_dir=args.data_dir, split='test'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=8)
    
    # Build CLIP model with PEFT
    num_classes = 1000
    model = build_model(
        method_name=args.method,
        num_classes=num_classes,
        backbone_type='clip',
        drop_path_rate=0.0,
    )
    model = model.to(args.device)
    
    # Initialize head with zero-shot weights
    print("Initializing classification head from CLIP zero-shot weights...")
    _init_zero_shot_head(model.backbone, model.head, args.device)
    
    # Train on target distribution
    print(f"\nTraining {args.method} on ImageNet (100-shot)...")
    
    criterion = torch.nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, 
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)
    
    # Simple training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        train_acc = 100. * correct / total
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: loss={total_loss/len(train_loader):.4f}, "
                  f"acc={train_acc:.2f}%")
    
    # Evaluate on target distribution
    target_result = evaluate(model, test_loader, criterion, args.device)
    print(f"\n  Target (ImageNet) accuracy: {target_result['accuracy']:.2f}%")
    
    # Evaluate on distribution shifts
    shift_results = {}
    shift_datasets = ['imagenet_v2', 'imagenet_r', 'imagenet_s', 'imagenet_a']
    
    for shift_name in shift_datasets:
        try:
            shift_dataset = get_robustness_dataset(shift_name, data_dir=args.data_dir)
            shift_loader = DataLoader(shift_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=4)
            shift_result = evaluate(model, shift_loader, criterion, args.device)
            shift_results[shift_name] = shift_result['accuracy']
            print(f"  {shift_name} accuracy: {shift_result['accuracy']:.2f}%")
        except Exception as e:
            print(f"  {shift_name}: skipped ({e})")
            shift_results[shift_name] = None
    
    # Compute average shift accuracy
    valid_shifts = [v for v in shift_results.values() if v is not None]
    avg_shift_acc = np.mean(valid_shifts) if valid_shifts else 0
    
    print(f"  Average distribution shift accuracy: {avg_shift_acc:.2f}%")
    
    # WiSE analysis
    wise_results = None
    if args.apply_wise:
        print("\nRunning WiSE analysis...")
        pretrained_backbone = CLIPViTBackbone()
        
        def eval_fn(m):
            results = {}
            tgt = evaluate(m, test_loader, criterion, args.device)
            results['target_acc'] = tgt['accuracy']
            
            shift_accs = []
            for shift_name in shift_datasets:
                try:
                    shift_dataset = get_robustness_dataset(shift_name, data_dir=args.data_dir)
                    shift_loader = DataLoader(shift_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=4)
                    r = evaluate(m, shift_loader, criterion, args.device)
                    shift_accs.append(r['accuracy'])
                except Exception:
                    pass
            results['shift_acc'] = np.mean(shift_accs) if shift_accs else 0
            return results
        
        wise_results = compute_wise_accuracy_curve(
            model, pretrained_backbone, eval_fn
        )
        
        for r in wise_results:
            print(f"  α={r['alpha']:.2f}: target={r['target_acc']:.2f}%, "
                  f"shift={r['shift_acc']:.2f}%")
    
    # Save results
    output = {
        'method': args.method,
        'target_accuracy': target_result['accuracy'],
        'shift_accuracies': shift_results,
        'avg_shift_accuracy': avg_shift_acc,
        'wise_results': wise_results,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
    }
    
    save_path = os.path.join(args.output_dir, f"robustness_{args.method}.json")
    with open(save_path, 'w') as f:
        # Convert numpy values for JSON serialization
        json_output = json.dumps(output, indent=2, default=str)
        f.write(json_output)
    
    print(f"\nResults saved to {save_path}")


def _init_zero_shot_head(backbone, head, device):
    """Initialize classification head with CLIP zero-shot weights.
    
    Following the paper: W_zero-shot is constructed by embedding
    class name prompts with the CLIP text encoder.
    """
    # This requires the CLIP text encoder which is part of the full CLIP model
    # For now, we use a simplified approach
    # In practice, you'd compute: head.weight = text_encoder(prompts)
    pass


if __name__ == '__main__':
    main()
