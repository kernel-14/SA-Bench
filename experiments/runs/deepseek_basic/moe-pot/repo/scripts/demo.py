"""
Demo script for MoE-POT.

Demonstrates the core components:
1. Model creation and parameter counting
2. Forward pass with synthetic data
3. Router-gating network analysis
4. Load balancing loss computation
5. Interpretability analysis
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from moe_pot import MoEPOT, MoELayer, FourierLayer, PatchEmbed, TemporalAggregation
from moe_pot.data_utils import (
    generate_synthetic_pde_data,
    prepare_pre_training_datasets,
    pad_channels,
)
from moe_pot.interpretability import (
    compute_dataset_expert_profiles,
    classify_input_by_routing,
    compute_classification_accuracy,
    analyze_expert_usage,
)
from moe_pot.training import PDEDataset, MultiPDEDataset, NoiseInjection


def demo_model_creation():
    """Demonstrate creating MoE-POT models of different sizes."""
    print("=" * 60)
    print("1. MODEL CREATION & PARAMETER COUNTS")
    print("=" * 60)
    
    configs = [
        ("Tiny", dict(dim=512, num_heads=4, num_layers=4)),
        ("Small", dict(dim=1024, num_heads=8, num_layers=6)),
        ("Medium", dict(dim=1024, num_heads=8, num_layers=8)),
    ]
    
    for name, cfg in configs:
        model = MoEPOT(**cfg)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Count experts separately
        expert_params = 0
        for block in model.blocks:
            for expert in block.moe.routed_experts:
                expert_params += sum(p.numel() for p in expert.parameters())
            for expert in block.moe.shared_experts:
                expert_params += sum(p.numel() for p in expert.parameters())
        
        # Activated params approx: only 4/16 routed + 2 shared per layer
        activated_ratio = (cfg['num_layers'] * (4 + 2)) / (cfg['num_layers'] * (16 + 2))
        
        print(f"\nMoE-POT-{name}:")
        print(f"  dim={cfg['dim']}, layers={cfg['num_layers']}, heads={cfg['num_heads']}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  MoE expert parameters: {expert_params:,}")
        print(f"  Approx activated params: {int(total_params * activated_ratio * 0.5 + total_params * 0.5):,}")
        print(f"  (Paper reports: Tiny=30M/17M, Small=166M/90M, Medium=489M/288M)")


def demo_forward_pass():
    """Demonstrate forward pass with synthetic data."""
    print("\n" + "=" * 60)
    print("2. FORWARD PASS WITH SYNTHETIC DATA")
    print("=" * 60)
    
    # Create model
    model = MoEPOT(
        dim=512,
        num_heads=4,
        num_layers=4,
        in_channels=3,
        out_channels=3,
        spatial_size=128,
        patch_size=8,
        T=10,
    )
    
    # Create synthetic input
    batch_size = 4
    T = 10
    x = torch.randn(batch_size, T, 3, 128, 128)
    
    print(f"Input shape: {x.shape}  (B={batch_size}, T={T}, C=3, H=128, W=128)")
    
    # Forward pass
    with torch.no_grad():
        out = model(x)
    
    print(f"Output shape: {out.shape}")
    
    # Forward pass with routing info
    with torch.no_grad():
        out, routing_info = model(x, return_routing=True)
    
    print(f"\nRouting info from {len(routing_info)} blocks:")
    for i, info in enumerate(routing_info):
        weights = info['weights']
        indices = info['indices']
        print(f"  Block {i}: weights shape={weights.shape}, indices shape={indices.shape}")
        print(f"    Top-{indices.shape[1]} experts selected: {indices[0].tolist()}")
        print(f"    Weights: {weights[0].tolist()}")
    
    # Compute load balancing loss
    balance_loss = model.get_load_balancing_loss(x)
    print(f"\nLoad balancing loss: {balance_loss.item():.6f}")


def demo_moe_layer():
    """Demonstrate the MoE layer in isolation."""
    print("\n" + "=" * 60)
    print("3. MIXTURE-OF-EXPERTS LAYER DETAILS")
    print("=" * 60)
    
    moe = MoELayer(
        dim=64,
        num_routed_experts=16,
        num_shared_experts=2,
        top_k=4,
    )
    
    x = torch.randn(2, 64, 16, 16)
    print(f"Input shape: {x.shape}")
    print(f"  Routed experts: {moe.num_routed_experts}")
    print(f"  Shared experts: {moe.num_shared_experts}")
    print(f"  Top-K selection: {moe.top_k}")
    print(f"  Activated experts: {moe.num_shared_experts + moe.top_k} (2 shared + 4 routed)")
    
    out = moe(x)
    print(f"Output shape: {out.shape}")
    
    # Check routing
    print(f"\nRouter output:")
    print(f"  Routing weights shape: {moe.routing_weights.shape}")
    print(f"  Routing indices shape: {moe.routing_indices.shape}")
    print(f"  Sample 0 selected experts: {moe.routing_indices[0].tolist()}")
    print(f"  Sample 0 weights: {moe.routing_weights[0].tolist()}")
    
    # Load balancing
    balance_loss = moe.get_load_balancing_loss(x)
    print(f"\nLoad balancing loss: {balance_loss.item():.6f}")


def demo_noise_injection():
    """Demonstrate noise injection for auto-regressive denoising."""
    print("\n" + "=" * 60)
    print("4. NOISE INJECTION (AUTO-REGRESSIVE DENOISING)")
    print("=" * 60)
    
    noise = NoiseInjection(epsilon=0.01)
    
    x = torch.randn(2, 10, 3, 128, 128)
    print(f"Original input shape: {x.shape}")
    print(f"Original norm: {torch.norm(x):.4f}")
    print(f"Epsilon: {noise.epsilon}")
    
    x_noisy = noise.add_noise(x)
    print(f"Noisy input norm: {torch.norm(x_noisy):.4f}")
    print(f"Noise added (relative): {(torch.norm(x_noisy - x) / torch.norm(x)):.6f}")


def demo_interpretability():
    """Demonstrate interpretability analysis."""
    print("\n" + "=" * 60)
    print("5. INTERPRETABILITY ANALYSIS")
    print("=" * 60)
    
    # Create small model for demo
    model = MoEPOT(
        dim=256,
        num_heads=4,
        num_layers=2,
        in_channels=3,
        out_channels=3,
        spatial_size=64,
        patch_size=8,
        T=10,
    )
    
    # Create synthetic datasets
    print("Creating synthetic PDE datasets...")
    ns_data = generate_synthetic_pde_data('ns', num_samples=10, num_timesteps=15, 
                                           spatial_size=64, num_channels=1)
    swe_data = generate_synthetic_pde_data('swe', num_samples=10, num_timesteps=15,
                                            spatial_size=64, num_channels=1)
    dr_data = generate_synthetic_pde_data('dr', num_samples=10, num_timesteps=15,
                                           spatial_size=64, num_channels=1)
    
    # Pad to 3 channels
    ns_data = pad_channels(ns_data, 3)
    swe_data = pad_channels(swe_data, 3)
    dr_data = pad_channels(dr_data, 3)
    
    # Create dataloaders for each dataset
    from torch.utils.data import DataLoader
    
    dataloaders = {}
    for name, data in [('Navier-Stokes', ns_data), ('SWE', swe_data), ('DR', dr_data)]:
        dataset = PDEDataset(data, T=10, dataset_id=0)
        dataloaders[name] = DataLoader(dataset, batch_size=4, shuffle=False)
    
    print(f"Datasets: {list(dataloaders.keys())}")
    
    # Compute expert profiles
    print("\nComputing expert selection profiles...")
    profiles = compute_dataset_expert_profiles(
        model, dataloaders, block_idx=0, max_samples_per_dataset=20
    )
    
    for name, profile in profiles.items():
        print(f"  {name}: top experts = {profile.topk(4).indices.tolist()}")
    
    # Classify a sample
    print("\nClassifying a test sample...")
    sample_x, _, _ = next(iter(dataloaders['Navier-Stokes']))
    predicted, distances = classify_input_by_routing(
        model, sample_x[0:1], profiles, block_idx=0
    )
    print(f"  Predicted: {predicted}")
    print(f"  Distances: {distances}")
    
    # Compute accuracy
    print("\nComputing classification accuracy...")
    accuracies = compute_classification_accuracy(
        model, dataloaders, profiles, block_idx=0, max_samples=10
    )
    print(f"  Overall accuracy: {accuracies.get('overall', 0):.2%}")
    for name, acc in accuracies.items():
        if name != 'overall':
            print(f"  {name}: {acc:.2%}")
    
    # Expert usage analysis
    print("\nExpert usage ratio per dataset:")
    usage = analyze_expert_usage(model, dataloaders, block_idx=0, max_samples=10)
    for name, ratio in usage.items():
        top_experts = np.argsort(ratio)[-4:][::-1]
        print(f"  {name}: top experts = {top_experts.tolist()}, ratios = {ratio[top_experts].tolist()}")


def demo_data_pipeline():
    """Demonstrate the data preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("6. DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Generate different datasets
    datasets = prepare_pre_training_datasets(synthetic=True, spatial_size=64, num_timesteps=15)
    
    print("Generated datasets:")
    for name, data in datasets.items():
        print(f"  {name}: shape={data.shape}, dtype={data.dtype}")
    
    # Unify channels
    max_ch = max(d.shape[2] for d in datasets.values())
    print(f"\nMax channels: {max_ch}")
    print(f"After padding to {max_ch} channels:")
    
    for name, data in datasets.items():
        padded = pad_channels(data, max_ch) if data.shape[2] < max_ch else data
        print(f"  {name}: shape={padded.shape}")


if __name__ == '__main__':
    import numpy as np
    
    print("MoE-POT Demo")
    print("============")
    print("Demonstrating key components of the MoE-POT architecture\n")
    
    demo_model_creation()
    demo_forward_pass()
    demo_moe_layer()
    demo_noise_injection()
    demo_data_pipeline()
    demo_interpretability()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
