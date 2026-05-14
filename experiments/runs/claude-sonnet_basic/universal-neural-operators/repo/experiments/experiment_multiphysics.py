"""
Experiment 3: General multi-physics learning.

From the paper:
"In the final stage, we evaluated the capabilities of the developed methods to transfer
knowledge from the dynamics of advection and Burgers' equation to reaction-diffusion,
based on the PDEBench dataset."

This script:
1. Pretrains on advection + Burgers' equation (multi-physics)
2. Fine-tunes on reaction-diffusion
3. Compares with training from scratch
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FNO1d, MambaFNO1d, PerceiverFNO1d, CoDANO1d
from datasets import BurgersDataset, AdvectionDataset, ReactionDiffusionDataset
from datasets.pdebench import MultiPhysicsDataset
from utils import Trainer, compute_metrics
from utils.transfer import freeze_backbone, create_new_adapters, print_param_summary


def run_multiphysics_experiment(
    n_pretrain_per_physics: int = 500,
    n_finetune: int = 200,
    n_test: int = 200,
    n_epochs_pretrain: int = 100,
    n_epochs_finetune: int = 50,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "checkpoints/multiphysics",
    pdebench_path: str = None,
):
    """
    Run multi-physics pretraining experiment.
    
    Pretrain on advection + Burgers, fine-tune on reaction-diffusion.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Multi-Physics Pretraining -> Reaction-Diffusion Fine-tuning")
    print("=" * 60)
    
    # Generate pretraining datasets
    print("\nGenerating pretraining data...")
    
    if pdebench_path:
        from datasets import PDEBenchDataset
        advection_dataset = PDEBenchDataset(
            "advection", 
            data_path=os.path.join(pdebench_path, "1D_Advection_Sols_beta0.1.hdf5"),
            n_samples=n_pretrain_per_physics,
        )
        burgers_dataset = PDEBenchDataset(
            "burgers",
            data_path=os.path.join(pdebench_path, "1D_Burgers_Sols_Nu0.001.hdf5"),
            n_samples=n_pretrain_per_physics,
        )
        rd_dataset = PDEBenchDataset(
            "reaction_diffusion",
            data_path=os.path.join(pdebench_path, "ReacDiff_Nu0.5_Rho1.0.hdf5"),
            n_samples=n_finetune + n_test,
        )
    else:
        print("  Using synthetic data (PDEBench not available)")
        advection_dataset = AdvectionDataset(n_samples=n_pretrain_per_physics, seed=42)
        burgers_dataset = BurgersDataset(n_samples=n_pretrain_per_physics, seed=43)
        rd_full = ReactionDiffusionDataset(n_samples=n_finetune + n_test, seed=44)
    
    # Split reaction-diffusion into fine-tune and test
    if pdebench_path:
        rd_train, rd_test = random_split(rd_dataset, [n_finetune, n_test])
    else:
        rd_train, rd_test = random_split(rd_full, [n_finetune, n_test])
    
    # Create data loaders
    adv_loader = DataLoader(advection_dataset, batch_size=batch_size, shuffle=True)
    burg_loader = DataLoader(burgers_dataset, batch_size=batch_size, shuffle=True)
    rd_loader = DataLoader(rd_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(rd_test, batch_size=batch_size)
    
    # Get input/output dimensions
    n_input_adv = advection_dataset.n_input  # 2
    n_input_burg = burgers_dataset.n_input   # 2
    n_input_rd = rd_full.n_input if not pdebench_path else rd_dataset.n_input  # 3
    n_output = 1
    
    print(f"Advection n_input: {n_input_adv}")
    print(f"Burgers n_input: {n_input_burg}")
    print(f"Reaction-Diffusion n_input: {n_input_rd}")
    
    results = {}
    
    # ---- FNO Baseline (from scratch on reaction-diffusion) ----
    print("\n--- FNO (from scratch) ---")
    fno_scratch = FNO1d(n_input=n_input_rd, n_output=n_output, width=64, modes=16, n_layers=4)
    
    optimizer = optim.Adam(fno_scratch.parameters(), lr=1e-3)
    trainer = Trainer(fno_scratch, optimizer, device=device)
    
    start = time.time()
    trainer.train(rd_loader, test_loader, n_epochs=n_epochs_pretrain + n_epochs_finetune)
    elapsed = time.time() - start
    
    metrics = trainer.evaluate(test_loader)
    results["FNO (scratch)"] = {**metrics, "total_time": elapsed}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- MambaFNO Multi-Physics Pretraining ----
    print("\n--- MambaFNO (multi-physics pretrained) ---")
    
    # Create separate models for each physics (sharing backbone)
    # For multi-physics, we use the same backbone but different adapters
    # We pretrain on advection and Burgers simultaneously
    
    # Use advection model as the "base" for pretraining
    mamba_adv = MambaFNO1d(n_input=n_input_adv, n_output=n_output, width=64, modes=16, n_layers=4)
    mamba_burg = MambaFNO1d(n_input=n_input_burg, n_output=n_output, width=64, modes=16, n_layers=4)
    
    # Share backbone parameters between the two models
    # Copy backbone from advection model to Burgers model
    adv_backbone_state = {
        k: v for k, v in mamba_adv.state_dict().items()
        if 'lifting' not in k and 'projection' not in k
    }
    burg_state = mamba_burg.state_dict()
    burg_state.update(adv_backbone_state)
    mamba_burg.load_state_dict(burg_state)
    
    # Share backbone by making them point to the same parameters
    # (simplified: train both models with shared backbone)
    all_params = list(mamba_adv.parameters()) + list(mamba_burg.get_adapter_params())
    optimizer = optim.Adam(all_params, lr=1e-3)
    
    print("  Multi-physics pretraining (advection + Burgers)...")
    start_pretrain = time.time()
    
    for epoch in range(n_epochs_pretrain):
        mamba_adv.train()
        mamba_burg.train()
        
        adv_iter = iter(adv_loader)
        burg_iter = iter(burg_loader)
        
        total_loss = 0
        n_batches = 0
        
        for adv_batch, burg_batch in zip(adv_iter, burg_iter):
            adv_inputs, adv_targets = adv_batch[0].to(device), adv_batch[1].to(device)
            burg_inputs, burg_targets = burg_batch[0].to(device), burg_batch[1].to(device)
            
            optimizer.zero_grad()
            
            adv_pred = mamba_adv(adv_inputs)
            burg_pred = mamba_burg(burg_inputs)
            
            loss = nn.MSELoss()(adv_pred, adv_targets) + nn.MSELoss()(burg_pred, burg_targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(mamba_adv.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs_pretrain}, Loss: {total_loss/n_batches:.6f}")
    
    pretrain_time = time.time() - start_pretrain
    torch.save(mamba_adv.state_dict(), os.path.join(save_dir, "mamba_multiphysics_pretrained.pt"))
    
    # Fine-tune on reaction-diffusion with new adapters
    mamba_rd = create_new_adapters(mamba_adv, n_input_rd, n_output)
    print_param_summary(mamba_rd, "MambaFNO (fine-tuning on RD)")
    
    adapter_params = [p for p in mamba_rd.parameters() if p.requires_grad]
    optimizer_ft = optim.Adam(adapter_params, lr=1e-3)
    trainer_ft = Trainer(mamba_rd, optimizer_ft, device=device)
    
    print("  Fine-tuning on reaction-diffusion...")
    start_ft = time.time()
    trainer_ft.train(rd_loader, test_loader, n_epochs=n_epochs_finetune)
    ft_time = time.time() - start_ft
    
    metrics = trainer_ft.evaluate(test_loader)
    results["Mamba FNO (pretr.)"] = {**metrics, "total_time": pretrain_time + ft_time}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- MambaFNO (from scratch) ----
    print("\n--- MambaFNO (from scratch) ---")
    mamba_scratch = MambaFNO1d(n_input=n_input_rd, n_output=n_output, width=64, modes=16, n_layers=4)
    
    optimizer = optim.Adam(mamba_scratch.parameters(), lr=1e-3)
    trainer = Trainer(mamba_scratch, optimizer, device=device)
    
    start = time.time()
    trainer.train(rd_loader, test_loader, n_epochs=n_epochs_pretrain + n_epochs_finetune)
    elapsed = time.time() - start
    
    metrics = trainer.evaluate(test_loader)
    results["Mamba FNO (scratch)"] = {**metrics, "total_time": elapsed}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- Perceiver (pretrained) ----
    print("\n--- Perceiver IO (pretrained) ---")
    perc_adv = PerceiverFNO1d(
        n_input=n_input_adv, n_output=n_output, width=64, modes=16, n_layers=4, n_latent=32
    )
    
    optimizer = optim.Adam(perc_adv.parameters(), lr=1e-3)
    trainer = Trainer(perc_adv, optimizer, device=device)
    
    print("  Pretraining on advection...")
    start_pretrain = time.time()
    trainer.train(adv_loader, None, n_epochs=n_epochs_pretrain // 2)
    
    # Continue with Burgers
    perc_burg = PerceiverFNO1d(
        n_input=n_input_burg, n_output=n_output, width=64, modes=16, n_layers=4, n_latent=32
    )
    # Share backbone
    burg_state = perc_burg.state_dict()
    burg_state.update({k: v for k, v in perc_adv.state_dict().items() 
                       if 'lifting' not in k and 'projection' not in k})
    perc_burg.load_state_dict(burg_state)
    
    trainer2 = Trainer(perc_burg, optim.Adam(perc_burg.parameters(), lr=1e-3), device=device)
    trainer2.train(burg_loader, None, n_epochs=n_epochs_pretrain // 2)
    pretrain_time = time.time() - start_pretrain
    
    # Fine-tune on reaction-diffusion
    perc_rd = create_new_adapters(perc_adv, n_input_rd, n_output)
    adapter_params = [p for p in perc_rd.parameters() if p.requires_grad]
    optimizer_ft = optim.Adam(adapter_params, lr=1e-3)
    trainer_ft = Trainer(perc_rd, optimizer_ft, device=device)
    
    print("  Fine-tuning on reaction-diffusion...")
    start_ft = time.time()
    trainer_ft.train(rd_loader, test_loader, n_epochs=n_epochs_finetune)
    ft_time = time.time() - start_ft
    
    metrics = trainer_ft.evaluate(test_loader)
    results["Perc. (pretr.)"] = {**metrics, "total_time": pretrain_time + ft_time}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- Perceiver (from scratch) ----
    print("\n--- Perceiver IO (from scratch) ---")
    perc_scratch = PerceiverFNO1d(
        n_input=n_input_rd, n_output=n_output, width=64, modes=16, n_layers=4, n_latent=32
    )
    
    optimizer = optim.Adam(perc_scratch.parameters(), lr=1e-3)
    trainer = Trainer(perc_scratch, optimizer, device=device)
    
    start = time.time()
    trainer.train(rd_loader, test_loader, n_epochs=n_epochs_pretrain + n_epochs_finetune)
    elapsed = time.time() - start
    
    metrics = trainer.evaluate(test_loader)
    results["Perc. (scratch)"] = {**metrics, "total_time": elapsed}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary (Multi-Physics -> Reaction-Diffusion)")
    print("=" * 70)
    print(f"{'Model':<25} {'MSE':>15} {'NMAE (%)':>12} {'Time (s)':>12}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<25} "
            f"{metrics['mse']:>15.4e} "
            f"{metrics['nmae']*100:>12.4f} "
            f"{metrics['total_time']:>12.2f}"
        )
    
    return results


if __name__ == "__main__":
    results = run_multiphysics_experiment()
