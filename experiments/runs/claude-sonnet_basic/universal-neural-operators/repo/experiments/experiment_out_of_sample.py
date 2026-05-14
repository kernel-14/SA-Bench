"""
Experiment 1: Out-of-sample parameter values scenario.

From the paper:
"First, we conducted several experiments on cases where the pretraining equations
and fine-tuning ones differed only in the coefficient values. The experiments were
conducted using Burgers' equation, the Gray-Scott model of the reaction-diffusion
process, and the Navier-Stokes equations for an incompressible flow."

This script:
1. Pretrains models on one range of parameters
2. Fine-tunes on a different (out-of-sample) range
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

from models import FNO1d, MambaFNO1d, PerceiverFNO1d, CoDANO1d, LocalAttnFNO1d
from datasets import BurgersDataset
from utils import Trainer, compute_metrics
from utils.transfer import freeze_backbone, create_new_adapters, print_param_summary


def run_burgers_experiment(
    n_pretrain: int = 800,
    n_finetune: int = 200,
    n_test: int = 200,
    n_epochs_pretrain: int = 100,
    n_epochs_finetune: int = 50,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "checkpoints/out_of_sample",
):
    """
    Run Burgers' equation out-of-sample experiment.
    
    Pretrain on nu in [0.001, 0.05], fine-tune on nu in [0.05, 0.1].
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Burgers' Equation - Out-of-Sample Parameter Values")
    print("=" * 60)
    
    # Generate datasets
    print("\nGenerating pretraining data (nu in [0.001, 0.05])...")
    pretrain_dataset = BurgersDataset(
        n_samples=n_pretrain + 100,
        nu_range=(0.001, 0.05),
        seed=42,
    )
    pretrain_train, pretrain_val = random_split(
        pretrain_dataset, [n_pretrain, 100]
    )
    
    print("Generating fine-tuning data (nu in [0.05, 0.1])...")
    finetune_dataset = BurgersDataset(
        n_samples=n_finetune + n_test,
        nu_range=(0.05, 0.1),
        seed=123,
    )
    finetune_train, finetune_test = random_split(
        finetune_dataset, [n_finetune, n_test]
    )
    
    pretrain_loader = DataLoader(pretrain_train, batch_size=batch_size, shuffle=True)
    pretrain_val_loader = DataLoader(pretrain_val, batch_size=batch_size)
    finetune_loader = DataLoader(finetune_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(finetune_test, batch_size=batch_size)
    
    n_input = pretrain_dataset.n_input
    n_output = pretrain_dataset.n_output
    
    results = {}
    
    # ---- FNO Baseline (from scratch) ----
    print("\n--- FNO (from scratch) ---")
    fno_scratch = FNO1d(n_input=n_input, n_output=n_output, width=64, modes=16, n_layers=4)
    print_param_summary(fno_scratch, "FNO")
    
    optimizer = optim.Adam(fno_scratch.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    trainer = Trainer(fno_scratch, optimizer, scheduler, device)
    
    start = time.time()
    trainer.train(finetune_loader, test_loader, n_epochs=n_epochs_pretrain + n_epochs_finetune)
    elapsed = time.time() - start
    
    metrics = trainer.evaluate(test_loader)
    results["FNO (scratch)"] = {**metrics, "total_time": elapsed}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- MambaFNO (pretrained + fine-tuned) ----
    print("\n--- MambaFNO (pretrained) ---")
    mamba_pretrain = MambaFNO1d(n_input=n_input, n_output=n_output, width=64, modes=16, n_layers=4)
    
    optimizer = optim.Adam(mamba_pretrain.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    trainer = Trainer(mamba_pretrain, optimizer, scheduler, device)
    
    print("  Pretraining...")
    start_pretrain = time.time()
    trainer.train(pretrain_loader, pretrain_val_loader, n_epochs=n_epochs_pretrain)
    pretrain_time = time.time() - start_pretrain
    
    torch.save(mamba_pretrain.state_dict(), os.path.join(save_dir, "mamba_pretrained.pt"))
    
    # Fine-tune: freeze backbone, train only adapters
    freeze_backbone(mamba_pretrain)
    print_param_summary(mamba_pretrain, "MambaFNO (fine-tuning)")
    
    adapter_params = [p for p in mamba_pretrain.parameters() if p.requires_grad]
    optimizer_ft = optim.Adam(adapter_params, lr=1e-3)
    trainer_ft = Trainer(mamba_pretrain, optimizer_ft, device=device)
    
    print("  Fine-tuning...")
    start_ft = time.time()
    trainer_ft.train(finetune_loader, test_loader, n_epochs=n_epochs_finetune)
    ft_time = time.time() - start_ft
    
    metrics = trainer_ft.evaluate(test_loader)
    results["Mamba FNO (pretr.)"] = {**metrics, "total_time": pretrain_time + ft_time}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- MambaFNO (from scratch) ----
    print("\n--- MambaFNO (from scratch) ---")
    mamba_scratch = MambaFNO1d(n_input=n_input, n_output=n_output, width=64, modes=16, n_layers=4)
    
    optimizer = optim.Adam(mamba_scratch.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    trainer = Trainer(mamba_scratch, optimizer, scheduler, device)
    
    start = time.time()
    trainer.train(finetune_loader, test_loader, n_epochs=n_epochs_pretrain + n_epochs_finetune)
    elapsed = time.time() - start
    
    metrics = trainer.evaluate(test_loader)
    results["Mamba FNO (scratch)"] = {**metrics, "total_time": elapsed}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- Perceiver (pretrained + fine-tuned) ----
    print("\n--- Perceiver IO (pretrained) ---")
    perc_pretrain = PerceiverFNO1d(
        n_input=n_input, n_output=n_output, width=64, modes=16, n_layers=4, n_latent=32
    )
    
    optimizer = optim.Adam(perc_pretrain.parameters(), lr=1e-3)
    trainer = Trainer(perc_pretrain, optimizer, device=device)
    
    print("  Pretraining...")
    start_pretrain = time.time()
    trainer.train(pretrain_loader, pretrain_val_loader, n_epochs=n_epochs_pretrain)
    pretrain_time = time.time() - start_pretrain
    
    freeze_backbone(perc_pretrain)
    adapter_params = [p for p in perc_pretrain.parameters() if p.requires_grad]
    optimizer_ft = optim.Adam(adapter_params, lr=1e-3)
    trainer_ft = Trainer(perc_pretrain, optimizer_ft, device=device)
    
    print("  Fine-tuning...")
    start_ft = time.time()
    trainer_ft.train(finetune_loader, test_loader, n_epochs=n_epochs_finetune)
    ft_time = time.time() - start_ft
    
    metrics = trainer_ft.evaluate(test_loader)
    results["Perc. (pretr.)"] = {**metrics, "total_time": pretrain_time + ft_time}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- Perceiver (from scratch) ----
    print("\n--- Perceiver IO (from scratch) ---")
    perc_scratch = PerceiverFNO1d(
        n_input=n_input, n_output=n_output, width=64, modes=16, n_layers=4, n_latent=32
    )
    
    optimizer = optim.Adam(perc_scratch.parameters(), lr=1e-3)
    trainer = Trainer(perc_scratch, optimizer, device=device)
    
    start = time.time()
    trainer.train(finetune_loader, test_loader, n_epochs=n_epochs_pretrain + n_epochs_finetune)
    elapsed = time.time() - start
    
    metrics = trainer.evaluate(test_loader)
    results["Perc. (scratch)"] = {**metrics, "total_time": elapsed}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- CoDA-NO (pretrained + fine-tuned) ----
    print("\n--- CoDA-NO (pretrained) ---")
    coda_pretrain = CoDANO1d(n_input=n_input, n_output=n_output, width=64, modes=16, n_layers=4)
    
    optimizer = optim.Adam(coda_pretrain.parameters(), lr=1e-3)
    trainer = Trainer(coda_pretrain, optimizer, device=device)
    
    print("  Pretraining...")
    start_pretrain = time.time()
    trainer.train(pretrain_loader, pretrain_val_loader, n_epochs=n_epochs_pretrain)
    pretrain_time = time.time() - start_pretrain
    
    freeze_backbone(coda_pretrain)
    adapter_params = [p for p in coda_pretrain.parameters() if p.requires_grad]
    optimizer_ft = optim.Adam(adapter_params, lr=1e-3)
    trainer_ft = Trainer(coda_pretrain, optimizer_ft, device=device)
    
    print("  Fine-tuning...")
    start_ft = time.time()
    trainer_ft.train(finetune_loader, test_loader, n_epochs=n_epochs_finetune)
    ft_time = time.time() - start_ft
    
    metrics = trainer_ft.evaluate(test_loader)
    results["CoDA-NO (pretr.)"] = {**metrics, "total_time": pretrain_time + ft_time}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- CoDA-NO (from scratch) ----
    print("\n--- CoDA-NO (from scratch) ---")
    coda_scratch = CoDANO1d(n_input=n_input, n_output=n_output, width=64, modes=16, n_layers=4)
    
    optimizer = optim.Adam(coda_scratch.parameters(), lr=1e-3)
    trainer = Trainer(coda_scratch, optimizer, device=device)
    
    start = time.time()
    trainer.train(finetune_loader, test_loader, n_epochs=n_epochs_pretrain + n_epochs_finetune)
    elapsed = time.time() - start
    
    metrics = trainer.evaluate(test_loader)
    results["CoDA-NO (scratch)"] = {**metrics, "total_time": elapsed}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("Results Summary (Burgers' Equation - Out-of-Sample Parameters)")
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
    results = run_burgers_experiment()
