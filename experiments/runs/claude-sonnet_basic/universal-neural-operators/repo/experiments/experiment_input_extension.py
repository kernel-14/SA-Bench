"""
Experiment 2: Input function set extension scenario.

From the paper:
"To assess the applicability of the adapter-based approach, several experiments were
conducted on scenarios where the equations were extended with additional terms.
Here, for fine-tuning, we added convection to the heat equation and extended
reaction-diffusion equations with advection."

This script:
1. Pretrains on heat equation (inputs: u0, alpha)
2. Fine-tunes on heat+convection (inputs: u0, alpha, v) using new adapters
3. Compares with training from scratch on the extended problem
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
from datasets import HeatEquationDataset, ReactionDiffusionDataset
from utils import Trainer, compute_metrics
from utils.transfer import freeze_backbone, create_new_adapters, print_param_summary


def run_heat_extension_experiment(
    n_pretrain: int = 800,
    n_finetune: int = 200,
    n_test: int = 200,
    n_epochs_pretrain: int = 100,
    n_epochs_finetune: int = 50,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "checkpoints/input_extension",
):
    """
    Run heat equation input extension experiment.
    
    Pretrain on heat equation (n_input=2), fine-tune on heat+convection (n_input=3).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Heat Equation - Input Function Set Extension")
    print("=" * 60)
    
    # Generate datasets
    print("\nGenerating pretraining data (heat equation, n_input=2)...")
    pretrain_dataset = HeatEquationDataset(
        n_samples=n_pretrain + 100,
        convection_range=None,  # Pure heat equation
        seed=42,
    )
    pretrain_train, pretrain_val = random_split(pretrain_dataset, [n_pretrain, 100])
    
    print("Generating fine-tuning data (heat+convection, n_input=3)...")
    finetune_dataset = HeatEquationDataset(
        n_samples=n_finetune + n_test,
        convection_range=(-1.0, 1.0),  # With convection
        seed=123,
    )
    finetune_train, finetune_test = random_split(finetune_dataset, [n_finetune, n_test])
    
    pretrain_loader = DataLoader(pretrain_train, batch_size=batch_size, shuffle=True)
    pretrain_val_loader = DataLoader(pretrain_val, batch_size=batch_size)
    finetune_loader = DataLoader(finetune_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(finetune_test, batch_size=batch_size)
    
    n_input_pretrain = pretrain_dataset.n_input  # 2
    n_input_finetune = finetune_dataset.n_input  # 3
    n_output = pretrain_dataset.n_output  # 1
    
    print(f"Pretrain n_input: {n_input_pretrain}, Finetune n_input: {n_input_finetune}")
    
    results = {}
    
    # ---- FNO Baseline (from scratch on extended problem) ----
    print("\n--- FNO (from scratch on extended problem) ---")
    fno_scratch = FNO1d(n_input=n_input_finetune, n_output=n_output, width=64, modes=16, n_layers=4)
    
    optimizer = optim.Adam(fno_scratch.parameters(), lr=1e-3)
    trainer = Trainer(fno_scratch, optimizer, device=device)
    
    start = time.time()
    trainer.train(finetune_loader, test_loader, n_epochs=n_epochs_pretrain + n_epochs_finetune)
    elapsed = time.time() - start
    
    metrics = trainer.evaluate(test_loader)
    results["FNO (scratch)"] = {**metrics, "total_time": elapsed}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- MambaFNO (pretrained + new adapters) ----
    print("\n--- MambaFNO (pretrained + new adapters) ---")
    # Pretrain on heat equation
    mamba_pretrain = MambaFNO1d(
        n_input=n_input_pretrain, n_output=n_output, width=64, modes=16, n_layers=4
    )
    
    optimizer = optim.Adam(mamba_pretrain.parameters(), lr=1e-3)
    trainer = Trainer(mamba_pretrain, optimizer, device=device)
    
    print("  Pretraining on heat equation...")
    start_pretrain = time.time()
    trainer.train(pretrain_loader, pretrain_val_loader, n_epochs=n_epochs_pretrain)
    pretrain_time = time.time() - start_pretrain
    
    torch.save(mamba_pretrain.state_dict(), os.path.join(save_dir, "mamba_heat_pretrained.pt"))
    
    # Create new adapters for extended problem (different n_input)
    mamba_ft = create_new_adapters(mamba_pretrain, n_input_finetune, n_output)
    print_param_summary(mamba_ft, "MambaFNO (fine-tuning with new adapters)")
    
    adapter_params = [p for p in mamba_ft.parameters() if p.requires_grad]
    optimizer_ft = optim.Adam(adapter_params, lr=1e-3)
    trainer_ft = Trainer(mamba_ft, optimizer_ft, device=device)
    
    print("  Fine-tuning with new adapters...")
    start_ft = time.time()
    trainer_ft.train(finetune_loader, test_loader, n_epochs=n_epochs_finetune)
    ft_time = time.time() - start_ft
    
    metrics = trainer_ft.evaluate(test_loader)
    results["Mamba FNO (pretr.)"] = {**metrics, "total_time": pretrain_time + ft_time}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- MambaFNO (from scratch on extended problem) ----
    print("\n--- MambaFNO (from scratch on extended problem) ---")
    mamba_scratch = MambaFNO1d(
        n_input=n_input_finetune, n_output=n_output, width=64, modes=16, n_layers=4
    )
    
    optimizer = optim.Adam(mamba_scratch.parameters(), lr=1e-3)
    trainer = Trainer(mamba_scratch, optimizer, device=device)
    
    start = time.time()
    trainer.train(finetune_loader, test_loader, n_epochs=n_epochs_pretrain + n_epochs_finetune)
    elapsed = time.time() - start
    
    metrics = trainer.evaluate(test_loader)
    results["Mamba FNO (scratch)"] = {**metrics, "total_time": elapsed}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # ---- Perceiver (pretrained + new adapters) ----
    print("\n--- Perceiver IO (pretrained + new adapters) ---")
    perc_pretrain = PerceiverFNO1d(
        n_input=n_input_pretrain, n_output=n_output, width=64, modes=16, n_layers=4, n_latent=32
    )
    
    optimizer = optim.Adam(perc_pretrain.parameters(), lr=1e-3)
    trainer = Trainer(perc_pretrain, optimizer, device=device)
    
    print("  Pretraining...")
    start_pretrain = time.time()
    trainer.train(pretrain_loader, pretrain_val_loader, n_epochs=n_epochs_pretrain)
    pretrain_time = time.time() - start_pretrain
    
    perc_ft = create_new_adapters(perc_pretrain, n_input_finetune, n_output)
    adapter_params = [p for p in perc_ft.parameters() if p.requires_grad]
    optimizer_ft = optim.Adam(adapter_params, lr=1e-3)
    trainer_ft = Trainer(perc_ft, optimizer_ft, device=device)
    
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
        n_input=n_input_finetune, n_output=n_output, width=64, modes=16, n_layers=4, n_latent=32
    )
    
    optimizer = optim.Adam(perc_scratch.parameters(), lr=1e-3)
    trainer = Trainer(perc_scratch, optimizer, device=device)
    
    start = time.time()
    trainer.train(finetune_loader, test_loader, n_epochs=n_epochs_pretrain + n_epochs_finetune)
    elapsed = time.time() - start
    
    metrics = trainer.evaluate(test_loader)
    results["Perc. (scratch)"] = {**metrics, "total_time": elapsed}
    print(f"MSE: {metrics['mse']:.4e}, NMAE: {metrics['nmae']*100:.4f}%")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary (Heat Equation - Input Extension)")
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
    results = run_heat_extension_experiment()
