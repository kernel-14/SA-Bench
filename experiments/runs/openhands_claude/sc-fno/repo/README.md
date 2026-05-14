# SC-FNO: Sensitivity-Constrained Fourier Neural Operators

Reproduction of "Sensitivity-Constrained Fourier Neural Operators for Forward and Inverse Problems in Parametric Differential Equations" (Behroozi, Shen, Kifer).

## Overview

SC-FNO enhances standard FNO training with a sensitivity loss term that supervises the Jacobians ∂û/∂p of the predicted solution with respect to physical parameters. This improves parameter inversion accuracy, robustness to input perturbations, and generalization with limited training data.

**Four model variants:**
| Variant | Loss |
|---------|------|
| FNO | L_u |
| FNO-PINN | L_u + L_eq |
| SC-FNO | L_u + L_s |
| SC-FNO-PINN | L_u + L_s + L_eq |

## Repository Structure

```
repo/
├── models/
│   ├── layers.py        # SpectralConv1d/2d/3d, FourierLayer1d/2d/3d
│   └── fno.py           # FNO1d, FNO2d, FNO3d architectures
├── equations/
│   ├── ode1.py          # Composite Harmonic Oscillator (analytical)
│   ├── ode2.py          # Duffing Oscillator (RK4)
│   ├── pde1.py          # Generalized Nonlinear Damped Wave Equation (RK4)
│   ├── pde2.py          # Forced Burgers' Equation (RK4, zoned variant)
│   ├── pde3.py          # Navier-Stokes vorticity-stream function (pseudo-spectral)
│   └── pde4.py          # Allen-Cahn Equation (RK4)
├── config.py            # All hyperparameters from Tables C.7, C.8, B.6
├── data.py              # Dataset classes and DataLoader construction
├── train.py             # Trainer with all 4 loss configurations (Algorithms 1-3)
├── evaluate.py          # R², relative L², parameter inversion
├── utils.py             # Model building, device management, utilities
├── generate_data.py     # Pre-generate and cache datasets
├── run_experiment.py    # Main experiment runner
└── requirements.txt
```

## Equations

| Name | Equation | Parameters | Domain |
|------|----------|------------|--------|
| ODE1 | du/dt = α sin(απt) + β cos(βπt) | α,β,γ | t∈[0,1], N=100 |
| ODE2 | Duffing oscillator | α,β,γ,δ,ω,ε,ζ | t∈[0,1], N=100 |
| PDE1 | Nonlinear damped wave | c,α,β,γ,ω | x,t∈[0,1], Sx=20, N=30 |
| PDE2 | Forced Burgers' | α,γ,δ,ω | x∈[0,1], t∈[0,π], Sx=40, N=30 |
| PDE2 (zoned) | Forced Burgers' with spatial zones | 82 params | same as PDE2 |
| PDE3 | Navier-Stokes (vorticity) | α,β | x,y∈[0,1], t∈[0,3], 64×64 |
| PDE4 | Allen-Cahn | c,α,β,ω,ε | x∈[0,1], t∈[0,1], Sx=40, N=30 |

## Usage

### 1. Generate datasets

```bash
# Generate ODE1 dataset (uses analytical solution)
python generate_data.py --equations ode1 --n_samples 2000

# Generate all datasets
python generate_data.py --all

# Use finite differences instead of AD
python generate_data.py --equations pde1 --use_fd
```

### 2. Train models

```bash
# Train FNO and SC-FNO on PDE1
python run_experiment.py --equation pde1 --variants FNO SC-FNO

# Train all variants on ODE1
python run_experiment.py --equation ode1 --variants FNO FNO-PINN SC-FNO SC-FNO-PINN

# High-dimensional zoned PDE2 (82 parameters)
python run_experiment.py --equation pde2_zoned_100 --variants FNO SC-FNO

# Use GPU
python run_experiment.py --equation pde1 --device cuda
```

### 3. Programmatic usage

```python
import torch
from equations import PDE1Solver
from data import PDE1DDataset, make_dataloaders
from models import FNO2d
from train import Trainer
from config import PDE1_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate data
solver = PDE1Solver(device=device)
data = solver.generate_dataset(n_samples=2000, use_ad=True)

# Build dataset
dataset = PDE1DDataset(
    params=data["params"], u=data["u"], jacobian=data["jacobian"], M=5
)
train_loader, val_loader, test_loader = make_dataloaders(dataset, batch_size=4)

# Build SC-FNO model (same architecture as FNO, different loss)
model = FNO2d(modes1=8, modes2=8, width=20, in_channels=3+5, out_channels=1)

# Train SC-FNO
trainer = Trainer(
    model=model,
    variant="SC-FNO",
    equation_type="pde1d",
    pinn_residual_fn=solver.pinn_residual,
    device=device,
)
history = trainer.train(train_loader, val_loader, n_epochs=500)
```

## Key Implementation Details

**Sensitivity loss** (Eq. 6): Computed via AD through the FNO at randomly sampled spatial-temporal points per epoch. The sampling varies between epochs to cover the full solution space (Section 2.4).

**Jacobian computation**: Both AD (via PyTorch autograd) and 4th-order finite differences are supported. AD is faster and more accurate (Table D.13).

**Parameter embedding**: Physical parameters p are repeated to match the spatial-temporal grid dimensions and concatenated with u, x, t as input channels to the FNO lifting layer (Section 2.4).

**Data split**: 70% train / 15% validation / 15% test, with validation and test sets containing parameter values not seen during training (Section 3.1).

## Hyperparameters (Table C.7)

All equations use: 4 Fourier layers, width=20, modes=8, lr=0.001, 500 epochs.

| Equation | Batch size | n_params | n_samples |
|----------|-----------|----------|-----------|
| ODE1/2 | 16 | 3/7 | 2000 |
| PDE1/2 | 4 | 5/4 | 2000 |
| PDE2 (zoned) | 1 | 82 | 100/500 |
| PDE3 | 4 | 2 | 1000 |
| PDE4 | 1 | 5 | 100/500 |
