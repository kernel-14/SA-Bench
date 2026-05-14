# Wavelet Diffusion Neural Operator (WDNO) - Reproduction

This repository contains a reproduction of the paper "Wavelet Diffusion Neural Operator (WDNO)" by Peiyan Hu et al.

## Overview

WDNO is a framework for PDE simulation and control that performs diffusion-based generative modeling in the wavelet domain with multi-resolution training. The key innovations are:

1. **Generation in the Wavelet Domain**: Diffusion in wavelet space handles abrupt changes effectively due to wavelet's space-frequency locality.
2. **Multi-Resolution Training**: Enables zero-shot super-resolution to finer grids via approximate scale invariance.

## Repository Structure

```
├── wdno/                        # Core library
│   ├── __init__.py              # Package exports
│   ├── wavelet_transform.py     # 1D, 2D, 3D wavelet transforms (pytorch_wavelets/ptwt)
│   ├── diffusion.py             # DDPM training + DDIM sampling
│   ├── unet.py                  # U-Net architectures (2D and 3D)
│   ├── wdno_base.py             # Base WDNO class (shared simulation & control logic)
│   ├── wdno_simulation.py       # WDNO for PDE simulation
│   ├── wdno_control.py          # WDNO for PDE control (energy-guided)
│   └── super_resolution.py      # Super-Resolution Model (SRM)
├── utils/                       # Utility modules
│   ├── __init__.py
│   └── data_generation.py       # PDE data generation & solvers
├── experiments/                 # Training scripts
│   ├── train_burgers_simulation.py
│   ├── train_burgers_control.py
├── configs/                     # Configuration YAML files
│   ├── burgers_simulation.yaml
│   ├── burgers_control.yaml
│   ├── cfd_simulation.yaml
│   ├── fluid_2d.yaml
│   └── era5.yaml
├── requirements.txt
└── README.md
```

## Core Contributions Reproduced

### 1. Diffusion in Wavelet Domain (Section 3.1)

The core WDNO class (`wdno_base.py`) implements:
- Wavelet encoding of spatiotemporal data using `pytorch_wavelets` (2D for 1D PDEs) and `ptwt` (3D for 2D PDEs)
- DDPM training with the denoising model predicting noise `ε_θ(W_u^{(k)}, W_a, k)`
- DDIM accelerated sampling (50 steps for 1D Burgers, 85 for CFD, 100 for 2D fluid)
- Classifier-free conditioning: the model conditions on wavelet-transformed equation parameters `W_a`

### 2. WDNO for Simulation (`wdno_simulation.py`, Section 3.1)

Learns `p(W_{u_{[0,T]}} | W_a)` - the conditional distribution of wavelet coefficients of the solution trajectory given wavelet-transformed parameters.

The denoising process (Eq. 3):
```
W_u^{(k-1)} = W_u^{(k)} - η · ε_θ(W_u^{(k)}, W_a, k) + ξ
```

### 3. WDNO for Control (`wdno_control.py`, Section 3.1)

Learns `p(W_{f_{[0,T]}} | W_a)` during training, then uses energy-based guidance during inference.

The guided denoising process (Eq. 4-5):
```
W_f^{(k-1)} = W_f^{(k)} - η(ε_θ(W_f^{(k)}, W_a, k) + λ ∇I(Ŵ_f^{(k)})) + ξ
```

where `Ŵ_f^{(k)}` is the predicted clean wavelet coefficients (Eq. 5):
```
Ŵ_f^{(k)} = (W_f^{(k)} - √(1-ᾱ_k) · ε_θ(W_f^{(k)}, W_a, k)) / √ᾱ_k
```

The objective `I` for 1D Burgers control (Eq. 6):
```
I = ∫_D |u(T,x) - u*(x)|² dx + α ∫_{[0,T]×D} |f(t,x)|² dt dx
```

### 4. Multi-Resolution Training (`super_resolution.py`, Section 3.2)

The Super-Resolution Model (SRM) implements:
- **Approximate scale invariance**: PDE dynamics follow approximately the same pattern across resolutions after rescaling
- **Multi-resolution dataset creation**: Downsampling the original data to create (hi_res, lo_res) pairs
- **SRM training**: Conditional diffusion model learning `p(W_h | W_l, W_{a_h})` with duplicated low-res coefficients
- **Zero-shot super-resolution inference**: Iteratively applying SRM to reach unseen finer resolutions

### 5. U-Net Architectures (`unet.py`)

Following the paper's specifications:
- **2D U-Net** (Table 18/19): init_dim=128, 4 down/up layers, kernel_size=3, dim_mult=[1,2,4,8], ResNet groups=8, attention heads=4, attention dim=32
- **3D U-Net** (Table 20): 3D convolutions with kernel (3,3,3), spatial-only downsampling with kernel (1,4,4) stride (1,2,2), 4 attention heads

### 6. Data Generation (`utils/data_generation.py`)

Implements the data generation procedures from:
- Appendix F.2: 1D Burgers' equation with Gaussian initial conditions and control forces
- Finite difference solver for Burgers' equation with high-resolution internal grid
- Support for loading PDEBench 1D CFD shock data
- ERA5 data preparation

## Experiments Covered

| Experiment | Config File | Training Script |
|---|---|---|
| 1D Burgers' Simulation (Table 1) | `configs/burgers_simulation.yaml` | `experiments/train_burgers_simulation.py` |
| 1D Burgers' Control (Table 2a) | `configs/burgers_control.yaml` | `experiments/train_burgers_control.py` |
| 1D CFD Simulation (Table 1) | `configs/cfd_simulation.yaml` | (requires PDEBench data) |
| 2D Fluid (Tables 1, 2b) | `configs/fluid_2d.yaml` | (requires fluid dataset) |
| ERA5 (Table 1) | `configs/era5.yaml` | (requires ERA5 data) |

## Key Hyperparameters (from paper)

### 1D Burgers' Equation (Tables 18)
- Wavelet: `bior2.4`, mode: `periodization` (via pytorch_wavelets)
- U-Net: init_dim=128, dim_mult=[1,2,4,8], groups=8, attn_heads=4, attn_dim=32
- Batch size: 16, Optimizer: Adam, LR: 1e-4
- Training steps: 190,000, LR schedule: cosine annealing
- DDIM: 50 iterations, η=1
- Control guidance weight λ: 120,000 (cosine schedule)

### 1D CFD (Tables 19)
- Same architecture as Burgers
- DDIM: 85 iterations, η=0

### 2D Fluid (Tables 20)
- Wavelet: `bior1.3`, mode: `zero` (via ptwt)
- 3D U-Net with spatial-only downsampling
- DDIM: 100 iterations, η=1
- Guidance weight: 1100

## Wavelet Transform Details

### 1D PDEs (Burgers, Advection, CFD)
- 2D wavelet transform on (time × space) data
- Input: `(B, C, 81, 120)` → Output: `(B, 4C, 41, 60)`
- 1 coarse (LL) + 3 detail (LH, HL, HH) coefficient sets

### 2D PDEs (Fluid, ERA5)
- 3D wavelet transform on (time × height × width) data
- Input: `(B, C, 32, 64, 64)` → Output: `(B, 8C, 18, 34, 34)`
- 1 coarse (LLL) + 7 detail coefficient sets

## Assumptions & Unresolved Details

1. **Wavelet reconstruction**: The 3D wavelet transform implementation relies on `ptwt` for inverse transforms. Some details about the exact 3D subband ordering may differ from the original implementation.

2. **Data duplication for boundary alignment**: When duplicating low-resolution data to match high-resolution dimensions (Section 3.2), we replicate the last temporal/spatial dimension for odd-sized dimensions, following the paper's description.

3. **Initial condition conditioning**: For 1D initial conditions, we expand them to match the full spatiotemporal dimensions before the wavelet transform, as described in Section 4.1.

4. **Classifier-free guidance**: The paper mentions using both classifier-based and classifier-free guidance. Our implementation focuses on classifier-free guidance with optional guidance weighting.

5. **Super-resolution training**: The SRM training needs a separate UNet with increased conditioning channels (to accommodate both low-res data and equation parameters). Our implementation handles this dynamically.

6. **Solver integration**: For control guidance, the paper uses a ground-truth solver to evaluate the objective I. Our implementation includes a basic FDM solver, but the exact solver configuration (16× internal refinement) may differ from the original.

7. **Training data sizes**: The paper uses 40,000 training trajectories for 1D Burgers. The exact dataset split ratios for validation are not specified.

8. **Ablation studies**: The Fourier-domain diffusion baseline (Diffusion + FFT) and FNO denoiser are not included in this reproduction.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train 1D Burgers' simulation
python experiments/train_burgers_simulation.py \
    --n_samples 40000 \
    --batch_size 16 \
    --n_epochs 100 \
    --device cuda

# Train 1D Burgers' control
python experiments/train_burgers_control.py \
    --n_samples 40000 \
    --batch_size 16 \
    --n_epochs 100 \
    --guidance_weight 120000 \
    --device cuda
```

## References

- Original paper: "Wavelet Diffusion Neural Operator (WDNO)" by Peiyan Hu et al.
- DDPM: Ho et al. (2020), "Denoising Diffusion Probabilistic Models"
- DDIM: Song et al. (2020), "Denoising Diffusion Implicit Models"
- pytorch_wavelets: Cotter (2019)
- ptwt: Wolter et al. (2024), "Pytorch Wavelet Toolbox"
