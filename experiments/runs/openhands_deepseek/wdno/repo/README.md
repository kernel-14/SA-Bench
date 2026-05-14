# Wavelet Diffusion Neural Operator (WDNO)

Official implementation of the paper "Wavelet Diffusion Neural Operator" (Hu et al.).

WDNO introduces two key innovations:
1. **Generation in the wavelet domain** - Performs diffusion-based generative modeling in the wavelet space to handle abrupt changes and long-term dependencies
2. **Multi-resolution training** - Enables zero-shot super-resolution to finer spatial and temporal resolutions

## Codebase Structure

```
repo/
├── config.yaml              # Hyperparameter configuration
├── requirements.txt         # Dependencies
├── README.md                # This file
├── wdno/                    # Core WDNO implementation
│   ├── __init__.py
│   ├── model.py             # WDNO1D, WDNO2D, SuperResolutionModel
│   ├── diffusion.py         # DDPM diffusion with DDIM sampling
│   ├── modules.py           # UNet2D, UNet3D architectures
│   ├── layers.py            # Building blocks (ResNet, Attention, etc.)
│   ├── wavelet_utils.py     # Wavelet transforms (1D, 2D, 3D)
│   ├── train.py             # Training loops (BRM, SRM, Control)
│   └── evaluate.py          # Evaluation metrics
├── baselines/               # Baseline implementations
│   ├── __init__.py
│   ├── fno.py               # Fourier Neural Operator
│   ├── wno.py               # Wavelet Neural Operator
│   ├── cnn.py               # CNN / U-Net baselines
│   ├── oformer.py           # Operator Transformer
│   └── mwt.py               # Multiwavelet Neural Operator
├── data/                    # Data generation and loading
│   ├── data_generation.py   # PDE solvers and data generation
│   └── dataset.py           # PyTorch Datasets and DataLoaders
└── experiments/             # Experiment scripts
    └── run_all.py           # Run all experiments from paper
```

## Experiments

The paper evaluates WDNO on five physical systems:

| Experiment | Type | Description |
|---|---|---|
| 1D Burgers' eq. | Sim + Ctrl | Shock waves, turbulence (81 steps) |
| 1D Advection eq. | Sim | Smooth advection dynamics |
| 1D Navier-Stokes | Sim | Compressible fluid with shocks |
| 2D Fluid | Sim + Ctrl | Incompressible flow, smoke in maze |
| ERA5 | Sim | Real-world weather data |

## Key Algorithms

### Base-Resolution Model (BRM)
- Learns conditional distribution p(W_u | W_a) in wavelet domain
- Uses DDPM with classifier-free guidance
- For simulation: Maps (u0, f) -> u_{[0,T]}
- For control: Maps (u0, uT) -> f_{[0,T]} with energy-based guidance

### Super-Resolution Model (SRM)
- Learns p(W_h | W_l, W_{a_h}) for multi-resolution pairs
- Enables zero-shot super-resolution during inference
- Trained on downsampled data following approximate scale invariance

### Control (Eq. 4-5)
- Energy-based guidance: lambda * grad_{W_f} J(W_hat_f)
- W_hat_f estimated from noisy sample using Eq. 5
- DDIM sampling with guidance for efficient generation

## Usage

Train and evaluate the Base-Resolution Model:
```python
from wdno.train import train_brm
model = train_brm(config, dataset_name='burgers', task='simulation')
```

Zero-shot super-resolution during inference:
```python
from wdno.model import WDNO1D, SuperResolutionModel
# Generate at base resolution, then super-resolve iteratively
```

## Hyperparameters

Key hyperparameters (from paper):
- UNet: init_dim=128, 4 down/up layers, dim_mult=[1,2,4,8], 8 resnet groups
- Diffusion: 1000 timesteps, linear schedule (beta: 1e-4 to 0.02)
- DDIM: 50 sampling steps, eta=1.0
- Training: lr=1e-4, 190k steps, cosine annealing, batch_size=16
- Control guidance: lambda=120000, cosine schedule

## References

- Ho et al. "Denoising Diffusion Probabilistic Models" (2020)
- Song et al. "Denoising Diffusion Implicit Models" (2020)
- Tripura & Chakraborty "Wavelet Neural Operator" (2022)
- Li et al. "Fourier Neural Operator" (2021)
