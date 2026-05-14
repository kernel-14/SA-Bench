# MoE-POT: Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training

Reproduction of the paper "MoE-POT: Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training" by Wang et al.

## Overview

This repository implements the MoE-POT architecture — a sparse-activated neural operator for PDE pre-training that uses Mixture-of-Experts (MoE) to scale model capacity while controlling inference costs. The key innovation is the combination of:

1. **MoE Architecture**: 16 routed experts (top-K=4 selected dynamically per input) + 2 shared experts
2. **Router-Gating Network**: Layer-wise CNN-based router that learns to assign PDE inputs to specialized experts
3. **Load Balancing Loss**: Coefficient of Variation (CV) based auxiliary loss to encourage uniform expert utilization
4. **Fourier Layers**: Multi-head spectral mixing layers for efficient PDE operator learning
5. **Auto-regressive Denoising Pre-training**: Next-frame prediction with noise injection

## Repository Structure

```
moe_pot/
├── __init__.py           # Package initialization
├── model.py              # Main MoE-POT model & block architecture
├── moe_layer.py          # Mixture-of-Experts layer (core contribution)
├── fourier_layer.py      # Multi-head Fourier neural operator layer
├── patch_embed.py        # Patchification & temporal aggregation
├── training.py           # Training, fine-tuning, data loading utilities
├── data_utils.py         # PDE data preprocessing & synthetic data generation
└── interpretability.py   # Router-gating network analysis
configs/
├── moe_pot_tiny.yaml     # Tiny config (30M total, 17M activated)
├── moe_pot_small.yaml    # Small config (166M total, 90M activated)
└── moe_pot_medium.yaml   # Medium config (489M total, 288M activated)
scripts/
├── demo.py               # Comprehensive demo of all components
└── run_pretrain.py       # Pre-training script
```

## Installation

```bash
pip install torch numpy pyyaml
```

## Quick Start (Demo)

```bash
python scripts/demo.py
```

This runs through all major components:
1. Model creation at different scales (Tiny/Small/Medium)
2. Forward pass with synthetic PDE data
3. MoE layer details (routing, expert selection)
4. Noise injection for denoising pre-training
5. Data preprocessing pipeline
6. Interpretability analysis (dataset classification via router)

## Pre-Training

```bash
python scripts/run_pretrain.py --config configs/moe_pot_tiny.yaml
```

This pre-trains MoE-POT-Tiny on 6 synthetic PDE datasets using:
- Balanced dataset sampling (Appendix B.1)
- Auto-regressive denoising objective (Section 2.2)
- Load balancing loss (Section 4)
- One-cycle learning rate schedule (Appendix B.3)

## Model Architecture

### Overall Pipeline (Figure 3)

```
Input [B, T, C, H, W]
  -> PatchEmbed (Conv2D, stride=P) -> [B, T, dim, H/P, W/P]
  -> + Positional Encoding
  -> TemporalAggregation (MLP + Fourier features) -> [B, dim, H/P, W/P]
  -> N x MoEPOTBlock:
       -> FourierLayer (FFT -> MLP -> IFFT, multi-head)
       -> MoELayer (2 shared + top-4/16 routed experts)
  -> Output Projection (ConvTranspose2d + Conv) -> [B, C_out, H, W]
```

### MoE Layer (Core Contribution, Section 4)

```
MoELayer(z_0):
  1. Shared experts: Σ E_i^{(s)}(z_0) / N_s    (always activated, N_s=2)
  2. Router: Softmax(G(z_0)) -> top-K weights  (K=4 out of N_r=16)
  3. Routed experts: Σ w_k · E_{i_k}^{(r)}(z_0) (sparse activation)
  4. Output: shared_out + routed_out
```

### Load Balancing Loss

```
Importance_i = Σ_{b=1}^B w_{i,b}(x)    (sum over batch)
L_balance = w_bal · CV({Importance_i})^2
```

where CV is the coefficient of variation (std/mean), and w_bal = 0.1.

## Model Configurations (Table 5)

| Size   | dim | heads | layers | Total Params | Activated | Paper Reported |
|--------|-----|-------|--------|-------------|-----------|----------------|
| Tiny   | 512 | 4     | 4      | ~30M        | ~17M      | 30M / 17M      |
| Small  | 1024| 8     | 6      | ~166M       | ~90M      | 166M / 90M     |
| Medium | 1024| 8     | 8      | ~489M       | ~288M     | 489M / 288M    |

## Implemented Components

### Core Contributions
- [x] MoE architecture with 16 routed + 2 shared experts
- [x] Top-K=4 sparse routing via router-gating network (CNN-based)
- [x] Load balancing loss using coefficient of variation
- [x] Fourier layer with multi-head spectral mixing
- [x] Patchification and temporal aggregation
- [x] Auto-regressive denoising pre-training objective
- [x] Three model scales (Tiny, Small, Medium)

### Training Infrastructure
- [x] Balanced dataset sampling across heterogeneous PDE datasets
- [x] Noise injection for training stability
- [x] One-cycle learning rate schedule
- [x] Router freezing during fine-tuning
- [x] Pre-training, fine-tuning, and downstream task training loops

### Interpretability
- [x] Router-gating network analysis (Section 5.4)
- [x] Dataset classification via expert selection patterns (Appendix B.4)
- [x] Expert usage ratio computation per dataset
- [x] Cross-entropy distance metric for PDE type identification

### Data Handling
- [x] Spatial resolution standardization (H=128)
- [x] Channel padding for variable unification
- [x] Mask channel for irregular geometries
- [x] Synthetic PDE data generation for testing

## Key Formulas from the Paper

### Primary Loss Function (Section 4)
```
L = Σ_t ||G_w(u^{<t} + ε) - u^t||_2^2 + Σ_l L_balance^l
```

### L2 Relative Error (Evaluation Metric)
```
L2RE = ||pred - target||_2 / ||target||_2
```

### MoE Output (Equation 6)
```
z^{l+1}(x) = (1/N_s) Σ E_i^{l(s)}(z_0^l(x)) + Σ w_k^l · E_{i_k}^{l(r)}(z_0^l(x))
```

### Balanced Sampling (Appendix B.1)
```
p_k = w_k / (K · |D_k| · Σ_k w_k)
```

## Datasets

The paper uses 6 PDE datasets for pre-training:
1. FNO-NS (ν=1e-5) - Navier-Stokes with low viscosity
2. FNO-NS (ν=1e-3) - Navier-Stokes with higher viscosity
3. PDEBench-CNS (η=0.1, ζ=0.01) - Compressible Navier-Stokes
4. PDEBench-SWE - Shallow Water Equations
5. PDEBench-DR - Diffusion-Reaction
6. CFDBench - CFD with irregular geometries

Downstream tasks: NS(1e-4), CNS(1, 0.01), PDEArena

## Assumptions & Unresolved Details

1. **Expert CNN structure**: The paper states experts use CNNs to preserve spatial information (Appendix B.2), but doesn't specify exact architecture. We use a 2-layer residual Conv2D block with GELU activation.

2. **Router architecture**: The paper describes a CNN-based router-gating network. We implement it as AdaptiveAvgPool2d + linear projections.

3. **Fourier mode truncation**: We use `mode=32` as a default; the paper does not explicitly state this value but it's common in FNO-based architectures.

4. **Temporal aggregation**: The paper describes `e^{-iγt}` modulation. We use `cos(γt)` as a real-valued approximation since features are real-valued.

5. **MLP dimension**: Table 5 lists "MLP dim" but the exact role is unclear. We don't use a separate MLP dim, following the DPOT architecture where the MoE layer replaces the feed-forward network.

6. **Channel counts**: The actual number of channels per dataset depends on the specific data format. We support configurable in/out channels with padding.

7. **Output decoder**: The paper does not detail the output projection architecture. We use ConvTranspose2d to upsample back to original resolution.

## References

- DPOT [15]: Hao et al., "DPOT: Auto-regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training", ICML 2024
- DeepSeekMoE [8]: Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models", 2024
- AFNO [13]: Guibas et al., "Adaptive Fourier Neural Operators", 2021
- FNO [28]: Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", 2020
- Switch Transformer [11]: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models", 2022
