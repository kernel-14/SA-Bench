# Improving Consistency Models with Generator-Augmented Flows

This repository reproduces the paper "Improving Consistency Models with Generator-Augmented Flows" by Issenhuth et al.

## Overview

The paper proposes **Generator-Augmented Flows (GC)**, a novel approach to improve consistency model training by:

1. **Identifying a discrepancy** between consistency training (CT) and consistency distillation (CD) that persists in the continuous-time limit
2. **Proposing generator-augmented coupling (GC)** that uses the consistency model itself to construct better data-noise pairs
3. **Demonstrating** that GC reduces the proxy regularizer term and transport costs compared to independent coupling (IC) and minibatch OT
4. **Achieving improved FID** on CIFAR-10, ImageNet 32x32, CelebA 64x64, and LSUN Church 64x64

## Key Contributions

### Theorem 1: Discrepancy Between CT and CD
The paper proves that in the continuous-time limit:
```
lim_{N→∞} N^α [L_CT(θ) - L_CD(θ)] = C T^{α-1} R(θ)
```
where R(θ) > 0 is a regularization term that depends on the discrepancy between the velocity field and its Monte Carlo estimate.

### Generator-Augmented Coupling (GC)
The GC algorithm (Algorithm 1):
1. Sample x_star ~ p_star, z ~ p_z (IC)
2. Compute IC intermediate point: x_{t_i} = x_star + σ_{t_i} * z
3. Predict endpoint: x_hat_{t_i} = sg(f_θ(x_{t_i}, σ_{t_i}))
4. Sample mask m ~ Binomial(μ, batch_size)
5. Mix IC and GC: x_hat_{t_i} = m * x_hat_{t_i} + (1-m) * x_star
6. Compute GC intermediate points: x_tilde_{t_i} = x_hat_{t_i} + σ_{t_i} * z
7. Compute consistency loss on GC points

### Joint Learning
The joint learning loss combines IC and GC:
```
L_{GC-μ}(θ) = μ L_GC(θ) + (1-μ) L_CT(θ)
```

## Results

From Table 1 (one-step generation FID on various datasets):

| Dataset | iCT-IC | iCT-OT | iCT-GC (μ=0.5) |
|---------|--------|--------|-----------------|
| CIFAR-10 | 7.42 | 6.75 | **5.95** |
| ImageNet 32×32 | 14.89 | 14.13 | **13.99** |
| CelebA 64×64 | 15.82 | 13.63 | **11.74** |
| LSUN Church 64×64 | 10.58 | 9.71 | 9.88 |

## Repository Structure

```
.
├── models/
│   ├── __init__.py
│   ├── ncsnpp.py          # SongUNet / NCSNpp architecture
│   └── consistency_model.py  # Consistency model wrapper
├── training/
│   ├── __init__.py
│   ├── trainer.py         # Main trainer with IC, OT, and GC modes
│   └── schedules.py       # Noise/timestep schedules from iCT
├── evaluation/
│   ├── __init__.py
│   └── fid.py             # FID, KID, IS evaluation
├── analysis/
│   ├── __init__.py
│   ├── proxy_regularizer.py  # Figure 2: proxy regularizer comparison
│   └── transport_cost.py     # Figure 3: transport cost comparison
├── utils/
│   ├── __init__.py
│   └── lion_optimizer.py  # Lion optimizer
├── configs/
│   ├── cifar10_ic.yaml    # iCT-IC config
│   ├── cifar10_ot.yaml    # iCT-OT config
│   ├── cifar10_gc.yaml    # iCT-GC config (μ=0.5)
│   ├── imagenet32_gc.yaml # ImageNet 32x32 GC config
│   ├── celeba64_gc.yaml   # CelebA 64x64 GC config
│   └── lsun_church64_gc.yaml  # LSUN Church 64x64 GC config
├── train.py               # Main training script
├── evaluate.py            # Evaluation script
└── requirements.txt
```

## Usage

### Training

```bash
# Train iCT-IC (baseline)
python train.py --config configs/cifar10_ic.yaml

# Train iCT-OT (minibatch OT baseline)
python train.py --config configs/cifar10_ot.yaml

# Train iCT-GC (our method, μ=0.5)
python train.py --config configs/cifar10_gc.yaml

# Override mode and mu from command line
python train.py --config configs/cifar10_gc.yaml --mode GC --mu 0.3
```

### Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/cifar10_gc/checkpoint_100000.pt \
    --config configs/cifar10_gc.yaml \
    --num_samples 50000
```

### Analysis

```bash
# Compute proxy regularizer comparison (Figure 2)
python analysis/proxy_regularizer.py \
    --gc_checkpoint checkpoints/cifar10_gc/checkpoint_100000.pt \
    --data_dir ./data

# Compute transport cost comparison (Figure 3)
python analysis/transport_cost.py \
    --gc_checkpoint checkpoints/cifar10_gc/checkpoint_100000.pt \
    --gc_config configs/cifar10_gc.yaml \
    --data_dir ./data
```

## Implementation Details

### Architecture
- **Backbone**: SongUNet (NCSNpp) from Karras et al. (2022) / Song et al. (2021)
- **Parametrization**: c_skip(σ) * x + c_out(σ) * F_θ(x, σ) ensuring boundary condition
- **Optimizer**: Lion optimizer (Chen et al., 2023)

### Schedules (from Song & Dhariwal, 2024)
- **Noise schedule**: σ_i = (σ_0^(1/ρ) + i/N * (σ_N^(1/ρ) - σ_0^(1/ρ)))^ρ with ρ=7
- **Timestep schedule**: N(k) = min(s0 * 2^⌊k/K'⌋, s1) + 1 with s0=10, s1=1280
- **Loss weighting**: λ(σ_i) = 1 / (σ_{i+1} - σ_i)
- **Timestep sampling**: p(σ_i) ∝ erf differences

### Key Hyperparameters
- σ_min = 0.002, σ_max = 80.0
- P_mean = -1.1, P_std = 2.0
- EMA decay = 0.9999
- μ = 0.5 for CIFAR-10, ImageNet, CelebA; μ = 0.3 for ECT setting

## Assumptions and Unresolved Details

1. **Loss function**: The paper uses pseudo-Huber loss D(x,y) = sqrt(||x-y||^2 + c^2) - c. The exact value of c is not specified in the main text but we use c = 0.00054 * sqrt(d) following Song & Dhariwal (2024).

2. **EMA for GC**: The paper mentions using EMA model for GC endpoint prediction (ablation in Table 3 shows this is important). We use EMA by default.

3. **Dropout**: The paper finds that dropout hurts iCT-GC performance (opposite to iCT-IC). We set dropout=0 by default for GC.

4. **Learning rate search**: The paper performs a hyperparameter search over {0.0001, 0.00003} for CIFAR-10. We use 0.0001 as default.

5. **ECT setting**: The ECT (Easy Consistency Tuning) setting requires a pre-trained diffusion model. This is not implemented here as it requires additional infrastructure.

## Citation

```bibtex
@article{issenhuth2024improving,
  title={Improving Consistency Models with Generator-Augmented Flows},
  author={Issenhuth, Thibaut and Lee, Sangchul and Dos Santos, Ludovic and Franceschi, Jean-Yves and Kim, Chansoo and Rakotomamonjy, Alain},
  year={2024}
}
```
