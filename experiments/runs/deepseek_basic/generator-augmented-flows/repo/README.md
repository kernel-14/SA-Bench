# Improving Consistency Models with Generator-Augmented Flows

Reproduction of the paper *"Improving Consistency Models with Generator-Augmented Flows"* by Thibaut Issenhuth et al.

## Overview

This repository implements the core contributions of the paper:

1. **Theorem 1**: Theoretical proof that consistency training (CT) and consistency distillation (CD) objectives differ in the continuous-time limit, with a closed-form expression for the discrepancy $R(\theta)$.

2. **Generator-Augmented Coupling (GC)**: A novel data-noise coupling that uses a consistency model's own predictions to construct improved probability paths. GC reduces the discrepancy proxy $\tilde{R}_t$ and lowers transport costs compared to independent coupling (IC) and minibatch optimal transport (batch-OT).

3. **Joint Learning Algorithm**: Practical training procedure (Algorithm 1) that mixes IC and GC trajectories with a mixing parameter $\mu$, enabling faster convergence and better final performance.

4. **Empirical Validation**: Configurations and training code for CIFAR-10, ImageNet 32×32, CelebA 64×64, and LSUN Church 64×64, with support for both iCT (Song & Dhariwal, 2024) and ECT (Geng et al., 2024) settings.

## Repository Structure

```
.
├── consistency_models/          # Core library
│   ├── __init__.py              # Package exports
│   ├── model.py                 # Consistency model with SongUNet architecture
│   ├── coupling.py              # IC, OT, GC coupling implementations
│   ├── losses.py                # CD, CT, GC, and joint learning losses
│   ├── training.py              # Training loops, EMA, Lion optimizer
│   ├── scheduling.py            # Noise schedules, discretization, timestep sampling
│   ├── metrics.py               # FID, KID, IS evaluation
│   └── theory.py                # Regularizer computation, transport cost analysis
├── configs/                     # Configuration files per dataset
│   ├── cifar10_ict.py           # CIFAR-10 iCT config (Table 4)
│   ├── imagenet32_ict.py        # ImageNet 32×32 iCT config (Table 6)
│   └── celeba64_ict.py          # CelebA 64×64 iCT config (Table 5)
├── train.py                     # Main training script
└── README.md                    # This file
```

## Key Mathematical Formulations

### Consistency Model Parameterization (Eq. 3)

$$f_\theta(x_t, \sigma_t) = c_{\text{skip}}(\sigma_t) x_t + c_{\text{out}}(\sigma_t) F_\theta(x_t, \sigma_t)$$

where $c_{\text{skip}}(\sigma) = \frac{\sigma_d^2}{\sigma_d^2 + (\sigma - \sigma_0)^2}$ and $c_{\text{out}}(\sigma) = \frac{\sigma_d(\sigma - \sigma_0)}{\sqrt{\sigma_d^2 + \sigma^2}}$.

### Generator-Augmented Coupling (Eqs. 13-14)

From IC samples $(x_*, z) \sim q_I$ with $x_{t_i} = x_* + \sigma_{t_i}z$:

1. Predict endpoint: $\hat{x}_{t_i} = \text{sg}(f_\theta(x_{t_i}, \sigma_{t_i}))$
2. Construct GC points: $\tilde{x}_{t_i} = \hat{x}_{t_i} + \sigma_{t_i}z$, $\tilde{x}_{t_{i+1}} = \hat{x}_{t_i} + \sigma_{t_{i+1}}z$

### Joint Learning Loss (Eq. 16)

$$\mathcal{L}_{\text{GC-}\mu}(\theta) = \mu \mathcal{L}_{\text{GC}}(\theta) + (1-\mu) \mathcal{L}_{\text{CT}}(\theta)$$

## Usage

### Training

```bash
# Train with independent coupling (baseline)
python train.py --config configs/cifar10_ict.py --coupling ic

# Train with minibatch optimal transport
python train.py --config configs/cifar10_ict.py --coupling ot

# Train with generator-augmented coupling (our method)
python train.py --config configs/cifar10_ict.py --coupling gc --mu 0.5
```

### Evaluation

```bash
python train.py --config configs/cifar10_ict.py --eval_only \
    --resume output/checkpoints/model_final.pt
```

### Using the Library

```python
from consistency_models import (
    ConsistencyModel, SongUNet,
    ConsistencyTrainingConfig, train_consistency_model,
    noise_schedule_karras, weighting_function,
)

# Build model
network = SongUNet(img_resolution=32, in_channels=3, out_channels=3,
                    model_channels=128, channel_mult=[1, 2, 2])
model = ConsistencyModel(network)

# Configure training
config = ConsistencyTrainingConfig(
    batch_size=512, total_steps=100000,
    coupling="gc", gc_mu=0.5,
)

# Train
stats = train_consistency_model(model, dataloader, config)
```

## Theoretical Components

### Theorem 1 (Discrepancy)

The function `consistency_models.theory.compute_regularizer_discrepancy()` computes $R(\theta)$, the regularizer term that measures the gap between CT and CD in the continuous-time limit.

### Proxy Term (Section 4.2.1)

$\tilde{R}_t = \mathbb{E}[\|\dot{x}_t - v_t(x_t)\|^2]$ is computed via `coupling.compute_r_proxy()` using a denoiser network.

### Transport Cost (Section 4.2.2)

$c(t) = \mathbb{E}[\|f(x_t, \sigma_t) - z\|^2]$ is computed via `theory.compute_transport_cost()`.

## Configuration Hyperparameters

| Parameter | CIFAR-10 | ImageNet 32 | CelebA 64 |
|-----------|----------|-------------|-----------|
| Batch size | 512 | 512 | 128 |
| Training steps | 100,000 | 150,000 | 150,000 |
| Learning rate | {1e-4, 3e-5} | 8e-5 | 8e-5 |
| Model channels | 128 | 128 | 128 |
| Channel mult | [1, 2, 2] | [1, 1, 2] | [1, 2, 2, 2] |
| Num blocks | 3 | [3, 5, 7] | [3, 3, 4, 5] |
| Attention | None | [16] | None |
| $s_0$ / $s_1$ | 10 / 1280 | 10 / 1280 | 10 / 1280 |
| $\sigma_0$ / $\sigma_{\max}$ | 0.002 / 80 | 0.002 / 80 | 0.002 / 80 |
| $\rho$ | 7 | 7 | 7 |

## Assumptions and Known Limitations

1. **Architecture**: This implementation uses a clean-room SongUNet based on the NCSN++/EDM architecture. The exact architecture from EDM (https://github.com/NVlabs/edm) may have additional features (e.g., per-resolution block counts, sophisticated resampling filters) not fully replicated here without access to the original codebase.

2. **Score Model**: The CD loss requires a pre-trained score/diffusion model. This implementation provides the CD loss function but assumes an external score model interface for the velocity field.

3. **ECT Setting**: The ECT experiments (Section 5.3) fine-tune from a pre-trained diffusion model. This repo provides the GC-ECT loss but assumes an external pre-trained model checkpoint.

4. **$\mu$ Sensitivity**: The paper finds $\mu = 0.5$ works well for iCT and $\mu = 0.3$ for ECT. We follow these recommendations.

5. **Dropout**: The paper finds dropout hurts GC performance (Table 3), opposite to IC findings. We default to no dropout in GC configurations.

6. **Theoretical Verifications**: The proxy term computation requires a trained denoiser for each coupling type. We provide the infrastructure but training separate denoisers for each coupling is left to the user.

## Dependencies

- PyTorch >= 2.0
- torchvision
- numpy
- scipy (for Hungarian OT solver)
- (optional) torchmetrics (for alternative metric implementations)

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{issenhuth2024improving,
  title={Improving Consistency Models with Generator-Augmented Flows},
  author={Issenhuth, Thibaut and Lee, Sangchul and Dos Santos, Ludovic and 
          Franceschi, Jean-Yves and Kim, Chansoo and Rakotomamonjy, Alain},
  year={2024}
}
```

## References

- Song et al. (2023) "Consistency Models"
- Song and Dhariwal (2024) "Improved Techniques for Training Consistency Models"
- Karras et al. (2022) "Elucidating the Design Space of Diffusion-Based Generative Models"
- Pooladian et al. (2023) "Multisample Flow Matching: Straightening Flows with Minibatch Couplings"
- Geng et al. (2024) "Consistency Models Made Easy" (ECT)
- Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms" (Lion optimizer)
