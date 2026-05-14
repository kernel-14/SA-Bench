# Adjoint Matching: Fine-tuning Flow and Diffusion Models with Memoryless SOC

This repository reproduces the core contributions of the paper:

> **Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control**  
> Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, Ricky T. Q. Chen  
> FAIR, Meta

## Overview

The paper introduces a theoretically-grounded approach to reward fine-tuning of Flow Matching and diffusion models. The key contributions are:

1. **Memoryless Noise Schedule** (Section 4.3): A specific noise schedule `σ(t) = √(2η_t)` that ensures the fine-tuned model converges to the tilted distribution `p*(x) ∝ p_base(x) exp(r(x))`.

2. **Adjoint Matching Algorithm** (Section 5, Algorithm 1): A novel SOC algorithm that casts the problem as a regression objective, combining the scalability of adjoint methods with the stability of least-squares objectives.

3. **Theoretical Guarantees** (Theorem 1): Proof that the memoryless noise schedule is both necessary and sufficient for unbiased fine-tuning.

## Repository Structure

```
submission/
├── src/
│   ├── __init__.py              # Package exports
│   ├── noise_schedules.py       # Memoryless noise schedule (Proposition 1, Theorem 1)
│   ├── adjoint_matching.py      # Adjoint Matching algorithm (Algorithm 1)
│   ├── adjoint_matching_ddim.py # Adjoint Matching for DDIM/DDPM (Algorithm 2)
│   ├── baselines.py             # DRaFT, ReFL, DPO, Continuous/Discrete Adjoint
│   ├── models.py                # Neural network architectures
│   ├── sde_simulation.py        # SDE simulation utilities
│   ├── evaluation.py            # Evaluation metrics (ClipScore, PickScore, etc.)
│   └── toy_experiment.py        # 1D toy experiment (Figure 2)
├── configs/
│   ├── adjoint_matching.yaml    # Adjoint Matching configuration
│   ├── draft_1.yaml             # DRaFT-1 configuration
│   └── refl.yaml                # ReFL configuration
├── scripts/
│   └── train_adjoint_matching.py # Main training script
└── README.md
```

## Core Algorithms

### Memoryless Noise Schedule

For the linear interpolant (α_t = t, β_t = 1-t), the memoryless noise schedule is:

```
σ(t) = √(2(1-t+h)/(t+h))
```

where h = 1/K is the step size. This is implemented in `src/noise_schedules.py`.

**Key insight**: This schedule ensures X_0 and X_1 are independent, removing the "initial value function bias" that plagues naive KL-regularized fine-tuning.

### Adjoint Matching (Algorithm 1)

The algorithm for Flow Matching fine-tuning:

1. **Forward pass**: Sample trajectories using the memoryless SDE:
   ```
   X_{t+h} = X_t + h*(2*v_ft(X_t,t) - κ_t*X_t) + √h*σ(t)*ε
   ```

2. **Lean adjoint**: Solve backwards in time:
   ```
   ã_{t-h} = ã_t + h * ã_t^T * ∇_{X_t}(2*v_base(X_t,t) - κ_t*X_t)
   ã_1 = -∇_{X_1} r(X_1)
   ```

3. **Loss**: Regression objective:
   ```
   L = Σ_t ||(2/σ(t))*(v_ft(X_t,t) - v_base(X_t,t)) + σ(t)*ã_t||²
   ```

### Key Implementation Details (Appendix G)

- **Timestep selection** (G.2): 10 early timesteps sampled uniformly from [0, 0.725] + last 10 timesteps [0.75, 0.975]
- **Loss clipping** (G.3): LCT = 1.6 × λ² to prevent high-magnitude early timesteps from dominating
- **Noiseless final step** (G.1): Terminal condition uses X̂_1 = X_{1-h} + h*v_base(X_{1-h}, 1-h) to avoid noise bias
- **Sigma offset** (G.1): σ(t) = √(2(1-t+h)/(t+h)) to avoid division by zero

## Baselines

The following baselines are implemented in `src/baselines.py`:

- **DRaFT-K** (Clark et al., 2024): Backpropagate reward through last K steps
- **ReFL** (Xu et al., 2023): Reward on denoised samples at random timesteps
- **DPO** (Wallace et al., 2023): Direct Preference Optimization adapted to Flow Matching
- **Continuous Adjoint**: Differentiate-then-discretize approach
- **Discrete Adjoint**: Discretize-then-differentiate approach

## Usage

### Training with Adjoint Matching

```python
from src import (
    MLPVelocityModel,
    AdjointMatchingTrainer,
    get_sigma_memoryless_fm,
)
import torch
import torch.optim as optim

# Create models
base_model = MLPVelocityModel(data_dim=2, hidden_dim=256)
finetune_model = MLPVelocityModel(data_dim=2, hidden_dim=256)
finetune_model.load_state_dict(base_model.state_dict())

# Define reward
def reward_fn(x):
    return -((x - torch.tensor([1.0, 1.0])) ** 2).sum(dim=-1)

# Create optimizer
optimizer = optim.Adam(finetune_model.parameters(), lr=2e-5, betas=(0.95, 0.999))

# Create trainer
trainer = AdjointMatchingTrainer(
    finetune_model=finetune_model,
    base_model=base_model,
    reward_fn=reward_fn,
    optimizer=optimizer,
    num_steps=40,
    lambda_reward=12500.0,
    lct_factor=1.6,
)

# Training loop
for iteration in range(1000):
    x0 = torch.randn(40, 2)
    metrics = trainer.train_step(x0)
    if (iteration + 1) % 100 == 0:
        print(f"Iter {iteration+1}: loss={metrics['loss']:.4f}, reward={metrics['reward']:.4f}")
```

### Running the Toy Experiment (Figure 2)

```bash
python -m src.toy_experiment
```

This reproduces Figure 2 from the paper, showing that:
- (a) Base Flow Matching model
- (b,c) Fine-tuning with constant σ leads to biased distributions
- (d) Fine-tuning with memoryless σ converges to the correct tilted distribution

### Training Script

```bash
python scripts/train_adjoint_matching.py \
    --method adjoint_matching \
    --lambda_reward 12500 \
    --num_steps 40 \
    --num_iterations 1000 \
    --batch_size 40 \
    --lr 2e-5
```

## Experimental Setup (Section 7)

The paper's main experiments use:
- **Base model**: Text-conditional Flow Matching on 512×512 images (latent space)
- **Reward**: ImageReward (Xu et al., 2023) scaled by λ
- **Evaluation**: ClipScore, PickScore, HPSv2, DreamSim Diversity
- **Hyperparameters**: K=40 steps, batch size 40, lr=2×10⁻⁵, Adam β₁=0.95

The evaluation metrics require external models:
- ClipScore: `open_clip` library
- PickScore: `transformers` library (yuvalkirstain/PickScore_v1)
- HPSv2: `hpsv2` library
- DreamSim: `dreamsim` library

## Key Results (Table 2)

| Method | ClipScore ↑ | PickScore ↑ | HPSv2 ↑ | Diversity ↑ |
|--------|-------------|-------------|---------|-------------|
| Base model | 28.32 | 18.15 | 17.89 | 56.53 |
| DRaFT-1 | 30.95 | 19.37 | 24.37 | 27.39 |
| Adj.-Matching λ=12500 | **31.65** | **19.76** | **24.49** | 37.24 |

Adjoint Matching achieves better consistency and human preference while maintaining more diversity than DRaFT-1.

## Assumptions and Unresolved Details

1. **Base model**: The paper uses a proprietary text-conditional Flow Matching model trained on licensed data. This reproduction uses a simple MLP for demonstration.

2. **Reward model**: The paper uses ImageReward. This reproduction uses a simple quadratic reward for demonstration.

3. **Latent space**: The paper operates in the latent space of a pre-trained autoencoder. This reproduction operates directly in data space.

4. **Scale**: The paper uses 2×80GB A100 GPUs. This reproduction is designed for smaller-scale experiments.

5. **Continuous adjoint details**: The exact implementation of the continuous adjoint method with the additional control cost gradient terms is approximated in our implementation.

## Dependencies

```
torch>=2.0
numpy
pyyaml
matplotlib (for visualization)
# Optional for evaluation:
open-clip-torch
transformers
hpsv2
dreamsim
```
