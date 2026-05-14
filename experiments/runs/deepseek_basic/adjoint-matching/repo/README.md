# Adjoint Matching: Fine-tuning Flow and Diffusion Models with Memoryless SOC

Reproduction of the paper:

> **Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control**
> Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, Ricky T. Q. Chen (FAIR, Meta)

## Overview

This repository implements the core contributions of the Adjoint Matching paper:

1. **Memoryless Noise Schedule** (Section 4.3): A theoretically-grounded noise schedule
   σ(t) = √(2η_t) that ensures the fine-tuned model converges to the tilted distribution
   p*(x) ∝ p^base(x) exp(r(x)). This resolves the "initial value function bias" problem
   that plagues naive KL-regularized fine-tuning.

2. **Adjoint Matching Algorithm** (Section 5.2): A novel SOC algorithm that casts
   stochastic optimal control as a least-squares regression problem, using a "lean
   adjoint" state that removes high-variance terms from the continuous adjoint method.

3. **Fine-tuning Recipes** for both Flow Matching and DDIM/DDPM models (Algorithms 1 & 2).

## Repository Structure

```
adjoint_matching/
├── __init__.py           # Package initialization
├── noise_schedule.py     # Memoryless noise schedule implementations
├── adjoint_matching.py   # Core Adjoint Matching algorithm and trainer
├── baselines.py          # Baseline methods (Cont. Adjoint, Disc. Adjoint, DRaFT, ReFL)
├── theory.py             # Theoretical foundations and reference implementations
└── evaluation.py         # Evaluation metrics (ClipScore, PickScore, HPS v2, etc.)
setup.py
README.md
```

## Key Theoretical Contributions

### 1. Memoryless Noise Schedule (Proposition 1, Theorem 1)

The paper proves that to fine-tune dynamical generative models to sample from the
tilted distribution p*(x) ∝ p^base(x) exp(r(x)), the fine-tuning must use the
memoryless noise schedule σ(t) = √(2η_t).

For **Flow Matching** (α_t = t, β_t = 1-t):
- η_t = β_t(α̇_t/α_t · β_t - β̇_t) = (1-t)/t
- σ(t) = √(2(1-t)/t)

For **DDIM/Diffusion**:
- η_t = ᾱ̇_t/(2ᾱ_t)
- σ(t) = √(ᾱ̇_t/ᾱ_t)  (recovers DDPM in the continuous-time limit)

With the numerical offset from Appendix G.1:
- σ(t) = √(2(1-t+h)/(t+h))

### 2. Value Function Bias Problem (Section 4.2)

Without the memoryless noise schedule, the SOC formulation produces:
p*(X₀, X₁) = p^base(X₀, X₁) · exp(r(X₁) + V(X₀, 0))

The term V(X₀, 0) biases the distribution away from the desired tilted distribution
unless X₀ and X₁ are independent (i.e., the process is memoryless).

### 3. Adjoint Matching (Section 5.2, Proposition 7)

The Adjoint Matching objective is:
L_Adj-Match(u; X) = ½ ∫₀¹ ||u(X_t, t) + σ(t)ᵀ ã(t; X)||² dt

Where the lean adjoint state ã solves:
dã/dt = -(ãᵀ ∇_x b(X_t, t) + ∇_x f(X_t, t))
ã(1) = ∇_x g(X₁)

The lean adjoint removes terms with zero expectation at the optimal control,
reducing variance and computational cost compared to the continuous adjoint.

## Usage

### Flow Matching Fine-tuning

```python
from adjoint_matching.noise_schedule import FlowMatchingNoiseSchedule
from adjoint_matching.adjoint_matching import AdjointMatchingTrainer

# Setup
noise_schedule = FlowMatchingNoiseSchedule(num_steps=40, offset=True)

trainer = AdjointMatchingTrainer(
    base_model=base_model,       # Pre-trained v^base
    fine_tuned_model=ft_model,   # Clone of base_model
    noise_schedule=noise_schedule,
    reward_fn=reward_model,      # ImageReward, etc.
    model_type="flow_matching",
    lambda_reg=12500,            # Reward scaling λ
    lr=2e-5,
    beta1=0.95,
)

# Training loop
for iteration in range(num_iterations):
    metrics = trainer.train_step(batch_size=40, device="cuda")
    print(f"Iter {iteration}: loss = {metrics['loss']:.4f}")

# Generate samples with ODE sampler (σ=0) after fine-tuning
samples = trainer.generate(batch_size=16, device="cuda", sigma_sampling=0.0)
```

### DDIM Fine-tuning

```python
from adjoint_matching.noise_schedule import DDIMMemorylessNoiseSchedule

noise_schedule = DDIMMemorylessNoiseSchedule(num_steps=40)
trainer = AdjointMatchingTrainer(
    ...,
    model_type="ddim",
    noise_schedule=noise_schedule,
)
```

## Reproduction Notes

### What was successfully implemented

1. **Memoryless noise schedule** (`noise_schedule.py`): Complete implementation for both
   Flow Matching and DDIM, including the numerical offset from Appendix G.1.

2. **Adjoint Matching algorithm** (`adjoint_matching.py`): Full implementation of
   Algorithm 1 (Flow Matching) and Algorithm 2 (DDIM) including:
   - Trajectory sampling with memoryless noise schedule
   - Lean adjoint ODE solver
   - Adjoint Matching loss computation
   - LCT clipping (Appendix G.3)
   - Timestep subsampling (Appendix G.2)

3. **Baseline methods** (`baselines.py`): Implementations of Continuous Adjoint,
   Discrete Adjoint, DRaFT-K, and ReFL for Flow Matching.

4. **Theoretical foundations** (`theory.py`): Reference implementations of key
   equations (control computation, lean vs full adjoint ODE, value function bias).

5. **Evaluation metrics** (`evaluation.py`): Framework for computing ClipScore,
   PickScore, HPS v2, DreamSim Diversity, and their diversity variants.

### Key hyperparameters (from Appendix G)

| Parameter | Value |
|-----------|-------|
| Number of timesteps K | 40 |
| Learning rate | 2×10⁻⁵ |
| Adam β₁ | 0.95 |
| Adam β₂ | 0.999 |
| Weight decay | 1×10⁻² |
| Gradient clipping | max norm 1.0 |
| Batch size | 40 |
| Precision | bfloat16 |
| LCT (Adj. Matching) | 1.6 × λ² |
| LCT (Cont. Adjoint) | 1600 × λ² |
| Timestep subsampling | Last 10 + 10 random from first 30 |

### Assumptions and missing details

1. **Pre-trained models**: The paper assumes a pre-trained Flow Matching model with
   a U-Net architecture on 512×512 latent space. This implementation is model-agnostic
   and works with any vector field model.

2. **Reward models**: The paper uses ImageReward (Xu et al., 2023). The code accepts
   any differentiable reward function.

3. **DPO baseline**: The paper notes that on-policy DPO (sampling preference pairs
   from the current model) performs poorly. The implementation uses reward-based
   preference weighting (Appendix F.2, Eq. 235).

4. **CFG**: Classifier-free guidance is applied after fine-tuning using the formula
   (1+w)·v(x,t|y) - w·v(x,t). The paper notes this is not principled for fine-tuned
   models but works empirically.

5. **Denoiser for final step** (Appendix G.1): The lean adjoint is initialized at
   X̂₁ = X_{1-h} + h·v^base(X_{1-h}, 1-h) rather than X₁, to avoid noise bias.

6. **σ(t) offset**: The paper uses σ(t) = √(2(1-t+h)/(t+h)) instead of the
   theoretical √(2(1-t)/t) for numerical stability (h = 1/40 = 0.025).

## References

- Lipman et al. (2023) - Flow Matching for Generative Modeling
- Ho et al. (2020) - Denoising Diffusion Probabilistic Models
- Song et al. (2021a) - Denoising Diffusion Implicit Models
- Domingo-Enrich et al. (2023) - Stochastic Optimal Control Matching
- Clark et al. (2024) - DRaFT: Directly Fine-tuning Diffusion Models on Differentiable Rewards
- Xu et al. (2023) - ImageReward: Learning and Evaluating Human Preferences
- Wallace et al. (2023a) - Diffusion Model Alignment Using Direct Preference Optimization
