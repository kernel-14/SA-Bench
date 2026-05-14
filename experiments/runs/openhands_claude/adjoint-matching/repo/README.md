# Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control

Implementation of the paper by Domingo-Enrich et al. (FAIR, Meta).

## Overview

This codebase reproduces the core contributions of Adjoint Matching:

1. **Memoryless noise schedule** (Theorem 1): The only noise schedule that ensures convergence to the tilted distribution `p*(x) ∝ p_base(x) exp(r(x))` during fine-tuning.

2. **Adjoint Matching algorithm** (Section 5.2): A least-squares SOC objective using the lean adjoint ODE, combining scalability of gradient methods with stability of regression objectives.

3. **Baselines**: DRaFT-1/40, ReFL, DPO, Continuous Adjoint, Discrete Adjoint.

## File Structure

```
repo/
├── config.py           # All hyperparameters and configuration dataclasses
├── noise_schedules.py  # Flow Matching and DDIM/DDPM noise schedules
│                       # Memoryless schedule: σ(t) = √(2(1-t+h)/(t+h))
├── sde_utils.py        # SDE/ODE simulation (Euler-Maruyama, DDPM, DDIM)
├── adjoint.py          # Lean adjoint ODE solver (Algorithm 1, Eq. 38-39)
│                       # Full adjoint ODE for Continuous Adjoint baseline
├── losses.py           # All loss functions:
│                       #   - Adjoint Matching (Eq. 37, 42)
│                       #   - Continuous Adjoint (Eq. 28, 32)
│                       #   - Discrete Adjoint
│                       #   - DRaFT-K (Clark et al., 2024)
│                       #   - ReFL (Xu et al., 2023)
│                       #   - DPO for FM (Wallace et al., 2023a)
├── model.py            # U-Net Flow Matching model with cross-attention
├── baselines.py        # Fine-tuner classes for all methods
├── data.py             # Prompt datasets, text encoding, reward model wrappers
├── evaluate.py         # ClipScore, PickScore, HPSv2, DreamSim Diversity
├── train.py            # Main training loop and CLI
└── requirements.txt    # Dependencies
```

## Key Algorithms

### Memoryless Noise Schedule (Section 4.3)

For Flow Matching with `α_t = t`, `β_t = 1-t`:
```
σ(t) = √(2η_t) = √(2(1-t)/t)
```
With numerical offset (Appendix G.1):
```
σ(t) = √(2(1-t+h)/(t+h))  where h = 1/K
```

This is the **only** schedule that removes the initial value function bias and ensures the fine-tuned model generates from the tilted distribution.

### Adjoint Matching (Algorithm 1)

1. Sample trajectory with memoryless SDE:
   ```
   X_{t+h} = X_t + h(2v_θ(X_t,t) - κ_t X_t) + √h σ(t) ε
   ```

2. Solve lean adjoint ODE backwards (Eq. 41):
   ```
   ã_{t-h} = ã_t + h ã_t^T ∇_{X_t}(2v_base(X_t,t) - κ_t X_t)
   ã_1 = -∇_{X_1} r(X_1)
   ```

3. Compute loss (Eq. 42):
   ```
   L = Σ_t ||2/σ(t)(v_θ(X_t,t) - v_base(X_t,t)) + σ(t)ã_t||²
   ```
   with loss clipping: `LCT = 1.6 × λ²`

### Unified SDE Framework (Eq. 10-11)

Both Flow Matching and DDIM/DDPM are unified as:
```
dX_t = b(X_t, t) dt + σ(t) dB_t
b(x,t) = κ_t x + (σ(t)²/2 + η_t) s(x,t)
```

## Usage

### Fine-tuning with Adjoint Matching

```bash
python train.py \
    --method adjoint_matching \
    --reward_lambda 12500 \
    --base_model_path checkpoints/flow_matching_base \
    --prompt_file data/prompts.txt \
    --output_dir outputs \
    --num_runs 3
```

### Running Baselines

```bash
# DRaFT-1
python train.py --method draft_1 --reward_lambda 12500

# DRaFT-40
python train.py --method draft_40 --reward_lambda 12500

# ReFL
python train.py --method refl --reward_lambda 12500

# DPO
python train.py --method dpo --reward_lambda 12500

# Continuous Adjoint
python train.py --method cont_adjoint --reward_lambda 12500

# Discrete Adjoint (use lower lr for stability)
python train.py --method disc_adjoint --reward_lambda 12500
```

### Ablation: Different λ values

```bash
for lambda in 1000 2500 12500; do
    python train.py --method adjoint_matching --reward_lambda $lambda
done
```

## Hyperparameters (Appendix G)

| Parameter | Value |
|-----------|-------|
| Timesteps K | 40 |
| Learning rate | 2×10⁻⁵ |
| Adam β₁ | 0.95 |
| Adam β₂ | 0.999 |
| Adam ε | 10⁻⁸ |
| Weight decay | 10⁻² |
| Grad norm clip | 1.0 |
| Batch size | 40 (2×20 GPUs) |
| Precision | bfloat16 |
| Training prompts | 40,000 |
| Test prompts | 1,000 |
| Runs per method | 3 |

### Loss Clipping Threshold (LCT)
- Adjoint Matching: `LCT = 1.6 × λ²`
- Continuous Adjoint: `LCT = 1600 × λ²`

### Gradient Timestep Selection (Appendix G.2)
- 10 uniformly sampled from `[0, 0.725]`
- Always include last 10 steps `[0.75, ..., 0.975]`

## Evaluation Metrics (Table 2)

| Metric | Description | Library |
|--------|-------------|---------|
| ClipScore | Text-image consistency | open_clip |
| PickScore | Human preference | transformers |
| HPSv2 | Unseen human preferences | hpsv2 |
| DreamSim Diversity | Sample diversity | dreamsim |

Diversity formula (Appendix G.4):
```
Diversity = (1/K) Σ_k (2/(N(N-1))) Σ_{i<j} ||f(g_i^k) - f(g_j^k)||²
```
where K=25 prompts, N=40 generations per prompt.

## Theoretical Background

### Value Function Bias Problem (Section 4.2)

Naïve KL-regularized fine-tuning leads to:
```
p*(X_0, X_1) = p_base(X_0, X_1) exp(r(X_1) + V(X_0, 0))
```
The `V(X_0, 0)` term biases the distribution away from the tilted distribution.

### Memoryless Property (Proposition 1)

A generative process is memoryless iff `X_0 ⊥ X_1`, which holds iff:
```
σ(t)² = 2η_t + χ(t)  where  lim_{t'→0+} α_{t'} exp(-∫_{t'}^t χ(s)/(2β_s²) ds) = 0
```
The simplest choice is `χ(t) = 0`, giving `σ(t) = √(2η_t)`.

### Lean Adjoint vs Full Adjoint (Section 5.2)

The lean adjoint removes terms with expectation zero at the optimum:
```
Full:  dã/dt = -(ã^T ∇_x(b + σu) + ∇_x(f + ½||u||²))
Lean:  dã/dt = -ã^T ∇_x b
```
This reduces variance and computational cost (no need for ∇_x u).

## Citation

```bibtex
@article{domingo2024adjoint,
  title={Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control},
  author={Domingo-Enrich, Carles and Drozdzal, Michal and Karrer, Brian and Chen, Ricky T. Q.},
  journal={arXiv preprint},
  year={2024}
}
```
