# Adjoint Matching

Reproduction of the paper:
**"Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control"**
by Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, Ricky T. Q. Chen (FAIR, Meta).

## Codebase Structure

```
repo/
├── configs/
│   └── config.yaml           # All hyperparameters and configuration
├── models/
│   ├── __init__.py
│   ├── unet.py               # U-Net architecture (based on Stable Diffusion)
│   ├── flow_matching.py      # Flow Matching model (ODE and SDE sampling)
│   └── diffusion.py          # Denoising Diffusion model (DDIM/DDPM)
├── soc/
│   ├── __init__.py
│   ├── memoryless_schedule.py # Memoryless noise schedule sigma(t)=sqrt(2*eta_t)
│   ├── control.py             # Fine-tuning base classes for FM and DDIM
│   └── adjoint_matching.py    # Adjoint Matching algorithm + Continuous/Full Adjoint
├── training/
│   ├── __init__.py
│   ├── train_adjoint_matching.py  # Training loop for Adjoint Matching (Algorithm 1)
│   └── train_baselines.py     # Baselines: DRaFT, ReFL, DPO, Cont/Disc Adjoint
├── data/
│   ├── __init__.py
│   └── dataset.py            # Dataset loading and preprocessing
├── evaluation/
│   ├── __init__.py
│   └── metrics.py            # Evaluation metrics (ClipScore, PickScore, HPSv2, etc.)
├── main.py                   # Main entry point
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Key Algorithms Implemented

### 1. Memoryless Noise Schedule (Theorem 1, Proposition 1)
- Implements `sigma(t) = sqrt(2 * eta_t)` where `eta_t = beta_t * (alpha_dot/alpha * beta_t - beta_dot)`
- Removes the initial value function bias problem
- Enables fine-tuning for arbitrary noise schedules after training

### 2. Adjoint Matching (Algorithm 1, Section 5.2)
- Combines continuous adjoint method with least-squares regression
- Uses "lean" adjoint state: removes zero-expectation terms at optimum
- No importance weighting needed

### 3. Baseline Methods
- **DRaFT-K**: Direct Reward Fine-Tuning through K steps (Clark et al., 2024)
- **ReFL**: Reward Feedback Learning (Xu et al., 2023)
- **DPO**: Direct Preference Optimization adapted for Flow Matching (Wallace et al., 2023a)
- **Continuous Adjoint**: Differentiate-then-discretize (Pontryagin, 1962)
- **Discrete Adjoint**: Discretize-then-differentiate

### 4. Evaluation Metrics
- ClipScore (text-to-image consistency)
- PickScore (human preference alignment)
- HPS v2 (generalization to unseen human preferences)
- DreamSim Diversity (sample diversity)
- ImageReward (fine-tuning reward)

## Usage

```bash
# Run Adjoint Matching fine-tuning
python main.py \
    --config configs/config.yaml \
    --method adjoint_matching \
    --prompt_file /path/to/prompts.json \
    --seed 42 \
    --output_dir ./outputs

# Run baseline methods
python main.py --method draft-1 --prompt_file prompts.json
python main.py --method draft-40 --prompt_file prompts.json
python main.py --method refl --prompt_file prompts.json
python main.py --method dpo --prompt_file prompts.json
python main.py --method continuous_adjoint --prompt_file prompts.json
python main.py --method discrete_adjoint --prompt_file prompts.json
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch size | 20 per GPU | 2 GPUs total (effective 40) |
| Learning rate | 2e-5 | Adam optimizer |
| Adam β₁ | 0.95 | |
| Adam β₂ | 0.999 | |
| Weight decay | 1e-2 | |
| Gradient clip | 1.0 | |
| Timesteps K | 40 | For both fine-tuning and inference |
| λ (reward scale) | 1000, 2500, 12500 | Trade-off between base model and reward |
| Precision | bfloat16 | |

## Paper References

- Flow Matching: Lipman et al. (2023), "Flow Matching for Generative Modeling"
- DDIM: Song et al. (2021), "Denoising Diffusion Implicit Models"
- ImageReward: Xu et al. (2023), "ImageReward: Learning and Evaluating Human Preferences"
- DRaFT: Clark et al. (2024), "Directly Fine-Tuning Diffusion Models on Differentiable Rewards"
- SOCM: Domingo-Enrich et al. (2023), "Stochastic Optimal Control Matching"
