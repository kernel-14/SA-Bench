# Prioritized Generative Replay (PGR)

Reproduction of the paper "Prioritized Generative Replay" by Wang et al. (UC Berkeley).

## Overview

PGR is a framework for sample-efficient online RL that uses conditional diffusion models as a parametric replay buffer. It conditions on relevance functions (e.g., curiosity) to generate more learning-relevant transitions.

## Codebase Structure

```
repo/
├── config.py       # All hyperparameters and experiment configurations
├── models/
│   ├── __init__.py
│   ├── diffusion.py   # Conditional diffusion model with CFG
│   ├── policy.py      # SAC, REDQ, DRQ-v2 policy architectures
│   └── curiosity.py   # ICM, RND, CTS, ECO relevance functions
├── pgr.py             # Core PGR Algorithm 1 (outer + inner loop)
├── replay.py          # Replay buffers (standard, synthetic, prioritized)
├── envs.py            # Environment wrappers (DMC, Gym, DMLab)
├── train.py           # Main training loop with all baselines
├── scaling.py         # Section 5.3 scaling experiments
├── baselines.py       # PER and exploration bonus baselines
├── utils.py           # Metrics, logging, dormant ratio
└── README.md
```

## Key Components

### Algorithm 1: PGR Framework
- **Outer loop**: Agent interacts with environment, storing transitions in D_real
- **Inner loop** (every 10K steps): Train conditional diffusion model, generate synthetic data with CFG, train policy on mixed real+synthetic data

### Relevance Functions (Eq. 3-5)
- `curiosity` (ICM): Forward dynamics prediction error
- `td_error`: Temporal difference error
- `return`: Q-value estimate
- `reward`: Episodic reward
- `rnd`: Random Network Distillation
- `cts`: Context-Tree Switching pseudo-counts
- `eco`: Episodic Curiosity via reachability

### Policy Architectures
- REDQ (state-based): Ensemble of 10 Q-networks with 2 random targets
- SAC: Standard Soft Actor-Critic
- DRQ-v2 (pixel-based): CNN encoder + SAC

### Baselines
- SYNTHER: Unconditional diffusion (p_uncond = 1.0)
- PER: Prioritized experience replay with TD-error or curiosity
- Model-free: REDQ / SAC without generation
- Exploration bonuses: REDQ + intrinsic reward

### Scaling Experiments (Section 5.3)
- Larger networks (6x params)
- Higher synthetic data ratios (r=0.75, 0.875)
- Combined with UTD=40

## Usage

### PGR with curiosity on Quadruped-Walk
```bash
python train.py --experiment dmc_state --relevance_fn curiosity \
    --env_domain quadruped --env_task walk --baseline pgr
```

### SYNTHER baseline
```bash
python train.py --experiment dmc_state \
    --env_domain quadruped --env_task walk --baseline synther
```

### Model-free REDQ baseline
```bash
python train.py --experiment dmc_state \
    --env_domain quadruped --env_task walk --baseline model_free
```

### PER baseline
```bash
python train.py --experiment dmc_state --relevance_fn curiosity \
    --env_domain quadruped --env_task walk --baseline per
```

### Pixel-based DMC
```bash
python train.py --experiment dmc_pixel --relevance_fn curiosity \
    --env_domain walker --env_task walk --baseline pgr
```

### Scaling experiments
```bash
# Larger network
python scaling.py --experiment larger_network --env_domain quadruped --env_task walk

# Higher synthetic ratios
python scaling.py --experiment higher_ratios --env_domain quadruped --env_task walk

# Combined scaling
python scaling.py --experiment combined --env_domain quadruped --env_task walk
```

### OpenAI Gym
```bash
python train.py --experiment gym --relevance_fn curiosity --baseline pgr
```

### DMLab (Noisy TV, Appendix A.2)
```bash
python train.py --experiment dmlab --relevance_fn eco --baseline pgr
```

## Implementation Notes

- The conditional diffusion model uses a residual MLP denoising architecture matching SynthER
- Classifier-free guidance (CFG) with p_uncond=0.25 and guidance_scale=1.5
- Top-k prompting strategy samples conditioning values from highest-F transitions
- Synthetic buffer regenerated every 10K environment steps
- UTD ratio of 20 applied to policy updates
- Pixel-based experiments generate in latent space of the CNN encoder

## References

- Wang et al., "Prioritized Generative Replay", 2024
- Lu et al., "Synthetic Experience Replay", NeurIPS 2023
- Chen et al., "Randomized Ensembled Double Q-Learning", ICLR 2021
- Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction", ICML 2017
