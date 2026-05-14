# Prioritized Generative Replay (PGR)

Implementation of **Prioritized Generative Replay** (Wang et al., 2024).

PGR replaces the standard replay buffer with a conditional diffusion model trained on online experience. A relevance function F guides generation toward transitions that are more useful for policy learning, improving sample efficiency in both state- and pixel-based RL tasks.

## Method Overview

The core idea (Algorithm 1):
1. **Outer loop**: collect real transitions with policy π, update relevance function F
2. **Inner loop** (every 10K steps): retrain conditional diffusion model G on D_real, generate synthetic transitions conditioned on high-F values, train π on mixed real+synthetic data

Relevance functions supported:
- `curiosity` — ICM prediction error (default, best performing)
- `td_error` — Bellman residual
- `return` — Q-value estimate
- `reward` — raw reward signal
- `rnd` — Random Network Distillation error
- `cts` — CTS pseudo-count novelty
- `eco` — Episodic Curiosity through reachability

## Repository Structure

```
repo/
├── config.py              # All hyperparameters and experiment configs
├── replay_buffer.py       # Real and synthetic replay buffers
├── train.py               # Main PGR training loop (Algorithm 1)
├── train_baselines.py     # Baseline training (SAC, REDQ, SYNTHER, PER, etc.)
├── evaluate.py            # Evaluation: dormant ratio, generation MSE, curiosity values
├── utils.py               # Seeding, logging, environment wrappers
├── models/
│   ├── __init__.py
│   ├── networks.py        # MLP, CNN, ResNet, NoisyLinear building blocks
│   ├── diffusion.py       # Conditional DDPM with classifier-free guidance
│   ├── rl_agents.py       # SAC, REDQ, DRQ-V2 implementations
│   └── relevance.py       # All relevance function modules
└── requirements.txt
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Real/synthetic buffer size | 1M |
| Synthetic data ratio r | 0.5 |
| Batch size | 256 |
| UTD ratio | 20 |
| CFG dropout p_uncond | 0.25 |
| Inner loop frequency | 10K steps |
| Diffusion steps N | 1000 |
| Guidance scale ω | 1.5 |
| Top-k prompting ratio | 0.1 |

## Usage

```bash
# Train PGR with curiosity guidance on DMC quadruped-walk
python train.py --env quadruped-walk --relevance curiosity --seed 0

# Train SYNTHER baseline (unconditional)
python train_baselines.py --method synther --env quadruped-walk --seed 0

# Train REDQ baseline
python train_baselines.py --method redq --env quadruped-walk --seed 0

# Pixel-based DMC
python train.py --env walker-walk --pixel True --relevance curiosity --seed 0

# OpenAI Gym
python train.py --env Walker2d-v2 --relevance curiosity --seed 0

# Scaling: larger network + higher UTD
python train.py --env quadruped-walk --relevance curiosity --hidden_dim 512 --n_hidden 3 \
    --utd_ratio 40 --synthetic_ratio 0.75 --batch_size 512 --syn_buffer_size 2000000
```

## Baselines

```bash
# PER with TD-error priority
python train_baselines.py --method per_td --env quadruped-walk

# PER with curiosity priority
python train_baselines.py --method per_curiosity --env quadruped-walk

# REDQ + curiosity exploration bonus
python train_baselines.py --method redq_curiosity --env quadruped-walk

# SYNTHER + curiosity exploration bonus
python train_baselines.py --method synther_curiosity --env quadruped-walk

# NoisyNets
python train_baselines.py --method noisynets --env quadruped-walk

# Bootstrapped DQN
python train_baselines.py --method boot_dqn --env quadruped-walk
```

## References

- Wang et al. (2024). Prioritized Generative Replay.
- Lu et al. (2024). Synthetic Experience Replay (SynthER).
- Chen et al. (2021). Randomized Ensembled Double Q-Learning (REDQ).
- Pathak et al. (2017). Curiosity-driven Exploration by Self-Supervised Prediction (ICM).
- Ho & Salimans (2022). Classifier-Free Diffusion Guidance.
