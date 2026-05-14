# Robotic World Model (RWM) - Reproduction

Reproduction of the paper:
**"Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics"**
by Chenhao Li, Andreas Krause, Marco Hutter (ETH Zurich)

## Overview

This repository implements the core contributions of the RWM paper:

1. **RWM Architecture** - GRU-based world model with dual-autoregressive mechanism
2. **Self-supervised Autoregressive Training** - Multi-step prediction with history and forecast horizons
3. **MBPO-PPO** - Model-based policy optimization using PPO on imagination rollouts
4. **Baseline Models** - MLP, RSSM, and Transformer for comparison

## Repository Structure

```
├── models/
│   ├── rwm.py          # Robotic World Model (GRU + MLP heads)
│   ├── baselines.py    # MLP, RSSM, Transformer baselines
│   └── policy.py       # Policy and value function networks
├── training/
│   ├── world_model_trainer.py  # Autoregressive + teacher-forcing trainers
│   └── mbpo_ppo.py             # MBPO-PPO algorithm (Algorithm 1)
├── envs/
│   └── rewards.py      # Velocity tracking reward functions
├── utils/
│   └── dataset.py      # Sliding window trajectory datasets
├── configs/
│   ├── rwm_anymal.yaml         # RWM config for ANYmal D
│   ├── rwm_g1.yaml             # RWM config for Unitree G1
│   ├── mbpo_ppo_anymal.yaml    # MBPO-PPO config for ANYmal D
│   └── baselines.yaml          # Baseline model configs
└── scripts/
    ├── train_world_model.py    # Train RWM
    ├── train_mbpo_ppo.py       # Train policy with MBPO-PPO
    ├── evaluate_models.py      # Compare model architectures
    └── ablation_horizons.py    # Ablation study on M and N
```

## Core Contributions Implemented

### 1. RWM Architecture (Section 3.2, Table S7)

The Robotic World Model uses:
- **GRU base**: 2 layers, hidden size 256
- **MLP heads**: hidden size 128, ReLU activation
  - Observation head: predicts mean and std of next observation
  - Privileged info head: predicts contacts/privileged information

**Dual-autoregressive mechanism**:
- *Inner autoregression*: GRU processes history horizon M step-by-step
- *Outer autoregression*: Predicted observations fed back for N forecast steps

### 2. Self-supervised Autoregressive Training (Section 3.2, Eq. 2)

Training loss:
```
L = (1/N) * sum_{k=1}^{N} alpha^k * [L_o(o'_{t+k}, o_{t+k}) + L_c(c'_{t+k}, c_{t+k})]
```

- Sliding window dataset of size M + N
- Reparameterization trick for gradient propagation
- Decay factor alpha for multi-step weighting

**Training parameters** (Table S10):
- History horizon M = 32
- Forecast horizon N = 8
- Forecast decay alpha = 1.0
- Learning rate = 1e-4
- Weight decay = 1e-5
- Batch size = 1024
- Max iterations = 2500

### 3. MBPO-PPO (Section 3.3, Algorithm 1, Table S11)

Policy optimization using learned world model:
1. Collect real environment data into replay buffer D
2. Train world model with autoregressive training
3. Initialize imagination agents from D
4. Roll out T=100 imagination steps using policy + world model
5. Update policy with PPO

**PPO parameters** (Table S11):
- Imagination environments: 4096
- Imagination steps: 100
- Learning rate: 0.001
- Learning epochs: 5
- Mini-batches: 4
- KL target: 0.01
- Discount factor: 0.99
- Clip range: 0.2
- Entropy coefficient: 0.005

### 4. Baseline Models (Table S8)

- **MLP**: 2 hidden layers of 256, ReLU, history concatenated as input
- **RSSM**: GRU hidden 256, latent dim 64, categorical prior with 32 categories
- **Transformer**: Decoder-only, d_model=64, 8 heads, 2 layers, sinusoidal PE

### 5. Reward Functions (Section A.1.2, Table S6)

Velocity tracking rewards for ANYmal D and Unitree G1:
- Linear velocity tracking (xy): exponential kernel, sigma=0.25
- Angular velocity tracking (z): exponential kernel, sigma=0.25
- Penalties: vertical velocity, roll/pitch rate, joint torques, action rate
- Bonuses: feet air time, foot clearance

## Observation and Action Spaces

### ANYmal D (Tables S2, S4, S5)

**World model observation** (45-dim):
- Base linear velocity (3), angular velocity (3), projected gravity (3)
- Joint positions (12), velocities (12), torques (12)

**Policy observation** (48-dim):
- Base linear velocity (3), angular velocity (3), projected gravity (3)
- Velocity command (3), joint positions (12), velocities (12), last actions (12)

**Action space** (12-dim): Joint position targets

**Privileged info** (8-dim): Knee contacts (4) + foot contacts (4)

### Unitree G1 (Tables S2, S4, S5)

**World model observation** (96-dim):
- Base linear velocity (3), angular velocity (3), projected gravity (3)
- Joint positions (29), velocities (29), torques (29)

**Policy observation** (99-dim):
- Base linear velocity (3), angular velocity (3), projected gravity (3)
- Velocity command (3), joint positions (29), velocities (29), last actions (29)

**Action space** (29-dim): Joint position targets

**Privileged info** (30-dim): Body contacts (26) + foot height (2) + foot velocity (2)

## Usage

### Train World Model

```bash
# Train RWM with autoregressive training (RWM-AR)
python scripts/train_world_model.py \
    --config configs/rwm_anymal.yaml \
    --data_path /path/to/trajectories.npz \
    --output_dir outputs/world_model

# Train with teacher forcing (RWM-TF baseline)
python scripts/train_world_model.py \
    --config configs/rwm_anymal.yaml \
    --teacher_forcing \
    --output_dir outputs/world_model_tf
```

### Train Policy with MBPO-PPO

```bash
python scripts/train_mbpo_ppo.py \
    --wm_checkpoint outputs/world_model/seed_42/best_model.pt \
    --config configs/mbpo_ppo_anymal.yaml \
    --output_dir outputs/mbpo_ppo
```

### Evaluate and Compare Models

```bash
# Compare all architectures (Figure 4)
python scripts/evaluate_models.py \
    --robot anymal \
    --eval_horizon 100 \
    --output_dir outputs/evaluation

# Ablation study on M and N (Figure S8)
python scripts/ablation_horizons.py \
    --robot anymal \
    --M_values 4 8 16 32 64 \
    --N_values 1 2 4 8 16 \
    --output_dir outputs/ablation
```

## Data Format

Trajectory data should be stored as `.npz` files with:
```python
np.savez("trajectories.npz",
    observations=observations,  # list of (T, obs_size) arrays
    actions=actions,            # list of (T, action_size) arrays
    privileged_info=priv_info,  # list of (T, priv_size) arrays (optional)
)
```

The paper uses Isaac Lab for simulation with ANYmal D and Unitree G1 robots.
Data is collected from velocity tracking policies at 50 Hz.

## Assumptions and Missing Details

1. **Exact data preprocessing**: The paper doesn't specify normalization details. We assume raw observations are used without normalization.

2. **Reward computation in imagination**: The world model obs space doesn't include velocity commands. In MBPO-PPO, velocity commands need to be tracked separately or embedded in the policy obs.

3. **Collision handling**: The paper mentions terminating rollouts when base contact is detected (Section A.4.3). We implement this via the privileged info prediction head.

4. **Pretraining data**: The paper uses simulation data from policies trained for similar tasks. We provide a synthetic data generator for testing.

5. **Isaac Lab integration**: The paper uses Isaac Lab for simulation. This reproduction provides the model and training code but not the Isaac Lab environment wrappers.

6. **Hardware deployment**: Zero-shot transfer to ANYmal D and Unitree G1 hardware is described but not reproducible without the physical robots.

## Key Results from Paper

- RWM-AR achieves lowest prediction errors across all environments (Figure 4)
- RWM-AR significantly outperforms RWM-TF (teacher forcing)
- MBPO-PPO achieves real tracking reward of 0.90 ± 0.04 (Table 1)
- Optimal hyperparameters: M=32, N=8 (Table S10, Figure S8)
- Training: ~50 min for world model pretraining, ~5 min for MBPO-PPO

## Dependencies

- PyTorch
- NumPy
- PyYAML
