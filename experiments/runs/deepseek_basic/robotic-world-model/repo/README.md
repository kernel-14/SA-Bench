# Robotic World Model (RWM) - Reproduction

This repository contains a reproduction of the paper:

**"Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics"**
by Chenhao Li, Andreas Krause, Marco Hutter (ETH Zurich)

## Overview

The Robotic World Model (RWM) is a framework for learning robust world models for robotic control tasks. Key innovations include:

1. **Dual-autoregressive mechanism**: Inner autoregression updates GRU hidden states after each historical step; outer autoregression feeds predicted observations back into the network
2. **Self-supervised autoregressive training**: Multi-step prediction with forecast decay to mitigate compounding errors
3. **MBPO-PPO**: Policy optimization combining Model-Based Policy Optimization with PPO on imagined rollouts

## Repository Structure

```
.
├── rwm/                    # Core library
│   ├── __init__.py
│   ├── world_model.py      # RWM architecture (GRU + MLP heads)
│   ├── baselines.py        # MLP, RSSM, Transformer baselines
│   ├── training.py         # Training utilities & dataset
│   ├── mbpo_ppo.py         # MBPO-PPO policy optimization
│   ├── evaluation.py       # Evaluation metrics & comparison
│   └── rewards.py          # Velocity tracking reward functions
├── configs/                # Configuration files
│   ├── rwm_anymal_d.yaml   # ANYmal D configuration (Tables S2-S11)
│   └── rwm_unitree_g1.yaml # Unitree G1 configuration
├── scripts/                # Training & evaluation scripts
│   ├── train_rwm.py        # World model training
│   ├── train_mbpo_ppo.py   # Policy optimization with MBPO-PPO
│   ├── ablation_horizons.py # Ablation study on M and N (Fig. S8)
│   └── evaluate_noise.py   # Noise robustness evaluation (Fig. 3b)
└── README.md
```

## Core Components Reproduced

### 1. RWM Architecture (Section 3.2, A.2.1, Table S7)
- GRU base with (256, 256) hidden dimensions
- MLP heads with 128 hidden units and ReLU activation
- Gaussian output (mean and standard deviation) for observations and privileged information
- Dual-autoregressive mechanism (Fig. S6)
- Implemented in `rwm/world_model.py`

### 2. Autoregressive Training (Section 3.2)
- Sliding window data construction (M + N steps)
- Multi-step prediction loss: L = (1/N) * Σ α^k * [L_o + L_c]
- Reparameterization trick for gradient propagation
- Forecast decay factor α
- Implemented in `rwm/training.py`

### 3. Baseline Architectures (Section 4.3, Table S8)
- **MLP**: (256, 256) hidden, ReLU
- **RSSM**: GRU type, hidden 256, 2 layers, latent 64, categorical 32
- **Transformer**: Decoder type, dim 64, 8 heads, 2 layers, sinusoidal PE
- Implemented in `rwm/baselines.py`

### 4. MBPO-PPO (Section 3.3, Algorithm 1)
- Replay buffer for environment interactions
- Imagination rollout from buffer samples
- PPO policy optimization on imagined data
- GAE advantage estimation
- Termination prediction from privileged information
- Implemented in `rwm/mbpo_ppo.py`

### 5. Reward Functions (Section A.1.2, Table S6)
- 12 reward components (velocity tracking, penalties, contacts, etc.)
- Separate weights for ANYmal D and Unitree G1
- Implemented in `rwm/rewards.py`

### 6. Evaluation Suite
- Relative autoregressive prediction error metric
- Noise robustness testing (Section 4.2)
- Model comparison framework (Section 4.3)
- Ablation study on history/forecast horizons (Section A.4.1)
- Implemented in `rwm/evaluation.py` and evaluation scripts

## Key Hyperparameters (from Tables S7, S10, S11)

| Parameter | Value |
|-----------|-------|
| History horizon M | 32 |
| Forecast horizon N | 8 |
| Forecast decay α | 1.0 |
| Learning rate (WM) | 1e-4 |
| Weight decay (WM) | 1e-5 |
| Batch size (WM) | 1024 |
| Max iterations (WM) | 2500 |
| Step time Δt | 0.02s |
| Imagination envs | 4096 |
| Imagination steps | 100 |
| Buffer size | 1000 |
| PPO learning rate | 0.001 |
| Discount factor γ | 0.99 |
| Clip range | 0.2 |
| Entropy coefficient | 0.005 |
| GAE λ | 0.95 |

## Robot-Specific Configurations

### ANYmal D
- Observation: 45D (base vel(3) + ang vel(3) + gravity(3) + joint pos(12) + joint vel(12) + torque(12))
- Action: 12D (joint position targets)
- Privileged: 8D (knee contact(4) + foot contact(4))

### Unitree G1
- Observation: 96D (base vel(3) + ang vel(3) + gravity(3) + joint pos(29) + joint vel(29) + torque(29))
- Action: 29D (joint position targets)
- Privileged: 30D (body contact(26) + foot height(2) + foot velocity(2))

## Usage

### Training a World Model
```bash
python scripts/train_rwm.py \
    --robot anymal_d \
    --data_path /path/to/trajectory_data.pkl \
    --output_dir ./outputs/rwm_anymal \
    --mode autoregressive
```

### Training Policy with MBPO-PPO
```bash
python scripts/train_mbpo_ppo.py \
    --robot anymal_d \
    --world_model_checkpoint ./outputs/rwm_anymal/best_model.pt \
    --output_dir ./outputs/policy
```

### Ablation Study
```bash
python scripts/ablation_horizons.py \
    --data_path /path/to/trajectory_data.pkl \
    --M_values 8 16 32 64 128 \
    --N_values 1 2 4 8 16
```

### Noise Robustness Evaluation
```bash
python scripts/evaluate_noise.py \
    --data_path /path/to/trajectory_data.pkl \
    --noise_levels 0.0 0.01 0.05 0.1 0.2 0.5
```

## Assumptions & Missing Details

The following aspects from the paper could not be fully reproduced due to missing details:

1. **Environment integration**: We provide the full MBPO-PPO and world model training code, but integration with Isaac Lab or other simulators requires the actual simulation environment which is not part of this reproduction.

2. **Real-world deployment**: The paper's hardware experiments with ANYmal D and Unitree G1 require physical robots and low-level controllers.

3. **Pretraining data**: The paper uses simulation data from policies trained for similar tasks. We provide the training framework but specific pretraining datasets are not included.

4. **Contact termination**: The paper predicts termination from privileged information (contacts). We implement the mechanism but exact contact thresholds may need task-specific tuning.

5. **Training from scratch vs pretraining**: The paper pretrains RWM with simulation data before policy optimization (Section A.4.3). Our code supports both pretraining and from-scratch initialization.

6. **Exact optimizer settings**: Some optimizer-specific details (e.g., AdamW betas, gradient clipping values beyond the stated max_norm=10) are assumed at standard defaults.

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- NumPy

## References

The paper is available at: https://sites.google.com/view/roboticworldmodel

*Note: This is a reproduction attempt based on the paper text. The official implementation is at https://github.com/leggedrobotics/robotic_world_model (not used in this reproduction per competition rules).*
