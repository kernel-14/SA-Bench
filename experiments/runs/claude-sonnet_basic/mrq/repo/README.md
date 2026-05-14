# MR.Q: Model-based Representations for Q-learning

Reproduction of **"Towards General-Purpose Model-Free Reinforcement Learning (MR.Q)"**  
Fujimoto et al., Meta FAIR, ICLR 2025

---

## Overview

MR.Q is a general-purpose model-free deep RL algorithm that achieves competitive performance across diverse benchmarks (Gym locomotion, DMC proprioceptive, DMC visual, Atari) using a **single fixed set of hyperparameters**.

The key idea is to learn state-action embeddings that approximately linearize the value function, using model-based objectives (reward prediction, dynamics prediction, terminal prediction) as representation learning signals — without actually doing planning or model rollouts at test time.

### Core Contributions Implemented

1. **MR.Q Algorithm** (`mrq/agent.py`)
   - State encoder `f_ω: s → z_s` (MLP for vectors, CNN for images)
   - State-action encoder `g_ω: (z_s, a) → z_sa`
   - Linear MDP predictor `m: z_sa → (z_s', r_logits, d)` (single linear layer)
   - Twin value networks `Q_θ: z_sa → ℝ` (TD3-style)
   - Policy network `π_φ: z_s → a` (deterministic policy gradient)

2. **Encoder Loss** (Section 4.2.1)
   - Unrolled over `H_enc = 5` steps
   - Reward loss: categorical cross-entropy with two-hot encoding (symexp bins)
   - Dynamics loss: MSE to target encoder output (stops policy dependency)
   - Terminal loss: MSE (enabled after first terminal seen)

3. **Value Loss** (Section 4.2.2)
   - Multi-step returns (`H_Q = 3`)
   - Reward scaling by mean absolute reward `r̄`
   - Huber loss + LAP prioritized replay
   - Twin critics with minimum target

4. **Policy Loss** (Section 4.2.3)
   - Deterministic policy gradient
   - Gumbel-Softmax for discrete actions (τ=10)
   - Tanh for continuous actions
   - Pre-activation regularization (λ=1e-5)

5. **Synchronized Target Updates** (every `T_target = 250` steps)
   - Hard copy of all networks to targets
   - Reward scale update

6. **LAP Replay Buffer** (`utils/replay_buffer.py`)
   - Priority-based sampling (α=0.4)
   - Minimum priority = 1

7. **Reward Encoding** (`utils/reward_encoding.py`)
   - symexp-spaced bins (65 bins, range [-10, 10] in symlog space → ~[-22k, 22k])
   - Two-hot encoding for categorical reward targets

8. **Evaluation & Normalization** (`utils/evaluation.py`)
   - TD3-normalized scores for Gym
   - Human-normalized scores for Atari
   - Raw reward for DMC
   - Stratified bootstrap confidence intervals (95%)
   - Mean, Median, IQM aggregate metrics

---

## Repository Structure

```
submission/
├── mrq/
│   ├── __init__.py
│   ├── networks.py      # Neural network architectures
│   └── agent.py         # MR.Q algorithm
├── envs/
│   ├── __init__.py
│   └── wrappers.py      # Environment wrappers (Gym, DMC, Atari)
├── utils/
│   ├── __init__.py
│   ├── replay_buffer.py # LAP replay buffer
│   ├── reward_encoding.py  # Symexp bins, two-hot encoding
│   └── evaluation.py    # Score normalization, aggregate metrics
├── train.py             # Main training script
├── run_experiments.py   # Batch experiment runner
└── README.md
```

---

## Installation

```bash
pip install torch numpy gymnasium

# For Gym locomotion (MuJoCo):
pip install gymnasium[mujoco]

# For DMC:
pip install dm_control

# For Atari:
pip install gymnasium[atari] ale-py
pip install opencv-python  # for frame preprocessing
```

**Software versions used in paper:**
- Python 3.11.8
- PyTorch 2.4.1
- Gymnasium 0.29.1
- MuJoCo 3.2.2
- NumPy 2.1.1

---

## Usage

### Single experiment

```bash
# Gym locomotion (1M steps)
python train.py --env_type gym --env_name HalfCheetah-v4 --seed 0

# DMC proprioceptive (500k steps)
python train.py --env_type dmc_proprio --env_name cheetah-run --seed 0

# DMC visual (500k steps)
python train.py --env_type dmc_visual --env_name cheetah-run --seed 0

# Atari (2.5M steps)
python train.py --env_type atari --env_name Pong --seed 0
```

### Full benchmark reproduction

```bash
# Gym (5 envs × 10 seeds)
python run_experiments.py --benchmark gym --seeds 10

# DMC proprioceptive (28 envs × 10 seeds)
python run_experiments.py --benchmark dmc_proprio --seeds 10

# DMC visual (28 envs × 10 seeds)
python run_experiments.py --benchmark dmc_visual --seeds 10

# Atari (57 games × 10 seeds)
python run_experiments.py --benchmark atari --seeds 10

# Parallel execution
python run_experiments.py --benchmark gym --seeds 10 --n_workers 4
```

Results are saved as CSV files in `results/`.

---

## Hyperparameters

All hyperparameters are fixed across all benchmarks (Table 3 of paper):

| Parameter | Value |
|-----------|-------|
| Discount γ | 0.99 |
| Replay buffer size | 1M |
| Batch size | 256 |
| Target update freq T_target | 250 |
| Encoder horizon H_enc | 5 |
| Multi-step horizon H_Q | 3 |
| Reward loss weight λ_R | 0.1 |
| Terminal loss weight λ_T | 0.1 |
| Pre-activation weight λ_preactiv | 1e-5 |
| Encoder LR | 1e-4 |
| Value LR | 3e-4 |
| Policy LR | 3e-4 |
| Weight decay | 1e-4 |
| Optimizer | AdamW |
| Target noise σ | N(0, 0.2²) |
| Target noise clip c | 0.3 |
| Exploration noise | N(0, 0.2²) |
| LAP α | 0.4 |
| Min priority | 1 |
| Reward bins | 65 |
| Reward range (symlog) | [-10, 10] |
| z_s dim | 512 |
| z_sa dim | 512 |
| z_a dim | 256 |
| Hidden dim | 512 |
| Encoder activation | ELU |
| Value activation | ELU |
| Policy activation | ReLU |
| Gumbel-Softmax τ | 10 |
| Initial random steps | 10k |

---

## Network Architectures

### State Encoder f_ω
- **Vector obs**: 3-layer MLP (512 hidden), LayerNorm+ELU after each layer → z_s ∈ ℝ^512
- **Image obs (84×84)**: 4 conv layers (32ch, k=3, strides 2,2,2,1) + linear → LayerNorm+ELU → z_s ∈ ℝ^512

### State-Action Encoder g_ω
- Action: Linear(action_dim → 256) + ELU
- Concat [z_s, z_a]: 3-layer MLP (512 hidden), LayerNorm+ELU after first 2 → z_sa ∈ ℝ^512
- Linear MDP predictor: Linear(512 → 512+65+1) — single linear layer

### Value Network Q_θ
- 4-layer MLP (512 hidden), LayerNorm+ELU after first 3 → scalar
- Two independent networks (twin critics)

### Policy Network π_φ
- 3-layer MLP (512 hidden), LayerNorm+ReLU after first 2
- Discrete: Gumbel-Softmax(τ=10)
- Continuous: Tanh

---

## Design Study

The paper's design study (Table 2) validates key choices. Our implementation supports all variants:

- **Linear value function**: Replace non-linear Q with linear weights
- **Dynamics target**: Use state-action embedding instead of state embedding
- **No target encoder**: Use current encoder for dynamics target
- **Non-linear model**: Replace linear MDP predictor with separate networks
- **MSE reward loss**: Replace categorical with MSE
- **No reward scaling**: Set r̄ = r̄' = 1
- **No min**: Use mean instead of min over twin critics
- **No LAP**: Use uniform sampling + MSE loss
- **No MR**: Remove model-based representation learning
- **1-step return**: Remove multi-step returns
- **No unroll**: Set H_enc = 1

---

## Assumptions and Unresolved Details

1. **Dynamics loss weight λ_Dynamics**: The paper's Table 3 shows "1" in the cell but the label is cut off. We use λ_Dynamics = 1.0 as the default.

2. **Replay ratio**: The paper mentions "replay ratio" in the hyperparameter table but the value is not clearly visible. We use 1 update per environment step (standard).

3. **Sequence sampling**: The paper samples sequences for encoder unrolling. We implement this by sampling contiguous transitions from the buffer, respecting episode boundaries.

4. **Gradient flow**: Encoder gradients are stopped when computing value and policy losses (the encoder is updated separately). This matches the paper's description of "decoupled RL."

5. **DMC visual frame stacking**: The paper uses 3 stacked RGB frames (9 channels total) for DMC visual, and 4 stacked grayscale frames for Atari.

6. **Action representation for discrete**: Actions are stored as one-hot vectors in the replay buffer and passed to the state-action encoder as continuous vectors.

7. **Target noise for discrete actions**: Gaussian noise is added to each dimension of the one-hot encoding, then argmax is taken.
