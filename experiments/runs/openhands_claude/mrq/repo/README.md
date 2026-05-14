# MR.Q – Model-based Representations for Q-learning

Reproduction of **"Towards General-Purpose Model-Free Reinforcement Learning"**
(Fujimoto, D'Oro, Zhang, Tian, Rabbat – Meta FAIR, 2024).

---

## Overview

MR.Q is a model-free deep RL algorithm that learns state and state-action
embeddings via model-based objectives (reward, dynamics, terminal prediction),
then uses those embeddings for standard TD3-style value and policy learning.
A single set of hyperparameters is used across all benchmarks.

---

## Repository structure

```
repo/
├── config.py          # MRQConfig dataclass + per-benchmark presets
├── utils.py           # symexp/symlog, two-hot encoding, reward bins
├── model.py           # All neural network components + MRQAgent
├── replay_buffer.py   # LAP prioritised replay buffer with sequence sampling
├── envs.py            # Environment wrappers (Gym, DMC, Atari)
├── train.py           # Training loop + CLI entry point
├── requirements.txt
└── README.md
```

---

## Algorithm

### Networks

| Network | Input | Output | Architecture |
|---------|-------|--------|--------------|
| `f_ω` (state encoder) | `s` | `z_s` (512) | CNN (image) or 3-layer MLP (vector) |
| `g_ω` (state-action encoder) | `(z_s, a)` | `z_sa` (512) | 3-layer MLP |
| `m` (linear MDP predictor) | `z_sa` | `(z̃_s', r̃, d̃)` | Single linear layer |
| `Q_θ` (×2) | `z_sa` | scalar | 4-layer MLP |
| `π_φ` | `z_s` | `a` | 3-layer MLP |

### Losses

**Encoder** (unrolled over `H_Enc = 5` steps):
```
L_Enc = Σ_t [ λ_R · CE(r̃^t, TwoHot(r^t))
             + λ_D · MSE(z̃_s'^t, f_{ω'}(s'^t))
             + λ_T · MSE(d̃^t, d^t) ]
```

**Value** (multi-step returns, `H_Q = 3`, Huber loss):
```
y = (1/r̄) · (Σ_{t=0}^{H_Q-1} γ^t r_t + γ^{H_Q} · r̄' · min_j Q_{θ'_j}(z_sa'))
L_Value = Huber(Q_i(z_sa), y)
```

**Policy** (deterministic policy gradient):
```
L_Policy = -0.5 · Σ_i Q_i(z_sa_π) + λ_pre-activ · ||z_π||²
```

### Update schedule

Every `T_target = 250` steps: hard-copy target networks and update reward scaling.
One gradient update per environment step (replay ratio = 1).

---

## Hyperparameters (Table 3)

| Parameter | Value |
|-----------|-------|
| `λ_Dynamics` | 1.0 |
| `λ_Reward` | 0.1 |
| `λ_Terminal` | 0.1 |
| `λ_pre-activ` | 1e-5 |
| `H_Enc` | 5 |
| `H_Q` | 3 |
| Target noise σ | N(0, 0.2²) |
| Target noise clip | ±0.3 |
| LAP α | 0.4 |
| Min priority | 1 |
| Init random steps | 10k |
| Exploration noise | N(0, 0.2²) |
| γ | 0.99 |
| Buffer size | 1M |
| Batch size | 256 |
| `T_target` | 250 |
| Encoder LR | 1e-4 |
| Value LR | 3e-4 |
| Policy LR | 3e-4 |
| Weight decay | 1e-4 |
| Value grad clip | 20 |
| `z_s` dim | 512 |
| `z_sa` dim | 512 |
| `z_a` dim | 256 |
| Hidden dim | 512 |
| Reward bins | 65 |
| Reward range | [-10, 10] (symexp) |
| Gumbel-Softmax τ | 10 |
| Activation (enc/value) | ELU |
| Activation (policy) | ReLU |
| Weight init | Xavier uniform |
| Bias init | 0 |
| Optimiser | AdamW |

---

## Usage

### Gym locomotion (1M steps)
```bash
python train.py --benchmark gym --env HalfCheetah-v4 --seed 0
```

### DMC proprioceptive (500k steps)
```bash
python train.py --benchmark dmc_proprio --env cheetah-run --seed 0
```

### DMC visual (500k steps)
```bash
python train.py --benchmark dmc_visual --env cheetah-run --seed 0
```

### Atari (2.5M steps)
```bash
python train.py --benchmark atari --env Breakout --seed 0
```

Results are written to `results/<env>_seed<N>.csv`.

---

## Benchmarks evaluated in the paper

| Benchmark | Environments | Steps | Obs | Actions |
|-----------|-------------|-------|-----|---------|
| Gym Locomotion | 5 MuJoCo tasks | 1M | vector | continuous |
| DMC Proprioceptive | 28 tasks | 500k | vector | continuous |
| DMC Visual | 28 tasks | 500k | 84×84 RGB (×3) | continuous |
| Atari-57 | 57 games | 2.5M | 84×84 grey (×4) | discrete |

---

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
gymnasium>=0.29.0
gymnasium[mujoco]
gymnasium[atari]
ale-py>=0.8.0
dm-control>=1.0.0
```
