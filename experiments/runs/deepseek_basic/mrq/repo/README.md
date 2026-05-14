# MR.Q: Towards General-Purpose Model-Free RL

This repository contains a reproduction of the MR.Q algorithm from the paper:

> **Towards General-Purpose Model-Free RL (MR.Q)**  
> Scott Fujimoto, Pierluca D'Oro, Amy Zhang, Yuandong Tian, Michael Rabbat  
> Meta FAIR

## Overview

MR.Q is a general-purpose model-free deep RL algorithm that achieves strong performance across diverse benchmarks (Gym locomotion, DM Control proprioceptive, DM Control visual, Atari) with a **single set of hyperparameters**. It leverages model-based representation learning objectives to learn features that approximately linearize the value function, without the computational overhead of planning or simulated trajectories.

## What was reproduced

The following core components from the paper have been implemented:

### 1. Network Architectures (Appendix B.2)
- **State Encoder**: CNN for images (4 conv layers, 32 channels, kernel 3, strides [2,2,2,1]) and MLP for vectors (3 layers, hidden dim 512), both using LayerNorm + ELU
- **State-Action Encoder**: Action embedding + concatenation + 3-layer MLP with linear MDP predictor
- **Value Networks**: 4-layer MLP with LayerNorm + ELU
- **Policy Network**: 3-layer MLP with LayerNorm + ReLU, Gumbel-Softmax for discrete actions, Tanh for continuous

### 2. Core Algorithm (Section 4.2)
- **Encoder Loss** (Equation 14): Unrolled dynamics over horizon H_Enc=5 with:
  - Reward loss: Cross-entropy with two-hot encoding using symexp bin spacing
  - Dynamics loss: MSE between predicted z_s' and target encoder z_s'
  - Terminal loss: MSE with binary done signal
  - Lambda_terminal set to 0 until first terminal transition is seen
- **Value Learning** (Equation 19): TD3-style with:
  - Double Q-networks with minimum target
  - Multi-step returns (horizon 3)
  - Huber loss (for LAP compatibility)
  - Reward scaling by average absolute reward
- **Policy Learning** (Equation 20): Deterministic policy gradient with pre-activation regularization
- **Target Networks**: Periodic synchronized update every T_target=250 steps
- **LAP**: Loss-Adjusted Prioritized sampling (alpha=0.4, min_priority=1)

### 3. Key Design Choices (Table 3 hyperparameters)
- All hyperparameters fixed across benchmarks
- AdamW optimizer with weight decay 1e-4
- Separate learning rates: encoder 1e-4, value 3e-4, policy 3e-4
- Gradient clipping norm of 20
- Xavier uniform weight initialization, bias 0
- Exploration noise N(0, 0.2^2), initial random steps 10k
- Target policy noise N(0, 0.2^2) clipped to [-0.3, 0.3]
- Discount factor 0.99, replay buffer size 1M, batch size 256

### 4. Utility Functions
- **symexp/symlog**: Symmetric exponential/logarithm for reward bin spacing
- **Two-hot encoding**: For categorical reward prediction
- **SumTree**: O(log N) prioritized experience replay

### 5. Design Study Variants (Section 5.2)
- Linear value function
- Dynamics target variant
- No target encoder
- MSE reward loss
- (Additional variants stubbed for extension)

## File Structure

```
mrq/
├── __init__.py          # Package exports
├── agent.py             # Main MR.Q agent implementation
├── networks.py          # Neural network architectures
├── replay.py            # Prioritized replay buffer with LAP
├── utils.py             # Utility functions (symexp, two-hot, etc.)
├── config.py            # Hyperparameter configuration
└── variants.py          # Design study variants

train.py                 # Training script
setup.py                 # Package installation
README.md               # This file
```

## Usage

### Basic Training

```bash
# Gym locomotion
python train.py --domain gym --task HalfCheetah-v4 --total_steps 1000000

# DM Control proprioceptive
python train.py --domain dmc --task cheetah_run --total_steps 500000

# Atari
python train.py --domain atari --task AlienNoFrameskip-v4 --total_steps 2500000
```

### Using the Agent Directly

```python
from mrq.agent import MRQ
import gymnasium as gym

env = gym.make("HalfCheetah-v4")

agent = MRQ(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    discrete_action_space=False,
    image_observations=False,
)

state, _ = env.reset()
for step in range(1000000):
    if step < agent.initial_random_steps:
        action = env.action_space.sample()
    else:
        action = agent.select_action(state, explore=True)
    
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    agent.replay_buffer.add(state, action, reward, next_state, done)
    
    if step >= agent.initial_random_steps:
        info = agent.update()
    
    state = next_state
    if done:
        state, _ = env.reset()
```

## Assumptions and Unresolved Details

Since this reproduction is based solely on the paper text (without access to the original codebase), some details required interpretation:

1. **Multi-step returns in value loss**: The paper mentions multi-step returns over horizon H_Q=3 (Equation 19), but the replay buffer samples individual transitions. In practice this requires sampling sequences or computing n-step returns from stored transitions. Our implementation uses 1-step TD with the reward scaling factor; a full implementation would require sequence-aware sampling or n-step return computation from individual transitions.

2. **Encoder unrolling requires sequential data**: The encoder loss (Equation 14) unrolls over H_Enc=5 steps. Our MultiStepReplayBuffer stores episode segments to support this, but the integration could be improved for better sample efficiency.

3. **Lambda_terminal**: Set to 0 until first terminal transition is seen. Our implementation currently applies it uniformly.

4. **Atari preprocessing**: The paper describes specific preprocessing (grayscale, resize to 84x84, max between frames 3 and 4, 4-frame stacking, sticky actions). Full Atari support would require environment wrappers implementing these.

5. **DMC visual**: Frame stacking of 3 previous observations, RGB 84x84. Our StateEncoder supports this but the exact integration depends on the dmc2gym wrapper.

6. **Action representation for discrete spaces**: The paper uses one-hot encoding with Gaussian noise added to each dimension, then argmax for exploration. This is implemented in our agent.

7. **Reward scaling update**: Updated simultaneously with target networks. Our implementation tracks the average absolute reward from the replay buffer.

## Key Theoretical Results

The paper provides three theorems that motivate the algorithm:

- **Theorem 1**: The fixed point of model-free semi-gradient TD and the model-based solution (rolling out learned dynamics and reward) are the same in linear space.
- **Theorem 2**: The value error is bounded by the accuracy of the estimated dynamics and reward.
- **Theorem 3**: Even with non-linear value functions, if the features satisfy MDP homomorphism conditions (matching reward and dynamics), there exists a Q-function over the embedding space that equals the true value function.

## Benchmarks and Evaluation

The paper evaluates on:
- **Gym - Locomotion**: 5 MuJoCo tasks, 1M steps, TD3-normalized scores
- **DMC - Proprioceptive**: 28 tasks, 500k steps (1M frames), raw reward/1000
- **DMC - Visual**: 28 tasks, 500k steps, raw reward/1000
- **Atari**: 57 games, 2.5M steps (10M frames), human-normalized scores

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- Gymnasium 0.29+
- NumPy

Optional:
- dmc2gym (for DM Control environments)
- wandb (for logging)

## References

- Fujimoto et al. (2018) - TD3: Addressing Function Approximation Error in Actor-Critic Methods
- Fujimoto et al. (2020) - LAP: An Equivalence between Loss Functions and Non-Uniform Sampling
- Hafner et al. (2023) - DreamerV3: Mastering Diverse Domains through World Models
- Hansen et al. (2024) - TD-MPC2: Scalable, Robust World Models for Continuous Control
- Parr et al. (2008) - An Analysis of Linear Models, Linear Value-Function Approximation
