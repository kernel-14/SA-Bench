# Robotic World Model (RWM)

Reproduction of the paper: **"Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics"** (Li, Krause, Hutter).

## Codebase Structure

```
repo/
├── config.py                 # All hyperparameters and configurations
├── model/
│   ├── __init__.py
│   ├── rwm.py                # Robotic World Model with dual-autoregressive mechanism
│   ├── policy.py             # PPO Actor & Critic networks
│   ├── baselines.py          # MLP, RSSM, Transformer baselines
│   └── modules.py            # Shared components (MLP builder, GaussianHead, RSSM)
├── training/
│   ├── __init__.py
│   ├── world_model_trainer.py  # Autoregressive training loop for RWM
│   └── mbpo_ppo_trainer.py    # MBPO-PPO policy optimization
├── data/
│   ├── __init__.py
│   ├── dataset.py            # SlidingWindowDataset for autoregressive training
│   └── replay_buffer.py      # Replay buffer for MBPO-PPO
├── env/
│   ├── __init__.py
│   ├── rewards.py            # Velocity tracking reward functions
│   └── tasks.py              # Task definitions & velocity command generation
├── utils/
│   ├── __init__.py
│   └── metrics.py            # Evaluation metrics
├── train_world_model.py      # Entry point: train RWM on collected trajectories
├── train_policy.py           # Entry point: MBPO-PPO policy optimization
├── evaluate.py               # Entry point: evaluate world model prediction error
├── noise_robustness.py       # Entry point: noise robustness experiments (Sec 4.2)
└── requirements.txt          # Python dependencies
```

## Key Contributions Reproduced

1. **RWM Architecture** (Sec 3.2): GRU-based world model with dual-autoregressive mechanism
   - Inner autoregression: GRU hidden states updated over M=32 history steps
   - Outer autoregression: Predicted observations fed back over N=8 forecast steps
   - Gaussian output heads for observations and privileged information

2. **Autoregressive Training** (Eq. 2): Multi-step prediction loss with decay factor α
   - L = (1/N) * Σ_{k=1}^{N} α^k [L_o(o'_{t+k}, o_{t+k}) + L_c(c'_{t+k}, c_{t+k})]

3. **MBPO-PPO** (Algorithm 1): Policy optimization in imagination
   - Imagination rollouts of T=100 steps over 4096 parallel environments
   - PPO with clipped surrogate objective, GAE advantage estimation
   - World model fine-tuning on real environment data

4. **Baselines** (Sec 4.3): MLP, RSSM (Dreamer-style), Transformer
   - All implemented with same M, N context for fair comparison

5. **Noise Robustness** (Sec 4.2): Evaluation under Gaussian noise perturbations

6. **Reward Functions** (Sec A.1.2): Complete velocity tracking reward terms with
   correct weights for both ANYmal D and Unitree G1

## Configuration

Hyperparameters are specified in `config.py` and follow Tables S2-S11 in the paper.

### RWM Training (Table S10)
- History horizon M = 32, Forecast horizon N = 8
- Learning rate = 1e-4, Weight decay = 1e-5
- Batch size = 1024, Max iterations = 2500

### MBPO-PPO Training (Table S11)
- Imagination environments = 4096, Steps per iteration = 100
- Buffer size = 1000, Learning rate = 0.001
- Learning epochs = 5, Mini-batches = 4
- Discount factor γ = 0.99, Clip range ε = 0.2

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train world model on collected data
python train_world_model.py --robot anymal_d

# Evaluate autoregressive prediction accuracy
python evaluate.py --model_path checkpoints/rwm.pt

# Noise robustness experiments
python noise_robustness.py --model_path checkpoints/rwm.pt

# Train policy with MBPO-PPO
python train_policy.py --robot anymal_d --world_model checkpoints/rwm.pt
```

## Robot Specifications

- **ANYmal D**: 45-dim observation, 12-dim action, 8-dim privileged info
- **Unitree G1**: 96-dim observation, 29-dim action, 30-dim privileged info

## References

- Li, Krause, Hutter. "Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics."
- Janner et al. "When to Trust Your Model: Model-Based Policy Optimization" (MBPO)
- Hafner et al. "Dream to Control: Learning Behaviors by Latent Imagination" (RSSM)
