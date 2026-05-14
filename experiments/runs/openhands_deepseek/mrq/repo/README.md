# MR.Q: Model-based Representations for Q-learning

Reproduction of "Towards General-Purpose Model-Free Reinforcement Learning" (Fujimoto et al., 2024).

## Structure

```
repo/
├── mrq/
│   ├── __init__.py
│   ├── config.py          # All hyperparameters (Table 3)
│   ├── utils.py           # symexp, two-hot encoding, reward scaler
│   ├── networks.py        # State encoder, SA encoder, Q nets, policy net
│   ├── replay_buffer.py   # LAP prioritized replay buffer with episode tracking
│   ├── agent.py           # MR.Q agent (encoder/value/policy losses)
│   ├── trainer.py         # Training loop + environment wrappers
│   └── main.py            # CLI entry point
├── requirements.txt
└── README.md
```

## Usage

```bash
pip install -r requirements.txt

# Gym Locomotion
python -m mrq.main --benchmark gym --env HalfCheetah-v4

# DMC Proprioceptive
python -m mrq.main --benchmark dmc_proprio --env cheetah-run

# DMC Visual
python -m mrq.main --benchmark dmc_visual --env cheetah-run

# Atari
python -m mrq.main --benchmark atari --env Alien

# Run all environments in a benchmark
python -m mrq.main --benchmark gym
```

## Key Hyperparameters

All hyperparameters are fixed across benchmarks (Table 3 of the paper):
- H_Enc = 5 (encoder unroll horizon)
- H_Q = 3 (multi-step returns)
- λ_Dynamics = 0.1, λ_Reward = 0.1, λ_Terminal = 0.1
- Target update every 250 steps
- LAP with α = 0.4
- AdamW optimizer
