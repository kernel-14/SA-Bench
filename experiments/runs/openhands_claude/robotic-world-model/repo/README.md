# Robotic World Model (RWM)

Implementation of **"Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics"** (Li et al., ETH Zurich).

## Structure

```
repo/
├── config.py          # All hyperparameters and robot configurations
├── layers.py          # Basic building blocks: MLP, GRU, GaussianHead, Transformer layers
├── modules.py         # Core modules: RWMCore, MLPWorldModel, RSSMWorldModel,
│                      #   TransformerWorldModel, PolicyNetwork, ValueNetwork
├── model.py           # Top-level wrappers with loss computation:
│                      #   RWM, MLPBaseline, RSSMBaseline, TransformerBaseline, ActorCritic
├── data.py            # Trajectory, TrajectoryDataset (sliding window), ReplayBuffer,
│                      #   ImaginaryRolloutBuffer, RunningNormalizer
├── rewards.py         # All reward functions from Sec. A.1.2 (Table S6)
├── train.py           # World model training loop (Algorithm 1, steps 3-4)
├── train_policy.py    # MBPO-PPO training loop (Algorithm 1, full)
├── evaluate.py        # Evaluation: relative error, noise robustness, ablations
└── requirements.txt
```

## Key Contributions Implemented

### 1. Dual-Autoregressive World Model (Sec. 3.2)

`RWMCore` in `modules.py` implements the dual-autoregressive mechanism:
- **Inner autoregression**: GRU hidden state updated step-by-step over history horizon M=32
- **Outer autoregression**: predicted observations fed back as input over forecast horizon N=8
- Architecture: GRU(256, 256) base + MLP(128, ReLU) heads for observation and privileged info

### 2. Self-Supervised Autoregressive Training (Eq. 2)

`WorldModelLoss` in `model.py` implements the multi-step prediction loss:

```
L = (1/N) * sum_{k=1}^{N} alpha^k * [L_o(o'_{t+k}, o_{t+k}) + L_c(c'_{t+k}, c_{t+k})]
```

with alpha=1.0 (no decay) and Gaussian NLL for both observation and privileged info losses.

### 3. MBPO-PPO Policy Optimization (Algorithm 1, Sec. 3.3)

`MBPOPPOTrainer` in `train_policy.py` implements the full loop:
1. Collect real data → replay buffer
2. Update world model with autoregressive training
3. Initialize 4096 imagination agents from replay buffer samples
4. Roll out T=100 imagination steps using policy + world model
5. Update policy with PPO (clip=0.2, entropy=0.005, epochs=5)

### 4. Baselines

All baselines from Sec. 4.3 / Fig. 4:
- **MLP**: hidden=(256,256), ReLU, trained autoregressively
- **RSSM**: GRU(256, 2 layers), latent_dim=64, categorical prior (32 categories)
- **Transformer**: decoder-only, d_model=64, 8 heads, 2 layers, sinusoidal PE, context=32
- **RWM-TF**: RWM trained with teacher-forcing (N=1 special case)

## Robot Configurations

| Robot | WM obs dim | Action dim | Privileged dim | Policy obs dim |
|-------|-----------|------------|----------------|----------------|
| ANYmal D | 45 | 12 | 8 | 48 |
| Unitree G1 | 96 | 29 | 30 | 99 |

## Usage

### Train World Model

```bash
# Train RWM with autoregressive training on ANYmal D data
python train.py --robot anymal_d --model rwm --data-dir /path/to/data

# Train all models for comparison (Fig. 4)
python train.py --robot anymal_d --model all --data-dir /path/to/data

# Quick test with synthetic data
python train.py --robot anymal_d --model rwm --synthetic
```

### Train Policy with MBPO-PPO

```bash
# Train policy using pretrained world model
python train_policy.py --robot anymal_d \
    --wm-checkpoint checkpoints/checkpoint_final.pt \
    --data-dir /path/to/data \
    --vel-cmd 1.0 0.0 0.0

# Quick test with synthetic data
python train_policy.py --robot anymal_d --synthetic
```

### Evaluate

```bash
python evaluate.py --robot anymal_d \
    --checkpoint checkpoints/checkpoint_final.pt \
    --data-dir /path/to/data \
    --noise-levels 0.0 0.01 0.05 0.1
```

## Training Parameters

### World Model (Table S10)
| Parameter | Value |
|-----------|-------|
| History horizon M | 32 |
| Forecast horizon N | 8 |
| Forecast decay α | 1.0 |
| Batch size | 1024 |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Max iterations | 2500 |

### MBPO-PPO (Table S11)
| Parameter | Value |
|-----------|-------|
| Imagination envs | 4096 |
| Imagination steps T | 100 |
| Buffer size \|D\| | 1000 |
| Learning rate | 0.001 |
| Learning epochs | 5 |
| Mini-batches | 4 |
| KL target | 0.01 |
| Discount γ | 0.99 |
| Clip range ε | 0.2 |
| Entropy coef | 0.005 |

## Notes

- The world model is pretrained on simulation data before policy optimization (Sec. A.4.3)
- Real environment interaction (Isaac Lab / hardware) requires external integration
- The `collect_real_data` method in `MBPOPPOTrainer` provides the interface for simulator/hardware integration
- Termination detection uses predicted privileged information (contact signals) as described in Sec. A.4.3
