# Interpreting Emergent Planning in Model-Free Reinforcement Learning

Reproduction of Bush et al. (2024) — "Interpreting Emergent Planning in Model-Free Reinforcement Learning".

## Overview

This codebase reproduces the core contributions of the paper:

1. **DRC Agent** — Deep Repeated ConvLSTM agent (Guez et al., 2019) trained on Sokoban via IMPALA
2. **Concept Probing** — Linear probes for C_A (Agent Approach Direction) and C_B (Box Push Direction)
3. **Plan Formation Analysis** — Test-time plan refinement via thinking steps
4. **Causal Interventions** — Steering agent behavior by modifying cell state representations
5. **Training Emergence** — Correlation between concept representations and planning-like behavior

## Structure

```
repo/
├── config.py                    # All hyperparameters and constants
├── requirements.txt
├── environment/
│   └── sokoban.py               # Sokoban environment (8x8, symbolic obs)
├── data/
│   └── boxoban.py               # Boxoban dataset loader
├── model/
│   ├── convlstm.py              # ConvLSTM cell + DRC stack
│   └── drc.py                   # DRC agent + ResNet agent
├── training/
│   ├── impala.py                # V-trace / IMPALA loss
│   └── train.py                 # Training loop
├── probing/
│   ├── concepts.py              # C_A, C_B and alternative concept computation
│   ├── probes.py                # 1x1, 3x3, NxN, global linear probes
│   └── evaluate.py              # Probe training and evaluation
├── interventions/
│   └── intervene.py             # Agent-Shortcut, Box-Shortcut, Cutoff interventions
├── utils/
│   └── metrics.py               # Macro F1, evaluation utilities
└── scripts/
    ├── run_probing.py           # Section 4 — Figure 4
    ├── run_interventions.py     # Section 6.1 — Table 1
    ├── analyze_emergence.py     # Section 6.2 — Figure 9
    ├── analyze_thinking_steps.py # Section 5 — Figure 6
    ├── evaluate_agent.py        # Appendix E.5 — Figure 45
    ├── probe_alternative_concepts.py  # Appendix D.4, D.5
    ├── evaluate_agent_sizes.py  # Appendix F — DRC(1,9), DRC(9,1)
    └── evaluate_resnet.py       # Appendix G — ResNet agent
```

## Setup

```bash
pip install -r requirements.txt
```

Download the Boxoban dataset:
```bash
git clone https://github.com/deepmind/boxoban-levels data/boxoban-levels
```

## Usage

### Training

```bash
python training/train.py \
    --boxoban_path data/boxoban-levels \
    --checkpoint_dir checkpoints \
    --device cuda
```

### Probing (Section 4, Figure 4)

```bash
python scripts/run_probing.py \
    --checkpoint checkpoints/final_model.pt \
    --boxoban_path data/boxoban-levels \
    --output_dir results/probing
```

### Interventions (Section 6.1, Table 1)

```bash
python scripts/run_interventions.py \
    --checkpoint checkpoints/final_model.pt \
    --boxoban_path data/boxoban-levels \
    --output_dir results/interventions
```

### Training Emergence (Section 6.2, Figure 9)

```bash
python scripts/analyze_emergence.py \
    --checkpoint_dir checkpoints \
    --boxoban_path data/boxoban-levels \
    --output_dir results/emergence
```

### Test-Time Plan Refinement (Section 5, Figure 6)

```bash
python scripts/analyze_thinking_steps.py \
    --checkpoint checkpoints/final_model.pt \
    --boxoban_path data/boxoban-levels \
    --output_dir results/thinking_steps
```

## Key Implementation Details

### DRC(3,3) Agent
- 3 ConvLSTM layers, 3 ticks per step
- 32 hidden channels, 3x3 kernels, 8x8 spatial dims
- Bottom-up skip connections (encoding to all layers)
- Top-down skip connections (final layer output to layer 1)
- Pool-and-inject (spatial mean+max pooling of hidden state)

### Training (Appendix E.4)
- IMPALA with V-trace (γ=0.97, λ=0.97)
- Adam optimizer, lr linear decay 4e-4 → 0
- Batch size 16, unroll length 20
- Entropy penalty 1e-2, logit L2 penalty 1e-3
- Head L2 regularization 1e-5
- 250M transitions on Boxoban unfiltered training set

### Probing (Section 4, Appendix D.1)
- 1x1 probes: 160 parameters; 3x3 probes: 1440 parameters
- AdamW optimizer, 10 epochs, lr=0.001, weight_decay=0.001
- Evaluated with macro F1 (due to class imbalance from NEVER class)
- 5 seeds per experiment

### Concepts
- **C_A (Agent Approach Direction)**: For each square, direction agent will next step onto it
- **C_B (Box Push Direction)**: For each square, direction box will next be pushed off it
- Classes: {NEVER=0, UP=1, DOWN=2, LEFT=3, RIGHT=4}

### Interventions (Section 6.1)
- Add concept vector w_k to cell state: g_{x,y} ← g_{x,y} + α·w_k
- Agent-Shortcut: steer agent to take longer path using C_A vectors
- Box-Shortcut: steer agent to push box longer route using C_B vectors
- Success = agent solves level via desired suboptimal route
