# Emergent Planning in Model-Free Reinforcement Learning

Reproduction of the experiments from:

> Thomas Bush, Stephen Chung, Usman Anwar, Adrià Garriga-Alonso, David Krueger
> *Interpreting Emergent Planning in Model-Free Reinforcement Learning*

This codebase reproduces the core contributions:
1. Training a Deep Repeated ConvLSTM (DRC) agent on Sokoban
2. Probing for planning-relevant concept representations (C_A, C_B)
3. Investigating plan formation through iterative refinement
4. Verifying causal influence via interventions on concept representations

## Codebase Structure

```
repo/
├── configs/
│   └── config.py             # All hyperparameters and configuration
├── environment/
│   └── sokoban.py            # Sokoban environment (symbolic 8x8x7)
├── models/
│   ├── convlstm.py           # ConvLSTM cell and layer implementations
│   └── drc.py                # DRC agent architecture (D=3, N=3)
├── training/
│   ├── vtrace.py             # V-trace loss computation
│   └── trainer.py            # IMPALA trainer
├── probing/
│   ├── concepts.py           # Concept labelers (C_A, C_B)
│   ├── linear_probe.py       # Linear probe training and evaluation
│   └── dataset.py            # Probe data collection
├── interventions/
│   ├── intervene.py          # Intervention engine
│   └── levels.py             # Handcrafted level generators
├── analysis/
│   ├── visualize.py          # Plan visualization tools
│   └── thinking_time.py      # Thinking time analysis
├── train.py                  # Main training script
├── run_probing.py            # Probe training experiments (Section 4)
├── run_interventions.py      # Intervention experiments (Section 6)
├── run_thinking_time.py      # Thinking time analysis (Sections 5 & 6.2)
├── requirements.txt
└── README.md
```

## Key Components

### DRC Agent (models/drc.py)
- 3 ConvLSTM layers, 3 internal ticks per step
- 32 hidden channels, kernel size 3
- Bottom-up and top-down skip connections
- Pool-and-inject mechanism

### Training (training/)
- IMPALA with V-trace (gamma=0.97, lambda=0.97)
- 250M transitions over Boxoban unfiltered training set
- Adam optimizer, linear LR decay from 4e-4 to 0
- L2 penalty on action logits (1e-3)
- Entropy bonus (0.01)

### Linear Probing (probing/)
- 1x1 and 3x3 probes on ConvLSTM cell states
- Concepts: Agent Approach Direction (C_A), Box Push Direction (C_B)
- AdamW optimizer, 10 epochs, 5 seeds

### Interventions (interventions/)
- Agent-Shortcut: steer agent to follow longer path
- Box-Shortcut: steer agent to push box longer route
- Cutoff: replicate effect of thinking time

## Usage

### Training
```bash
python train.py --data_dir data/ --device cuda
```

### Probing
```bash
python run_probing.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir data/ \
    --concept both
```

### Interventions
```bash
python run_interventions.py \
    --checkpoint checkpoints/best_model.pt \
    --probe_A results/probes/probe_agent_approach_layer2_k1.pt \
    --probe_B results/probes/probe_box_push_layer2_k1.pt \
    --intervention_type all
```

### Thinking Time Analysis
```bash
python run_thinking_time.py \
    --checkpoint checkpoints/best_model.pt \
    --probe_A results/probes/probe_agent_approach_layer2_k1.pt \
    --analysis_type all
```

## Requirements

See `requirements.txt`. Core dependencies:
- PyTorch >= 1.10
- NumPy
- scikit-learn
- Matplotlib
- TensorBoard

## Data

The code expects the Boxoban dataset from:
https://github.com/deepmind/boxoban-levels/

Place it under `data/boxoban-levels-master/`.
