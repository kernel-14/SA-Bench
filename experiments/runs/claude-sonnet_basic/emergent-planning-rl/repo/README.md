# Interpreting Emergent Planning in Model-Free Reinforcement Learning

Reproduction of the paper: **"Interpreting Emergent Planning in Model-Free Reinforcement Learning"** by Thomas Bush, Stephen Chung, Usman Anwar, Adria Garriga-Alonso, and David Krueger (ICLR 2025 Oral).

## Overview

This repository reproduces the core contributions of the paper, which provides the first mechanistic evidence that model-free RL agents can learn to plan. The paper studies a Deep Repeated ConvLSTM (DRC) agent playing Sokoban and demonstrates:

1. **Concept Probing** (Section 4): The agent linearly represents planning-relevant concepts C_A (Agent Approach Direction) and C_B (Box Push Direction) in its cell states.

2. **Plan Formation** (Section 5): The agent forms internal plans via iterative search, resembling parallelized bidirectional search.

3. **Causal Interventions** (Section 6.1): The agent's concept representations causally influence its behavior - interventions on cell states can steer the agent to follow suboptimal plans.

4. **Training Emergence** (Section 6.2): The emergence of concept representations coincides with the emergence of planning-like behavior (ability to benefit from extra test-time compute).

## Repository Structure

```
├── src/
│   ├── agent/
│   │   └── drc_agent.py          # DRC(D,N) agent implementation
│   ├── environment/
│   │   ├── sokoban.py            # Sokoban environment with symbolic observations
│   │   └── boxoban_loader.py     # Loader for Boxoban dataset
│   ├── probing/
│   │   ├── concepts.py           # C_A and C_B concept definitions
│   │   ├── linear_probe.py       # Linear probe implementation
│   │   └── probe_trainer.py      # Probe training and evaluation
│   ├── interventions/
│   │   └── interventions.py      # Agent-Shortcut and Box-Shortcut interventions
│   ├── training/
│   │   └── impala_trainer.py     # IMPALA-based training
│   └── utils/
│       ├── data_collection.py    # Episode data collection
│       └── visualization.py      # Visualization utilities
├── scripts/
│   ├── train_agent.py            # Train DRC agent
│   ├── train_probes.py           # Train linear probes (Section 4)
│   ├── analyze_thinking_steps.py # Analyze plan refinement (Section 5, Figure 6)
│   └── analyze_training_emergence.py  # Training emergence analysis (Section 6.2)
├── configs/
│   └── drc_config.yaml           # Configuration file
└── requirements.txt
```

## Key Components

### DRC Agent (`src/agent/drc_agent.py`)

Implements the Deep Repeated ConvLSTM (DRC) agent from Guez et al. (2019):
- **D=3** ConvLSTM layers, **N=3** ticks per step
- **32 hidden channels**, kernel size 3
- Bottom-up skip connections (input encoding to all layers)
- Top-down skip connections (final layer output to first layer)
- Pool-and-inject mechanism (spatial pooling for global context)

### Sokoban Environment (`src/environment/sokoban.py`)

Implements the Sokoban environment with:
- **Symbolic observations**: x_t ∈ R^{8×8×7} (one-hot encoded)
- **5 actions**: noop, up, down, left, right
- **Rewards**: -0.01/step, +1 box on target, -1 box off target, +10 solved

### Concept Definitions (`src/probing/concepts.py`)

Defines the two planning-relevant concepts:
- **C_A (Agent Approach Direction)**: For each square, encodes the direction from which the agent will next move onto it (UP/DOWN/LEFT/RIGHT/NEVER)
- **C_B (Box Push Direction)**: For each square, encodes the direction in which the next box will be pushed off it (UP/DOWN/LEFT/RIGHT/NEVER)

### Linear Probes (`src/probing/linear_probe.py`)

Implements linear probes as convolutions:
- **1x1 probes**: 160 parameters (32 channels × 5 classes)
- **3x3 probes**: 1440 parameters (32 × 9 × 5)
- Trained with AdamW, 10 epochs, lr=0.001, weight_decay=0.001

### Interventions (`src/interventions/interventions.py`)

Implements the two intervention types from Section 6.1:
- **Agent-Shortcut**: Steer agent to take longer path using C_A vectors
- **Box-Shortcut**: Steer agent to push box longer route using C_B vectors

Intervention formula: `g'_{x,y} ← g_{x,y} + α × w_k`

## Usage

### 1. Train the DRC Agent

```bash
python scripts/train_agent.py \
    --data_dir /path/to/boxoban \
    --output_dir checkpoints \
    --D 3 --N 3 --hidden_channels 32 \
    --total_transitions 250000000
```

### 2. Train Linear Probes (Section 4)

```bash
python scripts/train_probes.py \
    --agent_path checkpoints/final_agent.pt \
    --data_dir /path/to/boxoban \
    --output_dir probe_results \
    --num_train_episodes 3000 \
    --num_test_episodes 1000 \
    --num_seeds 5
```

This reproduces Figure 4 from the paper, showing macro F1 scores for C_A and C_B probes at each layer.

### 3. Analyze Plan Refinement (Section 5, Figure 6)

```bash
python scripts/analyze_thinking_steps.py \
    --agent_path checkpoints/final_agent.pt \
    --probe_dir probe_results \
    --data_dir /path/to/boxoban \
    --num_episodes 1000 \
    --num_thinking_steps 5
```

This reproduces Figure 6, showing that probe F1 improves with additional test-time compute.

### 4. Analyze Training Emergence (Section 6.2, Figure 9)

```bash
python scripts/analyze_training_emergence.py \
    --checkpoint_dir checkpoints \
    --data_dir /path/to/boxoban \
    --output_dir emergence_results
```

This reproduces Figure 9, showing the correlation between concept representation quality and planning-like behavior during training.

## Data

The Boxoban dataset can be downloaded from: https://github.com/deepmind/boxoban-levels

Expected directory structure:
```
boxoban/
├── train/
│   └── *.txt
├── valid/
│   └── *.txt
├── test/
│   └── *.txt
├── medium/
│   └── *.txt
└── hard/
    └── *.txt
```

## What Was Reproduced

### Core Contributions (Main Paper)

1. ✅ **DRC Agent Architecture** (Section 2.3, Appendix E.3): Full implementation of DRC(D,N) with all architectural features (skip connections, pool-and-inject, multiple ticks).

2. ✅ **Sokoban Environment** (Section 2.2, Appendix E.2): Symbolic observation representation, correct transition dynamics and reward structure.

3. ✅ **Concept Definitions** (Section 3.2): C_A (Agent Approach Direction) and C_B (Box Push Direction) with correct class assignments.

4. ✅ **Linear Probe Training** (Section 4, Appendix D.1): 1x1 and 3x3 probes, AdamW optimizer, 5 seeds, macro F1 evaluation.

5. ✅ **Thinking Steps Analysis** (Section 5, Figure 6): Plan refinement measurement during forced stationary steps.

6. ✅ **Intervention Framework** (Section 6.1): Agent-Shortcut and Box-Shortcut interventions with trained/random probe comparison.

7. ✅ **Training Emergence Analysis** (Section 6.2, Figure 9): Correlation between concept representations and planning-like behavior.

8. ✅ **IMPALA Training** (Appendix E.4): V-trace returns, entropy bonus, L2 regularization, linear LR decay.

### Assumptions and Unresolved Details

1. **DRC Architecture Details**: The paper describes the architecture at a high level. The exact implementation of skip connections and pool-and-inject may differ slightly from the original. Specifically:
   - The paper says "top-down skip connections" feed the final layer output to the first layer on the next tick. Our implementation does this.
   - The pool-and-inject projects mean+max pooled hidden state back to spatial dimensions.

2. **Training Setup**: The paper uses IMPALA with distributed actors. Our implementation is a simplified single-machine version. The exact training dynamics may differ.

3. **Intervention Levels**: The paper uses 25 handcrafted levels × 8 augmentations = 200 levels for each intervention type. We provide the intervention framework but not the specific handcrafted levels (which would need to be created manually).

4. **Boxoban Dataset**: The paper uses the "unfiltered" Boxoban training set. The exact split used may affect results.

5. **Concept Label Computation**: The paper computes C_A and C_B based on the agent's future behavior. Our implementation correctly tracks agent and box trajectories to compute these labels.

## References

- Bush et al. (2025): "Interpreting Emergent Planning in Model-Free Reinforcement Learning" (ICLR 2025 Oral)
- Guez et al. (2019): "An Investigation of Model-Free Planning" (ICML 2019)
- Espeholt et al. (2018): "IMPALA: Scalable Distributed Deep-RL" (ICML 2018)
- Guez et al. (2018): "An Investigation of Model-Free Planning: Boxoban Levels" (Dataset)
