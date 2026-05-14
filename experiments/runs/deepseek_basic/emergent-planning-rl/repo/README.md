# Interpreting Emergent Planning in Model-Free RL — Reproduction

This repository contains a reproduction of the core contributions from the ICLR 2025 Oral paper:

> **Interpreting Emergent Planning in Model-Free Reinforcement Learning**  
> Thomas Bush, Stephen Chung, Usman Anwar, Adrià Garriga-Alonso, David Krueger

## Overview

The paper presents the first mechanistic evidence that model-free RL agents can learn to plan. The authors demonstrate that a Deep Repeated ConvLSTM (DRC) agent trained on Sokoban internally forms plans using learned concept representations that predict long-term effects of actions.

The methodology involves three steps:
1. **Probing** for planning-relevant concepts (Section 4)
2. **Investigating plan formation** within representations (Section 5)
3. **Verifying causal influence** through interventions (Section 6)

## Repository Structure

```
├── src/                      # Core implementation
│   ├── sokoban.py            # Sokoban environment (8×8 symbolic)
│   ├── drc_agent.py          # DRC(D,N) architecture with ConvLSTM
│   ├── probes.py             # Linear probes (1×1, 3×3, K×K)
│   ├── concept_labels.py     # C_A and C_B concept labeling
│   ├── interventions.py      # Intervention procedures (Algorithms 1, 2)
│   ├── analysis.py           # Probing, emergence, plan refinement analysis
│   ├── visualization.py      # Plan visualization (Figures 4-9 style)
│   ├── training.py           # IMPALA-based training
│   ├── level_utils.py        # Level generation utilities
│   └── data_utils.py         # Boxoban dataset loading
├── scripts/
│   └── run_full_pipeline.py  # End-to-end experiment pipeline
├── notebooks/
│   └── demo_probing.ipynb    # Probing demonstration notebook
├── configs/
│   └── default.yaml          # Configuration file
└── README.md
```

## What Was Reproduced

### 1. Environment (Section 2.2, Appendix E.2)
- Full Sokoban environment with 8×8 grid, symbolic observations (7-channel one-hot)
- Correct transition dynamics (box pushing, wall collisions)
- Reward structure (-0.01 per step, +1 for box on target, -1 for box off target, +10 for solving)
- Episode length randomly sampled from [115, 120]

### 2. DRC Agent Architecture (Section 2.3, Appendix E.3)
- DRC(3,3) with 3 ConvLSTM layers and 3 internal ticks per step
- 32 hidden channels, kernel size 3, single-layer zero padding
- Bottom-up skip connections (input to all layers)
- Top-down skip connections (top layer output to bottom layer on next tick)
- Pool-and-Inject mechanism (mean+max spatial pooling with affine transform)
- Actor-critic architecture with policy and value heads
- DRC(1,9) and DRC(9,1) variants supported (Appendix F)
- ResNet agent architecture supported (Appendix G)

### 3. Concept-Based Interpretability (Section 3, Section 4)
- C_A (Agent Approach Direction): 5 classes {NEVER, UP, DOWN, LEFT, RIGHT}
- C_B (Box Push Direction): 5 classes {NEVER, UP, DOWN, LEFT, RIGHT}
- Behavioral-dependent concept labeling from episode trajectories
- Additional concepts: binary simplifications, reversed asymmetrical variants (Appendix D.4)
- Global probes for future action concepts (Appendix D.5)

### 4. Linear Probes (Section 4, Appendix D)
- 1×1, 3×3, 5×5, and 7×7 spatial probes implemented as convolutions
- Trainable via AdamW with cross-entropy loss
- Macro F1 evaluation metric
- Per-class precision, recall, F1 reporting
- Baseline probes using raw observations
- Probe vector extraction for interventions

### 5. Plan Formation Analysis (Section 5, Appendix A)
- Test-time plan refinement measurement during "thinking steps"
- Multi-layer plan decoding
- Evidence of iterative search behavior
- Corridor length experiments (Appendix A.3.2)

### 6. Intervention Experiments (Section 6, Appendix B)
- Agent-Shortcut interventions (Algorithm 1)
- Box-Shortcut interventions (Algorithm 2)
- Cutoff level interventions (Appendix B.3)
- `g'_{x,y} = g_{x,y} + α·w_k` intervention mechanism
- Variable intervention strength α and number of squares p
- Short-route (NEVER) and directional intervention components
- Random probe baseline comparison

### 7. Training & Emergence Analysis (Section 6.2, Appendix C)
- IMPALA-based training (discount γ=0.97, V-trace λ=0.97)
- L2 penalties on logits and heads
- Entropy regularization
- Checkpoint-based emergence tracking
- Correlation analysis between concept representation quality and planning behavior

### 8. Visualization (Figures 4-9, Appendices)
- Internal plan visualization with teal (C_A) and purple (C_B) arrows
- Probe performance bar charts (Figure 4 style)
- Test-time refinement plots (Figure 6 style)
- Emergence correlation plots (Figure 9 style)
- Intervention effect visualizations (Figures 7, 8 style)

## Key Assumptions & Unresolved Details

1. **Pretrained agent weights**: The paper analyzes a fully-trained DRC(3,3) agent trained for 250M transitions. We provide the architecture and training code but actual training requires substantial compute and the Boxoban dataset.

2. **Boxoban dataset**: The paper uses the DeepMind Boxoban dataset of 900K Sokoban levels. We provide a loader but the dataset must be downloaded separately from https://github.com/deepmind/boxoban-levels/.

3. **Probe training details**: The paper trains probes on 106.6K/25.7K transitions (train/test). Our implementation supports this scale but defaults to smaller datasets for quick iteration.

4. **Multi-seed averaging**: The paper uses 5 independent initialization seeds for probes and interventions. Our framework supports this.

5. **Intervention level details**: The exact Agent-Shortcut and Box-Shortcut level layouts are described schematically in the paper. We provide representative implementations that capture the key properties (short/long path choice).

6. **DRC(1,9) and DRC(9,1) agents**: These variants are described in Appendix F. We provide the flexible DRC(D,N) architecture that supports these configurations.

7. **Mini PacMan environment** (Appendix H): Not implemented as it's described as preliminary.

## Usage

### Installation
```bash
pip install torch numpy scikit-learn matplotlib scipy
```

### Running the Pipeline
```bash
python scripts/run_full_pipeline.py --device cpu --output_dir results/
```

### Quick Demo
```python
from src.sokoban import SokobanEnv
from src.drc_agent import DRCAgent
from src.probes import LinearProbe

# Create environment and agent
env = SokobanEnv()
agent = DRCAgent(D=3, N=3)

# Create a linear probe
probe = LinearProbe(in_channels=32, kernel_size=1)
```

## Dependencies
- Python 3.8+
- PyTorch 1.10+
- NumPy
- scikit-learn
- Matplotlib
- SciPy

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{bush2025interpreting,
  title={Interpreting Emergent Planning in Model-Free Reinforcement Learning},
  author={Bush, Thomas and Chung, Stephen and Anwar, Usman and Garriga-Alonso, Adri{\`a} and Krueger, David},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
