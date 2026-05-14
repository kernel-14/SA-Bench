# Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

Reproduction of the paper by Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham Kakade, and Sitan Chen (ICML 2025 Outstanding Paper).

## Overview

This repository reproduces the core contributions of the paper, which examines the role of token ordering in Masked Diffusion Models (MDMs):

1. **Training complexity**: MDMs train on exponentially more masking subproblems than ARMs, and many of these are computationally harder.
2. **Adaptive inference**: By choosing which tokens to unmask adaptively (based on model confidence), MDMs can sidestep hard subproblems and dramatically improve performance.

## Key Results Reproduced

### Table 1: L&O-NAE-SAT Adaptive Inference
| (N, P) | Vanilla Inference | Adaptive Inference |
|--------|------------------|-------------------|
| (25, 275) | 78.06% | 93.76% |
| (30, 270) | 75.70% | 93.54% |
| (40, 260) | 74.60% | 92.21% |
| (50, 250) | 67.94% | 90.01% |
| (100, 200) | 62.84% | 88.91% |

### Table 2: Sudoku Puzzle Solving
| Method | # Params | Accuracy |
|--------|----------|----------|
| MDM (vanilla) | 6M | 6.88% |
| MDM (Top probability) | 6M | 18.51% |
| MDM (Top prob. margin) | 6M | **89.49%** |
| ARM (w/o ordering) | 42M | 9.73% |
| ARM (with ordering) | 42M | 87.18% |

### Table 3: Zebra Puzzle Solving
| Method | # Params | Accuracy |
|--------|----------|----------|
| MDM (vanilla) | 19M | 76.9% |
| MDM (Top probability) | 19M | 98.5% |
| MDM (Top prob. margin) | 19M | 98.3% |

### Table 4: LLaDA 8B on Coding and Math
| Method | HumanEval-Single | HumanEval-Multi | HumanEval-Split | Math |
|--------|-----------------|-----------------|-----------------|------|
| Vanilla | 31.8% | 16.5% | 14.2% | 28.5% |
| Top probability | 32.9% | 20.8% | 18.4% | 31.3% |
| Top prob. margin | **33.5%** | **25.4%** | **22.3%** | **34.3%** |

## Repository Structure

```
submission/
├── src/
│   ├── lo_nae_sat.py              # L&O-NAE-SAT distribution (Section 3.3)
│   ├── mdm_model.py               # MDM transformer architecture
│   ├── mdm_training.py            # MDM training objective
│   ├── adaptive_inference.py      # Adaptive inference strategies (Section 4)
│   ├── pi_learner.py              # Pi-learner experiments (Section 3.2)
│   ├── sudoku.py                  # Sudoku puzzle dataset and evaluation
│   ├── zebra_puzzle.py            # Zebra puzzle dataset and evaluation
│   ├── error_imbalance_analysis.py # Error imbalance analysis (Section 3.3)
│   ├── train_mdm_lo_nae_sat.py    # Training script for L&O-NAE-SAT
│   ├── train_mdm_sudoku.py        # Training script for Sudoku
│   └── llada_adaptive_inference.py # LLaDA 8B adaptive inference (Section 4.4)
├── configs/
│   ├── lo_nae_sat_config.yaml     # Config for L&O-NAE-SAT experiments
│   ├── sudoku_config.yaml         # Config for Sudoku experiments
│   └── pi_learner_config.yaml     # Config for pi-learner scaling laws
├── data/                          # Data directory (see Data section)
├── experiments/                   # Output directory for results
├── requirements.txt
└── README.md
```

## Core Contributions

### 1. Adaptive Inference Strategies (Section 4)

The key contribution is two oracle strategies for selecting which tokens to unmask:

**Top Probability** (`top_prob`):
```
certainty(i) = max_j p_theta(x^i = j | x_t)
```
Select K positions with highest certainty.

**Top Probability Margin** (`top_prob_margin`):
```
certainty(i) = |p_theta(x^i = j1 | x_t) - p_theta(x^i = j2 | x_t)|
```
where j1, j2 are the top-2 most probable values. This is more robust when multiple values have similar high probabilities (e.g., in Sudoku).

### 2. L&O-NAE-SAT Distribution (Section 3.3)

A synthetic distribution with:
- N latent tokens sampled uniformly from {1, ..., m}
- P observation tokens determined by NAE(x_i1, x_i2, x_i3) for random triples

This demonstrates the error imbalance: MDMs perform well on observation positions but struggle with latent positions.

### 3. Pi-Learner Scaling Laws (Section 3.2)

A pi-learner predicts tokens in order defined by permutation π. The MDM loss equals the average pi-learner loss over all permutations. As π deviates from identity (left-to-right), performance degrades, showing MDMs train on harder subproblems.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### L&O-NAE-SAT Experiments (Table 1)

```bash
cd src

# Train and evaluate for each (N, P) pair
python train_mdm_lo_nae_sat.py --N 25 --P 275 --n_epochs 2000
python train_mdm_lo_nae_sat.py --N 50 --P 250 --n_epochs 2000
python train_mdm_lo_nae_sat.py --N 100 --P 200 --n_epochs 2000
```

### Sudoku Experiments (Table 2)

First, download the dataset from Shah et al. (2024):
```bash
# Dataset from: https://github.com/kulinshah98/llm-reasoning-logic-puzzles
# Place in data/sudoku/
```

Then train and evaluate:
```bash
cd src
python train_mdm_sudoku.py --data_path ../data/sudoku/train.csv
```

For testing without real data:
```bash
python train_mdm_sudoku.py --use_synthetic
```

### LLaDA 8B Adaptive Inference (Table 4)

```bash
cd src
python llada_adaptive_inference.py --task humaneval --strategy top_prob_margin
python llada_adaptive_inference.py --task math --strategy top_prob_margin
```

### Error Imbalance Analysis (Figure 2, right)

```bash
cd src
python error_imbalance_analysis.py --model_path /path/to/model.pt --N 20 --P 280
```

## Data

### Sudoku Dataset
- Training/test data: From Shah et al. (2024) - https://github.com/kulinshah98/llm-reasoning-logic-puzzles
- Hard test data: From Radcliffe (2020) - https://www.kaggle.com/dsv/1495975

### Zebra Dataset
- From Shah et al. (2024) - https://github.com/kulinshah98/llm-reasoning-logic-puzzles

### Text Data (Pi-learner experiments)
- SlimPajama dataset: https://huggingface.co/datasets/cerebras/SlimPajama-627B

## Implementation Details

### MDM Training
The MDM training objective (Proposition 2.1) is implemented as:
1. For each sequence, randomly mask tokens with probability sampled uniformly from [0, 1]
2. Compute cross-entropy loss on masked positions only
3. This is equivalent to the any-order autoregressive loss averaged over all permutations

### Adaptive Inference
The adaptive inference (Section 4.1) works as follows:
1. Start from fully masked sequence (or puzzle with empty cells)
2. At each step, get model predictions for all masked positions
3. Select K positions to unmask using the oracle (top_prob or top_prob_margin)
4. Assign tokens to selected positions (argmax or sampling)
5. Repeat until all positions are unmasked

The number K of tokens to unmask at each step follows the MDM reverse process:
```
K = n_masked * (alpha_s - alpha_t) / (1 - alpha_t)
```

### Gumbel Noise
For diversity in text generation, Gumbel noise is added to the oracle scores:
```
score(i) = margin(i) + gumbel_noise * Gumbel(0, 1)
```
This is used with coefficient 0.5 for logic puzzles (as per Appendix D.2).

## Assumptions and Unresolved Details

1. **Sudoku/Zebra dataset format**: The paper uses the dataset from Shah et al. (2024). We implement a CSV loader that expects columns [puzzle, solution] with digit strings.

2. **LLaDA model**: The paper uses LLaDA-8B-Instruct from Nie et al. (2025). Our implementation wraps the HuggingFace model with adaptive inference.

3. **Pi-learner experiments**: The full scaling law experiments require significant compute (training multiple models at different scales on SlimPajama). We implement the infrastructure but cannot run the full experiments.

4. **ARM baseline**: The paper compares against ARMs trained with and without ordering information (Shah et al., 2024). We implement the MDM side; the ARM baseline requires the Shah et al. codebase.

5. **Inference step count**: The paper uses 50 reverse sampling steps for all experiments.

6. **Gumbel noise coefficient**: 0.5 for logic puzzles (Appendix D.2), 0.0 for L&O-NAE-SAT accuracy evaluation.

## Citation

```bibtex
@inproceedings{kim2025train,
  title={Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions},
  author={Kim, Jaeyeon and Shah, Kulin and Kontonis, Vasilis and Kakade, Sham and Chen, Sitan},
  booktitle={ICML},
  year={2025}
}
```
