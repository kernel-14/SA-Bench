# Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

Reproduction of Kim et al. (2025) — "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions."

## Overview

This codebase reproduces the core contributions of the paper:

1. **MDM training complexity analysis** — empirical evidence that MDMs train on harder subproblems than ARMs via π-learner scaling laws on text data.
2. **Error imbalance across masking problems** — experiments on L&O-NAE-SAT and text showing MDMs perform worse on latent-position predictions.
3. **Adaptive MDM inference** — two oracle strategies (Top Probability, Top Probability Margin) that dramatically improve MDM performance at inference time without retraining.
4. **Logic puzzle experiments** — Sudoku and Zebra puzzle solving, showing adaptive MDM outperforms ARMs trained with ordering supervision.
5. **LLaDA 8B evaluation** — adaptive inference on coding and math benchmarks.

## Repository Structure

```
repo/
├── config.py               # All hyperparameters and experiment configurations
├── model.py                # Transformer backbone (shared by MDM and ARM)
├── mdm.py                  # MDM model wrapper, loss, and noise schedule
├── arm.py                  # ARM model wrapper and loss
├── data.py                 # Datasets: NAE-SAT, Sudoku, Zebra, text (Slimpajama)
├── inference.py            # Vanilla and adaptive MDM inference (Top-P, Top-P-Margin)
├── train_mdm.py            # MDM training script
├── train_arm.py            # ARM training script (left-to-right and order-aware)
├── evaluate.py             # Evaluation: accuracy, generative perplexity, entropy
├── utils.py                # Shared utilities (logging, checkpointing, noise schedule)
├── experiments/
│   ├── run_nae_sat.py      # L&O-NAE-SAT: training + vanilla vs. adaptive inference
│   ├── run_sudoku.py       # Sudoku: MDM and ARM training + evaluation
│   ├── run_zebra.py        # Zebra puzzle: MDM and ARM training + evaluation
│   ├── run_scaling_laws.py # π-learner scaling law experiments on text
│   └── run_llada.py        # LLaDA 8B adaptive inference evaluation
└── requirements.txt
```

## Key Algorithms

### MDM Training Loss (Eq. 1 in paper)
The MDM loss is equivalent to the average any-order autoregressive loss:
```
L_θ = -E_{π~Unif(S_L)} [ Σ_i log p_θ(x_0^{π(i)} | x_0[π{i,...,L-1}]) ]
```
In practice, implemented by randomly masking tokens and predicting them with cross-entropy.

### Adaptive MDM Inference
Instead of randomly selecting which tokens to unmask at each step, use an oracle F(θ, x_t):

- **Top Probability**: unmask the K positions with highest `max_j p_θ(x^i = j | x_t)`
- **Top Probability Margin**: unmask the K positions with highest `|p_θ(x^i=j1|x_t) - p_θ(x^i=j2|x_t)|`

### L&O-NAE-SAT Distribution
- N latent tokens sampled from uniform prior over {1,...,m}
- P observation tokens: `NAE(x_{i1}, x_{i2}, x_{i3}) = 1 - 1[x_{i1}=x_{i2}=x_{i3}]`
- Triples (i1,i2,i3) randomly chosen from [N]

## Experiments

### Sudoku (Table 2)
| Method | #Param | Accuracy |
|--------|--------|----------|
| ARM (w/o ordering) | 42M | 9.73% |
| ARM (with ordering) | 42M | 87.18% |
| MDM (vanilla) | 6M | 6.88% |
| MDM (Top probability) | 6M | 18.51% |
| MDM (Top prob. margin) | 6M | 89.49% |

### Zebra Puzzles (Table 3)
| Method | #Param | Accuracy |
|--------|--------|----------|
| ARM (w/o ordering) | 42M | 80.31% |
| ARM (with ordering) | 42M | 91.17% |
| MDM (vanilla) | 19M | 76.9% |
| MDM (Top probability) | 19M | 98.5% |
| MDM (Top prob. margin) | 19M | 98.3% |

## Usage

### Train MDM on Sudoku
```bash
python train_mdm.py --task sudoku --model_size 6M --epochs 300 --lr 0.001 --batch_size 128
```

### Train ARM on Sudoku (with ordering)
```bash
python train_arm.py --task sudoku --model_size 42M --epochs 300 --lr 0.001 --batch_size 128 --use_ordering
```

### Evaluate MDM with adaptive inference
```bash
python evaluate.py --task sudoku --checkpoint path/to/mdm.pt --inference_strategy top_prob_margin --num_steps 50
```

### Run L&O-NAE-SAT experiment
```bash
python experiments/run_nae_sat.py --N 25 --P 275
```

### Run scaling law experiment
```bash
python experiments/run_scaling_laws.py --permutation_type random
```
