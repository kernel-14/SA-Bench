# Train for the Worst, Plan for the Best

Reproduction code for "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions" by Kim et al.

## Structure

- `config.py` — All hyperparameters and model configurations matching the paper
- `models.py` — Transformer (GPT-2 style) model with configurable attention
- `diffusion.py` — Masked diffusion forward/reverse processes and score-entropy loss
- `inference.py` — Vanilla, top-probability, and top-probability-margin samplers
- `data.py` — Datasets: L&O-NAE-SAT, Sudoku, Zebra, Text (SlimPajama)
- `train.py` — Training loops for MDM, π-learner, ARM
- `evaluate.py` — Full evaluation suite (accuracy, perplexity, task imbalance)
- `main.py` — CLI entry point with preset configurations

## Quick Start

```bash
# Train MDM on L&O-NAE-SAT
python main.py --preset lonaesat_19m --mode train

# Train MDM on Sudoku
python main.py --preset sudoku_6m --mode train

# Evaluate with different inference strategies
python main.py --preset sudoku_6m --mode eval --strategy top_probability_margin

# Train ARM with ordering on Sudoku
python main.py --preset arm_42m_sudoku --mode train_arm --order_info

# Train π-learner on text
python main.py --preset text_pi_learner --mode train_pi
```

## Paper Experiments

| Experiment | Section | Command |
|---|---|---|
| L&O-NAE-SAT Accuracy | 4.2, Table 1 | `--preset lonaesat_19m` |
| Sudoku Accuracy | 4.3, Table 2 | `--preset sudoku_6m` |
| Zebra Accuracy | 4.3, Table 3 | `--preset zebra_19m` |
| Hard Sudoku Generalization | 4.5, Table 5 | `--preset sudoku_6m --hard` |
| Text Generative Perplexity | 4.2, Fig 3 | `--preset text_170m` |
| π-learner Scaling Laws | 3.2, Fig 2 | `--preset text_pi_learner` |
| ARM w/ & w/o ordering | 4.3, Tables 2,3 | `--preset arm_42m_sudoku` |

## Requirements

Python 3.10+, PyTorch 2.0+, tqdm, numpy, PyYAML, datasets, transformers, wandb
