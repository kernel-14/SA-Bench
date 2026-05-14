# Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

**Paper Reproduction Repository**

This repository contains a reproduction of the core contributions from the paper:

> *"Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions"*
> Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham Kakade, Sitan Chen
> ICML 2025 (Outstanding Paper)

## Overview

This paper examines two competing factors in Masked Diffusion Models (MDMs):

1. **Training complexity**: MDMs must solve exponentially many infilling subproblems, some of which are computationally intractable
2. **Inference flexibility**: MDMs can decode tokens in arbitrary order, allowing adaptive strategies to sidestep hard subproblems

The key insight is that while MDMs train on both easy and hard masking problems, simple adaptive inference strategies (Top probability and Top probability margin) can dramatically improve performance by selecting which tokens to unmask based on the model's certainty.

## Repository Structure

```
.
├── models/
│   ├── mdm.py              # Masked Diffusion Model (Section 2)
│   └── arm.py              # Autoregressive Model baseline (Sections 2.1.1, 4.3)
│
├── data_generation/
│   ├── lo_distribution.py  # L&O distributions (Definition 3.1, Example 3.2)
│   └── logic_puzzles.py    # Sudoku & Zebra puzzle generation (Sections 4.3-4.5)
│
├── training/
│   └── train_mdm.py        # MDM & π-learner training (Section 3.2, Appendix C)
│
├── inference/
│   └── (future: advanced sampling strategies)
│
├── evaluation/
│   └── metrics.py          # GenPPL, entropy, accuracy, error imbalance (Sections 3.3, 4.2)
│
├── analysis/
│   └── belief_propagation.py  # BP for planted CSPs (Appendix B.4, Proposition 3.3)
│
├── utils/
│   └── permutations.py     # Permutation utilities for π-learners (Section 3.2)
│
├── experiments/
│   ├── run_scaling_laws.py        # Scaling law analysis (Section 3.2, Figure 2 left)
│   ├── run_error_imbalance.py     # Error imbalance experiment (Section 3.3, Figure 2 right)
│   ├── run_adaptive_inference.py  # Adaptive inference comparison (Section 4.2, Table 1, Figure 3)
│   └── run_logic_puzzles.py       # Sudoku & Zebra experiments (Sections 4.3-4.5, Tables 2-5)
│
└── configs/
    └── (configuration files)
```

## What Was Reproduced

### 1. Theoretical Framework (Section 3.1, Appendix B)
- **L&O Distribution** (Definition 3.1): Full implementation of Latents-and-Observations distributions with latent tokens, observation functions, and oracle predictor
- **Sparse Predicate Observations** (Example 3.2): NAE predicate observations over k-sized subsets
- **Belief Propagation** (Appendix B.4): BP update rules for planted CSPs, Kesten-Stigum threshold computation
- **Computational hardness analysis**: Framework for connecting masking problems to planted CSP hardness

### 2. Training Analysis (Section 3.2-3.3)
- **π-learner training**: Training with arbitrary token order permutations
- **Scaling law comparison**: IsoFLOP analysis for different permutations (identity, much-closer, closer, uniform)
- **Error imbalance measurement**: Per-task error analysis for L&O-NAE-SAT and text data
- **MDM loss reformulation** (Proposition 2.1): Equivalence to uniform average over all masks

### 3. Adaptive Inference (Section 4.1-4.2)
- **Top probability oracle**: Select positions with highest maximum probability
- **Top probability margin oracle**: Select positions with largest gap between top-2 probabilities
- **Vanilla vs. adaptive comparison**: For L&O-NAE-SAT and text data
- **Temperature variant** with Gaussian noise for text generation (Appendix D.1.2)

### 4. Logic Puzzles (Section 4.3-4.5)
- **Sudoku puzzle generation**: 9×9 grid with configurable difficulty
- **Zebra (Einstein) puzzle generation**: Logic grid puzzles with constraints
- **ARM with/without ordering**: Two ARM baselines for comparison
- **Easy-to-hard generalization**: Hard test set evaluation (Section 4.5)

### 5. Model Architectures
- **MDM Transformer**: Bidirectional transformer with learnable positional embeddings
- **ARM Transformer**: Causal transformer with learnable positional embeddings
- **Noise schedules**: Cosine, linear, and log-linear schedules
- **Training configuration**: AdamW optimizer, cosine LR schedule, gradient clipping

## Key Results (Expected from Paper)

| Experiment | Method | Result |
|-----------|--------|--------|
| Sudoku (Table 2) | MDM Vanilla | ~6.88% |
| Sudoku (Table 2) | MDM Top prob margin | ~89.49% |
| Sudoku (Table 2) | ARM with ordering (42M) | ~87.18% |
| Sudoku hard (Table 5) | MDM Top prob margin | ~49.88% |
| Sudoku hard (Table 5) | ARM with ordering (42M) | ~32.57% |
| Zebra (Table 3) | MDM Top probability | ~98.5% |
| Zebra (Table 3) | ARM with ordering | ~91.17% |
| L&O-NAE-SAT (Table 1) | Vanilla → Adaptive | ~78% → ~94% |

## Assumptions and Missing Details

1. **Pre-trained model weights**: The code provides the training infrastructure but not pre-trained weights, as this is a static reproduction.

2. **Slimpajama dataset**: For the text experiments (Section 3.2), the paper uses the Slimpajama dataset. Our code includes dataset loading utilities but users must obtain the dataset separately.

3. **LLaDA-8B experiments** (Section 4.4): The inference strategies are implemented, but the actual LLaDA-8B model weights are not included.

4. **LLaMA2-7B evaluation** (Figure 3): The generative perplexity evaluation framework is provided, but the LLaMA2 model must be downloaded separately.

5. **Hyperparameter details**: Some hyperparameters from the paper are used as defaults. For exact reproduction, some tuning may be needed.

6. **Belief Propagation thresholds**: The Kesten-Stigum and condensation thresholds are computed numerically; exact values may vary slightly from analytical results in the literature.

## Usage

### Install dependencies
```bash
pip install torch numpy
```

### Run scaling laws experiment
```bash
python experiments/run_scaling_laws.py --device cpu --L 128 --vocab_size 1000
```

### Run logic puzzles experiment
```bash
python experiments/run_logic_puzzles.py --device cpu --num_train 1000 --num_test 100
```

### Run adaptive inference experiment
```bash
python experiments/run_adaptive_inference.py --device cpu
```

### Run error imbalance experiment
```bash
python experiments/run_error_imbalance.py --device cpu --N 20 --P 280
```

## References

- Shah et al. (2024): "Causal language modeling can elicit search and reasoning capabilities on logic puzzles"
- Sahoo et al. (2025): "Simple and effective masked diffusion language models"
- Shi et al. (2024): "Simplified and generalized masked diffusion for discrete data"
- Zheng et al. (2024): "Masked diffusion models are secretly time-agnostic masked models"
- Nie et al. (2024): "Scaling up masked diffusion models on text"
- Krzakala & Zdeborová (2009): "Hiding quiet solutions in random constraint satisfaction problems"
- Alaoui & Gamarnik (2024): "Hardness of sampling solutions from the symmetric binary perceptron"

## License

This reproduction is provided for research purposes.
