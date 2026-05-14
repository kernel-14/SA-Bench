# Masked Diffusion Token Ordering

## Overview
This repository implements the methodology and experiments described in "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions".

Masked Diffusion Models (MDMs) are generative models used for discrete data domains. They offer order-agnostic training and flexible inference compared to Autoregressive Models (ARMs). This repository reproduces the core contributions of the paper:

1. **Forward/Reverse Processes** of MDM
2. **Vanilla Inference**: Token unmasking randomly.
3. **Adaptive Inference Strategies**:
   - Top Probability
   - Top Probability Margin
4. Empirical validations for generative perplexity, Sudoku, and Zebra puzzles.

## Directory Structure

- **src/**: Contains implementation of Masked Diffusion Models ().
- **data/**: Placeholder for datasets (Sudoku/Zebra puzzles).
- **experiments/**: Code for executing experiments and storing results.
- **images/**: Figures generated during experiments.

## Implementation Highlights

The key components include:
- 
   - Defines the MDM framework, forward/reverse processes, inference methods, and placeholders for adaptive strategies.

## Future Steps

1. Implement adaptive oracles for inference.
2. Add puzzle-solving experiments (Sudoku/Zebra).
3. Evaluate model generative perplexity.

## Assumptions
- Pretrained models assumed available for generative perplexity tasks.
- Requires datasets aligned with paper specifications.

