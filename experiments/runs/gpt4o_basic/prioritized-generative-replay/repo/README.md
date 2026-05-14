# Prioritized Generative Replay

This repository contains an implementation of the Prioritized Generative Replay (PGR) framework, as described in the paper 'Prioritized Generative Replay'.

## Core Components

### PGR Framework
Located in , this script defines:
- Generative replay buffer mechanism
- Conditional diffusion model
- Synthetic data generation guided by relevance functions

### Relevance Functions
Implemented in , this includes:
- Curiosity-based relevance
- Return-based relevance
- Temporal Difference (TD) error-based relevance

### Policy Training
 handles:
- Policy optimization using mixed real and synthetic transitions
- Full integration with Prioritized Generative Replay

### Experiments

The  script provides:
- Examples showing how to run tasks with different relevance functions
- Integration with real and synthetic replay buffer mixing

## Instructions

1. Define or load an RL policy and its optimizer.
2. Use the  class for training.
3. Call different relevance functions (e.g., curiosity_relevance) with suitable guidance scales to evaluate performance.

## Future Work
The repository closely follows the methodology described in the paper but may not reproduce all experimental setups due to time constraints—additional scripts for scaling and pixel-based environments are required.
