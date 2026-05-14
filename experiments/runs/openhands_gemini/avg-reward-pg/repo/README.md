
# Global Convergence of Policy Gradient in Average Reward MDPs

This repository contains a Python implementation to reproduce the simulations from the paper "Global Convergence of Policy Gradient in Average Reward MDPs".

## Codebase Structure

- `config.py`: Defines configuration classes for MDPs, Policy Gradient, and simulations, along with specific configurations for reproducing the experiments in the paper.
- `model.py`: Implements the core MDP logic and the Policy Gradient algorithm, including methods for calculating average reward, value functions, Q-functions, and policy updates.
- `data.py`: Contains functions to generate specific MDP instances (transition kernels and reward functions) used in the paper's simulation sections (4.1, 4.2, 4.3).
- `train.py`: Orchestrates the simulation runs, trains the Policy Gradient agent on various MDPs, and plots the convergence of average reward over iterations.
- `requirements.txt`: Lists the necessary Python dependencies for the project.
