# Robotic World Model Implementation

This repository reproduces the core contributions of the paper 'Robotic World Model: A Neural Network Simulator for Robust Policy Optimization'.

## Key Components

1. **World Model (models/world_model.py)**:
   - Implements the GRU-based dual-autoregressive architecture described in the paper.
   - Predicts Gaussian parameters for next observations based on historical contexts and autoregressively utilizes predictions for subsequent steps.
   
2. **Autoregressive Training Framework (train/train_world_model.py)**:
   - Implements the sliding window approach and minimizes multi-step prediction error.
   - Facilitates capturing long-term dependencies and unobservable dynamics through autoregressive training.
   
3. **Policy Optimization Framework (scripts/policy_optimizer.py)**:
   - Combines the trained world model with Proximal Policy Optimization (PPO).
   - Performs simulated rollouts in imagined environments and optimizes the robot's policy for real-world deployment.

## Files Overview

- : GRU-based world model implementation.
- : Training script for the world model.
- : Policy training script using MBPO-PPO framework.

## Installation and Usage

1. Clone the repository.
2. Ensure Python is installed and includes necessary libraries such as PyTorch.
3. Train the world model using the commands for .
4. Use  to train robotic actions conditioned on world model predictions.

## Limitations

This reproduction uses placeholders for specific datasets, loss functions, and replay buffer data structures. These require real-world or simulation-specific implementations to demonstrate the full pipeline.
