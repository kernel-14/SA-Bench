
# Robotic World Model (RWM) Reproduction

This repository contains a reproduction of the "Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics" paper.

The codebase is structured as follows:

- `config.py`: Contains all hyperparameters and configuration settings for the RWM and MBPO-PPO training.
- `data.py`: Handles dataset loading and preprocessing, including data augmentation and sequence generation for autoregressive training.
- `layers.py`: Defines custom neural network layers used in the RWM and policy networks.
- `modules.py`: Implements the core building blocks and specialized modules for the RWM (GRU-based) and policy networks.
- `model.py`: Defines the complete RWM and policy network architectures, integrating components from `layers.py` and `modules.py`.
- `train.py`: Contains the main training loops for both the RWM (self-supervised autoregressive training) and the MBPO-PPO policy optimization.
- `requirements.txt`: Lists all necessary Python dependencies.
