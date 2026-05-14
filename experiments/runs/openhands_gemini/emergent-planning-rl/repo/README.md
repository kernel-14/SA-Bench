# Emergent Planning in Model-Free Reinforcement Learning

This repository contains a reproduction of the paper "Interpreting Emergent Planning in Model-Free Reinforcement Learning" by Bush et al. (2025). The paper investigates whether Deep Repeated ConvLSTM (DRC) agents, a type of model-free reinforcement learning agent, learn to perform decision-time planning in the Sokoban environment.

## Codebase Structure

- `config.py`: Contains all hyperparameters and configuration settings for training, probing, and interventions.
- `data.py`: Handles dataset generation, loading, and preprocessing for Sokoban environments and probe training data.
- `layers.py`: Defines fundamental neural network layers like ConvLSTM, and residual blocks.
- `model.py`: Implements the Deep Repeated ConvLSTM (DRC) agent architecture.
- `agent.py`: Encapsulates the DRC agent's forward pass, action selection, and internal state management.
- `probes.py`: Implements linear probe models used for concept-based interpretability.
- `train.py`: Contains the training loop for the DRC agent using the IMPALA algorithm.
- `evaluate.py`: Implements evaluation logic for agent performance, probing, and interventions.
- `requirements.txt`: Lists all Python dependencies.
