
# MR.Q - Towards General-Purpose Model-Free Reinforcement Learning

This repository contains a faithful reproduction of the MR.Q algorithm from the paper "TOWARDS GENERAL-PURPOSE MODEL-FREE REINFORCEMENT LEARNING" by Scott Fujimoto et al.

The codebase aims to implement the core contributions of the paper, including:
- **Model Architecture**: State Encoder, State-Action Encoder, Value Networks, Policy Network.
- **Loss Functions**: Encoder loss (Reward, Dynamics, Terminal), Value loss, Policy loss.
- **Training Procedure**: Multi-step returns, target network updates, reward scaling, prioritized experience replay.
- **Configuration**: All hyperparameters are defined in `config.py`.

## Files Structure:
- `config.py`: Contains all hyperparameters and configuration settings.
- `data.py`: Handles environment interaction, data sampling, and preprocessing.
- `layers.py`: Defines custom neural network layers.
- `modules.py`: Implements reusable neural network blocks.
- `model.py`: Defines the complete MR.Q neural network architecture.
- `train.py`: Contains the main training loop and evaluation logic.
- `requirements.txt`: Lists all Python dependencies.
