# MR.Q Implementation

This repository contains the implementation of the MR.Q algorithm as described in the paper "Towards General-Purpose Model-Free Reinforcement Learning".

## Codebase Structure

- `model.py`: Contains the neural network components including the state encoder, state-action encoder, value network, and policy network.
- `train.py`: Implements the training loop with all hyperparameters and periodic updates.
- `data.py`: Handles dataset loading and preprocessing for environments like Gym, DMC, and Atari.
- `config.py`: Defines hyperparameters and environment settings.
- `requirements.txt`: Lists all dependencies required to run the code.

## Installation

1. Clone the repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Configure hyperparameters in `config.py`.
2. Run the training script:
   ```bash
   python train.py
   ```

## Notes

- Ensure the environment specified in `config.py` is installed and accessible.
- Preprocessing steps are tailored for Gym, DMC, and Atari environments.

## License

This project is licensed under the MIT License.