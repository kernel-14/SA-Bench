
# NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints

This repository contains a reproduction of the "NaViL" paper, focusing on the core contributions as described in the paper. The implementation aims to be faithful to the architectural designs, algorithms, loss functions, and training configurations presented.

## Codebase Structure:

- `config.py`: Contains all hyperparameters and configuration settings for the model and training.
- `data.py`: Handles dataset loading and preprocessing.
- `layers.py`: Defines fundamental neural network layers used in the model.
- `modules.py`: Implements higher-level modules, such as the visual encoder and MoE components.
- `model.py`: Integrates the modules and layers to form the complete NaViL model architecture.
- `train.py`: Contains the main training loop, including loss functions, optimizers, and evaluation logic.
- `requirements.txt`: Lists all Python dependencies required to run the codebase.
