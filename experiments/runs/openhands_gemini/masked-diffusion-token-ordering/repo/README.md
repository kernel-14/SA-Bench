# Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

This repository contains a faithful reproduction of the paper "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions."

## Codebase Structure:

- `config.py`: Contains all hyperparameters and configuration settings for models and training.
- `model.py`: Defines the Masked Diffusion Model (MDM) architecture, including the denoising network.
- `modules.py`: Contains reusable neural network modules and layers used in `model.py`.
- `layers.py`: Contains custom layers, such as positional embeddings.
- `data.py`: Handles dataset loading, preprocessing, and batching.
- `train.py`: Implements the training loop, loss function, and optimization.
- `inference.py`: Implements vanilla and adaptive MDM inference strategies (Top probability, Top probability margin).
- `requirements.txt`: Lists all necessary Python dependencies.
