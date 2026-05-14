# Masked Diffusion Token Ordering

This repository contains the implementation of the paper "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions." The codebase includes the model architecture, training loop, dataset handling, and adaptive inference strategies for Masked Diffusion Models (MDMs).

## Codebase Structure

- `model.py`: Defines the Masked Diffusion Model architecture.
- `modules.py`: Contains the Denoising Network and other modules used in the model.
- `layers.py`: Implements custom layers used in the model.
- `train.py`: Training loop for the Masked Diffusion Model.
- `data.py`: Handles dataset loading and preprocessing.
- `config.py`: Configuration file containing hyperparameters and paths.
- `requirements.txt`: Lists all dependencies required for the project.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the dataset and save it to the path specified in `config.py`.

3. Train the model:
   ```bash
   python train.py
   ```

## Dependencies

See `requirements.txt` for the list of dependencies.