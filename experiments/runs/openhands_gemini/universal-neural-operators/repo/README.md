
# Universal Neural Operators through Multiphysics Pretraining

This repository contains a faithful reproduction of the paper "Towards Universal Neural Operators through Multiphysics Pretraining".

## Project Structure

- `config.py`: Configuration file for all hyperparameters and settings.
- `data.py`: Handles dataset loading, preprocessing, and augmentation for various PDE problems.
- `layers.py`: Implements fundamental neural network layers.
- `modules.py`: Contains implementations of specialized modules like FNO blocks, Mamba SSM, and Perceiver IO components.
- `model.py`: Defines the main neural operator architectures, including FNO, MambaFNO, PerceiverIONO, and CoDANO, along with adapter mechanisms.
- `train.py`: Manages the training and evaluation loops, supporting both pre-training and fine-tuning stages.
- `requirements.txt`: Lists all necessary Python dependencies.

## Models Implemented

- **FNO (Fourier Neural Operator)**: Baseline model.
- **Mamba FNO**: FNO enhanced with a Mamba State Space Model module for improved dependency encoding.
- **Perceiver IO NO**: Neural Operator incorporating Perceiver IO blocks for efficient latent representation learning.
- **CoDA-NO (Codomain Attention Neural Operator)**: Neural Operator using codomain attention mechanisms.
- **Swin-v2**: (Will be adapted as an operator block)

## Training and Evaluation

The `train.py` script supports pre-training on diverse multiphysics datasets and fine-tuning on specific downstream tasks, as described in the paper. It tracks metrics such as Mean Squared Error (MSE) and Range-Normalized Mean Absolute Error (NMAE).

## Datasets

The implementation aims to support datasets for:
- Burgers' equation
- Gray-Scott model
- Navier-Stokes equations
- Heat equation
- Reaction-Diffusion equations

## Usage

(Further instructions on how to run pre-training and fine-tuning will be added here once the implementation is complete.)
