
# Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model

This repository contains a faithful reproduction of the paper "Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model" by Chen and Deng.

## Project Structure

- `config.py`: Contains all hyperparameters and configuration settings for the models and training process.
- `data.py`: Handles dataset loading, preprocessing, and augmentation.
- `layers.py`: Implements various neural network layers used in the models, such as AdaLN-Zero, attention mechanisms, and SwiGLU.
- `modules.py`: Contains modular components like the P2VAE encoder/decoder, and the GRU for diffusion forcing.
- `model.py`: Defines the main model architectures, including the P2VAE (Pretrained Physics Variational Autoencoder) and FMT (Flow Marching Transformer).
- `train.py`: Implements the training loop, loss functions, and evaluation metrics.
- `requirements.txt`: Lists all Python dependencies required to run the code.

## Key Reproductions

- **P2VAE (Pretrained Physics Variational Autoencoder)**: Compresses high-resolution physical field snapshots into a compact latent grid.
- **FMT (Flow Marching Transformer)**: Implements the flow marching algorithm, bridging deterministic neural operators and stochastic flow matching.
- **Flow Marching Objective**: Implementation of the conditioned and preconditioned flow marching loss function.
- **Conditional Flow Marching with Diffusion Forcing**: Incorporates an RNN (GRU) to evolve latent states as conditions for the flow marching process.
- **Latent Temporal Pyramids**: Utilizes downsampling for efficiency in propagating PDE conditions.
- **Prediction and Generation Processes**: Implementation of Euler ODE sampler for propagation and handling of `k` parameter for deterministic vs. generative settings.
