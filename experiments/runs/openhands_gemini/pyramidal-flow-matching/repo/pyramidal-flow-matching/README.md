
# Pyramidal Flow Matching for Efficient Video Generative Modeling

This repository contains a faithful reproduction of the paper "PYRAMIDAL FLOW MATCHING FOR EFFICIENT VIDEO GENERATIVE MODELING" by Yang Jin et al. The core idea is to introduce a unified pyramidal flow matching algorithm that reinterprets the original denoising trajectory as a series of pyramid stages, enabling more efficient video generative modeling.

## Project Structure

The codebase is organized as follows:

- `config.py`: Contains all hyperparameters and configuration settings for training and model architecture.
- `data.py`: Handles dataset loading, preprocessing, and augmentation.
- `layers.py`: Defines fundamental neural network layers used in the model.
- `modules.py`: Implements higher-level neural network modules, composing layers from `layers.py`.
- `model.py`: Defines the overall Pyramidal Flow Matching model architecture, including the Diffusion Transformer (DiT) and the 3D VAE.
- `train.py`: Contains the main training loop, optimization, and evaluation logic.
- `requirements.txt`: Lists all Python dependencies required to run the codebase.
