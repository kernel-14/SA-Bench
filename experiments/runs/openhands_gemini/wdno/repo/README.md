
# Wavelet Diffusion Neural Operator (WDNO)

This repository contains a reproduction of the Wavelet Diffusion Neural Operator (WDNO) paper.
WDNO is a novel PDE simulation and control framework that enhances the handling of complexities in physical systems, particularly those with abrupt changes and issues of generalization to higher resolutions.

## Project Structure

- `config.py`: Configuration and hyperparameters for models and training.
- `data.py`: Handles data loading, preprocessing, and wavelet transforms for various PDE datasets.
- `layers.py`: Defines custom neural network layers used in the model.
- `modules.py`: Contains modular components that build up the main neural network architectures.
- `model.py`: Implements the core WDNO model, including the Base-Resolution Model (BRM) and Super-Resolution Model (SRM).
- `train.py`: Script for training and evaluating the WDNO model.
- `requirements.txt`: Python dependencies.

## Key Innovations

1. **Generation in the Wavelet Domain**: WDNO performs diffusion-based generative modeling in the wavelet domain for entire trajectories, effectively handling abrupt changes and long-term dependencies.
2. **Multi-resolution Training**: To address poor generalization across different resolutions, WDNO introduces multi-resolution training, enabling generalization to finer resolutions not seen during training.

