
# MoE-POT: Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training

This repository contains a faithful reproduction of the MoE-POT model proposed in the paper "Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training" by Wang et al.

The codebase aims to implement the core contributions of the paper, including:
- The novel MoE-POT architecture with routed and shared experts.
- Input encoding and temporal aggregation mechanisms.
- Fourier Layers for efficient kernel-based integral transformations.
- Mixture of Experts Layer with dynamic expert selection and load balancing.
- Auto-regressive denoising pre-training strategy.
- Training and evaluation procedures as described in the paper.

## Directory Structure

- `config.py`: Configuration and hyperparameters for the model and training.
- `requirements.txt`: Python dependencies required to run the codebase.
- `train.py`: Script for pre-training and fine-tuning the MoE-POT model.
- `evaluate.py`: Script for evaluating the model performance.
- `data/`: Contains modules for dataset loading and preprocessing.
    - `dataset.py`: Defines PyTorch Dataset classes for PDE data.
    - `preprocessing.py`: Implements data preprocessing steps like padding, masking, and noise injection.
- `models/`: Contains the implementation of the MoE-POT model architecture.
    - `__init__.py`: Initializes the models package.
    - `moe_pot.py`: Main MoE-POT model definition.
    - `layers.py`: Defines custom layers used in MoE-POT (e.g., Fourier Layer, Patchification).
    - `experts.py`: Defines the expert networks and router-gating network.
- `utils/`: Utility functions and helper classes.
    - `__init__.py`: Initializes the utils package.
    - `losses.py`: Implements custom loss functions, including the load balancing loss.
