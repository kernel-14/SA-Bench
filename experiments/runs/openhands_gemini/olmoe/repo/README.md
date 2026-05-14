
# OLMoE: Open Mixture-of-Experts Language Models

This repository contains a faithful reproduction of the OLMoE paper, implementing the core model architecture, training configuration, and data processing pipeline as described in the paper.

## Project Structure

- `config.py`: Defines all hyperparameters and configuration settings for the model and training.
- `layers.py`: Contains implementations of fundamental neural network layers like attention, feed-forward networks, RMSNorm, and QK-Norm.
- `modules.py`: Implements Mixture-of-Experts specific components, including the router, individual experts, and the MoE layer.
- `model.py`: Assembles the layers and modules into the complete OLMoE model architecture.
- `data.py`: Handles dataset loading and preprocessing, including data mixing and filtering as described.
- `train.py`: Contains the main training loop, including loss functions (cross-entropy, load balancing, router z-loss), optimizer, and FSDP setup.
- `requirements.txt`: Lists all Python dependencies required to run the code.

## Reproduction Details

The implementation strictly adheres to the specifications provided in the paper, including:
- OLMoE-1B-7B architecture: 1.3B active parameters out of 6.9B total, 64 small experts with 8 activated per layer.
- Training objective: Cross-entropy loss with auxiliary load balancing loss (weight 0.01) and router Z-loss (weight 0.001).
- Hyperparameters: All major hyperparameters are configured as per Appendix B and Table 1 of the paper.
- Initialization: Truncated normal initialization with specific std and cut-off values.
- Normalization: RMSNorm and QK-Norm applied as described.
- Dataset: OLMOE-MIX composition as detailed in Table 2, with specified filtering.

## Setup and Usage

(Instructions for setting up the environment and running the code will be added here.)
