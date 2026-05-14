# MoE-POT: Mixture-of-Experts Operator Transformer

This repository contains an implementation of the MoE-POT model, as described in the paper "Mixture-of-Experts Operator Transformer for Large-Scale PDE Pre-Training."

## Repository Structure

- **src/**: Source code for the implementation.
  - : Data loading and preprocessing for PDE datasets.
  - : Implementation of the Fourier Layer and Mixture-of-Experts Layer.
  - : Pipeline for training the MoE-POT model.
- **configs/**: Configuration files for experiments.
- **experiments/**: Directory to store experimental results and custom scripts.

## Assumptions and Simplifications

1. **Generalization**: The Fourier Layer and Mixture-of-Experts Layer are implemented based on the descriptions provided in the paper, focusing on core functionality.
2. **Dataset Simulation**: The code assumes raw PDE data to be available in  format.
3. **Preprocessing**: The  function provides a placeholder for normalization and masking.

## How to Use

1. **Install Dependencies**:
   Ensure PyTorch and NumPy are installed before running the scripts.

2. **Prepare Datasets**:
   Place your  datasets in the project directory and update the paths in .

3. **Train the Model**:
   Run the training pipeline directly:
   

## Limitations

1. Interpretability experiments and advanced evaluations may require additional implementations.
2. Exact dataset preprocessing and configurations from the paper are not fully detailed.

## Future Work
- Extend the model to support irregular point clouds and more advanced normalization techniques.
- Incorporate evaluation metrics for zero-shot learning.
- Add interpretability analysis for the router-gating network. 

