
# NGPT: NORMALIZED TRANSFORMER WITH REPRE-SENTATION LEARNING ON THE HYPERSPHERE

This repository contains a reproduction of the "NGPT: Normalized Transformer with Representation Learning on the Hypersphere" paper.

## Overview

The paper proposes a novel neural network architecture, the Normalized Transformer (nGPT), which normalizes all vectors forming the embeddings, MLP, attention matrices, and hidden states to unit norm. This approach aims to improve training stability and significantly accelerate convergence.

This implementation provides both the baseline Transformer (GPT) and the Normalized Transformer (nGPT) architectures, allowing for direct comparison of their training dynamics and performance.

## Project Structure

- `config.py`:
    Contains the `Config` class to manage hyperparameters for both GPT and nGPT models, including model dimensions, optimization parameters, and training settings.

- `data.py`:
    Handles dataset loading and preprocessing. It utilizes the `OpenWebText` dataset and `LLaMA-2` tokenizer, processing data into fixed-size blocks suitable for language model training.

- `ngpt_model/layers.py`:
    Defines foundational layers used in both Transformer variants, such as `Norm` (unit norm normalization), `RMSNorm` (Root Mean Square Layer Normalization for baseline GPT), and `RotaryPositionEmbedding` (RoPE).

- `ngpt_model/modules.py`:
    Implements key building blocks of the Transformer architecture, including `MultiHeadSelfAttention` and `MLP` (Multi-Layer Perceptron). These modules are designed to support both baseline and nGPT-specific modifications.

- `ngpt_model/model.py`:
    Integrates the layers and modules to construct the full `GPT` (which can be configured as either baseline or nGPT) model. It includes the `TransformerBlock` and handles embedding, normalization, and output logits.

- `train.py`:
    Contains the main training loop. It sets up the model, optimizer (AdamW for GPT, Adam for nGPT), learning rate schedule (Cosine Annealing), and orchestrates the training process, including periodic evaluation.

- `requirements.txt`:
    Lists all necessary Python dependencies to run the codebase.

## Key Features Implemented

- **Normalized Transformer (nGPT) Architecture**:
    - Unit norm normalization (`Norm`) applied to all relevant matrices and embeddings.
    - Modified update equations for attention and MLP blocks using learnable eigen learning rates (`alpha_A`, `alpha_M`).
    - Adjusted softmax scaling factor in attention.
    - Scaled intermediate states in the MLP block (`s_u`, `s_v`).
    - Scaled logits (`s_z`).
    - Removal of RMSNorm layers.
    - Adam optimizer with zero weight decay.
    - Custom weight initialization for nGPT.

- **Baseline Transformer (GPT) Architecture**:
    - RMSNorm layers before attention and MLP.
    - Standard Transformer update equations.
    - AdamW optimizer with weight decay.
    - Standard weight initialization.

- **Rotary Position Embeddings (RoPE)**: Incorporated into the attention mechanism for both models.
- **SwiGLU Activation**: Used in the MLP blocks.
- **OpenWebText Dataset**: Used for training, tokenized with LLaMA-2 tokenizer.
- **Cosine Annealing Learning Rate Schedule**.

## Usage

To run the training, ensure you have the dependencies installed:
```bash
pip install -r requirements.txt
```

Then, execute `train.py`:
```bash
python train.py
```
This script will train both a baseline GPT and an nGPT model for comparison, as configured in `config.py`.
