# nGPT Implementation

This repository contains the implementation of the normalized Transformer (nGPT) as described in the paper "NGPT: Normalized Transformer with Representation Learning on the Hypersphere."

## Directory Structure

- `model.py`: Defines the main nGPT model architecture, including normalized embeddings, attention, and MLP blocks.
- `modules.py`: Contains reusable modules such as attention mechanisms and MLP layers.
- `layers.py`: Implements lower-level building blocks like normalization and linear layers.
- `train.py`: Implements the training loop, including eigen learning rates, normalization steps, and hyperparameter settings.
- `data.py`: Handles dataset loading and preprocessing, specifically for the OpenWebText dataset.
- `config.py`: Contains all hyperparameters and settings described in the paper.
- `requirements.txt`: Lists all dependencies required to run the codebase.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train.py
   ```