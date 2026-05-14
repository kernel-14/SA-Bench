# Hi-MAR Codebase

This repository contains the implementation of the Hierarchical Masked Autoregressive Model (Hi-MAR) as described in the paper. Hi-MAR introduces a hierarchical autoregressive modeling approach for image generation, leveraging low-resolution tokens as pivots to enhance global context understanding.

## Structure

- `model.py`: Main Hi-MAR model architecture.
- `modules.py`: Reusable modules like scale-aware transformer blocks.
- `layers.py`: Low-level neural network layers.
- `train.py`: Training loop and hyperparameters.
- `data.py`: Dataset loading and preprocessing.
- `config.py`: Configuration and hyperparameters.
- `requirements.txt`: Dependencies.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up the dataset path in `config.py`.

3. Run training:
   ```bash
   python train.py
   ```

## Citation

If you use this code, please cite the original paper.