# LoRA-SB Implementation

This repository contains the implementation of LoRA-SB (Low-Rank Adapter Silver Bullet), a parameter-efficient fine-tuning method for large language models. LoRA-SB builds on LoRA-XS by introducing a learnable low-rank matrix `R` and an initialization strategy derived from the first step of full fine-tuning.

## Codebase Structure

- `model.py`: Contains the implementation of the LoRA-SB model and its components.
- `train.py`: Training loop with all hyperparameters and logic.
- `data.py`: Dataset loading and preprocessing.
- `config.py`: Configuration file with hyperparameters and settings.
- `requirements.txt`: List of dependencies.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train.py
   ```

## Dependencies

- PyTorch
- NumPy
- scikit-learn (optional, for additional preprocessing)