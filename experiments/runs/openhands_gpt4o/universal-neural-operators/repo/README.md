# Universal Neural Operators

This repository contains the implementation of the paper "Towards Universal Neural Operators through Multiphysics Pretraining." The codebase includes neural operator models, training scripts, and data handling for PDE-based problems.

## Codebase Structure

- `model.py`: Defines the main neural operator model.
- `modules.py`: Contains supporting modules such as FourierLayer, MambaSSM, and PerceiverIO.
- `train.py`: Training script with hyperparameters and training logic.
- `data.py`: Handles dataset loading and preprocessing.
- `config.py`: Configuration file with hyperparameters and paths.
- `requirements.txt`: Lists dependencies required for the project.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the dataset and place it in the `data/train` and `data/val` directories.

3. Run the training script:
   ```bash
   python train.py
   ```

## Citation

If you use this code, please cite the original paper.