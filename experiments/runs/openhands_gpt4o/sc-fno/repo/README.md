# SC-FNO Implementation

This repository contains the implementation of the Sensitivity-Constrained Fourier Neural Operator (SC-FNO) as described in the paper "Sensitivity-Constrained Fourier Neural Operators for Forward and Inverse Problems in Parametric Differential Equations".

## Codebase Structure

- `model.py`: Contains the SC-FNO model architecture and Fourier layers.
- `modules.py`: Additional modules for extended functionality (currently empty).
- `layers.py`: Custom layers for the neural network (currently empty).
- `train.py`: Training loop implementation with loss functions and hyperparameters.
- `data.py`: Dataset loading and preprocessing.
- `config.yaml`: Configuration file for hyperparameters and settings.
- `requirements.txt`: Dependencies required to run the code.

## How to Use

1. Install dependencies using `pip install -r requirements.txt`.
2. Configure hyperparameters in `config.yaml`.
3. Run `train.py` to train the SC-FNO model.

## Dependencies

See `requirements.txt` for a list of required Python packages.

## Contact

For questions or issues, please contact the authors of the paper.