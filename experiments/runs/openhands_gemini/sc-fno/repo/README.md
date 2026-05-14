# SC-FNO: Sensitivity-Constrained Fourier Neural Operators

This repository contains a faithful reproduction of the core contributions of the paper "SENSITIVITY-CONSTRAINED FOURIER NEURAL OPERATORS FOR FORWARD AND INVERSE PROBLEMS IN PARAMETRIC DIFFERENTIAL EQUATIONS".

The codebase is structured as follows:

- `config.py`: Configuration and hyperparameters for models and training.
- `data.py`: Dataset generation and loading utilities for various ODEs and PDEs.
- `layers.py`: Implementation of custom neural network layers, such as Fourier layers.
- `models.py`: Definitions of the FNO and SC-FNO models.
- `train.py`: Training loop for FNO, SC-FNO, FNO-PINN, and SC-FNO-PINN.
- `utils.py`: Utility functions for metrics, plotting, and general helpers.
- `differential_equations/`: Contains specific implementations for each ODE/PDE, including analytical solutions and their sensitivities.

