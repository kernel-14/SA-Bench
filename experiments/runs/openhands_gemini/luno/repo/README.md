
# LUNO: Linearization Turns Neural Operators into Function-Valued Gaussian Processes

This repository contains a faithful reproduction of the core contributions of the paper "Linearization Turns Neural Operators into Function-Valued Gaussian Processes" by Magnani et al. (2024).

The codebase is structured as follows:

- `luno/config.py`: Contains all hyperparameters and configuration settings for models, training, and data.
- `luno/data/pde_datasets.py`: Handles dataset loading and preprocessing for various PDE datasets (Burgers', Hyper Diffusion, Kuramoto-Sivashinsky, Advection-Diffusion).
- `luno/models/fno.py`: Implements the Fourier Neural Operator (FNO) architecture.
- `luno/models/luno_fno.py`: Implements the LUNO framework for FNOs, including last-layer Laplace approximation and probabilistic currying.
- `luno/train.py`: Contains the training loop for the FNO model.
- `luno/evaluate.py`: Implements evaluation metrics and procedures for uncertainty quantification methods.
- `luno/utils/metrics.py`: Defines various evaluation metrics like RMSE, NLL, and Chi-squared.
- `luno/requirements.txt`: Lists all Python dependencies.
- `luno/main.py`: Entry point for running experiments.
