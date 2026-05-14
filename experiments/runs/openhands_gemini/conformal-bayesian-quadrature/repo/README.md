# Conformal Prediction as Bayesian Quadrature

This repository contains a reproduction of the paper "Conformal Prediction as Bayesian Quadrature" by Jake C. Snell and Thomas L. Griffiths. The paper proposes a Bayesian framework for distribution-free uncertainty quantification, reinterpreting existing conformal prediction methods and offering a richer representation of predictive uncertainty.

## Project Structure

- `config.py`: Configuration file for hyperparameters and settings.
- `data.py`: Handles dataset loading, preprocessing, and generation (synthetic data).
- `methods.py`: Contains implementations of Conformal Risk Control (CRC), Risk-controlling Prediction Sets (RCPS), and our proposed Bayesian Quadrature (BQ) method.
- `experiments.py`: Defines the experimental setup for synthetic and real-world datasets, runs simulations, and collects results.
- `utils.py`: Utility functions, such as for calculating confidence intervals and handling Dirichlet distributions.
- `requirements.txt`: Python dependencies for the project.
