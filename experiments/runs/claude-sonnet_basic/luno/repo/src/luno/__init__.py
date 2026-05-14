"""
LUNO: Linearization Turns Neural Operators into Function-Valued Gaussian Processes.

Core framework for approximate Bayesian uncertainty quantification in trained neural operators.
"""
from .weight_space import IsotropicGaussian, LaplaceApproximation
from .luno import LUNO, LUNOPrediction
from .metrics import compute_rmse, compute_nll, compute_chi2
