"""Conformal Prediction as Bayesian Quadrature."""

from .bayesian_quadrature import (
    compute_conformal_risk_control_lambda,
    compute_split_conformal_lambda,
    compute_hpd_lambda,
    compute_L_plus_distribution,
    L_plus_random_variable,
)

__all__ = [
    "compute_conformal_risk_control_lambda",
    "compute_split_conformal_lambda",
    "compute_hpd_lambda",
    "compute_L_plus_distribution",
    "L_plus_random_variable",
]
