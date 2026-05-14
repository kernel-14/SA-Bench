import numpy as np

class Config:
    # General experiment settings
    num_trials: int = 10000  # M in the paper
    alpha: float = 0.4  # Target risk threshold for synthetic binomial, 0.1 for heteroskedastic/MS-COCO
    beta: float = 0.95 # Confidence level for HPD interval, 1 - target failure rate

    # Synthetic Binomial Data Experiment (Section 5.1)
    # L(z_i, lambda) = (1/K) * sum_{k=1 to K} 1{V_ik > lambda}
    # V_ik ~ Uniform(0,1)
    binomial_n: int = 10   # Number of calibration samples
    binomial_K: int = 4    # Number of Bernoulli trials per sample for binomial loss
    binomial_B: float = 1.0 # Maximum possible loss (B in paper)
    binomial_lambda_min: float = 0.0
    binomial_lambda_max: float = 1.0
    binomial_lambda_steps: int = 100 # Steps for lambda search

    # Synthetic Heteroskedastic Data Experiment (Section 5.2)
    hetero_n: int = 200    # Number of calibration samples
    hetero_alpha: float = 0.1 # Target miscoverage loss
    hetero_B: float = 1.0 # Max loss, e.g., 1 for binary miscoverage
    hetero_X_range: tuple = (0.0, 4.0) # X ~ U[0,4]
    hetero_mu: float = 0.0 # Y|X ~ N(0, X^2)
    hetero_sigma_multiplier: float = 1.0 # Used to scale X for sigma = X * sigma_multiplier
    hetero_lambda_min: float = 0.0
    hetero_lambda_max: float = 20.0 # Prediction interval [-lambda, lambda]
    hetero_lambda_steps: int = 200 # Steps for lambda search

    # MS-COCO Experiment (Section 5.3)
    coco_n: int = 1000 # Number of calibration examples
    coco_test_examples: int = 3952 # Number of test examples
    coco_alpha: float = 0.05 # Target false negative rate (5%)
    coco_B: float = 1.0 # Max loss (1 for FNR)

    # Bayesian Quadrature (BQ) Specific Settings
    dirichlet_samples: int = 100000 # Number of samples for Monte Carlo estimation of L+
