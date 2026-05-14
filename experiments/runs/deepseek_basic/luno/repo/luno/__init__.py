"""LUNO: Linearization Turns Neural Operators into Function-Valued Gaussian Processes.

LUNO provides a framework for approximate Bayesian uncertainty quantification in 
trained neural operators. It leverages model linearization to push Gaussian 
weight-space uncertainty forward to the neural operator's predictions, yielding 
a function-valued Gaussian process belief.

Key components:
- probabilistic_currying: Implementation of Theorem 3.2
- linearized_laplace: Linearized Laplace approximation for neural operators
- luno_fno: FNO-specific last-layer LUNO implementation
- weight_space: Weight-space uncertainty models (isotropic, low-rank LA)
- sampling: Sampling utilities for function-valued GPs
"""

from luno.probabilistic_currying import ProbabilisticCurrying
from luno.linearized_laplace import LinearizedLaplaceApproximation
from luno.luno_fno import LUNO_FNO, FourierGaussianRandomOperator
from luno.weight_space import (
    IsotropicGaussian,
    LowRankLaplace,
    DeepEnsembleWeightBelief,
)
from luno.sampling import (
    sample_function_valued_gp,
    compute_marginal_moments,
    compute_covariance_matrix,
)

__version__ = "0.1.0"
