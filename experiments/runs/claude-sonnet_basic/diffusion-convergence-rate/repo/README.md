# Reproduction: Instance-dependent Convergence Theory for Diffusion Models

This repository reproduces the key contributions of the paper:

**"Instance-dependent Convergence Theory for Diffusion Models"**  
by Yuchen Jiao and Gen Li (2025)

arXiv: 2410.13738

## Overview

The paper establishes an instance-dependent convergence rate for score-based diffusion models (SGMs) using a randomized midpoint sampler. The main result (Theorem 1) shows that the sampler achieves:

$$\text{TV}(q_K, p_{Y_K}) \leq C \cdot \frac{\min\{d^{3/2}, dL^{1/2}, d^{1/2}L^{3/2}\} \log^4 T}{T^{3/2}} + C \varepsilon_{\text{score}} \log^{1/2} T$$

which corresponds to an iteration complexity of:

$$T \gtrsim \min\{d, d^{2/3}L^{1/3}, d^{1/3}L\} \varepsilon^{-2/3} \log^{8/3} T$$

where $d$ is the data dimension, $L$ is the non-uniform Lipschitz constant (Definition 2),
$\varepsilon$ is the target TV distance accuracy, and $\varepsilon_{\text{score}}$ is the score estimation error.

## Key Contributions Reproduced

### 1. Randomized Midpoint Sampler (`src/sampler.py`)

Implements the sampling algorithm from Section 2.2:
- **Randomized schedule**: hat_alpha_{T+1} = 1/T^{c0}, hat_alpha_{t-1} = hat_alpha_t + c1*hat_alpha_t*(1-hat_alpha_t)*log(T)/T
- **Iterative update** (Equation 10): ODE discretization with randomized time steps
- **Noise injection** (Equation 11): Gaussian noise added between rounds

### 2. Score Functions (`src/score_functions.py`)

Implements exact score functions for:
- **Gaussian distributions** (Example 1): s_t*(x) = -Sigma_t^{-1} x
- **Gaussian Mixture Models** (Example 2): Score function with posterior weights

### 3. Theoretical Bounds (`src/theoretical_bounds.py`)

Implements the convergence rate formulas:
- `tv_bound_theorem1()`: TV distance upper bound from Theorem 1
- `iteration_complexity()`: Iteration complexity to achieve eps accuracy
- `compare_with_prior_works()`: Comparison with prior works
- `improvement_factor()`: Improvement factor over Li & Jiao (2024)
- `parallel_sampler_complexity()`: Parallel sampler complexity from Theorem 2

### 4. Numerical Experiments (`experiments/numerical_experiment.py`)

Reproduces Figure 2 from the paper (Appendix A):
- Target: d-dimensional Gaussian with diagonal covariance
- First k diagonal entries uniform in [0, 10], remaining set to 0
- Metric: KL divergence between Y_K and q_K = X_{tau_{K,0}}
- Theoretical rate: O(log^4(T)/T^3) in KL divergence

Three settings:
- (a) d=10, k=10
- (b) d=100, k=10
- (c) d=500, k=100

### 5. Comparison Plots (`experiments/comparison_plots.py`)

Reproduces Figures 1 and 3 from the paper:
- **Figure 1**: Iteration complexity vs L (left) and vs eps when L=infinity (right)
- **Figure 3**: TV distance achieved by various theories for fixed T

## Repository Structure

```
submission/
├── src/
│   ├── sampler.py              # Randomized midpoint sampler (Section 2.2)
│   ├── score_functions.py      # Score functions for Gaussian and GMM targets
│   ├── theoretical_bounds.py   # Convergence rate formulas (Theorem 1, 2)
│   └── gaussian_tracker.py     # Analytical Gaussian distribution tracker
├── experiments/
│   ├── numerical_experiment.py # Figure 2: Convergence rate validation
│   └── comparison_plots.py     # Figures 1 & 3: Comparison with prior works
├── tests/
│   └── test_sampler.py         # Unit tests
├── figures/                    # Output directory for generated figures
└── README.md                   # This file
```

## Running the Code

### Prerequisites

```bash
pip install numpy matplotlib
```

### Run Numerical Experiments (Figure 2)

```bash
cd experiments
python numerical_experiment.py
```

### Generate Comparison Plots (Figures 1 & 3)

```bash
cd experiments
python comparison_plots.py
```

### Run Tests

```bash
cd tests
python test_sampler.py
```

## Implementation Details

### Sampler Algorithm

The sampler implements the probability flow ODE discretization from Section 2.2.
The key update rule (Equation 10) in normalized coordinates u_{k,n} = Y_{k,n}/sqrt(1-tau_{k,n}) is:

u_{k,n} = u_{k,0} + s(Y_{k,0})/(2*(1-tau_{k,0})^{3/2}) * (tau_{k,0} - hat_tau_{k,0})
         + sum_{i=1}^{n-1} s(Y_{k,i})/(2*(1-tau_{k,i})^{3/2}) * (hat_tau_{k,i-1} - hat_tau_{k,i})
         + s(Y_{k,n-1})/(2*(1-tau_{k,n-1})^{3/2}) * (hat_tau_{k,n-1} - tau_{k,n})

The noise injection step (Equation 11) is:

Y_{k+1} = sqrt((1-tau_{k+1,0})/(1-tau_{k,N})) * Y_{k,N}
         + sqrt((tau_{k+1,0} - tau_{k,N})/(1-tau_{k,N})) * Z_k

### Analytical Gaussian Tracking

For Gaussian targets, the score function is linear: s_tau*(x) = -x/Sigma_tau.
This means the sampler output Y_{k,n} remains Gaussian throughout. We track the mean
and covariance analytically to compute the exact KL divergence.

### Non-uniform Lipschitz Condition

The paper introduces a relaxed Lipschitz condition (Definition 2) that only requires
the Lipschitz property to hold with high probability over x ~ X_t, rather than uniformly.
For Gaussian Mixture Models, this constant L scales only logarithmically with the number
of components and dimension, whereas the uniform Lipschitz constant can be extremely large.

## Assumptions and Limitations

1. **Schedule constants**: We use c0=5, c1=10 as defaults (paper requires sufficiently large values).
2. **Logarithmic factors**: The theoretical bound has log^4(T) factors; empirical results suggest log^2(T).
3. **Degenerate dimensions**: Approximated with small variance 1e-6 for numerical stability.
4. **Score estimation**: Experiments assume access to the exact score function.
5. **Parallel implementation**: Theorem 2 is described but not fully implemented as parallel code.

## Comparison with Prior Works

| Method | Iteration Complexity | Condition |
|--------|---------------------|-----------|
| Benton et al. (2023) | O(d*eps^{-2}) | General |
| Li & Yan (2024a) | O(d*eps^{-1}) | General |
| Li & Cai (2024) | O(d^{5/4}*eps^{-1/2}) | General |
| Li & Jiao (2024) | O(d^{1/3}*L*eps^{-2/3}) | Uniform Lipschitz |
| **Ours (Theorem 1)** | O(min{d, d^{2/3}*L^{1/3}, d^{1/3}*L}*eps^{-2/3}) | Non-uniform Lipschitz |

Our result improves over Li & Jiao (2024) by a factor of max{d^{-2/3}*L, d^{-1/3}*L^{2/3}, 1},
which is significant when L >= sqrt(d).
