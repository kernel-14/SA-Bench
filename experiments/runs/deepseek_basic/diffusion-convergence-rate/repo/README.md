# Improved Convergence Rate for Diffusion Probabilistic Models

Reproduction of the paper **"Instance-dependent Convergence Theory for Diffusion Models"** by Yuchen Jiao and Gen Li (2025).

## Overview

This repository reproduces the core theoretical contributions of the paper, which establishes an **instance-dependent convergence rate** for score-based diffusion models under a **relaxed (non-uniform) Lipschitz condition** on the score functions. The main result is an iteration complexity of:

```
min{d, d^{2/3} L^{1/3}, d^{1/3} L} * epsilon^{-2/3}
```

(up to logarithmic factors), where:
- `d` = data dimension
- `L` = non-uniform Lipschitz constant (Definition 2)
- `epsilon` = target TV distance accuracy

## Repository Structure

```
.
├── README.md                          # This file
├── demo.py                            # End-to-end demonstration of all contributions
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── schedule.py                    # Randomized learning rate schedule (Section 2.2)
│   ├── sampler.py                     # Randomized midpoint sampler (Section 2.2)
│   ├── score_functions.py             # Score functions for Gaussian and GMM targets
│   ├── lipschitz_analysis.py          # Non-uniform Lipschitz analysis (Section 3.1)
│   ├── discretization_analysis.py     # Discretization error bounds (Section 4)
│   └── parallel_sampler.py            # Parallel sampler analysis (Section 3.3)
├── experiments/
│   ├── numerical_experiment.py        # Numerical validation (Appendix A)
│   └── comparison_prior_works.py      # Comparison figures (Section 1.1, Appendix B)
└── figures/                           # Output figures directory
```

## Core Contributions Reproduced

### 1. Non-Uniform Lipschitz Condition (Definition 2)

**File:** `src/lipschitz_analysis.py`, `src/score_functions.py`

The paper introduces a relaxed Lipschitz condition that only holds with high probability
over the data distribution (and within a local neighborhood), rather than the uniform
Lipschitz condition used in prior work. Key demonstrations:

- **Example 1 (Gaussian):** `(1-bar_alpha)||s_t*(x) - s_t*(x')|| <= ||x-x'||` always holds,
  but the unscaled Lipschitz constant can be as large as `(1-bar_alpha)^{-1}` when
  the smallest variance is 0. Implemented in `GaussianScore` and `verify_gaussian_lipschitz()`.

- **Example 2 (GMM):** The non-uniform Lipschitz constant `L` scales as `O(log(H*(T+d)))`,
  while the uniform Lipschitz constant can be `O(||mu||^2/(1-bar_alpha+sigma^2)^2)`,
  which is extremely large when sigma^2 is small. Implemented in `GMMScore` and
  `verify_gmm_lipschitz_upper_bound()`.

### 2. Randomized Midpoint Sampler (Section 2.2)

**File:** `src/sampler.py`, `src/schedule.py`

The sampling algorithm discretizes the probability flow ODE:
```
dY_tau = -1/(2(1-tau)) * (Y_tau + s_tau(Y_tau)) dtau
```

using a randomized midpoint method with a carefully designed learning rate schedule:
```
hat_alpha_{T+1} = 1/T^{c_0}
hat_alpha_{t-1} = hat_alpha_t + c_1 * hat_alpha_t * (1 - hat_alpha_t) * log(T) / T
bar_alpha_t ~ Unif(hat_alpha_t, hat_alpha_{t-1})
```

The sampler operates in K rounds with N = 2T/K steps each. Total iterations = K*N = 2T.

### 3. Main Convergence Theorem (Theorem 1)

**File:** `src/lipschitz_analysis.py` (`theoretical_convergence_rate`)

The TV distance bound:
```
TV(q_K, p_{Y_K}) <= C * min{d^{3/2}, d L^{1/2}, d^{1/2} L^{3/2}} * log^4(T) / T^{3/2}
                    + C * epsilon_score * log^{1/2}(T)
```

Iteration complexity to achieve epsilon accuracy:
```
T >= min{d, d^{2/3} L^{1/3}, d^{1/3} L} * log^{8/3}(T) / epsilon^{2/3}
```

### 4. Discretization Error Analysis (Section 4)

**File:** `src/discretization_analysis.py`

Implements the key technical lemmas:
- **Lemma 7:** Schedule properties
- **Lemma 8:** Discretization error decomposition  
- **Lemma 9:** Bound on `||J_tau(x_tau) s_tau*(x_tau)||^2`
- **Lemma 10:** Derivative bound for score functions
- **Lemma 11:** Error propagation bound
- **Lemma 12:** Uniform score bound on typical set

### 5. Parallel Sampler (Theorem 2)

**File:** `src/parallel_sampler.py`

To achieve epsilon accuracy with parallel sampling:
```
N >= O((min{d^{2/3} L^{-2/3}, d^{1/3}} + 1) * log^{5/3}(T) / epsilon^{2/3})  [processors]
MK >= O(min{d log(T), L} * log^2(T))  [parallel rounds]
epsilon_score^2 <= O(epsilon^2 / log(T))
```

### 6. Numerical Experiments (Appendix A)

**File:** `experiments/numerical_experiment.py`

Validates the theoretical rate `O(log^4(T)/T^3)` for KL divergence using a Gaussian
target distribution with diagonal covariance. Configurations from Figure 2:
- (a) d=10, k=10
- (b) d=100, k=10  
- (c) d=500, k=100

### 7. Comparison with Prior Works (Section 1.1, Appendix B)

**File:** `experiments/comparison_prior_works.py`

Generates Figures 1 and 3 style comparisons between:
- Benton et al. (2023): `T ~ d/epsilon^2`
- Li & Yan (2024a): `T ~ d/epsilon`
- Li & Cai (2024): `T ~ d^{5/4}/sqrt(epsilon)`
- Li & Jiao (2024): `T ~ d^{1/3} L / epsilon^{2/3}`
- **This work:** `T ~ min{d, d^{2/3} L^{1/3}, d^{1/3} L} / epsilon^{2/3}`

## Usage

### Quick Demo

```bash
python demo.py
```

Runs through all core contributions with numerical examples.

### Numerical Experiment

```bash
python experiments/numerical_experiment.py
```

Reproduces the experimental validation of convergence rates (Appendix A).

### Comparison Figures

```bash
python experiments/comparison_prior_works.py
```

Generates comparison figures showing improvement over prior theoretical results.

## Key Assumptions

1. **Assumption 1 (Bounded second moment):** `E[||X_0||^2] < T^{c_R}` for arbitrarily large `c_R`
2. **Definition 2 (Non-uniform Lipschitz):** Lipschitz condition holds with high probability 
   over `x ~ X_t` and in a local neighborhood of radius `C*sqrt(d*(1-bar_alpha_t)*log(T))/L`
3. **Assumption 2 (Score estimation):** Access to score estimates with bounded `L_2` error

## Technical Innovations

The paper introduces several novel technical tools:

1. **Auxiliary sequences on typical sets** (Step 1-2 in Section 4): To handle error propagation
   without uniform Lipschitz, the proof introduces auxiliary sequences `\tilde{X}_k` and 
   `\tilde{Y}_k` constrained to typical sets, then relates TV distances.

2. **Direct derivative analysis** (Lemma 10): Instead of relying on log-concavity of 
   `p(X_tau | X_{tau+delta})` (which fails without uniform Lipschitz), the proof directly 
   analyzes derivatives using decomposition and statistical bounds.

3. **Adaptive bounds** (Lemma 9): The bound on `||J_tau s_tau*||^2` interpolates between 
   `O(d^2)` when L is large and `O(d L^2)` when L is small, enabling the min-form 
   convergence rate.

## Unresolved Details / Assumptions

1. **Universal constants:** The paper uses `C, c, c_0, c_1, c_2, c_R` as constants. 
   Their exact values are not specified but the paper requires `c_1/c_0` to be 
   "sufficiently large" and `c_0 >= max{c_R + 10, 10}`.

2. **Logarithmic factors:** The precise exponents of log(T) in the bounds may differ
   between the theoretical analysis (log^4(T)) and the empirical observation (log^2(T) 
   for TV, log^4(T) for KL). The paper acknowledges this suboptimality.

3. **Score estimation:** The paper assumes access to score estimates with bounded error.
   Our implementation uses exact score functions for demonstration purposes.

4. **Parallel implementation details:** The parallel sampler description references
   Appendix E.1 which is not fully provided in the main text.

## Dependencies

- Python 3.8+
- numpy
- scipy
- matplotlib (for figure generation)

## References

- Benton, J., De Bortoli, V., Doucet, A., & Deligiannidis, G. (2023). Nearly d-linear convergence bounds for diffusion models via stochastic localization.
- Li, G., & Yan, Y. (2024a). O(d/T) convergence theory for diffusion probabilistic models.
- Li, G., & Cai, Y. (2024). Provable acceleration for diffusion models.
- Li, G., & Jiao, Y. (2024). Improved convergence rate for diffusion probabilistic models.
