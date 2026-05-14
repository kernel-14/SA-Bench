# Reproduction: Global Convergence of Policy Gradient in Average Reward MDPs

This repository reproduces the key contributions of the paper:

> **"Global Convergence of Policy Gradient in Average Reward MDPs"**  
> Navdeep Kumar, Yashaswini Murthy, Itai Shufaro, Kfir Y. Levy, R. Srikant, Shie Mannor  
> ICLR 2025

## Overview

The paper presents the first comprehensive finite-time global convergence analysis of policy gradient for infinite horizon average reward MDPs. Key contributions:

1. **Smoothness of Average Reward** (Lemmas 2–4): Proves that the average reward is smooth w.r.t. the policy without requiring unverified assumptions, using a projection technique to handle the non-uniqueness of the value function.
2. **Sublinear Convergence Bounds** (Theorem 1): Shows that policy gradient iterates converge at O(1/T) rate with O(log T) regret.
3. **MDP Complexity Constants** (Table 1): Introduces constants (C_m, C_p, C_r, κ_r) that capture MDP complexity and appear explicitly in the convergence bounds.
4. **Extension to Discounted MDPs** (Section 3.2): Provides instance-dependent bounds for discounted reward MDPs.
5. **Experimental Validation** (Section 4): Simulations showing how MDP complexity affects convergence rates.

## Repository Structure

```
submission/
├── mdp.py                  # Core MDP class: value functions, gradients, constants
├── policy_gradient.py      # Projected policy gradient algorithm + theoretical bound
├── mdp_construction.py     # MDP construction for all three experiments
├── experiments.py          # Main experiment runner (Figures 1 & 2)
├── convergence_analysis.py # Theorem 1 validation and smoothness verification
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Key Algorithms

### Projected Policy Gradient (Equation 6 in paper)
```
π_{k+1} = Proj_Π [ π_k + η * ∂ρ^π/∂π |_{π=π_k} ]
```

### Average Reward Policy Gradient Theorem
```
∂ρ/∂π(a|s) = d^π(s) * Q^π(s, a)
```

### Projected Value Function (Lemma 1)
```
v_φ^π = (I - Φ P^π)^{-1} Φ r^π
```
where `Φ = I - 11^T/|S|` projects onto the subspace orthogonal to **1**.

This provides the unique value function satisfying **1**^T v = 0, enabling the smoothness analysis.

## Experiments

### Figure 1(a): Varying State/Action Space Sizes
- MDPs with (S, A) ∈ {(3,3), (9,9), (81,81)}
- Transition kernel (Appendix C.1): P(i|s,i) = (1+1/S)/2, P(i|s,j) = 1/(2S) for i≠j
- Reward: half actions get +1, half get -1 (maximal variance)
- 2000 iterations; plots average reward vs iteration
- Expected: slower convergence for larger (S, A) due to larger L₂^Π

### Figure 1(b): Varying Reward Variances
- Fixed (S, A) = (16, 16), randomly generated transition kernel
- Four reward variance levels (Appendix C.2): no variance, low (1/8 negative), high (1/4 negative), max (1/2 negative)
- 2000 iterations; plots average reward vs iteration
- Expected: higher variance → larger C_r → larger L₂^Π → slower convergence

### Figure 2: Varying Transition Kernels
- Fixed (S, A) = (16, 16), high-variance reward function
- Three kernel types (Appendix C.3): uniform, non-uniform, deterministic
- 3000 iterations; plots |Δ average reward| vs iteration
- Expected: deterministic > non-uniform > uniform in terms of C_p, hence slowest to fastest convergence

## Theoretical Results

### Theorem 1 (Main Result)
For step size η < 1/L₂^Π, the projected policy gradient iterates satisfy:

**General MDPs (sublinear convergence):**
```
ρ* - ρ^{π_k} ≤ 1 / (1/(ρ* - ρ^{π_0}) + ν·k)
```
where:
- c = 1 / (32 · C_PL² · |S| · L₂^Π)
- ν = c · (1 + 4c)^{-3/2}

This corresponds to O(log T) regret.

**Simple MDPs** (L₂^Π << 1, exponential convergence):
```
ρ* - ρ^{π_k} ≤ c^{-k/2} · (ρ* - ρ^{π_0})^{1/2^k}
```
where 1/c = 32 |S| L₂^Π C_PL² < 1.

### MDP Complexity Constants (Table 1)

| Constant | Definition | Range | Meaning |
|----------|-----------|-------|---------|
| C_m | max_π ‖(I - ΦP^π)^{-1}‖_∞ | [1, 2C_e\|S\|/(1-λ)] | Inverse mixing rate |
| C_p | max_{π,π'} ‖P^{π'}-P^π‖_∞/‖π'-π‖₂ | [0, √\|A\|] | Transition diameter |
| C_r | max_{π,π'} ‖r^{π'}-r^π‖_∞/‖π'-π‖₂ | [0, √\|A\|] | Reward diameter |
| κ_r | max_π ‖Φr^π‖_∞ | [0, 2) | Reward variance |
| L₁^Π | 2(C_r + C_p C_m κ_r + 2(C_m² C_p κ_r + C_m C_r)) | — | Restricted Lipschitz |
| L₂^Π | 4(C_p² C_m² κ_r + C_p C_m C_r + (C_p+1)(C_m² C_p κ_r + C_m C_r) + 4(C_m³ C_p² κ_r + C_m² C_p C_r)) | — | Restricted smoothness |
| C_PL | max_{π,s} d^{π*}(s)/d^π(s) | — | Gradient domination |

## Running the Code

```bash
# Install dependencies
pip install numpy matplotlib

# Run all experiments (Figures 1 and 2)
python experiments.py

# Run convergence analysis and Theorem 1 validation
python convergence_analysis.py
```

Output figures are saved to the `figures/` directory.

## Dependencies

- numpy
- matplotlib

## Implementation Notes

### Core Computations (mdp.py)
- `get_stationary_distribution`: Solves (P^π - I)^T d = 0 with normalization; falls back to power iteration
- `get_projected_value_function`: Solves (I - ΦP^π) v = Φr^π (Lemma 1)
- `get_q_function`: Q^π(s,a) = r(s,a) + Σ_{s'} P(s'|s,a) v_φ^π(s') - ρ^π
- `get_policy_gradient`: grad[s,a] = d^π(s) · Q^π(s,a)
- `get_optimal_policy`: Policy iteration
- `compute_mdp_constants`: Estimates C_m, C_p, C_r, κ_r, L₁, L₂, C_PL over a sample of policies

### Projection (policy_gradient.py)
- `project_onto_simplex`: Duchi et al. (2008) O(n log n) algorithm
- `project_policy`: Projects each state's action distribution onto the simplex

## Assumptions and Unresolved Details

1. **Ergodicity** (Assumption 1): All policies induce irreducible and aperiodic Markov chains. This is satisfied by construction for the uniform and non-uniform kernels. For the deterministic kernel, stochastic policies (mixtures) are ergodic.

2. **Step size**: The paper requires η < 1/L₂^Π but does not specify the exact value used in experiments. We use η = 0.01 throughout.

3. **Initial policy**: Not specified in the paper; we use the uniform policy π(a|s) = 1/|A|.

4. **Transition kernel for Experiment 1b**: The paper says "randomly generated transition kernel" without specifying the distribution. We use a Dirichlet(1,...,1)-sampled kernel (uniform Dirichlet), with the same seed for all variance types to ensure comparability.

5. **Deterministic kernel**: The paper says "random permutation of the identity matrix". We use a different random permutation per action, which creates a non-trivial deterministic MDP where each action maps states according to a permutation.

6. **MDP complexity constants**: The paper defines these as maxima over all policies. We approximate them by sampling a finite set of policies (uniform, deterministic, and random Dirichlet). The theoretical bounds are upper bounds, so the empirical convergence is typically much faster.
