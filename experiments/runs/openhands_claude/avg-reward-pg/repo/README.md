# Global Convergence of Policy Gradient in Average Reward MDPs

Reproduction of the paper:

> **Global Convergence of Policy Gradient in Average Reward MDPs**  
> Navdeep Kumar, Yashaswini Murthy, Itai Shufaro, Kfir Y. Levy, R. Srikant, Shie Mannor  
> ICLR 2025

## Overview

This codebase implements the Projected Policy Gradient (PPG) algorithm for infinite-horizon average-reward tabular MDPs and reproduces all three simulation experiments from Section 4 / Appendix C of the paper.

## File Structure

| File | Description |
|------|-------------|
| `mdp.py` | `AverageRewardMDP` class: transition matrix, stationary distribution, projected value function `v_φ^π`, Q-function, policy gradient |
| `policy_gradient.py` | PPG algorithm (Equation 6), theoretical convergence bound (Theorem 1) |
| `complexity.py` | MDP complexity constants `C_m, C_p, C_r, κ_r, L_1^Π, L_2^Π, C_PL` (Table 1) |
| `mdp_factory.py` | MDP construction for all three experiments (Appendix C) |
| `experiments.py` | Runs all experiments and saves figures |
| `plot.py` | Plotting utilities for Figures 1 and 2 |
| `utils.py` | Simplex projection, policy projection |
| `config.py` | All hyperparameters |
| `test_implementation.py` | Unit tests verifying mathematical properties |

## Key Algorithms

### Projected Policy Gradient (Equation 6)
```
π_{k+1} = Proj_Π[π_k + η ∂ρ^π/∂π|_{π=π_k}]
```

### Projected Value Function (Lemma 1)
```
v_φ^π = (I - Φ P^π)^{-1} Φ r^π
```
where `Φ = I - 11^T/|S|` projects onto the subspace orthogonal to `1`.

### Convergence Bound (Theorem 1)
```
ρ* - ρ^{π_k} ≤ 1 / (1/(ρ* - ρ^{π_0}) + ν k)
```
where `ν = c(1 + 4c)^{-3/2}` and `c = 1/(32 C_PL^2 |S| L_2^Π)`.

## Experiments

### Experiment 1 (Figure 1a): Varying (S, A) sizes
- MDPs: `(S,A) ∈ {(3,3), (9,9), (81,81)}`
- Transition: `P(i|s,i) = (1+1/S)/2`, `P(i|s,j) = 1/(2S)` for `i≠j`
- Reward: half actions → +1, half → -1 (maximal variance)
- 2000 iterations

### Experiment 2 (Figure 1b): Varying reward variance
- `S = A = 16`, fixed random transition kernel
- Four reward variance levels: none, low (1/8), high (1/4), max (1/2)
- 2000 iterations

### Experiment 3 (Figure 2): Varying transition kernels
- `S = A = 16`, high-variance reward
- Three kernels: uniform, non-uniform, deterministic
- 3000 iterations

## Usage

```bash
# Run all experiments
python experiments.py

# Run a single experiment
python experiments.py --experiment 1
python experiments.py --experiment 2
python experiments.py --experiment 3

# Print MDP complexity constants
python experiments.py --complexity

# Custom output directory
python experiments.py --output-dir results/

# Run tests
pytest test_implementation.py -v
```

## MDP Complexity Constants (Table 1)

| Constant | Definition | Bound | Meaning |
|----------|-----------|-------|---------|
| `C_m` | `max_π ||(I - ΦP^π)^{-1}||_∞` | `2C_e|S|/(1-λ)` | Mixing rate |
| `C_p` | `max_{π,π'} ||P^{π'}-P^π||_∞ / ||π'-π||_2` | `[0, √|A|]` | Transition diameter |
| `C_r` | `max_{π,π'} ||r^{π'}-r^π||_∞ / ||π'-π||_2` | `[0, √|A|]` | Reward diameter |
| `κ_r` | `max_π ||Φr^π||_∞` | `[0, 2)` | Reward variance |
| `L_2^Π` | (see Lemma 4) | `[0, k₂|A|C_m³]` | Smoothness constant |
| `C_PL` | `max_{π,s} d^{π*}(s)/d^π(s)` | — | Gradient domination |
