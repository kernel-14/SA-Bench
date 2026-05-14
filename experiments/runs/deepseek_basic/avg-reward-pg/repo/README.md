# Global Convergence of Policy Gradient in Average Reward MDPs

Reproduction of the paper:  
**"Global Convergence of Policy Gradient in Average Reward MDPs"**  
by Navdeep Kumar, Yashaswini Murthy, Itai Shufaro, Kfir Y. Levy, R. Srikant, and Shie Mannor.

## Overview

This repository implements the core theoretical and experimental contributions of the paper. The paper presents the first comprehensive finite-time global convergence analysis of policy gradient for infinite horizon average reward MDPs.

### Key Contributions Reproduced

1. **Average Reward MDP Formulation** (Section 2.1): Finite state/action MDP with randomized policies, average reward objective, and the policy gradient theorem.

2. **Unique Value Function via Projection** (Lemma 1): The projection matrix Φ = I - 11^T/|S| that provides a unique value function representation v_φ^π = (I - ΦP^π)^{-1} Φ r^π, overcoming the non-uniqueness challenge.

3. **Smoothness of Average Reward** (Lemmas 2-4, Appendix A): Proof that the average reward ρ^π is smooth with respect to the policy, without assuming it a priori. This is the key technical contribution that eliminates the smoothness assumption from previous work.

4. **MDP Complexity Constants** (Table 1, Lemma 18): Computation of constants C_m, C_p, C_r, κ_r that capture the complexity of the underlying MDP and appear in the convergence bounds.

5. **Projected Policy Gradient Algorithm** (Section 2.1): The update rule π_{k+1} = Proj_Π[π_k + η ∇ρ^π|_{π=π_k}] with exact gradient computation.

6. **Convergence Analysis** (Theorem 1, Appendix B):
   - Sublinear convergence: ρ^* - ρ^{π_k} ≤ 1/(1/(ρ^* - ρ^{π_0}) + νk) for all MDPs
   - Exponential convergence for simple MDPs (L_2^Π << 1)

7. **Extension to Discounted Reward MDPs** (Section 3.2, Appendix D): Instance-dependent bounds that can improve upon existing state-of-the-art bounds.

8. **Simulations** (Section 4, Appendix C):
   - Experiment 1: Convergence with different state/action space sizes
   - Experiment 2: Convergence with different reward variances
   - Experiment 3: Convergence with different transition kernels

## Repository Structure

```
/mnt/paper2any/pzw/proj/paperagent/hx/Research_space/SemanticAlign-Bench/experiments/runs/deepseek_basic/avg-reward-pg/avg-reward-pg-20260505-071827/workspace/repo/
├── README.md                    # This file
├── main.py                      # Main entry point with demonstrations
├── src/
│   ├── __init__.py              # Package initialization
│   ├── mdp.py                   # Average reward MDP formulation
│   ├── projection.py            # Φ projection matrix and unique value function
│   ├── constants.py             # MDP complexity constants (C_m, C_p, C_r, κ_r,
│   │                            #   L_1^Π, L_2^Π, C_PL)
│   ├── policy_gradient.py       # Projected policy gradient algorithm
│   ├── smoothness.py            # Smoothness analysis (directional derivatives)
│   ├── convergence.py           # Convergence lemmas and Theorem 1 bounds
│   ├── discounted.py            # Extension to discounted reward MDPs
│   └── simulations.py           # Three simulation experiments
└── tests/                       # Test directory
```

## Module Descriptions

### `src/mdp.py` - Average Reward MDP
Implements the core MDP model with finite state and action spaces. Key methods:
- `get_transition_matrix(pi)`: Compute P^π
- `get_stationary_distribution(pi)`: Compute d^π
- `average_reward(pi)`: Compute ρ^π = Σ d^π(s) r^π(s)
- `compute_value_function(pi)`: Compute v^π (unique under d^π·v=0)
- `compute_q_function(pi)`: Compute Q^π(s,a)

### `src/projection.py` - Unique Value Function (Lemma 1)
- `make_projection_matrix(n_states)`: Φ = I - 11^T/|S|
- `projected_value_function(P_pi, r_pi, Phi)`: v_φ^π = (I - ΦP^π)^{-1} Φ r^π
- `verify_projection_properties(n_states)`: Verifies Φ·1=0, Φ²=Φ, ||Φ||_∞ ≤ 2

### `src/constants.py` - MDP Complexity Constants (Table 1, Lemma 18)
Computes the constants that capture MDP complexity:
- `C_m`: Maximum operator norm of (I - ΦP^π)^{-1} [mixing rate]
- `C_p`: Diameter of transition kernel
- `C_r`: Diameter of reward function  
- `κ_r`: Variance of reward function
- `L_1^Π`: Restricted Lipschitz constant
- `L_2^Π`: Restricted smoothness constant
- `C_PL`: Distribution mismatch coefficient

### `src/policy_gradient.py` - Algorithm Implementation
- `compute_policy_gradient(mdp, pi)`: ∇ρ^π(s,a) = d^π(s) Q^π(s,a)
- `project_policy(pi)`: Euclidean projection onto simplex
- `run_projected_policy_gradient(...)`: Full algorithm with tracking
- `compute_theoretical_bound(...)`: Theorem 1 bounds

### `src/smoothness.py` - Smoothness Analysis (Appendix A)
Implements the analytical derivations of directional derivatives:
- First/second order directional derivatives of v_φ^π
- Verification of Lipschitz and smoothness constants

### `src/convergence.py` - Convergence Proofs (Appendix B)
Implements the key lemmas leading to Theorem 1:
- Lemma 5: Sufficient increase
- Lemma 6: Performance difference
- Lemma 7: Gradient domination
- Lemma 8: Directional derivative bound after update
- Lemmas 24-26: Recursion bounds
- Theorem 1: Main convergence result

### `src/discounted.py` - Discounted Extension (Section 3.2, Appendix D)
- `DiscountedRewardMDP` class
- Discounted smoothness constant computation
- Comparison with state-of-the-art bounds

### `src/simulations.py` - Experiments (Section 4, Appendix C)
Reproduces all three experiments from the paper with the exact setup described in Appendix C.

## Usage

```bash
# Run all demonstrations
python main.py

# Or import specific modules
from src.mdp import AverageRewardMDP, make_random_mdp
from src.policy_gradient import run_projected_policy_gradient
from src.constants import compute_all_constants
from src.simulations import run_all_simulations
```

## Assumptions and Limitations

1. **Assumption 1** (irreducible and aperiodic MDP): The code assumes all policies induce irreducible and aperiodic Markov chains. This is required for the existence of a unique stationary distribution.

2. **Exact Gradients**: The implementation computes exact policy gradients (planning setting). The paper notes that incorporating learning errors is future work.

3. **Tabular Policies**: The implementation uses tabular (direct) policy parameterization. Extension to parametric policies is discussed but not implemented.

4. **Optimal Policy**: In simulations, the optimal policy is approximated from the best performing policy during the run. For rigorous comparison, one would need to compute the true optimal policy via value iteration or policy iteration.

5. **Constant Computation**: The MDP complexity constants are computed over a sampled set of policies rather than the entire policy class (which would be infinite). This provides lower bounds on the true constants.

## Relationship to Paper

| Paper Section | Implementation |
|---------------|---------------|
| Section 2.1 (MDP formulation) | `src/mdp.py` |
| Lemma 1 (Projection) | `src/projection.py` |
| Lemmas 2-4 (Smoothness) | `src/smoothness.py`, `src/constants.py` |
| Table 1, Lemma 18 (Constants) | `src/constants.py` |
| Equation 6 (PG update) | `src/policy_gradient.py` |
| Lemmas 5-8 (Convergence) | `src/convergence.py` |
| Theorem 1 (Main result) | `src/convergence.py`, `src/policy_gradient.py` |
| Section 3.2 (Discounted extension) | `src/discounted.py` |
| Section 4, Appendix C (Simulations) | `src/simulations.py` |
| Appendix A (Smoothness proofs) | `src/smoothness.py` |
| Appendix B (Convergence proofs) | `src/convergence.py` |
| Appendix D (Discounted details) | `src/discounted.py` |

## Dependencies

- Python 3.7+
- NumPy

No other dependencies are required.
