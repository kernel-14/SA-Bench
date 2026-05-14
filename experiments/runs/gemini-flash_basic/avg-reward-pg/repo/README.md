# Reproduction of 'Global Convergence of Policy Gradient in Average Reward MDPs'

This repository aims to reproduce the core contributions of the paper 'Global Convergence of Policy Gradient in Average Reward MDPs'.

## Core Contributions to be Replicated:

The paper highlights several key contributions:

1.  **Elimination of Smoothness Assumption:** Proving the smoothness of the average reward function without prior assumptions, using a projection technique to ensure uniqueness of the value function.
2.  **Expression for Smooth Average Cost:** Deriving an explicit, smooth expression for the average cost in terms of the policy $\pi$.
3.  **Sublinear Convergence Bounds:** Presenting finite-time bounds on the optimality gap, showing sublinear convergence of $O(\log(T))$ for general MDPs and exponential convergence for 'simple' MDPs.
4.  **Extension to Discounted Reward MDPs:** Applying the analysis to discounted reward MDPs to provide stronger results.
5.  **Experimental Validation:** Simulating the performance of policy gradient across simple MDPs to empirically validate the theoretical results.

## Approach:

My approach will focus on the theoretical aspects first, specifically the mathematical derivations and definitions that lead to the elimination of the smoothness assumption and the convergence bounds. I will then consider the experimental validation if time permits.

**Phase 1: Understanding and Defining MDP Components**

- Define the Average Reward MDP formulation (Section 2.1).
- Understand the projection technique for unique value function representation (Lemma 1, Section 3.1.1).
- Formalize the definitions of Lipschitz and smoothness constants ($L_1^\Pi$, $L_2^\Pi$, $C_m$, $C_p$, $C_r$, $\kappa_r$).

**Phase 2: Deriving Smoothness and Convergence**

- Replicate the proof of smoothness for the average reward and relative value function (Lemma 2, 3, 4).
- Understand and implement the convergence analysis (Theorem 1, Lemma 5, 6).

**Phase 3: Code Implementation Structure**

I will create a modular Python codebase within the `repo/` directory. The structure will likely include:

-   `mdp.py`: Classes and functions to define MDPs (state space, action space, transition probabilities, rewards).
-   `policy.py`: Classes and functions to represent and manipulate policies.
-   `value_functions.py`: Functions to calculate value functions, Q-functions, and the projected value function.
-   `policy_gradient.py`: Implementation of the policy gradient update rule.
-   `analysis.py`: Functions to compute smoothness constants, convergence bounds, and optimality gaps.
-   `main.py`: Script to run simulations and generate results (if experiments are undertaken).

## Current Progress:

-   Read sections: Abstract, Introduction, Preliminaries (Section 2), Main Results (Section 3 up to 3.1.2).
-   Understood the problem, key contributions, and the initial mathematical setup.
-   Created the `repo/` directory and this `README.md`.

## Assumptions and Missing Details:

Currently, no specific assumptions are made beyond those explicitly stated in the paper. Detailed definitions of constants $C_m, C_p, C_r, \kappa_r$ from Appendix A are needed for full replication.

## Assumptions and Limitations for Constants Calculation

The theoretical constants ($C_m, C_p, C_r, \kappa_r, L_1^\Pi, L_2^\Pi, C_{PL}$) are defined as global maxima over the entire policy space. In this implementation, these constants are approximated locally:

-   `C_m` and `kappa_r` are calculated for the current policy `pi_k`.
-   `C_p` and `C_r` are calculated based on the difference between successive policies `pi_k` and `pi_{k+1}`.
-   `C_PL` requires an `optimal_stationary_distribution` (`d^*`), which is generally unknown. For simulation, a proxy for `d^*` (e.g., the stationary distribution of the best policy found so far, or an analytically derived optimal one if available for simple MDPs) would be needed.

These approximations are necessary due to the intractability of computing the true global maxima over the entire policy space. This approach is sufficient for demonstrating the calculations and behavior described in the paper, but for a strict theoretical validation, these would need to be global constants.
