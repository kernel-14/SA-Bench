"""
Global Convergence of Policy Gradient in Average Reward MDPs.

This package implements the core contributions of the paper:
"Global Convergence of Policy Gradient in Average Reward MDPs"
by Kumar, Murthy, Shufaro, Levy, Srikant, and Mannor.

Key modules:
- mdp: Average reward MDP formulation (Section 2.1)
- projection: Unique value function via Φ projection (Lemma 1)
- constants: MDP complexity constants (Table 1, Lemma 18)
- policy_gradient: Projected policy gradient algorithm (Section 2.1, Theorem 1)
- smoothness: Smoothness analysis of average reward (Appendix A)
- convergence: Convergence lemmas and Theorem 1 bounds (Appendix B)
- discounted: Extension to discounted reward MDPs (Section 3.2, Appendix D)
- simulations: Experimental validation (Section 4, Appendix C)
- plotting: Figure generation utilities
"""

from .mdp import AverageRewardMDP, make_random_mdp
from .projection import make_projection_matrix, projected_value_function
from .constants import (
    compute_C_m, compute_C_p, compute_C_r, compute_kappa_r,
    compute_L1_Pi, compute_L2_Pi, compute_C_PL, compute_all_constants,
)
from .policy_gradient import (
    compute_policy_gradient, project_policy,
    projected_policy_gradient_step, run_projected_policy_gradient,
    compute_theoretical_bound,
)
from .smoothness import (
    directional_derivative_first_order,
    directional_derivative_second_order,
    verify_smoothness_constant,
    verify_lipschitz_constant,
)
from .convergence import (
    sufficient_increase_lemma,
    performance_difference_lemma,
    gradient_domination_lemma,
    suboptimality_recursion,
    sublinear_convergence_bound,
    exponential_convergence_bound,
    theorem_1_bounds,
)
from .discounted import DiscountedRewardMDP, discounted_policy_gradient_step
from .simulations import (
    experiment_1_state_action_size,
    experiment_2_reward_variance,
    experiment_3_transition_kernel,
    run_all_simulations,
)
