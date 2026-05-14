import numpy as np
from mdp import AverageRewardMDP
from policy import Policy
from analysis import Analysis

def create_simple_mdp():
    """
    Creates a simple 3-state, 2-action MDP for demonstration.
    State 0: Start state
    State 1: Intermediate state
    State 2: Terminal-like state (high reward loop)
    """
    num_states = 3
    num_actions = 2

    # Transitions[s, a, s_prime]
    transitions = np.zeros((num_states, num_actions, num_states))

    # Action 0: move towards state 2
    # From state 0
    transitions[0, 0, 0] = 0.1 # Stay in 0
    transitions[0, 0, 1] = 0.9 # Go to 1
    # From state 1
    transitions[1, 0, 1] = 0.1 # Stay in 1
    transitions[1, 0, 2] = 0.9 # Go to 2
    # From state 2 (self-loop with high reward)
    transitions[2, 0, 2] = 1.0

    # Action 1: move towards state 0
    # From state 0 (self-loop with low reward)
    transitions[0, 1, 0] = 1.0
    # From state 1
    transitions[1, 1, 0] = 0.8 # Go to 0
    transitions[1, 1, 1] = 0.2 # Stay in 1
    # From state 2
    transitions[2, 1, 1] = 1.0 # Go to 1

    # Rewards[s, a]
    rewards = np.zeros((num_states, num_actions))
    rewards[0, 0] = 0.1
    rewards[0, 1] = 0.0
    rewards[1, 0] = 0.5
    rewards[1, 1] = 0.1
    rewards[2, 0] = 1.0
    rewards[2, 1] = 0.8

    # Normalize transitions to ensure each row sums to 1
    for s in range(num_states):
        for a in range(num_actions):
            if np.sum(transitions[s, a, :]) > 0:
                transitions[s, a, :] /= np.sum(transitions[s, a, :])
            else: # If all zeros, make it a self-loop
                transitions[s, a, s] = 1.0

    return AverageRewardMDP(num_states, num_actions, transitions, rewards)

def solve_mdp_for_optimal_rho(mdp: AverageRewardMDP, num_iterations=1000, tol=1e-6):
    """
    Approximates the optimal average reward (rho*) and optimal stationary distribution (d*)
    by running policy gradient for many steps.
    """
    best_rho = -np.inf
    best_policy_matrix = None

    policy = Policy(mdp.num_states, mdp.num_actions)
    current_policy_matrix = policy.get_policy_matrix()
    
    step_size = 0.1 
    
    for i in range(num_iterations):
        P_pi = mdp.get_policy_transition_kernel(current_policy_matrix)
        r_pi = mdp.get_policy_reward_function(current_policy_matrix)
        d_pi = mdp.get_stationary_distribution(P_pi)
        rho_pi = mdp.get_average_reward(r_pi, d_pi)
        
        if rho_pi > best_rho:
            best_rho = rho_pi
            best_policy_matrix = np.copy(current_policy_matrix) # Store a copy

        v_phi_pi = mdp.get_relative_value_function(P_pi, r_pi, rho_pi)
        q_pi = mdp.get_q_function(current_policy_matrix, P_pi, r_pi, rho_pi, v_phi_pi)
        
        policy_grad = policy.get_policy_gradient(d_pi, q_pi)
        policy.update_policy(policy_grad, step_size)
        current_policy_matrix = policy.get_policy_matrix()

    if best_policy_matrix is not None:
        P_best_pi = mdp.get_policy_transition_kernel(best_policy_matrix)
        d_best_pi = mdp.get_stationary_distribution(P_best_pi)
    else:
        d_best_pi = np.ones(mdp.num_states) / mdp.num_states # Fallback uniform dist

    print(f"Approximated optimal average reward (rho*): {best_rho:.4f}")
    return best_rho, d_best_pi, best_policy_matrix


def run_simulation():
    mdp = create_simple_mdp()
    policy = Policy(mdp.num_states, mdp.num_actions)
    analysis = Analysis(mdp)

    num_iterations = 200 # Number of policy gradient iterations
    step_size = 0.05 # Learning rate

    rho_history = []
    optimality_gap_history = []
    optimality_gap_bound_history = []
    
    # Approximate rho* and d* for C_PL calculation
    rho_star, d_star, _ = solve_mdp_for_optimal_rho(mdp, num_iterations=1000)

    # Initial policy
    current_policy_matrix = policy.get_policy_matrix()
    P_pi_0 = mdp.get_policy_transition_kernel(current_policy_matrix)
    r_pi_0 = mdp.get_policy_reward_function(current_policy_matrix)
    d_pi_0 = mdp.get_stationary_distribution(P_pi_0)
    rho_pi_0 = mdp.get_average_reward(r_pi_0, d_pi_0)

    rho_history.append(rho_pi_0)
    optimality_gap_history.append(rho_star - rho_pi_0)

    for k in range(num_iterations):
        # Calculate policy-dependent values
        P_pi = mdp.get_policy_transition_kernel(current_policy_matrix)
        r_pi = mdp.get_policy_reward_function(current_policy_matrix)
        d_pi = mdp.get_stationary_distribution(P_pi)
        rho_pi = mdp.get_average_reward(r_pi, d_pi)
        v_phi_pi = mdp.get_relative_value_function(P_pi, r_pi, rho_pi)
        q_pi = mdp.get_q_function(current_policy_matrix, P_pi, r_pi, rho_pi, v_phi_pi)

        rho_history.append(rho_pi)
        optimality_gap_history.append(rho_star - rho_pi)

        # Calculate policy gradient
        policy_grad = policy.get_policy_gradient(d_pi, q_pi)
        
        # Store previous policy for Cp and Cr calculation
        previous_policy_matrix = np.copy(current_policy_matrix)

        # Update policy
        policy.update_policy(policy_grad, step_size)
        current_policy_matrix = policy.get_policy_matrix()
        
        # Calculate analysis constants (local approximations)
        # Note: These constants (Cm, Cp, Cr, kappa_r) are approximations based on the current policy.
        # The theoretical bounds require global maxima of these, which are intractable to compute.
        Cm = analysis.get_Cm_constant(current_policy_matrix)
        # For Cp and Cr, we need the previous and current policy. For the very first iteration, Cp/Cr might be 0
        # if policy_matrix_k == policy_matrix_kp1. We should handle this by either skipping
        # the bound calculation for k=0 or ensuring a non-zero change in policy.
        Cp = analysis.get_Cp_constant(previous_policy_matrix, current_policy_matrix)
        Cr = analysis.get_Cr_constant(previous_policy_matrix, current_policy_matrix)
        kappa_r = analysis.get_kappa_r_constant(current_policy_matrix)
        
        L2_Pi = analysis.get_L2_Pi_constant(Cm, Cp, Cr, kappa_r)
        
        C_PL = analysis.get_CPL_constant(d_star, d_pi)

        # Calculate optimality gap bound
        if L2_Pi > 1e-9 and C_PL > 1e-9 and rho_star - rho_pi_0 > 1e-9: # Ensure constants are valid for calculation
            bound = analysis.calculate_optimality_gap_bound(rho_star, optimality_gap_history[0], L2_Pi, C_PL, k)
            optimality_gap_bound_history.append(bound)
        else:
            optimality_gap_bound_history.append(np.inf) # Cannot compute a meaningful bound

        if (k + 1) % 50 == 0:
            print(f"Iteration {k+1}: Avg Reward = {rho_pi:.4f}, Opt Gap = {rho_star - rho_pi:.4f}, Bound = {optimality_gap_bound_history[-1]:.4f}")

    print("
--- Simulation Results ---")
    print(f"Optimal Average Reward (approx): {rho_star:.4f}")
    print(f"Final Average Reward: {rho_history[-1]:.4f}")
    print(f"Final Optimality Gap: {rho_star - rho_history[-1]:.4f}")
    print(f"Final Optimality Gap Bound: {optimality_gap_bound_history[-1]:.4f}")

if __name__ == "__main__":
    run_simulation()
