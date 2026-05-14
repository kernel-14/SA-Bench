import numpy as np
from typing import Dict, List, Optional

# Assuming mdp_definitions.py and mdp_solver.py are in the same directory or importable path
from mdp_definitions import MDP
from mdp_solver import MDPSolver


class PPGAlgorithm:
    """
    Implements the core Projected Policy Gradient (PPG) algorithm for Average Reward MDPs.
    This class manages the iterative policy updates, calculates gradients, performs
    projections, and stores the convergence history.
    """

    def __init__(self, mdp: MDP, solver: MDPSolver, learning_rate: float,
                 initial_policy: Optional[np.ndarray] = None):
        """
        Initializes the PPGAlgorithm.

        Args:
            mdp (MDP): An instance of the MDP class containing environment details.
            solver (MDPSolver): An instance of the MDPSolver for computing MDP-related quantities.
            learning_rate (float): The step size (η) for the policy gradient update.
            initial_policy (Optional[np.ndarray]): The starting policy. If None, a uniform
                                                    random policy is generated. Shape (S, A).
        """
        if not isinstance(mdp, MDP):
            raise TypeError("mdp must be an instance of mdp_definitions.MDP")
        if not isinstance(solver, MDPSolver):
            raise TypeError("solver must be an instance of mdp_solver.MDPSolver")
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float.")
        if initial_policy is not None and (not isinstance(initial_policy, np.ndarray) or initial_policy.shape != (mdp.S, mdp.A)):
            raise ValueError(f"initial_policy must be a numpy array of shape ({mdp.S}, {mdp.A}) or None.")
        
        self.mdp: MDP = mdp
        self.solver: MDPSolver = solver
        self.learning_rate: float = learning_rate

        if initial_policy is None:
            self.current_policy: np.ndarray = self.mdp.generate_uniform_random_policy()
        else:
            self.current_policy: np.ndarray = initial_policy
        
        # History to store results for plotting
        self.history: Dict[str, List[float]] = {
            "average_rewards": [],
            "optimality_gaps": []
        }

    def _compute_gradient(self, d_pi: np.ndarray, Q_pi: np.ndarray) -> np.ndarray:
        """
        Computes the policy gradient ∂ρ/∂π(s,a) = d^π(s) Q^π(s,a).

        Args:
            d_pi (np.ndarray): The stationary distribution d^π of shape (S,).
            Q_pi (np.ndarray): The state-action value function Q^π of shape (S, A).

        Returns:
            np.ndarray: The policy gradient, a 2D NumPy array of shape (S, A).
        """
        if not isinstance(d_pi, np.ndarray) or d_pi.shape != (self.mdp.S,):
            raise ValueError(f"d_pi must be a numpy array of shape ({self.mdp.S},).")
        if not isinstance(Q_pi, np.ndarray) or Q_pi.shape != (self.mdp.S, self.mdp.A):
            raise ValueError(f"Q_pi must be a numpy array of shape ({self.mdp.S}, {self.mdp.A}).")
        
        # Reshape d_pi to (S, 1) for broadcasting across the action dimension
        gradient: np.ndarray = d_pi[:, np.newaxis] * Q_pi
        return gradient

    def _perform_policy_update(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates the current policy using the projected policy gradient update rule:
        π_{k+1} = Proj_Π[π_k + η * ∂ρ^π/∂π |_(π=π_k)].

        The update is performed in-place on `self.current_policy`.

        Args:
            gradient (np.ndarray): The computed policy gradient of shape (S, A).

        Returns:
            np.ndarray: The updated and projected policy of shape (S, A).
        """
        if not isinstance(gradient, np.ndarray) or gradient.shape != (self.mdp.S, self.mdp.A):
            raise ValueError(f"gradient must be a numpy array of shape ({self.mdp.S}, {self.mdp.A}).")

        # Calculate the unprojected next policy
        unprojected_policy: np.ndarray = self.current_policy + self.learning_rate * gradient
        
        new_policy: np.ndarray = np.zeros_like(self.current_policy)

        # Project each state's action probabilities onto the simplex
        for s in range(self.mdp.S):
            new_policy[s, :] = self.solver._project_onto_simplex(unprojected_policy[s, :])
        
        self.current_policy = new_policy
        return self.current_policy

    def run_iterations(self, num_iterations: int) -> Dict[str, List[float]]:
        """
        Executes the main PPG algorithm loop for a specified number of iterations.
        Records the average reward and optimality gap at each iteration.

        Args:
            num_iterations (int): The total number of iterations to run the algorithm.

        Returns:
            Dict[str, List[float]]: A dictionary containing lists of 'average_rewards'
                                     and 'optimality_gaps' over iterations.
        """
        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer.")
        if self.mdp.optimal_avg_reward is None:
            raise ValueError("Optimal average reward must be set in the MDP instance before running iterations.")

        for k in range(num_iterations):
            # 1. Obtain policy-dependent transition kernel P^π and reward vector r^π
            policy_P: np.ndarray = self.mdp.generate_policy_P(self.current_policy)
            policy_R_vec: np.ndarray = self.mdp.generate_policy_R(self.current_policy)

            # 2. Compute stationary distribution d^π
            d_pi: np.ndarray = self.solver.compute_stationary_distribution(policy_P)

            # 3. Compute current average reward ρ^π
            current_avg_reward: float = float(np.sum(d_pi * policy_R_vec))

            # 4. Compute projected relative value function v_φ^π
            v_phi: np.ndarray = self.solver.compute_projected_value_function(policy_P, policy_R_vec)

            # 5. Compute state-action value function Q^π
            Q_pi: np.ndarray = self.solver.compute_action_value_function(
                self.current_policy, v_phi, current_avg_reward
            )

            # 6. Compute the policy gradient
            gradient: np.ndarray = self._compute_gradient(d_pi, Q_pi)

            # 7. Update the current policy using projection
            self._perform_policy_update(gradient)

            # 8. Record history
            self.history["average_rewards"].append(current_avg_reward)
            optimality_gap: float = self.mdp.optimal_avg_reward - current_avg_reward
            self.history["optimality_gaps"].append(optimality_gap)

        return self.history

