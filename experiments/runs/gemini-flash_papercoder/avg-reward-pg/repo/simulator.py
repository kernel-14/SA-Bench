import numpy as np
from typing import Dict, List, Tuple
import logging

# Assuming config.py is in the same directory and contains the configuration loaded from config.yaml
import config
# Assuming mdp_definitions.py, mdp_solver.py, ppg_algorithm.py are in the same directory
from mdp_definitions import MDP
from mdp_solver import MDPSolver
from ppg_algorithm import PPGAlgorithm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Simulator:
    """
    Manages the setup and execution of the three experiments described in the paper.
    It generates MDPs with specific characteristics, runs the PPG algorithm on them,
    and collects the results.
    """

    def __init__(self, config_dict: Dict):
        """
        Initializes the Simulator with global configuration settings.

        Args:
            config_dict (Dict): A dictionary containing all configuration parameters
                                loaded from config.yaml.
        """
        self.config: Dict = config_dict
        # Initialize a reproducible random number generator
        self.rng: np.random.Generator = np.random.default_rng(self.config['general']['random_seed'])
        self.learning_rate: float = self.config['training']['learning_rate']
        logging.info("Simulator initialized with random seed %s and learning rate %s.",
                     self.config['general']['random_seed'], self.learning_rate)

    def _generate_mdp_exp1(self, s_val: int, a_val: int, reward_variance_level: str) -> MDP:
        """
        Generates an MDP instance for Experiment 1 based on state/action space sizes
        and a fixed reward variance level.

        Args:
            s_val (int): Number of states.
            a_val (int): Number of actions.
            reward_variance_level (str): Description of reward variance (e.g., "max variance").

        Returns:
            MDP: A configured MDP instance.
        """
        logging.info("Generating MDP for Exp1: S=%d, A=%d, Reward Variance: %s",
                     s_val, a_val, reward_variance_level)

        # Transition Kernel (P) construction as per Section C.1
        # P(s'|s,a) = (1 + 1/|S|) / 2 if s'=s
        # P(s'|s,a) = 1 / (2|S|) if s' != s
        transitions = np.zeros((s_val, a_val, s_val))
        for s in range(s_val):
            for a in range(a_val):
                for s_prime in range(s_val):
                    if s_prime == s:
                        transitions[s, a, s_prime] = (1 + 1 / s_val) / 2
                    else:
                        transitions[s, a, s_prime] = 1 / (2 * s_val)

        # Reward Function (R) construction as "max variance" per Section C.1 and C.2
        # For each state s, half of actions r(s,a)=1, other half r(s,a)=-1
        rewards = np.zeros((s_val, a_val))
        for s in range(s_val):
            action_indices = self.rng.permutation(a_val)
            half_a = a_val // 2
            rewards[s, action_indices[:half_a]] = 1.0
            rewards[s, action_indices[half_a:]] = -1.0
        
        return MDP(s_val, a_val, transitions, rewards)

    def _generate_mdp_exp2_kernel(self, s_val: int, a_val: int) -> np.ndarray:
        """
        Generates a single fixed random transition kernel P used across all
        reward variance levels in Experiment 2.

        Args:
            s_val (int): Number of states.
            a_val (int): Number of actions.

        Returns:
            np.ndarray: The generated transition probability kernel of shape (S, A, S).
        """
        logging.info("Generating fixed random transition kernel for Exp2: S=%d, A=%d",
                     s_val, a_val)
        transitions = np.zeros((s_val, a_val, s_val))
        for s in range(s_val):
            for a in range(a_val):
                # Use Dirichlet distribution for robust probability generation
                # Alpha parameter 1.0 creates a uniform distribution over the simplex
                transitions[s, a, :] = self.rng.dirichlet(np.ones(s_val))
        return transitions

    def _generate_mdp_exp2_rewards(self, s_val: int, a_val: int, reward_variance_level: str) -> np.ndarray:
        """
        Generates a reward function R for Experiment 2 based on the specified
        reward variance level.

        Args:
            s_val (int): Number of states.
            a_val (int): Number of actions.
            reward_variance_level (str): One of "no variance", "low variance",
                                         "high variance", "max variance".

        Returns:
            np.ndarray: The generated reward function of shape (S, A).
        """
        logging.info("Generating reward function for Exp2: S=%d, A=%d, Variance: %s",
                     s_val, a_val, reward_variance_level)
        rewards = np.zeros((s_val, a_val))
        s0 = 0  # Fixed special state as per Section C.2

        if reward_variance_level == "no variance":
            rewards[s0, :] = 1.0
        else:
            action_indices = self.rng.permutation(a_val)
            num_neg_rewards: int = 0
            if reward_variance_level == "low variance":
                num_neg_rewards = a_val // 8
            elif reward_variance_level == "high variance":
                num_neg_rewards = a_val // 4
            elif reward_variance_level == "max variance":
                num_neg_rewards = a_val // 2
            else:
                raise ValueError(f"Unknown reward variance level: {reward_variance_level}")
            
            rewards[s0, action_indices[:num_neg_rewards]] = -1.0
            rewards[s0, action_indices[num_neg_rewards:]] = 1.0
        
        return rewards

    def _generate_mdp_exp2(self, s_val: int, a_val: int, p_fixed: np.ndarray, reward_variance_level: str) -> MDP:
        """
        Combines a fixed transition kernel with a generated reward function to
        create an MDP instance for Experiment 2.

        Args:
            s_val (int): Number of states.
            a_val (int): Number of actions.
            p_fixed (np.ndarray): The pre-generated fixed transition kernel.
            reward_variance_level (str): The reward variance level for generating rewards.

        Returns:
            MDP: A configured MDP instance.
        """
        rewards = self._generate_mdp_exp2_rewards(s_val, a_val, reward_variance_level)
        return MDP(s_val, a_val, p_fixed, rewards)

    def _generate_mdp_exp3_rewards(self, s_val: int, a_val: int) -> np.ndarray:
        """
        Generates the fixed 'high variance' reward function R used across all
        transition types in Experiment 3. This reuses the logic from Experiment 2.

        Args:
            s_val (int): Number of states.
            a_val (int): Number of actions.

        Returns:
            np.ndarray: The generated reward function of shape (S, A).
        """
        reward_variance_level = self.config['mdp_config']['experiment3']['reward_variance_level']
        return self._generate_mdp_exp2_rewards(s_val, a_val, reward_variance_level)

    def _generate_mdp_exp3_transition_kernel(self, s_val: int, a_val: int, transition_type: str) -> np.ndarray:
        """
        Generates a transition kernel P for Experiment 3 based on the specified type.

        Args:
            s_val (int): Number of states.
            a_val (int): Number of actions.
            transition_type (str): One of "uniform", "non-uniform", "deterministic".

        Returns:
            np.ndarray: The generated transition probability kernel of shape (S, A, S).
        """
        logging.info("Generating transition kernel for Exp3: S=%d, A=%d, Type: %s",
                     s_val, a_val, transition_type)
        transitions = np.zeros((s_val, a_val, s_val))

        if transition_type == "uniform":
            # P(s'|s,a) = 1/|S| for all s, a, s'
            transitions.fill(1.0 / s_val)
        elif transition_type == "non-uniform":
            # P(i|s,i) = 1/(2S) + 1/2, P(i|s,j) = 1/(2S) for i!=j (Similar to Exp1)
            for s in range(s_val):
                for a in range(a_val):
                    for s_prime in range(s_val):
                        if s_prime == s:
                            transitions[s, a, s_prime] = (1 / (2 * s_val)) + (1 / 2)
                        else:
                            transitions[s, a, s_prime] = 1 / (2 * s_val)
        elif transition_type == "deterministic":
            # For each (s,a) pair, pick a random next state s' with probability 1.
            for s in range(s_val):
                for a in range(a_val):
                    # Each (s,a) pair leads deterministically to one random next state.
                    chosen_next_state = self.rng.integers(0, s_val)
                    transitions[s, a, chosen_next_state] = 1.0
        else:
            raise ValueError(f"Unknown transition kernel type: {transition_type}")
        
        return transitions

    def _generate_mdp_exp3(self, s_val: int, a_val: int, r_fixed: np.ndarray, transition_type: str) -> MDP:
        """
        Combines a fixed reward function with a generated transition kernel to
        create an MDP instance for Experiment 3.

        Args:
            s_val (int): Number of states.
            a_val (int): Number of actions.
            r_fixed (np.ndarray): The pre-generated fixed reward function.
            transition_type (str): The transition kernel type for generating P.

        Returns:
            MDP: A configured MDP instance.
        """
        transitions = self._generate_mdp_exp3_transition_kernel(s_val, a_val, transition_type)
        return MDP(s_val, a_val, transitions, r_fixed)

    def _run_single_ppg_instance(self, mdp: MDP, num_iterations: int) -> Dict[str, List[float]]:
        """
        A generic helper to run the PPG algorithm on a given MDP instance.

        Args:
            mdp (MDP): The MDP instance to run PPG on.
            num_iterations (int): The number of iterations for the PPG algorithm.

        Returns:
            Dict[str, List[float]]: The history of average rewards and optimality gaps.
        """
        # 1. Instantiate MDPSolver
        solver = MDPSolver(mdp)

        # 2. Compute Optimal Average Reward (ρ*)
        logging.info("  Computing optimal average reward (rho*)...")
        optimal_avg_reward = solver.find_optimal_average_reward()
        mdp.set_optimal_avg_reward(optimal_avg_reward)
        logging.info("  Optimal average reward (rho*): %f", optimal_avg_reward)

        # 3. Instantiate PPGAlgorithm (initial policy will be uniform random by default)
        ppg = PPGAlgorithm(mdp, solver, self.learning_rate)

        # 4. Run PPG iterations
        logging.info("  Running PPG for %d iterations...", num_iterations)
        history = ppg.run_iterations(num_iterations)
        logging.info("  PPG run completed.")
        return history

    def run_experiment1(self) -> Dict[Tuple[int, int], Dict[str, List[float]]]:
        """
        Orchestrates Experiment 1: generates MDPs of varying sizes, runs PPG,
        and collects results.

        Returns:
            Dict[Tuple[int, int], Dict[str, List[float]]]: Results for each (S, A) configuration.
        """
        logging.info("--- Starting Experiment 1: Convergence with Different State and Action Space Sizes ---")
        results: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
        
        mdp_sizes: List[Tuple[int, int]] = self.config['mdp_config']['experiment1']['mdp_sizes']
        reward_variance_level: str = self.config['mdp_config']['experiment1']['reward_variance_level']
        num_iterations: int = self.config['training']['num_iterations_exp1']

        for s_val, a_val in mdp_sizes:
            logging.info("Processing MDP (S=%d, A=%d)...", s_val, a_val)
            mdp = self._generate_mdp_exp1(s_val, a_val, reward_variance_level)
            history = self._run_single_ppg_instance(mdp, num_iterations)
            results[(s_val, a_val)] = history
        
        logging.info("--- Experiment 1 Completed ---")
        return results

    def run_experiment2(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Orchestrates Experiment 2: generates MDPs with fixed (S,A) and varying
        reward functions (variance), runs PPG, and collects results.

        Returns:
            Dict[str, Dict[str, List[float]]]: Results for each reward variance level.
        """
        logging.info("--- Starting Experiment 2: Convergence with Different Reward Functions (Variance) ---")
        results: Dict[str, Dict[str, List[float]]] = {}

        s_val: int = self.config['mdp_config']['experiment2']['fixed_S']
        a_val: int = self.config['mdp_config']['experiment2']['fixed_A']
        reward_variance_levels: List[str] = self.config['mdp_config']['experiment2']['reward_variance_levels']
        num_iterations: int = self.config['training']['num_iterations_exp2']

        # Generate fixed transition kernel once for all scenarios in Experiment 2
        p_fixed: np.ndarray = self._generate_mdp_exp2_kernel(s_val, a_val)

        for level in reward_variance_levels:
            logging.info("Processing MDP (S=%d, A=%d) with reward variance: %s",
                         s_val, a_val, level)
            mdp = self._generate_mdp_exp2(s_val, a_val, p_fixed, level)
            history = self._run_single_ppg_instance(mdp, num_iterations)
            results[level] = history
        
        logging.info("--- Experiment 2 Completed ---")
        return results

    def run_experiment3(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Orchestrates Experiment 3: generates MDPs with fixed (S,A) and reward function,
        and varying transition kernels, runs PPG, and collects results.

        Returns:
            Dict[str, Dict[str, List[float]]]: Results for each transition kernel type.
        """
        logging.info("--- Starting Experiment 3: Convergence with Different Transition Kernels (Cp) ---")
        results: Dict[str, Dict[str, List[float]]] = {}

        s_val: int = self.config['mdp_config']['experiment3']['fixed_S']
        a_val: int = self.config['mdp_config']['experiment3']['fixed_A']
        transition_types: List[str] = self.config['mdp_config']['experiment3']['transition_types']
        reward_variance_level: str = self.config['mdp_config']['experiment3']['reward_variance_level']
        num_iterations: int = self.config['training']['num_iterations_exp3']

        # Generate fixed reward function once for all scenarios in Experiment 3
        r_fixed: np.ndarray = self._generate_mdp_exp3_rewards(s_val, a_val)

        for tr_type in transition_types:
            logging.info("Processing MDP (S=%d, A=%d) with transition type: %s",
                         s_val, a_val, tr_type)
            mdp = self._generate_mdp_exp3(s_val, a_val, r_fixed, tr_type)
            history = self._run_single_ppg_instance(mdp, num_iterations)
            results[tr_type] = history
        
        logging.info("--- Experiment 3 Completed ---")
        return results

