
import numpy as np

# MDP Configuration
class MDPConfig:
    def __init__(self, S: int, A: int, name: str = ""):
        self.S = S  # Number of states
        self.A = A  # Number of actions
        self.name = name

# Policy Gradient Configuration
class PGConfig:
    def __init__(self,
                 learning_rate: float,
                 iterations: int,
                 initial_policy_type: str = "uniform", # or "random"
                 seed: int = 42):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.initial_policy_type = initial_policy_type
        self.seed = seed

# Simulation Configurations
class SimulationConfig:
    def __init__(self,
                 mdp_configs: list[MDPConfig],
                 pg_config: PGConfig,
                 plot_dir: str = "plots"):
        self.mdp_configs = mdp_configs
        self.pg_config = pg_config
        self.plot_dir = plot_dir

# Default configurations (can be overridden)

# Section 4.1: Convergence with different action and state space size
# MDPs with (S, A) = {(3,3), (9,9), (81,81)}
mdp_config_s3a3 = MDPConfig(S=3, A=3, name="S3A3")
mdp_config_s9a9 = MDPConfig(S=9, A=9, name="S9A9")
mdp_config_s81a81 = MDPConfig(S=81, A=81, name="S81A81")

pg_config_s41 = PGConfig(learning_rate=0.01, iterations=2000)

sim_config_s41 = SimulationConfig(
    mdp_configs=[mdp_config_s3a3, mdp_config_s9a9, mdp_config_s81a81],
    pg_config=pg_config_s41,
    plot_dir="plots/section_4_1"
)

# Section 4.2: Convergence with different reward functions
# MDP with (S, A) = (16, 16)
mdp_config_s16a16 = MDPConfig(S=16, A=16, name="S16A16")

pg_config_s42 = PGConfig(learning_rate=0.01, iterations=2000)

# Reward variance types
REWARD_VARIANTS = ["no_variance", "low_variance", "high_variance", "max_variance"]

sim_config_s42 = SimulationConfig(
    mdp_configs=[mdp_config_s16a16], # will be used with different reward functions
    pg_config=pg_config_s42,
    plot_dir="plots/section_4_2"
)

# Section 4.3: Convergence with different transition kernels
# MDP with (S, A) = (16, 16)
pg_config_s43 = PGConfig(learning_rate=0.01, iterations=3000)

TRANSITION_VARIANTS = ["uniform", "non_uniform", "deterministic"]

sim_config_s43 = SimulationConfig(
    mdp_configs=[mdp_config_s16a16], # will be used with different transition kernels
    pg_config=pg_config_s43,
    plot_dir="plots/section_4_3"
)


# Constants capturing MDP Complexity (from Table 1 and 2)
# These are theoretical bounds and might not be used directly in simulation,
# but are important for understanding the theoretical results.
# The actual values depend on the specific MDP instance.
class MDPConstants:
    def __init__(self,
                 Ce: float = None,  # For geometric ergodicity: ||(P^pi)^k - 1(d^pi)^T||_inf <= Ce * lambda^k
                 lambda_val: float = None, # Mixing coefficient
                 k1: float = None, # Numeric constant for L_1^II bound
                 k2: float = None  # Numeric constant for L_2^II bound
                 ):
        self.Ce = Ce
        self.lambda_val = lambda_val
        self.k1 = k1
        self.k2 = k2
