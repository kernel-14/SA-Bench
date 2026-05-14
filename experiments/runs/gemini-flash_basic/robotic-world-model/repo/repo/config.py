# This file centralizes configuration parameters for RWM and MBPO-PPO.
# Based on Tables S10 and S11.

class RWMConfig:
    def __init__(self):
        # Table S10: RWM training parameters
        self.step_time_seconds = 0.02 # ∆t
        self.max_iterations = 2500
        self.learning_rate = 1e-4 # √
        self.weight_decay = 1e-5
        self.batch_size = 1024
        self.history_horizon = 32 # M
        self.forecast_horizon = 8 # N
        self.forecast_decay_alpha = 1.0 # α
        self.approximate_training_hours = 1
        self.number_of_seeds = 5

        # RWM Architecture (from Table S7, also used in rwm/model.py)
        self.rwm_gru_hidden_shape = [256, 256]
        self.rwm_mlp_hidden_shape = 128
        self.rwm_mlp_activation = 'ReLU'


class MBPPOConfig:
    def __init__(self):
        # Table S11: MBPO-PPO training parameters (conceptual, as full table not provided)
        # The paper implies parameters like num_epochs, learning_rate, batch_size for PPO.
        # We will use general values or infer from common PPO setups where not explicit.

        self.ppo_epochs = 10 # Common PPO hyperparameter
        self.ppo_clip_param = 0.2 # Common PPO hyperparameter
        self.ppo_value_coeff = 0.5 # Common PPO hyperparameter
        self.ppo_entropy_coeff = 0.01 # Common PPO hyperparameter
        self.ppo_learning_rate = 3e-4 # Common PPO learning rate for actor/critic
        self.ppo_batch_size = 64 # Batch size for PPO updates
        self.gamma = 0.99 # Discount factor
        self.gae_lambda = 0.95 # GAE lambda parameter

        # From Algorithm 1, Step 6: imagination_horizon_T
        self.imagination_horizon_T = 100 # Implied by paper for robust policy optimization

        # Replay buffer capacity (not explicitly in tables but needed for ReplayBuffer class)
        self.replay_buffer_capacity = 1_000_000 # Large capacity for RL

        # Policy and Value Function Architecture (from Table S9, also used in policy/model.py)
        self.policy_mlp_hidden_shape = [128, 128, 128]
        self.policy_mlp_activation = 'ELU'
        self.value_mlp_hidden_shape = [128, 128, 128]
        self.value_mlp_activation = 'ELU'

class GlobalConfig:
    def __init__(self):
        self.rwm_config = RWMConfig()
        self.mbppo_config = MBPPOConfig()

        # General Environment parameters (e.g., robot type for reward functions)
        self.robot_type = 'anymal_d' # Default robot type

        # Dummy dimensions for initialization (these would be dynamically set in a real env setup)
        self.observation_dim_anymal_wm = 45  # Sum from env/spaces.py -> ObservationSpaces().world_model_anymal_d_dim
        self.action_dim_anymal = 12 # Sum from env/spaces.py -> ActionSpaces().anymal_d_dim
        self.privileged_dim_anymal = 8 # Sum from env/spaces.py -> PrivilegedInfoSpaces().anymal_d_dim

        self.observation_dim_anymal_policy = 48 # Sum from env/spaces.py -> ObservationSpaces().policy_anymal_d_dim

        self.observation_dim_unitree_wm = 96
        self.action_dim_unitree = 29
        self.privileged_dim_unitree = 30

        self.observation_dim_unitree_policy = 99

