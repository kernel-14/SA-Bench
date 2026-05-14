
import argparse

class RWMConfig:
    """
    Configuration for the Robotic World Model (RWM) and its training.
    Parameters are extracted from the paper, primarily from Table S10.
    """
    def __init__(self):
        self.step_time_seconds = 0.02 # Delta t
        self.max_iterations_rwm = 2500
        self.learning_rate_rwm = 1e-4
        self.weight_decay_rwm = 1e-5
        self.batch_size_rwm = 1024
        self.history_horizon_M = 32 # M
        self.forecast_horizon_N = 8 # N
        self.forecast_decay_alpha = 1.0 # alpha
        self.approx_training_hours_rwm = 1 # Not directly used in code, for info
        self.number_of_seeds = 5 # Not directly used in code, for info

        # Observation and action space dimensions (from Table S2, S3, S4)
        self.anymal_d_obs_dim = 45 # Base vel (3+3), gravity (3), joint pos (12), joint vel (12), joint torque (12)
        self.anymal_d_priv_info_dim = 8 # Knee contact (4), foot contact (4)
        self.anymal_d_action_dim = 12 # Joint position targets

        self.unitree_g1_obs_dim = 96 # Base vel (3+3), gravity (3), joint pos (29), joint vel (29), joint torque (29)
        self.unitree_g1_priv_info_dim = 30 # Body contact (26), foot height (2), foot velocity (2)
        self.unitree_g1_action_dim = 29 # Joint position targets

        # RWM Network Architecture (from Table S7)
        self.rwm_gru_hidden_size = 256
        self.rwm_mlp_head_hidden_size = 128
        self.rwm_mlp_head_activation = 'ReLU' # Implemented as nn.ReLU

        # Sigma values for velocity tracking rewards (common for both robots, from paper)
        self.sigma_v_xy = 0.25
        self.sigma_omega_z = 0.25

        # Reward weights (from Table S6)
        self.anymal_d_reward_weights = {
            'w_v_xy': 1.0, 'w_omega_z': 0.5, 'w_v_z': -2.0, 'w_omega_xy': -0.05,
            'w_q_tau': -2.5e-5, 'w_q_ddot': -2.5e-7, 'w_a_dot': -0.01, 'w_f_a': 0.5,
            'w_c': -1.0, 'w_g': -5.0, 'w_f_c': 0.0, 'w_q_d': 0.0
        }
        self.unitree_g1_reward_weights = {
            'w_v_xy': 1.0, 'w_omega_z': 0.5, 'w_v_z': -2.0, 'w_omega_xy': -0.05,
            'w_q_tau': -2.5e-5, 'w_q_ddot': -2.5e-7, 'w_a_dot': -0.05, 'w_f_a': 0.0,
            'w_c': -1.0, 'w_g': -5.0, 'w_f_c': 1.0, 'w_q_d': -1.0
        }
        # Default joint positions (q_0) - placeholder values, would be specific to robot
        self.anymal_d_q0 = [0.0] * self.anymal_d_action_dim # Assuming default is 0 for all joints
        self.unitree_g1_q0 = [0.0] * self.unitree_g1_action_dim # Assuming default is 0 for all joints

        # Observation, Privileged Info, and Policy Obs slicing indices (from Table S2, S3, S5)
        # ANYmal D RWM Obs
        self.anymal_d_rwm_obs_slices = {
            'v_xy': (0, 2), 'v_z': (2, 3), 'omega_xy': (3, 5), 'omega_z': (5, 6), # Base linear/angular velocities
            'g': (6, 9), # Projected gravity
            'q_pos': (9, 21), 'q_vel': (21, 33), 'tau': (33, 45) # Joint states
        }
        # ANYmal D RWM Privileged Info
        self.anymal_d_rwm_priv_info_slices = {
            'knee_contact': (0, 4), 'foot_contact': (4, 8)
        }
        # ANYmal D Policy Obs
        self.anymal_d_policy_obs_slices = {
            'v': (0, 3), 'omega': (3, 6), 'g': (6, 9), # Base velocities and gravity
            'c': (9, 12), # Velocity command
            'q_pos': (12, 24), 'q_vel': (24, 36), # Joint states
            'a_prev': (36, 48) # Last actions
        }

        # Unitree G1 RWM Obs
        self.unitree_g1_rwm_obs_slices = {
            'v_xy': (0, 2), 'v_z': (2, 3), 'omega_xy': (3, 5), 'omega_z': (5, 6), # Base linear/angular velocities
            'g': (6, 9), # Projected gravity
            'q_pos': (9, 38), 'q_vel': (38, 67), 'tau': (67, 96) # Joint states
        }
        # Unitree G1 RWM Privileged Info
        self.unitree_g1_rwm_priv_info_slices = {
            'body_contact': (0, 26), 'foot_height': (26, 28), 'foot_velocity': (28, 30)
        }
        # Unitree G1 Policy Obs
        self.unitree_g1_policy_obs_slices = {
            'v': (0, 3), 'omega': (3, 6), 'g': (6, 9), # Base velocities and gravity
            'c': (9, 12), # Velocity command
            'q_pos': (12, 41), 'q_vel': (41, 70), # Joint states
            'a_prev': (70, 99) # Last actions
        }

class MBPOPPOConfig:
    """
    Configuration for the Model-Based Policy Optimization with PPO (MBPO-PPO) and its training.
    Parameters are extracted from the paper, primarily from Table S11.
    """
    def __init__(self):
        self.imagination_environments = 4096
        self.imagination_steps_per_iteration = 100
        self.step_time_seconds = 0.02 # Delta t, same as RWM
        self.buffer_size = 1000 # |D|
        self.max_iterations_mbpo_ppo = 2500
        self.learning_rate_mbpo_ppo = 0.001
        self.weight_decay_mbpo_ppo = 0.0
        self.learning_epochs = 5
        self.mini_batches = 4
        self.kl_divergence_target = 0.01
        self.discount_factor_gamma = 0.99 # gamma
        self.clip_range_epsilon = 0.2 # epsilon
        self.entropy_coefficient = 0.005
        self.number_of_seeds = 5 # Not directly used in code, for info

        # Policy and Value Function Architecture (from Table S9)
        self.policy_mlp_hidden_shape = [128, 128, 128]
        self.value_mlp_hidden_shape = [128, 128, 128]
        self.policy_value_activation = 'ELU' # Implemented as nn.ELU

        # Policy Observation and Action Space dimensions (from Table S5 for ANYmal D)
        self.anymal_d_policy_obs_dim = 48 # Base vel (3+3), gravity (3), vel command (3), joint pos (12), joint vel (12), last actions (12)
        self.anymal_d_policy_action_dim = 12 # Joint position targets (same as RWM action space)

        # Unitree G1 Policy Observation and Action Space dimensions (from Table S5)
        self.unitree_g1_policy_obs_dim = 99 # Base vel (3+3), gravity (3), vel command (3), joint pos (29), joint vel (29), last actions (29)
        self.unitree_g1_policy_action_dim = 29 # Joint position targets (same as RWM action space)

class GlobalConfig:
    """
    Global configuration combining RWM and MBPO-PPO settings.
    """
    def __init__(self):
        self.rwm_config = RWMConfig()
        self.mbpo_ppo_config = MBPOPPOConfig()
        self.device = "cuda" # Assuming CUDA is available, otherwise "cpu"
        self.robot_type = "ANYmal D" # Can be "ANYmal D" or "Unitree G1"
        self.reward_weights = {} # Will be set based on robot_type
        self.q0 = [] # Default joint positions, will be set based on robot_type

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Robotic World Model and MBPO-PPO Configuration")
        # RWM arguments
        parser.add_argument('--rwm_lr', type=float, default=self.rwm_config.learning_rate_rwm,
                            help='Learning rate for RWM training.')
        parser.add_argument('--rwm_batch_size', type=int, default=self.rwm_config.batch_size_rwm,
                            help='Batch size for RWM training.')
        parser.add_argument('--history_horizon_M', type=int, default=self.rwm_config.history_horizon_M,
                            help='History horizon M for RWM.')
        parser.add_argument('--forecast_horizon_N', type=int, default=self.rwm_config.forecast_horizon_N,
                            help='Forecast horizon N for RWM.')
        parser.add_argument('--forecast_decay_alpha', type=float, default=self.rwm_config.forecast_decay_alpha,
                            help='Forecast decay alpha for RWM loss.')
        parser.add_argument('--max_iterations_rwm', type=int, default=self.rwm_config.max_iterations_rwm,
                            help='Max iterations for RWM training.')
        parser.add_argument('--rwm_gru_hidden_size', type=int, default=self.rwm_config.rwm_gru_hidden_size,
                            help='Hidden size of GRU in RWM base.')
        parser.add_argument('--rwm_mlp_head_hidden_size', type=int, default=self.rwm_config.rwm_mlp_head_hidden_size,
                            help='Hidden size of MLP heads in RWM.')

        # MBPO-PPO arguments
        parser.add_argument('--mbpo_ppo_lr', type=float, default=self.mbpo_ppo_config.learning_rate_mbpo_ppo,
                            help='Learning rate for MBPO-PPO policy training.')
        parser.add_argument('--imagination_envs', type=int, default=self.mbpo_ppo_config.imagination_environments,
                            help='Number of imagination environments.')
        parser.add_argument('--imagination_steps', type=int, default=self.mbpo_ppo_config.imagination_steps_per_iteration,
                            help='Imagination steps per iteration.')
        parser.add_argument('--buffer_size', type=int, default=self.mbpo_ppo_config.buffer_size,
                            help='Replay buffer size for MBPO-PPO.')
        parser.add_argument('--learning_epochs', type=int, default=self.mbpo_ppo_config.learning_epochs,
                            help='Learning epochs for PPO updates.')
        parser.add_argument('--mini_batches', type=int, default=self.mbpo_ppo_config.mini_batches,
                            help='Number of mini-batches for PPO updates.')
        parser.add_argument('--discount_factor', type=float, default=self.mbpo_ppo_config.discount_factor_gamma,
                            help='Discount factor gamma for MBPO-PPO.')
        parser.add_argument('--clip_range', type=float, default=self.mbpo_ppo_config.clip_range_epsilon,
                            help='Clip range epsilon for PPO.')
        parser.add_argument('--entropy_coeff', type=float, default=self.mbpo_ppo_config.entropy_coefficient,
                            help='Entropy coefficient for PPO.')
        parser.add_argument('--max_iterations_mbpo_ppo', type=int, default=self.mbpo_ppo_config.max_iterations_mbpo_ppo,
                            help='Max iterations for MBPO-PPO training.')

        parser.add_argument('--device', type=str, default=self.device,
                            help='Device to run training on (cuda or cpu).')
        parser.add_argument('--robot_type', type=str, default=self.robot_type,
                            choices=['ANYmal D', 'Unitree G1'],
                            help='Type of robot to configure dimensions for.')

        args = parser.parse_args()

        # Update RWM config from args
        self.rwm_config.learning_rate_rwm = args.rwm_lr
        self.rwm_config.batch_size_rwm = args.rwm_batch_size
        self.rwm_config.history_horizon_M = args.history_horizon_M
        self.rwm_config.forecast_horizon_N = args.forecast_horizon_N
        self.rwm_config.forecast_decay_alpha = args.forecast_decay_alpha
        self.rwm_config.max_iterations_rwm = args.max_iterations_rwm
        self.rwm_config.rwm_gru_hidden_size = args.rwm_gru_hidden_size
        self.rwm_config.rwm_mlp_head_hidden_size = args.rwm_mlp_head_hidden_size

        # Update MBPO-PPO config from args
        self.mbpo_ppo_config.learning_rate_mbpo_ppo = args.mbpo_ppo_lr
        self.mbpo_ppo_config.imagination_environments = args.imagination_envs
        self.mbpo_ppo_config.imagination_steps_per_iteration = args.imagination_steps
        self.mbpo_ppo_config.buffer_size = args.buffer_size
        self.mbpo_ppo_config.learning_epochs = args.learning_epochs
        self.mbpo_ppo_config.mini_batches = args.mini_batches
        self.mbpo_ppo_config.discount_factor_gamma = args.discount_factor
        self.mbpo_ppo_config.clip_range_epsilon = args.clip_range
        self.mbpo_ppo_config.entropy_coefficient = args.entropy_coeff
        self.mbpo_ppo_config.max_iterations_mbpo_ppo = args.max_iterations_mbpo_ppo

        self.device = args.device
        self.robot_type = args.robot_type

        # Set robot-specific dimensions
        if self.robot_type == "ANYmal D":
            self.rwm_obs_dim = self.rwm_config.anymal_d_obs_dim
            self.rwm_priv_info_dim = self.rwm_config.anymal_d_priv_info_dim
            self.rwm_action_dim = self.rwm_config.anymal_d_action_dim
            self.policy_obs_dim = self.mbpo_ppo_config.anymal_d_policy_obs_dim
            self.policy_action_dim = self.mbpo_ppo_config.anymal_d_policy_action_dim
            self.reward_weights = self.rwm_config.anymal_d_reward_weights
            self.q0 = torch.tensor(self.rwm_config.anymal_d_q0, device=self.device)
            self.rwm_obs_slices = self.rwm_config.anymal_d_rwm_obs_slices
            self.rwm_priv_info_slices = self.rwm_config.anymal_d_rwm_priv_info_slices
            self.policy_obs_slices = self.rwm_config.anymal_d_policy_obs_slices
        elif self.robot_type == "Unitree G1":
            self.rwm_obs_dim = self.rwm_config.unitree_g1_obs_dim
            self.rwm_priv_info_dim = self.rwm_config.unitree_g1_priv_info_dim
            self.rwm_action_dim = self.rwm_config.unitree_g1_action_dim
            self.policy_obs_dim = self.mbpo_ppo_config.unitree_g1_policy_obs_dim
            self.policy_action_dim = self.mbpo_ppo_config.unitree_g1_policy_action_dim
            self.reward_weights = self.rwm_config.unitree_g1_reward_weights
            self.q0 = torch.tensor(self.rwm_config.unitree_g1_q0, device=self.device)
            self.rwm_obs_slices = self.rwm_config.unitree_g1_rwm_obs_slices
            self.rwm_priv_info_slices = self.rwm_config.unitree_g1_rwm_priv_info_slices
            self.policy_obs_slices = self.rwm_config.unitree_g1_policy_obs_slices
        
        # Print final configuration for verification
        print("\n--- Final Configuration ---")
        print(f"Robot Type: {self.robot_type}")
        print(f"RWM Obs Dim: {self.rwm_obs_dim}")
        print(f"RWM Priv Info Dim: {self.rwm_priv_info_dim}")
        print(f"RWM Action Dim: {self.rwm_action_dim}")
        print(f"Policy Obs Dim: {self.policy_obs_dim}")
        print(f"Policy Action Dim: {self.policy_action_dim}")
        print(f"Reward Weights: {self.reward_weights}")
        print(f"Default Joint Positions (q0): {self.q0}")
        print(f"RWM Obs Slices: {self.rwm_obs_slices}")
        print(f"RWM Priv Info Slices: {self.rwm_priv_info_slices}")
        print(f"Policy Obs Slices: {self.policy_obs_slices}")
        print("---------------------------\n")

# Instantiate global config
cfg = GlobalConfig()

if __name__ == '__main__':
    # Example usage:
    cfg.parse_args()
    print("RWM Configuration:")
    print(f"  Learning Rate: {cfg.rwm_config.learning_rate_rwm}")
    print(f"  History Horizon (M): {cfg.rwm_config.history_horizon_M}")
    print(f"  Forecast Horizon (N): {cfg.rwm_config.forecast_horizon_N}")
    print("\nMBPO-PPO Configuration:")
    print(f"  Learning Rate: {cfg.mbpo_ppo_config.learning_rate_mbpo_ppo}")
    print(f"  Imagination Steps: {cfg.mbpo_ppo_config.imagination_steps_per_iteration}")
    print(f"  Device: {cfg.device}")
    print(f"  Robot Type: {cfg.robot_type}")
    print(f"  RWM Observation Dimension: {cfg.rwm_obs_dim}")
