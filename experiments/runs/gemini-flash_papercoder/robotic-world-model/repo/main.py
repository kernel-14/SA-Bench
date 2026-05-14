# main.py

# Imports
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np # Required for mock actions in pretraining

# Assuming these modules are in the same directory or accessible via Python path
import config as config_module
import utils
from environment import Environment
from data.replay_buffer import ReplayBuffer
from models.rwm_model import RWMModel
from models.policy_value_model import PolicyModel, ValueModel
from trainers.rwm_trainer import RWMTrainer
from trainers.mbpo_ppo_trainer import MBPOPPO_Trainer
from evaluation import Evaluator

def main():
    """
    This is the main entry point of the entire reproduction system.
    It orchestrates the loading of configurations, initialization of components,
    RWM pretraining, MBPO-PPO policy optimization, and evaluation phases.
    """

    # 1. Configuration Loading
    # Load settings from config.yaml using the Config class.
    # All hyperparameters and settings will be accessed through this config object.
    config = config_module.Config(config_path="config.yaml")

    # Determine the device for PyTorch operations
    device = torch.device(config.global.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories for logging and saving models
    log_dir_base = os.path.join(config.global.log_dir, config.environment.robot_type)
    model_dir_base = os.path.join(config.global.model_dir, config.environment.robot_type)
    os.makedirs(log_dir_base, exist_ok=True)
    os.makedirs(model_dir_base, exist_ok=True)

    # Loop over multiple seeds for robust evaluation as specified in config.global.num_seeds
    for seed_idx in range(config.global.num_seeds):
        current_seed = config.global.seed + seed_idx
        print(f"\n--- Starting run for seed: {current_seed} ---")

        # Set up random seeds for reproducibility
        utils.set_seed(current_seed)

        # Initialize TensorBoard writer for the current seed
        seed_log_dir = os.path.join(log_dir_base, f"seed_{current_seed}")
        writer = SummaryWriter(seed_log_dir)

        # 2. Environment Initialization
        # Initialize the Isaac Lab environment wrapper.
        # This wrapper handles interactions with the simulator and provides observation/action space dimensions.
        real_env = Environment(robot_type=config.environment.robot_type, config=config)
        
        # Retrieve environment dimensions for model initialization
        env_dims = real_env.get_obs_dims()
        obs_wm_dim = env_dims['obs_wm_dim']
        act_wm_dim = env_dims['action_dim'] # Action dimension for WM input is the same as policy action dim
        priv_dim = env_dims['priv_dim']
        obs_policy_dim = env_dims['obs_policy_dim']
        action_dim = env_dims['action_dim']

        # 3. Replay Buffer Initialization
        # The replay buffer stores real environment transitions for RWM training and policy imagination initialization.
        replay_buffer = ReplayBuffer(capacity=config.mbpo_ppo.training.replay_buffer_size, config=config)

        # 4. Model Initialization
        # Initialize the RWM, Policy, and Value Function models based on configurations.
        rwm_model = RWMModel(obs_wm_dim, act_wm_dim, priv_dim, config).to(device)
        policy_model = PolicyModel(obs_policy_dim, action_dim, config).to(device)
        value_model = ValueModel(obs_policy_dim, config).to(device)

        # 5. Optimizer Initialization
        # Initialize optimizers for each model.
        rwm_optimizer = torch.optim.Adam(
            rwm_model.parameters(),
            lr=config.rwm_model.training.learning_rate,
            weight_decay=config.rwm_model.training.weight_decay
        )
        policy_optimizer = torch.optim.Adam(
            policy_model.parameters(),
            lr=config.mbpo_ppo.training.learning_rate,
            weight_decay=config.mbpo_ppo.training.weight_decay
        )
        value_optimizer = torch.optim.Adam(
            value_model.parameters(),
            lr=config.mbpo_ppo.training.learning_rate,
            weight_decay=config.mbpo_ppo.training.weight_decay
        )

        # 6. Trainer and Evaluator Initialization
        # Initialize the trainers for RWM and MBPO-PPO, and the evaluator.
        rwm_trainer = RWMTrainer(rwm_model, rwm_optimizer, replay_buffer, config, writer, device)
        mbpo_ppo_trainer = MBPOPPO_Trainer(
            policy_model, value_model, rwm_model,
            policy_optimizer, value_optimizer,
            real_env, replay_buffer, config, rwm_trainer, writer, device
        )
        evaluator = Evaluator(rwm_model, policy_model, real_env, config, writer, device)

        # --- RWM Pretraining Phase ---
        print("\n--- Starting RWM Pretraining Phase ---")

        # Collect initial data for RWM pretraining (6M state transitions as per config).
        # This uses a simple random action policy for initial data generation.
        print(f"Collecting {config.environment.pretraining_data_collection_steps} steps for RWM pretraining...")
        current_obs_wm, current_obs_policy, current_priv_info, current_command_vel = real_env.reset()
        last_action_for_buffer = np.zeros(action_dim, dtype=np.float32) # For the very first step, no 'last_action' yet

        for i in range(config.environment.pretraining_data_collection_steps):
            # For pretraining, we need diverse data. A simple random action policy is used here.
            action = real_env.get_random_action()
            
            # Step the environment
            next_obs_wm, next_obs_policy, next_priv_info, env_reward, done, info = real_env.step(action)
            
            # Add transition to replay buffer.
            # act_wm refers to the action taken AT current_obs_wm to reach next_obs_wm
            # act_policy refers to the 'last_actions' component of current_obs_policy
            replay_buffer.add_transition(
                obs_wm=current_obs_wm,
                act_wm=action, # Action that generated next_obs_wm from current_obs_wm
                next_obs_wm=next_obs_wm,
                priv_info=current_priv_info,
                next_priv_info=next_priv_info,
                obs_policy=current_obs_policy,
                act_policy=last_action_for_buffer, # Action taken one step before current_obs_policy
                next_obs_policy=next_obs_policy,
                reward=env_reward,
                done=done,
                command_vel=current_command_vel
            )
            
            # Update current state for next iteration
            current_obs_wm = next_obs_wm
            current_obs_policy = next_obs_policy
            current_priv_info = next_priv_info
            current_command_vel = info.get('command_vel', current_command_vel) # Update command_vel if changed by env.reset
            last_action_for_buffer = action.copy() # The action taken in this step becomes 'last_action' for next_obs_policy input

            if done:
                current_obs_wm, current_obs_policy, current_priv_info, current_command_vel = real_env.reset()
                last_action_for_buffer = np.zeros(action_dim, dtype=np.float32) # Reset last action after episode end
            
            if (i + 1) % 100000 == 0:
                print(f"Collected {i + 1}/{config.environment.pretraining_data_collection_steps} transitions for RWM pretraining. Buffer size: {replay_buffer.size()}")
        print("RWM pretraining data collection complete.")

        # Train the RWM using the collected data.
        rwm_trainer.pretrain_rwm(num_iterations=config.rwm_model.training.max_iterations)
        print("RWM pretraining complete.")

        # Evaluate the pretrained RWM's autoregressive prediction accuracy and robustness.
        # This corresponds to evaluations in Section 4.1 and 4.2 of the paper.
        print("\n--- Evaluating pretrained RWM ---")
        eval_results_ar = evaluator.evaluate_rwm_autoregressive(
            num_trajectories=config.global.num_eval_trajectories,
            forecast_steps_max=config.rwm_model.training.history_horizon_M + config.rwm_model.training.forecast_horizon_N * 2 # Evaluate beyond training horizon N
        )
        print(f"RWM Autoregressive Evaluation Results (Seed {current_seed}): {eval_results_ar}")
        
        eval_results_robustness = evaluator.evaluate_rwm_robustness(
            noise_levels=[0.0, 0.01, 0.05, 0.1], # These levels could be defined in config.yaml
            num_trajectories=config.global.num_eval_trajectories
        )
        print(f"RWM Robustness Evaluation Results (Seed {current_seed}): {eval_results_robustness}")

        # --- MBPO-PPO Training Phase ---
        print("\n--- Starting MBPO-PPO Policy Optimization Phase ---")
        # Start the main MBPO-PPO training loop.
        mbpo_ppo_trainer.run_training_loop(total_iterations=config.mbpo_ppo.training.max_iterations, evaluator=evaluator)
        print("MBPO-PPO policy optimization complete.")

        # --- Final Evaluation ---
        print("\n--- Starting Final Policy Evaluation ---")
        # Evaluate the final policy in the real environment (simulator).
        final_avg_reward, final_avg_episode_len = evaluator.evaluate_policy_in_env(
            num_episodes=config.global.num_eval_episodes,
            render=False # Set to True for visualization if desired
        )
        print(f"Final Policy Evaluation (Seed {current_seed}): Avg Reward = {final_avg_reward:.2f}, Avg Episode Length = {final_avg_episode_len:.2f}")
        writer.add_scalar("Final_Eval/Average_Reward", final_avg_reward, current_seed)
        writer.add_scalar("Final_Eval/Average_Episode_Length", final_avg_episode_len, current_seed)


        # Save final models for this seed
        seed_model_dir = os.path.join(model_dir_base, f"seed_{current_seed}")
        os.makedirs(seed_model_dir, exist_ok=True)
        torch.save(policy_model.state_dict(), os.path.join(seed_model_dir, f"policy_model_final.pt"))
        torch.save(rwm_model.state_dict(), os.path.join(seed_model_dir, f"rwm_model_final.pt"))
        print(f"Final models saved for seed {current_seed} in {seed_model_dir}.")

        writer.close()
        real_env.close() # Close the environment at the end of each seed's run

    print("\n--- All training and evaluation phases complete ---")

if __name__ == "__main__":
    main()

