
import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm

from config import cfg
from data import ReplayBuffer, make_env
from model import PGRModel
import random

def train():
    # Set seeds
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    if cfg.DEVICE == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Initialize environment
    env = make_env(cfg.ENV_NAME, cfg.OBS_TYPE, cfg.ACTION_REPEAT, cfg.SEED, cfg.DMC_TASK)
    eval_env = make_env(cfg.ENV_NAME, cfg.OBS_TYPE, cfg.ACTION_REPEAT, cfg.SEED + 1, cfg.DMC_TASK)

    obs_dim = env.observation_space.shape[0] if cfg.OBS_TYPE == "state" else env.observation_space.shape
    action_dim = env.action_space.shape[0]

    # Initialize PGR Model
    pgr_model = PGRModel(env.observation_space, env.action_space, cfg.DEVICE)
    agent = pgr_model.agent # The SAC/REDQ agent
    
    # Optimizers for policy and critic
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=cfg.POLICY_LR)
    critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=cfg.Q_LR)
    alpha_optimizer = torch.optim.Adam([agent.log_alpha], lr=cfg.POLICY_LR)

    # Replay Buffers
    real_replay_buffer = ReplayBuffer(cfg.BUFFER_SIZE, obs_dim, action_dim, cfg.DEVICE)
    synthetic_buffer_obs_shape = (cfg.CNN_ENCODER_OUT_DIM,) if cfg.OBS_TYPE == "pixel" else obs_dim
    synthetic_replay_buffer = ReplayBuffer(cfg.SYNTHETIC_BUFFER_SIZE, synthetic_buffer_obs_shape, action_dim, cfg.DEVICE)

    # TensorBoard Logger
    log_dir = os.path.join(cfg.LOG_DIR, f"{cfg.ENV_NAME}_{cfg.DMC_TASK}_{cfg.RELEVANCE_FUNCTION}_{int(time.time())}")
    writer = SummaryWriter(log_dir)

    print("Starting training...")
    
    obs, info = env.reset(seed=cfg.SEED)
    episode_reward = 0
    episode_steps = 0
    done = False
    
    # Main training loop
    for global_step in tqdm(range(1, cfg.TOTAL_ENV_STEPS + 1)):
        # 1. Collect transitions
        if global_step <= cfg.BATCH_SIZE * cfg.UTD_RATIO: # Initial exploration phase
            action = env.action_space.sample()
        else:
            action = pgr_model.get_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        real_replay_buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_reward += reward
        episode_steps += 1

        if done:
            writer.add_scalar("train/episode_reward", episode_reward, global_step)
            writer.add_scalar("train/episode_steps", episode_steps, global_step)
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0

        # Ensure enough samples in buffer before training
        if real_replay_buffer.size < cfg.BATCH_SIZE * cfg.UTD_RATIO:
            continue

        # 2. Update Relevance Function (if learnable)
        # "updated for only 5% of all policy gradient steps."
        # This implies updating it within the inner gradient loop, not every INNER_LOOP_FREQUENCY.
        # However, for consistency with original placement, let's keep it here for now.
        # A more faithful implementation might integrate this into the gradient steps.
        # For relevance functions like ICM, RND, ECO that have an update method:
        if hasattr(pgr_model.relevance_function, 'update') and random.random() < cfg.ICM_UPDATE_RATIO:
            real_transitions_sample = real_replay_buffer.sample(cfg.BATCH_SIZE)
            relevance_loss = pgr_model.relevance_function.update(real_transitions_sample)
            writer.add_scalar("losses/relevance_function_loss", relevance_loss, global_step)


        # 3. Periodically update generative model and generate synthetic data
        if global_step % cfg.INNER_LOOP_FREQUENCY == 0:
            print(f"Updating generative model and generating synthetic data at step {global_step}")
            
            # Sample real transitions for training generative model
            all_real_transitions = real_replay_buffer.get_all_transitions()
            
            # Calculate relevance scores for all real transitions
            # This is crucial for "prompting" the diffusion model
            all_relevance_scores = pgr_model.calculate_relevance_scores(all_real_transitions)

            # Sort transitions by relevance and select top-k conditions
            # The paper says "choose some ratio k of the transitions in the real replay buffer
            # with the highest values for F(s, a, s', r), and sample their conditioning values randomly to pass to G"
            # Here we take the top `generation_batch_size` scores as conditions
            # Ensure we don't try to get more conditions than available
            num_conditions_to_sample = min(cfg.GENERATION_BATCH_SIZE, len(all_relevance_scores))
            sorted_indices = torch.argsort(all_relevance_scores, descending=True)
            top_k_indices = sorted_indices[:num_conditions_to_sample]
            conditions_for_generation = all_relevance_scores[top_k_indices]


            for _ in range(cfg.DIFF_TRAIN_EPOCHS): # Train diffusion model for a few epochs
                # Sample batch for diffusion training
                diff_train_idxs = np.random.randint(0, all_real_transitions["observations"].shape[0], size=cfg.BATCH_SIZE)
                diff_train_transitions = {k: v[diff_train_idxs] for k, v in all_real_transitions.items()}
                
                # Get relevance scores for this batch for diffusion training
                # These scores are used to condition the diffusion model during its training
                diff_train_relevance_scores = pgr_model.calculate_relevance_scores(diff_train_transitions)
                
                gen_loss = pgr_model.train_generative_model(diff_train_transitions)
                writer.add_scalar("losses/generative_model_loss", gen_loss, global_step)
            
            # Reinitialize synthetic buffer and re-generate
            # This ensures the synthetic buffer is always filled with fresh, relevant data
            print(f"Generating {cfg.SYNTHETIC_BUFFER_SIZE} synthetic transitions...")
            synthetic_replay_buffer = ReplayBuffer(cfg.SYNTHETIC_BUFFER_SIZE, synthetic_buffer_obs_shape, action_dim, cfg.DEVICE) # Reset
            num_generated = 0
            while num_generated < cfg.SYNTHETIC_BUFFER_SIZE:
                batch_to_generate = min(cfg.GENERATION_BATCH_SIZE, cfg.SYNTHETIC_BUFFER_SIZE - num_generated)
                
                # Sample conditions from the top relevant real transitions
                # If conditions_for_generation is empty, this will cause an error. Handle that.
                if conditions_for_generation.shape[0] == 0:
                    print("Warning: No conditions available for synthetic data generation. Skipping.")
                    break
                
                rand_indices = torch.randint(0, conditions_for_generation.shape[0], (batch_to_generate,), device=cfg.DEVICE)
                sampled_conds = conditions_for_generation[rand_indices]

                synthetic_transitions_batch = pgr_model.generate_synthetic_transitions_from_conditions(
                    sampled_conds, batch_to_generate
                )
                
                # Add generated transitions to synthetic buffer
                for i in range(batch_to_generate):
                    # Ensure numpy arrays for replay buffer
                    obs_syn = synthetic_transitions_batch["observations"][i].cpu().numpy()
                    action_syn = synthetic_transitions_batch["actions"][i].cpu().numpy()
                    reward_syn = synthetic_transitions_batch["rewards"][i].cpu().numpy()
                    next_obs_syn = synthetic_transitions_batch["next_observations"][i].cpu().numpy()
                    done_syn = synthetic_transitions_batch["dones"][i].cpu().numpy()
                    
                    synthetic_replay_buffer.add(obs_syn, action_syn, reward_syn, next_obs_syn, done_syn)
                num_generated += batch_to_generate
            print("Synthetic data generation complete.")

        # 4. Train policy on samples from D_real U D_syn
        if real_replay_buffer.size > cfg.BATCH_SIZE * cfg.UTD_RATIO:
            for _ in range(cfg.GRADIENT_STEPS):
                # Sample from real buffer
                real_batch = real_replay_buffer.sample(int(cfg.BATCH_SIZE * (1 - cfg.SYNTHETIC_DATA_RATIO)))
                
                # Sample from synthetic buffer (if available)
                if synthetic_replay_buffer.size > 0:
                    synthetic_batch = synthetic_replay_buffer.sample(int(cfg.BATCH_SIZE * cfg.SYNTHETIC_DATA_RATIO))
                    
                    # Combine batches
                    combined_batch = {}
                    for k in real_batch.keys():
                        combined_batch[k] = torch.cat([real_batch[k], synthetic_batch[k]], dim=0)
                else:
                    combined_batch = real_batch

                # Policy Training (SAC / REDQ)
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = (
                    combined_batch["observations"],
                    combined_batch["actions"],
                    combined_batch["rewards"],
                    combined_batch["next_observations"],
                    combined_batch["dones"]
                )

                # If pixel-based, encode observations for policy training
                if cfg.OBS_TYPE == "pixel":
                    obs_batch = pgr_model.agent.encoder(obs_batch)
                    next_obs_batch = pgr_model.agent.encoder(next_obs_batch)

                # Actor Loss
                pi, log_pi = agent.actor(obs_batch)
                q_pi = agent.critic.q_min(obs_batch, pi)
                actor_loss = (agent.log_alpha.exp() * log_pi - q_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Alpha Loss (for entropy tuning)
                alpha_loss = (-agent.log_alpha * (log_pi + agent.target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                # Critic Loss
                with torch.no_grad():
                    next_pi, next_log_pi = agent.actor(next_obs_batch)
                    target_q = agent.critic_target.q_min(next_obs_batch, next_pi)
                    target_q = reward_batch + (1 - done_batch) * cfg.DISCOUNT * (target_q - agent.log_alpha.exp() * next_log_pi)
                
                current_q_values = agent.critic(obs_batch, action_batch) # All Q-nets
                critic_loss = F.mse_loss(current_q_values, target_q.expand_as(current_q_values))

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Update target critic networks
                agent.update_critic_target()

                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                writer.add_scalar("metrics/alpha", agent.log_alpha.exp().item(), global_step)
        
        # Evaluation
        if global_step % cfg.EVAL_INTERVAL == 0:
            avg_reward = evaluate_policy(eval_env, pgr_model, cfg.EVAL_EPISODES)
            writer.add_scalar("eval/average_reward", avg_reward, global_step)
            print(f"Global Step: {global_step}, Average Eval Reward: {avg_reward:.2f}")

        # Save model
        if global_step % cfg.SAVE_MODEL_INTERVAL == 0:
            torch.save(pgr_model.state_dict(), os.path.join(log_dir, f"pgr_model_{global_step}.pth"))

    writer.close()
    env.close()
    eval_env.close()

def evaluate_policy(env, pgr_model: PGRModel, num_episodes: int) -> float:
    avg_reward = 0.
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.
        while not done:
            action = pgr_model.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        avg_reward += episode_reward
    avg_reward /= num_episodes
    return avg_reward

if __name__ == "__main__":
    train()
