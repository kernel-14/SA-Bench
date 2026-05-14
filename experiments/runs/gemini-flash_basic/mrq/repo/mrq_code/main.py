import torch
import numpy as np
import gymnasium as gym # Using gymnasium as it's the modern standard

from mrq_code.agent import MRQAgent
from mrq_code.config import MRQConfig

def main():
    # Configuration
    seed = 0
    env_name = "CartPole-v1" # Example environment
    # env_name = "Pendulum-v1" # Example continuous environment
    total_timesteps = 100000
    log_interval = 1000

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = gym.make(env_name)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Determine observation and action space properties
    is_discrete_action_space = isinstance(env.action_space, gym.spaces.Discrete)
    if is_discrete_action_space:
        action_dim = env.action_space.n
        print(f"Action Space: Discrete, Dim: {action_dim}")
    else:
        action_dim = env.action_space.shape[0]
        print(f"Action Space: Continuous, Dim: {action_dim}")
    
    # Assuming state is a vector for CartPole/Pendulum, adjust if using image-based envs
    image_observation_space = False
    state_dim = env.observation_space.shape[0]
    print(f"Observation Space: Vector, Dim: {state_dim}")

    # Initialize MR.Q Agent
    agent = MRQAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        is_discrete_action_space=is_discrete_action_space,
        image_observation_space=image_observation_space,
        device=device
    )

    print("Agent initialized.")

    # Training loop
    obs, info = env.reset(seed=seed)
    episode_reward = 0
    episode_steps = 0
    episode_num = 0

    for t in range(total_timesteps):
        # Select action
        if t < MRQConfig.INITIAL_RANDOM_EXPLORATION_TIME_STEPS:
            action = env.action_space.sample() # Random action for initial exploration
        else:
            action = agent.act(obs) # Agent selects action
        
        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store experience
        agent.remember(obs, action, reward, next_obs, done)
        
        obs = next_obs
        episode_reward += reward
        episode_steps += 1

        # Update agent
        if t >= MRQConfig.INITIAL_RANDOM_EXPLORATION_TIME_STEPS:
            agent.update()

        if done:
            episode_num += 1
            if episode_num % log_interval == 0:
                print(f"Episode: {episode_num}, Total Steps: {t+1}, Reward: {episode_reward}, Steps: {episode_steps}")
            
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0

    env.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
