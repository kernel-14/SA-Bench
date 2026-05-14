import gym
import numpy as np
import torch
from torchvision import transforms

def preprocess_observation(observation, env_name):
    if "Atari" in env_name:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(0, 255)
        ])
        return transform(observation)
    elif "DMC" in env_name:
        return torch.tensor(observation).float() / 255.0
    else:
        return torch.tensor(observation).float()

def load_environment(env_name):
    env = gym.make(env_name)
    return env

def sample_batch(env, batch_size):
    states, actions, rewards, next_states = [], [], [], []
    for _ in range(batch_size):
        state = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        states.append(preprocess_observation(state, env_name))
        actions.append(action)
        rewards.append(reward)
        next_states.append(preprocess_observation(next_state, env_name))
        if done:
            env.reset()
    return torch.stack(states), torch.tensor(actions), torch.tensor(rewards), torch.stack(next_states)