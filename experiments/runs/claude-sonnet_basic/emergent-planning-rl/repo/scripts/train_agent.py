"""
Script to train the DRC(3,3) agent on Sokoban.

From the paper (Appendix E.4):
- DRC(3,3) with 32 channels
- Trained on 900k levels from Boxoban unfiltered training set
- 250 million transitions using IMPALA
- Discount rate: gamma = 0.97
- V-trace lambda = 0.97
- Adam optimizer, lr decays from 4e-4 to 0
- Batch size: 16
- Unroll length: 20

Usage:
    python scripts/train_agent.py --data_dir /path/to/boxoban --output_dir /path/to/checkpoints
"""

import argparse
import os
import sys
import torch
import numpy as np
import random
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent.drc_agent import DRCAgent
from environment.sokoban import SokobanEnv
from environment.boxoban_loader import BoxobanLoader
from training.impala_trainer import create_trainer, IMPALALoss
import torch.nn.functional as F


def collect_rollout(
    agent: DRCAgent,
    envs: List[SokobanEnv],
    levels_pool: List[np.ndarray],
    hidden_states_list: List,
    device: torch.device,
    unroll_length: int = 20,
) -> dict:
    """
    Collect a rollout from multiple environments.
    
    Args:
        agent: DRC agent
        envs: List of Sokoban environments
        levels_pool: Pool of levels to sample from
        hidden_states_list: Hidden states for each environment
        device: Device
        unroll_length: Number of steps to collect
        
    Returns:
        Dict with rollout data
    """
    B = len(envs)
    
    observations = []
    actions = []
    rewards = []
    dones = []
    behavior_log_probs = []
    
    # Get initial observations
    current_obs = []
    for env in envs:
        if env.done or env.grid is None:
            level = random.choice(levels_pool)
            obs = env.reset(level)
        else:
            obs = env._get_observation()
        current_obs.append(obs)
    
    obs_tensor = torch.FloatTensor(np.stack(current_obs)).to(device)
    observations.append(obs_tensor)
    
    for t in range(unroll_length):
        with torch.no_grad():
            logits, values, hidden_states_list_new, _ = agent.forward(
                obs_tensor, 
                hidden_states_list if t == 0 else hidden_states_list,
            )
        
        # Sample actions
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        action_tensor = torch.multinomial(probs, 1).squeeze(-1)
        action_log_probs = log_probs.gather(-1, action_tensor.unsqueeze(-1)).squeeze(-1)
        
        # Step environments
        step_obs = []
        step_rewards = []
        step_dones = []
        
        for i, (env, action) in enumerate(zip(envs, action_tensor.cpu().numpy())):
            obs, reward, done, info = env.step(int(action))
            
            if done:
                # Reset environment
                level = random.choice(levels_pool)
                obs = env.reset(level)
                # Reset hidden state for this environment
                h, c = agent.convlstm_cells[0].init_hidden(1, device)
                hidden_states_list_new = list(hidden_states_list_new)
                for d in range(agent.D):
                    h_d, c_d = hidden_states_list_new[d]
                    h_d[i] = 0
                    c_d[i] = 0
            
            step_obs.append(obs)
            step_rewards.append(reward)
            step_dones.append(float(done))
        
        obs_tensor = torch.FloatTensor(np.stack(step_obs)).to(device)
        
        observations.append(obs_tensor)
        actions.append(action_tensor)
        rewards.append(torch.FloatTensor(step_rewards).to(device))
        dones.append(torch.FloatTensor(step_dones).to(device))
        behavior_log_probs.append(action_log_probs)
        
        hidden_states_list = hidden_states_list_new
    
    return {
        'observations': torch.stack(observations, dim=0),  # (T+1, B, H, W, C)
        'actions': torch.stack(actions, dim=0),             # (T, B)
        'rewards': torch.stack(rewards, dim=0),             # (T, B)
        'dones': torch.stack(dones, dim=0),                 # (T, B)
        'behavior_log_probs': torch.stack(behavior_log_probs, dim=0),  # (T, B)
        'hidden_states': hidden_states_list,
    }


def evaluate_agent(
    agent: DRCAgent,
    levels: List[np.ndarray],
    device: torch.device,
    num_episodes: int = 100,
    thinking_steps: int = 0,
) -> float:
    """Evaluate agent solve rate."""
    env = SokobanEnv()
    solved = 0
    
    for i in range(min(num_episodes, len(levels))):
        level = levels[i]
        obs = env.reset(level)
        hidden_states = agent.init_hidden(batch_size=1, device=device)
        
        # Thinking steps
        for _ in range(thinking_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                _, _, hidden_states, _ = agent.forward(obs_tensor, hidden_states)
            obs, _, done, _ = env.step(0)
            if done:
                break
        
        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, hidden_states, _ = agent.forward(obs_tensor, hidden_states)
            action = logits.argmax(dim=-1).item()
            obs, _, done, info = env.step(action)
        
        if info.get('solved', False):
            solved += 1
    
    return solved / min(num_episodes, len(levels))


def main():
    parser = argparse.ArgumentParser(description='Train DRC agent on Sokoban')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Boxoban dataset')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--D', type=int, default=3, help='Number of ConvLSTM layers')
    parser.add_argument('--N', type=int, default=3, help='Number of ticks per step')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Hidden channels')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--unroll_length', type=int, default=20, help='Unroll length')
    parser.add_argument('--total_transitions', type=int, default=250_000_000, help='Total transitions')
    parser.add_argument('--checkpoint_every', type=int, default=1_000_000, help='Checkpoint frequency')
    parser.add_argument('--eval_every', type=int, default=1_000_000, help='Evaluation frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load data
    print("Loading Boxoban levels...")
    loader = BoxobanLoader(args.data_dir)
    train_levels = loader.load_levels('train')
    test_levels = loader.load_levels('test', max_levels=1000)
    
    print(f"Loaded {len(train_levels)} training levels, {len(test_levels)} test levels")
    
    # Create agent
    agent = DRCAgent(
        D=args.D,
        N=args.N,
        hidden_channels=args.hidden_channels,
    ).to(device)
    
    print(f"Created DRC({args.D},{args.N}) agent with {sum(p.numel() for p in agent.parameters())} parameters")
    
    # Create trainer
    trainer, scheduler = create_trainer(
        agent,
        total_transitions=args.total_transitions,
        batch_size=args.batch_size,
        device=device,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize environments
    envs = [SokobanEnv() for _ in range(args.batch_size)]
    for env in envs:
        level = random.choice(train_levels)
        env.reset(level)
    
    hidden_states_list = agent.init_hidden(batch_size=args.batch_size, device=device)
    
    # Training loop
    total_transitions = 0
    step = 0
    
    print("Starting training...")
    
    while total_transitions < args.total_transitions:
        # Collect rollout
        rollout = collect_rollout(
            agent, envs, train_levels, hidden_states_list, device, args.unroll_length
        )
        
        hidden_states_list = rollout['hidden_states']
        
        # Training step
        loss_dict = trainer.train_step(
            rollout['observations'],
            rollout['actions'],
            rollout['rewards'],
            rollout['dones'],
            rollout['behavior_log_probs'],
        )
        
        scheduler.step()
        
        total_transitions += args.batch_size * args.unroll_length
        step += 1
        
        # Logging
        if step % 100 == 0:
            print(f"Step {step}, Transitions {total_transitions:,}, "
                  f"Loss: {loss_dict['total_loss']:.4f}, "
                  f"Entropy: {loss_dict['entropy']:.4f}")
        
        # Evaluation
        if total_transitions % args.eval_every < args.batch_size * args.unroll_length:
            agent.eval()
            solve_rate = evaluate_agent(agent, test_levels, device, num_episodes=100)
            print(f"Transitions {total_transitions:,}, Solve rate: {solve_rate:.3f}")
            agent.train()
        
        # Checkpoint
        if total_transitions % args.checkpoint_every < args.batch_size * args.unroll_length:
            checkpoint_path = os.path.join(
                args.output_dir, 
                f'checkpoint_{total_transitions // 1_000_000}M.pt'
            )
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'total_transitions': total_transitions,
                'D': args.D,
                'N': args.N,
                'hidden_channels': args.hidden_channels,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final save
    final_path = os.path.join(args.output_dir, 'final_agent.pt')
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'total_transitions': total_transitions,
        'D': args.D,
        'N': args.N,
        'hidden_channels': args.hidden_channels,
    }, final_path)
    print(f"Training complete! Saved final agent to {final_path}")


if __name__ == '__main__':
    main()
