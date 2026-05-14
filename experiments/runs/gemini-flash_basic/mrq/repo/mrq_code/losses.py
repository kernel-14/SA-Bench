import torch
import torch.nn as nn
import torch.nn.functional as F

from mrq_code.config import MRQConfig
from mrq_code.utils import two_hot_encode # Assuming two_hot_encode is in utils

def calculate_reward_loss(predicted_reward_logits, true_rewards, device):
    # predicted_reward_logits: (batch_size, reward_bins)
    # true_rewards: (batch_size,)
    
    # Two-hot encode true rewards
    true_rewards_encoded = two_hot_encode(true_rewards, MRQConfig.REWARD_BINS, MRQConfig.REWARD_RANGE).to(device)
    
    # Calculate cross-entropy loss
    # F.cross_entropy expects (N, C) for input and (N) for target (class indices)
    # or (N, C) for input and (N, C) for target (probabilities)
    # We have probabilities for target, so we use log_softmax on predictions
    log_probs = F.log_softmax(predicted_reward_logits, dim=-1)
    loss = -torch.sum(true_rewards_encoded * log_probs, dim=-1)
    return loss.mean()

def calculate_dynamics_loss(predicted_next_zs, target_next_zs):
    # predicted_next_zs: (batch_size, zs_dim)
    # target_next_zs: (batch_size, zs_dim)
    return F.mse_loss(predicted_next_zs, target_next_zs, reduction='mean')

def calculate_terminal_loss(predicted_terminal_logits, true_terminals):
    # predicted_terminal_logits: (batch_size,)
    # true_terminals: (batch_size,)
    return F.mse_loss(predicted_terminal_logits.squeeze(-1), true_terminals.float(), reduction='mean')

def calculate_encoder_loss(reward_logits, next_zs, terminal_logits, rewards, next_state_embeddings_target, terminals, device):
    reward_loss = calculate_reward_loss(reward_logits, rewards, device)
    dynamics_loss = calculate_dynamics_loss(next_zs, next_state_embeddings_target)
    terminal_loss = calculate_terminal_loss(terminal_logits, terminals)
    
    total_encoder_loss = (
        MRQConfig.LAMBDA_REWARD * reward_loss +
        MRQConfig.LAMBDA_DYNAMICS * dynamics_loss #+ # Temporarily commenting terminal loss for potential initial unroll issues
        # MRQConfig.LAMBDA_TERMINAL * terminal_loss
    )
    # The paper states terminal loss is set to 0 until the first terminal transition (d=0) is viewed.
    # This logic will be handled in the agent during the training loop.

    return total_encoder_loss, reward_loss, dynamics_loss, terminal_loss

def huber_loss(input, target, delta=1.0):
    # Custom Huber loss implementation to match the paper's description
    # This is a standard implementation of Huber loss
    error = torch.abs(input - target)
    quadratic = torch.min(error, torch.full_like(error, delta))
    linear = error - quadratic
    return 0.5 * quadratic**2 + delta * linear

def calculate_value_loss(q_predictions, target_q_values, is_prioritized_sampling=True):
    # q_predictions: (batch_size, 1) from Q network
    # target_q_values: (batch_size, 1) from target calculation
    
    if is_prioritized_sampling:
        # Use Huber loss with prioritized sampling as per paper (Fujimoto et al., 2020)
        loss = huber_loss(q_predictions, target_q_values).mean()
    else:
        loss = F.mse_loss(q_predictions, target_q_values, reduction='mean')
    return loss

def calculate_policy_loss(q_values_policy, policy_pre_activations, action_dim, is_discrete_action_space):
    # q_values_policy: (batch_size, 1) from Q networks for actions chosen by policy
    # policy_pre_activations: (batch_size, action_dim) raw output from policy MLP
    
    # The policy loss is -0.5 * sum(Q_i(zsa_pi)) + lambda_pre_activ * z_pi^2
    # Sum over Q_i is handled by providing sum/mean of two Q functions outside this function if needed
    q_loss = -q_values_policy.mean() # Paper specifies -0.5 * sum, but mean is more common for batches

    # Regularization on pre-activations
    pre_activ_loss = MRQConfig.LAMBDA_PRE_ACTIV * (policy_pre_activations**2).mean()
    
    return q_loss + pre_activ_loss

