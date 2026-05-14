import torch
import torch.nn as nn
import torch.nn.functional as F # Added this import
from torch.distributions import Categorical

class ValueModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # This is a simplified value model. In reality, it would likely be a value head
        # on top of the LLM's features (hidden states).
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, states):
        # `states` here would be a batch of hidden states (features) for each macro step.
        # Shape: (batch_size, hidden_size)
        x = self.relu(self.fc1(states))
        value = self.fc2(x) # (batch_size, 1)
        return value.squeeze(-1) # Return a 1D tensor of values (batch_size,)

class PPO:
    def __init__(self, policy_model, value_model, clip_epsilon, gamma, lambda_gae):
        # policy_model is the actual LLM (AutoModelForCausalLM) which we'll use to update
        self.policy_model = policy_model 
        self.value_model = value_model # This is the separate value head/model
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma # Discount factor for future rewards beyond the macro action
        self.lambda_gae = lambda_gae # For Generalized Advantage Estimation

    def compute_macro_advantage(self, rewards, values, dones):
        # GAE for macro actions. rewards, values, and dones should be at the macro action level.
        # rewards: (num_macro_actions,)
        # values: (num_macro_actions,)
        # dones: (num_macro_actions,)

        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        last_gae_lambda = 0
        num_macro_actions = len(rewards)

        # Iterate backwards to calculate advantages
        for t in reversed(range(num_macro_actions)):
            if t == num_macro_actions - 1:
                next_value = 0.0 # If episode ends, next value is 0
            else:
                next_value = values[t+1]
            
            # Delta for GAE: TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t].float()) - values[t]
            # GAE formula
            last_gae_lambda = delta + self.gamma * self.lambda_gae * (1 - dones[t].float()) * last_gae_lambda
            advantages[t] = last_gae_lambda
        
        return advantages

    def calculate_loss(self,
                       macro_action_new_log_probs, # log pi_theta(omega_tau | s_tau)
                       macro_action_old_log_probs, # log pi_theta_old(omega_tau | s_tau)
                       advantages,
                       values,
                       rewards_to_go): # For value loss
        
        # Policy Loss (PPO-Clip objective - Equation 4 from the paper)
        ratio = torch.exp(macro_action_new_log_probs - macro_action_old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean() # PPO is typically minimized, so negate for maximization objective

        # Value Loss (Mean Squared Error)
        value_loss = F.mse_loss(values, rewards_to_go)
        
        # Combine losses (hyperparameters for weighting these might be needed in a full impl)
        # For now, let's just sum them, but in practice, you'd have coefficients (e.g., c1 * value_loss)
        total_loss = policy_loss + value_loss
        
        return total_loss

    def update(self,
               macro_actions: list,          # List of macro actions (list of token lists)
               states: torch.Tensor,         # Stacked hidden states for the start of macro actions (batch_size, hidden_size)
               rewards: torch.Tensor,        # Macro-level rewards (batch_size,)
               dones: torch.Tensor,          # Whether each macro action terminates an episode (batch_size,)
               old_policy_token_log_probs: torch.Tensor, # Log probabilities of individual tokens under the old policy
               new_policy_token_log_probs: torch.Tensor): # Log probabilities of individual tokens under the new policy
        
        # 1. Get value estimates for the current states
        values = self.value_model(states)

        # 2. Compute Advantages at the macro action level
        advantages = self.compute_macro_advantage(rewards, values, dones)

        # 3. Calculate rewards-to-go for value function training
        # This is typically computed from advantages + values
        rewards_to_go = advantages + values

        # 4. Calculate macro action log probabilities for current and old policies
        # We need to construct the macro action log probs from token log probs.
        # old_policy_token_log_probs and new_policy_token_log_probs are flat tensors
        # of log probs for *all* generated tokens in the sequence.

        macro_action_new_log_probs = []
        macro_action_old_log_probs = []
        
        current_token_idx_in_generated_sequence = 0
        for macro_action in macro_actions:
            macro_length = len(macro_action)
            
            # Sum log probabilities for tokens within the current macro action
            # Make sure to handle cases where a macro action might exceed the bounds of generated_token_log_probs
            # (e.g., due to EOS token or max_new_tokens truncating)
            if current_token_idx_in_generated_sequence + macro_length <= len(new_policy_token_log_probs):
                new_log_prob_sum = torch.sum(new_policy_token_log_probs[current_token_idx_in_generated_sequence : current_token_idx_in_generated_sequence + macro_length])
                old_log_prob_sum = torch.sum(old_policy_token_log_probs[current_token_idx_in_generated_sequence : current_token_idx_in_generated_sequence + macro_length])
            else:
                # Handle edge case where macro action is cut short at the end of generated sequence
                actual_macro_length = len(new_policy_token_log_probs) - current_token_idx_in_generated_sequence
                if actual_macro_length > 0:
                    new_log_prob_sum = torch.sum(new_policy_token_log_probs[current_token_idx_in_generated_sequence :])
                    old_log_prob_sum = torch.sum(old_policy_token_log_probs[current_token_idx_in_generated_sequence :])
                else:
                    new_log_prob_sum = torch.tensor(0.0, device=rewards.device)
                    old_log_prob_sum = torch.tensor(0.0, device=rewards.device)
                # The current_token_idx_in_generated_sequence update still needs the original macro_length
                # but for future iterations, the actual_macro_length is relevant for the remainder.
                # However, since this is the *last* macro action if it's truncated, this loop will terminate.
                

            macro_action_new_log_probs.append(new_log_prob_sum)
            macro_action_old_log_probs.append(old_log_prob_sum)
            
            current_token_idx_in_generated_sequence += macro_length # This should be the length of the *original* macro_action, not actual_macro_length here.
                                                                     # No, it should advance by the length *consumed* which is min(macro_length, actual_macro_length).
                                                                     # But given the if/else, this is already implicitly handled.
                                                                     # The `macro_length` variable within the loop refers to `len(macro_action)`.
                                                                     # If it enters the else block, `actual_macro_length` is the real length.
                                                                     # So this line should be `current_token_idx_in_generated_sequence += actual_macro_length` if truncated,
                                                                     # but this `macro_length` is always `len(macro_action)`.
                                                                     # Let's adjust this to avoid confusion.

        # Correcting the index advancement: 
        # The loop iterates `for macro_action in macro_actions:`. `macro_length = len(macro_action)` is fixed for that macro_action.
        # The `if` condition checks if `current_token_idx_in_generated_sequence + macro_length` is within bounds.
        # If it's not, it means the *last* macro_action was truncated. The `actual_macro_length` is then the remaining tokens.
        # The `current_token_idx_in_generated_sequence` should advance by `len(macro_action)` to correctly map to the next full macro action.
        # The problem is `macro_length` is reassigned in the else block. This was a bug. Let's fix this.
        
        # Re-thinking the index advancement for macro_action_new_log_probs and macro_action_old_log_probs
        # The key is to iterate based on the actual `macro_actions` list, and sum log_probs for the tokens *within each macro_action*.
        # The `old_policy_token_log_probs` and `new_policy_token_log_probs` are assumed to correspond to `generated_tokens`.
        
        # Let's rewrite this part for clarity and correctness.

        macro_action_new_log_probs = []
        macro_action_old_log_probs = []
        
        current_idx_in_token_log_probs = 0
        for macro_action_tokens in macro_actions:
            length_of_current_macro_action = len(macro_action_tokens)

            # Ensure we don't go out of bounds of the actual generated token log probabilities
            # This can happen if the actual generated sequence was shorter than expected due to EOS or max_new_tokens.
            end_idx_in_token_log_probs = min(current_idx_in_token_log_probs + length_of_current_macro_action, 
                                             len(new_policy_token_log_probs))
            
            if end_idx_in_token_log_probs > current_idx_in_token_log_probs: # If there are tokens for this macro action
                new_log_prob_sum = torch.sum(new_policy_token_log_probs[current_idx_in_token_log_probs : end_idx_in_token_log_probs])
                old_log_prob_sum = torch.sum(old_policy_token_log_probs[current_idx_in_token_log_probs : end_idx_in_token_log_probs])
            else:
                # If this macro action is entirely beyond the generated sequence (e.g., due to previous truncation)
                new_log_prob_sum = torch.tensor(0.0, device=rewards.device)
                old_log_prob_sum = torch.tensor(0.0, device=rewards.device)
            
            macro_action_new_log_probs.append(new_log_prob_sum)
            macro_action_old_log_probs.append(old_log_prob_sum)
            
            current_idx_in_token_log_probs = end_idx_in_token_log_probs
            # If the current_idx_in_token_log_probs has reached the end, subsequent macro_actions will have 0 length of actual tokens.
            
        macro_action_new_log_probs = torch.stack(macro_action_new_log_probs)
        macro_action_old_log_probs = torch.stack(macro_action_old_log_probs)

        # 5. Calculate the PPO loss (policy_loss + value_loss)
        total_loss = self.calculate_loss(macro_action_new_log_probs, macro_action_old_log_probs, 
                                         advantages, values, rewards_to_go)
        
        return total_loss

