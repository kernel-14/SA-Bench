import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Any, Dict, List, Tuple, Union, Optional
from loguru import logger
from omegaconf import DictConfig

# To avoid circular imports, define Config as DictConfig directly.
# In `main.py` or `rlhf_trainer.py`, `Config` will be imported from `config.py`.
# Here, we use DictConfig directly.
Config = DictConfig

# Type hints for external components. In a real project, these would be imported.
# For a clean, modular design, we explicitly import them.
from models import PolicyModel, ValueModel, RewardModel, SFTModel
from utils import TokenizerWrapper, compute_kl_divergence
from macro_action_handler import MacroActionHandler


class PPOAlgorithm:
    """
    Implements the Proximal Policy Optimization (PPO) algorithm adapted for
    Macro Actions (MA-PPO). It manages experience collection, macro action
    segmentation, advantage estimation, and policy/critic updates.
    """

    def __init__(
        self,
        policy_model: PolicyModel,
        value_model: ValueModel,
        reward_model: RewardModel,
        sft_model: SFTModel,
        tokenizer_wrapper: TokenizerWrapper,
        config: Config,
        macro_action_handler: MacroActionHandler,
    ):
        """
        Initializes the PPO algorithm with all necessary components and hyperparameters.

        Args:
            policy_model: The PolicyModel to be optimized.
            value_model: The ValueModel (critic) to estimate state values.
            reward_model: The RewardModel to provide feedback.
            sft_model: The SFT reference model for KL divergence calculation.
            tokenizer_wrapper: The TokenizerWrapper instance.
            config: The global configuration object.
            macro_action_handler: The MacroActionHandler to manage macro actions.
        """
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.sft_model = sft_model
        self.tokenizer_wrapper = tokenizer_wrapper
        self.config = config
        self.macro_action_handler = macro_action_handler

        # Extract PPO hyperparameters from config. Ensure default values are accessible.
        ppo_cfg = config.ppo_config
        self.clip_ratio: float = ppo_cfg.clip_ratio
        self.gae_lambda: float = ppo_cfg.gae_lambda
        self.gae_gamma: float = ppo_cfg.gae_gamma
        self.kl_coefficient: float = ppo_cfg.kl_coefficient
        self.ppo_epochs: int = ppo_cfg.ppo_epochs
        self.max_response_length: int = ppo_cfg.max_response_length

        # Optimizers (AdamW is a common choice for LLM fine-tuning)
        self.policy_optimizer = AdamW(
            self.policy_model.parameters(), lr=ppo_cfg.policy_learning_rate
        )
        self.value_optimizer = AdamW(
            self.value_model.parameters(), lr=ppo_cfg.critic_learning_rate
        )

        self.device = self.policy_model.device  # Use the policy model's device
        self.dtype = self.policy_model.torch_dtype # Use the policy model's dtype for tensor creation

        logger.info(f"PPOAlgorithm initialized with KL coefficient: {self.kl_coefficient}")
        logger.info(f"PPOAlgorithm initialized with clip ratio: {self.clip_ratio}")
        logger.info(f"PPOAlgorithm initialized with GAE lambda: {self.gae_lambda}, gamma: {self.gae_gamma}")
        logger.info(f"PPOAlgorithm initialized with PPO epochs: {self.ppo_epochs}")

    def _generate_experience(self, batch_prompts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Generates responses using the policy model and collects all token-level data
        required for PPO.

        Args:
            batch_prompts: A dictionary containing 'prompt_ids' and 'attention_mask'
                           for a batch of prompts.

        Returns:
            A dictionary containing generated_ids_full (full sequence including prompt),
            generated_response_ids (response part only), generated_response_mask,
            old_policy_log_probs_tokens, token_values, sequence_rm_score,
            sft_log_probs_tokens, and token_rewards.
            All these are token-level or sequence-level data for the generated *response part*
            (except generated_ids_full which is the full sequence).
        """
        prompt_ids = batch_prompts['prompt_ids'].to(self.device)
        prompt_attention_mask = batch_prompts['attention_mask'].to(self.device)
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]

        # 1. Generate responses and get old policy log probabilities (for the full sequence)
        # The `policy_model.generate` method returns log_probs_full_seq with zeros for prompt tokens.
        generated_ids_full, old_policy_log_probs_full_seq = self.policy_model.generate(
            prompt_ids,
            prompt_attention_mask,
            max_new_tokens=self.max_response_length,
            temperature=self.config.ppo_config.temperature_sampling,
            top_p=self.config.ppo_config.top_p,
            top_k=self.config.ppo_config.top_k,
        )
        
        # 2. Extract generated response part and its mask
        # Slice from prompt_len to get only the tokens that were generated
        generated_response_ids = generated_ids_full[:, prompt_len:]
        # Create an attention mask for *only the generated response tokens*.
        # Padding tokens will be 0.
        generated_response_mask = (generated_response_ids != self.tokenizer_wrapper.tokenizer.pad_token_id).long()

        # 3. Get old policy log probabilities for the generated part
        # Since `policy_model.generate` already zeros out prompt log_probs, we just slice.
        old_policy_log_probs_tokens = old_policy_log_probs_full_seq[:, prompt_len:]

        # 4. Get token-level values for the generated response part
        # `value_model.get_token_values` is designed to return values *only for the generated part*.
        token_values = self.value_model.get_token_values(prompt_ids, generated_ids_full)
        # `token_values` shape is (batch_size, generated_response_len)

        # 5. Get sequence-level reward from Reward Model
        # `reward_model.get_reward` processes the full generated sequence to yield one scalar per example.
        sequence_rm_score = self.reward_model.get_reward(prompt_ids, generated_ids_full)
        # `sequence_rm_score` shape is (batch_size,)

        # 6. Get SFT model's token log probabilities for the generated part
        # Need to create an attention mask for the full generated sequence for SFT model's `get_log_probs`.
        sft_full_seq_mask = (generated_ids_full != self.tokenizer_wrapper.tokenizer.pad_token_id).long()
        sft_log_probs_full_seq = self.sft_model.get_log_probs(generated_ids_full, sft_full_seq_mask)
        # Slice to get SFT log probabilities for the generated response tokens only.
        sft_log_probs_tokens = sft_log_probs_full_seq[:, prompt_len:]

        # 7. Construct token-level rewards
        # R(x,y) = r_phi(x,y) - beta * D_KL(pi_theta || pi_sft)
        # KL term is applied per token, r_phi(x,y) is distributed to the last valid token.
        kl_div_tokens = compute_kl_divergence(
            old_policy_log_probs_tokens, sft_log_probs_tokens, generated_response_mask
        ) # Shape: (batch_size, generated_response_len)

        token_rewards_raw = torch.zeros_like(
            old_policy_log_probs_tokens, dtype=self.dtype, device=self.device
        )
        for i in range(batch_size):
            # Find the last valid token index in the generated response part for each example.
            # `sum()` gives number of non-padding tokens. Subtracting 1 gives the 0-based index.
            last_valid_token_idx = (generated_response_mask[i].sum() - 1).item()
            if last_valid_token_idx >= 0:
                # Assign the sequence-level RM score to the last valid token.
                # All other tokens in token_rewards_raw remain 0 for now.
                token_rewards_raw[i, last_valid_token_idx] = sequence_rm_score[i]
        
        # Final token-level effective rewards including KL penalty
        token_rewards = token_rewards_raw - self.kl_coefficient * kl_div_tokens
        
        return {
            'generated_ids_full': generated_ids_full, # Full sequence (prompt + response)
            'generated_response_ids': generated_response_ids, # Only the generated response tokens
            'generated_response_mask': generated_response_mask, # Mask for generated_response_ids
            'old_policy_log_probs_tokens': old_policy_log_probs_tokens,
            'token_values': token_values,
            'sequence_rm_score': sequence_rm_score,
            'sft_log_probs_tokens': sft_log_probs_tokens,
            'token_rewards': token_rewards,
            'kl_div_tokens': kl_div_tokens # For metrics logging
        }

    def _compute_macro_action_data(self, token_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Segments the token-level data into macro actions and aggregates values, rewards,
        and log probabilities to the macro level. Handles batching by padding to the
        maximum number of macro actions in the batch.

        Args:
            token_data: A dictionary from `_generate_experience` containing token-level data
                        for the generated response part.

        Returns:
            A dictionary containing:
            - macro_action_segments_batch: List[List[Tuple[int, int]]], original segments per example.
            - macro_values: Padded Tensor of shape (batch_size, max_num_macro_actions_in_batch).
            - macro_rewards: Padded Tensor of shape (batch_size, max_num_macro_actions_in_batch).
            - macro_old_policy_log_probs: Padded Tensor of shape (batch_size, max_num_macro_actions_in_batch).
            - macro_attention_mask: Padded Tensor of shape (batch_size, max_num_macro_actions_in_batch).
        """
        batch_size = token_data['generated_response_ids'].shape[0]
        
        # Lists to store macro data for each example in the batch
        macro_action_segments_batch: List[List[Tuple[int, int]]] = []
        macro_values_list: List[torch.Tensor] = []
        macro_rewards_list: List[torch.Tensor] = []
        macro_old_policy_log_probs_list: List[torch.Tensor] = []

        termination_method = self.config.macro_action_config.default_termination_method
        value_agg_method = self.config.macro_action_config.default_value_aggregation_method
        
        # Prepare kwargs for macro action handler's get_macro_action_positions
        handler_kwargs = {}
        if termination_method == "fixed_n_gram":
            handler_kwargs["n_gram_length"] = self.config.macro_action_config.fixed_n_gram.get(
                "n_values", [self.config.macro_action_config.default_n_gram_n]
            )[0] # Use the first value as default or the configured one
        elif termination_method == "randomized_n_gram":
            handler_kwargs["n_gram_list"] = self.config.macro_action_config.randomized_n_gram.list_of_lengths
            handler_kwargs["repeat_times"] = self.config.macro_action_config.randomized_n_gram.repeat_times
        elif termination_method == "parsing_based":
            handler_kwargs["cutoff"] = self.config.macro_action_config.parsing_based.cutoff
            # `self.macro_action_handler.nlp` is passed internally by MacroActionHandler._parsing_based_termination
        elif termination_method == "perplexity_based":
            handler_kwargs["sft_model"] = self.sft_model # SFTModel instance required for perplexity calculation

        max_num_macro_actions = 0

        for i in range(batch_size):
            # Extract single example's data for processing
            response_ids_single = token_data['generated_response_ids'][i]
            token_values_single = token_data['token_values'][i]
            token_rewards_single = token_data['token_rewards'][i]
            old_policy_log_probs_single = token_data['old_policy_log_probs_tokens'][i]
            
            # 1. Get macro action segments for this single response
            # `get_macro_action_positions` expects a 1D tensor representing just the response tokens.
            segments = self.macro_action_handler.get_macro_action_positions(
                response_ids_single,
                termination_method,
                self.tokenizer_wrapper,
                **handler_kwargs
            )
            macro_action_segments_batch.append(segments)

            # Handle case of empty segments (e.g., very short response or failure to segment)
            if not segments:
                macro_values_list.append(torch.empty(0, device=self.device, dtype=self.dtype))
                macro_rewards_list.append(torch.empty(0, device=self.device, dtype=self.dtype))
                macro_old_policy_log_probs_list.append(torch.empty(0, device=self.device, dtype=self.dtype))
                continue

            max_num_macro_actions = max(max_num_macro_actions, len(segments))

            # 2. Aggregate token-level values to macro-level
            macro_values_list.append(
                self.macro_action_handler.aggregate_values(
                    token_values_single, segments, value_agg_method
                )
            )

            # 3. Aggregate token-level rewards to macro-level
            macro_rewards_list.append(
                self.macro_action_handler.aggregate_rewards(
                    token_rewards_single, segments
                )
            )

            # 4. Aggregate old policy log probabilities to macro-level
            # _aggregate_token_log_probs_to_macro expects a batch of log_probs and a list of lists of segments.
            # For a single example, we temporarily unsqueeze it to create a batch of 1.
            macro_old_policy_log_probs_list.append(
                self._aggregate_token_log_probs_to_macro(
                    old_policy_log_probs_single.unsqueeze(0), [segments]
                ).squeeze(0) # Remove the batch dimension added by unsqueeze for the single example's result
            )

        # 5. Pad macro-level data to `max_num_macro_actions` for batching
        if max_num_macro_actions == 0: # If no macro actions were formed in the entire batch
            return {
                'macro_action_segments_batch': macro_action_segments_batch,
                'macro_values': torch.empty(batch_size, 0, device=self.device, dtype=self.dtype),
                'macro_rewards': torch.empty(batch_size, 0, device=self.device, dtype=self.dtype),
                'macro_old_policy_log_probs': torch.empty(batch_size, 0, device=self.device, dtype=self.dtype),
                'macro_attention_mask': torch.empty(batch_size, 0, device=self.device, dtype=torch.long),
            }

        # Initialize padded tensors with zeros (or a suitable default for log-probs if needed)
        macro_values_padded = torch.zeros(batch_size, max_num_macro_actions, device=self.device, dtype=self.dtype)
        macro_rewards_padded = torch.zeros(batch_size, max_num_macro_actions, device=self.device, dtype=self.dtype)
        macro_old_policy_log_probs_padded = torch.full( # Use -inf for log_probs of padded entries
            (batch_size, max_num_macro_actions),
            fill_value=float('-inf'),
            device=self.device,
            dtype=self.dtype
        )
        macro_attention_mask = torch.zeros(batch_size, max_num_macro_actions, device=self.device, dtype=torch.long)

        for i in range(batch_size):
            num_macro_actions_in_example = macro_values_list[i].numel()
            if num_macro_actions_in_example > 0:
                macro_values_padded[i, :num_macro_actions_in_example] = macro_values_list[i]
                macro_rewards_padded[i, :num_macro_actions_in_example] = macro_rewards_list[i]
                macro_old_policy_log_probs_padded[i, :num_macro_actions_in_example] = macro_old_policy_log_probs_list[i]
                macro_attention_mask[i, :num_macro_actions_in_example] = 1

        return {
            'macro_action_segments_batch': macro_action_segments_batch, # List of lists of segments for policy_loss calculation
            'macro_values': macro_values_padded,
            'macro_rewards': macro_rewards_padded,
            'macro_old_policy_log_probs': macro_old_policy_log_probs_padded,
            'macro_attention_mask': macro_attention_mask,
        }

    def _compute_advantages_and_returns(
        self, macro_values: torch.Tensor, macro_rewards: torch.Tensor, macro_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Generalized Advantage Estimation (GAE) for macro actions.

        Args:
            macro_values: Tensor of shape (batch_size, num_macro_actions_padded), representing V^pi(s_tau).
            macro_rewards: Tensor of shape (batch_size, num_macro_actions_padded), representing R_tau.
            macro_attention_mask: Tensor of shape (batch_size, num_macro_actions_padded), mask for valid macro actions.

        Returns:
            A tuple:
                - macro_advantages: Tensor of shape (batch_size, num_macro_actions_padded).
                - macro_returns: Tensor of shape (batch_size, num_macro_actions_padded).
        """
        batch_size, num_macro_actions = macro_values.shape
        
        advantages = torch.zeros_like(macro_values, device=self.device, dtype=self.dtype)
        returns = torch.zeros_like(macro_values, device=self.device, dtype=self.dtype)
        
        # `last_gae_lambda` is often used for a more efficient backward pass, but for clarity,
        # we compute deltas and advantages explicitly with a loop here, as described in standard GAE.
        
        # Iterate backward through macro actions
        for t in reversed(range(num_macro_actions)):
            # Only process if macro action is valid at this timestep for at least one example in batch
            valid_macro_mask_t = macro_attention_mask[:, t].bool()

            if not valid_macro_mask_t.any():
                continue # Skip if no valid macro action at this timestep for any example

            # Next value: value of the next macro action, or 0 if it's the last one for each example.
            # Initialize with 0s.
            next_v = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
            next_advantage = torch.zeros(batch_size, device=self.device, dtype=self.dtype)

            # Populate `next_v` and `next_advantage` only for valid examples where there is a next macro action
            if t < num_macro_actions - 1:
                has_next_macro_mask = macro_attention_mask[:, t + 1].bool() # Mask for examples that actually have a next macro action
                examples_with_next_macro = valid_macro_mask_t & has_next_macro_mask
                
                if examples_with_next_macro.any():
                    next_v[examples_with_next_macro] = macro_values[examples_with_next_macro, t + 1]
                    next_advantage[examples_with_next_macro] = advantages[examples_with_next_macro, t + 1]
            
            # TD error (delta) for the current macro action
            # Only calculate for currently valid macro actions at timestep t
            delta_t = macro_rewards[:, t] + self.gae_gamma * next_v - macro_values[:, t]
            
            # Advantage for the current macro action using GAE
            advantages[valid_macro_mask_t, t] = delta_t[valid_macro_mask_t] + \
                                               self.gae_lambda * self.gae_gamma * next_advantage[valid_macro_mask_t]

            # Returns for the current macro action (G_t = A_t + V_t)
            returns[valid_macro_mask_t, t] = advantages[valid_macro_mask_t, t] + macro_values[valid_macro_mask_t, t]

        # Apply mask to zero out advantages and returns for padded macro actions.
        # This is already implicitly handled by processing only `valid_macro_mask_t` above,
        # but multiplying by mask again ensures any leftover floating point noise is removed.
        advantages = advantages * macro_attention_mask
        returns = returns * macro_attention_mask

        return advantages, returns

    def _calculate_ppo_loss_macro_action_policy(
        self,
        macro_policy_log_probs: torch.Tensor,
        macro_old_policy_log_probs: torch.Tensor,
        macro_advantages: torch.Tensor,
        macro_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the PPO clipped objective function for macro actions.

        Args:
            macro_policy_log_probs: Tensor, joint log probabilities of macro actions
                                    under the current policy (batch_size, num_macro_actions).
            macro_old_policy_log_probs: Tensor, joint log probabilities of macro actions
                                        under the old policy (batch_size, num_macro_actions).
            macro_advantages: Tensor, estimated advantages for macro actions (batch_size, num_macro_actions).
            macro_attention_mask: Tensor, mask for valid macro actions (batch_size, num_macro_actions).

        Returns:
            policy_loss: Tensor (scalar).
        """
        # Only consider valid macro actions for normalization and loss calculation
        valid_macro_advantages = macro_advantages[macro_attention_mask.bool()]

        if valid_macro_advantages.numel() > 0:
            # Normalize advantages (globally over all valid macro actions in the batch)
            normalized_advantages = (valid_macro_advantages - valid_macro_advantages.mean()) / (valid_macro_advantages.std() + 1e-8)
            
            # Create a masked advantages tensor to multiply with ratios
            macro_advantages_masked = torch.zeros_like(macro_advantages, device=self.device, dtype=self.dtype)
            macro_advantages_masked[macro_attention_mask.bool()] = normalized_advantages
        else:
            # If no valid macro actions, return 0 loss
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Calculate probability ratio (pi_new / pi_old)
        # log(pi_new) - log(pi_old) = log(pi_new / pi_old)
        # Detach `macro_old_policy_log_probs` to ensure no gradients flow through the old policy.
        log_ratio = macro_policy_log_probs - macro_old_policy_log_probs.detach() 
        ratio = torch.exp(log_ratio)

        # Term 1 of clipped objective: `ratio * advantages`
        pg_loss1 = -macro_advantages_masked * ratio

        # Term 2 of clipped objective: `clipped_ratio * advantages`
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        pg_loss2 = -macro_advantages_masked * clipped_ratio

        # PPO objective: take the minimum of the two terms for each macro action
        ppo_loss_per_macro = torch.max(pg_loss1, pg_loss2)
        
        # Mask out padded macro actions and average over all valid ones
        num_valid_macros = macro_attention_mask.sum()
        if num_valid_macros > 0:
            policy_loss = (ppo_loss_per_macro * macro_attention_mask).sum() / num_valid_macros
        else:
            policy_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype) # Should be caught by earlier check but for robustness

        return policy_loss


    def _calculate_ppo_loss_macro_action_critic(
        self, macro_values_current: torch.Tensor, macro_returns: torch.Tensor, macro_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Mean Squared Error (MSE) loss for the critic (value model)
        based on macro-level returns.

        Args:
            macro_values_current: Tensor, estimated value of macro actions by the critic
                                  under the current model (batch_size, num_macro_actions).
            macro_returns: Tensor, GAE-computed returns for macro actions
                           (batch_size, num_macro_actions).
            macro_attention_mask: Tensor, mask for valid macro actions (batch_size, num_macro_actions).

        Returns:
            critic_loss: Tensor (scalar).
        """
        # Only compute loss for valid macro actions (where mask is 1)
        valid_macro_values = macro_values_current[macro_attention_mask.bool()]
        valid_macro_returns = macro_returns[macro_attention_mask.bool()]

        if valid_macro_values.numel() > 0:
            critic_loss = F.mse_loss(valid_macro_values, valid_macro_returns)
        else:
            critic_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype) # No valid macro actions, no loss

        return critic_loss


    def _aggregate_token_log_probs_to_macro(
        self, token_log_probs_batch: torch.Tensor, macro_action_segments_batch: List[List[Tuple[int, int]]]
    ) -> torch.Tensor:
        """
        Helper method to sum token-level log probabilities within each macro action
        to get the joint log probability for the macro action. Handles batch dimension.

        Args:
            token_log_probs_batch: Tensor of shape (batch_size, generated_response_len),
                                   token-level log probabilities for generated response part.
            macro_action_segments_batch: List of lists, where each inner list contains
                                         (start_idx, end_idx) tuples for one example in the batch.

        Returns:
            macro_log_probs: Tensor of shape (batch_size, max_num_macro_actions_in_batch).
                             Padded with -inf for log-probabilities.
        """
        batch_size = token_log_probs_batch.shape[0]
        max_num_macro_actions = max(len(segments) for segments in macro_action_segments_batch) if macro_action_segments_batch else 0

        if max_num_macro_actions == 0:
            return torch.empty(batch_size, 0, device=self.device, dtype=self.dtype)

        # Initialize padded tensor with -inf, which is the log(0) for padded elements.
        macro_log_probs_padded = torch.full(
            (batch_size, max_num_macro_actions),
            fill_value=float('-inf'), 
            device=self.device,
            dtype=self.dtype
        )

        for i in range(batch_size):
            segments = macro_action_segments_batch[i]
            if not segments:
                continue # Skip if no segments for this example

            current_macro_log_probs = []
            for start, end in segments:
                # Sum log probabilities for tokens within this macro action.
                # `segment_log_probs` are already masked by `generated_response_mask` (values of 0 for padding tokens).
                # So we can just sum. The `old_policy_log_probs_tokens` passed here
                # are already token-level log-probabilities *for the generated response part*,
                # and are `0` for any padding in that part.
                segment_log_probs = token_log_probs_batch[i, start:end]
                macro_log_prob = segment_log_probs.sum()
                current_macro_log_probs.append(macro_log_prob)
            
            if current_macro_log_probs: # If any macro actions were actually found
                macro_log_probs_padded[i, :len(current_macro_log_probs)] = torch.stack(current_macro_log_probs)

        return macro_log_probs_padded


    def step(self, batch_prompts: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Executes one full PPO optimization step, involving experience generation,
        macro action processing, advantage calculation, and policy/critic loss computation.

        Args:
            batch_prompts: A dictionary containing 'prompt_ids' (Tensor) and 'attention_mask' (Tensor)
                           for a batch of prompts.

        Returns:
            A tuple:
                - policy_loss: Tensor, the calculated policy loss for the current step.
                - critic_loss: Tensor, the calculated critic loss for the current step.
                - metrics: dict, containing various logging metrics.
        """
        # Ensure models are in appropriate modes for training
        self.policy_model.train()
        self.value_model.train() # Critic model is updated, so it needs to be in train mode
        self.reward_model.eval() # Reward model provides fixed feedback
        self.sft_model.eval() # SFT model provides fixed reference log_probs for KL

        # 1. Generate experience and calculate token-level rewards
        token_data = self._generate_experience(batch_prompts)
        
        # Detach necessary tensors for PPO's old policy/value references
        old_policy_log_probs_tokens = token_data['old_policy_log_probs_tokens'].detach() 
        generated_response_ids = token_data['generated_response_ids']
        generated_response_mask = token_data['generated_response_mask']

        # 2. Compute macro action data (segments, macro values, rewards, old log_probs, mask)
        macro_data = self._compute_macro_action_data(token_data)
        
        macro_action_segments_batch = macro_data['macro_action_segments_batch']
        macro_values_old = macro_data['macro_values'].detach() # Detach old values for GAE input
        macro_rewards = macro_data['macro_rewards']
        macro_old_policy_log_probs = macro_data['macro_old_policy_log_probs'].detach() # Detach old log_probs for PPO ratio
        macro_attention_mask = macro_data['macro_attention_mask']

        # Initialize total losses for averaging over PPO epochs
        total_policy_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total_critic_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Prepare metrics for logging
        metrics = {
            'mean_sequence_reward': token_data['sequence_rm_score'].mean().item(),
            'mean_kl_div': (token_data['kl_div_tokens'] * generated_response_mask).sum().item() / 
                           (generated_response_mask.sum().item() + 1e-8), # Mask and average KL
            'policy_loss': 0.0, 
            'critic_loss': 0.0,
            'num_generated_tokens': generated_response_mask.sum().item(),
            'num_macro_actions': macro_attention_mask.sum().item(),
        }

        # Check if there are any valid macro actions to process
        if macro_attention_mask.sum().item() == 0:
            logger.warning("No valid macro actions generated in this batch. Skipping PPO update.")
            return total_policy_loss, total_critic_loss, metrics


        # 3. PPO Optimization Loop (multiple gradient steps per rollout)
        for ppo_epoch in range(self.ppo_epochs):
            # Recalculate current policy log probabilities under the *current* policy
            # These are token-level log_probs for the generated response part.
            # We need to pass the full generated_ids_full to get_log_probs, then slice.
            current_policy_log_probs_full_seq = self.policy_model.get_log_probs(
                token_data['generated_ids_full'],
                (token_data['generated_ids_full'] != self.tokenizer_wrapper.tokenizer.pad_token_id).long() # Mask for full seq
            )
            # Slice to get log_probs only for the generated response part
            current_policy_log_probs_tokens = current_policy_log_probs_full_seq[:, token_data['generated_ids_full'].shape[1] - generated_response_ids.shape[1]:]

            # Aggregate current policy log probabilities to macro level
            macro_current_policy_log_probs = self._aggregate_token_log_probs_to_macro(
                current_policy_log_probs_tokens, macro_action_segments_batch
            )

            # Get current value estimates from the critic for macro actions
            # This requires recalculating token-level values and then aggregating to macro-level.
            macro_current_values_from_critic_list = []
            for i in range(generated_response_ids.shape[0]):
                segments = macro_action_segments_batch[i]
                if not segments:
                    macro_current_values_from_critic_list.append(torch.empty(0, device=self.device, dtype=self.dtype))
                    continue
                
                # Get current token values for generated response part from value model
                # `get_token_values` expects `prompt_ids` and `generated_ids_full`.
                current_token_values_for_example = self.value_model.get_token_values(
                    batch_prompts['prompt_ids'][i].unsqueeze(0), 
                    token_data['generated_ids_full'][i].unsqueeze(0)
                ).squeeze(0) # Shape: (generated_response_len,)
                
                macro_current_values_from_critic_list.append(
                    self.macro_action_handler.aggregate_values(
                        current_token_values_for_example, 
                        segments, 
                        value_agg_method=self.config.macro_action_config.default_value_aggregation_method
                    )
                )

            # Pad current macro values to match the batch tensor shape
            macro_current_values_padded = torch.zeros_like(macro_values_old, device=self.device, dtype=self.dtype)
            for i in range(len(macro_current_values_from_critic_list)):
                num_macro_actions_in_example = macro_current_values_from_critic_list[i].numel()
                if num_macro_actions_in_example > 0:
                    macro_current_values_padded[i, :num_macro_actions_in_example] = macro_current_values_from_critic_list[i]


            # Advantage and Return Calculation (using the detached `macro_values_old` as baseline for GAE)
            macro_advantages, macro_returns = self._compute_advantages_and_returns(
                macro_values_old, macro_rewards, macro_attention_mask
            )

            # Policy Loss
            policy_loss = self._calculate_ppo_loss_macro_action_policy(
                macro_current_policy_log_probs, macro_old_policy_log_probs, macro_advantages, macro_attention_mask
            )
            total_policy_loss += policy_loss

            # Critic Loss
            critic_loss = self._calculate_ppo_loss_macro_action_critic(
                macro_current_values_padded, macro_returns, macro_attention_mask
            )
            total_critic_loss += critic_loss

            # Optimization steps
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            critic_loss.backward()
            self.value_optimizer.step()

        # Average losses over PPO epochs
        avg_policy_loss = total_policy_loss / self.ppo_epochs
        avg_critic_loss = total_critic_loss / self.ppo_epochs
        
        metrics['policy_loss'] = avg_policy_loss.item()
        metrics['critic_loss'] = avg_critic_loss.item()

        return avg_policy_loss, avg_critic_loss, metrics

