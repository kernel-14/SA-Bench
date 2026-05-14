import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from marlhf.ppo import PPO, ValueModel
from marlhf.macro_actions import generate_ngram_macro_actions, generate_randomized_ngram_macro_actions

class MARLHF:
    def __init__(self,
                 policy_model_name: str = "gpt2", # Example
                 reward_model_name: str = "gpt2", # Example for a reward model (usually a critic)
                 n_macro_action_length: int = 4, # Default fixed n-gram length
                 clip_epsilon: float = 0.2,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 lr_policy: float = 1e-5, # Learning rate for policy model
                 lr_value: float = 1e-5): # Learning rate for value model

        # Load tokenizer and base LLM for policy and reference policy
        self.tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
        # Add a pad token if it doesn't exist, which is common for RL training
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.policy_lm = AutoModelForCausalLM.from_pretrained(policy_model_name)
        self.policy_lm_ref = AutoModelForCausalLM.from_pretrained(policy_model_name) # For old policy in PPO
        self.policy_lm_ref.eval() # Reference model should be in evaluation mode

        self.vocab_size = self.policy_lm.config.vocab_size
        self.hidden_size = self.policy_lm.config.hidden_size 

        self.value_head = ValueModel(self.hidden_size)

        self.ppo_agent = PPO(
            policy_model=self.policy_lm, 
            value_model=self.value_head, 
            clip_epsilon=clip_epsilon,
            gamma=gamma,
            lambda_gae=lambda_gae
        )

        # Optimizers
        # For the policy, we optimize the parameters of the LLM.
        # This is typically done by creating an optimizer over `self.policy_lm.parameters()`.
        # Note: In practice, fine-tuning large LLMs might involve more sophisticated optimizers
        # and learning rate schedules (e.g., AdamW with linear warmup and decay).
        self.policy_optimizer = optim.Adam(self.policy_lm.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_head.parameters(), lr=lr_value)

        # Reward Model (RM) - typically a different model or fine-tuned LLM
        # It's assumed to provide a scalar reward for a given (prompt, response) pair.
        # The paper uses r_phi(x, y), which is a single scalar.
        self.reward_model_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        # Using AutoModelForCausalLM as a placeholder for a reward model; in reality,
        # a reward model is usually a sequence classification head on top of an LLM.
        self.reward_model = AutoModelForCausalLM.from_pretrained(reward_model_name) 
        self.reward_model.eval() # Reward model should be in evaluation mode

        self.n_macro_action_length = n_macro_action_length

    def _get_reward(self, prompt: str, response: str) -> float:
        # Placeholder: Return a dummy reward.
        # In a real setup, this would involve passing prompt and response
        # through the reward model and getting its output. For example, a sentiment classifier.
        # Example: inputs = self.reward_model_tokenizer(prompt + response, return_tensors="pt")
        # with torch.no_grad():
        #     # Assuming a classification head outputting a scalar reward
        #     reward_score = self.reward_model(**inputs).logits.squeeze().item()
        # return reward_score
        return torch.randn(1).item() # Dummy reward for static code

    def _calculate_kl_penalty(self, policy_log_probs: torch.Tensor, sft_log_probs: torch.Tensor, beta: float) -> torch.Tensor:
        # D_KL(pi_theta || pi_sft). Approximated by mean difference of log probabilities of chosen actions.
        kl_div = (policy_log_probs - sft_log_probs).mean()
        return beta * kl_div

    def generate_sequence(self, prompt_tokens: list, max_new_tokens: int = 50, temperature: float = 1.0) -> tuple[list, torch.Tensor]:
        # Generate a sequence of tokens from the policy model and collect log probabilities.
        
        input_ids = torch.tensor([prompt_tokens]).long().to(self.policy_lm.device)
        generated_tokens = []
        log_probs = []
        
        self.policy_lm.eval() # Set policy_lm to evaluation mode for generation
        
        attention_mask = torch.ones(1, len(prompt_tokens)).long().to(self.policy_lm.device)
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.policy_lm(
                    input_ids=input_ids[:, -1:], 
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    return_dict=True
                )
                logits = outputs.logits[:, -1, :] 
                past_key_values = outputs.past_key_values
                
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                else: 
                    probs = torch.softmax(logits, dim=-1)

                dist = torch.distributions.Categorical(probs)
                next_token = dist.sample()
                
                log_prob = dist.log_prob(next_token)
                
                generated_tokens.append(next_token.item())
                log_probs.append(log_prob)
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones(1,1).long().to(self.policy_lm.device)], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        self.policy_lm.train() # Revert policy_lm to training mode after generation
        
        return generated_tokens, torch.stack(log_probs) if log_probs else torch.tensor([]).to(self.policy_lm.device)

    def _get_token_log_probs_and_hidden_states(self, prompt_tokens: list, generated_tokens: list, model: AutoModelForCausalLM) -> tuple[torch.Tensor, torch.Tensor]:
        # This function concatenates prompt and generated tokens, and gets the log probabilities
        # of the *generated* tokens and the last layer hidden states under the given model.
        
        full_sequence_ids = prompt_tokens + generated_tokens
        input_ids = torch.tensor([full_sequence_ids]).long().to(model.device)
        
        model.eval() # Ensure model is in eval mode for inference

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True, # We need hidden states for the value function
                return_dict=True
            )
            logits = outputs.logits
            # last_hidden_states is (batch_size, sequence_length, hidden_size)
            last_hidden_states = outputs.hidden_states[-1].squeeze(0) 

            log_probs_for_full_sequence = torch.log_softmax(logits, dim=-1)
            
            generated_token_log_probs = []
            for i, token_id in enumerate(generated_tokens):
                # log_prob of generated_tokens[i] is at index `len(prompt_tokens) + i - 1` in logits
                # for token `token_id`.
                if len(prompt_tokens) + i - 1 >= 0 and len(prompt_tokens) + i - 1 < log_probs_for_full_sequence.shape[1]:
                    log_prob_of_token = log_probs_for_full_sequence[0, len(prompt_tokens) + i - 1, token_id]
                    generated_token_log_probs.append(log_prob_of_token)
                else:
                    # This might happen if generated_tokens is empty or very short.
                    # For robustness, handle this gracefully.
                    pass
            
        model.train() # Revert model to training mode after inference

        return torch.stack(generated_token_log_probs) if generated_token_log_probs else torch.tensor([]).to(model.device), last_hidden_states

    def train_step(self, prompt: str, sft_model: AutoModelForCausalLM, beta: float = 0.1) -> float:
        # 1. Generate response using the current policy (self.policy_lm)
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0).tolist()
        generated_tokens, current_policy_token_log_probs = self.generate_sequence(prompt_tokens)
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if current_policy_token_log_probs.numel() == 0: 
            # print(f"Warning: No tokens generated for prompt: {prompt}")
            return 0.0 

        # 2. Get rewards from the reward model
        reward_score = self._get_reward(prompt, response_text)

        # 3. Calculate KL penalty with SFT model (reference policy)
        sft_token_log_probs, _ = self._get_token_log_probs_and_hidden_states(prompt_tokens, generated_tokens, sft_model)
        
        if sft_token_log_probs.numel() == 0:
            # print(f"Warning: No SFT token log probs for prompt: {prompt}")
            return 0.0

        kl_penalty = self._calculate_kl_penalty(current_policy_token_log_probs, sft_token_log_probs, beta)
        total_reward = reward_score - kl_penalty.item() 

        # 4. Generate macro actions
        macro_actions = generate_ngram_macro_actions(generated_tokens, self.n_macro_action_length)
        
        if not macro_actions: 
            # print(f"Warning: No macro actions generated for prompt: {prompt}")
            return 0.0

        macro_rewards = torch.zeros(len(macro_actions), dtype=torch.float32).to(self.policy_lm.device)
        macro_rewards[-1] = total_reward 
        
        macro_dones = torch.zeros(len(macro_actions), dtype=torch.bool).to(self.policy_lm.device)
        macro_dones[-1] = True
            
        # 5. Extract hidden states for macro actions (states for value function)
        _, last_hidden_states_full_seq = self._get_token_log_probs_and_hidden_states(
            prompt_tokens, generated_tokens, self.policy_lm
        )

        macro_states = []
        # The `current_token_global_idx` keeps track of the index in the `full_sequence_ids` (prompt + generated).
        # The prompt itself contributes `len(prompt_tokens)` to the sequence.
        # The first macro action starts after the prompt.
        current_token_global_idx_in_generated = 0 # Index within only the generated_tokens list
        
        for i, macro_action in enumerate(macro_actions):
            # The state s_tau for macro_action[i] is the hidden state *before* it starts.
            # This corresponds to the hidden state of the token at index `len(prompt_tokens) + current_token_global_idx_in_generated - 1`
            # in the `last_hidden_states_full_seq` tensor.
            
            state_hidden_idx = len(prompt_tokens) + current_token_global_idx_in_generated -1
            
            # Handle the very first macro action: its state is the hidden state of the last prompt token.
            if current_token_global_idx_in_generated == 0:
                # If prompt is empty and generation starts immediately, state_hidden_idx would be -1.
                # In such cases, we might use a zero vector or the first actual hidden state.
                # For now, let's assume prompt_tokens is non-empty enough to provide a state.
                # If prompt_tokens is empty and generated_tokens is not, state_hidden_idx will be -1.
                # So we take 0 in this edge case.
                if state_hidden_idx < 0:
                    macro_states.append(last_hidden_states_full_seq[0]) # Use the first hidden state if no prompt
                else:
                    macro_states.append(last_hidden_states_full_seq[state_hidden_idx])
            else:
                macro_states.append(last_hidden_states_full_seq[state_hidden_idx])
            
            current_token_global_idx_in_generated += len(macro_action)

        if not macro_states:
            # print(f"Warning: No macro states extracted for prompt: {prompt}")
            return 0.0
            
        # 6. Get old policy token log probabilities from the reference model
        old_policy_token_log_probs, _ = self._get_token_log_probs_and_hidden_states(
            prompt_tokens, generated_tokens, self.policy_lm_ref
        )
        
        if old_policy_token_log_probs.numel() == 0:
            # print(f"Warning: No old policy token log probs for prompt: {prompt}")
            return 0.0

        # 7. Update the policy and value function using MA-PPO
        # Set policy_lm to training mode for parameter updates
        self.policy_lm.train()
        self.value_head.train()

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        loss = self.ppo_agent.update(
            macro_actions=macro_actions,
            states=torch.stack(macro_states), 
            rewards=macro_rewards,
            dones=macro_dones,
            old_policy_token_log_probs=old_policy_token_log_probs,
            new_policy_token_log_probs=current_policy_token_log_probs
        )
        
        loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        return loss.item()

    def run_training(self, prompts: list, num_epochs: int = 10, sft_model: AutoModelForCausalLM = None, beta: float = 0.1):
        if sft_model is None:
            sft_model = self.policy_lm_ref
        
        for epoch in range(num_epochs):
            total_loss = 0
            # self.policy_lm.train() is handled in train_step
            # self.value_head.train() is handled in train_step

            for i, prompt in enumerate(prompts):
                # print(f"Epoch {epoch+1}/{num_epochs}, Processing prompt {i+1}/{len(prompts)}")
                loss = self.train_step(prompt, sft_model, beta)
                total_loss += loss
            
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(prompts):.4f}")
            
            # Update the reference model (policy_lm_ref) periodically
            self.policy_lm_ref.load_state_dict(self.policy_lm.state_dict())
            self.policy_lm_ref.eval() # Ensure reference model stays in eval mode

