import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
from typing import Any, Dict, List, Tuple, Union, Optional
from abc import ABC, abstractmethod
from loguru import logger
from omegaconf import DictConfig

# Define a type alias for Config to avoid circular imports.
# In `main.py` or `rlhf_trainer.py`, `Config` will be imported from `config.py`.
# Here, we use DictConfig directly.
Config = DictConfig


class BaseLLM(ABC, nn.Module):
    """
    Abstract base class for all Large Language Models used in the MA-RLHF framework.
    Handles common functionalities like model loading, device placement, and precision.
    """

    def __init__(self, model_name: str, config: Config):
        """
        Initializes the base LLM, loading it from Hugging Face `from_pretrained`.

        Args:
            model_name: The identifier for the pre-trained model (e.g., "google/gemma-2b").
            config: The global configuration object, used to access `global.precision`.
        """
        super().__init__()
        self.model_name = model_name
        self.config = config

        # Determine torch_dtype based on global precision setting
        self.torch_dtype = torch.bfloat16 if config.global.precision == "bfloat16" else torch.float32

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model '{model_name}' with dtype {self.torch_dtype} on device: {self.device}")

        # Load the pre-trained causal language model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True # Needed for some models like CodeGemma, Gemma
            )
            self.model.to(self.device)
            logger.info(f"Successfully loaded {self.__class__.__name__} from '{model_name}'.")
        except Exception as e:
            logger.error(f"Failed to load {self.__class__.__name__} '{model_name}': {e}")
            raise

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Abstract method for the model's forward pass. To be implemented by subclasses.
        """
        pass

    def get_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes token-level log probabilities for a given input sequence.

        Args:
            input_ids: Tokenized input sequences (batch_size, sequence_length).
            attention_mask: Attention mask to ignore padding tokens (batch_size, sequence_length).

        Returns:
            A tensor of token-level log probabilities (batch_size, sequence_length).
            Log probabilities for padding tokens are masked to 0.
        """
        # Ensure model is in evaluation mode for consistent log_probs calculation
        # and to avoid dropout/batchnorm effects.
        self.model.eval() 
        with torch.no_grad(): # Log probs are typically used for evaluation, so no grad needed
            # Move inputs to the correct device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits # (batch_size, sequence_length, vocab_size)
            
            log_probs = F.log_softmax(logits, dim=-1) # (batch_size, sequence_length, vocab_size)
            
            # Gather log probabilities for the actual tokens in `input_ids`
            # Unsqueeze `input_ids` to match `vocab_size` dimension for gathering
            token_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
            # (batch_size, sequence_length)

            # Mask out log probabilities for padding tokens
            # Ensure attention_mask is on the same device as token_log_probs
            attention_mask = attention_mask.to(token_log_probs.device)
            masked_token_log_probs = token_log_probs * attention_mask

            return masked_token_log_probs


class SFTModel(BaseLLM):
    """
    Supervised Fine-Tuning (SFT) model. Used for initial policy training and as a reference
    model for KL divergence during PPO.
    """

    def __init__(self, model_name: str, config: Config):
        """
        Initializes the SFT model.

        Args:
            model_name: Identifier for the pre-trained model.
            config: Global configuration.
        """
        super().__init__(model_name, config)
        # Ensure the model is in training mode by default for SFT,
        # but BaseLLM puts it in eval for get_log_probs. The trainer will manage this.
        self.model.train() 

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a forward pass for SFT, calculating the causal language modeling loss.

        Args:
            input_ids: Tokenized input sequences (batch_size, sequence_length).
            attention_mask: Attention mask (batch_size, sequence_length).
            labels: Target labels for causal LM loss (batch_size, sequence_length).
                    Padding tokens should be -100 to be ignored by loss.

        Returns:
            The computed causal language modeling loss (scalar).
        """
        # Ensure inputs are on the correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss


class RewardModel(BaseLLM):
    """
    Reward Model (RM) for assigning scalar scores to prompt-response pairs,
    trained via a ranking loss.
    """

    def __init__(self, model_name: str, config: Config):
        """
        Initializes the Reward Model. It loads a causal LM and adds a value head
        on top to output a scalar score.

        Args:
            model_name: Identifier for the pre-trained model.
            config: Global configuration.
        """
        super().__init__(model_name, config)
        self.model.eval() # RM is typically used in eval mode to provide fixed rewards

        # Add a value head on top of the base LM's last hidden state
        # The hidden size is typically `self.model.config.hidden_size`
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(self.device).to(self.torch_dtype)

        # Paper Appendix B.2: "Reward Model is initialized using the fine-tuned SFT model."
        # This implies the base_model_name here should align with the SFT model.
        # If specific freezing is desired, it should be in config.
        freeze_base_model = config.rm_config.get("freeze_base_model", False) # Default to False
        if freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info(f"RewardModel: Froze base LLM parameters. Only training value_head.")
        else:
            logger.info(f"RewardModel: Base LLM parameters are trainable.")

        # Initialize Tokenizer for length calculations if needed.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # Align with general generation setup

    def forward(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the ranking loss for RM training based on preferred and dispreferred responses.

        Args:
            prompt_ids: Tokenized prompts (batch_size, prompt_seq_len).
            chosen_ids: Tokenized chosen responses (batch_size, chosen_seq_len).
            rejected_ids: Tokenized rejected responses (batch_size, rejected_seq_len).

        Returns:
            The computed ranking loss (scalar).
        """
        # Ensure RM is in eval mode during reward calculation for loss,
        # but if base model is not frozen, it needs to be in train mode for backprop.
        # The trainer will handle model.train() if RM parameters are updated.
        # For a loss function involving evaluation of reward, eval() is appropriate for the base model.
        # However, the value_head is being trained. If the base model is frozen, it won't matter.
        # If not frozen, it needs to be in train mode. Let's assume the trainer sets it.

        # Get reward scores for chosen and rejected responses
        r_chosen = self.get_reward(prompt_ids, chosen_ids)
        r_rejected = self.get_reward(prompt_ids, rejected_ids)

        # Ranking loss: -log(sigmoid(r_chosen - r_rejected))
        loss = -F.logsigmoid(r_chosen - r_rejected).mean() # Mean over batch

        return loss

    def get_reward(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes the scalar reward score for a given prompt-response pair.

        Args:
            prompt_ids: Tokenized prompts (batch_size, prompt_seq_len).
            response_ids: Tokenized responses (batch_size, response_seq_len).

        Returns:
            A tensor of scalar reward scores for each pair in the batch (batch_size,).
        """
        # Ensure in eval mode for reward prediction, even if RM is trainable.
        # This is for consistent reward prediction during training steps.
        original_training_state = self.training
        self.eval() 
        with torch.no_grad(): # No gradient needed for reward prediction

            # Concatenate prompt and response for full sequence
            full_input_ids = []
            full_attention_masks = []
            
            for i in range(prompt_ids.shape[0]): # Iterate through batch
                p_ids = prompt_ids[i].to(self.device)
                r_ids = response_ids[i].to(self.device)
                
                # Filter out padding tokens if they exist already in p_ids/r_ids (shouldn't if `do_not_pad` was used initially)
                p_ids = p_ids[p_ids != self.tokenizer.pad_token_id]
                r_ids = r_ids[r_ids != self.tokenizer.pad_token_id]

                combined_ids = torch.cat((p_ids, r_ids), dim=0)

                # Add EOS token if not already present and not too long
                if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id not in combined_ids:
                    if len(combined_ids) < self.model.config.max_position_embeddings: # Basic check
                        combined_ids = torch.cat((combined_ids, torch.tensor([self.tokenizer.eos_token_id], device=self.device)))

                full_input_ids.append(combined_ids)

            # Pad the batch of full sequences to the longest sequence in the batch
            if not full_input_ids: # Handle empty batch case
                return torch.empty(0, device=self.device, dtype=self.torch_dtype)

            max_seq_len = max(len(ids) for ids in full_input_ids)
            
            padded_full_input_ids_list = []
            padded_full_attention_masks_list = []
            for ids in full_input_ids:
                padding_len = max_seq_len - len(ids)
                # Pad on the left as it's typically done for decoder-only models and RLHF
                padded_ids = F.pad(ids, (padding_len, 0), value=self.tokenizer.pad_token_id)
                attention_mask = torch.ones_like(ids, dtype=torch.long)
                padded_attention_mask = F.pad(attention_mask, (padding_len, 0), value=0)
                padded_full_input_ids_list.append(padded_ids)
                padded_full_attention_masks_list.append(padded_attention_mask)
            
            padded_full_input_ids_tensor = torch.stack(padded_full_input_ids_list).to(self.device)
            padded_full_attention_masks_tensor = torch.stack(padded_full_attention_masks_list).to(self.device)
            
            outputs = self.model(
                input_ids=padded_full_input_ids_tensor,
                attention_mask=padded_full_attention_masks_tensor,
                output_hidden_states=True # Need hidden states for value head
            )
            
            # Get the hidden state of the last non-padding token for each sequence
            # `attention_mask` is (batch_size, sequence_length)
            # `last_non_pad_token_indices` will be (batch_size,)
            last_non_pad_token_indices = padded_full_attention_masks_tensor.sum(dim=1) - 1
            
            # `outputs.hidden_states` is a tuple, take the last layer's hidden states
            # `last_hidden_state` is (batch_size, sequence_length, hidden_size)
            last_hidden_state = outputs.hidden_states[-1] 
            
            # Use `gather` to select the hidden state corresponding to the last actual token
            # `last_non_pad_token_indices` needs to be unsqueezed and expanded to match hidden_size dim
            hidden_size = last_hidden_state.shape[-1]
            last_token_hidden_states = last_hidden_state.gather(
                1, last_non_pad_token_indices.view(-1, 1).unsqueeze(-1).expand(-1, -1, hidden_size)
            ).squeeze(1) # (batch_size, hidden_size)
            
            # Pass through the value head
            reward_scores = self.value_head(last_token_hidden_states).squeeze(-1) # (batch_size,)
        
        if original_training_state: # Restore original training state if it was training
            self.train()

        return reward_scores


class PolicyModel(BaseLLM):
    """
    Policy model used during the RLHF stage to generate responses and
    whose parameters are optimized by MA-PPO.
    """

    def __init__(self, model_name: str, config: Config):
        """
        Initializes the Policy model.

        Args:
            model_name: Identifier for the pre-trained model.
            config: Global configuration.
        """
        super().__init__(model_name, config)
        self.model.train() # Policy model is trained in PPO
        # Initialize Tokenizer for generation parameters.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def generate(
        self, prompt_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates responses from prompts and records token-level log probabilities.

        Args:
            prompt_ids: Tokenized prompt sequences (batch_size, prompt_seq_len).
            attention_mask: Attention mask for prompts (batch_size, prompt_seq_len).
            **kwargs: Additional generation parameters (e.g., max_new_tokens, temperature, top_p, top_k).
                      These are typically populated from `config.ppo_config`.

        Returns:
            A tuple:
                - generated_ids (torch.Tensor): The full generated sequences, including prompts
                                              (batch_size, full_seq_len).
                - log_probs (torch.Tensor): Token-level log probabilities for the *entire* generated sequences
                                            (batch_size, full_seq_len).
        """
        # Generation is often done in eval mode for consistent sampling behavior.
        # The trainer will set the model back to train mode before PPO updates.
        original_training_state = self.training
        self.eval() 
        with torch.no_grad():
            prompt_ids = prompt_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            generation_output = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=kwargs.get("max_new_tokens", self.config.ppo_config.max_response_length),
                temperature=kwargs.get("temperature", self.config.ppo_config.temperature_sampling),
                top_p=kwargs.get("top_p", self.config.ppo_config.top_p),
                top_k=kwargs.get("top_k", self.config.ppo_config.top_k),
                do_sample=True, # Always sample in PPO rollout unless temperature is 0
                output_scores=True, # Needed to get logits for log_probs
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # Add support for `stopping_criteria` if needed for early stopping in generate
            )

            generated_ids = generation_output.sequences # (batch_size, full_sequence_length)
            
            # Scores is a tuple of (batch_size, vocab_size) logits for each new token
            # It only contains logits for newly generated tokens.
            scores = generation_output.scores # tuple of tensors (new_tokens_len, batch_size, vocab_size)

            # Convert scores (logits) to log_probs for *newly generated tokens*
            log_probs_new_tokens_list = []
            # 'scores' are for tokens *after* the prompt, so indices align with generated_ids from prompt_len onwards.
            for i, score_tensor in enumerate(scores): # Iterate over timesteps of generated tokens
                log_probs_t = F.log_softmax(score_tensor, dim=-1) # (batch_size, vocab_size)
                
                # Get the actual generated token at this timestep
                # This token is `generated_ids[:, prompt_ids.shape[1] + i]`
                current_generated_tokens = generated_ids[:, prompt_ids.shape[1] + i]
                
                # Gather the log_prob of the *chosen* token from the log_probs distribution
                log_prob_chosen = log_probs_t.gather(dim=-1, index=current_generated_tokens.unsqueeze(-1)).squeeze(-1)
                log_probs_new_tokens_list.append(log_prob_chosen)

            # Stack the log probabilities for the newly generated tokens
            if log_probs_new_tokens_list:
                log_probs_new_tokens = torch.stack(log_probs_new_tokens_list, dim=1) # (batch_size, new_tokens_len)
            else: # Handle case where no new tokens are generated (e.g., prompt is already EOS or max_new_tokens=0)
                log_probs_new_tokens = torch.empty(generated_ids.shape[0], 0, device=self.device, dtype=self.torch_dtype)


            # Create a full log_probs tensor with zeros for the prompt section and pad_token_id's log_probs for padding.
            full_log_probs = torch.zeros_like(generated_ids, dtype=log_probs_new_tokens.dtype, device=self.device)
            
            # Place the log_probs of generated tokens into the correct slice
            full_log_probs[:, prompt_ids.shape[1] : generated_ids.shape[1]] = log_probs_new_tokens
        
        if original_training_state: # Restore original training state
            self.train()

        return generated_ids, full_log_probs


class ValueModel(BaseLLM):
    """
    Value Model (Critic) used in PPO to estimate token-level state values.
    """

    def __init__(self, model_name: str, config: Config):
        """
        Initializes the Value Model. It loads a causal LM and adds a value head
        on top to output a scalar value *per token*.

        Args:
            model_name: Identifier for the pre-trained model.
            config: Global configuration.
        """
        super().__init__(model_name, config)
        self.model.eval() # Value model is used in eval mode to get fixed value estimates

        # Add a value head that maps hidden states to a single scalar for *each token*
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(self.device).to(self.torch_dtype)

        # Paper Appendix B.2: "the reward model initializes the critic model."
        # This implies that the initial weights for value_head might be similar to RM's value_head
        # or the base model part is shared/initialized similarly.
        # For this implementation, we initialize `ValueModel` from the same `model_name` as SFT/Policy,
        # but the `trainer` class will be responsible for copying weights from RM to ValueModel if needed.

        # Initialize Tokenizer for length calculations if needed.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass to estimate token-level value functions.

        Args:
            input_ids: Tokenized input sequences (batch_size, sequence_length).
            attention_mask: Attention mask (batch_size, sequence_length).

        Returns:
            A tensor of token-level value estimates (batch_size, sequence_length).
        """
        self.eval() # Ensure in eval mode for value prediction
        with torch.no_grad(): # No gradient needed for value prediction
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # `outputs.hidden_states` is a tuple, take the last layer's hidden states
            # `last_hidden_state` is (batch_size, sequence_length, hidden_size)
            last_hidden_state = outputs.hidden_states[-1]

            # Pass all hidden states through the value head to get token-level values
            # `token_values` will be (batch_size, sequence_length, 1)
            token_values = self.value_head(last_hidden_state).squeeze(-1) # (batch_size, sequence_length)

            # Mask out values for padding tokens (set to 0)
            token_values = token_values * attention_mask

        return token_values

    def get_token_values(
        self, prompt_ids: torch.Tensor, response_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimates token-level value functions specifically for the generated response part
        of a prompt-response pair.

        Args:
            prompt_ids: Tokenized prompt sequences (batch_size, prompt_seq_len). These are the
                        initial prompt tokens, used to determine the start of the *generated* response.
            response_ids: The full generated sequences, including the initial prompt tokens
                          (batch_size, full_seq_len). This is typically the output of
                          `PolicyModel.generate`.

        Returns:
            A tensor of token-level value estimates for the generated response tokens only
            (batch_size, generated_response_seq_len).
        """
        self.eval() # Ensure in eval mode
        with torch.no_grad():
            # `response_ids` already contains the prompt and is padded.
            # We need an attention mask for `response_ids`.
            response_attention_mask = (response_ids != self.tokenizer.pad_token_id).long().to(self.device)
            
            all_token_values = self.forward(
                response_ids, response_attention_mask
            ) # (batch_size, full_sequence_length)

            # `prompt_ids` is the original prompt input to PolicyModel.generate.
            # The generated part starts after its length.
            start_of_generated_response_idx = prompt_ids.shape[1]
            
            # Slice `all_token_values` to get only the values for the generated response tokens.
            # This slice also implicitly handles padding that might be present in the generated response,
            # as `all_token_values` are already multiplied by their attention mask.
            generated_response_token_values = all_token_values[:, start_of_generated_response_idx:]

        return generated_response_token_values


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(f"file_{__name__}.log", level="INFO")
    logger.add(sys.stderr, level="INFO")

    # Mock a minimal config object
    mock_config = DictConfig({
        "global": {
            "precision": "float32", # Use float32 for wider local testing compatibility
            "code_execution_timeout": 10 # Example, not used here
        },
        "rm_config": {
            "freeze_base_model": False
        },
        "ppo_config": {
            "max_response_length": 50,
            "temperature_sampling": 0.7,
            "top_p": 1.0,
            "top_k": 0
        },
        "model_configs": {
            "test_model": {"name": "gpt2"} # Using gpt2 for quick local testing
        }
    })

    # Initialize a tokenizer for testing purposes
    tokenizer_for_test = AutoTokenizer.from_pretrained(mock_config.model_configs.test_model.name)
    if tokenizer_for_test.pad_token is None:
        tokenizer_for_test.pad_token = tokenizer_for_test.eos_token
    tokenizer_for_test.padding_side = "left" # For generation consistency

    def _prepare_inputs(texts: Union[str, List[str]], max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = tokenizer_for_test(texts, return_tensors="pt", padding="longest", truncation=True, max_length=max_len)
        return encoded['input_ids'], encoded['attention_mask']

    print("--- Testing SFTModel ---")
    sft_model = SFTModel(mock_config.model_configs.test_model.name, mock_config)
    sft_model.model.train() # Set to train mode for SFT forward pass

    sft_prompts = ["Hello, how are you?", "What is the capital of France?"]
    sft_responses = ["I am fine, thank you.", "The capital of France is Paris."]
    sft_full_texts = [p + " " + r for p, r in zip(sft_prompts, sft_responses)] # Add space for concatenation

    sft_input_ids, sft_attention_mask = _prepare_inputs(sft_full_texts, max_len=64)
    sft_labels = sft_input_ids.clone()
    sft_labels[sft_labels == tokenizer_for_test.pad_token_id] = -100 # Mask padding for loss

    sft_loss = sft_model(sft_input_ids, sft_attention_mask, sft_labels)
    logger.info(f"SFT Loss: {sft_loss.item():.4f}")

    sft_log_probs = sft_model.get_log_probs(sft_input_ids, sft_attention_mask)
    logger.info(f"SFT Log Probs shape: {sft_log_probs.shape}")
    logger.info(f"SFT Log Probs (first example, first 5 tokens, masked): {sft_log_probs[0, :5]}")
    assert sft_log_probs.shape == sft_input_ids.shape
    sft_model.model.eval() # Set back to eval for next test


    print("\n--- Testing RewardModel ---")
    rm_model = RewardModel(mock_config.model_configs.test_model.name, mock_config)
    # rm_model.eval() # get_reward/forward implicitly handles eval state

    rm_prompts_text = ["Write a short story about a cat.", "Tell me a joke."]
    rm_chosen_text = ["Once there was a cat who loved to nap. He napped all day long.", "Why don't scientists trust atoms? Because they make up everything!"]
    rm_rejected_text = ["Dogs are cool too. Woof woof.", "Knock knock. Who's there? Banana."]

    rm_prompt_ids, _ = _prepare_inputs(rm_prompts_text)
    rm_chosen_ids, _ = _prepare_inputs(rm_chosen_text)
    rm_rejected_ids, _ = _prepare_inputs(rm_rejected_text)

    # Test get_reward
    r_chosen = rm_model.get_reward(rm_prompt_ids, rm_chosen_ids)
    r_rejected = rm_model.get_reward(rm_prompt_ids, rm_rejected_ids)
    logger.info(f"Reward for chosen: {r_chosen.tolist()}")
    logger.info(f"Reward for rejected: {r_rejected.tolist()}")

    # Test forward (loss calculation)
    rm_loss = rm_model(rm_prompt_ids, rm_chosen_ids, rm_rejected_ids)
    logger.info(f"RM Loss: {rm_loss.item():.4f}")
    assert r_chosen.shape == (len(rm_prompts_text),)
    assert r_rejected.shape == (len(rm_prompts_text),)


    print("\n--- Testing PolicyModel ---")
    policy_model = PolicyModel(mock_config.model_configs.test_model.name, mock_config)
    # policy_model.model.eval() # Generation implicitly handles eval mode

    policy_prompts_text = ["Once upon a time,", "The quick brown fox"]
    policy_prompt_ids, policy_attention_mask = _prepare_inputs(policy_prompts_text)

    # Test generate
    generated_ids, policy_log_probs_full_seq = policy_model.generate(
        policy_prompt_ids, policy_attention_mask,
        max_new_tokens=20, temperature=0.7, top_p=0.9
    )
    logger.info(f"Generated IDs shape: {generated_ids.shape}")
    logger.info(f"Policy Log Probs (full seq) shape: {policy_log_probs_full_seq.shape}")
    
    for i in range(len(policy_prompts_text)):
        logger.info(f"Prompt: '{tokenizer_for_test.decode(policy_prompt_ids[i], skip_special_tokens=True)}'")
        logger.info(f"Generated (full): '{tokenizer_for_test.decode(generated_ids[i], skip_special_tokens=True)}'")
        
        # Check log_probs for generated part
        generated_part_log_probs = policy_log_probs_full_seq[i, policy_prompt_ids.shape[1]:]
        logger.info(f"Log probs for generated part (first 5 non-zero tokens): {generated_part_log_probs[generated_part_log_probs != 0][:5]}")
    
    assert generated_ids.shape[0] == len(policy_prompts_text)
    assert generated_ids.shape == policy_log_probs_full_seq.shape
    assert torch.all(policy_log_probs_full_seq[:, :policy_prompt_ids.shape[1]] == 0.0) # Prompt part should be zeroed out

    # Test get_log_probs (for existing sequences)
    existing_text = ["This is an existing text.", "Another sequence."]
    existing_ids, existing_attention_mask = _prepare_inputs(existing_text)
    existing_log_probs = policy_model.get_log_probs(existing_ids, existing_attention_mask)
    logger.info(f"Existing Log Probs shape: {existing_log_probs.shape}")
    assert existing_log_probs.shape == existing_ids.shape


    print("\n--- Testing ValueModel ---")
    value_model = ValueModel(mock_config.model_configs.test_model.name, mock_config)
    # value_model.eval() # Value model implicitly handles eval mode

    value_prompts_text = ["How far is the moon?", "What is the capital of France?"]
    value_response_full_text = ["How far is the moon? The moon is approximately 384,400 kilometers (238,900 miles) from Earth on average.", "What is the capital of France? The capital of France is Paris."]
    
    value_prompt_ids, value_prompt_attention_mask = _prepare_inputs(value_prompts_text)
    value_response_full_ids, value_response_full_attention_mask = _prepare_inputs(value_response_full_text)

    # Test forward
    all_token_values = value_model(value_response_full_ids, value_response_full_attention_mask)
    logger.info(f"All Token Values shape: {all_token_values.shape}")
    assert all_token_values.shape == value_response_full_ids.shape

    # Test get_token_values (for response part)
    # Here, `value_response_full_ids` is the "response_ids" parameter as it's the full generated sequence.
    response_part_values = value_model.get_token_values(value_prompt_ids, value_response_full_ids)
    logger.info(f"Response Part Values shape: {response_part_values.shape}")
    
    # Calculate expected response length for the assertion. This should be the length of the full sequence
    # minus the length of the *original* prompt for each example in the batch.
    expected_response_lengths = [
        (value_response_full_ids[i] != tokenizer_for_test.pad_token_id).sum().item() - \
        (value_prompt_ids[i] != tokenizer_for_test.pad_token_id).sum().item()
        for i in range(len(value_prompts_text))
    ]
    # In this test setup, _prepare_inputs pads all to longest, so `value_prompt_ids.shape[1]` is consistent for batch.
    # The slicing `all_token_values[:, start_of_generated_response_idx:]` means the length is `full_seq_len - prompt_len`.
    # Let's verify against the first example.
    logger.info(f"Expected response length (first example): {expected_response_lengths[0]}")
    assert response_part_values.shape[1] == (value_response_full_ids.shape[1] - value_prompt_ids.shape[1])


    logger.info("All model tests passed successfully!")

