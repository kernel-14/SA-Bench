
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional

from config import ModelConfig

class PolicyModel(torch.nn.Module):
    """
    Policy model based on a CausalLM, potentially with LoRA for efficient fine-tuning.
    This model generates text and is optimized during the RLHF stage.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Quantization config for 4-bit training if enabled
        bnb_config = None
        if hasattr(config, 'load_in_4bit') and config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            quantization_config=bnb_config if bnb_config else None,
            device_map="auto" if bnb_config else None, # "auto" for multi-GPU, or specify device
            torch_dtype=torch.bfloat16 if bnb_config else None # Use bfloat16 for computation if 4-bit
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

        # Set pad_token if not already set, crucial for batching and generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if config.use_lora:
            # Prepare model for k-bit training if quantization is used
            if bnb_config:
                self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=config.target_modules if config.target_modules else None,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        # Ensure pad_token_id is set for generation
        if 'pad_token_id' not in kwargs and self.tokenizer.pad_token_id is not None:
            kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


class RewardModel(torch.nn.Module):
    """
    Reward model that takes text and outputs a scalar reward score.
    This model is typically an AutoModel (not CausalLM) with a regression head.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.model = AutoModel.from_pretrained(config.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

        # Add a regression head for reward prediction
        # The exact architecture of the RM head is not specified in the paper,
        # so a simple linear layer on top of the last hidden state (or pooled output) is a common choice.
        # Assuming last hidden state for simplicity, often CLS token or mean pooling is used.
        # The size of the hidden state can vary, so we'll need to infer it.
        hidden_size = self.model.config.hidden_size
        self.reward_head = torch.nn.Linear(hidden_size, 1)

        # Initialize the reward head's weights, common practice
        self.reward_head.weight.data.normal_(mean=0.0, std=1 / (hidden_size + 1))
        self.reward_head.bias.data.zero_()

        # Set pad_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the last hidden state of the first token (CLS token equivalent) for reward prediction
        # or mean pooling over all tokens.
        # For simplicity, we use the last hidden state of the last token before padding.
        # The paper does not specify the exact pooling strategy.
        last_hidden_states = outputs.last_hidden_state # (batch_size, sequence_length, hidden_size)
        
        # Simple pooling: take the last token's hidden state before padding
        # This requires finding the actual length of each sequence
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1 # lengths - 1 to get index of last token
        pooled_output = last_hidden_states[torch.arange(last_hidden_states.shape[0]), sequence_lengths]
        
        reward = self.reward_head(pooled_output)
        return reward


class ValueModel(torch.nn.Module):
    """
    Value model (Critic) that estimates the value of a state (sequence of tokens).
    This is typically a transformer model with a regression head, similar to the RewardModel,
    but trained to predict V(s) or Q(s, a). In PPO, it estimates V(s).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.model = AutoModel.from_pretrained(config.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

        hidden_size = self.model.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1)

        self.value_head.weight.data.normal_(mean=0.0, std=1 / (hidden_size + 1))
        self.value_head.bias.data.zero_()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        
        # In PPO, the value model estimates the value of each token in the sequence.
        # So we apply the head to all token hidden states.
        value = self.value_head(last_hidden_states) # (batch_size, sequence_length, 1)
        return value

