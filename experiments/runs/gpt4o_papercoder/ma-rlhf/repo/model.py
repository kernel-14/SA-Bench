# model.py
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Iterator, Dict

class Model:
    def __init__(self, pretrained_model: str, params: Dict):
        """Initialize the Gemma-based pre-trained language model and tokenizer.

        Args:
            pretrained_model (str): Name or path to the pre-trained model.
            params (Dict): Dictionary of task-specific configurations (e.g., learning rate, tokenizer rules).
        """
        self.pretrained_model_name = pretrained_model
        self.params = params

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model, 
            torch_dtype=torch.float16 if params.get("use_fp16", False) else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        # Apply gradient checkpointing for memory optimization (if specified in the config)
        if params.get("gradient_checkpoint", False):
            self.model.gradient_checkpointing_enable()

        # Freeze layers if specified (useful for reward modeling stage or large models)
        self.freeze_layers(params.get("freeze_layers", 0))

        # Max token length constraints
        self.max_prompt_length = params.get("max_prompt_length", 512)
        self.max_response_length = params.get("max_response_length", 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the language model.

        Args:
            x (torch.Tensor): Input tensor containing tokenized input sequences.

        Returns:
            torch.Tensor: Logits output from the model for each input token.
        """
        # Ensure input tensor respects max token length
        if x.size(-1) > self.max_prompt_length:
            raise ValueError(
                f"Input sequence length exceeds maximum limit: {self.max_prompt_length} tokens."
            )
        
        # Forward pass through the model
        outputs = self.model(input_ids=x, attention_mask=(x != self.tokenizer.pad_token_id))
        return outputs.logits

    def trainable_parameters(self) -> Iterator:
        """Retrieve trainable parameters for optimization.

        Returns:
            Iterator: An iterator over trainable parameters of the model.
        """
        return (param for param in self.model.parameters() if param.requires_grad)

    def freeze_layers(self, layers_to_freeze: int) -> None:
        """Freeze the lower layers of the model to limit trainable parameters.

        Args:
            layers_to_freeze (int): Number of transformer layers to freeze (from the bottom).
        """
        if layers_to_freeze > 0:
            for idx, layer in enumerate(self.model.transformer.h):
                if idx < layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False

    def tokenize(self, sequences: list, max_length: int = None) -> Dict[str, torch.Tensor]:
        """Tokenize input sequences and return tokenized tensor.

        Args:
            sequences (list): List of text strings to tokenize.
            max_length (int, optional): Maximum allowed sequence length. Defaults to max_prompt_length.

        Returns:
            Dict[str, torch.Tensor]: Tokenized tensors including input_ids and attention_mask.
        """
        max_length = max_length or self.max_prompt_length
        return self.tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def decode(self, token_ids: torch.Tensor) -> list:
        """Decode token IDs into human-readable text outputs.

        Args:
            token_ids (torch.Tensor): Tensor containing token IDs.

        Returns:
            list: List of decoded strings corresponding to the token IDs.
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def predict_reward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate reward logits for ranking loss during reward modeling stage.

        Args:
            x (torch.Tensor): Input tensor containing tokenized prompt and response sequences.

        Returns:
            torch.Tensor: Reward logits output by the model.
        """
        outputs = self.forward(x)
        # Optionally reshape logits or apply task-specific transformations here
        return outputs[:, -1]  # Take logits of the final token

    def adapt_for_task(self, task_type: str) -> None:
        """Adapt model architecture or behavior based on task type.

        Args:
            task_type (str): Canonical task name (e.g., "summarization", "code_generation").
        """
        if task_type == "code_generation":
            # Special token handling for CodeGemma models
            self.tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        elif task_type == "summarization":
            # Enable encoder-decoder behavior for summarization-based tasks
            pass  # Summarization mode is inherently handled by causal LM models
        elif task_type == "dialogue_generation":
            # Potential extensions for multi-turn dialogue setting (adding conversation separators)
            pass
        # Add additional task-specific adaptations as required

# Example usage
if __name__ == "__main__":
    # Example configuration dictionary (normally loaded from `config.yaml`)
    example_config = {
        "pretrained_model": "gemma-2b",
        "max_prompt_length": 512,
        "max_response_length": 256,
        "use_fp16": True,
        "gradient_checkpoint": False,
        "freeze_layers": 2
    }

    # Initialize model
    gemma_model = Model(example_config["pretrained_model"], example_config)

    # Tokenization test
    sample_input = ["This is a test input sequence for tokenization."]
    tokenized = gemma_model.tokenize(sample_input)
    print("Tokenized Input:", tokenized)

    # Debug forward pass
    logits = gemma_model.forward(tokenized["input_ids"])
    print(f"Logits Shape: {logits.shape}")
