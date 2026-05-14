## model.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Dict, Any


class Model:
    """
    Encapsulates the trained base model for multi-turn reasoning and self-correction tasks.
    Includes initialization, forward inference, and model checkpoint handling functionalities.
    """

    def __init__(self, pretrained_model: str = "Gemini 1.5 Flash", config: Dict[str, Any] = {}):
        """
        Initializes the pre-trained model and tokenizer using Hugging Face Transformers.

        Args:
            pretrained_model (str): Name of the pre-trained model to load (e.g., 'Gemini 1.5 Flash').
            config (dict): Configuration dictionary containing the model settings 
                          (e.g., `max_sequence_length`, `sampling_temperature`).

        Raises:
            ValueError: If the model is incompatible or fails to load correctly.
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)

        # Extract configurations
        self.max_sequence_length = config.get("datasets", {}).get("max_sequence_length", 512)  # Default to 512
        self.sampling_temperature = config.get("training", {}).get("stage1", {}).get("sampling_temperature", 1.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to the appropriate device
        self.model = self.model.to(self.device)
        torch.cuda.empty_cache()  # Ensure clean GPU memory setup


    def forward(self, x: str, attempt: int) -> Dict[str, Any]:
        """
        Performs forward inference for multi-turn reasoning and self-correction.

        Args:
            x (str): Input problem context or prompt text.
            attempt (int): Turn number (1 for first attempt, 2 for self-correction).

        Returns:
            dict: Model outputs containing predictions or token logits.

        Raises:
            ValueError: If `attempt` is not in {1, 2}.
            RuntimeError: If tokenized input exceeds `max_sequence_length`.
        """
        if attempt not in {1, 2}:
            raise ValueError("Invalid attempt number. Must be '1' or '2'.")

        # Tokenize input for different turns
        if attempt == 1:
            prompt = f"You are a reasoning expert. Solve the following problem:\n{x}\nAnswer:"
        elif attempt == 2:
            first_attempt_text = f"You have previously tried to solve:\n{x}\nPlease double-check this answer and correct any associated errors in reasoning:"
            prompt = first_attempt_text
        
        # Tokenize the input prompt
        tokenized_input = self.tokenizer(
            prompt, truncation=True, padding="max_length", max_length=self.max_sequence_length, return_tensors="pt"
        )

        # Handle overflowing sequence lengths
        if tokenized_input['input_ids'].size(1) > self.max_sequence_length:
            raise RuntimeError(f"Input sequence exceeds `max_sequence_length` of {self.max_sequence_length}. Truncated input!")

        # Move tokenized input to device
        tokenized_input = {key: value.to(self.device) for key, value in tokenized_input.items()}

        # Perform forward pass through the model
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model(**tokenized_input)

        # Decode outputs for human-readable predictions
        predicted_tokens = torch.argmax(outputs.logits, dim=-1)
        decoded_output = self.tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)

        # Return predictions and logits
        return {
            "predictions": decoded_output,
            "logits": outputs.logits
        }


    def load_model_weights(self, file_path: str) -> None:
        """
        Loads model weights from the specified checkpoint file.

        Args:
            file_path (str): Path to the pre-trained weights file.

        Returns:
            None

        Raises:
            FileNotFoundError: If the file path is invalid or weights file is missing.
            RuntimeError: If the loaded weights are incompatible with the model architecture.
        """
        try:
            checkpoint = torch.load(file_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)  # Load state_dict
            self.model.eval()  # Set the model to evaluation mode
            print(f"Model weights loaded from: {file_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}") from e
        except RuntimeError as e:
            raise RuntimeError(f"Error loading weights. Incompatible model architecture.") from e


    def save_model_weights(self, file_path: str) -> None:
        """
        Saves the current model weights to the specified checkpoint file.

        Args:
            file_path (str): Path to save the trained model weights.

        Returns:
            None

        Raises:
            IOError: If saving the weights file fails due to filesystem issues or permissions.
        """
        try:
            torch.save(self.model.state_dict(), file_path)
            print(f"Model weights saved to: {file_path}")
        except IOError as e:
            raise IOError(f"Error saving weights: {file_path}") from e
