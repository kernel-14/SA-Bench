"""
This module provides utility functions for tokenization, metric logging,
and KL divergence computation, supporting the MA-RLHF framework.
"""

import torch
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from typing import Dict, Any, Union, List, Optional
from omegaconf import DictConfig # Use DictConfig to avoid circular import with config.py


class TokenizerWrapper:
    """
    A wrapper class for Hugging Face tokenizers to provide consistent encoding
    and decoding functionalities across the project.
    """

    def __init__(self, model_name: str):
        """
        Initializes the tokenizer from a pre-trained model.

        Args:
            model_name: The name of the pre-trained model for the tokenizer
                        (e.g., "google/gemma-2b").
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set padding_side to "left" for decoder-only models, common in RLHF generation
        self.tokenizer.padding_side = "left"

        # Ensure a pad_token is defined, use eos_token if pad_token is None
        if self.tokenizer.pad_token is None:
            logger.warning(
                f"No pad token found for {model_name}. "
                f"Using eos_token ({self.tokenizer.eos_token}) as pad_token."
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer initialized for model: {model_name}")
        logger.info(f"Tokenizer pad_token: {self.tokenizer.pad_token_id} ({self.tokenizer.pad_token})")
        logger.info(f"Tokenizer eos_token: {self.tokenizer.eos_token_id} ({self.tokenizer.eos_token})")
        logger.info(f"Tokenizer bos_token: {self.tokenizer.bos_token_id} ({self.tokenizer.bos_token})")
        logger.info(f"Tokenizer unk_token: {self.tokenizer.unk_token_id} ({self.tokenizer.unk_token})")
        logger.info(f"Tokenizer padding side: {self.tokenizer.padding_side}")

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: Union[str, bool] = 'max_length',
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes input text(s) into token IDs and an attention mask.

        Args:
            text: The text or list of texts to encode.
            add_special_tokens: Whether to add special tokens (e.g., BOS, EOS).
            max_length: The maximum length to pad/truncate sequences to.
                        If None, uses tokenizer's default or no max length.
            truncation: Whether to truncate sequences longer than max_length.
            padding: Strategy for padding ('longest', 'max_length', True, False).
            return_tensors: Type of tensors to return ('pt' for PyTorch).

        Returns:
            A dictionary containing 'input_ids' and 'attention_mask' as PyTorch tensors.
        """
        if not isinstance(text, (str, list)):
            raise TypeError("Input 'text' must be a string or a list of strings.")
        
        # When padding='max_length', max_length must be provided.
        if padding == 'max_length' and max_length is None:
             raise ValueError("If padding is 'max_length', max_length must be specified.")

        encoding = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
        )
        return encoding

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decodes a tensor of token IDs back into human-readable text.

        Args:
            token_ids: A PyTorch tensor of token IDs. Can be 1D (single sequence)
                       or 2D (batch of sequences).
            skip_special_tokens: Whether to skip special tokens during decoding.

        Returns:
            The decoded string or a list of decoded strings.
        """
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("Input 'token_ids' must be a PyTorch tensor.")

        # Ensure token_ids are on CPU and converted to a list for decoding
        if token_ids.dim() == 1:
            return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=skip_special_tokens)
        elif token_ids.dim() == 2:
            return [
                self.tokenizer.decode(ids.tolist(), skip_special_tokens=skip_special_tokens)
                for ids in token_ids
            ]
        else:
            raise ValueError(f"Unsupported tensor dimension for decoding: {token_ids.dim()}")


def log_metrics(writer: SummaryWriter, metrics: Dict[str, Any], step: int, stage: str):
    """
    Logs training/evaluation metrics to TensorBoard and the console.

    Args:
        writer: The TensorBoard SummaryWriter instance.
        metrics: A dictionary of metrics to log.
        step: The current training/evaluation step.
        stage: The stage name (e.g., 'sft_train', 'ppo_eval').
    """
    logger.info(f"[{stage}] Step {step}: {metrics}")

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            tag = f'{stage}/{key}'
            writer.add_scalar(tag, value, step)
        else:
            logger.warning(
                f"Non-scalar metric '{key}' in stage '{stage}' "
                f"will not be logged to TensorBoard: {value}"
            )


def compute_kl_divergence(
    policy_log_probs: torch.Tensor,
    sft_log_probs: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes the token-level KL divergence between policy and SFT log probabilities.
    The KL divergence is approximated as `policy_log_probs - sft_log_probs`.

    Args:
        policy_log_probs: Log probabilities of tokens from the current policy model (batch_size, sequence_length).
        sft_log_probs: Log probabilities of tokens from the SFT reference model (batch_size, sequence_length).
        attention_mask: Attention mask to ignore padding tokens (batch_size, sequence_length).

    Returns:
        A tensor of token-level KL divergence values, masked for padding tokens
        (batch_size, sequence_length).
    """
    if not (policy_log_probs.shape == sft_log_probs.shape == attention_mask.shape):
        raise ValueError(
            "All input tensors (policy_log_probs, sft_log_probs, attention_mask) "
            "must have the same shape."
        )

    # Calculate token-level KL divergence as the difference in log probabilities
    # A higher policy_log_prob relative to sft_log_prob means a larger penalty
    # if the policy is diverging from the reference.
    kl_per_token = policy_log_probs - sft_log_probs

    # Ensure attention_mask is on the same device as kl_per_token
    attention_mask = attention_mask.to(kl_per_token.device)

    # Mask out KL divergence for padding tokens
    masked_kl_per_token = kl_per_token * attention_mask

    return masked_kl_per_token


if __name__ == "__main__":
    # This block demonstrates how the functions would be used and can be expanded
    # to include more comprehensive tests.
    
    # Mocking a DictConfig for demonstration
    mock_config = DictConfig({
        "model_configs": {
            "test_model": {"name": "gpt2"}
        }
    })

    # --- Test TokenizerWrapper ---
    logger.info("\n--- Testing TokenizerWrapper ---")
    tokenizer_wrapper = TokenizerWrapper(mock_config.model_configs.test_model.name)

    test_text_single = "Hello, world! This is a test sentence."
    test_text_batch = ["Hello, world!", "Another sentence here."]

    # Test encoding
    encoded_single = tokenizer_wrapper.encode(test_text_single, max_length=20)
    logger.info(f"Encoded single text input_ids: {encoded_single['input_ids']}")
    logger.info(f"Encoded single text attention_mask: {encoded_single['attention_mask']}")

    encoded_batch = tokenizer_wrapper.encode(test_text_batch, max_length=20)
    logger.info(f"Encoded batch input_ids: {encoded_batch['input_ids']}")
    logger.info(f"Encoded batch attention_mask: {encoded_batch['attention_mask']}")

    # Test decoding
    decoded_single = tokenizer_wrapper.decode(encoded_single['input_ids'][0])
    logger.info(f"Decoded single text: {decoded_single}")

    decoded_batch = tokenizer_wrapper.decode(encoded_batch['input_ids'])
    logger.info(f"Decoded batch texts: {decoded_batch}")

    # Test edge case for pad_token
    # GPT2 has no pad_token by default, so it should use eos_token
    tokenizer_wrapper_gpt2 = TokenizerWrapper("gpt2")
    assert tokenizer_wrapper_gpt2.tokenizer.pad_token == tokenizer_wrapper_gpt2.tokenizer.eos_token
    logger.info(f"GPT2 tokenizer pad_token: {tokenizer_wrapper_gpt2.tokenizer.pad_token}")


    # --- Test log_metrics ---
    logger.info("\n--- Testing log_metrics ---")
    # Create a dummy SummaryWriter
    if not os.path.exists("runs_test"):
        os.makedirs("runs_test")
    writer_test = SummaryWriter("runs_test/test_log_metrics")

    test_metrics = {
        "loss": 0.123,
        "accuracy": 0.98,
        "list_metric": [1, 2, 3],
        "string_metric": "test_value"
    }
    log_metrics(writer_test, test_metrics, 10, "test_stage")
    writer_test.close()
    logger.info("Check 'runs_test/test_log_metrics' for TensorBoard logs. "
                "Non-scalar metrics should show warnings in console.")
    import shutil
    shutil.rmtree("runs_test", ignore_errors=True)


    # --- Test compute_kl_divergence ---
    logger.info("\n--- Testing compute_kl_divergence ---")
    batch_size = 2
    sequence_length = 5

    # Simulate log probabilities for policy and SFT models
    # Example: policy diverges more in token 1 (index 1) for first sequence
    # and token 3 (index 3) for second sequence
    policy_log_probs_test = torch.tensor([
        [-1.0, -0.5, -2.0, -1.5, -3.0],
        [-1.0, -1.5, -2.0, -0.8, -3.0],
    ], dtype=torch.float32)

    sft_log_probs_test = torch.tensor([
        [-1.0, -1.0, -2.0, -1.5, -3.0],
        [-1.0, -1.5, -2.0, -1.2, -3.0],
    ], dtype=torch.float32)

    # Attention mask with padding (0 for padded tokens)
    attention_mask_test = torch.tensor([
        [1, 1, 1, 1, 0],  # First sequence has one padding token
        [1, 1, 1, 1, 1],  # Second sequence is full
    ], dtype=torch.float32)

    kl_div_result = compute_kl_divergence(
        policy_log_probs_test, sft_log_probs_test, attention_mask_test
    )
    logger.info(f"Policy log probs:\n{policy_log_probs_test}")
    logger.info(f"SFT log probs:\n{sft_log_probs_test}")
    logger.info(f"Attention mask:\n{attention_mask_test}")
    logger.info(f"Computed KL divergence (masked):\n{kl_div_result}")

    # Expected values for kl_div_result:
    # First sequence: [0.0, 0.5, 0.0, 0.0, 0.0] (0.5 for token 1 where policy was higher)
    # Second sequence: [0.0, 0.0, 0.0, 0.4, 0.0] (0.4 for token 3 where policy was higher)
    expected_kl = torch.tensor([
        [0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.4, 0.0],
    ])
    assert torch.allclose(kl_div_result, expected_kl), "KL divergence calculation is incorrect."
    logger.info("KL divergence calculation test passed.")

    # Test with incorrect shapes
    try:
        compute_kl_divergence(policy_log_probs_test, sft_log_probs_test[:, :-1], attention_mask_test)
    except ValueError as e:
        logger.info(f"Caught expected error for shape mismatch: {e}")
