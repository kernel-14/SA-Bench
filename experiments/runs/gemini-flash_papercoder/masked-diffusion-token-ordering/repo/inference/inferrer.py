import torch
import torch.nn.functional as F
import torch.distributions as dist
import random
import logging
import abc # Only for placeholder
from typing import Any, Dict, List, Optional, Tuple, Callable

# Placeholder for Config, TransformerMDM, and NoiseScheduler to avoid circular imports.
# In main.py, the actual imports will be used.
# For this file's standalone integrity and type hinting, placeholders are used.
class _ConfigPlaceholder:
    """
    A placeholder for the Config class. This ensures type hinting and method
    signatures are correctly defined without creating a direct import dependency
    that might lead to circular imports in a larger project structure.
    """
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value from the underlying config dictionary."""
        raise NotImplementedError("This is a placeholder for the Config object. "
                                  "Its 'get' method should not be called directly from here. "
                                  "Ensure the actual Config object is passed and used.")

class _TransformerMDMPlaceholder(torch.nn.Module):
    """
    A placeholder for the TransformerMDM class.
    """
    def __init__(self, config: _ConfigPlaceholder) -> None:
        super().__init__()
    def forward(self, x_t: torch.Tensor, masked_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError("This is a placeholder. Use actual TransformerMDM object.")

class _NoiseSchedulerPlaceholder:
    """
    A placeholder for the NoiseScheduler class.
    """
    def __init__(self, schedule_type: str, num_steps: int) -> None:
        pass
    def get_alpha(self, t: float) -> float:
        raise NotImplementedError("This is a placeholder. Use actual NoiseScheduler object.")
    def get_alpha_prime(self, t: float) -> float:
        raise NotImplementedError("This is a placeholder. Use actual NoiseScheduler object.")
    def get_mask_prob(self, t: float) -> float:
        raise NotImplementedError("This is a placeholder. Use actual NoiseScheduler object.")

# Re-assign for type hinting within this module.
# In the actual project, these would be:
# from config import Config
# from models.transformer_mdm import TransformerMDM
# from utils.noise_schedules import NoiseScheduler
Config = _ConfigPlaceholder
TransformerMDM = _TransformerMDMPlaceholder
NoiseScheduler = _NoiseSchedulerPlaceholder

# Get logger instance. The logger is set up in utils/logger.py and retrieved here.
logger = logging.getLogger("MDM_Project_Logger")


class Inferrer:
    """
    The Inferrer class handles the reverse diffusion process (generation) for MDMs
    using various strategies: vanilla (random) and adaptive (top_probability, top_probability_margin).
    It depends on a trained TransformerMDM and NoiseScheduler.
    """

    def __init__(
        self,
        config: Config,
        model: TransformerMDM,
        tokenizer: Any,
        noise_schedule: NoiseScheduler
    ) -> None:
        """
        Initializes the Inferrer.

        Args:
            config (Config): The global configuration object.
            model (TransformerMDM): An instance of the trained TransformerMDM model.
            tokenizer (Any): A tokenizer instance (e.g., from HuggingFace transformers)
                             or a custom object providing `mask_token_id`.
            noise_schedule (NoiseScheduler): An instance of the NoiseScheduler.
        """
        self.config: Config = config
        self.model: TransformerMDM = model
        self.tokenizer: Any = tokenizer
        self.noise_schedule: NoiseScheduler = noise_schedule

        self.device: str = self.config.get('general.device', 'cuda')
        self.model.eval() # Set model to evaluation mode
        self.model.to(self.device)

        self.max_sequence_length: int = self.config.get('data.max_sequence_length')
        self.vocab_size: int = self.config.get('data.vocab_size')

        # Determine mask_token_id based on tokenizer type or config
        if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
            self.mask_token_id: int = self.tokenizer.mask_token_id
        else:
            self.mask_token_id: int = self.config.get('data.mask_token_id', 0)
        
        self.gaussian_noise_std: float = self.config.get('inference.gaussian_noise_std', 0.0)
        self.gumbel_noise_coeff: float = self.config.get('inference.gumbel_noise_coeff', 0.0)

        # Dataset type for conditional noise application
        self.dataset_type: str = self.config.get('data.dataset_type', 'unknown')

        logger.info(f"Inferrer initialized on device: {self.device}")
        logger.info(f"Max sequence length: {self.max_sequence_length}, Vocab size: {self.vocab_size}, Mask token ID: {self.mask_token_id}")
        if self.gaussian_noise_std > 0:
            logger.info(f"Gaussian noise std for text-like tasks: {self.gaussian_noise_std}")
        if self.gumbel_noise_coeff > 0:
            logger.info(f"Gumbel noise coefficient for puzzle-like tasks: {self.gumbel_noise_coeff}")


    def _get_denoising_logits(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Computes the predicted logits for all token positions from the TransformerMDM
        given a partially masked sequence x_t.

        Args:
            x_t (torch.Tensor): The partially masked input sequence.
                                Shape: (batch_size, sequence_length).

        Returns:
            torch.Tensor: Logits for the vocabulary over all positions.
                          Shape: (batch_size, sequence_length, vocab_size).
        """
        with torch.no_grad(): # Inference should not calculate gradients
            # Ensure input is on the correct device
            x_t = x_t.to(self.device)
            all_logits: torch.Tensor = self.model(x_t)
        return all_logits

    def _calculate_k(self, num_masked_tokens: int, current_t_val: float, prev_t_val: float) -> int:
        """
        Calculates the number of tokens K to unmask in the current reverse step,
        based on the paper's formula (Section D.1.2).

        Args:
            num_masked_tokens (int): The number of tokens currently masked in the sequence.
            current_t_val (float): The continuous time value of the current sequence (x_t).
            prev_t_val (float): The continuous time value of the previous step in the schedule (x_s).

        Returns:
            int: The calculated number of tokens to unmask (K).
        """
        if num_masked_tokens == 0:
            return 0

        # Retrieve alpha values for the current and previous noise levels
        alpha_s: float = self.noise_schedule.get_alpha(prev_t_val)
        alpha_t: float = self.noise_schedule.get_alpha(current_t_val)

        # Apply the formula for K, adding epsilon for numerical stability if (1 - alpha_t) is near zero
        denominator = (1.0 - alpha_t)
        if denominator < 1e-8: # Avoid division by zero when alpha_t is very close to 1
            logger.warning(f"Denominator (1 - alpha_t) near zero ({denominator}) for t={current_t_val}. Setting K to 0.")
            return 0
            
        k_float: float = num_masked_tokens * (alpha_s - alpha_t) / denominator

        # Round to the nearest integer and ensure K is within valid bounds
        k: int = max(0, min(int(round(k_float)), num_masked_tokens))
        return k

    def _select_mask_indices(
        self,
        x_t: torch.Tensor,
        all_logits: torch.Tensor,
        current_t_idx: int,
        t_schedule: torch.Tensor,
        strategy: str
    ) -> List[int]:
        """
        Selects the indices of K tokens to unmask based on the specified adaptive strategy.

        Args:
            x_t (torch.Tensor): The current partially masked sequence (batch_size=1 assumed).
                                Shape: (1, sequence_length).
            all_logits (torch.Tensor): The model's predicted logits for the entire sequence.
                                       Shape: (1, sequence_length, vocab_size).
            current_t_idx (int): The index in the t_schedule corresponding to current_t_val.
            t_schedule (torch.Tensor): The discrete time schedule tensor.
            strategy (str): The adaptive selection method ("top_probability" or "top_probability_margin").

        Returns:
            List[int]: A list of indices of the tokens to be unmasked in this step.
        """
        # Identify currently masked positions
        masked_positions_mask: torch.Tensor = (x_t[0] == self.mask_token_id)
        # Convert boolean mask to 1D tensor of actual indices
        masked_indices: torch.Tensor = torch.nonzero(masked_positions_mask).squeeze(1)

        if masked_indices.numel() == 0: # No tokens left to unmask
            return []

        # Determine current and previous time values from the schedule
        # t_schedule goes from 1.0 down to 0.0
        current_t_val: float = t_schedule[current_t_idx].item()
        prev_t_val: float = t_schedule[current_t_idx - 1].item() # prev_t_val is always less noisy than current_t_val

        # Calculate K, the number of tokens to unmask
        num_masked_tokens: int = masked_indices.size(0)
        k_to_unmask: int = self._calculate_k(num_masked_tokens, current_t_val, prev_t_val)

        if k_to_unmask == 0:
            return []

        # Extract logits corresponding to currently masked positions
        # x_t is (1, seq_len), all_logits is (1, seq_len, vocab_size)
        masked_logits: torch.Tensor = all_logits[0, masked_indices, :]
        masked_probabilities: torch.Tensor = F.softmax(masked_logits, dim=-1)

        certainty_scores: torch.Tensor
        if strategy == "top_probability":
            # Certainty is the maximum probability assigned to any token
            certainty_scores = torch.max(masked_probabilities, dim=-1).values
        elif strategy == "top_probability_margin":
            # Certainty is the margin between the two most probable tokens
            # Sort probabilities in descending order
            sorted_probs: torch.Tensor = torch.sort(masked_probabilities, dim=-1, descending=True).values
            # Compute difference between top1 and top2 probabilities
            certainty_scores = sorted_probs[:, 0] - sorted_probs[:, 1]
        else:
            raise ValueError(f"Unknown adaptive inference strategy: {strategy}")

        # Apply noise if configured
        if self.dataset_type in ["slimpajama", "llada_tasks"]: # Text-like tasks
            if self.gaussian_noise_std > 0:
                epsilon: torch.Tensor = torch.randn_like(certainty_scores) * self.gaussian_noise_std
                certainty_scores += epsilon
        elif self.dataset_type in ["sudoku", "zebra", "lo_naesat"]: # Puzzle-like tasks
            if self.gumbel_noise_coeff > 0:
                # Gumbel noise is typically added to log-probabilities or scores for sampling
                # Here, applying it to certainty scores for selecting indices
                gumbel_noise: torch.Tensor = dist.Gumbel(loc=0.0, scale=1.0).sample(certainty_scores.shape).to(self.device)
                certainty_scores += gumbel_noise * self.gumbel_noise_coeff

        # Select the top K indices based on certainty scores
        # Ensure k_to_unmask does not exceed the number of available masked tokens
        k_to_select = min(k_to_unmask, num_masked_tokens)
        _, top_k_relative_indices: torch.Tensor = torch.topk(certainty_scores, k=k_to_select)
        
        # Map relative indices back to original sequence indices
        selected_indices: torch.Tensor = masked_indices[top_k_relative_indices]

        return selected_indices.tolist()

    def _sample_tokens(self, logits: torch.Tensor, indices_to_sample: List[int]) -> torch.Tensor:
        """
        Samples token values for a given list of indices from the model's output logits.

        Args:
            logits (torch.Tensor): The model's full output logits.
                                   Shape: (batch_size, sequence_length, vocab_size).
                                   (batch_size=1 assumed for this method's usage pattern).
            indices_to_sample (List[int]): A list of token indices (positions) in the sequence
                                           for which to sample new token values.

        Returns:
            torch.Tensor: A 1D tensor containing the sampled token IDs for the given positions.
                          Shape: (len(indices_to_sample),).
        """
        if not indices_to_sample:
            return torch.empty(0, dtype=torch.long, device=self.device)

        # Extract logits specifically for the positions to be sampled
        # logits is (1, seq_len, vocab_size), so we select from the first batch item
        selected_logits: torch.Tensor = logits[0, indices_to_sample, :]

        # Create a Categorical distribution and sample
        sampled_token_ids: torch.Tensor = dist.Categorical(logits=selected_logits).sample()
        
        return sampled_token_ids

    def vanilla_inference(self, initial_x_t: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Performs the standard, random-order MDM inference (reverse diffusion process).

        Args:
            initial_x_t (torch.Tensor): The starting sequence for inference,
                                        expected to be fully masked.
                                        Shape: (1, sequence_length).
            num_steps (int): The number of discrete reverse diffusion steps.

        Returns:
            torch.Tensor: The final denoised (generated) sequence.
                          Shape: (1, sequence_length).
        """
        logger.info(f"Starting vanilla inference with {num_steps} steps.")
        current_x_t: torch.Tensor = initial_x_t.clone().to(self.device)

        # Create a linear schedule of time values from 1.0 (fully masked) down to 0.0 (fully unmasked)
        t_schedule: torch.Tensor = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)

        # Iterate backward through the time schedule (from current_t_idx = num_steps down to 1)
        # t_schedule[num_steps] is 1.0, t_schedule[0] is 0.0
        for t_idx in range(num_steps, 0, -1):
            # Get model predictions (logits for original tokens) based on current partially masked sequence
            all_logits: torch.Tensor = self._get_denoising_logits(current_x_t)

            # Identify currently masked positions
            masked_positions_mask: torch.Tensor = (current_x_t[0] == self.mask_token_id)
            masked_indices: torch.Tensor = torch.nonzero(masked_positions_mask).squeeze(1)

            if masked_indices.numel() == 0:
                logger.debug(f"Step {num_steps - t_idx + 1}/{num_steps}: No more tokens masked. Early stop.")
                break # All tokens are unmasked

            # Determine current and previous time values from the schedule
            current_t_val: float = t_schedule[t_idx].item()
            prev_t_val: float = t_schedule[t_idx - 1].item()

            # Calculate K, the number of tokens to unmask in this step
            num_masked_tokens: int = masked_indices.size(0)
            k_to_unmask: int = self._calculate_k(num_masked_tokens, current_t_val, prev_t_val)
            
            if k_to_unmask == 0:
                logger.debug(f"Step {num_steps - t_idx + 1}/{num_steps}: K_to_unmask is 0. No tokens unmasked.")
                continue # No tokens to unmask in this step, move to the next

            # Randomly select K tokens to unmask from the currently masked ones
            selected_indices_list: List[int] = random.sample(masked_indices.tolist(), k=k_to_unmask)

            # Sample new token values for the selected indices
            sampled_token_ids: torch.Tensor = self._sample_tokens(all_logits, selected_indices_list)

            # Update the sequence with the newly sampled tokens
            current_x_t[0, torch.tensor(selected_indices_list, device=self.device)] = sampled_token_ids

            if (num_steps - t_idx + 1) % (num_steps // 10 if num_steps >= 10 else 1) == 0:
                logger.debug(f"Step {num_steps - t_idx + 1}/{num_steps}: Unmasked {k_to_unmask} tokens. Remaining masked: {num_masked_tokens - k_to_unmask}.")

        logger.info(f"Vanilla inference completed. Final sequence (first 10): {current_x_t[0, :10].tolist()}")
        return current_x_t

    def adaptive_inference(self, initial_x_t: torch.Tensor, strategy: str, num_steps: int) -> torch.Tensor:
        """
        Performs MDM inference using an adaptive token unmasking strategy.

        Args:
            initial_x_t (torch.Tensor): The starting sequence for inference,
                                        expected to be fully masked.
                                        Shape: (1, sequence_length).
            strategy (str): The adaptive selection method ("top_probability" or "top_probability_margin").
            num_steps (int): The number of discrete reverse diffusion steps.

        Returns:
            torch.Tensor: The final denoised (generated) sequence.
                          Shape: (1, sequence_length).
        """
        if strategy not in ["top_probability", "top_probability_margin"]:
            raise ValueError(f"Unsupported adaptive strategy: {strategy}. Must be 'top_probability' or 'top_probability_margin'.")
            
        logger.info(f"Starting adaptive inference with strategy '{strategy}' and {num_steps} steps.")
        current_x_t: torch.Tensor = initial_x_t.clone().to(self.device)

        # Create a linear schedule of time values from 1.0 (fully masked) down to 0.0 (fully unmasked)
        t_schedule: torch.Tensor = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)

        # Iterate backward through the time schedule (from current_t_idx = num_steps down to 1)
        for t_idx in range(num_steps, 0, -1):
            # Get model predictions (logits for original tokens)
            all_logits: torch.Tensor = self._get_denoising_logits(current_x_t)

            # Select tokens to unmask adaptively
            selected_indices_list: List[int] = self._select_mask_indices(
                current_x_t, all_logits, t_idx, t_schedule, strategy
            )

            if not selected_indices_list:
                # Check if there are still any masked tokens left in the sequence itself
                remaining_masked_count = (current_x_t[0] == self.mask_token_id).sum().item()
                if remaining_masked_count == 0:
                    logger.debug(f"Step {num_steps - t_idx + 1}/{num_steps}: No more tokens masked. Early stop.")
                    break
                logger.debug(f"Step {num_steps - t_idx + 1}/{num_steps}: _select_mask_indices returned empty list (K=0 or no masked tokens available).")
                continue # No tokens were selected to be unmasked in this step, move to the next

            # Sample new token values for the selected indices
            sampled_token_ids: torch.Tensor = self._sample_tokens(all_logits, selected_indices_list)

            # Update the sequence with the newly sampled tokens
            current_x_t[0, torch.tensor(selected_indices_list, device=self.device)] = sampled_token_ids
            
            if (num_steps - t_idx + 1) % (num_steps // 10 if num_steps >= 10 else 1) == 0:
                logger.debug(f"Step {num_steps - t_idx + 1}/{num_steps}: Unmasked {len(selected_indices_list)} tokens. Remaining masked: {(current_x_t[0] == self.mask_token_id).sum().item()}.")

        logger.info(f"Adaptive inference with '{strategy}' completed. Final sequence (first 10): {current_x_t[0, :10].tolist()}")
        return current_x_t


if __name__ == '__main__':
    # This block is for testing the Inferrer module in isolation.

    # Mock classes for testing
    class MockConfig(_ConfigPlaceholder):
        def __init__(self, data: Dict[str, Any]):
            self._data = data
        def get(self, key: str, default: Any = None) -> Any:
            keys = key.split('.')
            current = self._data
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            return current

    class MockTransformerMDM(TransformerMDM):
        def __init__(self, vocab_size: int, hidden_dim: int):
            super().__init__(None) # Pass None for config as it's a mock
            self._vocab_size = vocab_size
            self._hidden_dim = hidden_dim
            # Simulate a linear layer for logits, for testing purposes
            self.output_layer = torch.nn.Linear(hidden_dim, vocab_size)

        def forward(self, x_t: torch.Tensor, masked_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
            batch_size, seq_len = x_t.size()
            # Simulate some dummy hidden states
            dummy_hidden_states = torch.randn(batch_size, seq_len, self._hidden_dim, device=x_t.device)
            logits = self.output_layer(dummy_hidden_states)
            return logits

    class MockNoiseScheduler(NoiseScheduler):
        def __init__(self):
            super().__init__("linear", 1000) # Dummy init
        def get_alpha(self, t: float) -> float:
            # Simple linear schedule for alpha_t (1 at t=0, 0 at t=1)
            return 1.0 - t
        def get_alpha_prime(self, t: float) -> float:
            return -1.0
        def get_mask_prob(self, t: float) -> float:
            return t # mask_prob = 1 - alpha_t

    class MockTokenizer:
        def __init__(self, mask_token_id: int):
            self.mask_token_id = mask_token_id

    # Setup a mock logger for demonstration
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("MDM_Project_Logger")

    print("--- Testing Inferrer Module ---")

    # Configuration for testing
    TEST_VOCAB_SIZE = 10
    TEST_MASK_TOKEN_ID = 0
    TEST_SEQ_LEN = 20
    TEST_HIDDEN_DIM = 64
    TEST_NUM_INFERENCE_STEPS = 10
    
    # Text-like config
    mock_config_text = MockConfig({
        'general': {'device': 'cpu'},
        'data': {
            'max_sequence_length': TEST_SEQ_LEN,
            'vocab_size': TEST_VOCAB_SIZE,
            'mask_token_id': TEST_MASK_TOKEN_ID,
            'dataset_type': 'slimpajama' # To trigger gaussian noise
        },
        'inference': {
            'gaussian_noise_std': 0.1,
            'gumbel_noise_coeff': 0.0 # Not used for text
        }
    })

    # Puzzle-like config
    mock_config_puzzle = MockConfig({
        'general': {'device': 'cpu'},
        'data': {
            'max_sequence_length': TEST_SEQ_LEN,
            'vocab_size': TEST_VOCAB_SIZE,
            'mask_token_id': TEST_MASK_TOKEN_ID,
            'dataset_type': 'sudoku' # To trigger gumbel noise
        },
        'inference': {
            'gaussian_noise_std': 0.0, # Not used for puzzle
            'gumbel_noise_coeff': 0.5
        }
    })

    mock_model = MockTransformerMDM(TEST_VOCAB_SIZE, TEST_HIDDEN_DIM)
    mock_tokenizer = MockTokenizer(TEST_MASK_TOKEN_ID)
    mock_noise_scheduler = MockNoiseScheduler()

    # --- Test Inferrer with Text-like settings ---
    print("\n--- Inferrer with Text-like (Gaussian Noise) settings ---")
    inferrer_text = Inferrer(mock_config_text, mock_model, mock_tokenizer, mock_noise_scheduler)

    # Prepare an initial fully masked sequence (batch_size=1)
    initial_masked_seq = torch.full((1, TEST_SEQ_LEN), TEST_MASK_TOKEN_ID, dtype=torch.long)
    
    # Fill some positions with actual tokens to test mixed sequences
    initial_masked_seq[0, 0] = 1 # Keep first token unmasked for partial check
    initial_masked_seq[0, 5] = 2 # Keep 6th token unmasked

    print("\n--- Testing Vanilla Inference (Text-like) ---")
    vanilla_output = inferrer_text.vanilla_inference(initial_masked_seq, TEST_NUM_INFERENCE_STEPS)
    print(f"Vanilla Inference Output: {vanilla_output[0].tolist()}")
    print(f"Number of masked tokens remaining: {(vanilla_output[0] == TEST_MASK_TOKEN_ID).sum().item()}")
    assert (vanilla_output[0] == TEST_MASK_TOKEN_ID).sum().item() == 0, "Vanilla inference should ideally unmask all tokens"
    assert vanilla_output.shape == (1, TEST_SEQ_LEN)
    assert initial_masked_seq[0,0] == 1 # Check fixed tokens are unchanged
    assert initial_masked_seq[0,5] == 2
    
    print("\n--- Testing Adaptive Inference (Top Probability, Text-like) ---")
    adaptive_top_prob_output = inferrer_text.adaptive_inference(initial_masked_seq, "top_probability", TEST_NUM_INFERENCE_STEPS)
    print(f"Adaptive (Top Prob) Inference Output: {adaptive_top_prob_output[0].tolist()}")
    print(f"Number of masked tokens remaining: {(adaptive_top_prob_output[0] == TEST_MASK_TOKEN_ID).sum().item()}")
    assert (adaptive_top_prob_output[0] == TEST_MASK_TOKEN_ID).sum().item() == 0, "Adaptive inference should ideally unmask all tokens"
    assert adaptive_top_prob_output.shape == (1, TEST_SEQ_LEN)

    print("\n--- Testing Adaptive Inference (Top Probability Margin, Text-like) ---")
    adaptive_top_margin_output = inferrer_text.adaptive_inference(initial_masked_seq, "top_probability_margin", TEST_NUM_INFERENCE_STEPS)
    print(f"Adaptive (Top Prob Margin) Inference Output: {adaptive_top_margin_output[0].tolist()}")
    print(f"Number of masked tokens remaining: {(adaptive_top_margin_output[0] == TEST_MASK_TOKEN_ID).sum().item()}")
    assert (adaptive_top_margin_output[0] == TEST_MASK_TOKEN_ID).sum().item() == 0, "Adaptive inference should ideally unmask all tokens"
    assert adaptive_top_margin_output.shape == (1, TEST_SEQ_LEN)

    # --- Test Inferrer with Puzzle-like settings ---
    print("\n--- Inferrer with Puzzle-like (Gumbel Noise) settings ---")
    inferrer_puzzle = Inferrer(mock_config_puzzle, mock_model, mock_tokenizer, mock_noise_scheduler)

    # Prepare an initial fully masked sequence (batch_size=1)
    initial_masked_seq_puzzle = torch.full((1, TEST_SEQ_LEN), TEST_MASK_TOKEN_ID, dtype=torch.long)
    initial_masked_seq_puzzle[0, 0] = 3 # Keep first token unmasked for partial check
    initial_masked_seq_puzzle[0, 7] = 4 # Keep 8th token unmasked

    print("\n--- Testing Adaptive Inference (Top Probability, Puzzle-like) ---")
    adaptive_top_prob_puzzle_output = inferrer_puzzle.adaptive_inference(initial_masked_seq_puzzle, "top_probability", TEST_NUM_INFERENCE_STEPS)
    print(f"Adaptive (Top Prob) Puzzle Inference Output: {adaptive_top_prob_puzzle_output[0].tolist()}")
    print(f"Number of masked tokens remaining: {(adaptive_top_prob_puzzle_output[0] == TEST_MASK_TOKEN_ID).sum().item()}")
    assert (adaptive_top_prob_puzzle_output[0] == TEST_MASK_TOKEN_ID).sum().item() == 0, "Adaptive puzzle inference should ideally unmask all tokens"

    print("\n--- Testing Adaptive Inference (Top Probability Margin, Puzzle-like) ---")
    adaptive_top_margin_puzzle_output = inferrer_puzzle.adaptive_inference(initial_masked_seq_puzzle, "top_probability_margin", TEST_NUM_INFERENCE_STEPS)
    print(f"Adaptive (Top Prob Margin) Puzzle Inference Output: {adaptive_top_margin_puzzle_output[0].tolist()}")
    print(f"Number of masked tokens remaining: {(adaptive_top_margin_puzzle_output[0] == TEST_MASK_TOKEN_ID).sum().item()}")
    assert (adaptive_top_margin_puzzle_output[0] == TEST_MASK_TOKEN_ID).sum().item() == 0, "Adaptive puzzle inference should ideally unmask all tokens"
    
    print("\n--- Inferrer testing complete ---")

