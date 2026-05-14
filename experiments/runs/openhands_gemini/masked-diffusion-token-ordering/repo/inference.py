
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random

from model import MaskedDiffusionModel, MDMConfig
from config import Config

class MDMInferrer:
    def __init__(self, model: MaskedDiffusionModel, config: Config):
        self.model = model
        self.config = config
        self.device = config.device
        self.mask_token_id = config.mask_token_id

        # Noise schedule for masking (same as in training)
        self.mask_probs = np.linspace(0.0, 1.0, config.num_diffusion_steps)

    def _get_num_unmask_tokens(self, current_masked_count: int, step: int, total_steps: int):
        """
        Calculates the number of tokens to unmask at the current step.
        This is an approximation based on the idea that at each step, a certain
        fraction of the remaining masked tokens are unmasked.
        For simplicity, we can aim to unmask a roughly equal number of tokens
        at each step, or a fraction of the currently masked tokens.

        Section D.1.2 states:
        "For an inference transition from step t to s, vanilla MDM expects
        (# mask tokens in the current x_t) * (alpha_s - alpha_t) / (1 - alpha_t) unmasked."
        
        If we assume alpha_t decreases linearly from 1 to 0, and we have `total_steps`
        Then each step `k` (from 0 to total_steps-1) corresponds to a `t` value.
        Let `t_k = 1 - k/total_steps`. Then `alpha_k = t_k`.
        Then `alpha_{k+1} - alpha_k = -1/total_steps`.
        And `1 - alpha_k = 1 - t_k = k/total_steps`.
        This doesn't seem to make sense directly as a fraction.

        A more straightforward approach for fixed `sampling_steps` (say 50):
        At each step, unmask `total_masked_tokens / sampling_steps` tokens.
        Or, unmask `fraction_to_unmask * current_masked_count`.
        Let's use a simpler strategy: unmask a proportion of the current masked tokens.
        """
        # Determine the target number of unmasked tokens for this step
        # This can be `max_sequence_length / total_steps` for linear progression
        # Or a fraction of currently masked tokens.
        # Let's say we want to unmask roughly `max_sequence_length / self.config.sampling_steps` tokens per step.
        
        # This aims to unmask tokens gradually.
        tokens_to_unmask_per_step = self.config.max_sequence_length / total_steps
        
        # Ensure we don't unmask more than available or less than 1 (if masked_count > 0)
        num_unmask = max(1, min(current_masked_count, int(tokens_to_unmask_per_step)))
        return num_unmask

    def _get_masked_positions(self, current_sequence: torch.Tensor):
        return (current_sequence == self.mask_token_id).nonzero(as_tuple=True)[1]

    @torch.no_grad()
    def infer(self, initial_sequence: torch.Tensor, strategy: str = "vanilla", num_steps: int = None):
        """
        Performs MDM inference to denoise a sequence.
        Args:
            initial_sequence: The input sequence, potentially partially masked or fully masked.
                              Shape: (batch_size, sequence_length).
                              The values should be token IDs, with self.mask_token_id for masked positions.
            strategy: "vanilla", "top_probability", "top_probability_margin".
            num_steps: Number of reverse sampling steps. If None, uses config.sampling_steps.
        Returns:
            decoded_sequence: The fully unmasked sequence.
                              Shape: (batch_size, sequence_length)
        """
        self.model.eval()
        
        if num_steps is None:
            num_steps = self.config.sampling_steps

        # Start with a fully masked sequence if initial_sequence contains no masks, or use it as is.
        # The paper says "The reverse sampling process starts from the fully masked sentence x_1 = (0, ..., 0)"
        # But also, "Suppose we have a partially masked sequence x_t"
        # So we can start from a partially masked input if provided, or fully mask it.
        
        # Let's assume initial_sequence is the partially masked x_t (or x_1)
        current_x = initial_sequence.clone().to(self.device)
        batch_size, seq_len = current_x.shape

        for step_idx in tqdm(range(num_steps), desc=f"MDM Inference ({strategy})"):
            masked_positions_batch = [self._get_masked_positions(current_x[b]) for b in range(batch_size)]
            
            # If all sequences in batch are fully unmasked, break
            if all(len(pos) == 0 for pos in masked_positions_batch):
                break

            # Get logits for all positions
            logits = self.model(current_x) # (batch_size, sequence_length, vocab_size)

            newly_unmasked_tokens_batch = torch.full_like(current_x, self.mask_token_id)
            
            for b in range(batch_size):
                masked_indices = masked_positions_batch[b]
                if len(masked_indices) == 0:
                    continue # This sequence is already fully unmasked

                # Calculate number of tokens to unmask in this step for this sequence
                num_to_unmask = self._get_num_unmask_tokens(len(masked_indices), step_idx, num_steps)
                
                # Get relevant logits for masked positions
                masked_logits = logits[b, masked_indices, :] # (num_masked_tokens_in_this_seq, vocab_size)

                # Apply oracle strategy
                selected_indices_in_masked_logits = None
                if strategy == "vanilla":
                    # Vanilla: randomly select `num_to_unmask` from masked_indices
                    selected_indices_in_masked_logits = torch.randperm(len(masked_indices), device=self.device)[:num_to_unmask]
                elif strategy == "top_probability":
                    # Top probability: select positions where the model is most "certain"
                    # certainty = max_j p_theta(x^i = j | x_t)
                    # For logits, this is just max(softmax(logits), dim=-1) or max(logits, dim=-1) if comparing directly
                    # We need to compute probabilities or use logits directly for ranking. Using logits is simpler.
                    max_probs, _ = torch.max(masked_logits, dim=-1) # (num_masked_tokens_in_this_seq,)
                    if self.config.gumbel_noise_coeff > 0:
                        max_probs += self.gumbel_noise(max_probs.shape, device=self.device) * self.config.gumbel_noise_coeff
                    
                    # Select top K positions
                    _, sorted_indices = torch.topk(max_probs, k=num_to_unmask, largest=True, sorted=False)
                    selected_indices_in_masked_logits = sorted_indices

                elif strategy == "top_probability_margin":
                    # Top probability margin: select positions with largest margin between top two probabilities
                    # certainty = |p1 - p2|
                    # Convert logits to probabilities
                    probs = F.softmax(masked_logits, dim=-1) # (num_masked_tokens_in_this_seq, vocab_size)
                    
                    # Get top 2 probabilities
                    top_probs, _ = torch.topk(probs, k=2, dim=-1, largest=True, sorted=True) # (num_masked_tokens_in_this_seq, 2)
                    
                    # Calculate margin
                    margins = (top_probs[:, 0] - top_probs[:, 1]).abs() # (num_masked_tokens_in_this_seq,)
                    
                    if self.config.gumbel_noise_coeff > 0:
                        margins += self.gumbel_noise(margins.shape, device=self.device) * self.config.gumbel_noise_coeff
                        
                    # Select top K positions
                    _, sorted_indices = torch.topk(margins, k=num_to_unmask, largest=True, sorted=False)
                    selected_indices_in_masked_logits = sorted_indices
                else:
                    raise ValueError(f"Unknown inference strategy: {strategy}")

                # Get the actual sequence indices to unmask
                indices_to_unmask_in_seq = masked_indices[selected_indices_in_masked_logits]
                
                # Sample new tokens for these positions
                predicted_logits_for_sampling = logits[b, indices_to_unmask_in_seq, :]
                
                # Sample from categorical distribution
                sampled_tokens = torch.distributions.Categorical(logits=predicted_logits_for_sampling).sample()
                
                # Update the current sequence with sampled tokens
                current_x[b, indices_to_unmask_in_seq] = sampled_tokens
                newly_unmasked_tokens_batch[b, indices_to_unmask_in_seq] = sampled_tokens
            
        return current_x

    def gumbel_noise(self, shape, eps=1e-10, device="cpu"):
        """
        Generates Gumbel noise.
        """
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + eps) + eps)


if __name__ == "__main__":
    # Example usage:
    config = Config()
    
    # Update config for specific experiment, e.g., Sudoku
    config.dataset_name = "Sudoku"
    config.vocab_size = 10 # 0-9
    config.max_sequence_length = 81
    config.num_train_epochs = 1 # Just for demonstration, not actual training
    config.sampling_steps = 50
    config.gumbel_noise_coeff = 0.5

    print(f"Using device: {config.device}")
    print(f"Inference on {config.dataset_name} dataset.")

    # Create a dummy model (in a real scenario, load a trained model)
    mdm_config = MDMConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_layers=config.num_layers,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        max_sequence_length=config.max_sequence_length,
        initializer_range=config.initializer_range,
        layer_norm_eps=config.layer_norm_eps,
        use_learnable_pos_embeddings=config.use_learnable_pos_embeddings,
        pad_token_id=config.mask_token_id
    )
    model = MaskedDiffusionModel(mdm_config).to(config.device)
    # Load some dummy weights or pre-trained if available
    # For demonstration, we'll just use random weights

    inferrer = MDMInferrer(model, config)

    # Create a synthetic fully masked Sudoku puzzle for inference
    batch_size = 2
    fully_masked_puzzle = torch.full(
        (batch_size, config.max_sequence_length), 
        config.mask_token_id, 
        dtype=torch.long, 
        device=config.device
    )

    print("\n--- Vanilla Inference ---")
    decoded_vanilla = inferrer.infer(fully_masked_puzzle, strategy="vanilla")
    print("Decoded (Vanilla):\n", decoded_vanilla)

    print("\n--- Top Probability Inference ---")
    decoded_top_prob = inferrer.infer(fully_masked_puzzle, strategy="top_probability")
    print("Decoded (Top Probability):\n", decoded_top_prob)

    print("\n--- Top Probability Margin Inference ---")
    decoded_top_prob_margin = inferrer.infer(fully_masked_puzzle, strategy="top_probability_margin")
    print("Decoded (Top Probability Margin):\n", decoded_top_prob_margin)

