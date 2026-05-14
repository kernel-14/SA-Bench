import torch
from torch import nn
from torch.nn import functional as F
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

# Placeholder for Config to avoid circular imports.
# In main.py, the actual Config object will be imported.
# For this file's standalone integrity and type hinting, a placeholder is used.
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

# Re-assign for type hinting within this module.
# In the actual project, this would be: `from config import Config`
Config = _ConfigPlaceholder

# Get logger instance. The logger is set up in utils/logger.py and retrieved here.
logger = logging.getLogger("MDM_Project_Logger")


class CausalSelfAttention(nn.Module):
    """
    A multi-head self-attention module with a causal mask, ensuring that tokens
    can only attend to previous tokens in the sequence.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        """
        Initializes the CausalSelfAttention module.

        Args:
            hidden_dim (int): The dimensionality of the input and output features.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout probability.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads: int = num_heads
        self.head_dim: int = hidden_dim // num_heads
        self.hidden_dim: int = hidden_dim

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim)
        # Output projection
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask to ensure that attention is only paid to the left.
        # This is a buffer, meaning it's part of the model's state but not a trainable parameter.
        self.register_buffer("bias", torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024))
        logger.debug(f"CausalSelfAttention initialized with hidden_dim={hidden_dim}, num_heads={num_heads}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for CausalSelfAttention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        batch_size, seq_len, hidden_dim = x.size()

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # (B, T, 3*H) -> (B, T, 3, num_heads, head_dim) -> (3, B, num_heads, T, head_dim)
        qkv = self.c_attn(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # q, k, v are (B, num_heads, T, head_dim)

        # Causal self-attention; (B, num_heads, T, head_dim) x (B, num_heads, head_dim, T) -> (B, num_heads, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask.
        # The bias needs to be resized if current seq_len exceeds its pre-allocated size
        # This can happen if max_sequence_length is larger than the bias's internal size (1024 here).
        # Typically, a fixed max_sequence_length should be used to define this bias initially.
        if seq_len > self.bias.size(-1):
             # This is a fallback and can be inefficient. Ideally, bias should be pre-allocated
             # for the maximum possible sequence length.
             logger.warning(f"Sequence length {seq_len} exceeds pre-allocated attention bias size {self.bias.size(-1)}. "
                            "Dynamically resizing bias, which might impact performance. Consider increasing bias size.")
             # Recreate a larger bias or handle this by padding the input to a smaller max_len.
             # For now, we'll slice/expand as best as possible.
             new_bias = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len).to(attn_scores.device)
             self.bias = new_bias # Update the buffer for future calls (if this is acceptable behavior)
             attn_scores = attn_scores.masked_fill(new_bias[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        else:
            attn_scores = attn_scores.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # (B, num_heads, T, T) x (B, num_heads, T, head_dim) -> (B, num_heads, T, head_dim)
        y = attn_weights @ v 
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    A simple two-layer Multi-Layer Perceptron (MLP) with GELU activation and dropout.
    """
    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float) -> None:
        """
        Initializes the MLP module.

        Args:
            hidden_dim (int): The dimensionality of the input and output features.
            ff_dim (int): The dimensionality of the inner layer (feed-forward dimension).
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.c_fc = nn.Linear(hidden_dim, ff_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        logger.debug(f"MLP initialized with hidden_dim={hidden_dim}, ff_dim={ff_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer block, comprising a CausalSelfAttention layer and an MLP.
    It includes layer normalization and residual connections.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        """
        Initializes a TransformerBlock.

        Args:
            hidden_dim (int): The dimensionality of the input features.
            num_heads (int): The number of attention heads.
            ff_dim (int): The feed-forward dimension of the MLP.
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, ff_dim, dropout)
        logger.debug(f"TransformerBlock initialized with hidden_dim={hidden_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class PiLearnerARM(nn.Module):
    """
    Implements an Autoregressive Model (ARM) that acts as a pi-learner.
    It's a causal Transformer designed to predict the next token given a permuted
    input sequence, using learnable positional embeddings.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the PiLearnerARM model.

        Args:
            config (Config): The global configuration object.

        Raises:
            ValueError: If essential parameters are missing or if
                        `use_learnable_pos_embeddings` is not explicitly True.
        """
        super().__init__()
        self.config: Config = config
        self.model_params: Dict[str, Any] = {}

        # --- 1. Load common model parameters ---
        common_model_config = self.config.get('model.common', {})
        self.model_params.update(common_model_config)

        # --- 2. Determine and load size-specific parameters ---
        # The 'model.architecture_size' key specifies which size config (e.g., '42M') to use.
        # For ARM, "42M" is specified in the paper's tables for Sudoku/Zebra.
        model_size_key: str = self.config.get('model.architecture_size', '42M')
        
        # Override common parameters with size-specific parameters if they exist
        size_specific_config = self.config.get(f'model.size_configs.{model_size_key}', {})
        if size_specific_config:
            self.model_params.update(size_specific_config)
        
        # --- 3. Add essential data-related parameters required by the model ---
        vocab_size = self.config.get('data.vocab_size')
        if vocab_size is None:
            raise ValueError("Configuration 'data.vocab_size' is required for model initialization. "
                             "Please ensure it is set in the config.yaml.")
        self.model_params['vocab_size'] = vocab_size
            
        max_sequence_length = self.config.get('data.max_sequence_length')
        if max_sequence_length is None:
            raise ValueError("Configuration 'data.max_sequence_length' is required for model initialization. "
                             "Please ensure it is set in the config.yaml.")
        self.model_params['max_sequence_length'] = max_sequence_length

        # --- 4. Ensure all critical parameters have sensible defaults ---
        self.model_params.setdefault('num_layers', 8)  # Default for 42M ARM based on tables/common sense
        self.model_params.setdefault('num_heads', 8)
        self.model_params.setdefault('hidden_dim', 768)
        self.model_params.setdefault('ff_dim', 3072)
        self.model_params.setdefault('dropout', 0.1)
        
        # Crucial check for PiLearnerARM as per Section 3.2 of the paper
        use_learnable_pos_embeddings = self.model_params.get('use_learnable_pos_embeddings', False)
        if not use_learnable_pos_embeddings:
            logger.warning("PiLearnerARM is configured to not use learnable positional embeddings. "
                           "The paper (Section 3.2) explicitly states using learnable positional embeddings "
                           "for pi-learners to avoid RoPE's left-to-right bias. "
                           "Consider setting 'model.common.use_learnable_pos_embeddings' to True.")
        self.model_params['use_learnable_pos_embeddings'] = True # Force for PiLearnerARM as per paper

        # Extract resolved parameters for readability
        vocab_size_f = self.model_params['vocab_size']
        max_sequence_length_f = self.model_params['max_sequence_length']
        num_layers_f = self.model_params['num_layers']
        num_heads_f = self.model_params['num_heads']
        hidden_dim_f = self.model_params['hidden_dim']
        ff_dim_f = self.model_params['ff_dim']
        dropout_f = self.model_params['dropout']

        logger.info(f"Initializing PiLearnerARM with parameters: "
                    f"model_size={model_size_key}, vocab_size={vocab_size_f}, "
                    f"max_sequence_length={max_sequence_length_f}, num_layers={num_layers_f}, "
                    f"num_heads={num_heads_f}, hidden_dim={hidden_dim_f}, ff_dim={ff_dim_f}, "
                    f"dropout={dropout_f}, use_learnable_pos_embeddings=True (forced for PiLearnerARM)")

        self.token_embedding = nn.Embedding(vocab_size_f, hidden_dim_f)
        # Use learnable positional embeddings as required by the paper (Section 3.2)
        self.position_embedding = nn.Embedding(max_sequence_length_f, hidden_dim_f)
        self.drop = nn.Dropout(dropout_f)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim_f, num_heads_f, ff_dim_f, dropout_f)
            for _ in range(num_layers_f)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim_f)
        self.lm_head = nn.Linear(hidden_dim_f, vocab_size_f, bias=False)

        # Weight tying: share weights between token embedding and LM head, if bias is false.
        # This reduces parameters and often improves performance.
        self.token_embedding.weight = self.lm_head.weight
        logger.info("PiLearnerARM initialization complete with weight tying.")


    def forward(self, x_pi: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the PiLearnerARM.

        Args:
            x_pi (torch.Tensor): A batch of token IDs, already permuted according to
                                 the desired order. Shape: (batch_size, sequence_length).

        Returns:
            torch.Tensor: Logits for the vocabulary over all positions.
                          Shape: (batch_size, sequence_length, vocab_size).
                          `logits[b, t, v]` represents the model's unnormalized
                          prediction for the (t+1)-th token in sequence `b` being token `v`.
        """
        # Ensure input is on the correct device
        x_pi = x_pi.to(self.token_embedding.weight.device)

        batch_size, seq_len = x_pi.size()

        # Token embeddings
        token_embeddings = self.token_embedding(x_pi) # (B, T, H)

        # Positional embeddings (learnable)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=x_pi.device)
        position_embeddings = self.position_embedding(position_ids) # (T, H)

        # Combine embeddings
        x = token_embeddings + position_embeddings # (B, T, H)
        x = self.drop(x)

        # Transformer blocks (causal attention applied within each block)
        for block in self.transformer_blocks:
            x = block(x)

        # Final LayerNorm and projection to vocabulary logits
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, V)

        return logits

    def compute_likelihood(self, x0: torch.Tensor, permutation: List[int]) -> torch.Tensor:
        """
        Calculates the log-likelihood of an original sequence x0 under the model,
        given a specific permutation pi.

        Log p_theta(x_0) = sum_{j=0}^{L-1} log p_theta(x_0^pi(j) | x_0^pi(0), ..., x_0^pi(j-1)).

        Args:
            x0 (torch.Tensor): The original, unpermuted batch of sequences.
                               Shape: (batch_size, sequence_length).
            permutation (List[int]): A Python list of integers representing the
                                     permutation pi. Its length should be sequence_length.

        Returns:
            torch.Tensor: A tensor containing the log-likelihood for each sequence
                          in the batch, shape (batch_size,).
        """
        batch_size, sequence_length = x0.size()
        
        # Ensure permutation length matches sequence length
        if len(permutation) != sequence_length:
            raise ValueError(f"Permutation length ({len(permutation)}) must match "
                             f"sequence_length ({sequence_length}).")

        # Apply permutation to the input sequence
        # x0_permuted[b, j] will be x0[b, permutation[j]]
        # We need to unsqueeze permutation to act on the last dimension correctly if x0 is 2D.
        # However, it's simpler to index directly.
        x_permuted = x0[:, permutation].to(self.token_embedding.weight.device) # (B, T)

        # Obtain logits from the model's forward pass.
        # These logits are for predicting the *next* token in the permuted sequence.
        logits = self.forward(x_permuted) # (B, T, V)

        # For causal language modeling, logits[..., t, :] predicts x_permuted[..., t+1].
        # We need to shift targets and predictions to compute the loss.
        # Logits for tokens 0 to T-2 predict targets 1 to T-1.
        # So, we use logits up to T-1 and targets from 1.
        
        # Predictions are for x_permuted[:, 1:]
        shifted_logits = logits[:, :-1, :].contiguous() # (B, T-1, V)
        # Actual next tokens are x_permuted[:, 1:]
        shifted_target_tokens = x_permuted[:, 1:].contiguous() # (B, T-1)

        # Calculate log probabilities
        log_probs = F.log_softmax(shifted_logits, dim=-1) # (B, T-1, V)

        # Gather the log-probability of the actual next token at each position
        # unsqueeze(-1) is to make shifted_target_tokens broadcastable with log_probs for gather
        # squeeze(-1) removes the last dimension of size 1
        selected_log_probs = torch.gather(log_probs, -1, shifted_target_tokens.unsqueeze(-1)).squeeze(-1) # (B, T-1)

        # Sum the log probabilities across the sequence dimension for each sequence in the batch
        total_log_likelihood = selected_log_probs.sum(dim=1) # (B,)

        return total_log_likelihood


if __name__ == '__main__':
    # This block is for testing the PiLearnerARM module in isolation.

    # Mock Config for testing
    class MockConfig(Config):
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

    # Setup a mock logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MDM_Project_Logger")

    print("--- Testing PiLearnerARM ---")

    # Define a dummy config closely resembling config.yaml
    dummy_config_data = {
        'general': {
            'experiment_name': 'test_pi_learner',
            'seed': 42,
            'device': 'cpu'
        },
        'data': {
            'vocab_size': 100,
            'max_sequence_length': 64
        },
        'model': {
            'model_type': 'arm_transformer',
            'common': {
                'num_layers': 2,
                'num_heads': 4,
                'hidden_dim': 128,
                'ff_dim': 512,
                'dropout': 0.1,
                'use_learnable_pos_embeddings': True # Crucial for PiLearnerARM
            },
            'architecture_size': 'test_size', # Custom size for testing
            'size_configs': {
                'test_size': {
                    'num_layers': 2,
                    'num_heads': 4,
                    'hidden_dim': 128,
                    'ff_dim': 512
                },
                '42M': { # Example from paper
                    'num_layers': 8,
                    'num_heads': 8,
                    'hidden_dim': 768,
                    'ff_dim': 3072
                }
            }
        }
    }
    mock_config = MockConfig(dummy_config_data)

    # Instantiate the model
    try:
        model = PiLearnerARM(mock_config)
        print("PiLearnerARM instantiated successfully.")
        print(f"Model parameters: {model.model_params}")
        
        # Check if weight tying is applied
        assert model.token_embedding.weight is model.lm_head.weight
        print("Weight tying confirmed.")

        # Create dummy input data
        batch_size = 4
        sequence_length = 32
        dummy_input = torch.randint(0, mock_config.get('data.vocab_size'), (batch_size, sequence_length))
        
        # Test forward pass
        output_logits = model(dummy_input)
        print(f"Output logits shape: {output_logits.shape}")
        assert output_logits.shape == (batch_size, sequence_length, mock_config.get('data.vocab_size'))
        print("Forward pass successful.")

        # Test compute_likelihood
        print("\n--- Testing compute_likelihood ---")
        # Example permutation (identity permutation)
        identity_permutation = list(range(sequence_length))
        
        # Create a more structured dummy_input for easier likelihood verification
        # Let's say input is [1, 2, 3, 4, ..., T]
        x0_test = torch.tensor([[i for i in range(1, sequence_length + 1)]] * batch_size, dtype=torch.long)
        
        log_likelihoods = model.compute_likelihood(x0_test, identity_permutation)
        print(f"Computed log-likelihoods for identity permutation: {log_likelihoods}")
        assert log_likelihoods.shape == (batch_size,)
        print("compute_likelihood with identity permutation successful.")

        # Test with a different permutation (e.g., reverse order)
        reverse_permutation = list(range(sequence_length - 1, -1, -1))
        log_likelihoods_rev = model.compute_likelihood(x0_test, reverse_permutation)
        print(f"Computed log-likelihoods for reverse permutation: {log_likelihoods_rev}")
        print("compute_likelihood with reverse permutation successful.")

        # Test with a too short permutation
        try:
            model.compute_likelihood(x0_test, [0, 1])
        except ValueError as e:
            print(f"Caught expected error for incorrect permutation length: {e}")

    except Exception as e:
        logger.error(f"Error during PiLearnerARM testing: {e}", exc_info=True)

    print("\n--- PiLearnerARM testing complete ---")

