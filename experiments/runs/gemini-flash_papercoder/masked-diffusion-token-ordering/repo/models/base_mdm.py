import abc
import torch
from torch import nn
from typing import Any, Dict, Optional

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
        # This method should ideally not be called on the placeholder itself,
        # but is needed for subclasses. If it's called on the placeholder, it
        # indicates an issue in the import chain or usage pattern.
        raise NotImplementedError("This is a placeholder for the Config object. "
                                  "Its 'get' method should not be called directly from here. "
                                  "Ensure the actual Config object is passed and used.")

# Re-assign for type hinting within this module.
# In the actual project, this would be: `from config import Config`
Config = _ConfigPlaceholder


class BaseMDMModel(nn.Module, abc.ABC):
    """
    Abstract base class for all Masked Diffusion Model (MDM) implementations.
    This class provides a common interface and handles the parsing of shared
    model configuration parameters from the `Config` object.

    Attributes:
        config (Config): The global configuration object providing access to all
                         experiment settings and parameters.
        model_params (Dict[str, Any]): A dictionary containing resolved architectural
                                       parameters for the specific model instance,
                                       merged from common and size-specific configurations.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the BaseMDMModel with the provided configuration.
        It sets up common model parameters based on the `config.yaml` file,
        merging general settings with specific settings for a chosen model size.

        Args:
            config (Config): The global configuration object containing all
                             experiment settings, model parameters, etc.

        Raises:
            ValueError: If essential parameters like 'vocab_size' or
                        'max_sequence_length' are not found in the configuration.
        """
        super().__init__()
        self.config: Config = config
        self.model_params: Dict[str, Any] = {}

        # --- 1. Load common model parameters ---
        common_model_config = self.config.get('model.common', {})
        self.model_params.update(common_model_config)

        # --- 2. Determine and load size-specific parameters ---
        # The 'model.architecture_size' key specifies which size config (e.g., '6M', '19M') to use.
        # Defaulting to '170M' if not explicitly set, as it's a common example in the paper.
        model_size_key: str = self.config.get('model.architecture_size', '170M')
        
        # Override common parameters with size-specific parameters if they exist
        size_specific_config = self.config.get(f'model.size_configs.{model_size_key}', {})
        if size_specific_config:
            self.model_params.update(size_specific_config)
        
        # --- 3. Add essential data-related parameters required by the model ---
        # These are crucial for defining embedding layers and sequence processing.
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
        # These are defaults that a Transformer-based model typically needs.
        self.model_params.setdefault('num_layers', 12)  # Number of Transformer encoder layers
        self.model_params.setdefault('num_heads', 12)   # Number of attention heads
        self.model_params.setdefault('hidden_dim', 768) # Dimension of the hidden states
        self.model_params.setdefault('ff_dim', 3072)   # Dimension of the feed-forward network in Transformer (usually 4*hidden_dim)
        self.model_params.setdefault('dropout', 0.1)    # Dropout probability
        self.model_params.setdefault('use_learnable_pos_embeddings', False) # Whether to use learnable positional embeddings


    @abc.abstractmethod
    def forward(self, x_t: torch.Tensor, masked_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Abstract method for the forward pass of the MDM.

        This method must be implemented by all concrete subclasses to define
        how the model processes a partially masked sequence `x_t` to predict
        the original tokens.

        Args:
            x_t (torch.Tensor): The input sequence at a particular diffusion step `t`.
                                Shape: (batch_size, sequence_length). Contains token IDs,
                                potentially including a mask token ID.
            masked_positions (Optional[torch.Tensor]): A boolean tensor (or indices)
                                representing the positions that were masked in `x_t`
                                compared to the original `x_0`. This can be used
                                by specific model architectures if they need to
                                condition their output or computation on masked locations.
                                For standard MDMs, the model typically predicts for all positions
                                and the loss is applied only to the masked ones.
                                Shape: (batch_size, sequence_length) with boolean values,
                                or (batch_size, num_masked_tokens) with indices.

        Returns:
            torch.Tensor: Logits for the vocabulary over all positions.
                          Shape: (batch_size, sequence_length, vocab_size).
                          Each entry `[b, l, v]` is the unnormalized log-probability
                          of token `v` being the original token at position `l`
                          for batch item `b`.

        Raises:
            NotImplementedError: This method is abstract and must be implemented
                                 by concrete subclasses.
        """
        raise NotImplementedError("Forward method must be implemented by subclasses.")

