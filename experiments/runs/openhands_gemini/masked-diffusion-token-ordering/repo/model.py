
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from modules import TransformerEncoderBlock
from layers import LearnablePositionalEmbedding
from config import Config

class MDMConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=Config.vocab_size,
        hidden_size=Config.hidden_size,
        num_attention_heads=Config.num_attention_heads,
        num_layers=Config.num_layers,
        intermediate_size=Config.intermediate_size,
        hidden_act=Config.hidden_act,
        hidden_dropout_prob=Config.hidden_dropout_prob,
        attention_probs_dropout_prob=Config.attention_probs_dropout_prob,
        max_sequence_length=Config.max_sequence_length,
        initializer_range=Config.initializer_range,
        layer_norm_eps=Config.layer_norm_eps,
        use_learnable_pos_embeddings=Config.use_learnable_pos_embeddings,
        pad_token_id=Config.mask_token_id, # using mask_token_id as pad_token_id for transformers compatibility
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_learnable_pos_embeddings = use_learnable_pos_embeddings

class MaskedDiffusionModel(PreTrainedModel):
    """
    Masked Diffusion Model (MDM) architecture as the denoising network p_theta(x_0^i | x_t).
    It predicts the original token x_0^i for each masked position.
    The model is time-embedding free, inferring 't' from the masked input x_t.
    """
    config_class = MDMConfig

    def __init__(self, config: MDMConfig):
        super().__init__(config)
        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        if config.use_learnable_pos_embeddings:
            self.position_embeddings = LearnablePositionalEmbedding(config.max_sequence_length, config.hidden_size)
        else:
            # If not using learnable, we might default to no positional embeddings or sinusoidal ones (not specified, but for completeness)
            self.position_embeddings = None 

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.hidden_act,
                config.attention_probs_dropout_prob,
                config.hidden_dropout_prob
            )
            for _ in range(config.num_layers)
        ])

        self.cls_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            input_ids: Tokenized input sequences, where masked tokens are represented by mask_token_id.
                       Shape: (batch_size, sequence_length)
            attention_mask: Mask to avoid performing attention on padding token indices.
                            Shape: (batch_size, sequence_length).
                            Values: 1 for real tokens, 0 for padding.
        Returns:
            logits: Prediction logits for each token in the vocabulary for each position.
                    Shape: (batch_size, sequence_length, vocab_size)
        """
        input_shape = input_ids.size()
        seq_len = input_shape[1]
        
        # Word embeddings
        word_embeddings = self.word_embeddings(input_ids)
        
        # Positional embeddings
        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(input_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # Prepare attention mask for transformer
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)
        
        # Extended attention mask (batch_size, 1, 1, sequence_length)
        # This is for broadcasting with attention scores (batch_size, num_heads, sequence_length, sequence_length)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # Convert to additive mask

        hidden_states = embeddings
        for i, layer_module in enumerate(self.encoder_layers):
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        
        logits = self.cls_head(hidden_states)
        
        return logits

