
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from layers import RMSNorm, MultiHeadAttention, FeedForward
from modules import MoELayer
from config import ModelConfig

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.dim = config.dimension
        self.num_attention_heads = config.num_attention_heads
        self.ffn_dimension = config.ffn_dimension
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.use_qk_norm = config.use_qk_norm
        self.rope_theta = config.rope_theta
        self.use_bias = config.use_bias # Bias is generally false for these models

        self.attn_norm = RMSNorm(self.dim, eps=self.layer_norm_epsilon)
        self.attn = MultiHeadAttention(
            dim=self.dim,
            num_heads=self.num_attention_heads,
            dropout_rate=0.1, # Dropout rate not explicitly mentioned for individual layers, using a common value
            use_qk_norm=self.use_qk_norm,
            layer_norm_epsilon=self.layer_norm_epsilon,
            rope_theta=self.rope_theta,
            use_bias=self.use_bias
        )

        self.ffn_norm = RMSNorm(self.dim, eps=self.layer_norm_epsilon)
        # Check if this layer should be an MoE layer based on layer_idx
        # The paper states "Every layer is an MoE layer" for OLMOE-1B-7B (moe_layers_interval = 1)
        self.is_moe_layer = (config.moe_layers_interval > 0 and 
                             (layer_idx % config.moe_layers_interval == 0))

        if self.is_moe_layer:
            self.moe = MoELayer(
                dim=self.dim,
                ffn_dimension=self.ffn_dimension,
                num_experts=config.num_experts,
                num_activated_experts=config.num_activated_experts,
                dropout_rate=0.1, # Using a placeholder dropout
                use_bias=self.use_bias
            )
        else:
            self.ffn = FeedForward(
                dim=self.dim,
                ffn_hidden_dim=self.ffn_dimension,
                dropout_rate=0.1, # Using a placeholder dropout
                use_bias=self.use_bias
            )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Attention block
        norm_x = self.attn_norm(x)
        attn_output = self.attn(norm_x, mask=attention_mask)
        x = x + attn_output # Residual connection

        # FFN or MoE block
        norm_x = self.ffn_norm(x)
        if self.is_moe_layer:
            moe_output, router_logits, expert_gate_probabilities = self.moe(norm_x)
            x = x + moe_output # Residual connection
            return x, router_logits, expert_gate_probabilities
        else:
            ffn_output = self.ffn(norm_x)
            x = x + ffn_output # Residual connection
            return x, None, None # No MoE specific outputs

class OLMoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.dimension)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx) for layer_idx in range(config.num_layers)
        ])
        
        self.final_norm = RMSNorm(config.dimension, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.dimension, config.vocab_size, bias=self.config.use_bias)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.config.init_method == "truncated_normal":
                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.init_std,
                    a=-self.config.init_trunc_cutoff,
                    b=self.config.init_trunc_cutoff
                )
            elif self.config.init_method == "normal":
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            else:
                raise ValueError(f"Unknown initialization method: {self.config.init_method}")
            
            if module.bias is not None and self.config.use_bias:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if self.config.init_method == "truncated_normal":
                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.init_std,
                    a=-self.config.init_trunc_cutoff,
                    b=self.config.init_trunc_cutoff
                )
            elif self.config.init_method == "normal":
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            else:
                raise ValueError(f"Unknown initialization method: {self.config.init_method}")
            

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.token_embeddings(input_ids)

        all_router_logits = []
        all_expert_gate_probabilities = []

        for layer in self.layers:
            x, router_logits, expert_gate_probabilities = layer(x, attention_mask)
            if router_logits is not None:
                all_router_logits.append(router_logits)
                all_expert_gate_probabilities.append(expert_gate_probabilities)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, all_router_logits, all_expert_gate_probabilities
