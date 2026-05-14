
import torch
import torch.nn as nn
from typing import Optional

from modules import GatedMultiHeadAttention, FeedForward
from layers import RMSNorm
from config import ModelConfig

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = GatedMultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Pre-normalization (common in modern LLMs)
        h = self.attention_norm(x)
        h = self.attention(h, attention_mask)
        x = x + h # Residual connection

        h = self.ffn_norm(x)
        if isinstance(self.feed_forward, MoEFeedForward):
            h, router_z_loss = self.feed_forward(h)
            # Z-loss is returned by MoEFeedForward. This needs to be accumulated in the training loop.
            # For now, let's just return it from the block.
            x = x + h # Residual connection
            return x, router_z_loss
        else:
            h = self.feed_forward(h)
            x = x + h # Residual connection
            return x

class GatedTransformer(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.tok_embeddings = nn.Embedding(vocab_size, config.d_model)
        # Rotary Position Embeddings (RoPE) are mentioned in the paper for context length extension,
        # but a full implementation is complex and beyond the scope of this reproduction without more details.
        # For now, we'll omit explicit RoPE layer, assuming it's handled implicitly or by external utilities
        # if a pre-trained model is loaded. For this scratch implementation, it's a simplification.

        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(TransformerBlock(config))

        self.norm = RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size, bias=False)

        # Weight tying (if applicable, not explicitly mentioned but common for embeddings/output)
        self.tok_embeddings.weight = self.output.weight

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resizes the token embeddings to new_num_tokens."""
        old_embeddings = self.tok_embeddings
        new_embeddings = nn.Embedding(new_num_tokens, self.config.d_model)
        new_embeddings.to(old_embeddings.weight.device)
        self.tok_embeddings = new_embeddings
        self.output = nn.Linear(self.config.d_model, new_num_tokens, bias=False)
        self.tok_embeddings.weight = self.output.weight # Re-tie weights
        with torch.no_grad():
            new_embeddings.weight[:old_embeddings.weight.shape[0]] = old_embeddings.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        h = self.tok_embeddings(input_ids)

        # Causal mask for decoder-only transformers
        if attention_mask is None:
            seq_len = input_ids.shape[1]
            attention_mask = torch.full(
                (1, 1, seq_len, seq_len), float("-inf"), device=input_ids.device
            )
            attention_mask = torch.triu(attention_mask, diagonal=1).type_as(h)

        total_router_z_loss = torch.tensor(0.0, device=h.device)

        for layer in self.layers:
            # TransformerBlock now returns a tuple (output, z_loss) if MoE, else just output
            layer_output = layer(h, attention_mask)
            if isinstance(layer_output, tuple):
                h, router_z_loss = layer_output
                total_router_z_loss += router_z_loss
            else:
                h = layer_output

        h = self.norm(h)
        output = self.output(h).float()

        if self.config.model_type == "moe":
            return output, total_router_z_loss
        return output

class MoEFeedForward(nn.Module):
    """
    Mixture of Experts (MoE) Feed-Forward network.
    Paper mentions "128 total experts with top-8 softmax gating".
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.d_model = config.d_model
        self.d_ff = config.d_ff

        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor):
        # Router
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, expert_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True) # Normalize top-k weights

        final_output = torch.zeros_like(x)

        # Z-loss: squared sum of router logits
        router_z_loss = self.config.moe_loss_coeff * torch.sum(router_logits ** 2)

        # For each token in the batch, route to top-k experts
        # This naive implementation is very slow. For performance, typically
        # a more optimized approach like a 'scatter' operation or grouping
        # tokens by expert is used (e.g., in FairSeq's MoE implementation).
        # Given this is a reproduction of core logic, this suffices for clarity.
        for i, expert in enumerate(self.experts):
            # Find tokens assigned to this expert
            # expert_indices is (B, S, top_k)
            # We need to find (batch_idx, seq_idx, k_idx) for tokens assigned to expert 'i'
            # torch.where returns (row_indices, col_indices) for a 2D tensor.
            # For 3D (batch, seq, k), we need to handle it properly.
            # Here, we can iterate over the batch and sequence dimensions.

            # This is a simplified, non-optimized loop.
            # A more efficient implementation would use a combination of indexing and scatter_add.
            # For correctness of logic, this iteration works.

            # Identify indices where current expert 'i' is among the top-k for any token
            # mask (B, S, top_k)
            mask_expert_i = (expert_indices == i)

            # Get the (batch_idx, seq_idx) for tokens that use expert 'i'
            batch_indices, seq_indices, k_indices = torch.where(mask_expert_i)

            if batch_indices.numel() > 0: # If any tokens use this expert
                # Gather the input tokens for this expert
                current_tokens_input = x[batch_indices, seq_indices]
                # Get the output from this expert
                expert_output = expert(current_tokens_input)

                # Get the corresponding routing weights
                current_routing_weights = routing_weights[batch_indices, seq_indices, k_indices].unsqueeze(-1)

                # Add the weighted expert output to the final_output
                # Using index_put_ avoids sparse tensor issues and is more straightforward than scatter_add for this structure
                # However, it involves gathering and scattering which can be inefficient.
                # For faithful reproduction of logic, this is fine.
                final_output[batch_indices, seq_indices] += expert_output * current_routing_weights


        return final_output, router_z_loss


class GatedMoETransformer(GatedTransformer):
    """
    Transformer model with Mixture of Experts (MoE) Feed-Forward networks.
    """
    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__(config, vocab_size)
        self.model_type = "moe" # Override model type

        # Replace FFN in TransformerBlocks with MoEFeedForward
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            block = TransformerBlock(config)
            # Ensure MoEFeedForward is instantiated with the same config
            # so it can use the potentially reduced FFN width.
            block.feed_forward = MoEFeedForward(config)
            self.layers.append(block)

# Function to get the appropriate model based on config
def get_model(config: ModelConfig, vocab_size: int) -> nn.Module:
    if config.model_type == "dense":
        return GatedTransformer(config, vocab_size)
    elif config.model_type == "moe":
        return GatedMoETransformer(config, vocab_size)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

