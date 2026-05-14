
import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    """
    Learned linear layer to determine expert assignments.
    """
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        return self.gate(x)

class Expert(nn.Module):
    """
    A single FFN expert.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU() # Assuming ReLU as a common activation

    def forward(self, x):
        return self.w2(self.relu(self.w1(x)))

class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) layer.
    Replaces the FFN in a Transformer block.
    """\
    def __init__(self, d_model, num_experts, num_experts_per_token, d_ff):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.router = Router(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Flatten x for router input
        x_flat = x.view(-1, d_model) # (num_tokens, d_model), num_tokens = batch_size * sequence_length

        router_logits = self.router(x_flat) # (num_tokens, num_experts)
        routing_weights_all_experts = F.softmax(router_logits, dim=-1)

        # Select top-k experts and their weights
        # routing_weights: (num_tokens, num_experts_per_token)
        # selected_experts: (num_tokens, num_experts_per_token)
        routing_weights, selected_experts = torch.topk(routing_weights_all_experts, self.num_experts_per_token, dim=-1)

        # Normalize routing weights to sum to 1 for the selected experts
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        final_output = torch.zeros_like(x_flat) # (num_tokens, d_model)

        # In a real implementation, this loop would be replaced by more efficient
        # scatter/gather operations or custom CUDA kernels for performance.
        # This conceptual loop iterates through each token and dispatches it.
        for i in range(x_flat.shape[0]): # Iterate through num_tokens
            token_input = x_flat[i] # (d_model)
            
            for k in range(self.num_experts_per_token):
                expert_idx = selected_experts[i, k].item()
                weight = routing_weights[i, k]
                final_output[i] += weight * self.experts[expert_idx](token_input)

        return final_output.view(batch_size, seq_len, d_model), router_logits, routing_weights_all_experts, selected_experts


class OlmoeBlock(nn.Module):
    """
    A single OLMoE Transformer block.
    Assumes a standard Transformer decoder block structure with MoE instead of FFN.
    """\
    def __init__(self, d_model, num_heads, d_ff, num_experts, num_experts_per_token, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.moe = MoE(d_model, num_experts, num_experts_per_token, d_ff)

        # Store MoE outputs for loss calculation
        self.router_logits = None
        self.routing_weights_all_experts = None
        self.selected_experts = None

    def forward(self, x, mask=None):
        # Self-attention
        # MultiheadAttention expects query, key, value to be (sequence_length, batch_size, d_model)
        # and attn_mask to be (sequence_length, sequence_length)
        x_permuted = x.permute(1, 0, 2) # (seq_len, batch_size, d_model)

        attn_output, _ = self.attn(x_permuted, x_permuted, x_permuted, attn_mask=mask)
        attn_output = attn_output.permute(1, 0, 2) # (batch_size, seq_len, d_model) back

        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # MoE layer
        moe_output, router_logits, routing_weights_all_experts, selected_experts = self.moe(x)
        self.router_logits = router_logits
        self.routing_weights_all_experts = routing_weights_all_experts
        self.selected_experts = selected_experts

        x = x + self.dropout2(moe_output)
        x = self.norm2(x)
        return x


class OlmoeModel(nn.Module):
    """
    The main OLMoE model.
    """\
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, num_experts, num_experts_per_token):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            OlmoeBlock(d_model, num_heads, d_ff, num_experts, num_experts_per_token)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.token_embedding(input_ids)
        
        x = embeddings
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
        
        logits = self.lm_head(x)
        return logits


