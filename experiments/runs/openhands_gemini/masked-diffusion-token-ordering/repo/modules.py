
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """
    Multi-head self-attention module.
    """
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_probs_dropout_prob: float):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Take dot product between "query" and "key" to get raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TransformerModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is ostensibly actually just a dropout on the attention probabilities.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class SelfOutput(nn.Module):
    """
    Output layer for self-attention.
    """
    def __init__(self, hidden_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Attention(nn.Module):
    """
    Combines SelfAttention and SelfOutput.
    """
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_probs_dropout_prob: float, hidden_dropout_prob: float):
        super().__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class Intermediate(nn.Module):
    """
    Intermediate feed-forward layer.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = F.gelu if hidden_act == "gelu" else F.relu # using gelu as default as per config

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class Output(nn.Module):
    """
    Output layer for feed-forward network.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class TransformerEncoderBlock(nn.Module):
    """
    A single transformer encoder block.
    """
    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int,
                 hidden_act: str, attention_probs_dropout_prob: float, hidden_dropout_prob: float):
        super().__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size, hidden_act)
        self.output = Output(hidden_size, intermediate_size, hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
