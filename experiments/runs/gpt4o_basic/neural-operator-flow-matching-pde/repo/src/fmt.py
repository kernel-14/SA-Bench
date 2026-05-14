import torch
import torch.nn as nn
from flash_attn import FlashAttention

class FlowMarchingTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, rnn_dim):
        super(FlowMarchingTransformer, self).__init__()

        # Transformer layers
        self.attention = FlashAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.gru = nn.GRU(rnn_dim, rnn_dim, batch_first=True)

        # Projection layers
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, input_dim)

    def forward(self, x, h):
        x = self.input_proj(x)
        attn_out = self.attention(x)
        attn_out = self.norm(attn_out)
        ff_out = self.feed_forward(attn_out)
        gru_out, h_new = self.gru(ff_out, h)
        out = self.output_proj(gru_out)
        return out, h_new
