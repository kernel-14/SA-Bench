import torch
import torch.nn as nn
from embeddings import Embeddings
from ngpt_block import NGPTBlock
from normalization import normalize

class NGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_mlp: int,
                 s_z_init: float = 1.0, s_z_scale: float = None,
                 s_qk_init: float = 1.0, s_qk_scale: float = None,
                 s_u_init: float = 1.0, s_u_scale: float = 1.0,
                 s_nu_init: float = 1.0, s_nu_scale: float = 1.0,
                 alpha_A_init: float = 0.05, alpha_A_scale: float = None,
                 alpha_M_init: float = 0.05, alpha_M_scale: float = None):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.embeddings = Embeddings(vocab_size, d_model, s_z_init, s_z_scale)
        
        self.transformer_blocks = nn.ModuleList([
            NGPTBlock(d_model, n_heads, d_mlp,
                      alpha_A_init, alpha_A_scale,
                      alpha_M_init, alpha_M_scale)
            for _ in range(n_layers)
        ])

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Get input embeddings
        h = self.embeddings.forward_input(tokens)
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            h = block(h, mask)
            # No additional normalization after the final layer (Section 2.2.2)

        # Get logits for next token prediction
        logits = self.embeddings.get_logits(h)
        return logits

