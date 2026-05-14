
import torch
import torch.nn as nn
import torch.nn.functional as F
from ngpt_model.layers import Norm, RMSNorm
from ngpt_model.modules import MultiHeadSelfAttention, MLP
import math

class TransformerBlock(nn.Module):
    """
    A single Transformer block, composed of a Multi-Head Self-Attention and an MLP.
    Can be configured for both baseline Transformer and Normalized Transformer (nGPT).
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_mlp: int, dropout: float = 0.1, is_ngpt: bool = False, rope_base: int = 10000):
        super().__init__()
        self.is_ngpt = is_ngpt

        if not self.is_ngpt:
            self.attn_norm = RMSNorm(d_model)
            self.mlp_norm = RMSNorm(d_model)

        self.attn = MultiHeadSelfAttention(d_model, n_heads, d_k, dropout, is_ngpt, rope_base)
        self.mlp = MLP(d_model, d_mlp, dropout, is_ngpt)

        if self.is_ngpt:
            # Learnable eigen learning rates for nGPT
            self.alpha_A = nn.Parameter(torch.full((d_model,), 0.05)) # Initialized to 0.05
            self.alpha_M = nn.Parameter(torch.full((d_model,), 0.05)) # Initialized to 0.05
            # From paper, alpha_A,M_scale = 1/sqrt(d_model) for scaling effective LR, but for actual forward pass they are used directly.
            # Here we follow the simplified approach mentioned in A.7 that init is the direct value.

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.is_ngpt:
            # nGPT: hA = Norm(ATTN(h))
            h_attn_out = Norm(self.attn(h, mask))
            # nGPT: h = Norm(h + alpha_A * (hA - h))
            h = Norm(h + self.alpha_A * (h_attn_out - h))

            # nGPT: hM = Norm(MLP(h))
            h_mlp_out = Norm(self.mlp(h))
            # nGPT: h = Norm(h + alpha_M * (hM - h))
            h = Norm(h + self.alpha_M * (h_mlp_out - h))
        else:
            # Baseline Transformer
            h_attn_res = self.attn(self.attn_norm(h), mask)
            h = h + h_attn_res
            h_mlp_res = self.mlp(self.mlp_norm(h))
            h = h + h_mlp_res
        return h

class GPT(nn.Module):
    """
    Base Transformer (GPT) or Normalized Transformer (nGPT) model.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_mlp: int,
                 dropout: float = 0.1, is_ngpt: bool = False, rope_base: int = 10000,
                 d_k: int = None):
        super().__init__()
        self.is_ngpt = is_ngpt
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.output_embeddings = nn.Embedding(vocab_size, d_model) # Potentially tied, but paper implies separate
                                                                # unless explicitly tied.

        if d_k is None:
            d_k = d_model // n_heads

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_k, d_mlp, dropout, is_ngpt, rope_base)
            for _ in range(n_layers)
        ])

        if not self.is_ngpt:
            self.final_norm = RMSNorm(d_model)
        else:
            # Trainable scaling parameter for logits
            self.s_z = nn.Parameter(torch.ones(vocab_size))
            self.s_z.data.fill_(1.0) # Initialized to 1, scaled by 1/sqrt(d_model) effectively
            self.s_z_scale = 1.0 / math.sqrt(d_model) # From paper, s_z_scale = 1/sqrt(d_model)

        self.apply(self._init_weights)

        # Initialize weights for nGPT
        if self.is_ngpt:
            self.token_embeddings.weight.data = Norm(self.token_embeddings.weight.data)
            self.output_embeddings.weight.data = Norm(self.output_embeddings.weight.data)

        print(f"Number of parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.is_ngpt:
                # nGPT initialization: 1 / sqrt(d_model)
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
                # Output matrices W_o, W_o_mlp std scaled by sqrt(2 * n_layers)
                if isinstance(module, (MultiHeadSelfAttention, MLP)) and module.Wo.weight is module.weight: # This check needs refinement
                     nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model) * math.sqrt(2 * len(self.transformer_blocks)))

            else:
                # Baseline GPT initialization: 0.02
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # Output matrices std scaled by sqrt(2 * n_layers)
                if isinstance(module, (MultiHeadSelfAttention, MLP)) and module.Wo.weight is module.weight: # This check needs refinement
                    nn.init.normal_(module.weight, mean=0.0, std=0.02 * math.sqrt(2 * len(self.transformer_blocks)))

            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if self.is_ngpt:
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def normalize_ngpt_parameters(self):
        """
        Normalize all relevant matrices and embeddings in nGPT after each training step.
        This is called during the training loop.
        """
        with torch.no_grad():
            self.token_embeddings.weight.data = Norm(self.token_embeddings.weight.data)
            self.output_embeddings.weight.data = Norm(self.output_embeddings.weight.data)
            for block in self.transformer_blocks:
                block.attn.normalize_weights()
                block.mlp.normalize_weights()


    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        batch_size, seq_len = input_ids.size()

        # Causal mask for self-attention
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(1, 1, seq_len, seq_len)

        # Input embeddings
        h = self.token_embeddings(input_ids)
        if self.is_ngpt:
            h = Norm(h) # Input embeddings are normalized

        # Transformer blocks
        for block in self.transformer_blocks:
            h = block(h, mask)

        if not self.is_ngpt:
            h = self.final_norm(h)

        # Output logits
        # z_i = E_output @ h_i
        # Equivalent to matrix multiplication E_output.weight @ h_i.T
        logits = F.linear(h, self.output_embeddings.weight)

        if self.is_ngpt:
            # Scale logits for nGPT
            # s_z is per vocabulary token, so it scales each logit for each token in the sequence
            logits = logits * (self.s_z * self.s_z_scale)

        loss = None
        if targets is not None:
            # Reshape for cross_entropy: (batch_size * seq_len, vocab_size)
            logits_reshaped = logits.view(-1, self.vocab_size)
            targets_reshaped = targets.view(-1)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped, ignore_index=-1)

        return logits, loss

