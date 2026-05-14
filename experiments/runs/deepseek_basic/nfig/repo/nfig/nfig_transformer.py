"""
NFIG Autoregressive Transformer for Next-Frequency Image Generation.

This module implements the decoder-only transformer with block-wise causal attention
that generates images progressively from low to high frequency components, as described 
in Section 3.2 of the paper.

Key features:
- Next-Frequency Prediction: generates T_1, then T_2, ..., T_n sequentially
- Block-wise causal attention (from VAR)
- AdaLN (Adaptive Layer Normalization) for conditional generation
- Classifier-Free Guidance (CFG)
- Top-k sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with class conditioning.
    
    Given class embedding c, produces scale and shift parameters:
    AdaLN(h, c) = gamma(c) * Norm(h) + beta(c)
    """
    
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma_proj = nn.Linear(cond_dim, dim)
        self.beta_proj = nn.Linear(cond_dim, dim)
        
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(cond)
        beta = self.beta_proj(cond)
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * self.norm(x) + beta


class AdaLNSelfAttention(nn.Module):
    """Self-attention with AdaLN conditioning."""
    
    def __init__(self, dim: int, num_heads: int, cond_dim: int, 
                 dropout: float = 0.0, use_adaln: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        if use_adaln:
            self.adaln = AdaLayerNorm(dim, cond_dim)
        else:
            self.adaln = None
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        
        if self.adaln is not None and cond is not None:
            x_norm = self.adaln(x, cond)
        else:
            x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            # attn_mask: (N, N) or (B*num_heads, N, N) or (1, 1, N, N)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """MLP feed-forward layer with GELU activation."""
    
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with AdaLN, self-attention, and feed-forward."""
    
    def __init__(self, dim: int, num_heads: int, cond_dim: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0, use_adaln: bool = True):
        super().__init__()
        self.attn = AdaLNSelfAttention(dim, num_heads, cond_dim, dropout, use_adaln)
        self.mlp = FeedForward(dim, mlp_ratio, dropout)
        if use_adaln:
            self.adaln_mlp = AdaLayerNorm(dim, cond_dim)
        else:
            self.adaln_mlp = None
        self.norm_mlp = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(x, cond, attn_mask)
        if self.adaln_mlp is not None and cond is not None:
            x = x + self.mlp(self.adaln_mlp(x, cond))
        else:
            x = x + self.mlp(self.norm_mlp(x))
        return x


class NFIGTransformer(nn.Module):
    """
    NFIG Autoregressive Transformer.
    
    Generates images by predicting tokens for each frequency band sequentially.
    Uses block-wise causal attention: tokens in band i attend to all tokens in 
    bands < i and causally within band i.
    
    Args:
        scales: list of (h_i, w_i) tuples for each frequency band
        codebook_size: size of the codebook (vocabulary size)
        dim: transformer hidden dimension
        depth: number of transformer blocks (paper uses 16)
        num_heads: number of attention heads
        num_classes: number of class conditions (1000 for ImageNet)
        cond_drop_prob: classifier-free guidance dropout probability
        mlp_ratio: MLP hidden dimension ratio
        dropout: dropout rate
    """
    
    def __init__(
        self,
        scales: List[Tuple[int, int]] = None,
        codebook_size: int = 4096,
        dim: int = 1024,
        depth: int = 16,
        num_heads: int = 16,
        num_classes: int = 1000,
        cond_drop_prob: float = 0.1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if scales is None:
            scales = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), 
                      (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)]
        
        self.scales = scales
        self.n_scales = len(scales)
        self.codebook_size = codebook_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.cond_drop_prob = cond_drop_prob
        
        self.total_tokens = sum(h * w for h, w in scales)
        
        # Token embedding
        self.token_embed = nn.Embedding(codebook_size, dim)
        
        # Position embedding per scale
        self.pos_embeds = nn.ParameterList()
        for h, w in scales:
            self.pos_embeds.append(nn.Parameter(torch.randn(1, h * w, dim) * 0.02))
        
        # Scale embedding
        self.scale_embed = nn.Embedding(self.n_scales, dim)
        
        # Start token
        self.start_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Class conditioning
        self.class_embed = nn.Embedding(num_classes + 1, dim)  # +1 for null class
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, dim, mlp_ratio, dropout, use_adaln=True)
            for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, codebook_size)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.scale_embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
    
    def create_block_causal_mask(self, total_seq_len: int, 
                                  prev_lens: List[int], 
                                  curr_len: int) -> torch.Tensor:
        """
        Create block-wise causal attention mask.
        
        Layout: [prev_tokens | curr_tokens]
        - prev_tokens attend to all prev positions (full bidirectional within prev)
        - curr_tokens attend to all prev + causally within current
        
        Returns mask of shape (total_seq_len, total_seq_len) where 0=attend, -inf=mask
        """
        prev_total = sum(prev_lens)
        total = prev_total + curr_len
        
        mask = torch.zeros(total, total)
        
        # Current tokens: attend to all prev + causal within current
        for i in range(curr_len):
            # Attend to all previous scale tokens
            for j in range(prev_total):
                mask[prev_total + i, j] = 0.0  # attend
            # Causal within current scale
            for j in range(i + 1):
                mask[prev_total + i, prev_total + j] = 0.0
            for j in range(i + 1, curr_len):
                mask[prev_total + i, prev_total + j] = float('-inf')
        
        # Previous tokens: full bidirectional within previous scales
        # This is fine as-is since they're all 0
        
        return mask
    
    def get_logits(self, token_seqs: List[torch.Tensor], 
                   class_ids: Optional[torch.Tensor] = None,
                   scale_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get logits for predicting the next scale's tokens.
        
        Args:
            token_seqs: list of (B, n_i) token sequences for scales 0..k-1
            class_ids: (B,) class indices
            scale_idx: which scale to predict
        
        Returns:
            logits: (B, curr_n, vocab_size) prediction logits for the new scale
        """
        B = token_seqs[0].shape[0]
        device = token_seqs[0].device
        
        # Build input sequence from previous scales
        embeddings = []
        prev_lens = []
        
        for i in range(len(token_seqs)):
            n_i = token_seqs[i].shape[1]
            tok_emb = self.token_embed(token_seqs[i])
            tok_emb = tok_emb + self.pos_embeds[i]
            scale_emb = self.scale_embed(torch.full((B, 1), i, device=device))
            tok_emb = tok_emb + scale_emb
            embeddings.append(tok_emb)
            prev_lens.append(n_i)
        
        prev_total = sum(prev_lens)
        curr_scale_idx = len(token_seqs)  # next scale to predict
        curr_h, curr_w = self.scales[curr_scale_idx]
        curr_n = curr_h * curr_w
        
        # Query tokens for the new scale (initially zeros)
        query_tokens = torch.zeros(B, curr_n, self.dim, device=device)
        # Add scale embedding
        scale_emb = self.scale_embed(torch.full((B, 1), curr_scale_idx, device=device))
        query_tokens = query_tokens + scale_emb
        # Add position embedding for this scale
        pos_emb = self.pos_embeds[curr_scale_idx]
        query_tokens = query_tokens + pos_emb
        
        # Concatenate: [prev_scales | query_tokens]
        x = torch.cat(embeddings + [query_tokens], dim=1)
        
        # Class conditioning
        if class_ids is not None:
            class_emb = self.class_embed(class_ids).unsqueeze(1)
        else:
            class_emb = torch.zeros(B, 1, self.dim, device=device)
        
        # Block-wise causal mask
        attn_mask = self.create_block_causal_mask(
            prev_total + curr_n, prev_lens, curr_n
        ).to(device)
        
        # Pass through transformer
        for block in self.blocks:
            x = block(x, class_emb, attn_mask)
        
        x = self.final_norm(x)
        
        # Get logits for the query positions only
        logits = self.head(x[:, -curr_n:])
        return logits
    
    def forward(self, token_seqs: List[torch.Tensor], 
                class_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass. Predicts each scale's tokens given previous scales.
        
        Args:
            token_seqs: list of (B, n_i) token sequences for all scales
            class_ids: (B,) class indices
        
        Returns:
            logits: (B, total_tokens, vocab_size)
            loss: cross-entropy loss
        """
        B = token_seqs[0].shape[0]
        device = token_seqs[0].device
        
        # CFG dropout
        if self.training and self.cond_drop_prob > 0:
            drop_mask = torch.rand(B, device=device) < self.cond_drop_prob
            class_ids = class_ids.clone()
            class_ids[drop_mask] = self.num_classes
        
        all_logits = []
        all_targets = []
        
        for i in range(self.n_scales):
            if i == 0:
                # First scale: predict from start token
                h, w = self.scales[i]
                n = h * w
                
                # Build input: start token
                start_emb = self.start_token.expand(B, 1, -1)
                scale_emb = self.scale_embed(torch.full((B, 1), i, device=device))
                start_emb = start_emb + scale_emb
                
                # Query tokens
                query_tokens = torch.zeros(B, n, self.dim, device=device)
                query_tokens = query_tokens + self.pos_embeds[i]
                query_tokens = query_tokens + self.scale_embed(
                    torch.full((B, 1), i, device=device)
                )
                
                x = torch.cat([start_emb, query_tokens], dim=1)
                
                # Causal mask: start attends to itself, each query attends to start + previous queries
                total_n = 1 + n
                attn_mask = torch.triu(
                    torch.ones(total_n, total_n, device=device) * float('-inf'), 
                    diagonal=1
                )
                
                if class_ids is not None:
                    class_emb = self.class_embed(class_ids).unsqueeze(1)
                else:
                    class_emb = torch.zeros(B, 1, self.dim, device=device)
                
                for block in self.blocks:
                    x = block(x, class_emb, attn_mask)
                
                x = self.final_norm(x)
                logits = self.head(x[:, 1:])  # exclude start token
            else:
                logits = self.get_logits(token_seqs[:i], class_ids)
            
            all_logits.append(logits)
            all_targets.append(token_seqs[i])
        
        total_logits = torch.cat(all_logits, dim=1)
        total_targets = torch.cat(all_targets, dim=1)
        
        loss = F.cross_entropy(
            total_logits.reshape(-1, self.codebook_size),
            total_targets.reshape(-1)
        )
        
        return total_logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        class_ids: torch.Tensor,
        top_k: int = 990,
        cfg_scale: float = 4.5,
        temperature: float = 1.0,
        return_progressive: bool = False,
    ) -> List[torch.Tensor]:
        """
        Generate image tokens autoregressively by next-frequency prediction.
        """
        B = class_ids.shape[0]
        device = class_ids.device
        
        token_seqs = []
        
        for i in range(self.n_scales):
            h, w = self.scales[i]
            n = h * w
            
            if i == 0:
                # First scale: token-by-token with CFG
                start_emb = self.start_token.expand(B, 1, -1)
                scale_emb_start = self.scale_embed(torch.full((B, 1), i, device=device))
                start_emb = start_emb + scale_emb_start
                
                generated = []
                for pos in range(n):
                    if pos == 0:
                        x = start_emb
                    else:
                        tok_emb = self.token_embed(torch.stack(generated, dim=1))
                        pos_emb_part = self.pos_embeds[i][:, :pos]
                        scale_emb_curr = self.scale_embed(
                            torch.full((B, 1), i, device=device)
                        )
                        x = torch.cat([start_emb, tok_emb + pos_emb_part + scale_emb_curr], dim=1)
                    
                    curr_len = x.shape[1]
                    attn_mask = torch.triu(
                        torch.ones(curr_len, curr_len, device=device) * float('-inf'),
                        diagonal=1
                    )
                    
                    if cfg_scale > 0 and class_ids is not None:
                        class_emb = self.class_embed(class_ids).unsqueeze(1)
                        x_cond = x.clone()
                        for block in self.blocks:
                            x_cond = block(x_cond, class_emb, attn_mask)
                        logits_cond = self.head(self.final_norm(x_cond))[:, -1:]
                        
                        null_emb = torch.zeros(B, 1, self.dim, device=device)
                        x_uncond = x.clone()
                        for block in self.blocks:
                            x_uncond = block(x_uncond, null_emb, attn_mask)
                        logits_uncond = self.head(self.final_norm(x_uncond))[:, -1:]
                        
                        logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                    else:
                        class_emb = self.class_embed(class_ids).unsqueeze(1)
                        for block in self.blocks:
                            x = block(x, class_emb, attn_mask)
                        logits = self.head(self.final_norm(x))[:, -1:]
                    
                    logits = logits / temperature
                    if top_k > 0:
                        top_k_vals, _ = torch.topk(logits, min(top_k, self.codebook_size), dim=-1)
                        min_top_k = top_k_vals[:, :, -1:]
                        logits[logits < min_top_k] = float('-inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    token = torch.multinomial(probs.view(B, -1), 1)
                    generated.append(token)
                
                tokens_i = torch.stack(generated, dim=1).view(B, -1)
            else:
                # Predict using get_logits
                logits = self.get_logits(token_seqs, class_ids)
                
                if cfg_scale > 0 and class_ids is not None:
                    null_ids = torch.full_like(class_ids, self.num_classes)
                    logits_uncond = self.get_logits(token_seqs, null_ids)
                    logits = logits_uncond + cfg_scale * (logits - logits_uncond)
                
                logits = logits / temperature
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, self.codebook_size), dim=-1)
                    min_top_k = top_k_vals[:, :, -1:]
                    logits[logits < min_top_k] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                tokens_i = torch.multinomial(
                    probs.view(B * n, -1), 1
                ).view(B, n)
            
            token_seqs.append(tokens_i)
        
        return token_seqs
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
