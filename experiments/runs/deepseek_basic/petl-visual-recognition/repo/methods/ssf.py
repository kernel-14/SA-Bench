"""SSF (Scale & Shift deep Features) for PEFT.

SSF applies linear transformations to adapt intermediate features.
For each feature location, SSF applies:
    h = w ⊙ h + b
    
where w, b are learnable scale and shift parameters.

Modified features: h2, h3, h5, h7, h8, h9 per layer.
"""

import torch
import torch.nn as nn


class SSFModule(nn.Module):
    """Single SSF module: scales and shifts features element-wise."""
    
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        return self.scale * x + self.shift


class SSF(nn.Module):
    """SSF: Scale & Shift deep Features for parameter-efficient transfer learning.
    
    Applies scale and shift to intermediate features at specific locations:
    h2 = SSF2(h2), h3 = SSF3(h3), h5 = SSF5(h5), h7 = SSF7(h7), 
    h8 = SSF8(h8), h9 = SSF9(h9)
    
    Parameters: 0.205M (fixed for ViT-B/16)
    """
    
    def __init__(self, embed_dim=768, num_layers=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Per-layer SSF modules for each feature location
        # h2: after first LN, before MSA - [D]
        self.ssf_h2 = nn.ModuleList([SSFModule(embed_dim) for _ in range(num_layers)])
        # h3: Q/K/V in MSA - [3D] (Q, K, V concatenated)
        self.ssf_h3 = nn.ModuleList([SSFModule(3 * embed_dim) for _ in range(num_layers)])
        # h5: after MSA + residual - [D]
        self.ssf_h5 = nn.ModuleList([SSFModule(embed_dim) for _ in range(num_layers)])
        # h7: after MLP FC1 - [4D] (MLP hidden dim = 4*D)
        self.ssf_h7 = nn.ModuleList([SSFModule(4 * embed_dim) for _ in range(num_layers)])
        # h8: after MLP activation - [4D]
        self.ssf_h8 = nn.ModuleList([SSFModule(4 * embed_dim) for _ in range(num_layers)])
        # h9: after MLP FC2, before residual - [D]
        self.ssf_h9 = nn.ModuleList([SSFModule(embed_dim) for _ in range(num_layers)])
    
    def forward(self, x, vit_model):
        """Forward with SSF feature modulation."""
        B = x.shape[0]
        
        x = vit_model.patch_embed(x)
        cls_token = vit_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + vit_model.pos_embed
        
        for i, block in enumerate(vit_model.blocks):
            # h2: after first LN
            h2 = block.norm1(x)
            h2 = self.ssf_h2[i](h2)  # SSF on h2
            
            # MSA forward
            if hasattr(block.attn, 'qkv'):
                # Extract QKV
                qkv = block.attn.qkv(h2)
                # h3: SSF on QKV
                qkv = self.ssf_h3[i](qkv)
                
                D = self.embed_dim
                N = qkv.shape[1]
                h = block.attn.num_heads
                d = D // h
                
                qkv = qkv.reshape(B, N, 3, h, d).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
                h4 = block.attn.proj(attn_out)
            else:
                h4 = block.attn(h2)
            
            # h5: after MSA + residual
            h5 = x + h4
            h5 = self.ssf_h5[i](h5)  # SSF on h5
            
            # h6: after second LN
            h6 = block.norm2(h5)
            
            # MLP forward
            if hasattr(block, 'mlp'):
                # h7: after FC1
                h7 = block.mlp.fc1(h6)
                h7 = self.ssf_h7[i](h7)  # SSF on h7
                
                # h8: after activation
                h8 = block.mlp.act(h7)
                h8 = self.ssf_h8[i](h8)  # SSF on h8
                
                # h9: after FC2
                h9 = block.mlp.fc2(h8)
                h9 = self.ssf_h9[i](h9)  # SSF on h9
            else:
                h7 = block.mlp[0](h6)
                h7 = self.ssf_h7[i](h7)
                h8 = block.mlp[1](h7)
                h8 = self.ssf_h8[i](h8)
                h9 = block.mlp[3](h8)
                h9 = self.ssf_h9[i](h9)
            
            x = h5 + h9
        
        cls_out = x[:, 0, :]
        if hasattr(vit_model, 'head'):
            return vit_model.head(cls_out)
        return cls_out
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
