"""Efficient Selective Tuning: LoRA, FacT_TT, FacT_TK.

These methods learn additive residuals to original parameters with low-rank constraints.
The learned residuals can be merged into backbone weights for zero inference overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """LoRA linear layer wrapper: W + W_up @ W_down with low-rank decomposition.
    
    ΔW = W_up @ W_down, where W_down in R^{r×D}, W_up in R^{D×r}, r << D.
    """
    
    def __init__(self, linear_layer, rank=8, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # Store reference to original weight
        self.linear = linear_layer
        
        # LoRA decomposition
        self.lora_down = nn.Linear(self.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, self.out_features, bias=False)
        
        # Initialize: down with kaiming, up with zeros
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
        # Scaling
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Original linear output + LoRA residual
        return self.linear(x) + self.scaling * self.lora_up(self.lora_down(x))
    
    def merge_weights(self):
        """Merge LoRA weights into the original linear layer for inference."""
        merged_weight = self.linear.weight.data + self.scaling * (
            self.lora_up.weight @ self.lora_down.weight
        )
        self.linear.weight.data = merged_weight


class LoRA(nn.Module):
    """LoRA (Low-Rank Adaptation): Updates Q and V projection matrices via low-rank.
    
    Applies low-rank decomposition to Query and Value weight matrices in MSA:
        h3 = LoRA(h2) + h3
        LoRA(h2) = [W_down^Q @ W_up^Q @ h2, 0, W_down^V @ W_up^V @ h2]
    
    Hyperparameters: lora_rank in [1, 8, 16, 32]
    Parameters: 0.036M ~ 1.179M
    """
    
    def __init__(self, embed_dim=768, num_layers=12, rank=8, alpha=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA for Q and V in each layer
        self.lora_Q_down = nn.ParameterList()
        self.lora_Q_up = nn.ParameterList()
        self.lora_V_down = nn.ParameterList()
        self.lora_V_up = nn.ParameterList()
        
        for _ in range(num_layers):
            # Q: D x D -> r x D + D x r
            self.lora_Q_down.append(nn.Parameter(torch.zeros(rank, embed_dim)))
            self.lora_Q_up.append(nn.Parameter(torch.zeros(embed_dim, rank)))
            # V: D x D -> r x D + D x r
            self.lora_V_down.append(nn.Parameter(torch.zeros(rank, embed_dim)))
            self.lora_V_up.append(nn.Parameter(torch.zeros(embed_dim, rank)))
        
        self._init_weights()
    
    def _init_weights(self):
        for i in range(self.num_layers):
            nn.init.kaiming_uniform_(self.lora_Q_down[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_V_down[i], a=math.sqrt(5))
            # up matrices initialized to zero so initially ΔW = 0
    
    def apply_lora_to_attn(self, block, layer_idx):
        """Hook LoRA into a block's attention mechanism.
        
        This modifies the Q and V projections of the attention block.
        """
        # For timm-style ViT blocks
        if hasattr(block, 'attn'):
            attn = block.attn
            # Store original forward
            original_forward = attn.forward
            
            def lora_forward(x):
                B, N, D = x.shape
                
                # Original QKV projection (frozen)
                qkv = attn.qkv(x)
                qkv = qkv.reshape(B, N, 3, attn.num_heads, D // attn.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # Apply LoRA to Q
                lora_q = (x @ self.lora_Q_down[layer_idx].T) @ self.lora_Q_up[layer_idx].T
                lora_q = lora_q.reshape(B, N, attn.num_heads, D // attn.num_heads).permute(0, 2, 1, 3)
                q = q + self.scaling * lora_q
                
                # Apply LoRA to V
                lora_v = (x @ self.lora_V_down[layer_idx].T) @ self.lora_V_up[layer_idx].T
                lora_v = lora_v.reshape(B, N, attn.num_heads, D // attn.num_heads).permute(0, 2, 1, 3)
                v = v + self.scaling * lora_v
                
                # Attention
                attn_out = F.scaled_dot_product_attention(q, k, v)
                attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
                return attn.proj(attn_out)
            
            attn.forward = lora_forward
            self._original_forwards.append((attn, original_forward))
    
    def forward(self, x, vit_model):
        """Forward pass - must call apply_lora_to_attn first."""
        return vit_model(x)
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


class FacT_TT(nn.Module):
    """FacT with Tensor-Train decomposition.
    
    Stacks all weight matrices into a 3D tensor W_FacT in R^{12M × D × D}
    and decomposes the update ΔW_FacT using Tensor-Train format:
    
    FacT_TT: ΔW_FacT = s · Σ ×₂ U^T ×₃ V^T
    
    where U ∈ R^{D×r}, V ∈ R^{D×r}, Σ ∈ R^{12L×r×r}
    
    The update affects W_Q, W_K, W_V, W_O in MSA and W_1, W_2 in MLP.
    
    Hyperparameters: fact_scale in [0.01, 0.1, 1, 10, 100],
                     fact_bottleneck in [8, 16, 32]
    Parameters: 0.021M ~ 0.196M
    """
    
    def __init__(self, embed_dim=768, num_layers=12, bottleneck_dim=16, scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim  # r
        self.scale = scale
        
        # Number of matrices: 4 per MSA (Q,K,V,O) + 2 per MLP (W1,W2) = 6 per layer
        # Actually W1 is D×4D and W2 is 4D×D, we simplify to D×D
        self.num_matrices_per_layer = 4  # Q, K, V, O
        self.total_matrices = num_layers * self.num_matrices_per_layer
        
        # TT decomposition factors
        # U, V: shared across all layers
        self.U = nn.Parameter(torch.randn(embed_dim, bottleneck_dim))
        self.V = nn.Parameter(torch.randn(embed_dim, bottleneck_dim))
        
        # Σ: layer-specific core tensor
        self.Sigma = nn.Parameter(torch.randn(self.total_matrices, bottleneck_dim, bottleneck_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.Sigma)
    
    def get_delta_W(self, layer_idx):
        """Get the weight update for a specific layer's matrices.
        
        Returns: [num_matrices, D, D] tensor of updates
        """
        start_idx = layer_idx * self.num_matrices_per_layer
        end_idx = start_idx + self.num_matrices_per_layer
        
        Sigma_subset = self.Sigma[start_idx:end_idx]  # [4, r, r]
        
        # ΔW = s · Σ ×₂ U^T ×₃ V^T
        # For each matrix: ΔW = s · U @ Σ_i @ V^T
        delta_W = self.scale * torch.einsum('dr,frs,es->fde', self.U, Sigma_subset, self.V)
        
        return delta_W  # [4, D, D]
    
    def forward(self, x, vit_model):
        """Forward with FacT_TT modifications."""
        B = x.shape[0]
        
        x = vit_model.patch_embed(x)
        cls_token = vit_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + vit_model.pos_embed
        
        for i, block in enumerate(vit_model.blocks):
            delta_Ws = self.get_delta_W(i)  # [4, D, D]
            delta_W_q, delta_W_k, delta_W_v, delta_W_o = delta_Ws
            
            # h2
            z_norm = block.norm1(x)
            
            # Manual QKV with FacT updates
            # Original QKV weights are frozen
            if hasattr(block.attn, 'qkv'):
                # Single QKV projection
                D = self.embed_dim
                N = z_norm.shape[1]
                h = block.attn.num_heads
                d = D // h
                
                qkv_weight = block.attn.qkv.weight  # [3*D, D]
                qkv_bias = block.attn.qkv.bias if block.attn.qkv.bias is not None else None
                
                # Split QKV
                W_q = qkv_weight[:D]
                W_k = qkv_weight[D:2*D]
                W_v = qkv_weight[2*D:]
                
                # Apply FacT updates
                q = F.linear(z_norm, W_q + delta_W_q)
                k = F.linear(z_norm, W_k + delta_W_k)
                v = F.linear(z_norm, W_v + delta_W_v)
                
                q = q.reshape(B, N, h, d).permute(0, 2, 1, 3)
                k = k.reshape(B, N, h, d).permute(0, 2, 1, 3)
                v = v.reshape(B, N, h, d).permute(0, 2, 1, 3)
                
                attn_out = F.scaled_dot_product_attention(q, k, v)
                attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
                
                # Output projection with FacT update
                proj_weight = block.attn.proj.weight
                proj_bias = block.attn.proj.bias if block.attn.proj.bias is not None else None
                h4 = F.linear(attn_out, proj_weight + delta_W_o, proj_bias)
                
                x = x + h4
            else:
                x = x + block.attn(z_norm)
            
            # MLP (no FacT modifications in this simplified version)
            z_norm2 = block.norm2(x)
            x = x + block.mlp(z_norm2)
        
        cls_out = x[:, 0, :]
        if hasattr(vit_model, 'head'):
            return vit_model.head(cls_out)
        return cls_out
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


class FacT_TK(nn.Module):
    """FacT with Tucker decomposition.
    
    FacT_TK: ΔW_FacT = s · A ×₁ B^T ×₂ U^T ×₃ V^T
    
    where U ∈ R^{D×r}, V ∈ R^{D×r}, B ∈ R^{12L×r}, A ∈ R^{r×r×r}
    
    Hyperparameters: fact_bottleneck in [16, 32, 64],
                     fact_scale in [0.01, 0.1, 1, 10, 100]
    Parameters: 0.030M ~ 0.369M
    """
    
    def __init__(self, embed_dim=768, num_layers=12, bottleneck_dim=16, scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim  # r
        self.scale = scale
        
        self.num_matrices_per_layer = 4  # Q, K, V, O
        self.total_matrices = num_layers * self.num_matrices_per_layer
        
        # Tucker decomposition factors
        self.U = nn.Parameter(torch.randn(embed_dim, bottleneck_dim))
        self.V = nn.Parameter(torch.randn(embed_dim, bottleneck_dim))
        self.B = nn.Parameter(torch.randn(self.total_matrices, bottleneck_dim))
        self.A = nn.Parameter(torch.randn(bottleneck_dim, bottleneck_dim, bottleneck_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.A)
    
    def get_delta_W(self, layer_idx):
        """Get weight update via Tucker decomposition."""
        start_idx = layer_idx * self.num_matrices_per_layer
        end_idx = start_idx + self.num_matrices_per_layer
        
        B_subset = self.B[start_idx:end_idx]  # [4, r]
        
        # ΔW = s · A ×₁ B^T ×₂ U^T ×₃ V^T
        # For each matrix f: ΔW_f = s * sum_{i,j,k} A_{i,j,k} * B_{f,i} * U_{:,j} * V_{:,k}^T
        # Vectorized: ΔW = s · U @ (A ×₁ B_subset) @ V^T
        # A ×₁ B_subset: contract mode 1 of A (size r×r×r) with B_subset (size 4×r)
        # Result: [4, r, r]
        core = torch.einsum('fr,rijk->fjk', B_subset, self.A)  # [4, r, r]
        delta_W = self.scale * torch.einsum('dr,frs,es->fde', self.U, core, self.V)
        
        return delta_W
    
    def forward(self, x, vit_model):
        """Forward with FacT_TK modifications."""
        B = x.shape[0]
        
        x = vit_model.patch_embed(x)
        cls_token = vit_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + vit_model.pos_embed
        
        for i, block in enumerate(vit_model.blocks):
            delta_Ws = self.get_delta_W(i)
            delta_W_q, delta_W_k, delta_W_v, delta_W_o = delta_Ws
            
            z_norm = block.norm1(x)
            
            if hasattr(block.attn, 'qkv'):
                D = self.embed_dim
                N = z_norm.shape[1]
                h = block.attn.num_heads
                d = D // h
                
                qkv_weight = block.attn.qkv.weight
                W_q = qkv_weight[:D]
                W_k = qkv_weight[D:2*D]
                W_v = qkv_weight[2*D:]
                
                q = F.linear(z_norm, W_q + delta_W_q)
                k = F.linear(z_norm, W_k + delta_W_k)
                v = F.linear(z_norm, W_v + delta_W_v)
                
                q = q.reshape(B, N, h, d).permute(0, 2, 1, 3)
                k = k.reshape(B, N, h, d).permute(0, 2, 1, 3)
                v = v.reshape(B, N, h, d).permute(0, 2, 1, 3)
                
                attn_out = F.scaled_dot_product_attention(q, k, v)
                attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
                
                proj_weight = block.attn.proj.weight
                proj_bias = block.attn.proj.bias if block.attn.proj.bias is not None else None
                h4 = F.linear(attn_out, proj_weight + delta_W_o, proj_bias)
                
                x = x + h4
            else:
                x = x + block.attn(z_norm)
            
            z_norm2 = block.norm2(x)
            x = x + block.mlp(z_norm2)
        
        cls_out = x[:, 0, :]
        if hasattr(vit_model, 'head'):
            return vit_model.head(cls_out)
        return cls_out
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
