"""Prompt-based PEFT methods: VPT-Shallow and VPT-Deep.

Reference: Visual Prompt Tuning (Jia et al., ECCV 2022)
"""

import torch
import torch.nn as nn


class VPTShallow(nn.Module):
    """VPT-Shallow: Adds learnable prompts only to the first Transformer layer input.
    
    Formulation (Eq 11 in paper):
        [P̃_0, Z_0] = input
        [P̃_1, Z_1] = L_1([P_0, Z_0])
        [P̃_m, Z_m] = L_m([P̃_{m-1}, Z_{m-1}]) for m = 2..M
    
    Hyperparameters: prompt_number in [5, 10, 50, 100, 200]
    Parameters: 0.0003M ~ 0.153M
    """
    
    def __init__(self, embed_dim=768, num_layers=12, prompt_number=50):
        super().__init__()
        self.prompt_number = prompt_number
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Prompts for the input layer only
        self.prompts = nn.Parameter(torch.randn(1, prompt_number, embed_dim))
        nn.init.xavier_uniform_(self.prompts)
    
    def forward(self, x, vit_model):
        """Insert prompts at the first layer input.
        
        Args:
            x: input images [B, C, H, W]
            vit_model: the ViT model to hook into
        
        Returns:
            logits: classification logits
        """
        B = x.shape[0]
        
        # Get patch embeddings
        x = vit_model.patch_embed(x)
        cls_token = vit_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + vit_model.pos_embed
        
        # Prepend prompts: [prompts, class_token, patch_tokens]
        prompts = self.prompts.expand(B, -1, -1)
        x = torch.cat((prompts, x), dim=1)  # [B, 1+prompt_number+N, D]
        
        # Pass through Transformer layers
        # First layer processes prompts + tokens
        for i, block in enumerate(vit_model.blocks):
            if i == 0:
                # First layer: [prompts, class_token + patches]
                x = block(x)
                # Keep only class_token + patches (discard prompt outputs but they've
                # influenced through attention - actually VPT-Shallow keeps prompts
                # throughout since they influence all layers)
            else:
                x = block(x)
        
        # Extract class token (at position prompt_number after prompts were prepended)
        cls_out = x[:, self.prompt_number, :]
        
        if hasattr(vit_model, 'head'):
            return vit_model.head(cls_out)
        else:
            return cls_out
    
    def get_trainable_params(self):
        """Return count of trainable parameters in millions."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


class VPTDeep(nn.Module):
    """VPT-Deep: Inserts learnable prompts at every Transformer layer input.
    
    Formulation (Eq 12 in paper):
        [ , Z_m] = L_m([P_{m-1}, Z_{m-1}]) for m = 1..M
    
    The prompt outputs are discarded at each layer end; only Z_m propagates.
    
    Hyperparameters: prompt_number in [5, 10, 50, 100]
    Parameters: 0.046M ~ 0.921M
    """
    
    def __init__(self, embed_dim=768, num_layers=12, prompt_number=10):
        super().__init__()
        self.prompt_number = prompt_number
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # One set of prompts per layer
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, prompt_number, embed_dim))
            for _ in range(num_layers)
        ])
        
        for p in self.prompts:
            nn.init.xavier_uniform_(p)
    
    def forward(self, x, vit_model):
        """Insert prompts at each Transformer layer input.
        
        The prompts are prepended to the input of each layer, participate in 
        self-attention, but their outputs are discarded after each layer.
        """
        B = x.shape[0]
        
        # Get patch embeddings
        x = vit_model.patch_embed(x)
        cls_token = vit_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + vit_model.pos_embed
        # Shape: [B, 1+N, D]
        
        for i, block in enumerate(vit_model.blocks):
            # Prepend layer-specific prompts
            layer_prompts = self.prompts[i].expand(B, -1, -1)
            # Concatenate: [prompts, class_token + patches]
            combined = torch.cat((layer_prompts, x), dim=1)
            
            # Forward through block
            combined_out = block(combined)
            
            # Discard prompt outputs, keep only token outputs
            x = combined_out[:, self.prompt_number:, :]
        
        # Take class token
        cls_out = x[:, 0, :]
        
        if hasattr(vit_model, 'head'):
            return vit_model.head(cls_out)
        else:
            return cls_out
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
