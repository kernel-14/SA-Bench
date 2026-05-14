"""
VPT: Visual Prompt Tuning.
Jia et al., 2022.

VPT-Shallow: Adds learnable prompts to the input of the first Transformer layer.
VPT-Deep: Adds learnable prompts to the input of every Transformer layer.
"""

import torch
import torch.nn as nn
import math


class VPTViT(nn.Module):
    """ViT with Visual Prompt Tuning."""
    
    def __init__(self, vit_model, num_prompts=10, deep=True):
        super().__init__()
        self.vit = vit_model
        self.num_prompts = num_prompts
        self.deep = deep
        
        # Get embedding dimension
        embed_dim = vit_model.embed_dim
        num_layers = len(vit_model.blocks)
        
        if deep:
            # VPT-Deep: prompts for each layer
            self.prompt_embeddings = nn.ParameterList([
                nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
                for _ in range(num_layers)
            ])
        else:
            # VPT-Shallow: prompts only for first layer
            self.prompt_embeddings = nn.ParameterList([
                nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
            ])
        
        # Initialize prompts with truncated normal
        for prompt in self.prompt_embeddings:
            nn.init.trunc_normal_(prompt, std=0.02)
    
    def forward(self, x):
        # Patch embedding
        x = self.vit.patch_embed(x)
        
        # Add cls token
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        
        # Process through transformer blocks with prompts
        for i, block in enumerate(self.vit.blocks):
            if self.deep:
                # VPT-Deep: add prompts at each layer
                prompt = self.prompt_embeddings[i].expand(x.shape[0], -1, -1)
                # Prepend prompts to the sequence (after cls token)
                x = torch.cat([x[:, :1, :], prompt, x[:, 1:, :]], dim=1)
                x = block(x)
                # Remove prompts from output (keep cls token and patch tokens)
                x = torch.cat([x[:, :1, :], x[:, 1 + self.num_prompts:, :]], dim=1)
            else:
                # VPT-Shallow: only add prompts at first layer
                if i == 0:
                    prompt = self.prompt_embeddings[0].expand(x.shape[0], -1, -1)
                    x = torch.cat([x[:, :1, :], prompt, x[:, 1:, :]], dim=1)
                    x = block(x)
                    # Keep prompts for subsequent layers (they propagate)
                else:
                    x = block(x)
        
        # Final norm
        x = self.vit.norm(x)
        
        # Classification head using cls token
        cls_output = x[:, 0]
        return self.vit.head(cls_output)
    
    def get_classifier(self):
        return self.vit.head
    
    def reset_classifier(self, num_classes):
        self.vit.reset_classifier(num_classes)


def apply_vpt(model, num_prompts=10, deep=True, **kwargs):
    """
    Apply VPT to a ViT model.
    
    Args:
        model: ViT model (timm)
        num_prompts: Number of prompt tokens
        deep: If True, use VPT-Deep; otherwise VPT-Shallow
        **kwargs: Additional arguments (unused)
    
    Returns:
        VPTViT model
    """
    vpt_model = VPTViT(model, num_prompts=num_prompts, deep=deep)
    
    # Freeze all backbone parameters
    for param in vpt_model.vit.parameters():
        param.requires_grad = False
    
    # Unfreeze prompt embeddings
    for prompt in vpt_model.prompt_embeddings:
        prompt.requires_grad = True
    
    # Always keep head trainable
    for param in vpt_model.vit.head.parameters():
        param.requires_grad = True
    
    return vpt_model
