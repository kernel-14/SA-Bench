"""PEFT-wrapped Vision Transformer with all PEFT methods integrated.

This module provides a modified ViT forward pass that exposes intermediate
features (h1..h8) and applies PEFT adaptations at the correct positions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit import (
    PatchEmbed, MlpBlock, Attention, LayerNorm, DropPath, VisionTransformer
)
from models.peft import (
    VPTShallow, VPTDeep,
    PfeifAdapter, HoulAdapter, AdaptFormer, RepAdapter, Convpass,
    DiffFitScales, SSFModule,
    LoRAModule, FacT_TT, FacT_TK,
    count_trainable_params,
)


def build_peft_vit(name, vit_base, peft_config):
    """Build a PEFT-wrapped ViT with the specified method.

    Args:
        name: PEFT method name (e.g., 'vpt_shallow', 'lora', etc.)
        vit_base: base VisionTransformer instance
        peft_config: dict with method-specific hyperparameters

    Returns:
        PEFTViT instance
    """
    peft_vit = PEFTViT(vit_base, peft_method=name, peft_config=peft_config)
    return peft_vit


class PEFTViTBlock(nn.Module):
    """Transformer block with PEFT hooks at intermediate features.

    Feature positions (per paper Figure 9):
        h1 = input (Z_{m-1})
        h2 = LN1(h1)           -> before Q/K/V
        h3 = Q/K/V projections
        h4 = attention output
        h5 = after MSA + residual
        h6 = h5 (identity for notation)
        h7 = LN2(h6)           -> before MLP
        h8 = FC1(h7)           -> MLP first layer output
        h9 = after MLP + residual
        h10 = output
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mlp_hidden = int(dim * mlp_ratio)

        self.norm1 = LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = LayerNorm(dim)
        self.fc1 = nn.Linear(dim, self.mlp_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.mlp_hidden, dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp_drop = nn.Dropout(drop)

    def get_qkv(self, h2):
        """Return Q, K, V without computing attention."""
        B, N, C = h2.shape
        qkv = self.qkv(h2).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def compute_attention(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        B, nh, N, hd = attn.shape
        out = (attn @ v).transpose(1, 2).reshape(B, N, nh * hd)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def forward_vanilla(self, x):
        """Standard forward without PEFT hooks."""
        h1 = x
        h2 = self.norm1(h1)
        qkv = self.qkv(h2).reshape(h1.shape[0], h1.shape[1], 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        h4 = (attn @ v).transpose(1, 2).reshape(h1.shape[0], h1.shape[1], self.dim)
        h4 = self.proj(h4)
        h4 = self.proj_drop(h4)
        h5 = self.drop_path1(h4) + h1
        h7 = self.norm2(h5)
        h8 = self.fc1(h7)
        h8 = self.act(h8)
        h8 = self.mlp_drop(h8)
        h9 = self.fc2(h8)
        h9 = self.mlp_drop(h9)
        h9 = self.drop_path2(h9) + h5
        return h9

    def get_intermediate_features(self, x):
        """Extract intermediate features h1..h8."""
        h1 = x
        h2 = self.norm1(h1)
        q, k, v = self.get_qkv(h2)
        h4 = self.compute_attention(q, k, v)
        h5 = self.drop_path1(h4) + h1
        h7 = self.norm2(h5)
        h8 = self.fc1(h7)
        h8 = self.act(h8)
        h8 = self.mlp_drop(h8)
        return h1, h2, (q, k, v), h4, h5, h7, h8

    def forward_with_qkv_delta(self, x, delta_Q=None, delta_K=None, delta_V=None, delta_O=None):
        """Forward with LoRA/FacT delta applied to Q, K, V, O projections."""
        h1 = x
        h2 = self.norm1(h1)
        q, k, v = self.get_qkv(h2)
        B, N, C = h2.shape
        if delta_Q is not None:
            delta_Q_reshaped = delta_Q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = q + delta_Q_reshaped
        if delta_K is not None:
            delta_K_reshaped = delta_K.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k + delta_K_reshaped
        if delta_V is not None:
            delta_V_reshaped = delta_V.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v + delta_V_reshaped
        h4 = self.compute_attention(q, k, v)
        if delta_O is not None:
            h4 = h4 + delta_O
        h5 = self.drop_path1(h4) + h1
        h7 = self.norm2(h5)
        h8 = self.fc1(h7)
        h8 = self.act(h8)
        h8 = self.mlp_drop(h8)
        h9 = self.fc2(h8)
        h9 = self.mlp_drop(h9)
        h9 = self.drop_path2(h9) + h5
        return h9, h2, h7

    def forward_with_weight_delta(self, x, delta_Q=None, delta_K=None, delta_V=None,
                                   delta_O=None, delta_W1=None, delta_W2=None):
        """Forward with deltas applied to Q/K/V/O/W1/W2 weight matrices (for FacT)."""
        h1 = x
        h2 = self.norm1(h1)

        # Apply QKV with weight deltas
        B, N, C = h2.shape
        orig_qkv_weight = self.qkv.weight  # (3*C, C)
        orig_qkv_bias = self.qkv.bias

        if delta_Q is not None or delta_K is not None or delta_V is not None:
            delta_qkv = torch.cat([
                delta_Q if delta_Q is not None else torch.zeros(C, C, device=h2.device),
                delta_K if delta_K is not None else torch.zeros(C, C, device=h2.device),
                delta_V if delta_V is not None else torch.zeros(C, C, device=h2.device),
            ], dim=0)
            qkv_out = F.linear(h2, orig_qkv_weight + delta_qkv, orig_qkv_bias)
        else:
            qkv_out = self.qkv(h2)

        qkv = qkv_out.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        h4 = self.compute_attention(q, k, v)

        # O projection delta
        if delta_O is not None:
            h4 = F.linear(h4, self.proj.weight + delta_O, self.proj.bias)
        else:
            h4 = self.proj(h4)
        h4 = self.proj_drop(h4)

        h5 = self.drop_path1(h4) + h1
        h7 = self.norm2(h5)

        # MLP with weight deltas
        if delta_W1 is not None:
            h8 = F.linear(h7, self.fc1.weight + delta_W1, self.fc1.bias)
        else:
            h8 = self.fc1(h7)
        h8 = self.act(h8)
        h8 = self.mlp_drop(h8)

        if delta_W2 is not None:
            h9 = F.linear(h8, self.fc2.weight + delta_W2, self.fc2.bias)
        else:
            h9 = self.fc2(h8)
        h9 = self.mlp_drop(h9)
        h9 = self.drop_path2(h9) + h5
        return h9


class PEFTViT(nn.Module):
    """ViT with PEFT modules applied at every layer."""

    def __init__(self, vit_base, peft_method, peft_config=None):
        super().__init__()
        self.peft_method = peft_method
        self.peft_config = peft_config or {}

        # Copy backbone structure
        self.img_size = vit_base.patch_embed.img_size
        self.patch_size = vit_base.patch_embed.patch_size
        self.embed_dim = vit_base.embed_dim
        self.depth = vit_base.depth
        self.num_heads = len(vit_base.blocks) and vit_base.blocks[0].attn.num_heads or 12
        self.mlp_ratio = 4.0
        self.mlp_hidden = self.embed_dim * self.mlp_ratio

        self.patch_embed = vit_base.patch_embed
        self.cls_token = vit_base.cls_token
        self.pos_embed = vit_base.pos_embed
        self.pos_drop = vit_base.pos_drop

        # Build blocks with PEFT hooks
        drop_path_rate = self.peft_config.get('drop_path_rate', 0.1)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([
            PEFTViTBlock(self.embed_dim, self.num_heads, mlp_ratio=self.mlp_ratio,
                         drop_path=dpr[i])
            for i in range(self.depth)
        ])

        # Copy pretrained weights into blocks
        with torch.no_grad():
            for i in range(self.depth):
                src = vit_base.blocks[i]
                dst = self.blocks[i]
                dst.norm1.weight.copy_(src.norm1.weight)
                dst.norm1.bias.copy_(src.norm1.bias)
                dst.qkv.weight.copy_(src.attn.qkv.weight)
                dst.qkv.bias.copy_(src.attn.qkv.bias)
                dst.proj.weight.copy_(src.attn.proj.weight)
                dst.proj.bias.copy_(src.attn.proj.bias)
                dst.norm2.weight.copy_(src.norm2.weight)
                dst.norm2.bias.copy_(src.norm2.bias)
                dst.fc1.weight.copy_(src.mlp.fc1.weight)
                dst.fc1.bias.copy_(src.mlp.fc1.bias)
                dst.fc2.weight.copy_(src.mlp.fc2.weight)
                dst.fc2.bias.copy_(src.mlp.fc2.bias)

        self.norm = vit_base.norm
        self.head = vit_base.head

        # Initialize PEFT method
        self._build_peft()

    def _build_peft(self):
        self.vpt_shallow = None
        self.vpt_deep = None
        self.pfeif_adapter = None
        self.houl_adapter = None
        self.adaptformer = None
        self.repadapter = None
        self.convpass = None
        self.difffit_scales = None
        self.ssf = None
        self.lora = None
        self.fact_tt = None
        self.fact_tk = None

        if self.peft_method == 'vpt_shallow':
            prompt_num = self.peft_config.get('prompt_num', 10)
            self.vpt_shallow = VPTShallow(prompt_num, self.embed_dim, self.depth)

        elif self.peft_method == 'vpt_deep':
            prompt_num = self.peft_config.get('prompt_num', 10)
            self.vpt_deep = VPTDeep(prompt_num, self.embed_dim, self.depth)

        elif self.peft_method == 'pfeif_adapter':
            bn = self.peft_config.get('bottle_neck', 8)
            sf = self.peft_config.get('scale_factor', 1.0)
            self.pfeif_adapter = PfeifAdapter(self.embed_dim, bn, sf, self.depth)

        elif self.peft_method == 'houl_adapter':
            bn = self.peft_config.get('bottle_neck', 8)
            sf = self.peft_config.get('scale_factor', 1.0)
            self.houl_adapter = HoulAdapter(self.embed_dim, bn, sf, self.depth)

        elif self.peft_method == 'adaptformer':
            bn = self.peft_config.get('bottle_neck', 8)
            sf = self.peft_config.get('scale_factor', 0.1)
            self.adaptformer = AdaptFormer(self.embed_dim, bn, sf, self.depth)

        elif self.peft_method == 'repadapter':
            bn = self.peft_config.get('bottle_neck', 16)
            sf = self.peft_config.get('scale_factor', 1.0)
            ng = self.peft_config.get('num_groups', 4)
            self.repadapter = RepAdapter(self.embed_dim, bn, sf, ng, self.depth)

        elif self.peft_method == 'convpass':
            bn = self.peft_config.get('bottle_neck', 8)
            sf = self.peft_config.get('scale_factor', 1.0)
            ks = self.peft_config.get('kernel_size', 3)
            self.convpass = Convpass(self.embed_dim, bn, sf, ks, self.depth)

        elif self.peft_method == 'bitfit':
            pass  # handled via freeze/unfreeze

        elif self.peft_method == 'layernorm':
            pass  # handled via freeze/unfreeze

        elif self.peft_method == 'difffit':
            self.difffit_scales = DiffFitScales(self.embed_dim, self.depth)

        elif self.peft_method == 'ssf':
            self.ssf = SSFModule(self.embed_dim, self.depth, self.mlp_hidden)

        elif self.peft_method == 'lora':
            rank = self.peft_config.get('rank', 8)
            self.lora = LoRAModule(self.embed_dim, rank, self.depth)

        elif self.peft_method == 'fact_tt':
            bn = self.peft_config.get('bottle_neck', 16)
            sf = self.peft_config.get('scale_factor', 1.0)
            self.fact_tt = FacT_TT(self.embed_dim, self.depth, bn, sf)

        elif self.peft_method == 'fact_tk':
            bn = self.peft_config.get('bottle_neck', 32)
            sf = self.peft_config.get('scale_factor', 1.0)
            self.fact_tk = FacT_TK(self.embed_dim, self.depth, bn, sf)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        for i, block in enumerate(self.blocks):
            if self.peft_method == 'vpt_shallow':
                x = self.vpt_shallow(x, i)
                x = block.forward_vanilla(x)
                if i == 0 and self.vpt_shallow.prompt_num > 0:
                    x = x[:, self.vpt_shallow.prompt_num:, :]

            elif self.peft_method == 'vpt_deep' and self.vpt_deep is not None:
                x = self.vpt_deep(x, i)
                x = block.forward_vanilla(x)
                x = x[:, self.vpt_deep.prompt_num:, :]

            elif self.peft_method == 'pfeif_adapter' and self.pfeif_adapter is not None:
                x = block.forward_vanilla(x)
                x = self.pfeif_adapter(x, i)

            elif self.peft_method == 'houl_adapter' and self.houl_adapter is not None:
                h1, h2, qkv, h4, h5, h7, h8 = block.get_intermediate_features(x)
                h5 = self.houl_adapter.forward_msa(h5, i)
                h9 = block.fc2(h8)
                h9 = block.mlp_drop(h9)
                h9 = block.drop_path2(h9) + h5
                x = self.houl_adapter.forward_mlp(h9, i)

            elif self.peft_method == 'adaptformer' and self.adaptformer is not None:
                h1, h2, qkv, h4, h5, h7, h8 = block.get_intermediate_features(x)
                delta = self.adaptformer(h7, i)
                h9 = block.fc2(h8)
                h9 = block.mlp_drop(h9)
                h9 = block.drop_path2(h9) + h5 + delta
                x = h9

            elif self.peft_method == 'repadapter' and self.repadapter is not None:
                h1, h2, qkv, h4, h5, h7, h8 = block.get_intermediate_features(x)
                h2_adapted = self.repadapter.forward_msa(h2, i)
                # Recompute attention with adapted h2
                q, k, v = block.get_qkv(h2_adapted)
                h4 = block.compute_attention(q, k, v)
                h5 = block.drop_path1(h4) + h1
                h7 = block.norm2(h5)
                h7_adapted = self.repadapter.forward_mlp(h7, i)
                h8 = block.fc1(h7_adapted)
                h8 = block.act(h8)
                h8 = block.mlp_drop(h8)
                h9 = block.fc2(h8)
                h9 = block.mlp_drop(h9)
                h9 = block.drop_path2(h9) + h5
                x = h9

            elif self.peft_method == 'convpass' and self.convpass is not None:
                h1, h2, qkv, h4, h5, h7, h8 = block.get_intermediate_features(x)
                delta_msa = self.convpass.forward_msa(h2, i)
                h5 = h5 + delta_msa
                h7 = block.norm2(h5)
                h8 = block.fc1(h7)
                h8 = block.act(h8)
                h8 = block.mlp_drop(h8)
                h9 = block.fc2(h8)
                h9 = block.mlp_drop(h9)
                h9 = block.drop_path2(h9) + h5
                delta_mlp = self.convpass.forward_mlp(h7, i)
                x = h9 + delta_mlp

            elif self.peft_method == 'bitfit':
                x = block.forward_vanilla(x)

            elif self.peft_method == 'layernorm':
                x = block.forward_vanilla(x)

            elif self.peft_method == 'difffit' and self.difffit_scales is not None:
                h1, h2, qkv, h4, h5, h7, h8 = block.get_intermediate_features(x)
                h5 = self.difffit_scales.forward_msa(h5, i)
                h9 = block.fc2(h8)
                h9 = block.mlp_drop(h9)
                h9 = block.drop_path2(h9) + h5
                x = self.difffit_scales.forward_mlp(h9, i)

            elif self.peft_method == 'ssf' and self.ssf is not None:
                B = x.shape[0]
                h1 = x
                h2 = block.norm1(h1)
                h2 = self.ssf.modulate(h2, self.ssf.scale_h2, self.ssf.shift_h2, i)

                # Get QKV output and modulate h3 (concatenated QKV features)
                qkv_out = block.qkv(h2)  # B, N, 3*C
                h3_mod = self.ssf.modulate(qkv_out, self.ssf.scale_h3, self.ssf.shift_h3, i)
                qkv = h3_mod.reshape(B, -1, 3, block.num_heads, block.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                h4 = block.compute_attention(q, k, v)
                h5 = block.drop_path1(h4) + h1
                h5 = self.ssf.modulate(h5, self.ssf.scale_h5, self.ssf.shift_h5, i)

                h7 = block.norm2(h5)
                h7 = self.ssf.modulate(h7, self.ssf.scale_h7, self.ssf.shift_h7, i)
                h8 = block.fc1(h7)
                h8 = self.ssf.modulate(h8, self.ssf.scale_h8, self.ssf.shift_h8, i)
                h8 = block.act(h8)
                h8 = block.mlp_drop(h8)
                h9 = block.fc2(h8)
                h9 = block.mlp_drop(h9)
                h9 = block.drop_path2(h9) + h5
                x = self.ssf.modulate(h9, self.ssf.scale_h9, self.ssf.shift_h9, i)

            elif self.peft_method == 'lora' and self.lora is not None:
                h2 = block.norm1(x)
                delta_Q, delta_V = self.lora(h2, i)
                x, _, _ = block.forward_with_qkv_delta(x, delta_Q, delta_V)

            elif self.peft_method == 'fact_tt' and self.fact_tt is not None:
                delta_Q = self.fact_tt.get_delta(i, 'Q')
                delta_K = self.fact_tt.get_delta(i, 'K')
                delta_V = self.fact_tt.get_delta(i, 'V')
                delta_O = self.fact_tt.get_delta(i, 'O')
                delta_W1 = self.fact_tt.get_delta(i, 'W1')
                delta_W2 = self.fact_tt.get_delta(i, 'W2')
                x = block.forward_with_weight_delta(x, delta_Q, delta_K, delta_V,
                                                     delta_O, delta_W1, delta_W2)

            elif self.peft_method == 'fact_tk' and self.fact_tk is not None:
                delta_Q = self.fact_tk.get_delta(i, 'Q')
                delta_K = self.fact_tk.get_delta(i, 'K')
                delta_V = self.fact_tk.get_delta(i, 'V')
                delta_O = self.fact_tk.get_delta(i, 'O')
                delta_W1 = self.fact_tk.get_delta(i, 'W1')
                delta_W2 = self.fact_tk.get_delta(i, 'W2')
                x = block.forward_with_weight_delta(x, delta_Q, delta_K, delta_V,
                                                     delta_O, delta_W1, delta_W2)

            else:
                x = block.forward_vanilla(x)

        x = self.norm(x)
        return x

    def forward_features_with_wise(self, x, alpha=1.0):
        """Forward with WiSE alpha blending for PEFT modules."""
        if self.peft_method not in ('pfeif_adapter', 'houl_adapter', 'adaptformer',
                                     'convpass', 'lora', 'fact_tt', 'fact_tk'):
            return self.forward_features(x)

        B = x.shape[0]
        x = self.patch_embed(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        for i, block in enumerate(self.blocks):
            if self.peft_method == 'pfeif_adapter' and self.pfeif_adapter is not None:
                x = block.forward_vanilla(x)
                x = self.pfeif_adapter.apply_wise(x, i, alpha)

            elif self.peft_method == 'lora' and self.lora is not None:
                h2 = block.norm1(x)
                delta_Q, delta_V = self.lora.apply_wise(h2, i, alpha)
                x, _, _ = block.forward_with_qkv_delta(x, delta_Q, delta_V)

            else:
                x = block.forward_vanilla(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        if self.cls_token is not None:
            features = features[:, 0]
        else:
            features = features.mean(dim=1)
        return self.head(features)

    def forward_with_wise(self, x, alpha=1.0):
        features = self.forward_features_with_wise(x, alpha)
        if self.cls_token is not None:
            features = features[:, 0]
        else:
            features = features.mean(dim=1)
        return self.head(features)
