"""All PEFT (Parameter-Efficient Fine-Tuning) methods for Vision Transformers.

Implements 14 PEFT methods divided into four categories:
1. Prompt-based: VPT-Shallow, VPT-Deep
2. Adapter-based: Pfeif. Adapter, Houl. Adapter, AdaptFormer, RepAdapter, Convpass
3. Direct selective tuning: BitFit, DiffFit, LayerNorm, SSF
4. Efficient selective tuning: LoRA, FacT_TT, FacT_TK
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy


# ==============================================================================
# Prompt-based Methods
# ==============================================================================

class VPTShallow(nn.Module):
    """Visual Prompt Tuning - Shallow: prepend learnable prompts to first layer input."""

    def __init__(self, prompt_num=10, embed_dim=768, depth=12):
        super().__init__()
        self.prompt_num = prompt_num
        self.prompts = nn.Parameter(torch.zeros(1, prompt_num, embed_dim))
        nn.init.trunc_normal_(self.prompts, std=0.02)

    def forward(self, x, layer_idx):
        if layer_idx == 0:
            B = x.shape[0]
            prompts = self.prompts.expand(B, -1, -1)
            return torch.cat([prompts, x], dim=1)
        return x

    def get_trainable_params(self):
        return list(self.parameters())


class VPTDeep(nn.Module):
    """Visual Prompt Tuning - Deep: prepend learnable prompts to every layer input."""

    def __init__(self, prompt_num=10, embed_dim=768, depth=12):
        super().__init__()
        self.prompt_num = prompt_num
        self.depth = depth
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.zeros(1, prompt_num, embed_dim))
            for _ in range(depth)
        ])
        for p in self.prompts:
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x, layer_idx):
        B = x.shape[0]
        prompts = self.prompts[layer_idx].expand(B, -1, -1)
        out = torch.cat([prompts, x], dim=1)
        return out

    def filter_output(self, x, layer_idx):
        """Remove prompts from output after processing."""
        return x[:, self.prompt_num:, :]

    def get_trainable_params(self):
        return list(self.parameters())


# ==============================================================================
# Adapter-based Methods
# ==============================================================================

class AdapterModule(nn.Module):
    """Generic bottleneck adapter module: x + s * W_up * act(W_down * x)."""

    def __init__(self, embed_dim, bottle_neck, scale_factor=1.0, act_layer=nn.GELU):
        super().__init__()
        self.scale_factor = scale_factor
        self.down = nn.Linear(embed_dim, bottle_neck)
        self.act = act_layer()
        self.up = nn.Linear(bottle_neck, embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.trunc_normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = x
        out = self.down(x)
        out = self.act(out)
        out = self.up(out)
        return residual + self.scale_factor * out


class PfeifAdapter(nn.Module):
    """Pfeiffer Adapter: insert adapter only after MLP block (h9)."""

    def __init__(self, embed_dim=768, bottle_neck=8, scale_factor=1.0, depth=12):
        super().__init__()
        self.adapters = nn.ModuleList([
            AdapterModule(embed_dim, bottle_neck, scale_factor)
            for _ in range(depth)
        ])

    def forward(self, h9, layer_idx):
        return self.adapters[layer_idx](h9)

    def apply_wise(self, h9, layer_idx, alpha):
        """WiSE: scale adapter contribution by alpha."""
        adapter = self.adapters[layer_idx]
        residual = h9
        out = adapter.down(h9)
        out = adapter.act(out)
        out = adapter.up(out)
        return residual + alpha * adapter.scale_factor * out


class HoulAdapter(nn.Module):
    """Houlsby Adapter: insert adapters after MSA (h5) and MLP (h9)."""

    def __init__(self, embed_dim=768, bottle_neck=8, scale_factor=1.0, depth=12):
        super().__init__()
        self.msa_adapters = nn.ModuleList([
            AdapterModule(embed_dim, bottle_neck, scale_factor)
            for _ in range(depth)
        ])
        self.mlp_adapters = nn.ModuleList([
            AdapterModule(embed_dim, bottle_neck, scale_factor)
            for _ in range(depth)
        ])

    def forward_msa(self, h5, layer_idx):
        return self.msa_adapters[layer_idx](h5)

    def forward_mlp(self, h9, layer_idx):
        return self.mlp_adapters[layer_idx](h9)


class AdaptFormer(nn.Module):
    """Parallel Adapter after MLP: h9 = h9 + Adapter(h7)."""

    def __init__(self, embed_dim=768, bottle_neck=8, scale_factor=0.1, depth=12):
        super().__init__()
        self.adapters = nn.ModuleList([
            AdapterModule(embed_dim, bottle_neck, scale_factor)
            for _ in range(depth)
        ])

    def forward(self, h7, layer_idx):
        """Apply adapter to h7, return additive term for h9."""
        adapter = self.adapters[layer_idx]
        out = adapter.down(h7)
        out = adapter.act(out)
        out = adapter.up(out)
        return adapter.scale_factor * out


class RepAdapter(nn.Module):
    """RepAdapter: linear adapter with group-wise transformation, sequential placement.

    Placed after MSA and MLP: h5 = RepAdapter1(h2), h7 = RepAdapter2(h7).
    Linear (no activation) -> reparameterizable.
    """

    def __init__(self, embed_dim=768, bottle_neck=16, scale_factor=1.0, num_groups=4, depth=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottle_neck = bottle_neck
        self.num_groups = num_groups
        self.scale_factor = scale_factor
        assert bottle_neck % num_groups == 0
        group_dim_in = bottle_neck // num_groups
        group_dim_out = embed_dim // num_groups

        self.down_msa = nn.ModuleList([nn.Linear(embed_dim, bottle_neck) for _ in range(depth)])
        self.down_mlp = nn.ModuleList([nn.Linear(embed_dim, bottle_neck) for _ in range(depth)])
        self.up_msa = nn.ModuleList([
            nn.ModuleList([nn.Linear(group_dim_in, group_dim_out) for _ in range(num_groups)])
            for _ in range(depth)
        ])
        self.up_mlp = nn.ModuleList([
            nn.ModuleList([nn.Linear(group_dim_in, group_dim_out) for _ in range(num_groups)])
            for _ in range(depth)
        ])
        self._init_weights()

    def _init_weights(self):
        for mod_list in [self.down_msa, self.down_mlp]:
            for m in mod_list:
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        for up_list in [self.up_msa, self.up_mlp]:
            for groups in up_list:
                for m in groups:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)

    def _group_forward(self, x, down, up_groups, layer_idx):
        B, N, C = x.shape
        gs = C // self.num_groups
        h = down(x)  # B, N, r
        # Split into groups
        h_chunks = torch.chunk(h, self.num_groups, dim=-1)
        out_chunks = []
        for i, chunk in enumerate(h_chunks):
            o = up_groups[i](chunk)
            out_chunks.append(o)
        out = torch.cat(out_chunks, dim=-1)
        return x + self.scale_factor * out

    def forward_msa(self, h2, layer_idx):
        return self._group_forward(h2, self.down_msa[layer_idx], self.up_msa[layer_idx])

    def forward_mlp(self, h7, layer_idx):
        return self._group_forward(h7, self.down_mlp[layer_idx], self.up_mlp[layer_idx])


class Convpass(nn.Module):
    """Convolutional Adapter: parallel bypass with Conv2D for visual inductive bias.

    Placed parallel to MSA and MLP: h5 = Convpass1(h2) + h5, h9 = Convpass2(h7) + h9.
    """

    def __init__(self, embed_dim=768, bottle_neck=8, scale_factor=1.0, kernel_size=3, depth=12):
        super().__init__()
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim

        self.down_msa = nn.ModuleList([nn.Conv2d(embed_dim, bottle_neck, 1) for _ in range(depth)])
        self.conv_msa = nn.ModuleList([nn.Conv2d(bottle_neck, bottle_neck, kernel_size, padding=kernel_size // 2) for _ in range(depth)])
        self.up_msa = nn.ModuleList([nn.Conv2d(bottle_neck, embed_dim, 1) for _ in range(depth)])

        self.down_mlp = nn.ModuleList([nn.Conv2d(embed_dim, bottle_neck, 1) for _ in range(depth)])
        self.conv_mlp = nn.ModuleList([nn.Conv2d(bottle_neck, bottle_neck, kernel_size, padding=kernel_size // 2) for _ in range(depth)])
        self.up_mlp = nn.ModuleList([nn.Conv2d(bottle_neck, embed_dim, 1) for _ in range(depth)])

        self.act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for mod_list in [self.down_msa, self.conv_msa, self.up_msa,
                         self.down_mlp, self.conv_mlp, self.up_mlp]:
            for m in mod_list:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _conv_forward(self, x, down, conv, up, layer_idx):
        B, N, C = x.shape
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        H = W = int(math.sqrt(patch_tokens.size(1)))
        patches = patch_tokens.transpose(1, 2).reshape(B, C, H, W)

        out = down(patches)
        out = self.act(out)
        out = conv(out)
        out = self.act(out)
        out = up(out)

        out = out.reshape(B, C, -1).transpose(1, 2)
        cls_out = torch.zeros_like(cls_token)
        out = torch.cat([cls_out, out], dim=1)
        return self.scale_factor * out

    def forward_msa(self, h2, layer_idx):
        return self._conv_forward(h2, self.down_msa[layer_idx],
                                  self.conv_msa[layer_idx], self.up_msa[layer_idx])

    def forward_mlp(self, h7, layer_idx):
        return self._conv_forward(h7, self.down_mlp[layer_idx],
                                  self.conv_mlp[layer_idx], self.up_mlp[layer_idx])


# ==============================================================================
# Direct Selective Tuning
# ==============================================================================

def enable_bitfit(model):
    """BitFit: Only train bias terms. Freeze all other parameters."""
    trainable_params = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    return trainable_params


def enable_layernorm_tuning(model):
    """LayerNorm Tuning: Only train LN (weight + bias) parameters."""
    trainable_params = []
    for name, param in model.named_parameters():
        if 'norm' in name.lower():
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    return trainable_params


class DiffFitScales(nn.Module):
    """DiffFit: BitFit + LayerNorm + learnable scale factors after MSA and MLP."""

    def __init__(self, embed_dim=768, depth=12):
        super().__init__()
        self.gamma_msa = nn.ParameterList([
            nn.Parameter(torch.ones(embed_dim)) for _ in range(depth)
        ])
        self.gamma_mlp = nn.ParameterList([
            nn.Parameter(torch.ones(embed_dim)) for _ in range(depth)
        ])

    def forward_msa(self, h5, layer_idx):
        return h5 * self.gamma_msa[layer_idx]

    def forward_mlp(self, h9, layer_idx):
        return h9 * self.gamma_mlp[layer_idx]


class SSFModule(nn.Module):
    """SSF (Scale & Shift Features): linear modulation of intermediate features.

    Modulates h2, h3, h5, h7, h8, h9 per layer.
    """

    def __init__(self, embed_dim=768, depth=12, mlp_hidden_dim=None):
        super().__init__()
        if mlp_hidden_dim is None:
            mlp_hidden_dim = embed_dim * 4

        self.scale_h2 = nn.ParameterList([nn.Parameter(torch.ones(embed_dim)) for _ in range(depth)])
        self.shift_h2 = nn.ParameterList([nn.Parameter(torch.zeros(embed_dim)) for _ in range(depth)])

        self.scale_h3 = nn.ParameterList([nn.Parameter(torch.ones(embed_dim * 3)) for _ in range(depth)])
        self.shift_h3 = nn.ParameterList([nn.Parameter(torch.zeros(embed_dim * 3)) for _ in range(depth)])

        self.scale_h5 = nn.ParameterList([nn.Parameter(torch.ones(embed_dim)) for _ in range(depth)])
        self.shift_h5 = nn.ParameterList([nn.Parameter(torch.zeros(embed_dim)) for _ in range(depth)])

        self.scale_h7 = nn.ParameterList([nn.Parameter(torch.ones(embed_dim)) for _ in range(depth)])
        self.shift_h7 = nn.ParameterList([nn.Parameter(torch.zeros(embed_dim)) for _ in range(depth)])

        self.scale_h8 = nn.ParameterList([nn.Parameter(torch.ones(mlp_hidden_dim)) for _ in range(depth)])
        self.shift_h8 = nn.ParameterList([nn.Parameter(torch.zeros(mlp_hidden_dim)) for _ in range(depth)])

        self.scale_h9 = nn.ParameterList([nn.Parameter(torch.ones(embed_dim)) for _ in range(depth)])
        self.shift_h9 = nn.ParameterList([nn.Parameter(torch.zeros(embed_dim)) for _ in range(depth)])

    def modulate(self, x, scale, shift, idx):
        return x * scale[idx] + shift[idx]


# ==============================================================================
# Efficient Selective Tuning (Low-rank)
# ==============================================================================

class LoRAModule(nn.Module):
    """LoRA: Low-Rank Adaptation applied to Q and V projection weights.

    h3 = [W_Q*x, W_K*x, W_V*x]
    Delta applied: [W_down_Q @ W_up_Q @ x, 0, W_down_V @ W_up_V @ x]
    """

    def __init__(self, embed_dim=768, rank=8, depth=12):
        super().__init__()
        self.rank = rank
        self.down_Q = nn.ParameterList([nn.Parameter(torch.zeros(rank, embed_dim)) for _ in range(depth)])
        self.up_Q = nn.ParameterList([nn.Parameter(torch.zeros(embed_dim, rank)) for _ in range(depth)])
        self.down_V = nn.ParameterList([nn.Parameter(torch.zeros(rank, embed_dim)) for _ in range(depth)])
        self.up_V = nn.ParameterList([nn.Parameter(torch.zeros(embed_dim, rank)) for _ in range(depth)])
        self._init_weights()

    def _init_weights(self):
        for i in range(len(self.down_Q)):
            nn.init.kaiming_uniform_(self.down_Q[i], a=math.sqrt(5))
            nn.init.zeros_(self.up_Q[i])
            nn.init.kaiming_uniform_(self.down_V[i], a=math.sqrt(5))
            nn.init.zeros_(self.up_V[i])

    def forward(self, h2, layer_idx):
        B, N, C = h2.shape
        delta_Q = h2 @ self.down_Q[layer_idx].T @ self.up_Q[layer_idx].T
        delta_V = h2 @ self.down_V[layer_idx].T @ self.up_V[layer_idx].T
        return delta_Q, delta_V  # additive residuals for Q and V

    def apply_wise(self, h2, layer_idx, alpha):
        delta_Q = alpha * (h2 @ self.down_Q[layer_idx].T @ self.up_Q[layer_idx].T)
        delta_V = alpha * (h2 @ self.down_V[layer_idx].T @ self.up_V[layer_idx].T)
        return delta_Q, delta_V


class FacT_TT(nn.Module):
    """FacT with Tensor-Train decomposition.

    Stacks all 12L weight matrices into a tensor and learns additive residual
    via TT decomposition: Delta = s * Sigma x_2 U^T x_3 V^T.
    Applied to W_Q, W_K, W_V, W_O, W_1, W_2 across all layers.
    """

    def __init__(self, embed_dim=768, depth=12, bottle_neck=16, scale_factor=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.mlp_hidden = embed_dim * 4
        self.bottle_neck = bottle_neck
        self.scale_factor = scale_factor

        # 4 matrices from MSA + 2 from MLP = 6 per layer, total 12L slices
        self.num_slices = depth * 6
        # TT cores: U (D x r), V (D x r), Sigma (12L x r x r)
        self.U = nn.Parameter(torch.randn(embed_dim, bottle_neck) * 0.02)
        self.V = nn.Parameter(torch.randn(embed_dim, bottle_neck) * 0.02)
        self.Sigma = nn.Parameter(torch.randn(self.num_slices, bottle_neck, bottle_neck) * 0.02)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.U, std=0.02)
        nn.init.trunc_normal_(self.V, std=0.02)
        nn.init.trunc_normal_(self.Sigma, std=0.02)

    def get_delta(self, layer_idx, matrix_type):
        """matrix_type: 'Q', 'K', 'V', 'O', 'W1', 'W2'.

        Returns delta of shape matching the target weight matrix.
        For Q/K/V/O: (D, D); for W1: (4D, D); for W2: (D, 4D).
        """
        type_offset = {'Q': 0, 'K': 1, 'V': 2, 'O': 3, 'W1': 4, 'W2': 5}
        slice_idx = layer_idx * 6 + type_offset[matrix_type]
        sigma_slice = self.Sigma[slice_idx]  # (r, r)

        if matrix_type == 'W1':
            delta = self.scale_factor * (torch.cat([self.U] * 4, dim=0) @ sigma_slice @ self.V.T)
        elif matrix_type == 'W2':
            delta = self.scale_factor * (self.U @ sigma_slice @ torch.cat([self.V] * 4, dim=0).T)
        else:
            delta = self.scale_factor * (self.U @ sigma_slice @ self.V.T)
        return delta


class FacT_TK(nn.Module):
    """FacT with Tucker decomposition.

    Delta = s * A x_1 B^T x_2 U^T x_3 V^T.
    B (12L x r), A (r x r x r), U (D x r), V (D x r).
    """

    def __init__(self, embed_dim=768, depth=12, bottle_neck=16, scale_factor=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.bottle_neck = bottle_neck
        self.scale_factor = scale_factor
        self.num_slices = depth * 6

        self.U = nn.Parameter(torch.randn(embed_dim, bottle_neck) * 0.02)
        self.V = nn.Parameter(torch.randn(embed_dim, bottle_neck) * 0.02)
        self.B = nn.Parameter(torch.randn(self.num_slices, bottle_neck) * 0.02)
        self.A = nn.Parameter(torch.randn(bottle_neck, bottle_neck, bottle_neck) * 0.02)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.U, std=0.02)
        nn.init.trunc_normal_(self.V, std=0.02)
        nn.init.trunc_normal_(self.B, std=0.02)
        nn.init.trunc_normal_(self.A, std=0.02)

    def get_delta(self, layer_idx, matrix_type):
        """matrix_type: 'Q', 'K', 'V', 'O', 'W1', 'W2'.

        Returns delta of shape matching the target weight matrix.
        For Q/K/V/O: (D, D); for W1: (4D, D); for W2: (D, 4D).
        """
        type_offset = {'Q': 0, 'K': 1, 'V': 2, 'O': 3, 'W1': 4, 'W2': 5}
        slice_idx = layer_idx * 6 + type_offset[matrix_type]
        b_slice = self.B[slice_idx]  # (r,)
        A_contracted = torch.einsum('r,rij->ij', b_slice, self.A)

        if matrix_type == 'W1':
            delta = self.scale_factor * (torch.cat([self.U] * 4, dim=0) @ A_contracted @ self.V.T)
        elif matrix_type == 'W2':
            delta = self.scale_factor * (self.U @ A_contracted @ torch.cat([self.V] * 4, dim=0).T)
        else:
            delta = self.scale_factor * (self.U @ A_contracted @ self.V.T)
        return delta


# ==============================================================================
# Count trainable parameters
# ==============================================================================

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
