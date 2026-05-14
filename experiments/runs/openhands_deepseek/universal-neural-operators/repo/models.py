"""Neural operator architectures: baseline FNO, MambaFNO, PerceiverIOFNO,
CoDANO, and SwinV2FNO.

All models follow the lifting - operator blocks - projection architecture
described in Section 3, with adapters (lift/proj) serving as problem-specific
mappings that can be trained separately during fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from layers import (
    FNOBlock1d, FNOBlock2d, SpectralConv1d, SpectralConv2d,
    MambaSSM, CodomainAttention, PerceiverIOBlock,
    SwinV2Stage, LocalAttentionFNO,
    Lift, Project,
)


# ---------------------------------------------------------------------------
#  Baseline FNO
# ---------------------------------------------------------------------------

class FNO(nn.Module):
    """Baseline Fourier Neural Operator (Section 3)."""

    def __init__(self, config):
        super().__init__()
        self.ndim = config.get('ndim', 1)
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        hidden_channels = config['hidden_channels']
        n_layers = config.get('n_layers', 4)
        modes = config.get('modes', 12)
        modes1 = config.get('modes1', 12)
        modes2 = config.get('modes2', 12)
        lift_mode = config.get('lift_mode', 'mlp')
        resolution = config.get('resolution', None)

        self.lift = Lift(in_channels, hidden_channels, mode=lift_mode, resolution=resolution)
        self.project = Project(hidden_channels, out_channels, mode=lift_mode, resolution=resolution)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            if self.ndim == 1:
                self.blocks.append(FNOBlock1d(hidden_channels, modes))
            else:
                self.blocks.append(FNOBlock2d(hidden_channels, modes1, modes2))

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = self.project(x)
        return x


# ---------------------------------------------------------------------------
#  MambaFNO  –  Post-lifting Mamba SSM + FNO blocks (Section 3)
# ---------------------------------------------------------------------------

class MambaFNO(nn.Module):
    """
    MambaFNO: lift -> Mamba SSM -> FNO blocks -> project.

    The Mamba module (post-lifting) encodes long-range spatio-temporal
    dependencies, acting as a latent preconditioner for the Fourier layers.
    """

    def __init__(self, config):
        super().__init__()
        self.ndim = config.get('ndim', 1)
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        hidden_channels = config['hidden_channels']
        n_layers = config.get('n_layers', 4)
        modes = config.get('modes', 12)
        modes1 = config.get('modes1', 12)
        modes2 = config.get('modes2', 12)
        lift_mode = config.get('lift_mode', 'mlp')
        resolution = config.get('resolution', None)
        d_state = config.get('mamba_d_state', 16)
        d_conv = config.get('mamba_d_conv', 4)
        expand = config.get('mamba_expand', 2)

        self.lift = Lift(in_channels, hidden_channels, mode=lift_mode, resolution=resolution)
        self.mamba = MambaSSM(hidden_channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.project = Project(hidden_channels, out_channels, mode=lift_mode, resolution=resolution)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            if self.ndim == 1:
                self.blocks.append(FNOBlock1d(hidden_channels, modes))
            else:
                self.blocks.append(FNOBlock2d(hidden_channels, modes1, modes2))

    def forward(self, x):
        # lift
        if self.lift.mode in ('conv1d', 'conv2d'):
            x = self.lift(x)
            x = rearrange(x, 'b c ... -> b (...) c')
        else:
            x = self.lift(x)

        # Mamba SSM post-lifting
        x = self.mamba(x)

        # FNO blocks
        if self.ndim == 1:
            x = rearrange(x, 'b n c -> b c n')
        else:
            n = int(x.shape[1] ** 0.5)
            x = rearrange(x, 'b (h w) c -> b c h w', h=n, w=n)

        for block in self.blocks:
            x = block(x)

        # project
        if self.project.mode in ('conv1d', 'conv2d'):
            x = self.project(x)
        else:
            x = rearrange(x, 'b c ... -> b (...) c')
            x = self.project(x)
        return x


# ---------------------------------------------------------------------------
#  LocalAttnFNO  –  Post-lifting local attention + FNO blocks
# ---------------------------------------------------------------------------

class LocalAttnFNO(nn.Module):
    """Local attention applied post-lifting, followed by FNO blocks."""

    def __init__(self, config):
        super().__init__()
        self.ndim = config.get('ndim', 1)
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        hidden_channels = config['hidden_channels']
        n_layers = config.get('n_layers', 4)
        modes = config.get('modes', 12)
        modes1 = config.get('modes1', 12)
        modes2 = config.get('modes2', 12)
        lift_mode = config.get('lift_mode', 'mlp')
        resolution = config.get('resolution', None)
        window_size = config.get('local_attn_window', 16)

        self.lift = Lift(in_channels, hidden_channels, mode=lift_mode, resolution=resolution)
        self.local_attn = LocalAttentionFNO(hidden_channels, window_size=window_size)
        self.project = Project(hidden_channels, out_channels, mode=lift_mode, resolution=resolution)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            if self.ndim == 1:
                self.blocks.append(FNOBlock1d(hidden_channels, modes))
            else:
                self.blocks.append(FNOBlock2d(hidden_channels, modes1, modes2))

    def forward(self, x):
        if self.lift.mode in ('conv1d', 'conv2d'):
            x = self.lift(x)
            x = rearrange(x, 'b c ... -> b (...) c')
        else:
            x = self.lift(x)

        x = self.local_attn(x)

        if self.ndim == 1:
            x = rearrange(x, 'b n c -> b c n')
        else:
            n = int(x.shape[1] ** 0.5)
            x = rearrange(x, 'b (h w) c -> b c h w', h=n, w=n)

        for block in self.blocks:
            x = block(x)

        if self.project.mode in ('conv1d', 'conv2d'):
            x = self.project(x)
        else:
            x = rearrange(x, 'b c ... -> b (...) c')
            x = self.project(x)
        return x


# ---------------------------------------------------------------------------
#  PerceiverIOFNO  –  Perceiver IO-based neural operator (Section 3)
# ---------------------------------------------------------------------------

class PerceiverIOFNO(nn.Module):
    """
    Perceiver IO Neural Operator:
    lift -> Perceiver IO blocks -> project.

    Uses symmetrical cross-attention for input encoding and output decoding
    with latent arrays, as described in Section 3.
    """

    def __init__(self, config):
        super().__init__()
        self.ndim = config.get('ndim', 1)
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        hidden_channels = config['hidden_channels']
        n_layers = config.get('n_layers', 4)
        num_latents = config.get('num_latents', 128)
        num_heads = config.get('num_heads', 8)
        fno_modes = config.get('modes', 12)
        lift_mode = config.get('lift_mode', 'mlp')
        resolution = config.get('resolution', None)

        self.lift = Lift(in_channels, hidden_channels, mode=lift_mode, resolution=resolution)
        self.project = Project(hidden_channels, out_channels, mode=lift_mode, resolution=resolution)

        self.perceiver_blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.perceiver_blocks.append(
                PerceiverIOBlock(
                    input_dim=hidden_channels,
                    latent_dim=hidden_channels,
                    num_latents=num_latents,
                    num_heads=num_heads,
                    fno_modes=fno_modes,
                    use_fno_proj=True,
                )
            )

    def forward(self, x):
        if self.lift.mode in ('conv1d', 'conv2d'):
            x = self.lift(x)
            x = rearrange(x, 'b c ... -> b (...) c')
        else:
            x = self.lift(x)

        for block in self.perceiver_blocks:
            x = block(x)

        if self.project.mode in ('conv1d', 'conv2d'):
            x = rearrange(x, 'b n c -> b c n') if self.ndim == 1 else \
                rearrange(x, 'b (h w) c -> b c h w', h=int(x.shape[1] ** 0.5))
            x = self.project(x)
        else:
            x = self.project(x)
        return x


# ---------------------------------------------------------------------------
#  CoDANO  –  Codomain Attention Neural Operator (Section 3)
# ---------------------------------------------------------------------------

class CoDANO(nn.Module):
    """
    Codomain Attention Neural Operator (Rahman et al.):
    Attention computed in the codomain (feature space) using FNO-based
    projections for queries, keys, and values.
    """

    def __init__(self, config):
        super().__init__()
        self.ndim = config.get('ndim', 1)
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        hidden_channels = config['hidden_channels']
        n_layers = config.get('n_layers', 4)
        num_heads = config.get('num_heads', 8)
        fno_modes = config.get('modes', 12)
        lift_mode = config.get('lift_mode', 'mlp')
        resolution = config.get('resolution', None)

        self.lift = Lift(in_channels, hidden_channels, mode=lift_mode, resolution=resolution)
        self.project = Project(hidden_channels, out_channels, mode=lift_mode, resolution=resolution)

        self.codoma_blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.codoma_blocks.append(
                CodomainAttention(hidden_channels, num_heads=num_heads, use_fno_proj=True)
            )
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(n_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 4),
                nn.GELU(),
                nn.Linear(hidden_channels * 4, hidden_channels),
            ) for _ in range(n_layers)
        ])

    def _to_seq(self, x):
        if self.ndim == 1:
            return rearrange(x, 'b c n -> b n c')
        else:
            return rearrange(x, 'b c h w -> b (h w) c')

    def _from_seq(self, x):
        if self.ndim == 1:
            return rearrange(x, 'b n c -> b c n')
        else:
            n = int(x.shape[1] ** 0.5)
            return rearrange(x, 'b (h w) c -> b c h w', h=n, w=n)

    def forward(self, x):
        if self.lift.mode in ('conv1d', 'conv2d'):
            x = self.lift(x)
            x = self._to_seq(x)
        else:
            x = self.lift(x)

        for i, attn in enumerate(self.codoma_blocks):
            shortcut = x
            x = self.norms[i](x)
            x = attn(x) + shortcut
            x = self.ffns[i](x) + x

        if self.project.mode in ('conv1d', 'conv2d'):
            x = self._from_seq(x)
            x = self.project(x)
        else:
            x = self.project(x)
        return x


# ---------------------------------------------------------------------------
#  SwinV2FNO  –  Swin Transformer V2 + FNO (Hierarchical vision transformer)
# ---------------------------------------------------------------------------

class SwinV2FNO(nn.Module):
    """
    Swin Transformer V2 based neural operator.
    Uses hierarchical Swin-V2 stages to process spatial features,
    based on POSEIDON approach described in Section 2/3.
    """

    def __init__(self, config):
        super().__init__()
        self.ndim = config.get('ndim', 1)
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        hidden_channels = config['hidden_channels']
        n_layers = config.get('n_layers', 4)
        resolution = config.get('resolution', 64)
        window_size = config.get('swin_window_size', 8)
        num_heads = config.get('num_heads', 8)
        lift_mode = config.get('lift_mode', 'conv2d')

        self.lift = Lift(in_channels, hidden_channels, mode=lift_mode, resolution=resolution)
        self.project = Project(hidden_channels, out_channels, mode=lift_mode, resolution=resolution)

        self.stage = SwinV2Stage(
            dim=hidden_channels,
            input_resolution=resolution,
            depth=n_layers,
            window_size=window_size,
            num_heads=num_heads,
        )

    def forward(self, x):
        x = self.lift(x)  # (B, C, H, W)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.stage(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.project(x)
        return x


# ---------------------------------------------------------------------------
#  Multi-physics wrapper with adapters (Section 3, Figure 1)
# ---------------------------------------------------------------------------

class MultiPhysicsNO(nn.Module):
    """
    Multi-physics neural operator with problem-specific adapters.

    Each physics problem gets its own lift/project adapters, sharing the
    core operator body (theta_F). During pre-training all parameters are
    optimized; during fine-tuning only the new adapter parameters are trained.

    Architecture: lift_i -> core_body -> proj_i    for problem i
    """

    def __init__(self, core_config, problem_configs):
        super().__init__()
        self.problem_configs = problem_configs
        self.hidden_channels = core_config['hidden_channels']
        self.ndim = core_config.get('ndim', 1)
        self.model_type = core_config.get('model_type', 'fno')

        self.core_body = self._build_core(core_config)

        self.lifts = nn.ModuleDict()
        self.projects = nn.ModuleDict()
        for pcfg in problem_configs:
            name = pcfg['name']
            lift_mode = pcfg.get('lift_mode', 'mlp')
            self.lifts[name] = Lift(
                pcfg['in_channels'], self.hidden_channels, mode=lift_mode,
            )
            self.projects[name] = Project(
                self.hidden_channels, pcfg['out_channels'], mode=lift_mode,
            )

    def _build_core(self, config):
        model_type = config.get('model_type', 'fno')
        ndim = config.get('ndim', 1)
        hidden = config['hidden_channels']
        n_layers = config.get('n_layers', 4)
        modes = config.get('modes', 12)

        self.mamba_module = None
        self.local_attn_module = None

        if model_type in ('fno', 'mamba_fno', 'local_attn_fno'):
            if model_type == 'mamba_fno':
                self.mamba_module = MambaSSM(
                    hidden,
                    d_state=config.get('mamba_d_state', 16),
                    d_conv=config.get('mamba_d_conv', 4),
                    expand=config.get('mamba_expand', 2),
                )
            elif model_type == 'local_attn_fno':
                self.local_attn_module = LocalAttentionFNO(
                    hidden, window_size=config.get('local_attn_window', 16)
                )
            blocks = nn.ModuleList()
            for _ in range(n_layers):
                if ndim == 1:
                    blocks.append(FNOBlock1d(hidden, modes))
                else:
                    blocks.append(FNOBlock2d(hidden, config.get('modes1', 12), config.get('modes2', 12)))
            return blocks
        elif model_type == 'codano':
            blocks = nn.ModuleList()
            for _ in range(n_layers):
                blocks.append(CodomainAttention(hidden, num_heads=config.get('num_heads', 8)))
            return blocks
        elif model_type == 'perceiver_io_fno':
            blocks = nn.ModuleList()
            for _ in range(n_layers):
                blocks.append(PerceiverIOBlock(
                    input_dim=hidden, latent_dim=hidden,
                    num_latents=config.get('num_latents', 128),
                    num_heads=config.get('num_heads', 8),
                    fno_modes=modes,
                ))
            return blocks
        else:
            blocks = nn.ModuleList()
            for _ in range(n_layers):
                if ndim == 1:
                    blocks.append(FNOBlock1d(hidden, modes))
                else:
                    blocks.append(FNOBlock2d(hidden, config.get('modes1', 12), config.get('modes2', 12)))
            return blocks

    def forward(self, x, problem_name):
        lift = self.lifts[problem_name]
        proj = self.projects[problem_name]

        is_conv_input = lift.mode in ('conv1d', 'conv2d')

        if is_conv_input:
            x = lift(x)  # (B, C, N) or (B, C, H, W)
        else:
            x = lift(x)  # (B, N, hidden)

        # Apply mamba or local attention post-lifting if present
        if self.mamba_module is not None:
            if not is_conv_input:
                x = self.mamba_module(x)
            else:
                if self.ndim == 1:
                    x_seq = rearrange(x, 'b c n -> b n c')
                else:
                    x_seq = rearrange(x, 'b c h w -> b (h w) c')
                x_seq = self.mamba_module(x_seq)
                if self.ndim == 1:
                    x = rearrange(x_seq, 'b n c -> b c n')
                else:
                    n = int(x_seq.shape[1] ** 0.5)
                    x = rearrange(x_seq, 'b (h w) c -> b c h w', h=n, w=n)
        if self.local_attn_module is not None:
            if not is_conv_input:
                x = self.local_attn_module(x)
            else:
                if self.ndim == 1:
                    x_seq = rearrange(x, 'b c n -> b n c')
                else:
                    x_seq = rearrange(x, 'b c h w -> b (h w) c')
                x_seq = self.local_attn_module(x_seq)
                if self.ndim == 1:
                    x = rearrange(x_seq, 'b n c -> b c n')
                else:
                    n = int(x_seq.shape[1] ** 0.5)
                    x = rearrange(x_seq, 'b (h w) c -> b c h w', h=n, w=n)

        # Apply core body blocks — handle shape conversion per block type
        if not is_conv_input and self.ndim == 1:
            x = rearrange(x, 'b n c -> b c n')
        elif not is_conv_input and self.ndim == 2:
            n = int(x.shape[1] ** 0.5)
            x = rearrange(x, 'b (h w) c -> b c h w', h=n, w=n)

        for block in self.core_body:
            if isinstance(block, (CodomainAttention, PerceiverIOBlock)):
                if x.dim() == 4 and self.ndim == 2:
                    x = rearrange(x, 'b c h w -> b (h w) c')
                elif x.dim() == 3 and self.ndim == 1:
                    x = rearrange(x, 'b c n -> b n c')
                x = block(x)
            elif isinstance(block, FNOBlock1d):
                if x.dim() == 3 and x.shape[-2] > x.shape[-1]:
                    pass  # already (B, C, N)
                elif x.dim() == 3:
                    x = rearrange(x, 'b n c -> b c n')
                x = block(x)
            elif isinstance(block, FNOBlock2d):
                if x.dim() == 3:
                    n = int(x.shape[1] ** 0.5)
                    x = rearrange(x, 'b (h w) c -> b c h w', h=n, w=n)
                x = block(x)
            else:
                x = block(x)

        # Ensure correct format for projection
        if is_conv_input:
            if x.dim() == 3 and isinstance(proj, Project) and proj.mode in ('conv1d', 'conv2d'):
                pass
            elif x.dim() == 4 and isinstance(proj, Project) and proj.mode in ('conv1d', 'conv2d'):
                pass
            elif x.dim() == 3:
                x = rearrange(x, 'b n c -> b c n') if not (
                    isinstance(proj, Project) and proj.mode == 'mlp') else x
        else:
            if proj.mode == 'mlp':
                if x.dim() == 3 and x.shape[-2] > x.shape[-1]:
                    x = rearrange(x, 'b c n -> b n c')
                elif x.dim() == 4:
                    x = rearrange(x, 'b c h w -> b (h w) c')

        x = proj(x)
        return x

    def _all_core_params(self):
        """Iterator over all core parameters (blocks + mamba + local_attn)."""
        for p in self.core_body.parameters():
            yield p
        if self.mamba_module is not None:
            for p in self.mamba_module.parameters():
                yield p
        if self.local_attn_module is not None:
            for p in self.local_attn_module.parameters():
                yield p

    def freeze_core(self):
        """Freeze core body parameters for fine-tuning."""
        for p in self._all_core_params():
            p.requires_grad = False

    def unfreeze_core(self):
        for p in self._all_core_params():
            p.requires_grad = True


# ---------------------------------------------------------------------------
#  Factory function
# ---------------------------------------------------------------------------

def get_model(model_type, config):
    """Return the requested neural operator model."""
    model_map = {
        'fno': FNO,
        'mamba_fno': MambaFNO,
        'local_attn_fno': LocalAttnFNO,
        'perceiver_io_fno': PerceiverIOFNO,
        'codano': CoDANO,
        'swinv2_fno': SwinV2FNO,
    }
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
    return model_map[model_type](config)
