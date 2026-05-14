
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules import FNOBlock, MambaSSM, PerceiverIOBlock, CodomainAttention
from layers import MLP, PositionalEmbedding

class Adapter(nn.Module):
    """
    Problem-specific adapter for lifting or projection layers.
    These contain different cardinality input sets, projecting into the fixed number of hidden features.
    """
    def __init__(self, in_features, out_features, hidden_features, num_layers=2):
        super().__init__()
        self.net = MLP(in_features, out_features, hidden_features, num_layers)

    def forward(self, x):
        return self.net(x)

class NeuralOperator(nn.Module):
    """
    Base class for Neural Operators with a lifting-operator-projection architecture.
    The core idea is to map input functions to a higher-dimensional hidden representation (lifting),
    apply an operator (e.g., FNO, Mamba, Perceiver) in this hidden space,
    and then project back to the output function space.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, lifting_channels, projection_channels, num_layers,
                 num_physics_tasks=1, current_physics_idx=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.num_layers = num_layers
        self.num_physics_tasks = num_physics_tasks
        self.current_physics_idx = current_physics_idx

        # Lifting layers (adapters)
        self.lifting_adapters = nn.ModuleList([
            Adapter(in_channels, lifting_channels, hidden_channels)
            for _ in range(num_physics_tasks)
        ])

        # Core operator (to be defined by subclasses)
        self.operator_blocks = nn.ModuleList()

        # Projection layers (adapters)
        self.projection_adapters = nn.ModuleList([
            Adapter(lifting_channels, out_channels, hidden_channels)
            for _ in range(num_physics_tasks)
        ])

    def forward(self, x, physics_idx=None):
        if physics_idx is None:
            physics_idx = self.current_physics_idx

        # x: (batch, spatial_dim, in_channels)
        x = self.lifting_adapters[physics_idx](x) # (batch, spatial_dim, lifting_channels)

        # Rearrange for FNO-style processing (batch, channels, spatial_dim)
        x = rearrange(x, 'b s c -> b c s')

        # Apply operator blocks
        for block in self.operator_blocks:
            x = block(x)

        # Rearrange back for projection (batch, spatial_dim, channels)
        x = rearrange(x, 'b c s -> b s c')

        x = self.projection_adapters[physics_idx](x) # (batch, spatial_dim, out_channels)
        return x

class FNO(NeuralOperator):
    def __init__(self, in_channels, out_channels, hidden_channels, lifting_channels, projection_channels, num_layers, modes,
                 num_physics_tasks=1, current_physics_idx=0):
        super().__init__(in_channels, out_channels, hidden_channels, lifting_channels, projection_channels, num_layers,
                         num_physics_tasks, current_physics_idx)
        self.modes = modes

        self.operator_blocks = nn.ModuleList([
            FNOBlock(lifting_channels if i == 0 else hidden_channels,
                     hidden_channels,
                     modes)
            for i in range(num_layers)
        ])

        # Final MLP in the FNO usually projects from hidden_channels to hidden_channels before final projection layer
        self.fno_mlp_out = MLP(hidden_channels, lifting_channels, hidden_channels, 2)


    def forward(self, x, physics_idx=None):
        if physics_idx is None:
            physics_idx = self.current_physics_idx

        x = self.lifting_adapters[physics_idx](x) # (batch, spatial_dim, lifting_channels)
        x = rearrange(x, 'b s c -> b c s')

        for i, block in enumerate(self.operator_blocks):
            x = block(x)
            if i < len(self.operator_blocks) - 1:
                x = F.gelu(x) # Activation between FNO blocks

        x = rearrange(x, 'b c s -> b s c')
        x = self.fno_mlp_out(x) # Apply final MLP
        x = self.projection_adapters[physics_idx](x)
        return x


class MambaFNO(NeuralOperator):
    def __init__(self, in_channels, out_channels, hidden_channels, lifting_channels, projection_channels, num_layers, modes,
                 mamba_d_state, mamba_d_conv, mamba_expand,
                 num_physics_tasks=1, current_physics_idx=0):
        super().__init__(in_channels, out_channels, hidden_channels, lifting_channels, projection_channels, num_layers,
                         num_physics_tasks, current_physics_idx)
        self.modes = modes
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand

        # Mamba module after lifting, acts as a latent preconditioner
        self.mamba_preconditioner = MambaSSM(lifting_channels, mamba_d_state, mamba_d_conv, mamba_expand)

        self.operator_blocks = nn.ModuleList([
            FNOBlock(lifting_channels if i == 0 else hidden_channels,
                     hidden_channels,
                     modes)
            for i in range(num_layers)
        ])
        self.fno_mlp_out = MLP(hidden_channels, lifting_channels, hidden_channels, 2) # Final MLP in FNO

    def forward(self, x, physics_idx=None):
        if physics_idx is None:
            physics_idx = self.current_physics_idx

        x = self.lifting_adapters[physics_idx](x) # (batch, spatial_dim, lifting_channels)

        # Apply Mamba preconditioner
        x = self.mamba_preconditioner(x)

        x = rearrange(x, 'b s c -> b c s') # For FNO blocks

        for i, block in enumerate(self.operator_blocks):
            x = block(x)
            if i < len(self.operator_blocks) - 1:
                x = F.gelu(x)

        x = rearrange(x, 'b c s -> b s c')
        x = self.fno_mlp_out(x)
        x = self.projection_adapters[physics_idx](x)
        return x

class PerceiverIONO(NeuralOperator):
    def __init__(self, in_channels, out_channels, hidden_channels, lifting_channels, projection_channels, num_layers,
                 num_latents, latent_dim, num_cross_attention_heads, num_self_attention_heads, num_perceiver_blocks,
                 num_physics_tasks=1, current_physics_idx=0):
        super().__init__(in_channels, out_channels, hidden_channels, lifting_channels, projection_channels, num_layers,
                         num_physics_tasks, current_physics_idx)

        self.num_latents = num_latents
        self.latent_dim = latent_dim

        # Latent array, shared across batch
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Perceiver IO blocks
        self.perceiver_blocks = nn.ModuleList([
            PerceiverIOBlock(lifting_channels, latent_dim, num_cross_attention_heads, latent_dim // num_cross_attention_heads, num_latents)
            for _ in range(num_perceiver_blocks)
        ])

        # Output projection from latents back to spatial dimension
        # The paper describes "cross-attention, matching the queries from the inputs with the keys and values,
        # taken from the transformed latent representations." This suggests a final cross-attention from
        # latent space back to the spatial input query to generate output.
        self.final_cross_attn = CrossAttention(lifting_channels, latent_dim, num_cross_attention_heads,
                                               lifting_channels // num_cross_attention_heads)

    def forward(self, x, physics_idx=None):
        if physics_idx is None:
            physics_idx = self.current_physics_idx

        # (batch, spatial_dim, in_channels)
        input_features = self.lifting_adapters[physics_idx](x) # (batch, spatial_dim, lifting_channels)

        b, s, c = input_features.shape

        # Expand latents to batch size
        latents = self.latents.unsqueeze(0).repeat(b, 1, 1) # (batch, num_latents, latent_dim)

        for block in self.perceiver_blocks:
            latents = block(input_features, latents)

        # Final projection: cross-attention from input_features (as queries) to latents (keys/values)
        output_features = self.final_cross_attn(input_features, latents) # (batch, spatial_dim, lifting_channels)

        output = self.projection_adapters[physics_idx](output_features) # (batch, spatial_dim, out_channels)
        return output

class CoDANO(NeuralOperator):
    def __init__(self, in_channels, out_channels, hidden_channels, lifting_channels, projection_channels, num_layers,
                 codano_num_heads, modes,
                 num_physics_tasks=1, current_physics_idx=0):
        super().__init__(in_channels, out_channels, hidden_channels, lifting_channels, projection_channels, num_layers,
                         num_physics_tasks, current_physics_idx)
        self.codano_num_heads = codano_num_heads

        # Initial FNOBlock to get into a suitable representation
        self.initial_fno = FNOBlock(lifting_channels, hidden_channels, modes)

        # Sequence of CodomainAttention blocks interleaved with FNO-like operations or MLPs
        self.operator_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.operator_blocks.append(CodomainAttention(hidden_channels, hidden_channels, heads=codano_num_heads))
            self.operator_blocks.append(FNOBlock(hidden_channels, hidden_channels, modes)) # Paper doesn't specify how CoDA-NO integrates, but often with FNO.

    def forward(self, x, physics_idx=None):
        if physics_idx is None:
            physics_idx = self.current_physics_idx

        x = self.lifting_adapters[physics_idx](x) # (batch, spatial_dim, lifting_channels)
        x = rearrange(x, 'b s c -> b c s') # For FNO-like processing

        x = self.initial_fno(x)
        x = F.gelu(x)

        for block in self.operator_blocks:
            x = block(x)
            x = F.gelu(x) # Activation after each block

        x = rearrange(x, 'b c s -> b s c')
        x = self.projection_adapters[physics_idx](x)
        return x

# SwinV2NO: This is a placeholder as Swin-v2 is a vision transformer.
# Adapting it to an operator block requires more specific information on how it would process
# function data (e.g., how to form "patches" from a PDE solution field).
# For now, it will be a conceptual placeholder or potentially omitted if details aren't clarified in the paper.
class SwinV2NO(NeuralOperator):
    def __init__(self, in_channels, out_channels, hidden_channels, lifting_channels, projection_channels, num_layers,
                 swin_embed_dim, swin_depths, swin_num_heads, swin_window_size,
                 num_physics_tasks=1, current_physics_idx=0):
        super().__init__(in_channels, out_channels, hidden_channels, lifting_channels, projection_channels, num_layers,
                         num_physics_tasks, current_physics_idx)
        # Swin Transformer implementation would go here.
        # This would require careful consideration of how to adapt image-based patching
        # and windowing to function inputs (e.g., spatial discretizations of functions).
        # For a 1D PDE output, this would be highly non-trivial to adapt directly.
        # Given the paper's focus on 1D functions for FNO, a Swin-v2 based NO would likely
        # require reshaping the 1D spatial input into something 2D-like, or a 1D variant of Swin.
        print("SwinV2NO is a complex adaptation; its implementation details are beyond the scope of direct inference from the paper's abstract description. A placeholder will be used.")

        # Placeholder for Swin-like processing:
        self.placeholder_fno_blocks = nn.ModuleList([
            FNOBlock(lifting_channels if i == 0 else hidden_channels,
                     hidden_channels,
                     modes=8) # Using an arbitrary modes for now
            for i in range(num_layers)
        ])
        self.fno_mlp_out = MLP(hidden_channels, lifting_channels, hidden_channels, 2)

    def forward(self, x, physics_idx=None):
        if physics_idx is None:
            physics_idx = self.current_physics_idx

        x = self.lifting_adapters[physics_idx](x)
        x = rearrange(x, 'b s c -> b c s')

        for i, block in enumerate(self.placeholder_fno_blocks):
            x = block(x)
            if i < len(self.placeholder_fno_blocks) - 1:
                x = F.gelu(x)

        x = rearrange(x, 'b c s -> b s c')
        x = self.fno_mlp_out(x)
        x = self.projection_adapters[physics_idx](x)
        return x

def get_model(config):
    if config.model_type == 'FNO':
        model = FNO(
            in_channels=config.input_channels, # Needs to be defined in config
            out_channels=config.output_channels, # Needs to be defined in config
            hidden_channels=config.hidden_channels,
            lifting_channels=config.lifting_channels,
            projection_channels=config.projection_channels,
            num_layers=config.num_layers,
            modes=config.modes,
            num_physics_tasks=config.num_physics_tasks,
            current_physics_idx=config.current_physics_idx
        )
    elif config.model_type == 'MambaFNO':
        model = MambaFNO(
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            hidden_channels=config.hidden_channels,
            lifting_channels=config.lifting_channels,
            projection_channels=config.projection_channels,
            num_layers=config.num_layers,
            modes=config.modes,
            mamba_d_state=config.mamba_d_state,
            mamba_d_conv=config.mamba_d_conv,
            mamba_expand=config.mamba_expand,
            num_physics_tasks=config.num_physics_tasks,
            current_physics_idx=config.current_physics_idx
        )
    elif config.model_type == 'PerceiverIONO':
        model = PerceiverIONO(
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            hidden_channels=config.hidden_channels,
            lifting_channels=config.lifting_channels,
            projection_channels=config.projection_channels,
            num_layers=config.num_layers,
            num_latents=config.num_latents,
            latent_dim=config.latent_dim,
            num_cross_attention_heads=config.num_cross_attention_heads,
            num_self_attention_heads=config.num_self_attention_heads,
            num_perceiver_blocks=config.num_perceiver_blocks,
            num_physics_tasks=config.num_physics_tasks,
            current_physics_idx=config.current_physics_idx
        )
    elif config.model_type == 'CoDANO':
        model = CoDANO(
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            hidden_channels=config.hidden_channels,
            lifting_channels=config.lifting_channels,
            projection_channels=config.projection_channels,
            num_layers=config.num_layers,
            codano_num_heads=config.codano_num_heads,
            modes=config.modes, # CoDANO integrates with FNO, so modes are relevant
            num_physics_tasks=config.num_physics_tasks,
            current_physics_idx=config.current_physics_idx
        )
    elif config.model_type == 'SwinV2NO':
        model = SwinV2NO(
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            hidden_channels=config.hidden_channels,
            lifting_channels=config.lifting_channels,
            projection_channels=config.projection_channels,
            num_layers=config.num_layers,
            swin_embed_dim=config.swin_embed_dim,
            swin_depths=config.swin_depths,
            swin_num_heads=config.swin_num_heads,
            swin_window_size=config.swin_window_size,
            num_physics_tasks=config.num_physics_tasks,
            current_physics_idx=config.current_physics_idx
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    return model

