import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make q, k, v broadcastable with query_points

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.query_proj = nn.Linear(query_dim, dim, bias=qkv_bias)
        self.key_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.value_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, context):
        B_q, N_q, C_q = query.shape
        B_c, N_c, C_c = context.shape

        q = self.query_proj(query).reshape(B_q, N_q, self.num_heads, -1).permute(0, 2, 1, 3) # (B, num_heads, N_q, head_dim)
        k = self.key_proj(context).reshape(B_c, N_c, self.num_heads, -1).permute(0, 2, 1, 3) # (B, num_heads, N_c, head_dim)
        v = self.value_proj(context).reshape(B_c, N_c, self.num_heads, -1).permute(0, 2, 1, 3) # (B, num_heads, N_c, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, -1) # (B, N_q, C_q)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PerceiverIOModule(nn.Module):
    def __init__(self, query_dim, context_dim, latent_dim, num_latent_tokens, num_heads=8, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.latent_tokens = nn.Parameter(torch.randn(num_latent_tokens, latent_dim))

        # Cross-attention from input to latent
        self.cross_attn1 = CrossAttention(
            query_dim=latent_dim, context_dim=context_dim, dim=latent_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, proj_drop=drop_rate
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.mlp1 = MLP(latent_dim, int(latent_dim * mlp_ratio), drop=drop_rate)
        self.norm2 = nn.LayerNorm(latent_dim)

        # Self-attention on latent
        self.self_attn = SelfAttention(
            dim=latent_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, proj_drop=drop_rate
        )
        self.norm3 = nn.LayerNorm(latent_dim)
        self.mlp2 = MLP(latent_dim, int(latent_dim * mlp_ratio), drop=drop_rate)
        self.norm4 = nn.LayerNorm(latent_dim)

        # Cross-attention from latent to output
        self.cross_attn2 = CrossAttention(
            query_dim=query_dim, context_dim=latent_dim, dim=latent_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, proj_drop=drop_rate
        )
        self.norm5 = nn.LayerNorm(query_dim) # This normalization should be applied to the output query dimension
        self.mlp3 = MLP(query_dim, int(query_dim * mlp_ratio), drop=drop_rate)
        self.norm6 = nn.LayerNorm(query_dim)


    def forward(self, x_input): # x_input: (B, N_input, C_input)
        # Repeat latent tokens for the batch
        latent = self.latent_tokens.unsqueeze(0).repeat(x_input.shape[0], 1, 1) # (B, num_latent_tokens, latent_dim)

        # Cross-attention from input to latent
        latent = latent + self.cross_attn1(self.norm1(latent), x_input)
        latent = latent + self.mlp1(self.norm2(latent))

        # Self-attention on latent
        latent = latent + self.self_attn(self.norm3(latent))
        latent = latent + self.mlp2(self.norm4(latent))

        # Cross-attention from latent to output
        # For the final cross-attention, we want to project the latent representation back to the input space
        # using the original input as queries. This mirrors the symmetric cross-attention mentioned.
        output = x_input + self.cross_attn2(self.norm5(x_input), latent)
        output = output + self.mlp3(self.norm6(output))

        return output

class PerceiverFNO1d(nn.Module):
    def __init__(self, modes, width, in_channels, out_channels, num_fno_layers, 
                 latent_dim=64, num_latent_tokens=16, num_heads=8, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super(PerceiverFNO1d, self).__init__()
        self.modes = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fno_layers = num_fno_layers

        self.padding = 9 # pad the domain if input is non-periodic

        # Lifting layer
        self.p = nn.Linear(in_channels, self.width)
        
        # Perceiver IO module after lifting
        # The input to PerceiverIOModule will be (batchsize, x_dim, width)
        self.perceiver_io = PerceiverIOModule(
            query_dim=self.width, context_dim=self.width, latent_dim=latent_dim, 
            num_latent_tokens=num_latent_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate
        )

        # FNO blocks
        self.fno_blocks = FNOBlock1d(self.width, self.modes, self.num_fno_layers)

        # Projection layer
        self.q = nn.Linear(self.width, out_channels)

    def forward(self, x): # x: (batchsize, x_dim, in_channels)
        # Lifting
        x = self.p(x) # (batchsize, x_dim, width)
        
        # Apply Perceiver IO module
        x = self.perceiver_io(x) # (batchsize, x_dim, width)
        
        x = x.permute(0, 2, 1) # (batchsize, width, x_dim)

        # FNO blocks
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic
        x = self.fno_blocks(x)
        x = x[..., :-self.padding] # unpad

        # Projection
        x = x.permute(0, 2, 1) # (batchsize, x_dim, width)
        x = self.q(x) # (batchsize, x_dim, out_channels)

        return x


class PerceiverFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels, num_fno_layers, 
                 latent_dim=64, num_latent_tokens=16, num_heads=8, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super(PerceiverFNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fno_layers = num_fno_layers

        self.padding = 9 # pad the domain if input is non-periodic

        # Lifting layer
        self.p = nn.Linear(in_channels, self.width)
        
        # Perceiver IO module after lifting
        # The input to PerceiverIOModule will be (batchsize, x_dim * y_dim, width)
        self.perceiver_io = PerceiverIOModule(
            query_dim=self.width, context_dim=self.width, latent_dim=latent_dim, 
            num_latent_tokens=num_latent_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate
        )

        # FNO blocks
        self.fno_blocks = FNOBlock2d(self.width, self.modes1, self.modes2, self.num_fno_layers)

        # Projection layer
        self.q = nn.Linear(self.width, out_channels)

    def forward(self, x): # x: (batchsize, x_dim, y_dim, in_channels)
        batch_size, x_dim, y_dim, in_channels = x.shape

        # Lifting
        x = self.p(x) # (batchsize, x_dim, y_dim, width)

        # Reshape for PerceiverIOModule: (B, H, W, D) -> (B, H*W, D)
        x_reshaped = x.view(batch_size, -1, self.width)
        
        # Apply Perceiver IO module
        x_perceiver = self.perceiver_io(x_reshaped) # (batchsize, H*W, width)
        
        # Reshape back: (B, H*W, D) -> (B, D, H, W)
        x = x_perceiver.view(batch_size, x_dim, y_dim, self.width)
        x = x.permute(0, 3, 1, 2) # (batchsize, width, x_dim, y_dim)

        # FNO blocks
        x = F.pad(x, [0, self.padding, 0, self.padding]) # pad the domain if input is non-periodic
        x = self.fno_blocks(x)
        x = x[..., :-self.padding, :-self.padding] # unpad

        # Projection
        x = x.permute(0, 2, 3, 1) # (batchsize, x_dim, y_dim, width)
        x = self.q(x) # (batchsize, x_dim, y_dim, out_channels)

        return x

