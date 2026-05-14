
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, activation=nn.GELU):
        super().__init__()
        self.num_layers = num_layers
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.activation = activation()
        self.fc_hidden = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers - 2)])
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        for i in range(self.num_layers - 2):
            x = self.fc_hidden[i](x)
            x = self.activation(x)
        x = self.fc_out(x)
        return x

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        # (batch, in_channels, x), (in_channels, out_channels, x) -> (batch, out_channels, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # FNO uses 1D convolution on the last dimension of the input feature map (e.g. spatial dimension for 1D PDE)
        # Therefore, we assume x has shape (batch_size, channels, spatial_dim)
        
        # 1. Fourier transform
        x_ft = torch.fft.rfft(x)

        # 2. Multiply relevant modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights1)

        # 3. Inverse Fourier transform
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pos_embed[:, :seq_len]

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, *kv))

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

