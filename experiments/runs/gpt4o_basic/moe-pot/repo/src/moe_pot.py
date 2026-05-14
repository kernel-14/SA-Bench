import torch
import torch.nn as nn
import torch.fft

class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        # Learnable parameters for frequency-dependent transformations
        self.weight1 = nn.Parameter(torch.randn(num_heads, in_channels // num_heads))
        self.weight2 = nn.Parameter(torch.randn(num_heads, out_channels // num_heads))
        self.bias1 = nn.Parameter(torch.randn(num_heads))
        self.bias2 = nn.Parameter(torch.randn(num_heads))

    def forward(self, x):
        B, C, H, W = x.shape  # batch size, channels, spatial dimensions
        grouped_features = x.view(B, self.num_heads, C // self.num_heads, H, W)

        # Fourier transform
        freq_features = torch.fft.fftn(grouped_features, dim=(-2, -1))

        # Apply transformations in the frequency domain
        transformed = torch.fft.ifftn(
            self.weight2 * torch.relu(self.weight1 * freq_features + self.bias1) + self.bias2,
            dim=(-2, -1))

        return transformed.view(B, C, H, W)

# FourierLayer designed for complex integral approximations


class MoELayer(nn.Module):
    def __init__(self, in_features, num_routed_experts, num_shared_experts, top_k):
        super(MoELayer, self).__init__()
        
        # Define routed and shared experts
        self.num_routed = num_routed_experts
        self.num_shared = num_shared_experts
        self.top_k = top_k  # Top-K gating mechanism
        self.shared_experts = nn.ModuleList([nn.Conv2d(in_features, in_features, kernel_size=3, padding=1) for _ in range(num_shared_experts)])
        self.routed_experts = nn.ModuleList([nn.Conv2d(in_features, in_features, kernel_size=3, padding=1) for _ in range(num_routed_experts)])
        self.gating_network = nn.Sequential(
            nn.Conv2d(in_features, num_routed_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Compute gating weights for routed experts
        gating_logits = self.gating_network(x)  # Shape: (Batch, Routed Experts, H, W)
        top_k_weights, indices = torch.topk(gating_logits, self.top_k, dim=1)

        # Shared experts activation
        shared_output = sum(expert(x) for expert in self.shared_experts) / self.num_shared

        # Routed experts activation via Top-K
        routed_output = torch.zeros_like(shared_output)
        for batch_idx in range(x.size(0)):
            for k, expert_idx in enumerate(indices[batch_idx]):
                routed_output[batch_idx] += top_k_weights[batch_idx, k] * self.routed_experts[expert_idx](x[batch_idx])

        return shared_output + routed_output

# MoELayer combines routed and shared experts for sparse activation

