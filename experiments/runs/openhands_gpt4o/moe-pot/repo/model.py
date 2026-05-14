import torch
import torch.nn as nn
from modules import RouterGatingNetwork, SharedExpert, RoutedExpert
from layers import FourierLayer

class MoEPOT(nn.Module):
    def __init__(self, attention_dim: int, mlp_dim: int, num_layers: int, num_heads: int,
                 num_routed_experts: int, num_shared_experts: int, top_k: int):
        super(MoEPOT, self).__init__()
        self.layers = nn.ModuleList([
            MoEBlock(attention_dim, mlp_dim, num_heads, num_routed_experts, num_shared_experts, top_k)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MoEBlock(nn.Module):
    def __init__(self, attention_dim: int, mlp_dim: int, num_heads: int,
                 num_routed_experts: int, num_shared_experts: int, top_k: int):
        super(MoEBlock, self).__init__()
        self.fourier_layer = FourierLayer(attention_dim, num_heads)
        self.router_gating = RouterGatingNetwork(attention_dim, num_routed_experts, top_k)
        self.shared_experts = nn.ModuleList([SharedExpert(attention_dim, mlp_dim) for _ in range(num_shared_experts)])
        self.routed_experts = nn.ModuleList([RoutedExpert(attention_dim, mlp_dim) for _ in range(num_routed_experts)])

    def forward(self, x):
        x = self.fourier_layer(x)
        routing_weights, selected_experts = self.router_gating(x)

        shared_output = sum(expert(x) for expert in self.shared_experts) / len(self.shared_experts)
        routed_output = sum(routing_weights[i] * self.routed_experts[i](x) for i in selected_experts)

        return shared_output + routed_output