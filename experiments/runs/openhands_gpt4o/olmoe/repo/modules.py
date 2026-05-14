# modules.py

import torch
import torch.nn as nn

class MoELayer(nn.Module):
    def __init__(self, d_model: int, num_experts: int, num_active_experts: int):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        routing_logits = self.router(x)
        routing_probs = torch.softmax(routing_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.num_active_experts, dim=-1)

        expert_outputs = []
        for i in range(self.num_active_experts):
            expert_idx = top_k_indices[:, i]
            expert_output = self.experts[expert_idx](x)
            expert_outputs.append(expert_output * top_k_probs[:, i].unsqueeze(-1))

        return sum(expert_outputs)