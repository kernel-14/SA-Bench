import torch
import torch.nn as nn

class MoE(nn.Module):
    def __init__(self, num_experts, active_experts, input_dim, expert_dim):
        super(MoE, self).__init__()
        # Experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.active_experts = active_experts
        
        # Router
        self.router = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Routing scores
        routing_scores = self.router(x)
        probabilities = self.softmax(routing_scores)

        # Select top-k experts based on probabilities
        top_k_indices = torch.topk(probabilities, self.active_experts, dim=-1).indices
        outputs = []

        for i, indices in enumerate(top_k_indices):
            expert_outputs = []
            for index in indices:
                expert_outputs.append(self.experts[index](x[i]))
            weighted_output = sum(expert_outputs) / len(expert_outputs) # Average weighting
            outputs.append(weighted_output)

        return torch.stack(outputs)

# Auxiliary losses
class LoadBalancingLoss(nn.Module):
    def __init__(self):
        super(LoadBalancingLoss, self).__init__()

    def forward(self, probabilities):
        loss = -torch.mean(torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1))
        return loss

class RouterZLoss(nn.Module):
    def __init__(self):
        super(RouterZLoss, self).__init__()

    def forward(self, router_logits):
        loss = torch.mean(torch.sum(torch.square(router_logits), dim=-1))
        return loss

