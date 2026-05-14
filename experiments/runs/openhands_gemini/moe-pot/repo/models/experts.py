
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Expert(nn.Module):
    """
    A single expert network, implemented as a simple Convolutional Neural Network (CNN).
    Maps input feature map to an output feature map of the same shape.
    """
    def __init__(self, embed_dim: int, mlp_dim: int):
        super().__init__()
        # The paper mentions "convolutional subnetwork" for experts.
        # A simple block with two conv layers and GELU activation.
        # Assuming 1x1 convolutions to maintain spatial dimensions,
        # often used as MLP in CNN context (e.g. FNO has MLP blocks).
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mlp_dim, embed_dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, H', W', embed_dim)
        # Permute to (B, embed_dim, H', W') for Conv2d
        x = rearrange(x, 'b h w c -> b c h w')
        
        # Apply CNN
        x = self.net(x)
        
        # Permute back to (B, H', W', embed_dim)
        x = rearrange(x, 'b c h w -> b h w c')
        return x

class RouterGatingNetwork(nn.Module):
    """
    Router-gating network to dynamically select Top-K routed experts.
    Implemented using CNNs to preserve spatial information.
    """
    def __init__(self, embed_dim: int, num_routed_experts: int):
        super().__init__()
        self.num_routed_experts = num_routed_experts
        
        # The paper mentions CNNs for router-gating network.
        # A simple Conv2d followed by flattening and a linear layer for logits.
        # This will output N_r logits per patch.
        # The routing decision can be per patch or global.
        # "router-gating network G^l(z_0^l(x)), which computes a vector of routing logits s^l(z_0^l(x))"
        # suggests per-patch routing.
        
        # Let's use a 1x1 conv to project embed_dim to num_routed_experts for logits.
        # This means each spatial location (patch) will have its own routing weights.
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, num_routed_experts, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, H', W', embed_dim)
        B, H_prime, W_prime, embed_dim = x.shape
        
        # Permute to (B, embed_dim, H', W') for Conv2d
        x = rearrange(x, 'b h w c -> b c h w')
        
        # Compute logits: (B, num_routed_experts, H', W')
        logits = self.gate(x)
        
        # Apply softmax per patch: (B, num_routed_experts, H', W')
        # The paper: w^l(z_0^l(x)) = Softmax(s^l(z_0^l(x)))
        # So softmax should be applied over the expert dimension for each spatial location.
        weights = F.softmax(logits, dim=1) # Softmax over expert dimension
        
        # Permute back to (B, H', W', num_routed_experts)
        weights = rearrange(weights, 'b c h w -> b h w c')

        return logits, weights # Logits for load balancing, weights for expert selection

class MoELayer(nn.Module):
    """
    Mixture-of-Experts (MoE) layer with shared and routed experts.
    """
    def __init__(self, embed_dim: int, mlp_dim: int, num_routed_experts: int, num_shared_experts: int, top_k: int):
        super().__init__()
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.embed_dim = embed_dim

        # Shared experts
        self.shared_experts = nn.ModuleList([
            Expert(embed_dim, mlp_dim) for _ in range(num_shared_experts)
        ])

        # Routed experts
        self.routed_experts = nn.ModuleList([
            Expert(embed_dim, mlp_dim) for _ in range(num_routed_experts)
        ])

        # Router-gating network
        self.router = RouterGatingNetwork(embed_dim, num_routed_experts)

    def forward(self, x: torch.Tensor):
        # x: (B, H', W', embed_dim)
        B, H_prime, W_prime, embed_dim = x.shape

        # 1. Shared experts: always activated
        shared_expert_outputs = []
        for expert in self.shared_experts:
            shared_expert_outputs.append(expert(x))
        
        # Average shared expert outputs: (B, H', W', embed_dim)
        # Equation: (1/Ns) * sum(E_i^l(s)(z_0^l(x)))
        shared_output = sum(shared_expert_outputs) / self.num_shared_experts

        # 2. Routed experts: dynamic selection
        # logits for load balancing loss, expert_weights for selection
        router_logits, expert_weights = self.router(x) # (B, H', W', num_routed_experts)

        # Select Top-K experts per patch
        # expert_weights are (B, H', W', num_routed_experts)
        # We need to find top-k indices and values for each patch.
        # This will result in (B, H', W', top_k) indices and weights.
        
        # Flatten spatial dimensions for TopK selection
        expert_weights_flat = rearrange(expert_weights, 'b h w c -> (b h w) c') # (N_patches, num_routed_experts)
        router_logits_flat = rearrange(router_logits, 'b c h w -> (b h w) c') # (N_patches, num_routed_experts)


        # Get top-k weights and indices
        top_k_weights, top_k_indices = torch.topk(expert_weights_flat, self.top_k, dim=-1)
        
        # Normalize top-k weights to sum to 1
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Initialize routed output
        routed_output = torch.zeros_like(x) # (B, H', W', embed_dim)
        
        # Iterate over each patch and apply selected experts
        # This is where the sparse activation happens.
        # This loop can be slow. Can optimize by gathering inputs for each expert.
        
        # A more efficient way: create a mask to apply to expert outputs
        # This is conceptually similar to how Switch Transformers implement this.
        
        # Gather inputs for each expert (might be less efficient than direct application for conv experts)
        # A more standard approach for MoE for dense layers involves dispatching tokens.
        # For CNN experts operating on feature maps, a mask-based approach or
        # using `index_select` to apply to parts of the feature map is common.
        
        # Let's iterate through the top_k selected experts for each patch.
        # This is computationally intensive if B*H'*W' is large.
        # Given the paper uses CNN experts and it's a "sparse-activated architecture",
        # the experts are applied to the full feature map (B, H', W', embed_dim).
        # This means, for each spatial location (x), we select K experts.
        # So, the summation `sum_k w_k^l(x) * E_i_k^l(r)(z_0^l(x))`
        # means each expert `E_j` gets applied to the *entire* input `z_0^l(x)`,
        # but its output is weighted by `w_j^l(x)`.

        # So, for each expert, calculate its output, then weight it by its routing weight (which is 0 for non-selected).
        all_expert_outputs = torch.zeros(
            B, H_prime, W_prime, self.num_routed_experts, embed_dim,
            dtype=x.dtype, device=x.device
        )

        for i, expert in enumerate(self.routed_experts):
            all_expert_outputs[:, :, :, i, :] = expert(x)
        
        # expert_weights: (B, H', W', num_routed_experts)
        # all_expert_outputs: (B, H', W', num_routed_experts, embed_dim)
        # Expand expert_weights to (B, H', W', num_routed_experts, 1) for broadcasting
        weighted_expert_outputs = all_expert_outputs * expert_weights.unsqueeze(-1)
        
        # Sum over num_routed_experts dimension
        routed_output = weighted_expert_outputs.sum(dim=-2) # (B, H', W', embed_dim)

        # Final output is sum of shared and routed outputs
        final_output = shared_output + routed_output
        
        # Calculate load balancing loss components
        # Importance_i^l = sum_b=1^B w_i,b^l(x)
        # This sums weights over batch and spatial dimensions.
        # expert_weights: (B, H', W', num_routed_experts)
        
        # Sum over batch and spatial dimensions for each expert
        # Result: (num_routed_experts,)
        expert_importance = expert_weights.sum(dim=(0, 1, 2))
        
        # Compute the coefficient of variation (CV) for load balancing.
        # CV = std_dev / mean
        mean_importance = expert_importance.mean()
        std_importance = expert_importance.std()
        
        # Handle case where mean might be zero to avoid NaNs
        if mean_importance == 0:
            load_balancing_loss = torch.tensor(0.0, device=x.device)
        else:
            load_balancing_loss = (std_importance / mean_importance)**2

        return final_output, load_balancing_loss, expert_weights_flat # expert_weights_flat for dataset classification

if __name__ == '__main__':
    # Test Expert
    B, H_prime, W_prime, embed_dim, mlp_dim = 2, 16, 16, 32, 64
    expert_input = torch.randn(B, H_prime, W_prime, embed_dim)
    expert_net = Expert(embed_dim, mlp_dim)
    expert_output = expert_net(expert_input)
    print(f"Expert output shape: {expert_output.shape}")
    assert expert_output.shape == (B, H_prime, W_prime, embed_dim)

    # Test RouterGatingNetwork
    num_routed_experts = 16
    router_net = RouterGatingNetwork(embed_dim, num_routed_experts)
    logits, weights = router_net(expert_input)
    print(f"Router logits shape: {logits.shape}")
    print(f"Router weights shape: {weights.shape}")
    assert logits.shape == (B, num_routed_experts, H_prime, W_prime) # Conv2D output format
    assert weights.shape == (B, H_prime, W_prime, num_routed_experts) # Softmax output format

    # Test MoELayer
    num_shared_experts = 2
    top_k = 4
    moe_layer = MoELayer(embed_dim, mlp_dim, num_routed_experts, num_shared_experts, top_k)
    moe_output, lb_loss, expert_weights_flat = moe_layer(expert_input)
    print(f"MoE output shape: {moe_output.shape}")
    print(f"Load Balancing Loss: {lb_loss.item()}")
    print(f"Expert weights flat shape: {expert_weights_flat.shape}")
    assert moe_output.shape == (B, H_prime, W_prime, embed_dim)
    assert isinstance(lb_loss, torch.Tensor)
    assert expert_weights_flat.shape == (B * H_prime * W_prime, num_routed_experts)

    print("All expert-related layers tested successfully!")
