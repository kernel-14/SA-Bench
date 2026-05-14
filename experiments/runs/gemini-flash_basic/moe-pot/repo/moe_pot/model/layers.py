import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        
        # Section 4: "we apply a patchification layer with positional embeddings inspired by vision transformers [10]:"
        # "where P is a convolutional layer"
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # "and p^t_{i,j} = W_p(x_i, y_j, t) denotes learnable positional encodings."
        # Assuming 2D spatial dimensions (H, W) and then flattened. The positional encoding should be added to the projected patches.
        # The paper states output Z_p^t is in R^(H/p x W/p x C), meaning C is the embed_dim.
        # W_p is in R^(n x 3) where n=C. This implies 3 dimensions for positional encoding (x, y, t).
        # We'll create a learnable positional embedding that's added *after* the convolution.
        # (H/patch_size) * (W/patch_size) patches in spatial dimension
        num_patches_h = img_size[0] // patch_size
        num_patches_w = img_size[1] // patch_size
        self.num_patches = num_patches_h * num_patches_w
        
        # Learnable positional embeddings. The 't' in W_p(x,y,t) suggests time-dependent positional encoding
        # which will be handled in the TemporalAggregation layer, or added per time step.
        # For a single time step u^t, the positional embedding is spatial.
        self.position_embeddings = nn.Parameter(torch.zeros(1, embed_dim, num_patches_h, num_patches_w))

    def forward(self, x):
        # x: (B, C_in, H, W) for a single time step u^t
        x = self.proj(x) # (B, embed_dim, H/patch_size, W/patch_size)
        x = x + self.position_embeddings # Add learnable spatial positional embedding
        return x

class TemporalAggregation(nn.Module):
    def __init__(self, embed_dim, fourier_feature_constant_dim):
        super().__init__()
        # Section 4: "For each local node feature z_p^t in Z_p^t, we apply a learnable MLP transformation W_t
        # combined with Fourier feature constant gamma in R^C"
        # "z_agg = sum_t W_t * z_p^t * e^(-i * gamma * t)"
        # W_t is a learnable MLP transformation. We'll use a linear layer as a simple MLP for W_t.
        # The paper states 'W_t' which implies a different W for each time step or a shared W_t that operates on each time step input.
        # Given "sum_t W_t * z_p^t * e^(-i * gamma * t)", it suggests a shared W_t applied across time, and then summed.
        # We will assume W_t is a single linear layer applied to the flattened spatial features.
        self.mlp_w_t = nn.Linear(embed_dim, embed_dim)
        
        # gamma is a Fourier feature constant.
        self.gamma = nn.Parameter(torch.randn(fourier_feature_constant_dim)) # R^C, where C is embed_dim

    def forward(self, z_p_t_sequence):
        # z_p_t_sequence: (B, T, embed_dim, H_patches, W_patches)
        # We need to flatten the spatial dimensions to apply W_t to each "local node feature"
        B, T, C, H_p, W_p = z_p_t_sequence.shape
        z_p_t_reshaped = z_p_t_sequence.permute(0, 1, 3, 4, 2).reshape(B * T * H_p * W_p, C) # (B*T*num_patches, embed_dim)
        
        # Apply W_t
        transformed_z_p_t = self.mlp_w_t(z_p_t_reshaped) # (B*T*num_patches, embed_dim)
        transformed_z_p_t = transformed_z_p_t.reshape(B, T, H_p, W_p, C).permute(0, 1, 4, 2, 3) # (B, T, embed_dim, H_p, W_p)
        
        # Prepare for complex multiplication with e^(-i * gamma * t)
        # We need 't' values. Assuming t is normalized or directly corresponds to time step index.
        # Let's assume t ranges from 0 to T-1.
        
        # Create a tensor for time 't' for each element in the batch and spatial location.
        # This part requires careful handling of complex numbers.
        # z_agg = sum_t W_t * z_p^t * cos(-gamma*t) + i * sum_t W_t * z_p^t * sin(-gamma*t)
        # The paper uses e^(-i * gamma * t), suggesting complex-valued features.
        # If z_p^t is real, then z_agg will be complex. For now, let's assume we handle real input
        # and result in a real output, potentially by taking the real part or magnitude.
        # However, the subsequent Fourier layer implies complex numbers will be handled.
        
        # For simplicity, let's assume gamma is applied per channel and per time step, and we simulate the complex multiplication
        # by creating complex numbers for both z_p^t and the exponential term.
        
        # Extend gamma to match the shape of transformed_z_p_t for broadcasting
        # We'll assume fourier_feature_constant_dim is the same as embed_dim for gamma.
        
        # Construct time vector
        time_steps = torch.arange(T, device=z_p_t_sequence.device, dtype=torch.float32).view(1, T, 1, 1, 1)
        
        # gamma is R^C. It needs to interact with 't' to form R^(T x C).
        # Reshape gamma to (1, 1, C, 1, 1) to broadcast with transformed_z_p_t
        gamma_reshaped = self.gamma.view(1, 1, C, 1, 1)
        
        # Compute the argument for the complex exponential: -gamma * t
        arg = -gamma_reshaped * time_steps # (1, T, C, 1, 1) * (1, T, 1, 1, 1) -> (1, T, C, 1, 1)
        
        # Create the complex exponential term e^(-i * gamma * t)
        exp_term = torch.exp(torch.complex(torch.zeros_like(arg), arg)) # (1, T, C, 1, 1) complex
        
        # Convert transformed_z_p_t to complex
        transformed_z_p_t_complex = torch.complex(transformed_z_p_t, torch.zeros_like(transformed_z_p_t))
        
        # Element-wise multiplication
        product = transformed_z_p_t_complex * exp_term # (B, T, C, H_p, W_p) complex
        
        # Sum over time dimension
        z_agg = torch.sum(product, dim=1) # (B, C, H_p, W_p) complex
        
        return z_agg

class FourierLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, img_size_patches, activation_fn=F.gelu):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.activation_fn = activation_fn
        self.img_size_patches = img_size_patches # (H_p, W_p)

        # Section 4: "we first divide spatial features z^l(x) into h groups."
        # "z_0i^l(x) = F^-1[W_2,i^l * sigma(W_1,i^l * F[z_i^l] + b_1,i^l) + b_2,i^l](x)"
        # W_1,i^l, W_2,i^l are in R^(d_z/h x d_z/h)
        # b_1,i^l, b_2,i^l are in R^(d_z/h)
        
        # We need to apply FFT, then linear layers, then IFFT.
        # These are frequency-dependent learnable transformations.
        
        # Store weights and biases for each head
        self.weights1 = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim, img_size_patches[0], img_size_patches[1], 2)) # Real and Imaginary parts
        self.biases1 = nn.Parameter(torch.empty(num_heads, self.head_dim, img_size_patches[0], img_size_patches[1], 2))
        self.weights2 = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim, img_size_patches[0], img_size_patches[1], 2))
        self.biases2 = nn.Parameter(torch.empty(num_heads, self.head_dim, img_size_patches[0], img_size_patches[1], 2))
        
        nn.init.kaiming_uniform_(self.weights1, a=1)
        nn.init.kaiming_uniform_(self.weights2, a=1)
        nn.init.zeros_(self.biases1)
        nn.init.zeros_(self.biases2)

    def forward(self, z_l):
        # z_l: (B, embed_dim, H_p, W_p) complex tensor from TemporalAggregation
        B, C, H_p, W_p = z_l.shape

        # Divide into heads
        z_l_heads = z_l.view(B, self.num_heads, self.head_dim, H_p, W_p) # (B, num_heads, head_dim, H_p, W_p)

        output_heads = []
        for i in range(self.num_heads):
            z_i_l = z_l_heads[:, i, :, :, :] # (B, head_dim, H_p, W_p)

            # F[z_i^l] - FFT
            # torch.fft.fftn operates on the last 'n' dimensions. Here, we want 2D FFT on H_p, W_p.
            fft_z_i_l = torch.fft.fftn(z_i_l, dim=(-2, -1)) # (B, head_dim, H_p, W_p) complex

            # Convert learnable parameters to complex
            w1_complex = torch.complex(self.weights1[i, ..., 0], self.weights1[i, ..., 1])
            b1_complex = torch.complex(self.biases1[i, ..., 0], self.biases1[i, ..., 1])
            w2_complex = torch.complex(self.weights2[i, ..., 0], self.weights2[i, ..., 1])
            b2_complex = torch.complex(self.biases2[i, ..., 0], self.biases2[i, ..., 1])

            # W_1,i^l * F[z_i^l] + b_1,i^l
            # This is a batched matrix multiplication. The weights are frequency-dependent.
            # fft_z_i_l: (B, head_dim, H_p, W_p)
            # w1_complex: (head_dim, head_dim, H_p, W_p)
            
            # einsum for matrix multiplication per spatial frequency and batch
            # (b h_in h_p w_p), (h_out h_in h_p w_p) -> (b h_out h_p w_p)
            term1 = torch.einsum('b c h w, d c h w -> b d h w', fft_z_i_l, w1_complex) + b1_complex
            
            activated_term1 = self.activation_fn(term1.real) + 1j * self.activation_fn(term1.imag) # Apply activation to real and imag parts

            # W_2,i^l * sigma(...) + b_2,i^l
            term2 = torch.einsum('b c h w, d c h w -> b d h w', activated_term1, w2_complex) + b2_complex
            
            # F^-1[...] - IFFT
            ifft_term2 = torch.fft.ifftn(term2, dim=(-2, -1)) # (B, head_dim, H_p, W_p) complex
            
            output_heads.append(ifft_term2)

        # Concatenate heads
        z_0_l = torch.cat(output_heads, dim=1) # (B, embed_dim, H_p, W_p) complex
        return z_0_l

class MoELayer(nn.Module):
    def __init__(self, embed_dim, num_routed_experts, num_shared_experts, top_k, capacity_factor_mlp=1.25, capacity_factor_router=1.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.capacity_factor_mlp = capacity_factor_mlp
        self.capacity_factor_router = capacity_factor_router # Not directly used for simple TopK, but good to keep for future.

        # Section 4: "Both expert networks and router-gating networks are implemented using convolutional neural networks (CNNs) to preserve spatial information."
        # For experts, we will use a simple CNN block.
        # A common choice for expert networks is a small MLP or a bottleneck design.
        # Given "output feature map of the same shape", we'll use a Conv2d block with kernel size 1.
        
        # Shared Experts
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
            ) for _ in range(num_shared_experts)
        ])

        # Routed Experts
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
            ) for _ in range(num_routed_experts)
        ])

        # Router-Gating Network
        # "router-gating network G^l(z_0^l(x)), which computes a vector of routing logits s^l(z_0^l(x)) in R^N_r"
        # It's a CNN to preserve spatial information, outputting N_r logits per spatial location.
        self.router = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, num_routed_experts, kernel_size=1) # Output N_r logits for experts
        )

    def forward(self, z_0_l):
        # z_0_l: (B, embed_dim, H_p, W_p) real tensor (after taking real part from Fourier Layer output or if Fourier Layer output was processed)
        # Assuming FourierLayer output is now real or its magnitude/real part is used.
        # The paper's equations show z_0^l(x) as input to MoE layer, which implies a real value.
        # Let's explicitly take the real part to handle potential complex output from Fourier layer for experts input.
        if z_0_l.is_complex():
            z_0_l_real = z_0_l.real
        else:
            z_0_l_real = z_0_l

        B, C, H_p, W_p = z_0_l_real.shape

        # Shared Experts calculation
        shared_expert_outputs = []
        for expert in self.shared_experts:
            shared_expert_outputs.append(expert(z_0_l_real)) # (B, embed_dim, H_p, W_p)

        # Average shared expert outputs
        sum_shared_experts_output = sum(shared_expert_outputs)
        # Equation 145: (1 / N_s) * sum(E_i^l(s)(z_0^l(x)))
        shared_output = sum_shared_experts_output / self.num_shared_experts

        # Router-Gating Network
        routing_logits = self.router(z_0_l_real) # (B, num_routed_experts, H_p, W_p)
        
        # Apply softmax per spatial location across experts
        # "gating weights are computed via a softmax function"
        # w^l(z_0^l(x)) = Softmax(s^l(z_0^l(x)))
        gating_weights_all = F.softmax(routing_logits, dim=1) # (B, num_routed_experts, H_p, W_p)

        # Top-K selection
        # "only the Top-K entries in w^l are retained, and the rest are masked to zero"
        # This requires finding the top_k values and their indices for each spatial location and batch item.
        # Flatten spatial dimensions to apply topk easily
        gating_weights_flat = gating_weights_all.view(B, self.num_routed_experts, -1) # (B, N_r, H_p*W_p)
        
        # Get top-k values and indices for each (B, H_p*W_p) combination
        # The topk operation is usually on the last dimension of values.
        # Here we want top_k experts per spatial location.
        # So we need to transpose to get experts on the last dim.
        gating_weights_per_location = gating_weights_all.permute(0, 2, 3, 1).reshape(B * H_p * W_p, self.num_routed_experts)
        
        top_k_weights, top_k_indices = torch.topk(gating_weights_per_location, self.top_k, dim=-1)
        
        # Create a mask for selected experts
        mask = torch.zeros_like(gating_weights_per_location, dtype=torch.bool)
        mask.scatter_(-1, top_k_indices, True)
        
        # Apply mask to gating weights, zeroing out non-selected experts
        gating_weights_selected = gating_weights_per_location * mask
        
        # Normalize the selected weights to sum to 1 (if the original topk values didn't sum to 1)
        # "w_k^l(x) is the normalized routing weight"
        # If we just masked, they might not sum to 1. So re-normalize.
        sum_top_k_weights = gating_weights_selected.sum(dim=-1, keepdim=True)
        # Avoid division by zero for locations where all top_k_weights might be zero (unlikely with softmax)
        sum_top_k_weights = torch.where(sum_top_k_weights == 0, torch.tensor(1.0, device=sum_top_k_weights.device), sum_top_k_weights)
        gating_weights_normalized = gating_weights_selected / sum_top_k_weights
        
        # Reshape back to (B, N_r, H_p, W_p)
        gating_weights_final = gating_weights_normalized.view(B, H_p, W_p, self.num_routed_experts).permute(0, 3, 1, 2)
        
        # Compute routed expert outputs
        routed_expert_outputs = torch.zeros_like(shared_expert_outputs[0]) # (B, embed_dim, H_p, W_p)
        
        # Iterate over experts and multiply by their gating weight
        for i, expert in enumerate(self.routed_experts):
            # Only consider experts that were selected for at least one location
            # To apply gating_weights_final properly, we need to multiply expert output by its specific weight.
            # E_i^l(r)(z_0^l(x)) is (B, embed_dim, H_p, W_p)
            # gating_weights_final is (B, N_r, H_p, W_p)
            
            # The expert output itself is (B, embed_dim, H_p, W_p)
            expert_output = expert(z_0_l_real)
            
            # Multiply by the gating weight for this expert across all spatial locations
            # The gating weight for expert 'i' is gating_weights_final[:, i, :, :]
            # This needs to be broadcast to (B, embed_dim, H_p, W_p)
            weighted_expert_output = expert_output * gating_weights_final[:, i, :, :].unsqueeze(1)
            routed_expert_outputs += weighted_expert_output
            
        # Equation 145: sum_k=1^K w_k^l(z_0^l(x)) * E_i_k^l(r)(z_0^l(x))
        # The way I've implemented above, `routed_expert_outputs` already sums the weighted contributions.
        
        # Combine shared and routed outputs
        z_l_plus_1 = shared_output + routed_expert_outputs
        
        # Store gating_weights_all for load balancing loss calculation (outside this module)
        # This will be the (B, N_r, H_p, W_p) before TopK for calculating importance.
        self.gating_weights_all_for_loss = gating_weights_all

        return z_l_plus_1

