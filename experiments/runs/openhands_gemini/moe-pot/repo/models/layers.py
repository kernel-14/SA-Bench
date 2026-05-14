
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class PatchEmbedding(nn.Module):
    """
    Patchification layer with positional embeddings inspired by vision transformers.
    P(u^t + p^t)
    """
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int, H: int, W: int):
        super().__init__()
        self.patch_size = patch_size
        self.H_prime = H // patch_size
        self.W_prime = W // patch_size
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable positional encoding p^t = W_p(x_i, y_j, t)
        # The paper states W_p in R^(n x 3) where n is feature dimension (embed_dim)
        # This implies positional embeddings are added per patch.
        # Here we assume a fixed positional embedding across time for now,
        # and it's added to each patch. The paper's formulation is a bit ambiguous here.
        # We'll follow a common ViT approach: learnable embeddings per patch location.
        self.position_embeddings = nn.Parameter(torch.zeros(1, embed_dim, self.H_prime, self.W_prime))

    def forward(self, x: torch.Tensor):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        
        # Reshape to (B*T, C, H, W) for Conv2d
        x_reshaped = rearrange(x, 'b t h w c -> (b t) c h w')
        
        # Apply convolution for patchification: (B*T, embed_dim, H', W')
        patch_tokens = self.conv(x_reshaped)
        
        # Add positional embeddings. The paper mentions p^t, suggesting time-dependent
        # positional embeddings. For simplicity and common practice in ViT, we use
        # a single learnable embedding for each spatial patch location.
        # If the paper implies time-dependent, this might need adjustment.
        patch_tokens = patch_tokens + self.position_embeddings
        
        # Reshape back to (B, T, H', W', embed_dim)
        patch_tokens = rearrange(patch_tokens, '(b t) c h_prime w_prime -> b t h_prime w_prime c', b=B, t=T)
        
        return patch_tokens

class TemporalAggregation(nn.Module):
    """
    Temporal aggregation layer to extract information across adjacent time steps.
    z_agg = sum_t W_t * z_p^t * e^(-i * gamma * t)
    """
    def __init__(self, embed_dim: int, time_steps: int):
        super().__init__()
        self.time_steps = time_steps
        self.mlps = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(time_steps)])
        # Fourier feature constant gamma is a learnable parameter in R^C (embed_dim)
        self.gamma = nn.Parameter(torch.randn(embed_dim))

    def forward(self, z_p: torch.Tensor):
        # z_p: (B, T, H', W', embed_dim)
        B, T, H_prime, W_prime, embed_dim = z_p.shape
        
        # Initialize aggregated features
        z_agg = torch.zeros(B, H_prime, W_prime, embed_dim, dtype=z_p.dtype, device=z_p.device)
        
        for t in range(T):
            # z_p^t: (B, H', W', embed_dim)
            z_p_t = z_p[:, t, :, :, :]
            
            # Apply MLP transformation W_t: (B, H', W', embed_dim)
            mlp_out = self.mlps[t](z_p_t)
            
            # e^(-i * gamma * t) term. Using complex exponential for Fourier features.
            # Assuming gamma is real, then e^(-i*gamma*t) = cos(gamma*t) - i*sin(gamma*t)
            # For simplicity, if working with real-valued features, we can use a real-valued
            # weighting or split into real/imaginary parts.
            # The paper's formulation implies complex numbers, but typical neural networks operate on real.
            # We will use a real-valued cosine and sine for now.
            # A common way to handle this in real networks is to apply two linear layers
            # for real and imaginary parts or use a single real weight.
            # Here we follow a simplified interpretation for real networks.
            # A more precise implementation would involve complex tensors or
            # 2xMLP with careful handling.
            
            # Let's interpret `e^(-i * gamma * t)` as a frequency-based weighting.
            # We can use a learnable periodic function or map to real space.
            # For current purposes, assume it's a scalar weight applied element-wise.
            # Or perhaps, gamma is a scalar and element-wise multiply.
            # "Fourier feature constant gamma in R^C" implies vector gamma.
            
            # Revisit: sum_t W_t * z_p^t * e^(-i * gamma * t)
            # This looks like a Fourier transform. If z_p_t are real,
            # then the result is generally complex.
            # Since the output `z_agg` is used in subsequent layers, we'll keep it real for now.
            # A pragmatic way to handle this: the complex exponential is absorbed into W_t.
            # Or: just use `cos(gamma * t)` as a real-valued modulator.
            
            # For now, let's assume `e^(-i * gamma * t)` is a real-valued scaling factor.
            # This is a simplification to avoid complex numbers in PyTorch without a clear directive.
            # If the output must be real, one common way is to take the real part or magnitude.
            # Let's use `torch.cos(self.gamma * t_val)` as the weighting factor.
            # t_val = torch.tensor(float(t), device=z_p.device, dtype=z_p.dtype)
            # modulator = torch.cos(self.gamma * t_val) # (embed_dim,)
            # z_agg += mlp_out * modulator
            
            # More accurate interpretation for real-valued output using Fourier Series terms:
            # We can interpret W_t as learning the "Fourier coefficients" for each time step.
            # Let's simplify and just sum up the MLP outputs, effectively learning a time-agnostic sum
            # or implicitly encoding time in W_t. This deviates slightly from the explicit e^(-i*gamma*t)
            # but avoids immediate complex number handling.
            
            # To be more faithful, let's include a real-valued cosine modulator.
            # This still simplifies, but explicitly uses `gamma` and `t`.
            t_val = torch.tensor(float(t), device=z_p.device, dtype=z_p.dtype)
            modulator = torch.cos(self.gamma * t_val) # (embed_dim,)
            z_agg += mlp_out * modulator.unsqueeze(0).unsqueeze(0).unsqueeze(0) # (1,1,1,embed_dim) broadcast
            
        return z_agg # (B, H', W', embed_dim)

class FourierLayer(nn.Module):
    """
    Multi-head Fourier layer for learning kernel-based integral transformations.
    Approximates (K_phi z^l)(x) using h smaller MLPs in Fourier domain.
    """
    def __init__(self, embed_dim: int, num_heads: int, H_prime: int, W_prime: int, activation: nn.Module = nn.GELU()):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.activation = activation
        
        # Parameters for each head
        self.w1 = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
        self.b1 = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.w2 = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
        self.b2 = nn.Parameter(torch.empty(num_heads, self.head_dim))

        self.init_weights()

        # The paper uses FFT, which requires the spatial dimensions.
        # This layer operates on (B, H', W', embed_dim)
        # For FFT, we need to move channel dim.
        self.H_prime = H_prime
        self.W_prime = W_prime

    def init_weights(self):
        for i in range(self.num_heads):
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w1[i])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b1[i], -bound, bound)
            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w2[i])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b2[i], -bound, bound)

    def forward(self, z_agg: torch.Tensor):
        # z_agg: (B, H', W', embed_dim)
        B, H_prime, W_prime, embed_dim = z_agg.shape
        
        # Rearrange for multi-head processing and FFT: (B, embed_dim, H', W')
        z_agg = rearrange(z_agg, 'b h_prime w_prime c -> b c h_prime w_prime')
        
        output_features = []
        
        # Split features into heads: (B, num_heads, head_dim, H', W')
        z_heads = rearrange(z_agg, 'b (num_heads head_dim) h_prime w_prime -> b num_heads head_dim h_prime w_prime', num_heads=self.num_heads)

        for i in range(self.num_heads):
            z_i = z_heads[:, i, :, :, :] # (B, head_dim, H', W')
            
            # Apply 2D FFT
            # The paper's formulation (R_phi * F[z^l]) implies complex multiplication in Fourier space.
            # torch.fft.fft2 operates on the last two dimensions.
            # We need to apply it to H' x W' for each B x head_dim.
            
            # Move head_dim to the end for easier FFT application over spatial dims
            z_i_fft = torch.fft.fft2(z_i) # Output is complex (B, head_dim, H', W')
            
            # Apply W1, b1. These operate on `head_dim` dimension, frequency-wise.
            # The parameters w1/b1 are real, so this implies point-wise application per frequency.
            # Need to handle complex numbers explicitly for matrix multiplication if W1/B1 were complex.
            # Given W1, B1 are real, we apply them to real and imaginary parts separately.
            
            # Let's perform multiplication in the frequency domain.
            # z_i_fft is complex. W1, W2 are real.
            # F[z_i] is (B, head_dim, H', W')
            
            # Need to expand W1, b1 to match (B, head_dim, H', W') for broadcasting,
            # or permute z_i_fft to (B, H', W', head_dim) for standard matmul.
            # Let's permute for standard matmul with W1, W2.
            z_i_fft_permuted = rearrange(z_i_fft, 'b c h w -> b h w c') # (B, H', W', head_dim)
            
            # Apply W1, b1: (B, H', W', head_dim)
            # torch.matmul handles complex numbers
            temp = torch.matmul(z_i_fft_permuted, self.w1[i]) + self.b1[i]
            
            # Activation function (e.g., GELU) typically for real numbers.
            # If temp is complex, applying GELU directly might not be standard.
            # A common approach for complex activation is to apply it to magnitude,
            # or real/imag parts separately.
            # Given no specific instruction, we apply GELU to real and imaginary parts.
            # But the paper says sigma(.) is an activation function. For FNO, it's typically GELU.
            # If the output is R_phi * F[z^l], then R_phi is complex.
            # W1 and W2 in the paper are in R^(d_z/h x d_z/h), meaning real matrices.
            # This implies they operate on the real and imaginary parts independently,
            # or the entire computation is implicitly real.
            # The phrasing "R_phi(k) in C^(d_z x d_z)" implies R_phi is a complex matrix.
            # If W1, W2 are real, then they are not R_phi directly.
            # This indicates a common approximation or simplification in the paper.

            # Re-reading: z_0i^l(x) = F^-1[W_2,i * sigma(W_1,i * F[z_i^l] + b_1,i) + b_2,i](x)
            # This means W_1, b_1, W_2, b_2 are applied *in the frequency domain*.
            # If F[z_i^l] is complex, and W_1, b_1 are real, then W_1 * F[z_i^l] + b_1 will be complex.
            # Applying `sigma` (GELU) to complex numbers: torch.gelu(complex_tensor) works by applying
            # GELU to real and imaginary parts independently. This is a common heuristic.

            # Apply activation: (B, H', W', head_dim)
            activated_temp = self.activation(temp)
            
            # Apply W2, b2: (B, H', W', head_dim)
            result_fft_permuted = torch.matmul(activated_temp, self.w2[i]) + self.b2[i]
            
            # Permute back for IFFT: (B, head_dim, H', W')
            result_fft = rearrange(result_fft_permuted, 'b h w c -> b c h w')
            
            # Apply Inverse 2D FFT
            z_0i_x = torch.fft.ifft2(result_fft).real # Take real part as output features are real
            
            output_features.append(z_0i_x)
            
        # Concatenate outputs from all heads: (B, embed_dim, H', W')
        z_concat = torch.cat(output_features, dim=1)
        
        # Permute back to (B, H', W', embed_dim)
        z_concat = rearrange(z_concat, 'b c h_prime w_prime -> b h_prime w_prime c')

        return z_concat

if __name__ == '__main__':
    # Test PatchEmbedding
    B, T, H, W, C = 2, 5, 64, 64, 3
    patch_size = 8
    embed_dim = 32
    
    x = torch.randn(B, T, H, W, C)
    patch_embed_layer = PatchEmbedding(patch_size, C, embed_dim, H, W)
    z_p = patch_embed_layer(x)
    print(f"PatchEmbedding output shape: {z_p.shape}")
    assert z_p.shape == (B, T, H // patch_size, W // patch_size, embed_dim)

    # Test TemporalAggregation
    time_steps = T
    temporal_agg_layer = TemporalAggregation(embed_dim, time_steps)
    z_agg = temporal_agg_layer(z_p)
    print(f"TemporalAggregation output shape: {z_agg.shape}")
    assert z_agg.shape == (B, H // patch_size, W // patch_size, embed_dim)

    # Test FourierLayer
    num_heads = 4
    H_prime = H // patch_size
    W_prime = W // patch_size
    fourier_layer = FourierLayer(embed_dim, num_heads, H_prime, W_prime)
    z_out_fourier = fourier_layer(z_agg)
    print(f"FourierLayer output shape: {z_out_fourier.shape}")
    assert z_out_fourier.shape == (B, H_prime, W_prime, embed_dim)

    print("All layers tested successfully!")
