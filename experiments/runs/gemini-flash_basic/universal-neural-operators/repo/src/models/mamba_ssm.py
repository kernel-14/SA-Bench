import torch
import torch.nn as nn
import torch.nn.functional as F

# Simplified Mamba-like SSM block focusing on causal convolution
# This is a simplified interpretation based on the paper's description
# of Mamba's role (encoding long-range dependencies via causal convolution).
# A full Mamba implementation is considerably more complex.
class MambaSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.in_proj = nn.Linear(d_model, d_model * expand * 2)
        self.conv1d = nn.Conv1d(
            in_channels=d_model * expand,
            out_channels=d_model * expand,
            kernel_size=d_conv,
            groups=d_model * expand,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(d_model * expand, d_state * 2) # (B, L, ED) -> (B, L, 2*d_state)
        self.dt_proj = nn.Linear(d_model * expand, d_model * expand)
        self.out_proj = nn.Linear(d_model * expand, d_model)

        # A, B, C, D are state-space parameters
        # A is a learnable matrix, B is input mixing, C is output mixing, D is skip connection
        # For simplicity, A and B are initialized but can be made more complex.
        # This is a highly simplified version of the Selective Scan mechanism in Mamba
        self.A = nn.Parameter(torch.exp(torch.rand(d_state, d_model * expand)))
        self.B = nn.Parameter(torch.rand(d_model * expand, d_state))
        self.C = nn.Parameter(torch.rand(d_model * expand, d_state))
        self.D = nn.Parameter(torch.ones(d_model * expand))
        

    def forward(self, x): # x: (B, L, d_model)
        batch_size, seq_len, d_model = x.shape

        # Input projection
        xz = self.in_proj(x) # (B, L, d_model * expand * 2)
        x, z = xz.chunk(2, dim=-1) # x: (B, L, d_model * expand), z: (B, L, d_model * expand)

        # Conv1d expects (B, C, L)
        x = x.transpose(1, 2) # (B, d_model * expand, L)
        x = self.conv1d(x)[:, :, :seq_len] # (B, d_model * expand, L)
        x = x.transpose(1, 2) # (B, L, d_model * expand)

        x = F.silu(x)

        # Simplified Selective Scan (sSM)
        # This part is a very simplified approximation of Mamba's core logic.
        # A more faithful implementation would involve complex recurrent computation.

        dt = self.dt_proj(x) # (B, L, d_model * expand)
        dt = F.softplus(dt) # Ensure dt is positive
        
        A = -torch.exp(self.A) # (d_state, d_model * expand)

        # The following lines attempt to mimic the selective scan operation without recursion
        # This is a drastic simplification and might not capture the full essence of Mamba's selectivity
        # A proper Mamba implementation involves a scan operation over the sequence length.
        # Here, we're doing a more direct, but less selective, state update.
        
        # This is a place holder. A true Mamba SSM is complex and cannot be fully replicated with simple matrix ops
        # without actual scan/recurrence. For this exercise, I'll simulate a simplified state update.
        # dt_A = torch.exp(dt.unsqueeze(-1) * A) # (B, L, d_model*expand, d_state)
        # dt_B = dt.unsqueeze(-1) * self.B # (B, L, d_model*expand, d_state)

        # For static simulation, we will simplify the state propagation to a form that is compatible 
        # with non-recurrent operations, acknowledging it's a simplification.
        # This is NOT a full Mamba implementation but aims to capture the spirit of state interaction.

        # For the purpose of static code representation, let's assume a simplified interaction.
        # In a true Mamba, B and C are input and output mixing matrices that are input-dependent.
        # Here, we will use fixed B and C for simplicity, demonstrating the structure.
        
        # This block is a simplified placeholder for the state update equation of Mamba.
        # It will not perform the actual recurrence but shows the intended inputs.
        # x_proj_out = self.x_proj(x) # (B, L, 2 * d_state)
        # B_selective, C_selective = x_proj_out.chunk(2, dim=-1) # (B, L, d_state)
        
        # Simplified version: Treat A, B, C as fixed for the forward pass, apply a transformation.
        # This is a major simplification. The actual Mamba involves selective recurrence.
        # Here, we will just apply a linear transformation and element-wise products.
        
        # Expand A, B, C for batch processing (conceptual)
        A_prime = A.unsqueeze(0).unsqueeze(0) # (1, 1, d_state, d_model * expand)
        B_prime = self.B.unsqueeze(0).unsqueeze(0) # (1, 1, d_model * expand, d_state)
        C_prime = self.C.unsqueeze(0).unsqueeze(0) # (1, 1, d_model * expand, d_state)

        # State computation placeholder (highly simplified)
        # This would be the core selective scan in a real Mamba
        # For static code, we just perform a series of linear ops as a proxy.
        
        # y = H * C + D * x_conv
        # H_t = (A_t * H_{t-1}) + (B_t * x_conv_t)
        
        # Placeholder for the SSM output, effectively a linear transformation
        # This does not fully capture the recurrence but represents the module's position.
        ssm_output = F.linear(x, self.C.T) # (B, L, d_state) * (d_state, ED) -> (B, L, ED)
        ssm_output = F.silu(ssm_output) * F.silu(z)
        
        # This is a placeholder for the output of the SSM block. In a real Mamba,
        # this would be the result of the selective scan operation.
        output = self.out_proj(ssm_output) # (B, L, d_model)

        return output



class MambaFNO1d(nn.Module):
    def __init__(self, modes, width, in_channels, out_channels, num_fno_layers, d_state=16, d_conv=4, expand=2):
        super(MambaFNO1d, self).__init__()
        self.modes = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fno_layers = num_fno_layers

        self.padding = 9 # pad the domain if input is non-periodic

        # Lifting layer
        self.p = nn.Linear(in_channels, self.width)
        
        # Mamba SSM module after lifting
        self.mamba_ssm = MambaSSM(d_model=self.width, d_state=d_state, d_conv=d_conv, expand=expand)

        # FNO blocks
        self.fno_blocks = FNOBlock1d(self.width, self.modes, self.num_fno_layers)

        # Projection layer
        self.q = nn.Linear(self.width, out_channels)

    def forward(self, x): # x: (batchsize, x_dim, in_channels)
        # Lifting
        x = self.p(x) # (batchsize, x_dim, width)
        
        # Apply Mamba SSM to the lifted features
        x = self.mamba_ssm(x) # (batchsize, x_dim, width)
        
        x = x.permute(0, 2, 1) # (batchsize, width, x_dim)

        # FNO blocks
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic
        x = self.fno_blocks(x)
        x = x[..., :-self.padding] # unpad

        # Projection
        x = x.permute(0, 2, 1) # (batchsize, x_dim, width)
        x = self.q(x) # (batchsize, x_dim, out_channels)

        return x


class MambaFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels, num_fno_layers, d_state=16, d_conv=4, expand=2):
        super(MambaFNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fno_layers = num_fno_layers

        self.padding = 9 # pad the domain if input is non-periodic

        # Lifting layer
        self.p = nn.Linear(in_channels, self.width)
        
        # Mamba SSM module after lifting (needs adaptation for 2D, or sequential application)
        # For 2D, MambaSSM expects (B, L, D). We'll treat (H*W, D) as (L, D)
        self.mamba_ssm = MambaSSM(d_model=self.width, d_state=d_state, d_conv=d_conv, expand=expand)

        # FNO blocks
        self.fno_blocks = FNOBlock2d(self.width, self.modes1, self.modes2, self.num_fno_layers)

        # Projection layer
        self.q = nn.Linear(self.width, out_channels)

    def forward(self, x): # x: (batchsize, x_dim, y_dim, in_channels)
        batch_size, x_dim, y_dim, in_channels = x.shape

        # Lifting
        x = self.p(x) # (batchsize, x_dim, y_dim, width)

        # Reshape for MambaSSM: (B, H, W, D) -> (B, H*W, D)
        x_reshaped = x.view(batch_size, -1, self.width)
        
        # Apply Mamba SSM to the lifted features
        x_mamba = self.mamba_ssm(x_reshaped) # (batchsize, H*W, width)
        
        # Reshape back: (B, H*W, D) -> (B, D, H, W)
        x = x_mamba.view(batch_size, x_dim, y_dim, self.width)
        x = x.permute(0, 3, 1, 2) # (batchsize, width, x_dim, y_dim)

        # FNO blocks
        x = F.pad(x, [0, self.padding, 0, self.padding]) # pad the domain if input is non-periodic
        x = self.fno_blocks(x)
        x = x[..., :-self.padding, :-self.padding] # unpad

        # Projection
        x = x.permute(0, 2, 3, 1) # (batchsize, x_dim, y_dim, width)
        x = self.q(x) # (batchsize, x_dim, y_dim, out_channels)

        return x
