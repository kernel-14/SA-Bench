
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# Helper for freezing parameters
def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

# --- Prompt-based Methods ---
class VPTPrompt(nn.Module):
    def __init__(self, prompt_number, embed_dim, deep_vpt=False):
        super().__init__()
        self.prompt_number = prompt_number
        self.embed_dim = embed_dim
        self.deep_vpt = deep_vpt
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, prompt_number, embed_dim))
        nn.init.xavier_uniform_(self.prompt_embeddings)

    def forward(self, x):
        # x is B, N, D where N includes the class token
        # For shallow VPT, prompts are prepended to the first layer's input.
        # For deep VPT, prompts are prepended to each layer's input and discarded at the end of the layer.
        # This module will be used by the main model to inject prompts.
        return self.prompt_embeddings.expand(x.shape[0], -1, -1)

# --- Adapter-based Methods ---
class Adapter(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim, adapter_scale_factor=1.0):
        super().__init__()
        self.down_project = nn.Linear(embed_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(bottleneck_dim, embed_dim)
        self.scale = adapter_scale_factor

        nn.init.kaiming_uniform_(self.down_project.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x * self.scale + residual

class ConvPass(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim, kernel_size=3, convpass_scale_factor=1.0, xavier_init=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv_down = nn.Conv2d(embed_dim, bottleneck_dim, kernel_size=1, stride=1, padding=0)
        self.conv_middle = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=kernel_size, stride=1, padding=padding, groups=bottleneck_dim)
        self.conv_up = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.activation = nn.GELU()
        self.scale = convpass_scale_factor

        if xavier_init:
            nn.init.xavier_uniform_(self.conv_down.weight)
            nn.init.xavier_uniform_(self.conv_middle.weight)
            nn.init.xavier_uniform_(self.conv_up.weight)
        else:
            nn.init.kaiming_uniform_(self.conv_down.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.conv_middle.weight, a=math.sqrt(5))
            nn.init.zeros_(self.conv_up.weight) # Initialize up-projection to zero

        nn.init.zeros_(self.conv_down.bias)
        nn.init.zeros_(self.conv_middle.bias)
        nn.init.zeros_(self.conv_up.bias)

    def forward(self, x):
        # x shape: (B, N, D) where N is num_tokens (including CLS), D is embed_dim
        # ConvPass expects 2D image-like input. Assuming patches are arranged in a grid.
        # This requires careful handling of CLS token and grid reconstruction.
        # For simplicity, we'll process the tokens excluding CLS, assuming square patches.
        
        # Original paper details: "performing 2D convolution over nearby patch tokens"
        # This implies we should reconstruct the spatial layout.
        
        # Temporarily detach CLS token if present
        cls_token, patch_tokens = x[:, :1], x[:, 1:] if x.shape[1] > 1 else (None, x)

        B, N_patches, D = patch_tokens.shape
        # Assuming square image patches for reconstruction
        H_P = W_P = int(N_patches**0.5)
        
        if H_P * W_P != N_patches:
            raise ValueError("Number of patch tokens is not a perfect square, cannot apply 2D convolution.")

        # Reshape to (B, D, H_P, W_P) for Conv2d
        y = rearrange(patch_tokens, 'b (h w) d -> b d h w', h=H_P, w=W_P)

        y = self.conv_down(y)
        y = self.activation(y)
        y = self.conv_middle(y)
        y = self.activation(y)
        y = self.conv_up(y)

        # Reshape back to (B, N_patches, D)
        y = rearrange(y, 'b d h w -> b (h w) d')

        # Re-attach CLS token
        if cls_token is not None:
            y = torch.cat((cls_token, y), dim=1)
        
        # ConvPass applies in parallel and adds to residual
        return y * self.scale + x


class RepAdapter(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim, groups, repadapter_scale_factor=1.0):
        super().__init__()
        # down_project: D -> bottleneck_dim
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.down_project = nn.Linear(embed_dim, bottleneck_dim)
        # up_project: bottleneck_dim -> D using group-wise transformation
        self.group_wise_weights = nn.Parameter(torch.Tensor(groups, bottleneck_dim // groups, embed_dim // groups))
        self.groups = groups
        self.scale = repadapter_scale_factor

        nn.init.kaiming_uniform_(self.down_project.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.group_wise_weights, a=math.sqrt(5))
        nn.init.zeros_(self.down_project.bias)

    def forward(self, x):
        residual = x
        # x: (B, N, D)
        h = self.down_project(x) # h: (B, N, bottleneck_dim)
        
        # Apply group-wise transformation
        # Reshape h for group-wise multiplication: (B, N, groups, bottleneck_dim // groups)
        h_reshaped = h.view(h.shape[0], h.shape[1], self.groups, h.shape[2] // self.groups)
        
        # Expand group_wise_weights: (1, 1, groups, D // groups, bottleneck_dim // groups)
        # Result: (B, N, groups, D // groups)
        # Permute to (B, N, D // groups, groups) then reshape to (B, N, D)
        
        # To perform batched matrix multiplication, we need to adjust dimensions
        # group_wise_weights: (groups, D_per_group, bottleneck_dim_per_group)
        # h_reshaped: (B*N, groups, bottleneck_dim_per_group)
        
        B, N, _ = x.shape
        h_grouped = h.view(B * N, self.groups, self.bottleneck_dim // self.groups)
        
        # Perform batched matrix multiplication
        # (B*N, groups, D_per_group)
        up_h = torch.einsum('bgi,gio->bgo', h_grouped, self.group_wise_weights)
        
        y = up_h.view(B, N, self.embed_dim)

        return y * self.scale + residual

# --- Efficient Selective Tuning ---
class LoRAModule(nn.Module):
    def __init__(self, in_features, out_features, rank, lora_alpha=1.0):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = lora_alpha / rank # Original LoRA scaling

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        # Instead of directly modifying nn.Linear weights,
        # we compute the low-rank update and add it to the original output.
        # This module replaces the weight update logic.
        # For inference efficiency, these can be merged into original weights: W_new = W_orig + (lora_A @ lora_B) * scale
        return (x @ self.lora_A @ self.lora_B) * self.scale

class FacTT(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim, scale_factor=1.0):
        super().__init__()
        # Simplified FacT_TT as described in paper, assuming
        # U, V are D x r and Sigma is 12L x r x r
        # This implementation requires more context about which specific weights to apply this to.
        # For a general module, this is tricky. The paper applies it to W_QKV/O, W1/W2.
        # For simplicity, we'll create a placeholder for the concept.
        # This means applying low-rank updates to weight matrices, similar to LoRA but with tensor-train decomposition.
        # This specific implementation will need more context about the target weight matrix dimensions.
        self.scale = scale_factor
        self.U = nn.Parameter(torch.randn(embed_dim, bottleneck_dim))
        self.V = nn.Parameter(torch.randn(embed_dim, bottleneck_dim))
        # Sigma would connect U and V, forming a low-rank matrix Delta_W = U @ Sigma @ V.T
        # But Sigma is a 3D tensor (12L x r x r) in the paper, implying multiple weight matrices.
        # For a single application (e.g., to a linear layer's weight), it would be simpler:
        self.sigma = nn.Parameter(torch.randn(bottleneck_dim, bottleneck_dim))
        
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.sigma)

    def forward(self, x):
        # This module conceptually represents Delta_W.
        # In practice, it would modify a specific weight matrix W
        # x would be the input feature to the layer that uses W
        # The output of this module would be x @ Delta_W
        delta_W = (self.U @ self.sigma @ self.V.T) * self.scale
        return x @ delta_W

class FacTK(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim, scale_factor=1.0):
        super().__init__()
        self.scale = scale_factor
        # Simplified FacT_TK: Delta_W = B @ U @ V.T, where A is the core tensor
        # This is also a placeholder due to the complexity of full tensor decomposition.
        # It would similarly approximate Delta_W for a target weight matrix.
        self.U = nn.Parameter(torch.randn(embed_dim, bottleneck_dim))
        self.V = nn.Parameter(torch.randn(embed_dim, bottleneck_dim))
        self.B = nn.Parameter(torch.randn(bottleneck_dim, bottleneck_dim)) # Represents one mode of core tensor A
        
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.B)

    def forward(self, x):
        delta_W = (self.U @ self.B @ self.V.T) * self.scale
        return x @ delta_W


class SSF(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        return x * self.scale + self.shift


# --- PEFT specific modifications to ViT blocks ---

def _inject_vpt_prompts(config, model):
    if config.peft_method == 'vpt_shallow':
        prompt_model = VPTPrompt(config.vpt_prompt_number, model.embed_dim, deep_vpt=False)
        model.prompt_model = prompt_model
        # Modify the forward pass of the first encoder layer to prepend prompts
        # This is a conceptual modification. Actual implementation needs to override
        # or hook into the ViT's forward to insert prompts at Z_0
        print(f"VPT-Shallow: {config.vpt_prompt_number} prompts injected.")
    elif config.peft_method == 'vpt_deep':
        prompt_models = nn.ModuleList([VPTPrompt(config.vpt_prompt_number, model.embed_dim, deep_vpt=True) for _ in range(model.depth)])
        model.prompt_models = prompt_models
        # Modify forward pass of each encoder layer to prepend prompts
        # This also requires overriding or hooking into ViT's forward at each layer
        print(f"VPT-Deep: {config.vpt_prompt_number} prompts injected into each of {model.depth} layers.")
    else:
        raise ValueError(f"Unknown VPT method: {config.peft_method}")

    # Freeze all other parameters
    freeze_parameters(model)
    for param in model.prompt_model.parameters() if hasattr(model, 'prompt_model') else model.prompt_models.parameters():
        param.requires_grad = True
    print("VPT: All backbone parameters frozen, only prompts are trainable.")


def _inject_adapter_modules(config, model):
    for i, block in enumerate(model.blocks):
        if config.peft_method == 'houl_adapter':
            # Houl. Adapter after MSA and MLP
            block.attn.adapter = Adapter(model.embed_dim, config.adapter_bottleneck, config.adapter_scale_factor)
            block.mlp.adapter = Adapter(model.embed_dim, config.adapter_bottleneck, config.adapter_scale_factor)
        elif config.peft_method == 'pfeif_adapter':
            # Pfeif. Adapter after MLP
            block.mlp.adapter = Adapter(model.embed_dim, config.adapter_bottleneck, config.adapter_scale_factor)
        elif config.peft_method == 'adaptformer':
            # AdaptFormer in parallel with MLP
            block.mlp.adaptformer = Adapter(model.embed_dim, config.adapter_bottleneck, config.adapter_scale_factor)
        elif config.peft_method == 'convpass':
            # ConvPass in parallel with MSA and MLP
            # This requires careful integration as ConvPass expects 2D structure
            # For simplicity, we directly add it to the block for now,
            # but its forward method needs to handle token-to-image reshape
            block.attn.convpass = ConvPass(model.embed_dim, config.adapter_bottleneck, config.convpass_kernel_size, config.adapter_scale_factor)
            block.mlp.convpass = ConvPass(model.embed_dim, config.adapter_bottleneck, config.convpass_kernel_size, config.adapter_scale_factor)
        elif config.peft_method == 'repadapter':
            # RepAdapter sequentially after MSA and MLP
            block.attn.repadapter = RepAdapter(model.embed_dim, config.adapter_bottleneck, config.repadapter_groups, config.adapter_scale_factor)
            block.mlp.repadapter = RepAdapter(model.embed_dim, config.adapter_bottleneck, config.repadapter_groups, config.adapter_scale_factor)
        else:
            raise ValueError(f"Unknown Adapter method: {config.peft_method}")
    
    freeze_parameters(model)
    # Enable gradients for adapter parameters
    for name, param in model.named_parameters():
        if "adapter" in name or "convpass" in name or "repadapter" in name:
            param.requires_grad = True
    print(f"Adapter method '{config.peft_method}': All backbone parameters frozen, only adapter parameters are trainable.")


def _selective_tune_parameters(config, model):
    if config.peft_method == 'bitfit':
        freeze_parameters(model)
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
        print("BitFit: Only bias parameters are trainable.")
    elif config.peft_method == 'layernorm':
        freeze_parameters(model)
        for name, param in model.named_parameters():
            if "norm" in name: # Matches LayerNorm layers
                param.requires_grad = True
        print("LayerNorm: Only LayerNorm parameters are trainable.")
    elif config.peft_method == 'difffit':
        freeze_parameters(model)
        for name, param in model.named_parameters():
            if "bias" in name or "norm" in name:
                param.requires_grad = True
        # DiffFit also inserts learnable scale factors (gamma) after MSA and MLP
        # This requires modifying the Transformer blocks to add these factors.
        # For now, we'll just implement the bias and LayerNorm part.
        # Adding gamma factors would require hooking into each Transformer block.
        # Example conceptual addition:
        for i, block in enumerate(model.blocks):
            block.gamma1 = nn.Parameter(torch.ones(model.embed_dim))
            block.gamma2 = nn.Parameter(torch.ones(model.embed_dim))
            block.gamma1.requires_grad = True
            block.gamma2.requires_grad = True
        print("DiffFit: Bias, LayerNorm, and custom gamma parameters are trainable.")
    elif config.peft_method == 'ssf':
        # SSF modulates features with scale and shift factors
        # This requires injecting SSF modules at various points.
        freeze_parameters(model)
        for i, block in enumerate(model.blocks):
            # Example: inject SSF at input to MSA, output of MSA, input to MLP, output of MLP
            block.ssf_attn_in = SSF(model.embed_dim)
            block.ssf_attn_out = SSF(model.embed_dim)
            block.ssf_mlp_in = SSF(model.embed_dim)
            block.ssf_mlp_out = SSF(model.embed_dim)
            block.ssf_attn_in.requires_grad = True
            block.ssf_attn_out.requires_grad = True
            block.ssf_mlp_in.requires_grad = True
            block.ssf_mlp_out.requires_grad = True
        print("SSF: Scale and shift factors are trainable.")
    else:
        raise ValueError(f"Unknown Direct Selective Tuning method: {config.peft_method}")


def _inject_lora_modules(config, model):
    # LoRA applies to Q/V projection weights in MSA
    for i, block in enumerate(model.blocks):
        # Original Linear layers: W_Q, W_K, W_V in model.blocks[i].attn.qkv
        # W_O in model.blocks[i].attn.proj
        
        # We need to replace or wrap the Linear layers to inject LoRA.
        # This typically involves replacing the nn.Linear with a custom LoRA-enabled Linear.
        # For demonstration, we'll just add LoRA modules and assume forward pass is modified.
        
        # Assume qkv is a single linear layer that produces Q, K, V
        # QKV weights: model.blocks[i].attn.qkv (embed_dim, embed_dim * 3)
        # Proj weights: model.blocks[i].attn.proj (embed_dim, embed_dim)
        
        # For simplicity, let's inject LoRA to Q and V projections.
        # A more robust implementation would replace the Linear layer with a LoRALinear layer.
        
        block.attn.lora_q = LoRAModule(model.embed_dim, model.embed_dim, config.lora_rank)
        block.attn.lora_v = LoRAModule(model.embed_dim, model.embed_dim, config.lora_rank)
    
    freeze_parameters(model)
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    print(f"LoRA: All backbone parameters frozen, only LoRA modules are trainable (rank={config.lora_rank}).")


def _inject_fact_modules(config, model):
    # FacT applies tensor decomposition to a collection of weight matrices
    # This is a complex modification. For now, we'll assume it modifies weights directly.
    # Similar to LoRA, this would typically involve custom Linear layers.
    
    for i, block in enumerate(model.blocks):
        if config.peft_method == 'fact_tt':
            block.fact_tt_attn_qkv = FacTT(model.embed_dim * 3, config.fact_bottleneck, config.adapter_scale_factor)
            block.fact_tt_attn_proj = FacTT(model.embed_dim, config.fact_bottleneck, config.adapter_scale_factor)
            block.fact_tt_mlp_fc1 = FacTT(model.embed_dim * 4, config.fact_bottleneck, config.adapter_scale_factor)
            block.fact_tt_mlp_fc2 = FacTT(model.embed_dim, config.fact_bottleneck, config.adapter_scale_factor)
        elif config.peft_method == 'fact_tk':
            block.fact_tk_attn_qkv = FacTK(model.embed_dim * 3, config.fact_bottleneck, config.adapter_scale_factor)
            block.fact_tk_attn_proj = FacTK(model.embed_dim, config.fact_bottleneck, config.adapter_scale_factor)
            block.fact_tk_mlp_fc1 = FacTK(model.embed_dim * 4, config.fact_bottleneck, config.adapter_scale_factor)
            block.fact_tk_mlp_fc2 = FacTK(model.embed_dim, config.fact_bottleneck, config.adapter_scale_factor)
        else:
            raise ValueError(f"Unknown FacT method: {config.peft_method}")
            
    freeze_parameters(model)
    for name, param in model.named_parameters():
        if "fact" in name:
            param.requires_grad = True
    print(f"FacT method '{config.peft_method}': All backbone parameters frozen, only FacT modules are trainable.")



