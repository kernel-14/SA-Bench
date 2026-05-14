
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Block, Attention, Mlp
from timm.models.layers import DropPath
from functools import partial
import copy

from models.modules import (
    VPTPrompt, Adapter, ConvPass, RepAdapter, LoRAModule, FacTT, FacTK, SSF,
    freeze_parameters
)

class PEFTAttention(Attention):
    def __init__(self, original_attention, peft_method=None, peft_args=None):
        super().__init__(
            dim=original_attention.qkv.in_features,
            num_heads=original_attention.num_heads,
            qkv_bias=original_attention.qkv.bias is not None,
            attn_drop=original_attention.attn_drop,
            proj_drop=original_attention.proj_drop
        )
        self.qkv = original_attention.qkv
        self.proj = original_attention.proj
        self.attn_drop = original_attention.attn_drop
        self.proj_drop = original_attention.proj_drop
        self.num_heads = original_attention.num_heads
        self.head_dim = original_attention.head_dim
        self.scale = original_attention.scale
        
        self.peft_method = peft_method
        
        if peft_method == 'houl_adapter':
            self.adapter = Adapter(self.qkv.in_features, peft_args['bottleneck_dim'], peft_args['scale_factor'])
        elif peft_method == 'convpass':
            self.convpass = ConvPass(self.qkv.in_features, peft_args['bottleneck_dim'], peft_args['kernel_size'], peft_args['scale_factor'])
        elif peft_method == 'repadapter':
            self.repadapter = RepAdapter(self.qkv.in_features, peft_args['bottleneck_dim'], peft_args['groups'], peft_args['scale_factor'])
        elif peft_method == 'lora':
            # LoRA for Q and V projections
            self.lora_q = LoRAModule(self.qkv.in_features, self.qkv.in_features // 3, peft_args['rank'])
            self.lora_v = LoRAModule(self.qkv.in_features, self.qkv.in_features // 3, peft_args['rank'])
        elif peft_method == 'fact_tt':
            self.fact_tt_qkv = FacTT(self.qkv.in_features * 3, peft_args['bottleneck_dim'], peft_args['scale_factor'])
        elif peft_method == 'fact_tk':
            self.fact_tk_qkv = FacTK(self.qkv.in_features * 3, peft_args['bottleneck_dim'], peft_args['scale_factor'])

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.peft_method == 'lora':
            # Apply LoRA to Q and V
            q = q + self.lora_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v + self.lora_v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        elif self.peft_method in ['fact_tt', 'fact_tk']:
            # For FacT, assume we are modifying the qkv output directly by adding a residual
            # This is a simplification; actual FacT implementation would modify the weights themselves.
            # Here, we model it as an additive residual to the combined QKV output.
            if self.peft_method == 'fact_tt':
                qkv_residual = self.fact_tt_qkv(x.reshape(B, -1, C)) # Simplified
            else: # fact_tk
                qkv_residual = self.fact_tk_qkv(x.reshape(B, -1, C)) # Simplified
            
            # Reshape qkv_residual to match qkv and add
            qkv_residual = qkv_residual.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0] + qkv_residual[0], qkv[1] + qkv_residual[1], qkv[2] + qkv_residual[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if self.peft_method == 'houl_adapter':
            x = self.adapter(x)
        elif self.peft_method == 'convpass':
            x = self.convpass(x)
        elif self.peft_method == 'repadapter':
            x = self.repadapter(x)

        return x

class PEFTMlp(Mlp):
    def __init__(self, original_mlp, peft_method=None, peft_args=None):
        super().__init__(
            in_features=original_mlp.fc1.in_features,
            hidden_features=original_mlp.fc2.in_features,
            act_layer=original_mlp.act,
            drop=original_mlp.drop
        )
        self.fc1 = original_mlp.fc1
        self.fc2 = original_mlp.fc2
        self.act = original_mlp.act
        self.drop = original_mlp.drop
        
        self.peft_method = peft_method

        if peft_method in ['houl_adapter', 'pfeif_adapter']:
            self.adapter = Adapter(self.fc1.in_features, peft_args['bottleneck_dim'], peft_args['scale_factor'])
        elif peft_method == 'adaptformer':
            self.adaptformer = Adapter(self.fc1.in_features, peft_args['bottleneck_dim'], peft_args['scale_factor'])
        elif peft_method == 'convpass':
            self.convpass = ConvPass(self.fc1.in_features, peft_args['bottleneck_dim'], peft_args['kernel_size'], peft_args['scale_factor'])
        elif peft_method == 'repadapter':
            self.repadapter = RepAdapter(self.fc1.in_features, peft_args['bottleneck_dim'], peft_args['groups'], peft_args['scale_factor'])
        elif peft_method == 'fact_tt':
            self.fact_tt_fc1 = FacTT(self.fc1.in_features, peft_args['bottleneck_dim'], peft_args['scale_factor'])
            self.fact_tt_fc2 = FacTT(self.fc2.in_features, peft_args['bottleneck_dim'], peft_args['scale_factor'])
        elif peft_method == 'fact_tk':
            self.fact_tk_fc1 = FacTK(self.fc1.in_features, peft_args['bottleneck_dim'], peft_args['scale_factor'])
            self.fact_tk_fc2 = FacTK(self.fc2.in_features, peft_args['bottleneck_dim'], peft_args['scale_factor'])
        
    def forward(self, x):
        x = self.fc1(x)
        if self.peft_method in ['fact_tt', 'fact_tk']:
            if self.peft_method == 'fact_tt':
                x = x + self.fact_tt_fc1(x)
            else: # fact_tk
                x = x + self.fact_tk_fc1(x)
        x = self.act(x)
        x = self.drop(x)
        
        if self.peft_method == 'adaptformer':
            x = x + self.adaptformer(x) # AdaptFormer runs parallel to MLP
            
        x = self.fc2(x)
        if self.peft_method in ['fact_tt', 'fact_tk']:
            if self.peft_method == 'fact_tt':
                x = x + self.fact_tt_fc2(x)
            else: # fact_tk
                x = x + self.fact_tk_fc2(x)
        x = self.drop(x)
        
        if self.peft_method in ['houl_adapter', 'pfeif_adapter']:
            x = self.adapter(x)
        elif self.peft_method == 'convpass':
            x = self.convpass(x)
        elif self.peft_method == 'repadapter':
            x = self.repadapter(x)
        
        return x

class PEFTBlock(Block):
    def __init__(self, original_block, peft_method=None, peft_args=None):
        super().__init__(
            dim=original_block.norm1.normalized_shape[0],
            num_heads=original_block.attn.num_heads,
            mlp_ratio=original_block.mlp.fc2.out_features / original_block.norm1.normalized_shape[0], # Approx
            qkv_bias=original_block.attn.qkv.bias is not None,
            drop_path=original_block.drop_path.drop_prob,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), # Original was nn.LayerNorm
            act_layer=original_block.mlp.act # Original was nn.GELU
        )
        # Copy original components
        self.norm1 = original_block.norm1
        self.attn = PEFTAttention(original_block.attn, peft_method, peft_args)
        self.drop_path = original_block.drop_path
        self.norm2 = original_block.norm2
        self.mlp = PEFTMlp(original_block.mlp, peft_method, peft_args)
        
        self.peft_method = peft_method
        self.peft_args = peft_args

        if peft_method == 'difffit':
            self.gamma1 = nn.Parameter(torch.ones(self.dim))
            self.gamma2 = nn.Parameter(torch.ones(self.dim))
        elif peft_method == 'ssf':
            self.ssf_attn_in = SSF(self.dim)
            self.ssf_attn_out = SSF(self.dim)
            self.ssf_mlp_in = SSF(self.dim)
            self.ssf_mlp_out = SSF(self.dim)

    def forward(self, x, deep_vpt_prompts=None):
        # Original Block forward pass
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        # PEFT modified forward pass
        residual = x
        
        if self.peft_method == 'vpt_deep' and deep_vpt_prompts is not None:
            x = torch.cat((deep_vpt_prompts, x), dim=1) # Prepend prompts

        if self.peft_method == 'ssf':
            x_attn = self.ssf_attn_in(self.norm1(x))
        else:
            x_attn = self.norm1(x)

        x_attn = self.attn(x_attn)
        
        if self.peft_method == 'difffit':
            x_attn = x_attn * self.gamma1
        elif self.peft_method == 'ssf':
            x_attn = self.ssf_attn_out(x_attn)
        
        x = residual + self.drop_path(x_attn)

        residual = x
        if self.peft_method == 'ssf':
            x_mlp = self.ssf_mlp_in(self.norm2(x))
        else:
            x_mlp = self.norm2(x)
        
        x_mlp = self.mlp(x_mlp)

        if self.peft_method == 'difffit':
            x_mlp = x_mlp * self.gamma2
        elif self.peft_method == 'ssf':
            x_mlp = self.ssf_mlp_out(x_mlp)

        x = residual + self.drop_path(x_mlp)
        
        if self.peft_method == 'vpt_deep' and deep_vpt_prompts is not None:
            # Remove prompts before returning, as they are layer-specific and discarded.
            # This assumes the class token and patch tokens are separate.
            # If the class token is part of the 'x' after attention, need to be careful.
            # Original ViT forward passes the cls_token along with patch_tokens.
            # If prompts are prepended, they become part of the sequence.
            # The paper says: "discards their output at the end of the layer".
            # This implies the original sequence should be returned.
            return x[:, deep_vpt_prompts.shape[1]:, :] # Remove prompts
        
        return x


class PEFTVisionTransformer(VisionTransformer):
    def __init__(self, original_vit_model, peft_method=None, peft_args=None):
        super().__init__(
            img_size=original_vit_model.img_size,
            patch_size=original_vit_model.patch_size,
            in_chans=original_vit_model.in_chans,
            num_classes=original_vit_model.num_classes,
            global_pool=original_vit_model.global_pool,
            embed_dim=original_vit_model.embed_dim,
            depth=original_vit_model.depth,
            num_heads=original_vit_model.num_heads,
            mlp_ratio=original_vit_model.mlp_ratio,
            qkv_bias=original_vit_model.qkv_bias,
            norm_layer=original_vit_model.norm_layer,
            act_layer=original_vit_model.act_layer,
            drop_rate=original_vit_model.drop_rate,
            attn_drop_rate=original_vit_model.attn_drop_rate,
            drop_path_rate=original_vit_model.drop_path_rate,
        )

        self.patch_embed = original_vit_model.patch_embed
        self.cls_token = original_vit_model.cls_token
        self.pos_embed = original_vit_model.pos_embed
        self.pos_drop = original_vit_model.pos_drop
        self.norm = original_vit_model.norm
        self.head = original_vit_model.head

        self.peft_method = peft_method
        self.peft_args = peft_args
        
        # Replace original blocks with PEFT-aware blocks
        self.blocks = nn.ModuleList([
            PEFTBlock(original_vit_model.blocks[i], peft_method, peft_args)
            for i in range(original_vit_model.depth)
        ])

        if peft_method == 'vpt_shallow':
            self.prompt_model = VPTPrompt(peft_args['prompt_number'], self.embed_dim, deep_vpt=False)
        elif peft_method == 'vpt_deep':
            self.prompt_models = nn.ModuleList([VPTPrompt(peft_args['prompt_number'], self.embed_dim, deep_vpt=True) for _ in range(self.depth)])
        elif peft_method == 'bitfit':
            # BitFit tunes biases. This means setting requires_grad=True for all biases.
            # The actual forward pass is mostly unchanged, but we need to ensure biases are tunable.
            # This is handled by `freeze_parameters` and then setting specific `requires_grad=True`.
            pass # Handled by apply_peft later
        elif peft_method == 'layernorm':
            pass # Handled by apply_peft later
        elif peft_method == 'difffit':
            # DiffFit tunes biases, LN, and adds gamma factors.
            # Gamma factors are in PEFTBlock. Biases and LN are handled by apply_peft.
            pass
        elif peft_method == 'ssf':
            # SSF modules are already in PEFTBlock
            pass
        
        self.apply_peft_policy()

    def apply_peft_policy(self):
        if self.peft_method == 'linear_probe':
            freeze_parameters(self)
            for param in self.head.parameters():
                param.requires_grad = True
        elif self.peft_method == 'full_finetune':
            for param in self.parameters():
                param.requires_grad = True
        elif self.peft_method in ['vpt_shallow', 'vpt_deep']:
            freeze_parameters(self)
            if self.peft_method == 'vpt_shallow':
                for param in self.prompt_model.parameters():
                    param.requires_grad = True
            else: # vpt_deep
                for prompt_module in self.prompt_models:
                    for param in prompt_module.parameters():
                        param.requires_grad = True
            for param in self.head.parameters(): # Head is always finetuned
                param.requires_grad = True
        elif self.peft_method in ['houl_adapter', 'pfeif_adapter', 'adaptformer', 'convpass', 'repadapter']:
            freeze_parameters(self)
            for name, param in self.named_parameters():
                if any(peft_tag in name for peft_tag in ['adapter', 'convpass', 'repadapter']):
                    param.requires_grad = True
            for param in self.head.parameters(): # Head is always finetuned
                param.requires_grad = True
        elif self.peft_method in ['bitfit', 'layernorm', 'difffit', 'ssf']:
            freeze_parameters(self)
            for name, param in self.named_parameters():
                if self.peft_method == 'bitfit' and "bias" in name:
                    param.requires_grad = True
                elif self.peft_method == 'layernorm' and "norm" in name:
                    param.requires_grad = True
                elif self.peft_method == 'difffit' and ("bias" in name or "norm" in name or "gamma" in name):
                    param.requires_grad = True
                elif self.peft_method == 'ssf' and "ssf" in name:
                    param.requires_grad = True
            for param in self.head.parameters(): # Head is always finetuned
                param.requires_grad = True
        elif self.peft_method == 'lora':
            freeze_parameters(self)
            for name, param in self.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
            for param in self.head.parameters(): # Head is always finetuned
                param.requires_grad = True
        elif self.peft_method in ['fact_tt', 'fact_tk']:
            freeze_parameters(self)
            for name, param in self.named_parameters():
                if "fact_tt" in name or "fact_tk" in name:
                    param.requires_grad = True
            for param in self.head.parameters(): # Head is always finetuned
                param.requires_grad = True
        else:
            raise ValueError(f"Unsupported PEFT method: {self.peft_method}")

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"PEFT method: {self.peft_method}")
        print(f"Trainable parameters: {trainable_params} ({100 * trainable_params / total_params:.2f}%)")
        print(f"Total parameters: {total_params}")

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from dulyapha a timm
        if self.peft_method == 'vpt_shallow':
            # Prepend shallow prompts to the input sequence
            prompts = self.prompt_model(x)
            x = torch.cat((cls_token, prompts, x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        for i, block in enumerate(self.blocks):
            if self.peft_method == 'vpt_deep':
                deep_vpt_prompts = self.prompt_models[i](x)
                x = block(x, deep_vpt_prompts=deep_vpt_prompts)
            else:
                x = block(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def build_peft_model(config, original_vit_model):
    peft_args = {}
    if config.peft_method in ['vpt_shallow', 'vpt_deep']:
        peft_args['prompt_number'] = config.vpt_prompt_number
    elif config.peft_method in ['houl_adapter', 'pfeif_adapter', 'adaptformer', 'convpass', 'repadapter']:
        peft_args['bottleneck_dim'] = config.adapter_bottleneck
        peft_args['scale_factor'] = config.adapter_scale_factor
        if config.peft_method == 'convpass':
            peft_args['kernel_size'] = config.convpass_kernel_size
        elif config.peft_method == 'repadapter':
            peft_args['groups'] = config.repadapter_groups
    elif config.peft_method == 'lora':
        peft_args['rank'] = config.lora_rank
    elif config.peft_method in ['fact_tt', 'fact_tk']:
        peft_args['bottleneck_dim'] = config.fact_bottleneck
        peft_args['scale_factor'] = config.adapter_scale_factor # Assuming adapter_scale_factor for Fact

    peft_model = PEFTVisionTransformer(original_vit_model, config.peft_method, peft_args)
    return peft_model

