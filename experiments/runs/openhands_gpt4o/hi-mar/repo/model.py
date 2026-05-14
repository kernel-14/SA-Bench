import torch
import torch.nn as nn
from modules import ScaleAwareTransformer, DiffusionTransformerHead

class HiMAR(nn.Module):
    def __init__(self, config):
        super(HiMAR, self).__init__()
        self.low_res_transformer = ScaleAwareTransformer(config['low_res'])
        self.high_res_transformer = ScaleAwareTransformer(config['high_res'])
        self.low_res_diffusion_head = DiffusionTransformerHead(config['low_res_diffusion'])
        self.high_res_diffusion_head = DiffusionTransformerHead(config['high_res_diffusion'])

    def forward(self, low_res_tokens, high_res_tokens, context_tokens):
        # Phase 1: Low-resolution token prediction
        low_res_conditional = self.low_res_transformer(low_res_tokens, context_tokens)
        low_res_output = self.low_res_diffusion_head(low_res_conditional)

        # Phase 2: High-resolution token prediction
        high_res_conditional = self.high_res_transformer(high_res_tokens, low_res_conditional, context_tokens)
        high_res_output = self.high_res_diffusion_head(high_res_conditional)

        return low_res_output, high_res_output