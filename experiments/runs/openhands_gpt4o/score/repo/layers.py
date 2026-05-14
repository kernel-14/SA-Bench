import torch
import torch.nn as nn

class KLDivergencePenalty(nn.Module):
    def __init__(self, beta: float):
        super(KLDivergencePenalty, self).__init__()
        self.beta = beta

    def forward(self, logits, ref_logits):
        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits, dim=-1),
            torch.nn.functional.softmax(ref_logits, dim=-1),
            reduction="batchmean"
        )
        return self.beta * kl_loss