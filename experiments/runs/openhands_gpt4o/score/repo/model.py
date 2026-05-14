import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

class SCoReModel(nn.Module):
    def __init__(self, base_model_name: str, kl_beta: float):
        super(SCoReModel, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.kl_beta = kl_beta

    def forward(self, input_ids, attention_mask, labels=None, ref_logits=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        loss = None
        if labels is not None:
            ce_loss = outputs.loss
            kl_loss = 0
            if ref_logits is not None:
                kl_loss = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(ref_logits, dim=-1),
                    reduction="batchmean"
                )
            loss = ce_loss + self.kl_beta * kl_loss

        return logits, loss

    def generate(self, input_ids, attention_mask, **generate_kwargs):
        return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)