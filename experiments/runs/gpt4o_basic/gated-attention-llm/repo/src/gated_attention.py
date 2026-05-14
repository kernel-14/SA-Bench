import torch
import torch.nn as nn

class GatedAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(GatedAttention, self).__init__()
        
        self.model_dim = model_dim
        self.num_heads = num_heads

        # Linear layer for gating scores
        self.gating_score_layer = nn.Linear(model_dim, num_heads)
        
        # Sigmoid activation for gating
        self.sigmoid = nn.Sigmoid()

    def forward(self, attention_output, query):
        
        # Compute gating scores
        gating_scores = self.gating_score_layer(query)
        gating_scores = self.sigmoid(gating_scores)  # Apply sigmoid activation
        
        # Expand dimensions for element-wise gating
        gating_scores = gating_scores.unsqueeze(-1)

        # Apply gating to attention output
        gated_output = attention_output * gating_scores
        
        return gated_output

