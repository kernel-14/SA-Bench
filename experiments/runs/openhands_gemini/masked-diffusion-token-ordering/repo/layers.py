
import torch
import torch.nn as nn

class LearnablePositionalEmbedding(nn.Module):
    """
    Learnable Positional Embedding layer.
    As mentioned in Section 3.2: "we employ a learnable positional embedding layer for all experiments to correct this."
    """
    def __init__(self, max_sequence_length: int, hidden_size: int):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_sequence_length, hidden_size)

    def forward(self, input_ids: torch.Tensor):
        """
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length).
                       The values are token IDs, but we only use its shape
                       to create position IDs.
        Returns:
            position_embeddings: Tensor of shape (batch_size, sequence_length, hidden_size)
        """
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings

