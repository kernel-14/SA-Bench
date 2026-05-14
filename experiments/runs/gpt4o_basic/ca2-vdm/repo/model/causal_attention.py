import torch
import torch.nn.functional as F

def causal_attention(query, key, value):
    """
    Implements causal attention where each frame attends only to its preceding frames.
    Args:
        query: Tensor of shape (batch_size, num_frames, embed_dim)
        key: Tensor of shape (batch_size, num_frames, embed_dim)
        value: Tensor of shape (batch_size, num_frames, embed_dim)

    Returns:
        updated_values: Tensor of shape (batch_size, num_frames, embed_dim)
    """
    batch_size, num_frames, embed_dim = query.size()
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(embed_dim)
    
    # Creating the causal mask
    causal_mask = torch.tril(torch.ones(num_frames, num_frames)).to(query.device)
    attention_scores = attention_scores.masked_fill(causal_mask == 0, float("-inf"))
    
    attention_probs = F.softmax(attention_scores, dim=-1)
    updated_values = torch.matmul(attention_probs, value)
    
    return updated_values
