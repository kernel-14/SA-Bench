import torch
import torch.nn.functional as F

def prefix_spatial_attention(query, key, value, prefix):
    """
    Implements prefix-enhanced spatial attention.
    Args:
        query: Tensor of shape (batch_size, num_frames, spatial_dim, embed_dim)
        key: Tensor of shape (batch_size, num_frames, spatial_dim, embed_dim)
        value: Tensor of shape (batch_size, num_frames, spatial_dim, embed_dim)
        prefix: Tensor of shape (batch_size, prefix_length, spatial_dim, embed_dim)

    Returns:
        updated_values: Tensor of shape (batch_size, num_frames, spatial_dim, embed_dim)
    """
    batch_size, num_frames, spatial_dim, embed_dim = query.size()
    prefix_length = prefix.size(1)
    
    # Concatenate prefix with key and value vectors for enhancement
    prefix_repeated = prefix.repeat(1, num_frames, 1, 1)  # Repeat prefix over frames
    enhanced_key = torch.cat([prefix_repeated, key], dim=1)
    enhanced_value = torch.cat([prefix_repeated, value], dim=1)

    attention_scores = torch.matmul(query.view(batch_size, -1, embed_dim), enhanced_key.transpose(-2, -1)) / torch.sqrt(embed_dim)
    attention_probs = F.softmax(attention_scores, dim=-1)
    updated_values = torch.matmul(attention_probs, enhanced_value.view(batch_size, -1, embed_dim))
    
    return updated_values.view(batch_size, num_frames, spatial_dim, embed_dim)
