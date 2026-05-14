"""
Patch n' Pack implementation for length-balanced training batches.

Based on NaViT (Dehghani et al., 2023): "Patch n' Pack: NaViT, a vision transformer
for any aspect ratio and resolution."

This allows packing training samples with varying token counts together to form
length-balanced training batches, as described in Section 3.4 of the paper.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class PatchNPackBatch:
    """
    Represents a packed batch of sequences with varying lengths.
    
    Sequences are packed together to fill a target sequence length,
    with attention masks to prevent cross-sequence attention.
    """
    
    def __init__(
        self,
        sequences: List[torch.Tensor],
        target_length: int,
        pad_value: float = 0.0,
    ):
        """
        Args:
            sequences: List of tensors with varying lengths (L_i, D)
            target_length: Target packed sequence length
            pad_value: Value to use for padding
        """
        self.sequences = sequences
        self.target_length = target_length
        self.pad_value = pad_value
        
        # Pack sequences
        self.packed_tokens, self.attention_mask, self.sequence_ids = self._pack()
    
    def _pack(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pack sequences into a single tensor with attention masks.
        
        Returns:
            Tuple of (packed_tokens, attention_mask, sequence_ids)
        """
        D = self.sequences[0].shape[-1]
        
        # Sort sequences by length (descending) for better packing
        sorted_seqs = sorted(self.sequences, key=lambda x: x.shape[0], reverse=True)
        
        # Greedy bin packing
        bins = []  # Each bin is a list of (sequence, start_pos)
        bin_lengths = []
        
        for seq in sorted_seqs:
            seq_len = seq.shape[0]
            
            # Find a bin with enough space
            placed = False
            for i, bin_len in enumerate(bin_lengths):
                if bin_len + seq_len <= self.target_length:
                    bins[i].append((seq, bin_len))
                    bin_lengths[i] += seq_len
                    placed = True
                    break
            
            if not placed:
                # Create new bin
                bins.append([(seq, 0)])
                bin_lengths.append(seq_len)
        
        # Create packed tensors for each bin
        packed_list = []
        mask_list = []
        ids_list = []
        
        for bin_idx, (bin_seqs, bin_len) in enumerate(zip(bins, bin_lengths)):
            # Create packed sequence
            packed = torch.zeros(self.target_length, D)
            mask = torch.zeros(self.target_length, self.target_length, dtype=torch.bool)
            seq_ids = torch.full((self.target_length,), -1, dtype=torch.long)
            
            for seq_idx, (seq, start_pos) in enumerate(bin_seqs):
                end_pos = start_pos + seq.shape[0]
                packed[start_pos:end_pos] = seq
                
                # Allow attention within the same sequence
                mask[start_pos:end_pos, start_pos:end_pos] = True
                seq_ids[start_pos:end_pos] = seq_idx
            
            packed_list.append(packed)
            mask_list.append(mask)
            ids_list.append(seq_ids)
        
        packed_tokens = torch.stack(packed_list)  # (num_bins, target_length, D)
        attention_mask = torch.stack(mask_list)   # (num_bins, target_length, target_length)
        sequence_ids = torch.stack(ids_list)       # (num_bins, target_length)
        
        return packed_tokens, attention_mask, sequence_ids


def pack_sequences(
    sequences: List[torch.Tensor],
    target_length: int,
    pad_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Pack a list of variable-length sequences into fixed-length batches.
    
    This implements the Patch n' Pack strategy from NaViT for efficient
    training with variable-resolution inputs.
    
    Args:
        sequences: List of tensors (L_i, D) with varying lengths
        target_length: Target packed sequence length
        pad_value: Padding value
    
    Returns:
        Dictionary with:
        - 'tokens': Packed tokens (B, target_length, D)
        - 'attention_mask': Attention mask (B, target_length, target_length)
        - 'sequence_ids': Sequence IDs (B, target_length)
    """
    batch = PatchNPackBatch(sequences, target_length, pad_value)
    
    return {
        'tokens': batch.packed_tokens,
        'attention_mask': batch.attention_mask,
        'sequence_ids': batch.sequence_ids,
    }


def compute_token_budget(
    resolutions: List[Tuple[int, int]],
    patch_size: int,
    max_tokens: int,
) -> int:
    """
    Compute the token budget for a given set of resolutions.
    
    Args:
        resolutions: List of (H, W) resolutions
        patch_size: Patch size
        max_tokens: Maximum number of tokens per sequence
    
    Returns:
        Recommended target_length for packing
    """
    token_counts = []
    for H, W in resolutions:
        H_p = H // patch_size
        W_p = W // patch_size
        token_counts.append(H_p * W_p)
    
    # Use the maximum token count as the target length
    return min(max(token_counts), max_tokens)
