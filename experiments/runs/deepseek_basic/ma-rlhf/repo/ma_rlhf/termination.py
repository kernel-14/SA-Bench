"""
Macro Action Termination Strategies for MA-RLHF.

Implements four termination conditions (ζ) for defining macro actions:
1. Fixed n-gram: Groups tokens into fixed-length sequences of length n.
2. Randomized n-gram: Randomly selects n-gram lengths from {2, 3, 5, 10}.
3. Parsing-based: Uses constituent tree parsing with DFS traversal.
4. Perplexity-based: Terminates when a token increases the perplexity.

Reference: Section 3.2.1 and Appendix B.4 of the MA-RLHF paper.
"""

import torch
from typing import List, Optional


def get_macro_action_positions_fixed_ngram(
    start: int,
    mask: torch.Tensor,
    n_gram: int,
) -> List[int]:
    """
    Fixed n-gram termination condition.
    
    Groups tokens into macro actions of exactly n tokens each.
    If the remaining tokens are fewer than n, they form the final macro action.
    
    The sequence boundaries follow the paper's pseudocode:
    sequence = [start]
    for each position, count valid tokens; when n_gram reached, append boundary.
    
    Args:
        start: Starting index (prompt length - 1).
        mask: Attention mask tensor of shape (1, seq_len).
        n_gram: Number of tokens per macro action.
    
    Returns:
        List of boundary positions defining macro actions.
    
    Reference: Appendix B.4, Section 1 and Appendix E pseudocode.
    """
    sequence = [start]
    current_count = 0
    
    # Total sequence length including start position
    total_len = mask.size(1)
    
    # Iterate over all positions from start to end-1
    for i in range(start, total_len - 1):
        current_count += mask[0, i].item()
        if current_count == n_gram:
            sequence.append(i + 1)
            current_count = 0
    
    # Always include the final position as a boundary
    sequence.append(int(mask.size(1)))
    
    return sequence


def get_macro_action_positions_randomized_ngram(
    start: int,
    mask: torch.Tensor,
    repeat_times: int = 3,
) -> List[int]:
    """
    Randomized n-gram termination condition.
    
    Randomly selects n-gram lengths from {2, 3, 5, 10}, repeating the pattern
    'repeat_times' times and shuffling.
    
    Args:
        start: Starting index (prompt length - 1).
        mask: Attention mask tensor of shape (1, seq_len).
        repeat_times: Number of times to repeat the base length pattern.
    
    Returns:
        List of boundary positions defining macro actions.
    
    Reference: Appendix B.4, Section 2.
    """
    k_list = torch.tensor([2, 3, 5, 10], dtype=torch.int)
    k_list = torch.repeat_interleave(k_list, repeat_times)
    k_list = k_list[torch.randperm(k_list.size(-1))]
    
    # Cumulative sum to get boundary indices
    indexed_k_list = torch.cumsum(k_list, dim=-1)
    
    total_len = mask.size(1) - start
    indexed_k_list = [start + x.item() for x in indexed_k_list if x.item() < total_len]
    
    sequence = [start] + indexed_k_list
    
    if sequence[-1] != int(mask.size(1) - 1):
        sequence.append(int(mask.size(1)))
    
    return sequence


def get_macro_action_positions_perplexity(
    start: int,
    mask: torch.Tensor,
    ppl: torch.Tensor,
) -> List[int]:
    """
    Perplexity-based termination condition.
    
    A macro action terminates when a token increases the perplexity
    (i.e., when ppl[i] > ppl[i-1]). The perplexity should exhibit a 
    monotonic decreasing pattern within each macro action.
    
    Args:
        start: Starting index (prompt length - 1).
        mask: Attention mask tensor of shape (1, seq_len).
        ppl: Perplexity values for each token position, shape (seq_len,).
    
    Returns:
        List of boundary positions defining macro actions.
    
    Reference: Section 3.2.1 and Appendix B.4, Section 4.
    """
    sequence = [start]
    for i in range(1, len(ppl)):
        if ppl[i] > ppl[i - 1]:
            sequence.append(start + i)
    
    if sequence[-1] != int(mask.size(1) - 1):
        sequence.append(int(mask.size(1)))
    
    return sequence


def get_macro_action_positions_parsing(
    start: int,
    mask: torch.Tensor,
    parse_tree,
    cutoff: int = 5,
) -> List[int]:
    """
    Parsing-based termination condition using constituent tree DFS.
    
    Traverses the constituent tree using depth-first search. A macro action
    terminates when the current node contains <= cutoff leaf tokens.
    
    Args:
        start: Starting index (prompt length - 1).
        mask: Attention mask tensor.
        parse_tree: Constituent parse tree of the generated response.
        cutoff: Maximum number of leaf tokens for a single macro action (C=5).
    
    Returns:
        List of boundary positions defining macro actions.
    
    Reference: Section 3.2.1 and Appendix B.4, Section 3.
    """
    sequence = [start]
    
    def dfs(node, ma_length):
        """DFS traversal of constituent tree."""
        if len(node.leaves()) < 1:
            return False, ma_length + 1
        
        if len(node.leaves()) <= cutoff:
            sequence.append(ma_length + len(node.leaves()))
            return True, ma_length + len(node.leaves())
        
        for child in node.childs():
            state, ma_length_ = dfs(child, ma_length)
            if not state:
                # Merge with previous macro action (avoid single-token termination)
                if len(sequence) > 1:
                    sequence[-1] = ma_length_
                ma_length = ma_length_
            else:
                ma_length = ma_length_
        
        return True, ma_length
    
    dfs(parse_tree, start)
    
    if sequence[-1] != int(mask.size(1) - 1):
        sequence.append(int(mask.size(1)))
    
    return sequence


def get_macro_action_positions(
    start: int,
    mask: torch.Tensor,
    termination: str = 'ngram',
    n_gram: Optional[int] = None,
    ppl: Optional[torch.Tensor] = None,
    repeat_times: Optional[int] = None,
    cutoff: Optional[int] = None,
    parse_tree=None,
) -> List[int]:
    """
    Main dispatch function for obtaining macro action boundary positions.
    
    Args:
        start: Starting index (prompt length - 1).
        mask: Attention mask tensor.
        termination: Type of termination condition.
            Options: 'ngram', 'randomized_ngram', 'ppl', 'parser'.
        n_gram: n-gram length (for 'ngram' termination).
        ppl: Perplexity values (for 'ppl' termination).
        repeat_times: Repeat times for randomized n-gram (default 3).
        cutoff: Cutoff threshold for parsing (default 5).
        parse_tree: Constituent parse tree (for 'parser' termination).
    
    Returns:
        List of boundary positions defining macro actions.
    
    Reference: Algorithm 1 and Appendix E of the MA-RLHF paper.
    """
    if termination == 'ngram':
        assert n_gram is not None, "n_gram must be provided for ngram termination"
        return get_macro_action_positions_fixed_ngram(start, mask, n_gram)
    
    elif termination == 'randomized_ngram':
        rt = repeat_times if repeat_times is not None else 3
        return get_macro_action_positions_randomized_ngram(start, mask, rt)
    
    elif termination == 'ppl':
        assert ppl is not None, "ppl must be provided for perplexity termination"
        return get_macro_action_positions_perplexity(start, mask, ppl)
    
    elif termination == 'parser':
        assert parse_tree is not None, "parse_tree must be provided for parser termination"
        c = cutoff if cutoff is not None else 5
        return get_macro_action_positions_parsing(start, mask, parse_tree, c)
    
    else:
        raise ValueError(f"Unknown termination type: {termination}")
