"""
This module implements the MacroActionHandler, a central component in MA-RLHF.
It defines, segments, and aggregates information based on macro actions using
various termination conditions (fixed n-gram, randomized n-gram, parsing-based,
and perplexity-based) and value aggregation methods.
"""

import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, List, Tuple, Union, Optional
from loguru import logger
from omegaconf import DictConfig

# Avoid circular imports by defining Config as DictConfig and using string references or Any for types
# that would cause circular dependencies.
Config = DictConfig

# Import necessary components from other modules
from utils import TokenizerWrapper  # Assuming utils.py is in the same package
from parsers import (
    load_spacy_model,
    get_parsing_based_macro_segments,
    _SPACY_PARSE_TREE_AVAILABLE # To check if spacy_parse_tree is loaded
)
# For perplexity-based termination, we need the SFTModel.
# We use a string literal to avoid a direct import that could create a circular dependency
# if SFTModel somehow needed MacroActionHandler.
# The actual SFTModel instance will be passed at runtime.
SFTModel = Any # Placeholder for type hinting, actual type is models.SFTModel


class MacroActionHandler:
    """
    Manages the creation, segmentation, and aggregation of data for macro actions.
    Supports various termination conditions and value/reward aggregation methods.
    """

    def __init__(self, config: Config, tokenizer_wrapper: TokenizerWrapper):
        """
        Initializes the MacroActionHandler.

        Args:
            config: The global configuration object.
            tokenizer_wrapper: An instance of TokenizerWrapper for tokenization.
        """
        self.config: Config = config
        self.tokenizer_wrapper: TokenizerWrapper = tokenizer_wrapper
        self.nlp: Optional[Any] = None  # SpaCy Language model for parsing-based termination

        # Pre-load spaCy model if parsing-based termination is a configured option
        if "parsing_based" in self.config.macro_action_config:
            parser_model_name = self.config.macro_action_config.parsing_based.parser_model
            if _SPACY_PARSE_TREE_AVAILABLE:
                try:
                    self.nlp = load_spacy_model(parser_model_name)
                except Exception as e:
                    logger.warning(
                        f"Failed to load spaCy model '{parser_model_name}' for parsing-based "
                        f"macro actions: {e}. Parsing-based termination will not be fully functional."
                    )
                    self.nlp = None
            else:
                logger.warning(
                    "spacy_parse_tree is not available. Parsing-based macro action termination "
                    "will not be functional even if configured."
                )
        
        logger.info("MacroActionHandler initialized.")

    def get_macro_action_positions(
        self,
        full_response_ids: torch.Tensor,
        termination_method: str,
        tokenizer_wrapper: TokenizerWrapper,
        sft_model: Optional[SFTModel] = None,
        **kwargs
    ) -> List[Tuple[int, int]]:
        """
        Dispatches to the appropriate macro action termination method to get segment positions.

        Args:
            full_response_ids: A 1D PyTorch tensor of token IDs representing the full response.
                               Assumed to be already on the correct device.
            termination_method: The name of the termination method to use
                                (e.g., 'fixed_n_gram', 'randomized_n_gram',
                                'parsing_based', 'perplexity_based').
            tokenizer_wrapper: The TokenizerWrapper instance, needed for text decoding in parsing.
            sft_model: The SFTModel instance, required for perplexity-based termination.
            **kwargs: Additional parameters for specific termination methods (e.g., batch_size).

        Returns:
            A list of (start_idx, end_idx) tuples, where each tuple defines the token
            indices of a macro action within the `full_response_ids`. Returns an empty list
            if the response is empty or no macro actions can be formed.
        """
        if full_response_ids.numel() == 0:
            return []

        macro_action_config = self.config.macro_action_config

        if termination_method == "fixed_n_gram":
            n_gram_length = macro_action_config.default_n_gram_n
            return self._fixed_ngram_termination(full_response_ids, n_gram_length)
        elif termination_method == "randomized_n_gram":
            n_gram_list = macro_action_config.randomized_n_gram.list_of_lengths
            repeat_times = macro_action_config.randomized_n_gram.repeat_times
            return self._randomized_ngram_termination(
                full_response_ids, n_gram_list, repeat_times
            )
        elif termination_method == "parsing_based":
            if self.nlp is None:
                logger.warning(
                    "Parsing-based termination configured but spaCy model not loaded. "
                    "Falling back to fixed n-gram termination."
                )
                return self._fixed_ngram_termination(
                    full_response_ids, macro_action_config.default_n_gram_n
                )
            cutoff = macro_action_config.parsing_based.cutoff
            return self._parsing_based_termination(
                full_response_ids, self.nlp, tokenizer_wrapper, cutoff
            )
        elif termination_method == "perplexity_based":
            if sft_model is None:
                raise ValueError(
                    "SFTModel instance must be provided for perplexity-based termination."
                )
            return self._perplexity_based_termination(full_response_ids, sft_model)
        else:
            raise ValueError(f"Unknown termination method: {termination_method}")

    def _fixed_ngram_termination(
        self, response_ids: torch.Tensor, n_gram_length: Union[int, str]
    ) -> List[Tuple[int, int]]:
        """
        Segments the response into macro actions of a fixed length.

        Args:
            response_ids: 1D PyTorch tensor of token IDs for a single response.
            n_gram_length: Integer or string "infinity" indicating the fixed length of each macro action.

        Returns:
            A list of (start_idx, end_idx) tuples for macro actions.
        """
        macro_action_segments: List[Tuple[int, int]] = []
        seq_len = response_ids.numel()

        if seq_len == 0:
            return []

        if n_gram_length == "infinity":
            macro_action_segments.append((0, seq_len))
            return macro_action_segments

        effective_n_gram_length = int(n_gram_length)
        if effective_n_gram_length <= 0:
            logger.warning(
                f"Invalid n_gram_length: {effective_n_gram_length}. Using default of 1."
            )
            effective_n_gram_length = 1

        start_idx = 0
        while start_idx < seq_len:
            end_idx = min(start_idx + effective_n_gram_length, seq_len)
            macro_action_segments.append((start_idx, end_idx))
            start_idx = end_idx
        return macro_action_segments

    def _randomized_ngram_termination(
        self, response_ids: torch.Tensor, n_gram_list: List[int], repeat_times: int
    ) -> List[Tuple[int, int]]:
        """
        Segments the response using randomized n-gram lengths from a predefined list.

        Args:
            response_ids: 1D PyTorch tensor of token IDs for a single response.
            n_gram_list: List of integers representing possible n-gram lengths.
            repeat_times: Integer, how many times to repeat the list before shuffling.

        Returns:
            A list of (start_idx, end_idx) tuples for macro actions.
        """
        macro_action_segments: List[Tuple[int, int]] = []
        seq_len = response_ids.numel()

        if seq_len == 0:
            return []

        if not n_gram_list:
            logger.warning("n_gram_list is empty for randomized termination. Falling back to fixed n-gram.")
            return self._fixed_ngram_termination(response_ids, self.config.macro_action_config.default_n_gram_n)

        # Create an extended and shuffled list of n-gram lengths
        extended_n_gram_lengths = n_gram_list * repeat_times
        random.shuffle(extended_n_gram_lengths)
        
        # Ensure that the list can provide at least one segment if it's too short for 'repeat_times'
        if not extended_n_gram_lengths and seq_len > 0:
            extended_n_gram_lengths = [self.config.macro_action_config.default_n_gram_n] # Fallback to default if list becomes empty

        current_idx = 0
        list_ptr = 0
        while current_idx < seq_len:
            n_gram_length: int
            if list_ptr < len(extended_n_gram_lengths):
                n_gram_length = extended_n_gram_lengths[list_ptr]
                list_ptr += 1
            else:
                # If shuffled list exhausted, use remaining length for the last segment
                n_gram_length = seq_len - current_idx
                if n_gram_length <= 0: # Avoid infinite loop if somehow current_idx already at seq_len
                    break 

            n_gram_length = max(1, n_gram_length) # Ensure length is at least 1

            end_idx = min(current_idx + n_gram_length, seq_len)
            macro_action_segments.append((current_idx, end_idx))
            current_idx = end_idx

        return macro_action_segments

    def _parsing_based_termination(
        self,
        response_ids: torch.Tensor,
        nlp: Any, # spacy.Language
        tokenizer_wrapper: TokenizerWrapper,
        cutoff: int,
    ) -> List[Tuple[int, int]]:
        """
        Segments the response based on linguistic parsing (constituent tree).

        Args:
            response_ids: 1D PyTorch tensor of token IDs for a single response.
            nlp: The loaded spaCy Language model.
            tokenizer_wrapper: The TokenizerWrapper instance.
            cutoff: Integer, the maximum number of leaf tokens for a non-terminal node
                    to terminate a macro action.

        Returns:
            A list of (start_idx, end_idx) tuples for macro actions.
            Falls back to fixed n-gram if parsing fails or encounters discrepancies.
        """
        if response_ids.numel() == 0:
            return []

        # 1. Convert response_ids to text
        response_text = tokenizer_wrapper.decode(response_ids, skip_special_tokens=True)

        # 2. Get macro action segments using the parsers module
        # The parsers.get_parsing_based_macro_segments handles the DFS and token mapping
        # and returns a success flag and the segments list.
        success, segments = get_parsing_based_macro_segments(
            response_text, response_ids, nlp, tokenizer_wrapper, cutoff
        )

        if not success:
            logger.warning(
                "Parsing-based macro action segmentation failed or had discrepancies. "
                f"Falling back to fixed n-gram ({self.config.macro_action_config.default_n_gram_n}) termination."
            )
            return self._fixed_ngram_termination(
                response_ids, self.config.macro_action_config.default_n_gram_n
            )

        return segments

    def _perplexity_based_termination(
        self, response_ids: torch.Tensor, sft_model: SFTModel
    ) -> List[Tuple[int, int]]:
        """
        Segments the response based on changes in token-level perplexity.
        A macro action terminates until it reaches a token that has a negative impact
        on the perplexity of the macro action (i.e., would increase it).

        Args:
            response_ids: 1D PyTorch tensor of token IDs for a single response.
            sft_model: The SFTModel instance, used to get token-level log probabilities.

        Returns:
            A list of (start_idx, end_idx) tuples for macro actions.
        """
        macro_action_segments: List[Tuple[int, int]] = []
        seq_len = response_ids.numel()

        if seq_len == 0:
            return []

        # Perplexity calculation needs full sequence log probs.
        # SFTModel.get_log_probs expects a batch dimension, so unsqueeze.
        # We need attention_mask for sft_model.get_log_probs
        attention_mask = (response_ids != self.tokenizer_wrapper.tokenizer.pad_token_id).long()
        token_log_probs = sft_model.get_log_probs(
            response_ids.unsqueeze(0), attention_mask.unsqueeze(0)
        ).squeeze(0) # (seq_len,)

        # NLLs for each token (negative log-likelihood)
        nlls = -token_log_probs

        start_idx = 0
        while start_idx < seq_len:
            current_macro_end_idx = start_idx + 1 # Start with at least one token in the macro
            
            # The perplexity definition is `ppl(ωτ) ∝ − 1|ωτ| Pa∈ωτ`. This is just the average NLL.
            # Mathematical condition:
            # 1. PPL should be decreasing or stable within the macro action.
            # 2. A macro action terminates when adding the next token would increase overall PPL.

            while current_macro_end_idx <= seq_len:
                # NLLs for the current candidate macro action
                segment_nlls = nlls[start_idx:current_macro_end_idx]
                if segment_nlls.numel() == 0: # Should not happen if current_macro_end_idx > start_idx
                    current_macro_end_idx += 1
                    continue

                ppl_current_macro = torch.mean(segment_nlls)

                # Check if we are at the end of the response
                if current_macro_end_idx == seq_len:
                    # If end of sequence, this is the last macro action
                    macro_action_segments.append((start_idx, current_macro_end_idx))
                    start_idx = current_macro_end_idx
                    break # Exit inner while loop
                else:
                    # Consider adding the next token
                    segment_nlls_with_next = nlls[start_idx : current_macro_end_idx + 1]
                    ppl_with_next_token = torch.mean(segment_nlls_with_next)
                    
                    # Termination condition: if adding the next token increases perplexity
                    # or if the current segment perplexity is already higher than with the next token
                    # (which implies the sequence has been "bad" for a while and getting better)
                    # The paper's condition: `ppl(ωτ U a_{t_τ+1}) > ppl(ωτ)`
                    if ppl_with_next_token > ppl_current_macro:
                        macro_action_segments.append((start_idx, current_macro_end_idx))
                        start_idx = current_macro_end_idx
                        break # Terminate current macro action, start a new one
                    else:
                        current_macro_end_idx += 1
            
            # If the inner loop breaks due to `current_macro_end_idx == seq_len`
            # and no segment was added (e.g., `start_idx` didn't update),
            # this means the entire remaining sequence should form one macro action.
            if start_idx == current_idx and start_idx < seq_len:
                macro_action_segments.append((start_idx, seq_len))
                start_idx = seq_len # Ensure outer loop terminates

        return macro_action_segments

    def aggregate_values(
        self,
        token_values: torch.Tensor,
        macro_action_segments: List[Tuple[int, int]],
        aggregation_method: str,
    ) -> torch.Tensor:
        """
        Aggregates token-level value estimates into macro-level value estimates.

        Args:
            token_values: A 1D PyTorch tensor of token-level value estimates (for a single response).
            macro_action_segments: A list of (start_idx, end_idx) tuples defining macro actions.
            aggregation_method: String ('equal_assignment', 'unit_assignment', 'position_decayed_assignment').

        Returns:
            A 1D PyTorch tensor of macro-level value estimates.
        """
        macro_values_list: List[torch.Tensor] = []
        device = token_values.device
        dtype = token_values.dtype

        for start, end in macro_action_segments:
            segment_values = token_values[start:end]
            if segment_values.numel() == 0:
                macro_values_list.append(torch.tensor(0.0, device=device, dtype=dtype))
                continue

            macro_value: torch.Tensor
            if aggregation_method == "equal_assignment":
                macro_value = torch.mean(segment_values)
            elif aggregation_method == "unit_assignment":
                macro_value = segment_values[-1]  # Value of the last token
            elif aggregation_method == "position_decayed_assignment":
                length = segment_values.numel()
                # Compute H = sum(1 / (length - i) for i in range(length))
                # Using 1-based indexing for 1/(len-i) to align with paper's formula (i=0 to L-1).
                # The denominator would be len, len-1, ..., 1. So i goes from 0 to len-1
                denominators = torch.arange(length, 0, -1, device=device, dtype=dtype)
                inverse_denominators = 1.0 / denominators
                H = torch.sum(inverse_denominators)
                
                if H == 0: # Avoid division by zero for very short segments
                    macro_value = torch.mean(segment_values) # Fallback to equal assignment
                else:
                    weights = inverse_denominators / H
                    macro_value = torch.sum(segment_values * weights)
            else:
                raise ValueError(f"Unknown value aggregation method: {aggregation_method}")
            macro_values_list.append(macro_value)

        if not macro_values_list:
            return torch.empty(0, device=device, dtype=dtype)
            
        return torch.stack(macro_values_list)

    def aggregate_rewards(
        self, token_rewards: torch.Tensor, macro_action_segments: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Aggregates token-level rewards into macro-level rewards.
        The paper states `rho=1`, implying summation of rewards within a macro action.

        Args:
            token_rewards: A 1D PyTorch tensor of token-level rewards (for a single response).
            macro_action_segments: A list of (start_idx, end_idx) tuples defining macro actions.

        Returns:
            A 1D PyTorch tensor of macro-level reward sums.
        """
        macro_rewards_list: List[torch.Tensor] = []
        device = token_rewards.device
        dtype = token_rewards.dtype

        for start, end in macro_action_segments:
            segment_rewards = token_rewards[start:end]
            if segment_rewards.numel() == 0:
                macro_rewards_list.append(torch.tensor(0.0, device=device, dtype=dtype))
            else:
                macro_rewards_list.append(torch.sum(segment_rewards))

        if not macro_rewards_list:
            return torch.empty(0, device=device, dtype=dtype)

        return torch.stack(macro_rewards_list)

    def split_token_data_by_macro_action(
        self, token_data: torch.Tensor, macro_action_segments: List[Tuple[int, int]]
    ) -> List[torch.Tensor]:
        """
        Helper function to split a token-level tensor into a list of tensors,
        each corresponding to a macro action segment.

        Args:
            token_data: A PyTorch tensor (e.g., log probabilities, old log probabilities).
                        Can be 1D (seq_len) or 2D (seq_len, dim).
            macro_action_segments: A list of (start_idx, end_idx) tuples.

        Returns:
            A list of PyTorch tensors, each representing a segment of the original data.
        """
        split_data: List[torch.Tensor] = []
        for start, end in macro_action_segments:
            split_data.append(token_data[start:end])
        return split_data


if __name__ == "__main__":
    import sys
    # For testing, we mock the SFTModel
    class MockSFTModel:
        def __init__(self, device="cpu", dtype=torch.float32):
            self.device = device
            self.dtype = dtype
        
        def get_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            # Simulate log probabilities.
            # Perplexity termination expects higher NLL for 'bad' tokens.
            # Let's make some tokens have higher NLL to trigger termination.
            seq_len = input_ids.shape[1]
            # Random uniform log probs, then some spikes
            log_probs = -torch.rand(input_ids.shape[0], seq_len, device=self.device, dtype=self.dtype) * 2
            
            # Introduce a pattern for perplexity changes
            # For a 1D tensor representing a single response:
            if input_ids.shape[0] == 1:
                # Force some perplexity increases
                # e.g., token at index 5 and 10 might have higher NLL than previous
                if seq_len > 5: log_probs[0, 5] = -0.5 # lower prob -> higher NLL
                if seq_len > 10: log_probs[0, 10] = -0.3
                if seq_len > 15: log_probs[0, 15] = -0.8
            
            return log_probs * attention_mask

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Mock a minimal config object matching config.yaml structure
    mock_config = OmegaConf.create({
        "global": {
            "precision": "float32",
            "seed": 42
        },
        "macro_action_config": {
            "default_termination_method": "fixed_n_gram",
            "default_n_gram_n": 5,
            "default_value_aggregation_method": "equal_assignment",
            "fixed_n_gram": {
                "n_values": [3, 5, 10, "infinity"]
            },
            "randomized_n_gram": {
                "list_of_lengths": [2, 3, 5, 10],
                "repeat_times": 3
            },
            "parsing_based": {
                "cutoff": 5,
                "parser_model": "en_core_web_sm" # Need to ensure this is installed for tests
            },
            "perplexity_based": {}
        }
    })

    # Mock a TokenizerWrapper (assuming gpt2 for simplicity)
    mock_tokenizer_wrapper = TokenizerWrapper("gpt2")
    
    # Instantiate the handler
    handler = MacroActionHandler(mock_config, mock_tokenizer_wrapper)

    # Sample response IDs
    test_response_text = "This is a sample sentence for testing macro actions. It has several words."
    test_response_ids = mock_tokenizer_wrapper.encode(test_response_text, add_special_tokens=False, padding=False)['input_ids'].squeeze(0)
    logger.info(f"Original response length (tokens): {test_response_ids.numel()}")
    logger.info(f"Original response (decoded): '{mock_tokenizer_wrapper.decode(test_response_ids)}'")

    # --- Test Fixed N-gram Termination ---
    logger.info("\n--- Testing Fixed N-gram Termination (n=5) ---")
    fixed_segments_5 = handler.get_macro_action_positions(
        test_response_ids, "fixed_n_gram", mock_tokenizer_wrapper
    )
    logger.info(f"Fixed 5-gram segments: {fixed_segments_5}")
    expected_fixed_5_segments = [(0, 5), (5, 10), (10, 15), (15, 18)] # Based on gpt2 tokenizer (18 tokens)
    assert len(fixed_segments_5) == math.ceil(test_response_ids.numel() / 5)
    assert fixed_segments_5[-1][1] == test_response_ids.numel()

    logger.info("\n--- Testing Fixed N-gram Termination (n=infinity) ---")
    # Temporarily override config for this test
    old_n_gram_n = mock_config.macro_action_config.default_n_gram_n
    mock_config.macro_action_config.default_n_gram_n = "infinity"
    fixed_segments_inf = handler.get_macro_action_positions(
        test_response_ids, "fixed_n_gram", mock_tokenizer_wrapper
    )
    logger.info(f"Fixed 'infinity' n-gram segments: {fixed_segments_inf}")
    assert fixed_segments_inf == [(0, test_response_ids.numel())]
    mock_config.macro_action_config.default_n_gram_n = old_n_gram_n # Restore


    # --- Test Randomized N-gram Termination ---
    logger.info("\n--- Testing Randomized N-gram Termination ---")
    random.seed(mock_config.global.seed) # Ensure reproducibility for test
    randomized_segments = handler.get_macro_action_positions(
        test_response_ids, "randomized_n_gram", mock_tokenizer_wrapper
    )
    logger.info(f"Randomized n-gram segments: {randomized_segments}")
    assert sum(end - start for start, end in randomized_segments) == test_response_ids.numel()


    # --- Test Parsing-based Termination ---
    logger.info("\n--- Testing Parsing-based Termination ---")
    # This part requires spaCy model 'en_core_web_sm' to be downloaded and spacy_parse_tree installed.
    # If not, it will fall back to fixed n-gram.
    if handler.nlp:
        parsing_segments = handler.get_macro_action_positions(
            test_response_ids, "parsing_based", mock_tokenizer_wrapper
        )
        logger.info(f"Parsing-based segments: {parsing_segments}")
        assert sum(end - start for start, end in parsing_segments) == test_response_ids.numel()
        for start, end in parsing_segments:
            logger.debug(f"  Segment: '{mock_tokenizer_wrapper.decode(test_response_ids[start:end])}'")
    else:
        logger.info("Skipping full parsing-based test due to missing spaCy model or spacy_parse_tree.")
        # Test fallback behavior:
        fallback_segments = handler.get_macro_action_positions(
            test_response_ids, "parsing_based", mock_tokenizer_wrapper
        )
        logger.info(f"Parsing-based (fallback to fixed n-gram) segments: {fallback_segments}")
        assert fallback_segments == fixed_segments_5 # Should match default fixed n-gram


    # --- Test Perplexity-based Termination ---
    logger.info("\n--- Testing Perplexity-based Termination ---")
    mock_sft_model = MockSFTModel(device="cpu", dtype=torch.float32)
    perplexity_segments = handler.get_macro_action_positions(
        test_response_ids, "perplexity_based", mock_tokenizer_wrapper, sft_model=mock_sft_model
    )
    logger.info(f"Perplexity-based segments: {perplexity_segments}")
    assert sum(end - start for start, end in perplexity_segments) == test_response_ids.numel()


    # --- Test Aggregation Methods ---
    logger.info("\n--- Testing Aggregation Methods ---")
    mock_token_values = torch.randn(test_response_ids.numel())
    mock_token_rewards = torch.randn(test_response_ids.numel())

    # Use fixed_segments_5 for aggregation tests
    segments_to_test = fixed_segments_5

    # Equal Assignment
    macro_values_equal = handler.aggregate_values(
        mock_token_values, segments_to_test, "equal_assignment"
    )
    logger.info(f"Macro values (equal): {macro_values_equal.tolist()}")
    
    # Unit Assignment
    macro_values_unit = handler.aggregate_values(
        mock_token_values, segments_to_test, "unit_assignment"
    )
    logger.info(f"Macro values (unit): {macro_values_unit.tolist()}")

    # Position Decayed Assignment
    macro_values_pos_decay = handler.aggregate_values(
        mock_token_values, segments_to_test, "position_decayed_assignment"
    )
    logger.info(f"Macro values (position decayed): {macro_values_pos_decay.tolist()}")

    # Aggregate Rewards (summation)
    macro_rewards = handler.aggregate_rewards(mock_token_rewards, segments_to_test)
    logger.info(f"Macro rewards (sum): {macro_rewards.tolist()}")

    assert macro_values_equal.shape[0] == len(segments_to_test)
    assert macro_values_unit.shape[0] == len(segments_to_test)
    assert macro_values_pos_decay.shape[0] == len(segments_to_test)
    assert macro_rewards.shape[0] == len(segments_to_test)


    # --- Test split_token_data_by_macro_action ---
    logger.info("\n--- Testing split_token_data_by_macro_action ---")
    split_data = handler.split_token_data_by_macro_action(mock_token_values, segments_to_test)
    logger.info(f"Split data (first few items): {[t.tolist() for t in split_data[:3]]}")
    assert len(split_data) == len(segments_to_test)
    assert sum(len(t) for t in split_data) == mock_token_values.numel()


    logger.info("\nAll MacroActionHandler tests completed successfully!")
