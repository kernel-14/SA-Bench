import spacy
from spacy.tokens import Doc
from spacy.language import Language
from loguru import logger
import torch
import re
import nltk
from typing import Any, List, Tuple, Optional, Union, Dict

# Check if spacy_parse_tree can be imported. This library extends spaCy's Doc object
# to provide constituent parse trees.
_SPACY_PARSE_TREE_AVAILABLE = False
try:
    import spacy_parse_tree
    # spacy_parse_tree works by globally extending Doc/Span objects upon import.
    # No explicit `add_pipe` is usually needed if the `ParseTree` component
    # is not explicitly added to the pipeline (which it doesn't seem to be
    # in standard usage for just accessing `doc._.parse_tree`).
    # However, to be safe, we check if the extension is registered.
    if not Doc.has_extension("parse_tree"):
        logger.warning("`spacy_parse_tree` is imported, but 'parse_tree' extension not found on Doc. "
                       "Attempting to register (this might not be sufficient if a specific component is needed).")
        # Attempt to manually register if not auto-registered by spacy_parse_tree import
        # This is a heuristic; actual behavior depends on spacy_parse_tree's internal logic.
        # Typically, a simple `import spacy_parse_tree` should handle this if it's installed.
        # For full pipeline integration, one might define a custom component.
        # The paper implies simple access to a parse tree, so we assume the extension is there.
        # We will set `_SPACY_PARSE_TREE_AVAILABLE` to True only if Doc.has_extension("parse_tree") is true.
    if Doc.has_extension("parse_tree"):
        _SPACY_PARSE_TREE_AVAILABLE = True
    else:
        logger.warning("`spacy_parse_tree` imported but `parse_tree` extension not available. "
                       "Constituent parsing for macro actions will be disabled.")
except ImportError:
    logger.warning(
        "Could not import 'spacy_parse_tree'. "
        "Parsing-based macro action termination will be disabled."
    )

# NLTK dependencies for tokenizing or other utility
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')


# Mocking the TokenizerWrapper class from utils.py to avoid circular imports.
# In a real scenario, this would be imported from utils.py.
class TokenizerWrapper:
    """Mock TokenizerWrapper for type hinting and to satisfy parser dependencies."""
    tokenizer: Any # AutoTokenizer instance
    def __init__(self, model_name: str):
        pass # Not used in this mock
    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = True, max_length: Optional[int] = None, truncation: bool = True, padding: Union[str, bool] = 'max_length', return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Mock method should not be called directly.")
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> Union[str, List[str]]:
        raise NotImplementedError("Mock method should not be called directly.")


def load_spacy_model(lang_model_name: str) -> Language:
    """
    Loads a spaCy language model. Assumes `spacy_parse_tree` is installed and extends
    the Doc object with `parse_tree` attribute if `_SPACY_PARSE_TREE_AVAILABLE` is True.

    Args:
        lang_model_name: The name of the spaCy model to load (e.g., 'en_core_web_sm').

    Returns:
        A spaCy Language object.

    Raises:
        OSError: If the specified spaCy model cannot be found or loaded.
    """
    try:
        nlp = spacy.load(lang_model_name)
        logger.info(f"SpaCy model '{lang_model_name}' loaded successfully.")

        if _SPACY_PARSE_TREE_AVAILABLE:
            logger.info("`spacy_parse_tree` library detected and assumed to be active for constituent parsing.")
        else:
            logger.warning(
                "`spacy_parse_tree` is not available or its extension is not registered. "
                "Constituent tree parsing will be disabled for parsing-based macro actions."
            )
        return nlp
    except OSError:
        logger.error(
            f"SpaCy model '{lang_model_name}' not found. "
            f"Please install it using: python -m spacy download {lang_model_name}"
        )
        raise


def get_constituent_tree(text: str, nlp: Language) -> Any:
    """
    Parses text into a constituent tree. This function relies on the `spacy_parse_tree`
    library extending spaCy's `Doc` object with a `parse_tree` attribute.

    Args:
        text: The input text to parse.
        nlp: The loaded spaCy Language model.

    Returns:
        The root node of the constituent tree (`spacy_parse_tree.Node` object),
        or None if `spacy_parse_tree` is not available/configured or parsing fails.
    """
    if not _SPACY_PARSE_TREE_AVAILABLE or not Doc.has_extension("parse_tree"):
        return None

    doc = nlp(text)
    return doc._.parse_tree


def _map_parsed_span_to_llm_tokens(
    parsed_text_span: str,
    current_global_llm_token_idx: int,
    tokenizer_wrapper: TokenizerWrapper,
    full_llm_token_ids: torch.Tensor,
) -> Tuple[bool, int, int]:
    """
    Maps a text span identified by a linguistic parser to a contiguous block
    of LLM subword tokens within the full response token IDs.

    Args:
        parsed_text_span: The text string corresponding to a segment from the parser.
        current_global_llm_token_idx: The starting index in `full_llm_token_ids`
                                      from where to search for the `parsed_text_span`.
        tokenizer_wrapper: The tokenizer wrapper for the LLM.
        full_llm_token_ids: The complete sequence of LLM token IDs for the response.

    Returns:
        A tuple: (success: bool, start_llm_token_idx: int, end_llm_token_idx: int).
        `success` is True if a contiguous, exact match was found.
        `start_llm_token_idx` and `end_llm_token_idx` are the boundaries in `full_llm_token_ids`.
        If `success` is False, `start_llm_token_idx` and `end_llm_token_idx` will be
        `current_global_llm_token_idx`, indicating no progress.
    """
    # 1. Pre-tokenize Parser Span
    # Ensure no special tokens are added for the segment itself.
    # Also ensure padding=False to avoid adding padding tokens for single span.
    encoded_parsed_span = tokenizer_wrapper.encode(
        parsed_text_span,
        add_special_tokens=False,
        padding=False,
        truncation=False, # We want to encode the full span
        return_tensors="pt"
    )
    # Squeeze to get a 1D tensor of token IDs. Handle potential empty output for very short/whitespace spans.
    encoded_parsed_span_ids = encoded_parsed_span['input_ids'].squeeze(0) # Assumes batch_size 1

    if encoded_parsed_span_ids.numel() == 0:
        # If the parsed span was effectively empty or only whitespace for the LLM tokenizer,
        # it cannot form a meaningful segment in LLM tokens. Treat as a failure to map for this part.
        logger.debug(
            f"Parser span '{parsed_text_span}' resulted in 0 LLM tokens. "
            f"Cannot map. Current index: {current_global_llm_token_idx}"
        )
        return False, current_global_llm_token_idx, current_global_llm_token_idx

    llm_token_len = encoded_parsed_span_ids.numel()
    
    # 2. Search for Match in full_llm_token_ids
    # Search from current_global_llm_token_idx onwards in the full response's LLM tokens.
    # The search should only go as far as `full_llm_token_ids` allows for a full `llm_token_len` match.
    search_end_limit = full_llm_token_ids.numel() - llm_token_len + 1

    for i in range(current_global_llm_token_idx, search_end_limit):
        if torch.equal(full_llm_token_ids[i : i + llm_token_len], encoded_parsed_span_ids):
            # Found a match based on token IDs. Now, perform a text re-verification.
            # This is crucial because different tokenizers might encode the same text
            # slightly differently (e.g., leading/trailing spaces, special characters),
            # or the LLM tokenizer might have split words differently than a linguistic parser.
            decoded_llm_span = tokenizer_wrapper.decode(
                full_llm_token_ids[i : i + llm_token_len], skip_special_tokens=True
            )
            
            # Simple normalization for comparison (e.g., collapsing multiple spaces, stripping).
            # This makes the comparison more robust to minor whitespace differences.
            normalized_parsed = re.sub(r'\s+', ' ', parsed_text_span).strip()
            normalized_decoded = re.sub(r'\s+', ' ', decoded_llm_span).strip()

            if normalized_parsed == normalized_decoded:
                # logger.debug(f"Mapped parser span '{parsed_text_span}' to LLM tokens {i}:{i+llm_token_len}")
                return True, i, i + llm_token_len
            else:
                logger.debug(
                    f"LLM token ID match found ({i}:{i+llm_token_len}) but text decode mismatch: "
                    f"'{normalized_parsed}' != '{normalized_decoded}'. "
                    f"Continuing search for better match or declaring failure."
                )

    logger.debug(
        f"Failed to map parser span '{parsed_text_span}' "
        f"starting from LLM token index {current_global_llm_token_idx}. "
        f"Full LLM tokens length: {full_llm_token_ids.numel()}. "
        f"Parsed span length (LLM tokens): {llm_token_len}. "
        f"LLM decoded response: '{tokenizer_wrapper.decode(full_llm_token_ids, skip_special_tokens=True)}'"
    )
    return False, current_global_llm_token_idx, current_global_llm_token_idx


def _dfs_extract_macro_segments(
    node: Any, # spacy_parse_tree.Node object
    current_global_llm_token_idx: int,
    tokenizer_wrapper: TokenizerWrapper,
    full_llm_token_ids: torch.Tensor,
    cutoff: int,
    all_segments: List[Tuple[int, int]]
) -> Tuple[bool, int]:
    """
    Performs Depth-First Search on the constituent parse tree to identify macro action boundaries.
    Segments found are mapped to LLM token indices and appended to `all_segments`.

    Args:
        node: The current node in the constituent parse tree (e.g., `spacy_parse_tree.Node`).
        current_global_llm_token_idx: The current starting index in `full_llm_token_ids`
                                      from which the current `node`'s text span is expected to start.
        tokenizer_wrapper: The LLM tokenizer wrapper.
        full_llm_token_ids: The complete sequence of LLM token IDs for the response.
        cutoff: The maximum number of leaf tokens (parser's tokens) for a node to define a macro action.
        all_segments: A list to accumulate the identified macro action segments (modified in-place).

    Returns:
        A tuple: (success: bool, next_llm_token_idx: int).
        `success` is True if the node and its children were successfully mapped to LLM tokens.
        `next_llm_token_idx` is the LLM token index immediately after the current node's span.
        If `success` is False, `next_llm_token_idx` will be `current_global_llm_token_idx`,
        signaling that a part of the text could not be mapped.
    """
    node_start_llm_token_idx = current_global_llm_token_idx
    
    # Base case: If it's a leaf node in the spaCy parse tree, map its text.
    if not node.children:
        parsed_text_span = node.text
        success_map, mapped_start, mapped_end = _map_parsed_span_to_llm_tokens(
            parsed_text_span,
            node_start_llm_token_idx,
            tokenizer_wrapper,
            full_llm_token_ids,
        )
        if not success_map:
            logger.debug(f"DFS: Leaf node text '{parsed_text_span}' could not be mapped. Failing.")
            return False, node_start_llm_token_idx # Propagate failure
        node_end_llm_token_idx = mapped_end
    else:
        # Recursive step: Process children first to get their LLM token spans
        current_child_llm_token_idx = node_start_llm_token_idx
        for child in node.children:
            success_child, child_processed_llm_token_idx = _dfs_extract_macro_segments(
                child,
                current_child_llm_token_idx,
                tokenizer_wrapper,
                full_llm_token_ids,
                cutoff,
                all_segments
            )
            if not success_child:
                logger.debug(f"DFS: Child segment failed. Propagating failure for node '{node.text}'.")
                return False, node_start_llm_token_idx # Propagate failure
            current_child_llm_token_idx = child_processed_llm_token_idx
        node_end_llm_token_idx = current_child_llm_token_idx

    # After children are processed (or if it was a leaf), apply termination condition to the current node's span
    llm_token_length = node_end_llm_token_idx - node_start_llm_token_idx

    if llm_token_length <= 0:
        # This node mapped to no LLM tokens or negative length. It's not a valid segment.
        # However, it doesn't necessarily mean failure if its children were processed.
        return True, node_start_llm_token_idx # Effectively consume no LLM tokens from this point.

    num_parser_leaves = len(node.leaves()) # Number of terminal tokens (words/punctuation) in this parser node

    # Rule 1: nodes with fewer than C tokens mark the end of a macro action
    if num_parser_leaves < cutoff:
        # Rule 2: nodes with single token are included in the last macro action, avoiding single-token termination conditions like punctuation.
        if llm_token_length == 1:
            decoded_token = tokenizer_wrapper.decode(
                full_llm_token_ids[node_start_llm_token_idx : node_end_llm_token_idx],
                skip_special_tokens=True
            ).strip()
            
            # Simple check for common punctuation (single character, non-alphanumeric)
            is_punctuation = len(decoded_token) == 1 and not decoded_token.isalnum()

            if is_punctuation and all_segments:
                # Merge with the last existing macro action
                last_start, _ = all_segments[-1]
                all_segments[-1] = (last_start, node_end_llm_token_idx)
                logger.debug(
                    f"Merged single-token punctuation '{decoded_token}' (LLM {node_start_llm_token_idx}:{node_end_llm_token_idx}) "
                    f"into previous segment. New last segment: {all_segments[-1]}"
                )
            else:
                # If not punctuation, or no previous segment to merge with, add as a new segment.
                all_segments.append((node_start_llm_token_idx, node_end_llm_token_idx))
                logger.debug(f"Added single-LLM-token segment: '{decoded_token}' (LLM {node_start_llm_token_idx}:{node_end_llm_token_idx})")
        else:
            # Multi-token segment satisfying cutoff.
            all_segments.append((node_start_llm_token_idx, node_end_llm_token_idx))
            logger.debug(f"Added multi-LLM-token segment: '{tokenizer_wrapper.decode(full_llm_token_ids[node_start_llm_token_idx:node_end_llm_token_idx])}' (LLM {node_start_llm_token_idx}:{node_end_llm_token_idx})")

    # Return success and the index of the LLM token right after this node's span.
    return True, node_end_llm_token_idx


def get_parsing_based_macro_segments(
    response_text: str,
    full_llm_token_ids: torch.Tensor,
    nlp: Language,
    tokenizer_wrapper: TokenizerWrapper,
    cutoff: int
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Generates macro action segments for a response based on its constituent parse tree.
    This function handles the overall logic, including error handling for parsing failures
    and ensuring all LLM tokens are covered.

    Args:
        response_text: The full text of the response, used for linguistic parsing.
        full_llm_token_ids: The PyTorch tensor of LLM token IDs for the response.
        nlp: The loaded spaCy Language model, expected to be configured for constituent parsing.
        tokenizer_wrapper: The LLM tokenizer wrapper instance.
        cutoff: The maximum number of leaf tokens (parser's tokens) for a parser node
                to be considered as a macro action boundary.

    Returns:
        A tuple: (success: bool, List[Tuple[int, int]]).
        `success` is True if parsing and all LLM token mapping were successful for the entire response.
        The list contains (start_idx, end_idx) tuples of LLM token indices for each macro action.
        If `success` is False, the list will be empty, signaling that the `MacroActionHandler`
        should revert to a fallback (e.g., n-gram based) for this response.
    """
    if full_llm_token_ids.numel() == 0:
        return True, [] # An empty response has no macro actions.

    # 1. Obtain the constituent parse tree
    root_node = get_constituent_tree(response_text, nlp)

    if root_node is None:
        logger.warning(
            "Could not obtain constituent tree (spacy_parse_tree not available/configured). "
            "Parsing-based segmentation failed for this response."
        )
        return False, []

    all_segments: List[Tuple[int, int]] = []
    
    # 2. Perform DFS to extract macro action segments
    success, last_processed_llm_token_idx = _dfs_extract_macro_segments(
        root_node,
        0, # Start mapping from the first LLM token
        tokenizer_wrapper,
        full_llm_token_ids,
        cutoff,
        all_segments
    )

    if not success:
        logger.warning(
            "DFS for parsing-based macro actions encountered unresolvable token mapping discrepancies. "
            "Returning failure for fallback."
        )
        return False, []
    
    # 3. Post-processing: Ensure all LLM tokens are covered
    if not all_segments:
        if full_llm_token_ids.numel() > 0:
            # If no segments were found by DFS but there are LLM tokens,
            # treat the entire response as a single macro action.
            logger.warning(
                "Parsing-based DFS completed without adding any segments, but LLM tokens exist. "
                "Defaulting to a single macro action for the entire response."
            )
            all_segments.append((0, full_llm_token_ids.numel()))
    elif last_processed_llm_token_idx < full_llm_token_ids.numel():
        # If there are trailing LLM tokens not covered by the last segment,
        # extend the last segment to include them. This handles cases where
        # the parser might have missed a tail part or mapping was imperfect.
        logger.warning(
            f"Trailing LLM tokens from index {last_processed_llm_token_idx} to {full_llm_token_ids.numel()} "
            f"not covered by parser segments. Extending the last macro action to include them."
        )
        last_start, _ = all_segments[-1]
        all_segments[-1] = (last_start, full_llm_token_ids.numel())
    
    # Ensure segments are sorted and non-overlapping (should be guaranteed by DFS logic if successful)
    # If any gaps somehow appeared, it would lead to an issue. The DFS is designed to avoid this.

    return True, all_segments

