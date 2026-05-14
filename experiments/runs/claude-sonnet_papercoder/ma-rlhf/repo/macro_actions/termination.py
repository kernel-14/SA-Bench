```python
## macro_actions/termination.py
"""Macro action termination strategies for MA-RLHF.

This module implements MacroActionTermination, the component responsible for
segmenting a generated token sequence into macro actions. It is the most
critical and novel component of the MA-RLHF framework.

Four termination strategies are supported, as described in Section 3.2.1
and Appendix B.4 of the paper:
  1. fixed_ngram: Group every n consecutive tokens into one macro action.
  2. randomized_ngram: Variable-length macro actions drawn from {2,3,5,10}.
  3. parsing: Use constituency parse tree (NL) or AST (code) boundaries.
  4. ppl: Terminate when adding the next token increases perplexity.

The output of get_positions() is always a list of integer boundary indices:
    [start, b1, b2, ..., end]
where torch.diff(torch.tensor(sequence)) gives the length of each macro action.
This list is consumed by MacroActionValueEstimator and MacroPolicyLoss.

Paper alignments:
  - Section 3.2.1: Four termination conditions described.
  - Appendix B.4: Detailed implementation of each termination rule.
  - Appendix E: PyTorch pseudocode for get_macro_action_positions().
  - config.yaml macro_action section: All hyperparameters.

Dependencies:
    External: torch, nltk, benepar (optional), tree_sitter (optional)
    Internal: config.py (MacroActionConfig)
"""

import logging
import random
from typing import Any, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer

from config import MacroActionConfig

logger = logging.getLogger(__name__)


class MacroActionTermination:
    """Determines macro action boundaries in a generated token sequence.

    This class is instantiated once at trainer initialization (not per-call)
    because parser loading is expensive. The termination strategy is fixed
    at construction time based on config.termination.

    The primary interface is get_positions(), which returns a list of integer
    boundary indices. All four termination strategies produce the same output
    format, making them interchangeable from the perspective of downstream
    components.

    Attributes:
        config: MacroActionConfig with all termination hyperparameters.
        tokenizer: HuggingFace tokenizer for BPE encoding (used in parsing).
        termination: The active termination strategy string.
        parser: benepar constituency parser (only for NL parsing strategy).
        code_parser: tree-sitter parser (only for code parsing strategy).
        is_code_task: Whether the parsing strategy uses code AST vs NL tree.
        rng: Seeded random.Random instance for reproducible n-gram shuffling.
    """

    def __init__(
        self,
        config: MacroActionConfig,
        tokenizer: PreTrainedTokenizer,
        is_code_task: bool = False,
    ) -> None:
        """Initialize the termination strategy and load parsers if needed.

        Parser initialization is guarded with try/except because benepar
        and tree-sitter are optional dependencies. If a parser fails to
        load, the strategy falls back to fixed_ngram with a warning.

        Args:
            config: MacroActionConfig instance. The termination field
                determines which strategy is active. All hyperparameters
                (n_gram, randomized_lengths, parse_cutoff, etc.) are read
                from this object.
            tokenizer: HuggingFace PreTrainedTokenizer. Used in the parsing
                strategy to detect BPE/parser tokenizer mismatches and to
                convert parser-level positions to BPE token indices.
            is_code_task: Whether the task is code generation (APPS).
                When True and termination='parsing', uses tree-sitter AST
                instead of benepar constituency parsing. Defaults to False.
        """
        self.config: MacroActionConfig = config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.termination: str = config.termination
        self.is_code_task: bool = is_code_task

        # Parser instances (only populated for 'parsing' strategy).
        self.parser: Optional[Any] = None
        self.code_parser: Optional[Any] = None

        # Seeded RNG for reproducible randomized n-gram shuffling.
        # Fixed seed 42 ensures reproducibility across runs.
        self.rng: random.Random = random.Random(42)

        # Initialize parsers if needed.
        if self.termination == "parsing":
            if is_code_task:
                self._init_code_parser()
            else:
                self._init_nl_parser()

        logger.info(
            "MacroActionTermination initialized: termination='%s', "
            "n_gram=%s, parse_cutoff=%d, is_code_task=%s.",
            self.termination,
            str(config.n_gram),
            config.parse_cutoff,
            is_code_task,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_positions(
        self,
        start: int,
        mask: torch.Tensor,
        ppl: Optional[List[float]] = None,
        response_text: Optional[str] = None,
    ) -> List[int]:
        """Compute macro action boundary indices for a generated response.

        Dispatches to the appropriate private method based on the configured
        termination strategy. Always guarantees:
          - sequence[0] == start
          - sequence[-1] == mask.size(1) - 1
          - All elements are strictly increasing integers
          - No duplicate entries

        The returned list has length (num_macro_actions + 1). For example,
        with n=5 and response_length=13:
            sequence = [start, start+5, start+10, start+13]
            torch.diff(torch.tensor(sequence)) = [5, 5, 3]

        Args:
            start: Index where the response begins in the full sequence.
                Corresponds to prompts.size()[-1] - 1 in the PPO pseudocode
                (Appendix E). The response tokens occupy positions
                [start, mask.size(1) - 1].
            mask: Attention mask of shape [1, full_seq_len] or
                [batch_size, full_seq_len]. 1 for real tokens, 0 for padding.
                Only the first batch element is used (mask[0]).
            ppl: Per-token perplexity values for the response tokens.
                Required when termination='ppl'. ppl[i] corresponds to
                the token at position start + i. Computed from reference
                model logits in MAPPOTrainer._get_ppl(). Defaults to None.
            response_text: Decoded response string. Required when
                termination='parsing'. Decoded from response token IDs
                in MAPPOTrainer._rollout(). Defaults to None.

        Returns:
            A sorted list of unique integer boundary indices:
                [start, b1, b2, ..., end]
            where end = mask.size(1) - 1.

        Raises:
            ValueError: If termination='ppl' and ppl is None.
            ValueError: If termination='parsing' and response_text is None.
        """
        end: int = int(mask.size(1)) - 1

        # Handle degenerate case: empty response (start >= end).
        if start >= end:
            return [start, end] if start != end else [start]

        # Dispatch to the appropriate termination strategy.
        if self.termination == "fixed_ngram":
            sequence: List[int] = self._fixed_ngram(start, mask)

        elif self.termination == "randomized_ngram":
            sequence = self._randomized_ngram(start, mask)

        elif self.termination == "ppl":
            if ppl is None:
                raise ValueError(
                    "ppl must be provided when termination='ppl'. "
                    "Compute per-token perplexity from reference model logits "
                    "in MAPPOTrainer._get_ppl() before calling get_positions()."
                )
            sequence = self._ppl_based(start, ppl)

        elif self.termination == "parsing":
            if response_text is None:
                logger.warning(
                    "response_text is None for parsing termination; "
                    "falling back to fixed_ngram."
                )
                sequence = self._fixed_ngram(start, mask)
            else:
                sequence = self._parsing_based(start, mask, response_text)

        else:
            logger.warning(
                "Unknown termination strategy '%s'; falling back to fixed_ngram.",
                self.termination,
            )
            sequence = self._fixed_ngram(start, mask)

        # Guarantee sequence starts with start.
        if not sequence or sequence[0] != start:
            sequence = [start] + [x for x in sequence if x != start]

        # Guarantee sequence ends with end.
        if sequence[-1] != end:
            sequence.append(end)

        # Remove duplicates while preserving order.
        # dict.fromkeys preserves insertion order in Python 3.7+.
        sequence = list(dict.fromkeys(sequence))

        # Ensure strictly increasing (remove any out-of-order entries).
        sequence = self._ensure_sorted_unique(sequence)

        # Final guarantee: must start with start and end with end.
        if sequence[0] != start:
            sequence = [start] + sequence
        if sequence[-1] != end:
            sequence.append(end)

        return sequence

    # ------------------------------------------------------------------
    # Private termination strategy implementations
    # ------------------------------------------------------------------

    def _fixed_ngram(
        self,
        start: int,
        mask: torch.Tensor,
    ) -> List[int]:
        """Group every n consecutive valid tokens into one macro action.

        When n_gram=1, every token is its own macro action (vanilla PPO).
        When n_gram=None (infinity), the entire response is one macro action
        (REINFORCE/RLOO equivalent).

        Follows the paper's pseudocode from Appendix E:
            current_count = 0
            for i in range(mask[:, start:].size(1) - 1):
                current_count += mask[0, start + i].item()
                if current_count == n_gram:
                    sequence.append(start + i + 1)
                    current_count = 0

        Args:
            start: Response start index in the full sequence.
            mask: Attention mask of shape [1, full_seq_len] or larger.

        Returns:
            List of boundary indices starting with start. The final
            boundary (end) is appended by get_positions().
        """
        # n_gram=None means infinity: entire response is one macro action.
        if self.config.n_gram is None:
            return [start]

        n: int = self.config.n_gram
        sequence: List[int] = [start]
        current_count: int = 0

        # Iterate over response positions (excluding the final end position,
        # which get_positions() appends separately).
        response_mask_slice: torch.Tensor = mask[0, start:]
        response_len: int = response_mask_slice.size(0)

        for i in range(response_len - 1):
            # Only count non-padding tokens.
            token_valid: int = int(response_mask_slice[i].item())
            if token_valid != 0:
                current_count += 1
                if current_count == n:
                    # Complete n-gram: mark the start of the next macro action.
                    sequence.append(start + i + 1)
                    current_count = 0

        return sequence

    def _randomized_ngram(
        self,
        start: int,
        mask: torch.Tensor,
    ) -> List[int]:
        """Create variable-length macro actions from shuffled length list.

        Follows the paper's pseudocode exactly (Appendix E):
            k_list = torch.tensor([2, 3, 5, 10], dtype=int)
            k_list = torch.repeat_interleave(k_list, 3)
            k_list = k_list[torch.randperm(k_list.size()[-1])]
            indexed_k_list = torch.cumsum(k_list, dim=-1)
            sequence = [n for n in range(start, mask[:, start:].size(1) - 1)]
            indexed_k_list = [x.item() for x in indexed_k_list
                              if x.item() < len(sequence)]
            sequence = [start] + [sequence[i] for i in indexed_k_list]

        Note: torch.repeat_interleave([2,3,5,10], 3) produces
        [2,2,2,3,3,3,5,5,5,10,10,10], not [2,3,5,10,2,3,5,10,2,3,5,10].
        This matches the paper's pseudocode exactly.

        Args:
            start: Response start index in the full sequence.
            mask: Attention mask of shape [1, full_seq_len] or larger.

        Returns:
            List of boundary indices starting with start.
        """
        # Build the length list following the paper's pseudocode exactly.
        base_lengths: torch.Tensor = torch.tensor(
            self.config.randomized_lengths, dtype=torch.long
        )
        # repeat_interleave repeats each element randomized_repeat times.
        # e.g., [2,3,5,10] with repeat=3 → [2,2,2,3,3,3,5,5,5,10,10,10]
        k_list: torch.Tensor = torch.repeat_interleave(
            base_lengths, self.config.randomized_repeat
        )

        # Shuffle using torch.randperm for reproducibility with torch's RNG.
        # The paper uses torch.randperm, so we follow that exactly.
        perm: torch.Tensor = torch.randperm(k_list.size(0))
        k_list = k_list[perm]

        # Cumulative sum gives the boundary offsets from start.
        indexed_k_list: torch.Tensor = torch.cumsum(k_list, dim=0)

        # All valid positions in the response (excluding the final end position).
        # This matches the paper: sequence = [n for n in range(start, size-1)]
        full_seq_len: int = int(mask.size(1))
        all_positions: List[int] = list(range(start, full_seq_len - 1))

        # Filter: only keep offsets that fall within the valid response range.
        # indexed_k_list values are used as indices into all_positions.
        valid_indices: List[int] = [
            x.item()
            for x in indexed_k_list
            if x.item() < len(all_positions)
        ]

        # Build sequence: start + positions at the valid cumulative offsets.
        sequence: List[int] = [start] + [
            all_positions[i] for i in valid_indices
        ]

        return sequence

    def _ppl_based(
        self,
        start: int,
        ppl: List[float],
    ) -> List[int]:
        """Segment response at positions where perplexity increases.

        A macro action terminates when adding the next token would increase
        the perplexity of the current macro action. This implements the
        paper's definition from Section 3.2.1:
            "A macro action terminates until it reaches a token that has
            negative impact on the perplexity of the macro action."

        Formally: ω_τ terminates at position i when ppl[i] > ppl[i-1].

        The ppl values are computed from the reference model's logits
        (reusing the KL computation — no extra forward pass needed).
        Per-token perplexity contribution: p_t = -log π_ref(a_t | a_{<t}).

        Args:
            start: Response start index in the full sequence.
            ppl: Per-token perplexity values for response tokens.
                ppl[i] corresponds to the token at position start + i.
                Length should equal the number of response tokens.

        Returns:
            List of boundary indices starting with start.
        """
        if len(ppl) <= 1:
            # Single token or empty: one macro action.
            return [start]

        sequence: List[int] = [start]

        for i in range(1, len(ppl)):
            # Perplexity increased: current token hurts fluency.
            # Start a new macro action at this position.
            if ppl[i] > ppl[i - 1]:
                sequence.append(start + i)

        return sequence

    def _parsing_based(
        self,
        start: int,
        mask: torch.Tensor,
        response_text: str,
    ) -> List[int]:
        """Use linguistic/code structure to determine macro action boundaries.

        For NL tasks: uses benepar constituency parsing. The parse tree is
        traversed with DFS, and nodes with fewer than parse_cutoff leaf
        tokens mark macro action boundaries.

        For code tasks (APPS): uses tree-sitter AST parsing. Statement-level
        nodes in the AST mark macro action boundaries.

        Falls back to fixed_ngram when:
          - The parser is not available (import failed during __init__).
          - A tokenizer mismatch is detected (BPE count != expected count).
          - The parser fails to produce a valid tree.

        Args:
            start: Response start index in the full sequence.
            mask: Attention mask of shape [1, full_seq_len].
            response_text: Decoded response string to parse.

        Returns:
            List of boundary indices starting with start, in BPE token space.
        """
        if not response_text or not response_text.strip():
            logger.debug(
                "Empty response_text for parsing termination; "
                "falling back to fixed_ngram."
            )
            return self._fixed_ngram(start, mask)

        if self.is_code_task:
            return self._parsing_based_code(start, mask, response_text)
        else:
            return self._parsing_based_nl(start, mask, response_text)

    def _parsing_based_nl(
        self,
        start: int,
        mask: torch.Tensor,
        response_text: str,
    ) -> List[int]:
        """NL constituency parsing using benepar.

        Args:
            start: Response start index.
            mask: Attention mask.
            response_text: Decoded response string.

        Returns:
            List of BPE-level boundary indices, or fixed_ngram fallback.
        """
        if self.parser is None:
            logger.debug(
                "benepar parser not available; falling back to fixed_ngram."
            )
            return self._fixed_ngram(start, mask)

        try:
            import nltk

            # Tokenize with NLTK word tokenizer (word-level).
            word_tokens: List[str] = nltk.word_tokenize(response_text)

            if not word_tokens:
                return self._fixed_ngram(start, mask)

            # Parse with benepar to get constituency tree.
            sentence = nltk.Tree.fromstring(
                list(self.parser.parse(word_tokens))[0].__str__()
            )

            # Collect word-level macro action boundaries via DFS.
            word_sequence: List[int] = [0]
            self._dfs_parse(sentence, 0, word_sequence)

            # Verify BPE token count matches the mask.
            bpe_ids: List[int] = self.tokenizer.encode(
                response_text, add_special_tokens=False
            )
            bpe_count: int = len(bpe_ids)
            valid_token_count: int = int(mask[0, start:].sum().item())

            if bpe_count != valid_token_count:
                logger.debug(
                    "BPE tokenizer mismatch: bpe_count=%d, "
                    "valid_token_count=%d. Falling back to fixed_ngram.",
                    bpe_count,
                    valid_token_count,
                )
                return self._fixed_ngram(start, mask)

            # Build word-to-BPE position mapping.
            word_to_bpe_start: List[int] = self._build_word_to_bpe_mapping(
                word_tokens, bpe_ids
            )

            if word_to_bpe_start is None or len(word_to_bpe_start) != len(word_tokens):
                logger.debug(
                    "Word-to-BPE mapping failed; falling back to fixed_ngram."
                )
                return self._fixed_ngram(start, mask)

            # Convert word-level boundaries to BPE-level absolute positions.
            bpe_sequence: List[int] = [start]
            for word_pos in word_sequence[1:]:
                if word_pos < len(word_to_bpe_start):
                    bpe_pos: int = word_to_bpe_start[word_pos]
                    abs_pos: int = start + bpe_pos
                    bpe_sequence.append(abs_pos)
                else:
                    # word_pos == len(word_tokens): end of sequence.
                    # get_positions() will append the actual end.
                    pass

            return bpe_sequence

        except Exception as exc:
            logger.debug(
                "NL parsing failed: %s. Falling back to fixed_ngram.", exc
            )
            return self._fixed_ngram(start, mask)

    def _parsing_based_code(
        self,
        start: int,
        mask: torch.Tensor,
        response_text: str,
    ) -> List[int]:
        """Code AST parsing using tree-sitter.

        Uses statement-level AST nodes as macro action boundaries.
        Each top-level statement in the code becomes one macro action.

        Args:
            start: Response start index.
            mask: Attention mask.
            response_text: Decoded code string.

        Returns:
            List of BPE-level boundary indices, or fixed_ngram fallback.
        """
        if self.code_parser is None:
            logger.debug(
                "tree-sitter parser not available; falling back to fixed_ngram."
            )
            return self._fixed_ngram(start, mask)

        try:
            source_bytes: bytes = response_text.encode("utf-8")
            tree = self.code_parser.parse(source_bytes)
            root_node = tree.root_node

            # Collect character-level boundaries from top-level AST nodes.
            char_boundaries: List[int] = [0]
            for child in root_node.children:
                if child.type not in ("comment", "ERROR") and child.child_count > 0:
                    # Use the start byte of each top-level statement.
                    char_boundaries.append(child.start_byte)

            # Verify BPE token count.
            bpe_ids: List[int] = self.tokenizer.encode(
                response_text, add_special_tokens=False
            )
            bpe_count: int = len(bpe_ids)
            valid_token_count: int = int(mask[0, start:].sum().item())

            if bpe_count != valid_token_count:
                logger.debug(
                    "BPE tokenizer mismatch for code: bpe_count=%d, "
                    "valid_token_count=%d. Falling back to fixed_ngram.",
                    bpe_count,
                    valid_token_count,
                )
                return self._fixed_ngram(start, mask)

            # Build character-to-BPE-token mapping.
            char_to_bpe: dict = self._build_char_to_bpe_mapping(
                response_text, bpe_ids
            )

            # Convert character boundaries to BPE token positions.
            bpe_sequence: List[int] = [start]
            for char_pos in char_boundaries[1:]:
                bpe_pos: Optional[int] = char_to_bpe.get(char_pos)
                if bpe_pos is not None:
                    abs_pos: int = start + bpe_pos
                    bpe_sequence.append(abs_pos)
                else:
                    # Character position doesn't align with a BPE boundary.
                    # Fall back to fixed_ngram for safety.
                    logger.debug(
                        "Character position %d not in char_to_bpe mapping; "
                        "falling back to fixed_ngram.",
                        char_pos,
                    )
                    return self._fixed_ngram(start, mask)

            return bpe_sequence

        except Exception as exc:
            logger.debug(
                "Code parsing failed: %s. Falling back to fixed_ngram.", exc
            )
            return self._fixed_ngram(start, mask)

    # ------------------------------------------------------------------
    # DFS parse tree traversal
    # ------------------------------------------------------------------

    def _dfs_parse(
        self,
        node: Any,
        ma_length: int,
        sequence: List[int],
    ) -> Tuple[bool, int]:
        """Recursive DFS traversal of a constituency parse tree.

        Implements the paper's pseudocode from Appendix E:
            if len(node.leaves()) < 1:
                return False, ma_length + 1
            if len(node.leaves()) < cutoff:
                sequence.append(ma_length + node.leaves())
                return True, ma_length + node.leaves()
            for nxt_node in node.childs():
                state, ma_length_ = dfs(nxt_node, ma_length)
                if !state:
                    sequence[-1] = ma_length_
                    ma_length = ma_length_
            return True, ma_length

        Two termination rules (Appendix B.4):
          1. Nodes with fewer than parse_cutoff leaf tokens mark the end
             of a macro action.
          2. Single-token nodes are merged into the previous macro action
             (avoiding punctuation isolation).

        Args:
            node: An nltk.Tree node (for NL) or similar tree node.
            ma_length: Current cumulative token count (word-level position).
            sequence: Mutable list of word-level boundary positions.
                Modified in-place by appending new boundaries.

        Returns:
            A tuple (state, ma_length) where:
                state: True if this node produced a complete macro action
                    boundary, False if it was a single token merged into
                    the previous macro action.
                ma_length: Updated cumulative token count after processing
                    this node.
        """
        try:
            import nltk

            # Count leaf tokens under this node.
            if isinstance(node, nltk.Tree):
                leaves: List[str] = node.leaves()
                leaf_count: int = len(leaves)
            else:
                # Leaf node (string token): no children.
                leaf_count = 1

        except ImportError:
            # nltk not available; treat as single token.
            leaf_count = 1

        # Base case: empty node (shouldn't occur in valid parse trees).
        if leaf_count < 1:
            return False, ma_length + 1

        # Termination condition: node has fewer than parse_cutoff leaves.
        # This node's tokens form one complete macro action.
        if leaf_count < self.config.parse_cutoff:
            new_ma_length: int = ma_length + leaf_count
            sequence.append(new_ma_length)
            return True, new_ma_length

        # Recurse into children for larger nodes.
        try:
            import nltk

            if isinstance(node, nltk.Tree):
                children = list(node)
            else:
                # Leaf node with leaf_count >= parse_cutoff (shouldn't happen).
                sequence.append(ma_length + 1)
                return True, ma_length + 1
        except ImportError:
            sequence.append(ma_length + 1)
            return True, ma_length + 1

        for child in children:
            state: bool
            ma_length_new: int
            state, ma_length_new = self._dfs_parse(child, ma_length, sequence)

            if not state:
                # Child returned False (single token): merge into previous
                # macro action by updating the last sequence entry.
                if sequence:
                    sequence[-1] = ma_length_new
                else:
                    sequence.append(ma_length_new)
                ma_length = ma_length_new
            else:
                ma_length = ma_length_new

        return True, ma_length

    # ------------------------------------------------------------------
    # Parser initialization helpers
    # ------------------------------------------------------------------

    def _init_nl_parser(self) -> None:
        """Initialize the benepar constituency parser for NL tasks.

        Downloads the benepar_en3 model if not already present.
        Sets self.parser to the initialized parser, or None on failure.
        """
        try:
            import benepar
            import nltk

            # Download benepar model if not already present.
            try:
                benepar.download("benepar_en3")
            except Exception as download_exc:
                logger.debug(
                    "benepar.download('benepar_en3') raised: %s. "
                    "Assuming model is already downloaded.",
                    download_exc,
                )

            # Initialize the parser.
            # benepar.BeneparComponent is used with spacy, but for standalone
            # use we use the nltk interface via benepar.Parser.
            self.parser = benepar.Parser("benepar_en3")

            logger.info(
                "benepar constituency parser initialized (benepar_en3)."
            )

        except ImportError:
            logger.warning(
                "benepar not installed. Parsing-based termination will "
                "fall back to fixed_ngram. Install with: pip install benepar"
            )
            self.parser = None
        except Exception as exc:
            logger.warning(
                "Failed to initialize benepar parser: %s. "
                "Parsing-based termination will fall back to fixed_ngram.",
                exc,
            )
            self.parser = None

    def _init_code_parser(self) -> None:
        """Initialize the tree-sitter Python parser for code tasks (APPS).

        Sets self.code_parser to the initialized parser, or None on failure.
        """
        try:
            import tree_sitter_python as tspython
            from tree_sitter import Language, Parser

            # Build the Python language grammar.
            PY_LANGUAGE = Language(tspython.language())

            # Initialize the parser with the Python language.
            self.code_parser = Parser(PY_LANGUAGE)

            logger.info(
                "tree-sitter Python parser initialized for code generation."
            )

        except ImportError:
            logger.warning(
                "tree-sitter or tree-sitter-python not installed. "
                "Code parsing-based termination will fall back to fixed_ngram. "
                "Install with: pip install tree-sitter tree-sitter-python"
            )
            self.code_parser = None
        except Exception as exc:
            logger.warning(
                "Failed to initialize tree-sitter parser: %s. "
                "Code parsing-based termination will fall back to fixed_ngram.",
                exc,
            )
            self.code_parser = None

    # ------------------------------------------------------------------
    # Tokenizer alignment helpers
    # ------------------------------------------------------------------

    def _build_word_to_bpe_mapping(
        self,
        word_tokens: List[str],
        bpe_ids: List[int],
    ) -> Optional[List[int]]:
        """Build a mapping from word token index to BPE token start index.

        For each word token at index i, word_to_bpe_start[i] gives the
        index of the first BPE token that corresponds to that word.

        This mapping is used to convert word-level parse tree boundaries
        to BPE-level positions for the macro action sequence.

        Args:
            word_tokens: List of word-level tokens from NLTK tokenizer.
            bpe_ids: List of BPE token IDs from the HuggingFace tokenizer.

        Returns:
            A list of length len(word_tokens) where element i is the
            BPE start index for word i. Returns None if the mapping
            cannot be constructed (tokenizer mismatch).
        """
        if not word_tokens or not bpe_