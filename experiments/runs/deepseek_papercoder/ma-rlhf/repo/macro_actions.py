# macro_actions.py
"""
MacroActionModule: Core logic for segmenting token sequences into macro actions
and aggregating token-level statistics for MA-RLHF.

Supports termination strategies: fixed n-gram, randomized n-gram, parsing-based,
and perplexity-based, as described in the paper "MA-RLHF: Reinforcement Learning
from Human Feedback with Macro Actions".

All operations are designed to be used within the PPO training loop.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer

# Lazy imports for parsing-based termination
try:
    import spacy
    import benepar
    HAS_BENEPAR = True
except ImportError:
    HAS_BENEPAR = False

logger = logging.getLogger(__name__)


class MacroActionModule:
    """
    Handles the creation of macro-action boundaries and the aggregation of
    token-level values/rewards into macro-level values/rewards.

    Args:
        tokenizer: HuggingFace tokenizer for the language model.
        termination: One of 'fixed_ngram', 'randomized_ngram', 'parsing', 'perplexity'.
        n: Length of macro action for fixed n-gram (ignored for other strategies).
        cutoff: Maximum leaf tokens for parsing-based termination (default C=5).
        lengths_pool: List of integer lengths for randomized n-gram (default [2,3,5,10]).
        randomized_repeat: Number of times to repeat the lengths_pool before shuffling
                           (default 3, as per paper).
        weighting: Aggregation weighting scheme for macro values: 'equal', 'unit',
                   or 'position_decayed' (default 'equal').
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        termination: str,
        n: int,
        cutoff: int,
        lengths_pool: Optional[List[int]] = None,
        randomized_repeat: int = 3,
        weighting: str = "equal",
    ):
        if termination not in ("fixed_ngram", "randomized_ngram", "parsing", "perplexity"):
            raise ValueError(f"Unknown termination strategy: {termination}")
        if weighting not in ("equal", "unit", "position_decayed"):
            raise ValueError(f"Unknown weighting scheme: {weighting}")

        self.tokenizer = tokenizer
        self.termination = termination
        self.n = n
        self.cutoff = cutoff
        self.lengths_pool = lengths_pool if lengths_pool is not None else [2, 3, 5, 10]
        self.randomized_repeat = randomized_repeat
        self.weighting = weighting

        # Lazy-loaded NLP resources for parsing-based termination
        self._nlp = None
        self._parser = None

        # Validate that required resources are available if needed
        if termination == "parsing" and not HAS_BENEPAR:
            raise ImportError(
                "Parsing-based termination requires spaCy and benepar. "
                "Install them with: pip install spacy benepar && python -m spacy download en_core_web_sm"
            )

    def _setup_parser(self):
        """Initialize spaCy + benepar parser (only when parsing termination is used)."""
        if self._nlp is not None:
            return
        if not HAS_BENEPAR:
            raise RuntimeError("spaCy/benepar not installed; cannot use parsing termination.")
        try:
            self._nlp = spacy.load("en_core_web_sm")
            self._nlp.add_pipe("benepar", config={"model": "benepar_en3"})
            logger.info("Loaded en_core_web_sm with benepar pipeline.")
        except Exception as e:
            logger.error(f"Failed to load spaCy/benepar: {e}")
            raise

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_boundaries(
        self,
        response_tokens: List[int],
        ref_logprobs: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Compute macro-action boundaries for a single response. The boundaries are
        half-open intervals (start, end) relative to the start of the response tokens.

        Args:
            response_tokens: List of token IDs of the generated response (after the prompt).
            ref_logprobs: 1‑D tensor of per‑token log‑probabilities under the reference model.
                          Required for 'perplexity' termination.
            text: Decoded string of the response. Required for 'parsing' termination.

        Returns:
            List of (start, end) tuples, or None if the segmentation fails (e.g., parsing mismatch).
        """
        length = len(response_tokens)
        if length == 0:
            return []

        if self.termination == "fixed_ngram":
            return self._fixed_ngram_boundaries(0, length, self.n)
        elif self.termination == "randomized_ngram":
            return self._randomized_ngram_boundaries(0, length, self.lengths_pool)
        elif self.termination == "parsing":
            if text is None:
                raise ValueError("Parsing termination requires the decoded text.")
            return self._parsing_boundaries(text, response_tokens)
        elif self.termination == "perplexity":
            if ref_logprobs is None:
                raise ValueError("Perplexity termination requires ref_logprobs tensor.")
            return self._perplexity_boundaries(ref_logprobs, 0, length)
        else:
            # Should not reach here
            raise ValueError(f"Unknown termination strategy: {self.termination}")

    def aggregate_values(
        self,
        values: torch.Tensor,
        boundaries: List[Tuple[int, int]],
        weighting: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Aggregate token‑level value predictions (shape (1, L)) into macro‑level values
        (shape (1, num_macro_actions)) using the specified weighting scheme.

        Args:
            values: Token‑level values for the response part, shape (1, L).
            boundaries: List of (start, end) intervals from get_boundaries.
            weighting: Override the default weighting scheme (self.weighting).

        Returns:
            Tensor of macro‑level values with shape (1, len(boundaries)).
        """
        w = weighting if weighting is not None else self.weighting
        macro_vals = []
        for start, end in boundaries:
            seg = values[:, start:end]  # shape (1, seg_len)
            if seg.numel() == 0:
                macro_vals.append(torch.zeros(1, device=values.device))
                continue

            if w == "equal":
                val = seg.mean(dim=1)  # shape (1,)
            elif w == "unit":
                val = seg[:, -1]  # last token value
            elif w == "position_decayed":
                seg_len = end - start
                raw_weights = 1.0 / torch.arange(seg_len, 0, -1, device=values.device, dtype=values.dtype)
                weights = raw_weights / raw_weights.sum()
                # seg is (1, seg_len); weights is (seg_len,)
                val = (seg.squeeze(0) * weights).sum().unsqueeze(0)
            else:
                raise ValueError(f"Unknown weighting: {w}")
            macro_vals.append(val)

        if len(macro_vals) == 0:
            return torch.zeros(1, 0, device=values.device)
        return torch.stack(macro_vals, dim=1)  # (1, num_macro)

    def aggregate_rewards(
        self,
        rewards: torch.Tensor,
        boundaries: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Sum token‑level rewards (shape (1, L)) within each macro action.
        The paper uses discount factor ρ=1, so simple sum.

        Args:
            rewards: Token‑level rewards for the response part, shape (1, L).
            boundaries: List of (start, end) intervals from get_boundaries.

        Returns:
            Tensor of macro‑level rewards with shape (1, len(boundaries)).
        """
        macro_rewards = []
        for start, end in boundaries:
            seg = rewards[:, start:end]
            if seg.numel() == 0:
                macro_rewards.append(torch.zeros(1, device=rewards.device))
            else:
                macro_rewards.append(seg.sum(dim=1))
        if len(macro_rewards) == 0:
            return torch.zeros(1, 0, device=rewards.device)
        return torch.stack(macro_rewards, dim=1)

    # ------------------------------------------------------------------
    # Private segmentation methods
    # ------------------------------------------------------------------

    def _fixed_ngram_boundaries(self, start: int, length: int, n: int) -> List[Tuple[int, int]]:
        """Create boundaries for fixed n-gram macro actions."""
        boundaries = []
        for i in range(start, start + length, n):
            end = min(i + n, start + length)
            boundaries.append((i, end))
        return boundaries

    def _randomized_ngram_boundaries(
        self, start: int, length: int, lengths_pool: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Create boundaries by randomly shuffling a repeated pool of lengths.
        Remaining tokens form a final macro action.
        """
        import random
        # Build the list K by repeating lengths_pool self.randomized_repeat times and shuffling
        K = lengths_pool * self.randomized_repeat
        random.shuffle(K)

        boundaries = []
        curr = start
        for l in K:
            if curr + l > start + length:
                # The next segment would exceed the sequence; break and assign remainder later
                break
            boundaries.append((curr, curr + l))
            curr += l
        # Remaining tokens
        if curr < start + length:
            boundaries.append((curr, start + length))
        return boundaries

    def _perplexity_boundaries(
        self, ref_logprobs: torch.Tensor, start: int, length: int
    ) -> List[Tuple[int, int]]:
        """
        Greedy segmentation where a new macro action is created whenever adding
        a token would increase the macro action's perplexity (i.e., decrease the
        average log-probability).
        """
        boundaries = []
        seg_start = start
        sum_logp = 0.0
        cnt = 0

        for i in range(start, start + length):
            logp = ref_logprobs[i].item()
            if cnt == 0:
                sum_logp = logp
                cnt = 1
            else:
                new_avg = (sum_logp + logp) / (cnt + 1)
                old_avg = sum_logp / cnt
                if new_avg < old_avg:  # perplexity increased, stop before i
                    boundaries.append((seg_start, i))
                    # start new segment with current token
                    seg_start = i
                    sum_logp = logp
                    cnt = 1
                else:
                    sum_logp += logp
                    cnt += 1

        if cnt > 0:
            boundaries.append((seg_start, start + length))
        return boundaries

    def _parsing_boundaries(
        self, text: str, response_tokens: List[int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Segment using constituency parsing.  The response text is parsed and the tree is
        traversed depth‑first.  Each subtree that contains ≤ self.cutoff leaf tokens
        becomes a macro action; smaller single-leaf punctuation nodes are merged into the
        previous macro action.
        Returns None if token alignment fails.
        """
        self._setup_parser()

        # Tokenize the text with offset mapping to align spacy leaves with subword tokens
        try:
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=len(response_tokens),
                return_offsets_mapping=True,
                return_tensors=None,
            )
        except Exception as e:
            logger.warning(f"Failed to tokenize for parsing boundaries: {e}")
            return None

        input_ids = tokenized["input_ids"]
        offset_mapping = tokenized["offset_mapping"]  # list of (char_start, char_end)
        if len(input_ids) != len(response_tokens):
            logger.warning("Tokenized length differs from response_tokens; alignment impossible.")
            return None

        # Parse the text with spaCy
        try:
            doc = self._nlp(text)
        except Exception as e:
            logger.warning(f"spaCy parsing failed: {e}")
            return None

        # Helper to map a spacy token (character span) to subword token indices
        def _leaf_to_token_range(spacy_token) -> Tuple[int, int]:
            """Returns (start_inclusive, end_exclusive) subword indices covering this spacy token."""
            char_start = spacy_token.idx
            char_end = char_start + len(spacy_token.text)
            # Find subword tokens whose offset lies within this character span
            sub_starts = []
            for idx, (off_start, off_end) in enumerate(offset_mapping):
                if off_start >= char_end or off_end <= char_start:
                    continue  # no overlap
                sub_starts.append(idx)
            if not sub_starts:
                raise ValueError(f"Could not map spacy token '{spacy_token.text}' to any subword token.")
            start_idx = sub_starts[0]
            end_idx = sub_starts[-1] + 1  # inclusive end -> exclusive
            return start_idx, end_idx

        # Recursively collect macro action spans from a constituent tree node.
        # The tree leaves are spacy Token objects.
        # Returns a list of (token_start, token_end, is_single_leaf) for macro actions.
        def _collect_spans(node, is_leaf_flag=False):
            if isinstance(node, str):
                # This should not happen because benepar trees have Token leaves
                return []
            leaves = node.leaves()
            leaf_count = len(leaves)
            if leaf_count == 0:
                return []

            # Base: if leaf count <= cutoff, whole node becomes a macro action
            if leaf_count <= self.cutoff:
                # Determine overall token range from the first and last leaf
                try:
                    start_token, _ = _leaf_to_token_range(leaves[0])
                    _, end_token = _leaf_to_token_range(leaves[-1])
                except ValueError:
                    return None  # alignment failure
                # A node is considered "single leaf" if it has exactly one leaf
                is_single = (leaf_count == 1)
                return [(start_token, end_token, is_single)]

            # Otherwise, recursively process children
            spans = []
            for child in node:
                res = _collect_spans(child)
                if res is None:
                    return None
                spans.extend(res)
            return spans

        # Process each sentence
        all_spans = []
        for sent in doc.sents:
            try:
                tree = sent._.constituent
            except AttributeError:
                logger.warning("Benepar constituent tree not available for sentence.")
                return None
            sent_spans = _collect_spans(tree)
            if sent_spans is None:
                return None
            all_spans.extend(sent_spans)

        if not all_spans:
            return []  # no spans – maybe empty text? this should not happen

        # Post‑processing: merge single‑leaf spans (e.g., punctuation) into previous macro action
        merged = []
        for span in all_spans:
            start, end, is_single = span
            if is_single and merged:
                # merge with previous
                prev_start, prev_end, _ = merged.pop()
                merged.append((prev_start, end, False))
            else:
                merged.append((start, end, is_single))

        # Extract only the (start, end) tuples
        boundaries = [(s, e) for (s, e, _) in merged]

        # Validate that boundaries cover the entire response
        # Should be contiguous and start at 0, end at len(response_tokens)
        if not boundaries:
            return []  # empty, but that's fine if 0-length response
        if boundaries[0][0] != 0:
            logger.warning("Parsed macro actions do not start at token 0; alignment mismatch.")
            return None
        if boundaries[-1][1] != len(response_tokens):
            logger.warning("Parsed macro actions do not cover the entire response; alignment mismatch.")
            return None
        for i in range(1, len(boundaries)):
            if boundaries[i][0] != boundaries[i-1][1]:
                logger.warning("Gaps between parsed macro actions.")
                return None

        return boundaries
