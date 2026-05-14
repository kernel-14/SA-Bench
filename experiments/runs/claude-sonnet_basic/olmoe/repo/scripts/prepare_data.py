"""
Data Preparation for OLMoE Pretraining

Implements the data filtering described in Section 2 of the paper:
1. Remove documents with 32+ repeated n-grams (n=1 to 13 tokens)
2. StarCoder-specific filters:
   - Remove repos with < 2 GitHub stars
   - Remove docs where top-1 word > 30% of document
   - Remove docs where top-2 words > 50% of document

OLMoE-Mix composition (Table 2):
- DCLM-Baseline: 3,860B tokens (web pages)
- StarCoder: 101B tokens (code)
- peS2o: 57.2B tokens (STEM papers)
- arXiv: 21.1B tokens (STEM papers)
- OpenWebMath: 12.7B tokens (math web pages)
- Algebraic Stack: 12.6B tokens (math proofs/code)
- Wikipedia+Wikibooks: 3.69B tokens (encyclopedic)
Total: ~4,060B tokens
"""

import re
import json
import argparse
from collections import Counter
from typing import List, Optional, Iterator
import logging

logger = logging.getLogger(__name__)


def has_repeated_ngrams(
    tokens: List[int],
    max_repeated: int = 32,
    max_ngram_size: int = 13,
) -> bool:
    """
    Check if a document has a sequence of 32+ repeated n-grams.

    From Section 2 of the paper:
    "we apply a filter that removes all documents with a sequence of 32 or more
    repeated n-grams, where an n-gram is any span of 1 to 13 tokens"

    Args:
        tokens: List of token IDs
        max_repeated: Maximum allowed consecutive repetitions (default 32)
        max_ngram_size: Maximum n-gram size to check (default 13)

    Returns:
        True if the document should be filtered out (has too many repeated n-grams)
    """
    n = len(tokens)

    for ngram_size in range(1, max_ngram_size + 1):
        # Check for sequences of repeated n-grams
        consecutive_count = 1
        i = ngram_size

        while i <= n - ngram_size:
            # Check if current n-gram matches previous n-gram
            current_ngram = tokens[i:i + ngram_size]
            prev_ngram = tokens[i - ngram_size:i]

            if current_ngram == prev_ngram:
                consecutive_count += 1
                if consecutive_count >= max_repeated:
                    return True
                i += ngram_size
            else:
                consecutive_count = 1
                i += 1

    return False


def filter_starcoder_document(
    text: str,
    github_stars: Optional[int] = None,
    min_stars: int = 2,
    max_top1_freq: float = 0.30,
    max_top2_freq: float = 0.50,
) -> bool:
    """
    Apply StarCoder-specific filters.

    From Section 2 of the paper:
    "For the StarCoder subset, we also remove any document from a repository with
    fewer than 2 stars on GitHub, whose most frequent word constitutes over 30% of
    the document, or whose top-2 most frequent words constitute over 50% of the document."

    Args:
        text: Document text
        github_stars: Number of GitHub stars for the repository
        min_stars: Minimum required stars (default 2)
        max_top1_freq: Maximum frequency for top-1 word (default 0.30)
        max_top2_freq: Maximum frequency for top-2 words combined (default 0.50)

    Returns:
        True if the document should be filtered out
    """
    # Filter by GitHub stars
    if github_stars is not None and github_stars < min_stars:
        return True

    # Compute word frequencies
    words = text.lower().split()
    if not words:
        return False

    word_counts = Counter(words)
    total_words = len(words)

    # Get top-2 most frequent words
    top_words = word_counts.most_common(2)

    if top_words:
        top1_freq = top_words[0][1] / total_words
        if top1_freq > max_top1_freq:
            return True

    if len(top_words) >= 2:
        top2_freq = (top_words[0][1] + top_words[1][1]) / total_words
        if top2_freq > max_top2_freq:
            return True

    return False


def filter_document(
    text: str,
    tokens: Optional[List[int]] = None,
    source: str = "web",
    github_stars: Optional[int] = None,
    tokenizer=None,
) -> bool:
    """
    Apply all document filters.

    Args:
        text: Document text
        tokens: Pre-tokenized document (if available)
        source: Data source ("web", "code", "papers", "math", "encyclopedic")
        github_stars: GitHub stars (for StarCoder)
        tokenizer: Tokenizer to use if tokens not provided

    Returns:
        True if the document should be filtered out
    """
    # Tokenize if needed
    if tokens is None and tokenizer is not None:
        tokens = tokenizer.encode(text)

    # Apply n-gram repetition filter to all sources
    if tokens is not None:
        if has_repeated_ngrams(tokens):
            return True

    # Apply StarCoder-specific filters
    if source == "code":
        if filter_starcoder_document(text, github_stars=github_stars):
            return True

    return False


def process_dataset(
    input_path: str,
    output_path: str,
    source: str = "web",
    tokenizer=None,
    max_documents: Optional[int] = None,
) -> dict:
    """
    Process a dataset file, applying all filters.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        source: Data source type
        tokenizer: Tokenizer for n-gram filtering
        max_documents: Maximum number of documents to process

    Returns:
        Statistics about filtering
    """
    stats = {
        "total": 0,
        "filtered_ngram": 0,
        "filtered_starcoder": 0,
        "kept": 0,
    }

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for i, line in enumerate(fin):
            if max_documents and i >= max_documents:
                break

            stats["total"] += 1

            try:
                doc = json.loads(line)
                text = doc.get("text", doc.get("content", ""))
                github_stars = doc.get("stars", None)

                # Tokenize for n-gram filter
                tokens = None
                if tokenizer is not None:
                    tokens = tokenizer.encode(text)

                # Check n-gram filter
                if tokens is not None and has_repeated_ngrams(tokens):
                    stats["filtered_ngram"] += 1
                    continue

                # Check StarCoder filter
                if source == "code" and filter_starcoder_document(
                    text, github_stars=github_stars
                ):
                    stats["filtered_starcoder"] += 1
                    continue

                # Document passes all filters
                fout.write(line)
                stats["kept"] += 1

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse line {i}")
                continue

    stats["filter_rate"] = 1 - stats["kept"] / max(1, stats["total"])
    return stats


def create_olmoe_mix_config() -> dict:
    """
    Create the OLMoE-Mix data configuration.

    Returns the composition of the pretraining dataset as described in Table 2.
    """
    return {
        "sources": [
            {
                "name": "DCLM-Baseline",
                "hf_path": "mlfoundations/dclm-baseline-1.0",
                "type": "web",
                "tokens_billions": 3860,
                "weight": 3860 / 4068.29,
            },
            {
                "name": "StarCoder",
                "hf_path": "bigcode/starcoderdata",
                "type": "code",
                "tokens_billions": 101,
                "weight": 101 / 4068.29,
                "filters": {
                    "min_github_stars": 2,
                    "max_top1_word_freq": 0.30,
                    "max_top2_word_freq": 0.50,
                }
            },
            {
                "name": "peS2o",
                "hf_path": "allenai/peS2o",
                "type": "papers",
                "tokens_billions": 57.2,
                "weight": 57.2 / 4068.29,
            },
            {
                "name": "arXiv",
                "hf_path": "allenai/dolma",
                "subset": "arxiv",
                "type": "papers",
                "tokens_billions": 21.1,
                "weight": 21.1 / 4068.29,
            },
            {
                "name": "OpenWebMath",
                "hf_path": "open-web-math/open-web-math",
                "type": "math",
                "tokens_billions": 12.7,
                "weight": 12.7 / 4068.29,
            },
            {
                "name": "Algebraic Stack",
                "hf_path": "EleutherAI/proof-pile-2",
                "subset": "algebraic-stack",
                "type": "math",
                "tokens_billions": 12.6,
                "weight": 12.6 / 4068.29,
            },
            {
                "name": "Wikipedia+Wikibooks",
                "hf_path": "allenai/dolma",
                "subset": "wiki",
                "type": "encyclopedic",
                "tokens_billions": 3.69,
                "weight": 3.69 / 4068.29,
            },
        ],
        "total_tokens_billions": 4068.29,
        "training_tokens_billions": 5133,  # 1.3 epochs
        "annealing_tokens_billions": 100,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare OLMoE training data")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--source", default="web",
                        choices=["web", "code", "papers", "math", "encyclopedic"])
    parser.add_argument("--max-documents", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    stats = process_dataset(
        args.input,
        args.output,
        source=args.source,
        max_documents=args.max_documents,
    )

    logger.info(f"Processing complete:")
    logger.info(f"  Total documents: {stats['total']}")
    logger.info(f"  Filtered (n-gram): {stats['filtered_ngram']}")
    logger.info(f"  Filtered (StarCoder): {stats['filtered_starcoder']}")
    logger.info(f"  Kept: {stats['kept']}")
    logger.info(f"  Filter rate: {stats['filter_rate']:.2%}")
