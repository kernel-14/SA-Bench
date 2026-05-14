"""
Shared utilities for MA-RLHF.

Includes:
  - Seed setting
  - Logging helpers
  - Checkpoint save/load
  - DeepSpeed / distributed training helpers
  - Constituent tree parsing (for parsing-based termination)
  - Perplexity computation helpers
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    step: int,
    extra_state: Optional[Dict[str, Any]] = None,
):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    if extra_state:
        with open(os.path.join(ckpt_dir, "extra_state.json"), "w") as f:
            json.dump(extra_state, f)
    logger.info(f"Checkpoint saved to {ckpt_dir}")


def load_checkpoint(model, ckpt_dir: str, device: torch.device):
    state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded checkpoint from {ckpt_dir}")


# ---------------------------------------------------------------------------
# Distributed training helpers
# ---------------------------------------------------------------------------

def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Constituent tree parsing (for parsing-based macro action termination, §B.4)
# ---------------------------------------------------------------------------

_parser = None


def get_constituency_parser():
    """Lazy-load the benepar constituency parser."""
    global _parser
    if _parser is None:
        try:
            import benepar
            import spacy
            nlp = spacy.load("en_core_web_sm")
            if spacy.__version__.startswith("3"):
                nlp.add_pipe("benepar", config={"model": "benepar_en3"})
            else:
                nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
            _parser = nlp
        except Exception as e:
            logger.warning(f"Could not load benepar parser: {e}. Falling back to n-gram.")
            _parser = None
    return _parser


def parse_to_constituent_tree(text: str):
    """Parse text to a constituent tree using benepar.

    Returns a nltk.Tree or None if parsing fails.
    """
    nlp = get_constituency_parser()
    if nlp is None:
        return None
    try:
        import nltk
        doc = nlp(text)
        sent = list(doc.sents)[0]
        tree = sent._.parse_string
        return nltk.Tree.fromstring(tree)
    except Exception as e:
        logger.debug(f"Parsing failed: {e}")
        return None


def parse_code_to_tree(code: str):
    """Parse Python code to an AST-based constituent structure.

    Used for parsing-based termination on APPS (§D.2).
    Returns a simplified tree structure compatible with _dfs_constituent.
    """
    import ast

    class SimpleNode:
        def __init__(self, children, n_leaves):
            self._children = children
            self._n_leaves = n_leaves

        def leaves(self):
            return list(range(self._n_leaves))

        def __iter__(self):
            return iter(self._children)

    def ast_to_simple(node, tokens):
        if isinstance(node, ast.AST):
            children = [ast_to_simple(c, tokens) for c in ast.iter_child_nodes(node)]
            n_leaves = sum(c._n_leaves for c in children) if children else 1
            return SimpleNode(children, n_leaves)
        return SimpleNode([], 1)

    try:
        tree = ast.parse(code)
        tokens = code.split()
        return ast_to_simple(tree, tokens)
    except SyntaxError:
        return None


# ---------------------------------------------------------------------------
# Perplexity utilities (for PPL-based termination, §B.4)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_sequence_perplexity(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    start: int,
    end: Optional[int] = None,
) -> float:
    """Compute perplexity of the response tokens [start:end].

    ppl = exp(-1/T * Σ log p(a_t | a_{<t}))
    """
    import torch.nn.functional as F

    if end is None:
        end = input_ids.size(1)

    log_probs = F.log_softmax(logits[0, start - 1 : end - 1], dim=-1)
    target = input_ids[0, start:end]
    token_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    avg_neg_log_prob = -token_log_probs.mean().item()
    return float(np.exp(avg_neg_log_prob))


# ---------------------------------------------------------------------------
# Response decoding
# ---------------------------------------------------------------------------

def decode_response(
    tokenizer: PreTrainedTokenizer,
    generated_ids: torch.Tensor,
    prompt_len: int,
) -> str:
    """Decode only the response portion of generated_ids."""
    response_ids = generated_ids[0, prompt_len:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Running average tracker for training metrics."""

    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0) + 1

    def averages(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()

    def log(self, step: int, prefix: str = ""):
        avgs = self.averages()
        parts = [f"{prefix}step={step}"] + [f"{k}={v:.4f}" for k, v in avgs.items()]
        logger.info(" | ".join(parts))


# ---------------------------------------------------------------------------
# Agreement computation (Table 1)
# ---------------------------------------------------------------------------

def compute_agreement(
    labels_a: List[str],
    labels_b: List[str],
) -> float:
    """Compute pairwise agreement between two sets of binary labels."""
    assert len(labels_a) == len(labels_b)
    matches = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    return matches / len(labels_a)


# ---------------------------------------------------------------------------
# Scaling trend helpers (Figure 9)
# ---------------------------------------------------------------------------

def format_scaling_results(
    model_sizes: List[str],
    rm_scores: List[float],
    gpt4_win_rates: List[float],
    human_win_rates: List[float],
) -> Dict:
    """Format scaling trend results for logging/plotting."""
    return {
        "model_sizes": model_sizes,
        "rm_scores": rm_scores,
        "gpt4_win_rates": gpt4_win_rates,
        "human_win_rates": human_win_rates,
    }
