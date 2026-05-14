# evaluate.py
"""
Evaluator: Offline evaluation module for MA‑RLHF and vanilla RLHF trained policies.

Provides methods to:
  - compute reward model (RM) scores on validation sets,
  - run GPT‑4 pairwise judgements,
  - compute pass@k for code generation (APPS).

All evaluation settings are derived from the experiment configuration (config.yaml).
The class is designed to be initialised once and then used for multiple evaluation calls.
"""

import logging
import json
import math
import re
import subprocess
import tempfile
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from datasets import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> str:
    """Extract a Python code block from a model's completion."""
    # Try to find a fenced code block with python specifier
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    # Try generic code fence
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    # Fallback: assume the whole output is code (after removing leading/trailing whitespace)
    # Remove any preceding natural language (e.g., "Here is the solution:")
    lines = text.strip().split("\n")
    code_lines = []
    started = False
    for line in lines:
        if line.strip().startswith("def ") or line.strip().startswith("class ") or line.strip().startswith("import "):
            started = True
        if started:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines)
    # If nothing else, return the stripped text
    return text.strip()


def _execute_code_against_tests(code: str, input_list: List[str], output_list: List[str], timeout: float = 10.0) -> bool:
    """
    Run the Python code in a subprocess, feed each input and compare output to expected.
    Returns True if all test cases pass, False otherwise.
    """
    # Write the code to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        # Build a wrapper script that runs the code for each test case
        # We'll create a simple harness that imports the code as a module? Safer: run code and capture prints.
        # Since coding problems often read from stdin and write to stdout,
        # we'll simulate that via subprocess with input fed as string.
        all_pass = True
        for inp, out in zip(input_list, output_list):
            proc = subprocess.run(
                ["python", tmp_path],
                input=inp.strip(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                # runtime error or non‑zero exit
                all_pass = False
                break
            # Compare output
            expected = out.strip()
            actual = proc.stdout.strip()
            # Some tolerance for whitespace differences
            if actual != expected:
                # Try floating point comparison? Many APPS problems output numbers.
                # Basic float tolerance: split and compare numerically if possible.
                # If both can be parsed as floats, compare with relative error 1e-6.
                try:
                    expected_floats = [float(x) for x in expected.split()]
                    actual_floats = [float(x) for x in actual.split()]
                    if len(expected_floats) != len(actual_floats):
                        all_pass = False
                        break
                    for e, a in zip(expected_floats, actual_floats):
                        if math.isclose(e, a, rel_tol=1e-6, abs_tol=1e-6):
                            continue
                        else:
                            all_pass = False
                            break
                except ValueError:
                    all_pass = False
                    break
        return all_pass

    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        logger.warning(f"Exception during code execution: {e}")
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Offline evaluation of a trained RLHF policy.

    Args:
        actor_path: Path to the fine‑tuned actor model (HuggingFace checkpoint).
        reward_path: Path to the trained reward model (None if not needed, e.g., for APPS).
        tokenizer: Pre‑trained tokenizer matching the actor.
        config: Experiment configuration dictionary (from config.yaml, per experiment).
    """

    def __init__(
        self,
        actor_path: str,
        reward_path: Optional[str],
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
    ):
        self.config = config
        self.dataset_name = config.get("dataset_name", "tldr")
        self.tokenizer = tokenizer

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA for evaluation.")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for evaluation.")

        # Load actor model
        logger.info(f"Loading actor from {actor_path}")
        self.actor = AutoModelForCausalLM.from_pretrained(
            actor_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.actor.eval()

        # Load reward model (if applicable)
        self.reward_model = None
        if reward_path is not None:
            logger.info(f"Loading reward model from {reward_path}")
            # First, try loading as a custom RewardModel (from rm_trainer.py)
            # Since reward model may have been saved as a custom class, we need to handle both.
            # We'll attempt to load using AutoModelForSequenceClassification first, as it's common.
            try:
                self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                    reward_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
            except Exception as e:
                logger.warning(f"Failed to load reward model as sequence classification: {e}")
                # Fallback: try loading as AutoModel and adding a head manually, or load a custom state dict.
                raise RuntimeError("Reward model loading failed; check the saved format.")

            self.reward_model.eval()
        else:
            logger.info("No reward model provided; RM scoring will be unavailable.")

        # Generation parameters
        ppo_cfg = config.get("ppo", {})
        self.temperature = ppo_cfg.get("temperature", 0.8)
        self.top_p = ppo_cfg.get("top_p", 1.0)
        self.top_k = ppo_cfg.get("top_k", 50)
        self.max_prompt_length = config.get("max_prompt_length", 512)
        self.max_response_length = config.get("max_response_length", 512)

        # For code evaluation
        self.reward_type = ppo_cfg.get("reward_type", "model")
        # Number of samples for unbiased pass@k (default 200, but can be overridden)
        self.eval_num_samples = config.get("eval_num_samples", 200)

        # GPT‑4 client setup
        self.gpt4_client = None
        if HAS_OPENAI:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if openai.api_key:
                self.gpt4_client = openai
                logger.info("OpenAI client initialised for GPT‑4 evaluation.")
            else:
                logger.warning("OPENAI_API_KEY not set; GPT‑4 evaluation will be unavailable.")
        else:
            logger.warning("openai package not installed; GPT‑4 evaluation unavailable.")

        # Ensure tokenizer padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info("Set tokenizer pad_token to eos_token.")

    # ------------------------------------------------------------------
    # RM Score
    # ------------------------------------------------------------------

    def compute_rm_score(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Generate responses with the policy and score them using the loaded reward model.

        Args:
            prompts: List of raw prompt strings (e.g., validation set).

        Returns:
            Dictionary containing:
                - "mean": average RM score
                - "std": standard deviation
                - "scores": list of individual scores (floats)
        """
        if self.reward_model is None:
            raise RuntimeError("Reward model not loaded; cannot compute RM score.")

        # Determine formatting template
        if self.dataset_name == "tldr":
            def format_text(prompt, response):
                return f"{prompt}\n\nTL;DR:{response}"
        elif self.dataset_name in ("hhrlhf", "webgpt"):
            def format_text(prompt, response):
                return f"Human: {prompt}\n\nAssistant: {response}"
        else:
            raise ValueError(f"Unsupported dataset for RM scoring: {self.dataset_name}")

        scores = []
        # Batch generation for efficiency (mini‑batch)
        batch_size = 8
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            # Tokenize prompts (left padding)
            tokenized = self.tokenizer(
                batch_prompts,
                max_length=self.max_prompt_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            prompt_ids = tokenized["input_ids"].to(self.device)
            prompt_mask = tokenized["attention_mask"].to(self.device)

            with torch.no_grad():
                generated = self.actor.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=self.max_response_length,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
            full_ids = generated.sequences  # (batch, prompt_len + response_len)
            prompt_len = prompt_ids.shape[1]

            # Decode responses
            responses = []
            for j in range(full_ids.size(0)):
                resp_tokens = full_ids[j, prompt_len:]
                text = self.tokenizer.decode(resp_tokens, skip_special_tokens=True)
                responses.append(text)

            # Format and score with reward model
            full_texts = [format_text(p, r) for p, r in zip(batch_prompts, responses)]
            with torch.no_grad():
                tokenized_rm = self.tokenizer(
                    full_texts,
                    truncation=True,
                    max_length=self.max_prompt_length + self.max_response_length,
                    padding="max_length",
                    return_tensors="pt",
                ).to(self.device)
                rm_outputs = self.reward_model(
                    input_ids=tokenized_rm["input_ids"],
                    attention_mask=tokenized_rm["attention_mask"],
                )
                # reward model outputs logits of shape (batch, 1) or (batch,)
                if hasattr(rm_outputs, "logits"):
                    logits = rm_outputs.logits
                else:
                    logits = rm_outputs[0]  # for some classification models
                if logits.dim() > 1 and logits.size(1) == 1:
                    logits = logits[:, 0]
                batch_scores = logits.cpu().tolist()
                scores.extend(batch_scores)

        mean_score = sum(scores) / len(scores)
        std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5

        return {"mean": mean_score, "std": std_score, "scores": scores}

    # ------------------------------------------------------------------
    # GPT‑4 Pairwise Evaluation
    # ------------------------------------------------------------------

    def _get_gpt4_prompt_template(self, criteria: Optional[str] = None) -> str:
        """Select and return the appropriate GPT‑4 evaluation prompt template."""
        ds = self.dataset_name if criteria is None else criteria
        if ds == "tldr":
            template = (
                "You will be given two summaries written for an article. Your task is to pick the better one "
                "between them, based on the four criteria.\n"
                "Please make sure you read and understand these instructions carefully.\n"
                "Relevance - selection of important content from the source. The summary should include only "
                "important information from the source document. Annotators were instructed to penalize summaries "
                "which contained redundancies and excess information.\n"
                "Coherence - the collective quality of all sentences. We align this dimension with the DUC quality "
                "question of structure and coherence whereby “the summary should be well-structured and well-organized. "
                "The summary should not just be a heap of related information, but should build from sentence to a "
                "coherent body of information about a topic.”\n"
                "Consistency - the factual alignment between the summary and the summarized source. A factually "
                "consistent summary contains only statements that are entailed by the source document. Annotators "
                "were also asked to penalize summaries that contained hallucinated facts.\n"
                "Fluency - the quality of the summary in terms of grammar, spelling, punctuation, word choice, and "
                "sentence structure.\n"
                "You should output single character to indicate which summary you think is better. "
                "‘A’ stands for Summary A and ‘B’ stands for Summary B. If you think both summaries are equally good, "
                "output ‘E’.\n\n"
                "Article / Post:\n{article}\n\n"
                "Summary A:\n{summary_a}\n\n"
                "Summary B:\n{summary_b}\n\n"
                "Your Choice (only a single character):"
            )
        elif ds == "hhrlhf":
            template = (
                "For the following query to a chatbot assistant, which response is more helpful?\n"
                "First provide a one-sentence comparison of the two responses and explain which you feel is more "
                "helpful. Second, on a new line, state only ‘A’ or ‘B’ to indicate which response is more helpful. "
                "If they are equally good or bad, state ‘E’.\n"
                "Your response should use the json format, with “comparison” and “choice” as keys.\n\n"
                "Query: {query}\n\n"
                "Response A: {response_a}\n\n"
                "Response B: {response_b}\n\n"
                "Your Judgment:"
            )
        elif ds == "webgpt":
            template = (
                "You will be given two response written for an question. Your task is to pick the better one "
                "between them, based on these criteria.\n"
                "Factual accuracy - which answer is more factually accurate?\n"
                "Coherence - which answer is easier to follow?\n"
                "Usefulness overall - all things considered, which answer would be more helpful to the person "
                "who asked this question?\n"
                "You should output with a json format where the key is the criteria and the value is the choice "
                "you made, using ‘A’ stands for Response A and ‘B’ stands for Response B. If you think both "
                "responses are equally good, output ‘E’.\n\n"
                "Question: {question}\n\n"
                "Answer A: {answer_a}\n\n"
                "Answer B: {answer_b}\n\n"
                "Your Judgment (you should also output the reason, note that you are allowed to think both "
                "responses are equally good, then output with ‘E’):"
            )
        else:
            raise ValueError(f"Unknown GPT‑4 evaluation template for dataset: {ds}")
        return template

    def _parse_gpt4_choice(self, raw_response: str, dataset_name: str) -> str:
        """Extract the 'A', 'B', or 'E' preference from GPT‑4’s output."""
        raw = raw_response.strip()
        if dataset_name == "tldr":
            # Expect last character to be choice
            # Remove trailing newlines
            lines = raw.split("\n")
            for line in reversed(lines):
                line = line.strip()
                if line in ("A", "B", "E"):
                    return line
            # Fallback: regex
            match = re.search(r"\b([ABE])\b", raw)
            if match:
                return match.group(1)
            return "E"  # default tie
        elif dataset_name in ("hhrlhf", "webgpt"):
            # Try JSON parsing
            try:
                data = json.loads(raw)
                return data.get("choice", "E").strip()
            except json.JSONDecodeError:
                # Regex hunt for 'choice'
                pattern = r"['\"]choice['\"]\s*:\s*['\"]([ABE])['\"]"
                match = re.search(pattern, raw)
                if match:
                    return match.group(1)
                # Last resort
                match = re.search(r"\b([ABE])\b", raw)
                if match:
                    return match.group(1)
            return "E"
        else:
            return "E"

    def gpt4_eval(
        self,
        prompts: List[str],
        responses_a: List[str],
        responses_b: List[str],
        criteria: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run GPT‑4 pairwise evaluation on a set of instances.

        Args:
            prompts: List of prompt strings.
            responses_a: List of first responses (order preserved, caller shuffles).
            responses_b: List of second responses.
            criteria: Optional task name for template selection; defaults to dataset_name.

        Returns:
            Dictionary with:
                - "preferences": list of single‑character choices ('A', 'B', 'E')
                - "win_A": count of A wins
                - "win_B": count of B wins
                - "tie": count of ties
        """
        if not HAS_OPENAI or self.gpt4_client is None:
            raise RuntimeError("OpenAI client not available; GPT‑4 evaluation skipped.")

        template = self._get_gpt4_prompt_template(criteria)
        preferences = []
        model_name = "gpt-4o-05-13"  # as used in the paper

        for prompt, resp_a, resp_b in zip(prompts, responses_a, responses_b):
            # Build the prompt
            if self.dataset_name == "tldr":
                filled = template.format(
                    article=prompt,
                    summary_a=resp_a,
                    summary_b=resp_b,
                )
            elif self.dataset_name == "hhrlhf":
                filled = template.format(
                    query=prompt,
                    response_a=resp_a,
                    response_b=resp_b,
                )
            elif self.dataset_name == "webgpt":
                filled = template.format(
                    question=prompt,
                    answer_a=resp_a,
                    answer_b=resp_b,
                )
            else:
                raise ValueError(f"Unsupported dataset for GPT‑4 eval: {self.dataset_name}")

            # Call OpenAI API with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.gpt4_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": filled},
                        ],
                        temperature=0.0,
                        max_tokens=300,
                    )
                    raw_output = response.choices[0].message.content
                    break
                except Exception as e:
                    logger.warning(f"OpenAI API call failed (attempt {attempt+1}): {e}")
                    if attempt == max_retries - 1:
                        logger.error("Max retries exceeded; defaulting to tie.")
                        raw_output = "E"
                        break
                    time.sleep(2 ** attempt)

            choice = self._parse_gpt4_choice(raw_output, self.dataset_name)
            preferences.append(choice)

        win_A = sum(1 for c in preferences if c == "A")
        win_B = sum(1 for c in preferences if c == "B")
        tie = sum(1 for c in preferences if c == "E")

        return {
            "preferences": preferences,
            "win_A": win_A,
            "win_B": win_B,
            "tie": tie,
        }

    # ------------------------------------------------------------------
    # Pass@k for Code Generation (APPS)
    # ------------------------------------------------------------------

    def pass_k_eval(self, test_dataset: Dataset, k: int = 1) -> Dict[str, float]:
        """
        Evaluate pass@k on the APPS test set.

        Args:
            test_dataset: HuggingFace Dataset containing columns:
                - "problem": prompt string
                - "input_output": dict with "inputs" and "outputs" lists
            k: k in pass@k (1 or 5; can be called multiple times).

        Returns:
            Dictionary with {"pass@<k>": average_score}.
        """
        if self.reward_type != "compiler":
            raise RuntimeError("pass_k_eval only supported for compiler reward tasks (APPS).")

        n = self.eval_num_samples
        if n < k:
            logger.warning(f"Number of samples ({n}) is less than k ({k}); pass@k may be overestimated.")

        # Prepare prompt template for code generation
        def format_code_prompt(problem: str) -> str:
            return f"Write a Python program to solve the following problem:\n\n{problem}\n\nSolution:\n"

        total_problems = 0
        pass_k_cumulative = 0.0

        # Process each problem
        # We'll iterate over the dataset; we assume it's small enough to work sequentially.
        for sample in test_dataset:
            problem = sample["problem"]
            prompt_str = format_code_prompt(problem)
            # Tokenize
            tokenized = self.tokenizer(
                prompt_str,
                max_length=self.max_prompt_length,
                truncation=True,
                return_tensors="pt",
            )
            prompt_ids = tokenized["input_ids"].to(self.device)
            prompt_mask = tokenized["attention_mask"].to(self.device)

            # Generate n samples
            correct_count = 0
            for _ in range(n):
                with torch.no_grad():
                    generated = self.actor.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        max_new_tokens=self.max_response_length,
                        do_sample=True,
                        temperature=1.0,  # as per config for apps
                        top_p=1.0,
                        top_k=5,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                # Extract only newly generated tokens
                start_idx = prompt_ids.shape[1]
                new_tokens = generated[0, start_idx:]
                full_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                # Extract Python code
                code = _extract_code(full_response)

                # Test against provided test cases
                io_data = sample.get("input_output", None)
                if io_data is None:
                    logger.warning("No input_output found for a problem; skipping.")
                    continue
                inputs = io_data.get("inputs", [])
                outputs = io_data.get("outputs", [])
                if len(inputs) == 0:
                    # Fallback: maybe test_list? some APPS versions have test_list
                    continue
                if _execute_code_against_tests(code, inputs, outputs, timeout=10.0):
                    correct_count += 1

            # Compute pass@k for this problem
            if n - correct_count < k:
                prob_score = 1.0
            else:
                prob_score = 1.0 - math.comb(n - correct_count, k) / math.comb(n, k)
            pass_k_cumulative += prob_score
            total_problems += 1
            logger.debug(f"Problem: correct={correct_count}/{n}, pass@{k}={prob_score:.4f}")

        if total_problems == 0:
            return {f"pass@{k}": 0.0}

        avg_pass_k = pass_k_cumulative / total_problems
        return {f"pass@{k}": avg_pass_k}

