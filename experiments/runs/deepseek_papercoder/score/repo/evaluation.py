"""
evaluation.py

Self‑correction evaluation and inference‑compute scaling for the SCoRe method.

Implements the Evaluator class as specified in the project design.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from config import Config
from policy_model import PolicyModel
from reward import RewardFunction

logger = logging.getLogger(__name__)

# Default maximum number of tokens to generate during evaluation.
DEFAULT_MAX_NEW_TOKENS = 512


class Evaluator:
    """
    Evaluates a trained SCoRe policy on held‑out problem sets.

    Parameters
    ----------
    policy : PolicyModel
        The trained policy model (typically the best checkpoint after Stage II).
    reward_fn : RewardFunction
        Binary reward function that provides answer extraction and correctness checks.
    config : Config
        Global configuration object containing evaluation parameters and prompt templates.
    """

    def __init__(
        self,
        policy: PolicyModel,
        reward_fn: RewardFunction,
        config: Config,
    ) -> None:
        self.policy = policy
        self.reward_fn = reward_fn
        self.config = config

        # Ensure the policy is in evaluation mode and gradient tracking is off.
        self.policy.model.eval()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        """Generate a model response given a prompt string.

        Parameters
        ----------
        prompt : str
            The input prompt (already tokenizable).
        max_new_tokens : int
            Maximum number of new tokens to generate.
        temperature : float
            Sampling temperature; 0.0 for greedy decoding.

        Returns
        -------
        str
            The decoded output, skipping special tokens.
        """
        # Tokenize the prompt (left‑padding is already configured in the tokenizer).
        inputs = self.policy.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.policy.max_seq_length,
        )
        input_ids = inputs["input_ids"].to(self.policy.device)

        # Generate new tokens and decode only the newly added part.
        gen_ids = self.policy.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return self.policy.tokenizer.decode(
            gen_ids[0], skip_special_tokens=True
        ).strip()

    def _build_turn2(self, turn1_prompt: str, y1: str) -> str:
        """Construct the second‑turn input by appending the previous attempt
        and the appropriate self‑correction instruction.

        Parameters
        ----------
        turn1_prompt : str
            The exact prompt used for the first attempt.
        y1 : str
            The model's first‑attempt response.

        Returns
        -------
        str
            The full second‑turn input string.
        """
        if self.reward_fn.is_code:
            correction = self.config.prompts.code_self_correction
        else:
            correction = self.config.prompts.math_self_correction
        return f"{turn1_prompt}\n{y1}\n\n{correction}"

    def _check_math_answer(
        self, candidate_answer: str, ground_truth: str
    ) -> bool:
        """Verify a MATH candidate answer against the ground truth.

        Constructs a minimal response containing the candidate answer and
        delegates to the reward function for extraction + exact‑match.

        Parameters
        ----------
        candidate_answer : str
            The extracted answer (normalised).
        ground_truth : str
            The ground‑truth answer string (expected to be in boxed form).

        Returns
        -------
        bool
            True if the answers match.
        """
        dummy_response = f"Final Answer: \\boxed{{{candidate_answer}}}"
        return self.reward_fn(dummy_response, ground_truth) == 1.0

    # ------------------------------------------------------------------ #
    #  Public evaluation methods (matching the design interface)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate_self_correction(
        self,
        dataset: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Dict[str, float]:
        """Run standard two‑turn self‑correction evaluation (greedy decoding).

        For each problem, generates a first attempt and a second attempt
        conditioned on the first, then computes the five paper‑defined metrics.

        Parameters
        ----------
        dataset : list of dict
            Each entry must contain:
                - ``"prompt"`` : str – the full first‑turn input string.
                - ``"answer"`` : str or dict – ground‑truth answer (MATH) or
                  test‑case dictionary (code).
        temperature : float, optional
            Sampling temperature. Default 0.0 (greedy) for deterministic results.
        max_new_tokens : int, optional
            Maximum number of new tokens to generate per turn.

        Returns
        -------
        dict
            Keys: ``"accuracy@t1"``, ``"accuracy@t2"``, ``"delta"``,
            ``"i_to_c"``, ``"c_to_i"``.
        """
        correct_t1: List[float] = []
        correct_t2: List[float] = []

        for entry in dataset:
            prompt = entry["prompt"]
            ground_truth = entry["answer"]

            # ---------- Turn 1 ----------
            y1 = self._generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
            r1 = 1.0 if self.reward_fn(y1, ground_truth) else 0.0
            correct_t1.append(r1)

            # ---------- Turn 2 ----------
            turn2_input = self._build_turn2(prompt, y1)
            y2 = self._generate(turn2_input, max_new_tokens=max_new_tokens, temperature=temperature)
            r2 = 1.0 if self.reward_fn(y2, ground_truth) else 0.0
            correct_t2.append(r2)

        # -------- Metrics --------
        acc_t1 = float(np.mean(correct_t1))
        acc_t2 = float(np.mean(correct_t2))
        delta = acc_t2 - acc_t1

        # i→c : fraction of wrong‑at‑t1 problems that become correct at t2
        incorrect_first = [i for i, c in enumerate(correct_t1) if c == 0.0]
        i_to_c = (
            float(np.mean([correct_t2[i] for i in incorrect_first]))
            if incorrect_first
            else 0.0
        )

        # c→i : fraction of correct‑at‑t1 problems that become wrong at t2
        correct_first = [i for i, c in enumerate(correct_t1) if c == 1.0]
        c_to_i = (
            float(np.mean([1.0 - correct_t2[i] for i in correct_first]))
            if correct_first
            else 0.0
        )

        return {
            "accuracy@t1": acc_t1,
            "accuracy@t2": acc_t2,
            "delta": delta,
            "i_to_c": i_to_c,
            "c_to_i": c_to_i,
        }

    @torch.no_grad()
    def run_parallel_and_sequential(
        self,
        dataset: List[Dict[str, Any]],
        K: int,
        temperature: float = 0.7,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Dict[str, float]:
        """Inference‑compute scaling experiment (Section 6.2 of the paper).

        Compares two strategies on each problem:
          - **parallel‑only** : sample ``2K`` first attempts, majority‑vote
            over the answers.
          - **sequential self‑correction** : sample ``K`` first attempts,
            apply one round of self‑correction to each, then majority‑vote
            over the corrected answers.

        Parameters
        ----------
        dataset : list of dict
            Same format as in ``evaluate_self_correction``.
        K : int
            Number of base samples for the sequential branch (parallel uses 2K).
        temperature : float, optional
            Sampling temperature (default 0.7, as in the paper).
        max_new_tokens : int, optional
            Maximum new tokens per generation.

        Returns
        -------
        dict
            Keys: ``"parallel_accuracy"`` and ``"sequential_accuracy"``.
        """
        parallel_correct: List[float] = []
        sequential_correct: List[float] = []

        for entry in dataset:
            prompt = entry["prompt"]
            ground_truth = entry["answer"]
            is_code = self.reward_fn.is_code

            # ------------------ Parallel branch (2K) ------------------
            parallel_y1_list: List[str] = []
            for _ in range(2 * K):
                y = self._generate(
                    prompt, max_new_tokens=max_new_tokens, temperature=temperature
                )
                parallel_y1_list.append(y)

            if is_code:
                # Majority‑vote over the full code text.
                counter = Counter(y.strip() for y in parallel_y1_list)
                majority_code = counter.most_common(1)[0][0]
                # Retrieve any raw output that produced this code.
                majority_raw = next(
                    y for y in parallel_y1_list if y.strip() == majority_code
                )
                correct = 1.0 if self.reward_fn(majority_raw, ground_truth) else 0.0
            else:
                # Majority‑vote over extracted answers.
                answers = [
                    self.reward_fn.extract_answer(y) for y in parallel_y1_list
                ]
                counter = Counter(answers)
                majority_ans = counter.most_common(1)[0][0]
                correct = (
                    1.0 if self._check_math_answer(majority_ans, ground_truth) else 0.0
                )
            parallel_correct.append(correct)

            # ------------------ Sequential branch (K + correction) ------------------
            corrected_list: List[str] = []
            for _ in range(K):
                y1 = self._generate(
                    prompt, max_new_tokens=max_new_tokens, temperature=temperature
                )
                turn2_input = self._build_turn2(prompt, y1)
                y2 = self._generate(
                    turn2_input, max_new_tokens=max_new_tokens, temperature=temperature
                )
                corrected_list.append(y2)

            if is_code:
                counter = Counter(y.strip() for y in corrected_list)
                majority_code = counter.most_common(1)[0][0]
                majority_raw = next(
                    y for y in corrected_list if y.strip() == majority_code
                )
                correct_seq = 1.0 if self.reward_fn(majority_raw, ground_truth) else 0.0
            else:
                answers_seq = [
                    self.reward_fn.extract_answer(y) for y in corrected_list
                ]
                counter_seq = Counter(answers_seq)
                majority_ans_seq = counter_seq.most_common(1)[0][0]
                correct_seq = (
                    1.0 if self._check_math_answer(majority_ans_seq, ground_truth) else 0.0
                )
            sequential_correct.append(correct_seq)

        parallel_acc = float(np.mean(parallel_correct))
        sequential_acc = float(np.mean(sequential_correct))

        return {
            "parallel_accuracy": parallel_acc,
            "sequential_accuracy": sequential_acc,
        }
