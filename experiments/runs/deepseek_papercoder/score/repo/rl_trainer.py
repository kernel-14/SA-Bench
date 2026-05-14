```python
"""
rl_trainer.py

Core on‑policy multi‑turn REINFORCE training loop for SCoRe.
Implements the two‑stage algorithm with reward shaping and optional offline mixing.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import Config
from policy_model import PolicyModel
from reward import RewardFunction

logger = logging.getLogger(__name__)

# Default maximum number of new tokens to generate during sampling.
DEFAULT_MAX_NEW_TOKENS = 512


# --------------------------------------------------------------------------- #
# Trajectory – stores a single two‑turn rollout.
# --------------------------------------------------------------------------- #

@dataclass
class Trajectory:
    problem: str                           # raw problem statement
    y1: str                                # first‑attempt response
    y2: str                                # second‑attempt response
    r1: float                              # binary reward for y1
    r2: float                              # binary reward for y2
    logprobs_policy_y1: Optional[torch.Tensor] = None   # (L1,) on tokens of y1
    logprobs_policy_y2: torch.Tensor = None             # (L2,) on tokens of y2
    logprobs_ref_y1: Optional[torch.Tensor] = None      # (L1,) on tokens of y1 (ref)
    logprobs_ref_y2: torch.Tensor = None                # (L2,) on tokens of y2 (ref)
    is_offline: bool = False               # True if y1 came from the offline pool


# --------------------------------------------------------------------------- #
# RLTrainer – main class.
# --------------------------------------------------------------------------- #

class RLTrainer:
    """
    Handles Stage I and Stage II on‑policy RL for SCoRe.

    Parameters
    ----------
    policy : PolicyModel
        The policy model to train.
    ref : PolicyModel
        Frozen reference model (base model).
    reward_fn : RewardFunction
        Binary reward function (MATH / code).
    config : Config
        Global configuration object.
    offline_dataset : Optional[List[Dict[str, Any]]]
        Pre‑generated offline first‑attempt data (provided to Stage II).
    task : str
        Either ``"math"`` or ``"code"``. Determines which sub‑config to use
        and how prompts are formatted.
    """

    def __init__(
        self,
        policy: PolicyModel,
        ref: PolicyModel,
        reward_fn: RewardFunction,
        config: Config,
        offline_dataset: Optional[List[Dict[str, Any]]] = None,
        task: str = "math",
    ) -> None:
        self.policy = policy
        self.ref = ref
        self.reward_fn = reward_fn
        self.config = config
        self.task = task

        # For convenience, store the relevant sub‑config block.
        if task == "math":
            self.task_config = config.math
        elif task == "code":
            self.task_config = config.code
        else:
            raise ValueError(f"Unknown task '{task}'; expected 'math' or 'code'.")

        self.tokenizer = policy.tokenizer  # shared tokenizer

        # Private state
        self.current_stage: Optional[str] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.best_reward: float = -float("inf")
        self.best_checkpoint_path: Optional[str] = None

        # Offline data – transform into lookup table.
        self.offline_mapping: Dict[str, List[Dict[str, Any]]] = {}
        if offline_dataset:
            for entry in offline_dataset:
                prob = entry["problem"]
                if prob not in self.offline_mapping:
                    self.offline_mapping[prob] = []
                self.offline_mapping[prob].append(
                    {
                        "y1": entry["y1"],
                        "r1": entry["r1"],
                        "ground_truth": entry["ground_truth"],
                    }
                )

        # Pipeline state
        self.stage_cfg = None
        self.offline_mixing_prob = 0.0
        self.use_offline = False

    # ------------------------------------------------------------------ #
    #  Prompt formatters (rely on config templates)
    # ------------------------------------------------------------------ #

    def _format_turn1(self, raw_problem: str) -> str:
        """Build the first‑turn prompt string."""
        if self.task == "math":
            return f"{self.config.prompts.math_zero_shot}\nProblem: {raw_problem}\n"
        else:
            # For code, the raw_problem is already a fully formatted prompt
            return raw_problem

    def _format_turn2(self, raw_problem: str, y1: str) -> str:
        """Build the second‑turn prompt, including y1."""
        turn1_text = self._format_turn1(raw_problem)
        if self.task == "math":
            correction = self.config.prompts.math_self_correction
        else:
            correction = self.config.prompts.code_self_correction
        return f"{turn1_text}\n{y1}\n\n{correction}"

    # ------------------------------------------------------------------ #
    #  Tokenization helper
    # ------------------------------------------------------------------ #

    def _tokenize(self, text: str) -> torch.LongTensor:
        """
        Tokenize a single string and return a tensor of shape (1, L) on device.
        """
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_seq_length,
            padding=False,
        )
        input_ids = tokenized["input_ids"].to(self.policy.device)
        return input_ids

    # ------------------------------------------------------------------ #
    #  Sampling trajectories
    # ------------------------------------------------------------------ #

    def sample_trajectories(
        self,
        prompts: List[str],
        ground_truths: List[Any],
        temperature: float,
        use_offline: bool = False,
    ) -> List[Trajectory]:
        """
        Generate a batch of two‑turn trajectories.

        Parameters
        ----------
        prompts : list[str]
            Raw problem strings.
        ground_truths : list
            Corresponding ground truth answers or test dicts.
        temperature : float
            Sampling temperature.
        use_offline : bool
            If True, offline first attempts may be mixed in (Stage II only).

        Returns
        -------
        list of Trajectory
        """
        trajectories: List[Trajectory] = []
        for i, (prompt_str, gt) in enumerate(zip(prompts, ground_truths)):
            # Decide offline vs online for the first attempt.
            is_offline = (
                use_offline
                and (prompt_str in self.offline_mapping)
                and (random.random() < self.offline_mixing_prob)
            )

            if is_offline:
                # ---------- offline first attempt ----------
                candidates = self.offline_mapping[prompt_str]
                chosen = random.choice(candidates)
                y1 = chosen["y1"]
                r1 = chosen["r1"]
                offline_gt = chosen["ground_truth"]
                # Use the offline ground truth for the reward computation of y2.
                gt = offline_gt

                # Build turn‑2 input.
                turn2_text = self._format_turn2(prompt_str, y1)
                input_ids2 = self._tokenize(turn2_text)
                gen_ids2 = self.policy.generate(
                    input_ids2, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, temperature=temperature
                )
                y2 = self.tokenizer.decode(gen_ids2[0], skip_special_tokens=True)

                # Reward for y2.
                r2 = self.reward_fn(y2, gt, is_code=(self.task == "code"))

                # Log‑probs for y2 only (reference with no grad).
                logprobs_policy_y2 = self.policy.compute_logprobs(input_ids2, gen_ids2)  # (1, L2)
                with torch.no_grad():
                    logprobs_ref_y2 = self.ref.compute_logprobs(input_ids2, gen_ids2)

                # Strip batch dimension.
                logprobs_policy_y2 = logprobs_policy_y2.squeeze(0)
                logprobs_ref_y2 = logprobs_ref_y2.squeeze(0)

                traj = Trajectory(
                    problem=prompt_str,
                    y1=y1,
                    y2=y2,
                    r1=r1,
                    r2=r2,
                    logprobs_policy_y1=None,
                    logprobs_policy_y2=logprobs_policy_y2,
                    logprobs_ref_y1=None,
                    logprobs_ref_y2=logprobs_ref_y2,
                    is_offline=True,
                )
                trajectories.append(traj)

            else:
                # ---------- online first attempt (and then second) ----------
                turn1_text = self._format_turn1(prompt_str)
                input_ids1 = self._tokenize(turn1_text)
                gen_ids1 = self.policy.generate(
                    input_ids1, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, temperature=temperature
                )
                y1 = self.tokenizer.decode(gen_ids1[0], skip_special_tokens=True)
                r1 = self.reward_fn(y1, gt, is_code=(self.task == "code"))

                # Log‑probs for y1.
                logprobs_policy_y1 = self.policy.compute_logprobs(input_ids1, gen_ids1).squeeze(0)
                with torch.no_grad():
                    logprobs_ref_y1 = self.ref.compute_logprobs(input_ids1, gen_ids1).squeeze(0)

                # Turn 2.
                turn2_text = self._format_turn2(prompt_str, y1)
                input_ids2 = self._tokenize(turn2_text)
                gen_ids2 = self.policy.generate(
                    input_ids2, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, temperature=temperature
                )
                y2 = self.tokenizer.decode(gen_ids2[0], skip_special_tokens=True)
                r2 = self.reward_fn(y2, gt, is_code=(self.task == "code"))

                logprobs_policy_y2 = self.policy.compute_logprobs(input_ids2, gen_ids2).squeeze(0)
                with torch.no_grad():
                    logprobs_ref_y2 = self.ref.compute_logprobs(input_ids2, gen_ids2).squeeze(0)

                traj = Trajectory(
                    problem=prompt_str,
                    y1=y1,
                    y2=y2,
                    r1=r1,
                    r2=r2,
                    logprobs_policy_y1=logprobs_policy_y1,
                    logprobs_policy_y2=logprobs_policy_y2,
                    logprobs_ref_y1=logprobs_ref_y1,
                    logprobs_ref_y2=logprobs_ref_y2,
                    is_offline=False,
                )
                trajectories.append(traj)

        return trajectories

    # ------------------------------------------------------------------ #
    #  Loss computation (REINFORCE + KL)
    # ------------------------------------------------------------------ #

    def compute_loss(self, trajectories: List[Trajectory], stage: str) -> torch.Tensor:
        """
        Build the scalar loss for a batch of trajectories according to the
        active stage.

        Parameters
        ----------
        trajectories : list[Trajectory]
        stage : str
            Either ``"I"`` or ``"II"``.

        Returns
        -------
        torch.Tensor
            Scalar loss ready for backpropagation.
        """
        device = self.policy.device
        batch_size = len(trajectories)

        # Extract rewards.
        r2_list = torch.tensor([t.r2 for t in trajectories], device=device)

        if stage == "I":
            # ---------- Stage I ----------
            advantage2 = r2_list - r2_list.mean(dim=0).detach()

            loss = torch.tensor(0.0, device=device)

            # Turn‑1 KL (only for online trajectories) with β2.
            kl1_total = 0.0
            num_tokens1 = 0
            for t in trajectories:
                if t.logprobs_policy_y1 is not None:
                    kl = (t.logprobs_policy_y1 - t.logprobs_ref_y1).sum()
                    kl1_total += kl
                    num_tokens1 += t.logprobs_policy_y1.numel()
            if num_tokens1 > 0:
                loss = loss + self.beta2 * (kl1_total / num_tokens1)

            # Turn‑2 policy gradient + KL with β1.
            loss2_adv = 0.0
            loss2_kl = 0.0
            num_tokens2 = 0
            for i, t in enumerate(trajectories):
                logp = t.logprobs_policy_y2
                ref_logp = t.logprobs_ref_y2
                n = logp.numel()
                # advantage2[i] * sum of log‑probs over tokens.
                loss2_adv += advantage2[i] * logp.sum()
                loss2_kl += (logp - ref_logp).sum()
                num_tokens2 += n

            # REINFORCE: average over batch.
            loss = loss - (loss2_adv / batch_size)
            # KL: average over all tokens.
            loss = loss + self.beta1 * (loss2_kl / max(num_tokens2, 1))

            return loss

        else:
            # ---------- Stage II ----------
            r1_list = torch.tensor([t.r1 for t in trajectories], device=device)
            # Shaped reward for second turn.
            alpha = self.task_config.reward_shaping.alpha
            r2_shaped = r2_list + alpha * (r2_list - r1_list)

            advantage1 = r1_list - r1_list.mean(dim=0).detach()
            advantage2 = r2_shaped - r2_shaped.mean(dim=0).detach()

            loss = torch.tensor(0.0, device=device)

            # Turn‑1: only online trajectories.
            loss1_adv = 0.0
            loss1_kl = 0.0
            num_tokens1 = 0
            for i, t in enumerate(trajectories):
                if not t.is_offline:
                    logp = t.logprobs_policy_y1
                    ref_logp = t.logprobs_ref_y1
                    n = logp.numel()
                    loss1_adv += advantage1[i] * logp.sum()
                    loss1_kl += (logp - ref_logp).sum()
                    num_tokens1 += n
            if num_tokens1 > 0:
                loss = loss - (loss1_adv / batch_size)
                loss = loss + self.beta1 * (loss1_kl / num_tokens1)

            # Turn‑2: all trajectories.
            loss2_adv = 0.0
            loss2_kl = 0.0
            num_tokens2 = 0
            for i, t in enumerate(trajectories):
                logp = t.logprobs_policy_y2
                ref_logp = t.logprobs_ref_y2
                n = logp.numel()
                loss2_adv += advantage2[i] * logp.sum()
                loss2_kl += (logp - ref_logp).sum()
                num_tokens2 += n
            loss = loss - (loss2_adv / batch_size)
            loss = loss + self.beta1 * (loss2_kl / max(num_tokens2, 1))

            return loss

    # ------------------------------------------------------------------ #
    #  Single training step
    # ------------------------------------------------------------------ #

    def train_step(self, batch: List[Tuple[str, Any]]) -> Dict[str, float]:
        """
        Perform one parameter update using a batch of (problem, ground_truth).

        Parameters
        ----------
        batch : list of (str, Any)
            The training batch.

        Returns
        -------
        dict
            Dictionary of scalars for logging (loss, avg_r1, avg_r2, ...).
        """
        prompts, ground_truths = zip(*batch)

        # Use offline mixing only in Stage II when the offline pool is available.
        use_offline = (
            self.current_stage == "II"
            and self.use_offline
            and self.offline_mapping
        )

        # Sample trajectories.
        # The temperature comes from the current stage configuration.
        temperature = self.stage_cfg.temperature
        trajectories = self.sample_trajectories(
            list(prompts), list(ground_truths), temperature, use_offline=use_offline
        )

        # Compute loss.
        loss = self.compute_loss(trajectories, self.current_stage)

        # Backward.
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Gather metrics.
        r1s = [t.r1 for t in trajectories]
        r2s = [t.r2 for t in trajectories]
        avg_r1 = sum(r1s) / len(r1s)
        avg_r2 = sum(r2s) / len(r2s)

        # Optional KL metrics (approximate, averaged over tokens).
        def mean_kl(trajs, turn=1):
            vals = []
            for