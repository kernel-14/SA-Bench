import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from config import Config
from dataset_utils import Problem, build_dataloader, load_code_dataset # load_code_dataset for MBPP-R specific loading
from model_utils import LLMForSelfCorrection
from prompt_manager import PromptManager
from reward_utils import BaseRewardFunction

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator class for assessing the self-correction performance of a trained LLM.
    It runs two-turn inference (first attempt and self-correction attempt) and
    calculates various self-correction metrics.
    """

    def __init__(
        self,
        model_wrapper: LLMForSelfCorrection,
        dataloader: torch.utils.data.DataLoader,
        prompt_manager: PromptManager,
        reward_function: BaseRewardFunction,
        config: Config,
    ):
        """
        Initializes the Evaluator.

        Args:
            model_wrapper: The LLM model wrapper to be evaluated.
            dataloader: PyTorch DataLoader for the evaluation dataset.
            prompt_manager: Manages the generation of prompts for the model.
            reward_function: Function to calculate correctness rewards.
            config: Configuration object containing hyperparameters and settings.
        """
        self.model_wrapper: LLMForSelfCorrection = model_wrapper
        self.dataloader: torch.utils.data.DataLoader = dataloader
        self.prompt_manager: PromptManager = prompt_manager
        self.reward_function: BaseRewardFunction = reward_function
        self.config: Config = config

        # Ensure model is in evaluation mode for consistent inference
        self.model_wrapper.get_current_model().eval()
        logger.info("Evaluator initialized. Model set to evaluation mode.")

    def _run_two_turn_inference(
        self, problem: Problem
    ) -> Tuple[str, float, str, float]:
        """
        Performs a two-turn inference for a single problem.

        Args:
            problem: The problem instance (containing text and ground_truth).

        Returns:
            A tuple: (response_t1, reward_t1, response_t2, reward_t2)
            where responses are strings and rewards are floats (1.0 for correct, 0.0 for incorrect).
        """
        # Configuration for inference, obtained from config.yaml's evaluation section
        if self.config.evaluation is None:
            raise ValueError("Evaluation configuration is missing in config.yaml.")
        sampling_temperature: float = self.config.evaluation.get("sampling_temperature", 0.0)
        max_new_tokens: int = self.config.evaluation.get("max_new_tokens", 1024)
        task_type_for_prompts: str = self.config.task_type # Use this for prompt manager

        # Determine task context string for prompt manager based on task_type and current evaluation dataset
        # For evaluation, distinguish between human_eval and mbpp_r if task_type is code
        if task_type_for_prompts == "math":
            prompt_task_context = "math"
        elif task_type_for_prompts == "code":
            # Assuming main evaluation dataset is HumanEval if task_type is code,
            # and MBPP-R is handled separately.
            # This needs to be consistent with how main.py creates the DataLoader for evaluation.
            if "HumanEval" in problem.problem_id: # Heuristic for HumanEval
                prompt_task_context = "human_eval_eval"
            elif "MBPP" in problem.problem_id: # Heuristic for MBPP problems (e.g. if eval on raw MBPP)
                prompt_task_context = "mbpp_train" # Use 3-shot MBPP format
            else:
                # Fallback, perhaps for other code datasets. Default to generic code prompt.
                logger.warning(f"Unknown problem ID format for code task: {problem.problem_id}. Defaulting to 'human_eval_eval' context for prompts.")
                prompt_task_context = "human_eval_eval"
        else:
            raise ValueError(f"Unsupported task_type: {task_type_for_prompts}")


        # --- Turn 1 Inference ---
        first_turn_prompt: str = self.prompt_manager.get_first_turn_prompt(
            problem_text=problem.text, task_context=prompt_task_context
        )
        response_t1, _ = self.model_wrapper.generate(
            prompt=first_turn_prompt,
            temperature=sampling_temperature,
            max_new_tokens=max_new_tokens,
        )
        reward_t1: float = self.reward_function.calculate_reward(
            response=response_t1, ground_truth=problem.ground_truth
        )

        # --- Turn 2 Inference (Self-Correction) ---
        second_turn_prompt: str = self.prompt_manager.get_second_turn_prompt(
            problem_text=problem.text,
            first_response=response_t1,
            task_context=prompt_task_context,
        )
        response_t2, _ = self.model_wrapper.generate(
            prompt=second_turn_prompt,
            temperature=sampling_temperature,
            max_new_tokens=max_new_tokens,
        )
        reward_t2: float = self.reward_function.calculate_reward(
            response=response_t2, ground_truth=problem.ground_truth
        )

        return response_t1, reward_t1, response_t2, reward_t2

    def _run_mbpp_r_inference(self, problem: Problem) -> Tuple[str, float]:
        """
        Performs single-turn inference for MBPP-R, where the first attempt
        (which is known to be incorrect) is provided in the problem's metadata.
        The model generates only the corrected second attempt.

        Args:
            problem: The MBPP-R problem instance, including the incorrect first attempt
                     in its metadata.

        Returns:
            A tuple: (response_t2, reward_t2)
        """
        incorrect_t1: str = problem.metadata.get("incorrect_first_attempt", "")
        if not incorrect_t1:
            logger.error(
                f"MBPP-R problem {problem.problem_id} must contain "
                "'incorrect_first_attempt' in metadata to run _run_mbpp_r_inference."
            )
            # Cannot proceed with MBPP-R evaluation for this problem, return 0 reward
            return "", 0.0

        if self.config.evaluation is None:
            raise ValueError("Evaluation configuration is missing in config.yaml.")
        sampling_temperature: float = self.config.evaluation.get("sampling_temperature", 0.0)
        max_new_tokens: int = self.config.evaluation.get("max_new_tokens", 1024)
        task_type_for_prompts: str = "mbpp_r_eval" # Specific context for MBPP-R self-correction

        # Generate only the second turn based on the provided incorrect first attempt
        second_turn_prompt: str = self.prompt_manager.get_second_turn_prompt(
            problem_text=problem.text,
            first_response=incorrect_t1,
            task_context=task_type_for_prompts,
        )
        response_t2, _ = self.model_wrapper.generate(
            prompt=second_turn_prompt,
            temperature=sampling_temperature,
            max_new_tokens=max_new_tokens,
        )
        reward_t2: float = self.reward_function.calculate_reward(
            response=response_t2, ground_truth=problem.ground_truth
        )
        return response_t2, reward_t2

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates the model's self-correction performance across the entire evaluation dataset.

        Calculates Accuracy@1, Accuracy@2, Delta(t1,t2), Delta_ic, and Delta_ci.
        If task_type is 'code' and mbpp_r_path is configured, also calculates MBPP-R score.

        Returns:
            A dictionary containing all calculated metrics.
        """
        logger.info("Starting main evaluation...")
        results: Dict[str, float] = {}

        # --- Standard Two-Turn Evaluation (MATH or HumanEval) ---
        correct_t1_count: int = 0
        correct_t2_count: int = 0
        inc_to_corr_count: int = 0  # Incorrect in t1, Correct in t2
        corr_to_inc_count: int = 0  # Correct in t1, Incorrect in t2
        total_problems: int = 0

        # We disable gradient calculations for efficiency during evaluation
        with torch.no_grad():
            for batch_problems in tqdm(self.dataloader, desc="Evaluating main dataset"):
                for problem in batch_problems:  # DataLoader typically yields batches, iterate over individual problems
                    total_problems += 1
                    _, reward_t1, _, reward_t2 = self._run_two_turn_inference(problem)

                    if reward_t1 == 1.0:
                        correct_t1_count += 1
                    if reward_t2 == 1.0:
                        correct_t2_count += 1

                    if reward_t1 == 0.0 and reward_t2 == 1.0:
                        inc_to_corr_count += 1
                    elif reward_t1 == 1.0 and reward_t2 == 0.0:
                        corr_to_inc_count += 1
        
        # Calculate standard metrics
        if total_problems > 0:
            results["Accuracy@t1"] = correct_t1_count / total_problems
            results["Accuracy@t2"] = correct_t2_count / total_problems
            results["Delta(t1,t2)"] = results["Accuracy@t2"] - results["Accuracy@t1"]
            results["Delta_ic(t1,t2)"] = inc_to_corr_count / total_problems
            results["Delta_ci(t1,t2)"] = corr_to_inc_count / total_problems
        else:
            logger.warning("No problems processed in main evaluation. All metrics set to 0.0.")
            results["Accuracy@t1"] = 0.0
            results["Accuracy@t2"] = 0.0
            results["Delta(t1,t2)"] = 0.0
            results["Delta_ic(t1,t2)"] = 0.0
            results["Delta_ci(t1,t2)"] = 0.0

        logger.info(f"Main evaluation metrics: {results}")

        # --- MBPP-R Specific Evaluation (if applicable) ---
        if self.config.task_type == "code" and self.config.mbpp_r_path:
            logger.info(f"Starting MBPP-R evaluation on {self.config.mbpp_r_path}...")
            mbpp_r_problems: List[Problem] = self._load_mbpp_r_dataset()
            mbpp_r_correct_count: int = 0
            total_mbpp_r_problems: int = 0

            if not mbpp_r_problems:
                logger.warning("MBPP-R dataset is empty. MBPP-R score not calculated.")
                results["MBPP-R"] = 0.0
            else:
                with torch.no_grad():
                    for problem in tqdm(mbpp_r_problems, desc="Evaluating MBPP-R"):
                        total_mbpp_r_problems += 1
                        _, reward_t2 = self._run_mbpp_r_inference(problem)
                        if reward_t2 == 1.0:
                            mbpp_r_correct_count += 1

                if total_mbpp_r_problems > 0:
                    results["MBPP-R"] = mbpp_r_correct_count / total_mbpp_r_problems
                else:
                    results["MBPP-R"] = 0.0
                logger.info(f"MBPP-R score: {results['MBPP-R']:.4f}")
        
        logger.info("Evaluation complete.")
        return results

    def _load_mbpp_r_dataset(self) -> List[Problem]:
        """
        Helper method to load the MBPP-R dataset using dataset_utils.
        Assumes mbpp_r_path points to a JSONL file where problems contain
        'incorrect_first_attempt' in their metadata.
        """
        if not self.config.mbpp_r_path:
            logger.warning("MBPP-R path not configured. Returning empty list.")
            return []
        try:
            # load_code_dataset is designed to parse generic code problem formats including metadata.
            # We assume 'incorrect_first_attempt' is placed in metadata during generation.
            problems: List[Problem] = load_code_dataset(self.config.mbpp_r_path)
            logger.info(f"Loaded {len(problems)} problems for MBPP-R evaluation.")
            return problems
        except FileNotFoundError:
            logger.error(f"MBPP-R dataset file not found at: {self.config.mbpp_r_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading MBPP-R dataset from {self.config.mbpp_r_path}: {e}")
            return []

