# evaluation.py

import os
import yaml
import torch
from typing import List, Tuple, Dict
from dataset_loader import DatasetLoader
from model import Model


class Evaluation:
    """Evaluation class implementing RM score computation, GPT-4 pairwise evaluation, and human evaluation."""

    def __init__(self, model: Model, dataset: DatasetLoader, evaluation_config: Dict):
        """
        Initialize the Evaluation class.

        Args:
            model (Model): Trained policy/reward model for evaluation.
            dataset (DatasetLoader): Loaded dataset object providing validation splits.
            evaluation_config (Dict): Dictionary containing evaluation settings, prompts, and other configurations.
        """
        self.model = model
        self.dataset = dataset
        self.evaluation_config = evaluation_config

        # Paths to evaluation datasets
        self.validation_paths = self.evaluation_config.get("evaluation", {}).get("validation_paths", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_reward_model(self) -> Dict[str, float]:
        """
        Evaluate the reward model using RM scores on validation datasets.

        Returns:
            Dict[str, float]: RM evaluation metrics including mean RM score and standard deviation.
        """
        print("Starting Reward Model Evaluation...")
        task_type = self.evaluation_config.get("evaluation", {}).get("task_type", "tl_dr")
        validation_path = self.validation_paths.get(task_type, None)

        if not validation_path or not os.path.exists(validation_path):
            raise FileNotFoundError(f"Validation dataset path '{validation_path}' not found!")

        raw_validation_data = self.dataset.load_data()

        total_rm_score = []
        for entry in raw_validation_data:
            # Tokenize the prompt and response
            prompt = entry.get("prompt", "")
            response = entry.get("response", "")
            tokenized_prompt = self.dataset.tokenize([prompt])[0].to(self.device)
            tokenized_response = self.dataset.tokenize([response])[0].to(self.device)

            # Predict RM score using the model
            rm_logit = self.model.predict_reward(tokenized_prompt)
            rm_score = rm_logit.item()
            total_rm_score.append(rm_score)

        mean_rm_score = sum(total_rm_score) / len(total_rm_score)
        std_rm_score = torch.std(torch.tensor(total_rm_score, dtype=torch.float32)).item()

        print(f"Computed RM Scores -> Mean: {mean_rm_score:.2f}, Std Dev: {std_rm_score:.2f}")
        return {
            "mean_rm_score": mean_rm_score,
            "std_rm_score": std_rm_score,
        }

    def evaluate_gpt4(self, task: str, responses: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Simulate GPT-4 pairwise evaluation for task-specific metrics.

        Args:
            task (str): Name of the task (e.g., "TL;DR", "HH-RLHF", "WebGPT").
            responses (List[Tuple[str, str]]): Paired responses formatted as tuples (response_A, response_B).

        Returns:
            Dict[str, float]: GPT-4 evaluation results including win rate, tie rate, and loss rate.
        """
        print("Starting GPT-4 Evaluation...")
        gpt4_prompts = self.evaluation_config.get("evaluation", {}).get("gpt4_prompts", {})
        task_prompt_template = gpt4_prompts.get(task, None)

        if not task_prompt_template:
            raise ValueError(f"No GPT-4 prompt template found for task '{task}'.")

        gpt4_results = {"win_rate": 0.0, "tie_rate": 0.0, "loss_rate": 0.0}
        total_pairs = len(responses)

        for resp_a, resp_b in responses:
            # Format the GPT-4 prompt
            prompt = task_prompt_template.format(response_a=resp_a, response_b=resp_b)

            # Here, you would query GPT-4 API (simulated result for illustration)
            # Example query: gpt4_output = self.query_gpt4(prompt)
            simulated_result = torch.randint(0, 3, (1,)).item()  # Simulated: 0 -> A wins, 1 -> Tie, 2 -> B wins

            # Aggregate results
            if simulated_result == 0:
                gpt4_results["win_rate"] += 1
            elif simulated_result == 1:
                gpt4_results["tie_rate"] += 1
            else:
                gpt4_results["loss_rate"] += 1

        for key in gpt4_results.keys():
            gpt4_results[key] = gpt4_results[key] / total_pairs

        print(f"GPT-4 Evaluation Results -> Win Rate: {gpt4_results['win_rate']:.2f}, Tie Rate: {gpt4_results['tie_rate']:.2f}, Loss Rate: {gpt4_results['loss_rate']:.2f}")
        return gpt4_results

    def evaluate_human(self, task: str, responses: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Conduct human evaluations for paired responses on task-specific metrics.

        Args:
            task (str): Name of the task (e.g., "TL;DR", "HH-RLHF", "WebGPT").
            responses (List[Tuple[str, str]]): Paired responses formatted as tuples (response_A, response_B).

        Returns:
            Dict[str, float]: Human evaluation metrics including win rate, tie rate, loss rate, and inter-rater agreement.
        """
        print("Starting Human Evaluation...")
        human_criteria = self.evaluation_config.get("evaluation", {}).get("human_criteria", {})
        task_criteria = human_criteria.get(task, None)

        if not task_criteria:
            raise ValueError(f"No human evaluation criteria found for task '{task}'.")

        human_results = {"win_rate": 0.0, "tie_rate": 0.0, "loss_rate": 0.0, "inter_rater_agreement": 0.0}
        total_pairs = len(responses)
        annotated_results = []

        for resp_a, resp_b in responses:
            # Randomize response order to avoid bias
            randomized_responses = [(resp_a, resp_b), (resp_b, resp_a)][torch.randint(0, 2, (1,)).item()]

            # Human annotators manually compare responses (simulated result for illustration)
            simulated_result = torch.randint(0, 3, (1,)).item()  # Simulated: 0 -> A wins, 1 -> Tie, 2 -> B wins
            annotated_results.append(simulated_result)

            # Aggregate results
            if simulated_result == 0:
                human_results["win_rate"] += 1
            elif simulated_result == 1:
                human_results["tie_rate"] += 1
            else:
                human_results["loss_rate"] += 1

        human_results["inter_rater_agreement"] = self.compute_inter_rater_agreement(annotated_results)
        for key in ["win_rate", "tie_rate", "loss_rate"]:
            human_results[key] = human_results[key] / total_pairs

        print(f"Human Evaluation Results -> Win Rate: {human_results['win_rate']:.2f}, Tie Rate: {human_results['tie_rate']:.2f}, Loss Rate: {human_results['loss_rate']:.2f}, Inter-Rater Agreement: {human_results['inter_rater_agreement']:.2f}")
        return human_results

    def compute_inter_rater_agreement(self, results: List[int]) -> float:
        """
        Compute inter-rater agreement for human evaluation results.

        Args:
            results (List[int]): Annotated results with votes from multiple annotators.

        Returns:
            float: Calculated inter-rater agreement score.
        """
        # Simulate inter-rater agreement computation (placeholder implementation)
        return torch.rand(1).item()

