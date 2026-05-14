from typing import Dict, Any, List
import numpy as np

from score.model import SCoReModel
from score.rewards import calculate_base_reward, get_total_stage_ii_reward
from score.utils import generate_first_attempt_prompt, generate_self_correction_instruction

class SCoReTrainer:
    """
    Implements the two-stage RL training process for SCoRe.
    """

    def __init__(
        self, 
        model: SCoReModel,
        ref_model: SCoReModel, # Reference model for KL-divergence
        config: Dict[str, Any]
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config

        self.alpha = config['alpha']
        self.beta_1 = config['beta_1'] # KL penalty for general RL
        self.beta_2 = config['beta_2'] # KL penalty for Stage I
        self.task_type = config['task']['type']

        # Placeholder for optimizer. In a real setting, this would be an actual PyTorch/JAX optimizer.
        self.optimizer = None # Or define a dummy optimizer class

    def _calculate_kl_divergence(self, prompt: str, generated_response: str) -> float:
        """
        Calculates a symbolic KL divergence between the current model and the reference model.
        In a real implementation, this would involve tokenizing the prompt and response,
        getting log probabilities from both models, and computing KL.
        """
        # Placeholder for actual KL divergence calculation
        # Assuming prompt + generated_response forms the sequence for KL calculation
        # For static benchmark, we return a symbolic value.
        # This would involve self.model.get_log_prob and self.ref_model.get_log_prob
        print(f"[DEBUG] Calculating KL divergence for prompt (first 50 chars): {prompt[:50]}... and response (first 50 chars): {generated_response[:50]}...")
        return 0.1 # Placeholder value

    def _update_model(self, loss: float):
        """
        Placeholder for updating model parameters based on the loss.
        In a real scenario, this would involve optimizer.step() and zero_grad().
        """
        print(f"[DEBUG] Updating model with loss: {loss}")
        if self.optimizer:
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            pass

    def train_stage_i(self, problems: List[Dict[str, str]], ground_truths: List[str]):
        """
        Implements Stage I of SCoRe training: Decoupling Attempts (Section 5.1).
        Objective: max E[r(y2, y*) - beta_2 * D_KL(pi_theta(.|x1) || pi_ref(.|x1))]
        """
        print("
--- Starting SCoRe Stage I Training ---")
        total_loss = 0.0

        for i, problem_data in enumerate(problems):
            problem = problem_data['problem']
            y_star = ground_truths[i]

            # 1. Generate first attempt (y1) from the current model
            prompt_t1 = generate_first_attempt_prompt(problem)
            y1 = self.model.generate_response(prompt_t1, temperature=0.0)

            # 2. Generate second attempt (y2) based on y1 and correction instruction
            correction_instruction = generate_self_correction_instruction(y1)
            prompt_t2 = f"{prompt_t1}{y1}
{correction_instruction}"
            y2 = self.model.generate_response(prompt_t2, temperature=0.0)

            # Calculate base rewards
            reward_y1 = calculate_base_reward(y1, y_star, self.task_type)
            reward_y2 = calculate_base_reward(y2, y_star, self.task_type)

            # Calculate KL-divergence for the first attempt only
            # The KL is calculated for the policy that generated y1 given x1
            # For this static reproduction, we simulate the KL calculation.
            # In a real setup, this would compare self.model.log_prob(y1|prompt_t1) with self.ref_model.log_prob(y1|prompt_t1)
            kl_penalty_t1 = self._calculate_kl_divergence(prompt_t1, y1)

            # Stage I objective: Maximize reward_y2, penalize KL of y1
            # So, loss = - (reward_y2 - beta_2 * kl_penalty_t1)
            loss = - (reward_y2 - self.beta_2 * kl_penalty_t1)
            total_loss += loss

            self._update_model(loss)

            print(f"Stage I - Problem {i+1}: Reward_y1={reward_y1}, Reward_y2={reward_y2}, KL_t1={kl_penalty_t1:.4f}, Loss={loss:.4f}")

        print(f"--- Stage I Training Finished. Average Loss: {total_loss / len(problems):.4f} ---
")

    def train_stage_ii(self, problems: List[Dict[str, str]], ground_truths: List[str]):
        """
        Implements Stage II of SCoRe training: Multi-Turn RL with Reward Shaping (Section 5.2).
        Objective: max E[sum(r(yi, y*)) + shaped_bonus - beta_1 * D_KL(pi_theta(.|xi) || pi_ref(.|xi))]
        where shaped_bonus = alpha * (r(y2, y*) - r(y1, y*))
        """
        print("
--- Starting SCoRe Stage II Training ---")
        total_loss = 0.0

        for i, problem_data in enumerate(problems):
            problem = problem_data['problem']
            y_star = ground_truths[i]

            # 1. Generate first attempt (y1)
            prompt_t1 = generate_first_attempt_prompt(problem)
            y1 = self.model.generate_response(prompt_t1, temperature=0.0)

            # 2. Generate second attempt (y2)
            correction_instruction = generate_self_correction_instruction(y1)
            prompt_t2 = f"{prompt_t1}{y1}
{correction_instruction}"
            y2 = self.model.generate_response(prompt_t2, temperature=0.0)

            # Calculate base rewards
            reward_y1 = calculate_base_reward(y1, y_star, self.task_type)
            reward_y2 = calculate_base_reward(y2, y_star, self.task_type)

            # Calculate total reward for Stage II with shaping
            total_shaped_reward = get_total_stage_ii_reward(reward_y1, reward_y2, self.alpha)

            # Calculate KL-divergence for both attempts
            # In a real setup, this would be a sum of KL for y1|prompt_t1 and y2|prompt_t2
            kl_penalty_t1 = self._calculate_kl_divergence(prompt_t1, y1)
            kl_penalty_t2 = self._calculate_kl_divergence(prompt_t2, y2)
            total_kl_penalty = kl_penalty_t1 + kl_penalty_t2

            # Stage II objective: Maximize total_shaped_reward - beta_1 * total_kl_penalty
            # So, loss = - (total_shaped_reward - beta_1 * total_kl_penalty)
            loss = - (total_shaped_reward - self.beta_1 * total_kl_penalty)
            total_loss += loss

            self._update_model(loss)

            print(f"Stage II - Problem {i+1}: Total_Shaped_Reward={total_shaped_reward:.4f}, Total_KL={total_kl_penalty:.4f}, Loss={loss:.4f}")

        print(f"--- Stage II Training Finished. Average Loss: {total_loss / len(problems):.4f} ---
")

    def evaluate(self, problems: List[Dict[str, str]], ground_truths: List[str]) -> Dict[str, float]:
        """
        Evaluates the current model on a given set of problems and ground truths.
        """
        print("
--- Starting Evaluation ---")
        predictions_t1 = []
        predictions_t2 = []

        for i, problem_data in enumerate(problems):
            problem = problem_data['problem']

            prompt_t1 = generate_first_attempt_prompt(problem)
            y1 = self.model.generate_response(prompt_t1, temperature=0.0) # Greedy decoding for evaluation
            predictions_t1.append(y1)

            correction_instruction = generate_self_correction_instruction(y1)
            prompt_t2 = f"{prompt_t1}{y1}
{correction_instruction}"
            y2 = self.model.generate_response(prompt_t2, temperature=0.0) # Greedy decoding for evaluation
            predictions_t2.append(y2)
            print(f"Evaluation - Problem {i+1}: Problem={problem[:30]}..., Pred_t1={y1[:30]}..., Pred_t2={y2[:30]}...")

        metrics = {
            "model_name": self.model.model_name,
            **calculate_metrics(predictions_t1, predictions_t2, ground_truths, self.task_type)
        }
        print("---
Evaluation Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("---
")

        return metrics

