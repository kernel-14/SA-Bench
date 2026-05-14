"""
This module provides the EvaluationManager class, responsible for quantitatively
assessing the performance of trained MA-RLHF models across various tasks and metrics.
It includes functionalities for computing Reward Model scores, orchestrating
GPT-4 based pairwise evaluations, facilitating human evaluations, and calculating
pass@k metrics for code generation tasks.
"""

import os
import json
import random
import numpy as np
import torch
import math
import openai
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Union, Optional
from loguru import logger
from omegaconf import DictConfig

# To avoid circular imports, define Config as DictConfig directly.
Config = DictConfig

# Import custom modules needed for EvaluationManager
from data_loader import DataLoader
from models import PolicyModel, RewardModel, SFTModel
from utils import TokenizerWrapper
from code_executor import CodeExecutor # Optional, for APPS task


class EvaluationManager:
    """
    Manages various evaluation procedures for MA-RLHF models.
    """

    def __init__(
        self,
        config: Config,
        data_loader: DataLoader,
        policy_model: PolicyModel, # This is typically the MA-PPO model being evaluated
        reward_model: RewardModel,
        sft_model: SFTModel,
        tokenizer_wrapper: TokenizerWrapper,
        code_executor: Optional[CodeExecutor] = None,
        # An optional baseline policy model could be passed here for live comparisons
        # or baseline responses loaded from file for simplicity.
        # For this implementation, assume baseline responses are loaded from file or `sft_model` if needed.
    ):
        """
        Initializes the EvaluationManager.

        Args:
            config: A DictConfig object containing the global and evaluation configurations.
            data_loader: An instance of DataLoader for loading evaluation datasets.
            policy_model: The PolicyModel (e.g., MA-PPO) to be evaluated.
            reward_model: The RewardModel used to assign scores.
            sft_model: The SFT model, potentially used as a reference or weak baseline.
            tokenizer_wrapper: The TokenizerWrapper instance.
            code_executor: An optional CodeExecutor instance for code generation tasks.
        """
        self.config: Config = config
        self.data_loader: DataLoader = data_loader
        self.policy_model: PolicyModel = policy_model
        self.reward_model: RewardModel = reward_model
        self.sft_model: SFTModel = sft_model # Keep SFT model for potential baselines if needed
        self.tokenizer_wrapper: TokenizerWrapper = tokenizer_wrapper
        self.code_executor: Optional[CodeExecutor] = code_executor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"EvaluationManager initialized. Evaluations will run on device: {self.device}")

        # Set models to evaluation mode
        self.policy_model.eval()
        self.reward_model.eval()
        self.sft_model.eval()

        # Load GPT-4 prompt templates if enabled
        self.gpt4_prompt_templates: Dict[str, str] = {}
        gpt4_config = self.config.evaluation_config
        if gpt4_config.gpt4_model_name: # Check if GPT-4 evaluation is configured
            self._load_gpt4_prompt_templates(gpt4_config.gpt4_prompts_dir)
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            if not openai.api_key:
                logger.warning("OPENAI_API_KEY environment variable not set. GPT-4 evaluation will not work.")
            self.openai_client = openai.OpenAI(api_key=openai.api_key)
        else:
            logger.info("GPT-4 evaluation not configured.")
        
        # Ensure output directory for evaluation results exists
        os.makedirs(os.path.join(self.config.global.output_dir, "eval_results"), exist_ok=True)


    def _load_gpt4_prompt_templates(self, prompts_dir: str):
        """Loads GPT-4 prompt templates from specified directory."""
        if not os.path.isdir(prompts_dir):
            logger.warning(f"GPT-4 prompts directory not found: {prompts_dir}. GPT-4 evaluation templates will be empty.")
            return

        for filename in os.listdir(prompts_dir):
            if filename.endswith(".txt"):
                task_name = filename.replace("_gpt4_prompt.txt", "")
                filepath = os.path.join(prompts_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.gpt4_prompt_templates[task_name] = f.read()
                logger.info(f"Loaded GPT-4 prompt template for task: {task_name} from {filepath}")
        if not self.gpt4_prompt_templates:
            logger.warning(f"No GPT-4 prompt templates found in {prompts_dir}.")

    def _generate_responses(
        self,
        prompts_batch: Dict[str, torch.Tensor],
        policy_model: PolicyModel,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        num_samples: int = 1 # For pass@k, generate multiple samples
    ) -> List[Tuple[torch.Tensor, torch.Tensor, List[str]]]:
        """
        Helper to generate responses from a given policy model.

        Args:
            prompts_batch: A dictionary with 'prompt_ids' and 'attention_mask'.
            policy_model: The policy model to use for generation.
            max_new_tokens: Maximum number of tokens to generate.
            temperature, top_p, top_k: Sampling parameters.
            num_samples: Number of independent samples to generate for each prompt.

        Returns:
            A list of tuples: (generated_ids_full, generated_response_ids, decoded_texts).
            Each tuple contains the full generated sequence, just the response part,
            and the decoded text of the response part.
        """
        generated_data = []
        for _ in range(num_samples):
            with torch.no_grad():
                generated_ids_full, _ = policy_model.generate(
                    prompts_batch['prompt_ids'],
                    prompts_batch['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=(temperature > 0 and (top_p > 0 or top_k > 0)) # Ensure do_sample for non-greedy
                )
            
            prompt_len = prompts_batch['prompt_ids'].shape[1]
            generated_response_ids = generated_ids_full[:, prompt_len:]
            decoded_texts = self.tokenizer_wrapper.decode(generated_response_ids, skip_special_tokens=True)
            
            generated_data.append((generated_ids_full, generated_response_ids, decoded_texts))
        
        return generated_data

    def evaluate_rm_scores(self, task: str) -> Dict[str, Any]:
        """
        Evaluates the generated responses by computing their Reward Model (RM) scores
        and analyzing their distribution.

        Args:
            task: The name of the task (e.g., 'tldr_summarization').

        Returns:
            A dictionary containing evaluation metrics for RM scores.
        """
        logger.info(f"Starting RM score evaluation for task: {task}")
        eval_dataloader = self.data_loader.load_eval_data(task)

        if eval_dataloader is None:
            logger.warning(f"No evaluation data found for task '{task}'. Skipping RM score evaluation.")
            return {"rm_score_mean": float('nan'), "rm_score_std": float('nan')}

        all_rm_scores: List[float] = []
        sampling_params = {
            "max_new_tokens": self.config.ppo_config.max_response_length,
            "temperature": self.config.ppo_config.temperature_sampling,
            "top_p": self.config.ppo_config.top_p,
            "top_k": self.config.ppo_config.top_k,
        }

        with torch.no_grad():
            for batch_idx, batch_prompts in enumerate(eval_dataloader):
                # Ensure prompts are moved to device
                prompt_ids = batch_prompts['prompt_ids'].to(self.device)
                prompt_attention_mask = batch_prompts['attention_mask'].to(self.device)

                # Generate responses using the policy model (MA-PPO)
                ma_ppo_generated_data = self._generate_responses(
                    {'prompt_ids': prompt_ids, 'attention_mask': prompt_attention_mask},
                    self.policy_model,
                    **sampling_params,
                    num_samples=1 # Only 1 sample needed for RM score evaluation
                )
                ma_ppo_generated_ids_full, _, _ = ma_ppo_generated_data[0] # Take the first sample

                # Calculate RM scores for MA-PPO responses
                rm_scores_batch = self.reward_model.get_reward(
                    prompt_ids, ma_ppo_generated_ids_full
                )
                all_rm_scores.extend(rm_scores_batch.cpu().numpy().tolist())

        if not all_rm_scores:
            logger.warning(f"No RM scores collected for task '{task}'.")
            return {"rm_score_mean": float('nan'), "rm_score_std": float('nan')}

        mean_score = np.mean(all_rm_scores)
        std_score = np.std(all_rm_scores)
        min_score = np.min(all_rm_scores)
        max_score = np.max(all_rm_scores)
        
        # Calculate percentiles to describe distribution
        percentiles = {
            f"p{p}": np.percentile(all_rm_scores, p) for p in [10, 25, 50, 75, 90]
        }

        results = {
            "rm_score_mean": mean_score,
            "rm_score_std": std_score,
            "rm_score_min": min_score,
            "rm_score_max": max_score,
            **percentiles,
            "total_samples": len(all_rm_scores)
        }
        logger.info(f"RM Score Evaluation Results for {task}: {results}")
        return results

    def _get_baseline_responses_from_file(self, task: str) -> Dict[str, List[str]]:
        """
        Loads pre-generated baseline responses from a file.
        The file is expected to be a JSONL where each line is:
        {"prompt_text": "...", "response_text": "..."}
        
        This is a placeholder for actual integration with vanilla PPO outputs.
        """
        baseline_path_key = f"{task}_vanilla_ppo_responses_path"
        baseline_file_path = self.config.evaluation_config.get(baseline_path_key)

        if not baseline_file_path or not os.path.exists(baseline_file_path):
            logger.warning(f"Baseline responses file not found for task '{task}' at '{baseline_file_path}'. "
                           "Falling back to SFT model as a weak baseline for GPT-4/Human evaluations.")
            return {} # Return empty dict, caller should handle.

        baseline_responses: Dict[str, List[str]] = defaultdict(list)
        with open(baseline_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    baseline_responses[data["prompt_text"]].append(data["response_text"])
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from baseline responses file: {e} in line: {line}")
                except KeyError as e:
                    logger.error(f"Missing key in baseline responses file: {e} in line: {line}")
        logger.info(f"Loaded {len(baseline_responses)} unique prompts with baseline responses from {baseline_file_path}.")
        return baseline_responses

    def evaluate_gpt4(self, task: str, baseline_responses_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs pairwise comparison of generated responses (MA-PPO vs. a baseline)
        using GPT-4 as an automated judge.

        Args:
            task: The name of the task.
            baseline_responses_path: Optional path to a JSONL file containing baseline responses.
                                     If None, `evaluation_config` will be checked for a default path.

        Returns:
            A dictionary containing MA-PPO's win, tie, and loss rates against the baseline.
        """
        logger.info(f"Starting GPT-4 pairwise evaluation for task: {task}")

        if not self.openai_client:
            logger.error("OpenAI client not initialized. Cannot perform GPT-4 evaluation.")
            return {"ma_ppo_win_rate": 0.0, "ma_ppo_tie_rate": 0.0, "ma_ppo_loss_rate": 0.0}

        gpt4_config = self.config.evaluation_config
        template = self.gpt4_prompt_templates.get(task)
        if not template:
            logger.error(f"GPT-4 prompt template for task '{task}' not found. Cannot proceed with GPT-4 evaluation.")
            return {"ma_ppo_win_rate": 0.0, "ma_ppo_tie_rate": 0.0, "ma_ppo_loss_rate": 0.0}

        # Load limited set of evaluation data
        # We need original prompts for GPT-4. data_loader.load_eval_data yields tokenized prompts.
        # For GPT-4 and human eval, we often need the raw text of the prompt.
        # Let's adjust data_loader to include `prompt_text` in its output for eval_data.
        eval_dataloader = self.data_loader.load_eval_data(task)
        
        # Manually extract a subset of examples (prompts and original content) for GPT-4 eval
        num_samples_for_gpt4 = gpt4_config.gpt4_human_eval_samples
        sampled_prompts_info: List[Dict[str, Any]] = []

        # Iterate through the dataloader to collect `num_samples_for_gpt4` raw examples.
        # This assumes the raw examples are available after tokenization or original prompt text is passed.
        # The `data_loader.load_eval_data` should handle sampling the correct number of raw examples.
        
        # Re-fetch raw data directly as eval_dataloader usually returns tokenized.
        full_eval_dataset = self.data_loader._load_dataset(task, 'eval')
        if len(full_eval_dataset) > num_samples_for_gpt4:
            random.seed(self.config.global.seed)
            sampled_indices = random.sample(range(len(full_eval_dataset)), num_samples_for_gpt4)
            sampled_raw_data = [full_eval_dataset[i] for i in sampled_indices]
        else:
            sampled_raw_data = full_eval_dataset.to_list()
        
        if not sampled_raw_data:
            logger.warning(f"No samples for GPT-4 evaluation for task '{task}'.")
            return {"ma_ppo_win_rate": 0.0, "ma_ppo_tie_rate": 0.0, "ma_ppo_loss_rate": 0.0}

        # Get baseline responses
        baseline_responses = self._get_baseline_responses_from_file(task)
        
        ma_ppo_wins = 0
        ma_ppo_ties = 0
        ma_ppo_losses = 0

        sampling_params = {
            "max_new_tokens": self.config.ppo_config.max_response_length,
            "temperature": self.config.ppo_config.temperature_sampling,
            "top_p": self.config.ppo_config.top_p,
            "top_k": self.config.ppo_config.top_k,
        }

        for i, raw_example in enumerate(sampled_raw_data):
            # Prepare prompt for LLM generation
            prompt_text_for_llm = self.data_loader._get_prompt_text(raw_example, task)
            encoded_prompt = self.tokenizer_wrapper.encode(
                prompt_text_for_llm,
                max_length=self.config.ppo_config.max_prompt_length,
                truncation=True,
                padding='max_length', # Pad to max_length for consistency
                return_tensors="pt"
            )
            prompt_ids = encoded_prompt['input_ids'].to(self.device)
            attention_mask = encoded_prompt['attention_mask'].to(self.device)

            # Generate MA-PPO response
            ma_ppo_generated_data = self._generate_responses(
                {'prompt_ids': prompt_ids, 'attention_mask': attention_mask},
                self.policy_model,
                **sampling_params,
                num_samples=1
            )
            ma_ppo_response_text = ma_ppo_generated_data[0][2][0] # First sample, decoded text

            # Get baseline response
            baseline_response_text: str
            if prompt_text_for_llm in baseline_responses and baseline_responses[prompt_text_for_llm]:
                # If multiple baseline responses are available, pick one (e.g., first one)
                baseline_response_text = baseline_responses[prompt_text_for_llm][0]
            else:
                # If no specific baseline response found, use SFT model as a weak baseline for comparison
                # Generate SFT response
                sft_generated_data = self._generate_responses(
                    {'prompt_ids': prompt_ids, 'attention_mask': attention_mask},
                    self.sft_model,
                    **sampling_params,
                    num_samples=1
                )
                baseline_response_text = sft_generated_data[0][2][0]
                logger.debug(f"Using SFT model response as baseline for prompt {i}.")
            

            # Randomize order for GPT-4
            if random.random() < 0.5:
                response_A = ma_ppo_response_text
                response_B = baseline_response_text
                ma_ppo_is_A = True
            else:
                response_A = baseline_response_text
                response_B = ma_ppo_response_text
                ma_ppo_is_A = False
            
            # Construct GPT-4 request payload
            formatted_prompt_for_gpt4 = self._format_gpt4_prompt(
                task, raw_example, response_A, response_B
            )
            
            messages = [{"role": "user", "content": formatted_prompt_for_gpt4}]

            try:
                chat_completion = self.openai_client.chat.completions.create(
                    model=gpt4_config.gpt4_model_name,
                    messages=messages,
                    temperature=gpt4_config.gpt4_temperature,
                    response_format={"type": "json_object"} # Expect JSON output
                )
                gpt4_choice_str = chat_completion.choices[0].message.content
                gpt4_judgment = json.loads(gpt4_choice_str)
                choice = gpt4_judgment.get('choice', 'E').upper() # Get the choice from JSON, default to 'E'
            except openai.APIError as e:
                logger.error(f"GPT-4 API error: {e}")
                choice = 'E' # Assume tie on API error
            except json.JSONDecodeError as e:
                logger.error(f"GPT-4 response not valid JSON: {gpt4_choice_str}. Error: {e}")
                choice = 'E'
            except Exception as e:
                logger.error(f"An unexpected error occurred during GPT-4 call: {e}")
                choice = 'E'

            # Tally results
            if choice == 'A':
                if ma_ppo_is_A:
                    ma_ppo_wins += 1
                else:
                    ma_ppo_losses += 1
            elif choice == 'B':
                if ma_ppo_is_A:
                    ma_ppo_losses += 1
                else:
                    ma_ppo_wins += 1
            else: # 'E' or any other invalid choice
                ma_ppo_ties += 1

            logger.info(f"GPT-4 Eval {i+1}/{num_samples_for_gpt4}: MA-PPO ({'A' if ma_ppo_is_A else 'B'}) vs Baseline ({'B' if ma_ppo_is_A else 'A'}). Judge: {choice}. Current scores: Wins={ma_ppo_wins}, Ties={ma_ppo_ties}, Losses={ma_ppo_losses}")

        total_evals = ma_ppo_wins + ma_ppo_ties + ma_ppo_losses
        if total_evals == 0:
            return {"ma_ppo_win_rate": 0.0, "ma_ppo_tie_rate": 0.0, "ma_ppo_loss_rate": 0.0}

        win_rate = ma_ppo_wins / total_evals
        tie_rate = ma_ppo_ties / total_evals
        loss_rate = ma_ppo_losses / total_evals

        results = {
            "ma_ppo_win_rate": win_rate,
            "ma_ppo_tie_rate": tie_rate,
            "ma_ppo_loss_rate": loss_rate,
            "total_evaluations": total_evals
        }
        logger.info(f"GPT-4 Evaluation Results for {task}: {results}")
        return results

    def _format_gpt4_prompt(self, task: str, raw_example: Dict[str, Any], response_A: str, response_B: str) -> str:
        """
        Formats the prompt for GPT-4 based on the task and example data.
        """
        template = self.gpt4_prompt_templates.get(task)
        if not template:
            raise ValueError(f"No GPT-4 template found for task: {task}")

        if task == "tldr_summarization":
            # Assume 'post' is the key for the article/post content in raw_example
            return template.format(article=raw_example.get('post', ''), summaryA=response_A, summaryB=response_B)
        elif task == "hh_rlhf":
            # Assume 'query' is the key for the user query in raw_example
            return template.format(query=raw_example.get('query', ''), response_a=response_A, response_b=response_B)
        elif task == "webgpt_comparison":
            # Assume 'question' is the key for the question in raw_example
            return template.format(question=raw_example.get('question', ''), answer_a=response_A, answer_b=response_B)
        else:
            raise ValueError(f"Unsupported task '{task}' for GPT-4 prompt formatting.")


    def evaluate_human(self, task: str, baseline_responses_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepares data for human evaluation by generating responses and formatting them
        for annotation, then provides a placeholder for reporting results.

        Args:
            task: The name of the task.
            baseline_responses_path: Optional path to a JSONL file containing baseline responses.

        Returns:
            A dictionary containing instructions for human evaluation or collected results.
        """
        logger.info(f"Preparing data for human evaluation for task: {task}")

        gpt4_config = self.config.evaluation_config
        num_samples_for_human_eval = gpt4_config.gpt4_human_eval_samples # Reusing this config value

        # Load raw data examples
        full_eval_dataset = self.data_loader._load_dataset(task, 'eval')
        if len(full_eval_dataset) > num_samples_for_human_eval:
            random.seed(self.config.global.seed)
            sampled_indices = random.sample(range(len(full_eval_dataset)), num_samples_for_human_eval)
            sampled_raw_data = [full_eval_dataset[i] for i in sampled_indices]
        else:
            sampled_raw_data = full_eval_dataset.to_list()
        
        if not sampled_raw_data:
            logger.warning(f"No samples for human evaluation for task '{task}'.")
            return {"status": "No samples prepared."}

        # Get baseline responses
        baseline_responses = self._get_baseline_responses_from_file(task)

        # Output file for human annotation tasks
        output_filename = os.path.join(
            self.config.global.output_dir, "eval_results", f"human_eval_data_{task}.jsonl"
        )
        human_eval_records: List[Dict[str, Any]] = []

        sampling_params = {
            "max_new_tokens": self.config.ppo_config.max_response_length,
            "temperature": self.config.ppo_config.temperature_sampling,
            "top_p": self.config.ppo_config.top_p,
            "top_k": self.config.ppo_config.top_k,
        }

        for i, raw_example in enumerate(sampled_raw_data):
            prompt_text_for_llm = self.data_loader._get_prompt_text(raw_example, task)
            encoded_prompt = self.tokenizer_wrapper.encode(
                prompt_text_for_llm,
                max_length=self.config.ppo_config.max_prompt_length,
                truncation=True,
                padding='max_length',
                return_tensors="pt"
            )
            prompt_ids = encoded_prompt['input_ids'].to(self.device)
            attention_mask = encoded_prompt['attention_mask'].to(self.device)

            ma_ppo_generated_data = self._generate_responses(
                {'prompt_ids': prompt_ids, 'attention_mask': attention_mask},
                self.policy_model,
                **sampling_params,
                num_samples=1
            )
            ma_ppo_response_text = ma_ppo_generated_data[0][2][0]

            baseline_response_text: str
            if prompt_text_for_llm in baseline_responses and baseline_responses[prompt_text_for_llm]:
                baseline_response_text = baseline_responses[prompt_text_for_llm][0]
            else:
                sft_generated_data = self._generate_responses(
                    {'prompt_ids': prompt_ids, 'attention_mask': attention_mask},
                    self.sft_model,
                    **sampling_params,
                    num_samples=1
                )
                baseline_response_text = sft_generated_data[0][2][0]

            # Randomize order for human annotators
            if random.random() < 0.5:
                response_A = ma_ppo_response_text
                response_B = baseline_response_text
                model_A_type = "MA-PPO"
                model_B_type = "Baseline"
            else:
                response_A = baseline_response_text
                response_B = ma_ppo_response_text
                model_A_type = "Baseline"
                model_B_type = "MA-PPO"
            
            record = {
                "task": task,
                "prompt": prompt_text_for_llm,
                "response_A": response_A,
                "response_B": response_B,
                "model_A_type": model_A_type, # For internal tracking, not shown to annotator
                "model_B_type": model_B_type, # For internal tracking, not shown to annotator
                "instruction_for_annotators": "Refer to Appendix F.2 in the paper for detailed annotation rules.",
                "human_choice": None # To be filled by annotator ('A', 'B', 'E')
            }
            human_eval_records.append(record)
        
        # Save to file
        with open(output_filename, 'w', encoding='utf-8') as f:
            for record in human_eval_records:
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Prepared {len(human_eval_records)} samples for human evaluation. "
                    f"Saved to: {output_filename}. Annotators should fill in 'human_choice'.")
        
        # This is a placeholder. Realistically, an external script would read this file,
        # collect annotations, and then another script would process the results.
        # For this exercise, we just return the path to the prepared data.
        return {
            "status": "Human evaluation data prepared.",
            "data_file": output_filename,
            "num_annotators_per_sample": gpt4_config.human_eval_annotators_7b if "7b" in self.config.resolved_model_id else gpt4_config.human_eval_annotators_others
        }

    def evaluate_pass_at_k(self, task: str) -> Dict[str, float]:
        """
        Evaluates code generation models (specifically for the APPS dataset)
        using the pass@k metric.

        Args:
            task: The name of the task, must be 'apps_code_gen'.

        Returns:
            A dictionary containing pass@k metrics (e.g., 'pass@1', 'pass@5').
        """
        if task != 'apps_code_gen':
            raise ValueError(f"pass@k evaluation is only supported for 'apps_code_gen' task, got '{task}'.")

        if self.code_executor is None:
            raise ValueError("CodeExecutor must be initialized for pass@k evaluation.")
        
        logger.info(f"Starting pass@k evaluation for task: {task}")
        eval_dataloader = self.data_loader.load_eval_data(task) # This returns raw examples for APPS

        if eval_dataloader is None:
            logger.warning(f"No evaluation data found for task '{task}'. Skipping pass@k evaluation.")
            return {}

        pass_at_k_metrics = self.config.evaluation_config.pass_at_k_metrics
        max_k_to_generate = max(pass_at_k_metrics)

        all_results: List[List[bool]] = [] # List of lists, each inner list is [passed_sample_1, passed_sample_2, ...] for one problem

        sampling_params = {
            "max_new_tokens": self.config.ppo_config.max_response_length,
            "temperature": self.config.ppo_config.temperature_sampling, # Use ppo's sampling params
            "top_p": self.config.ppo_config.top_p,
            "top_k": self.config.ppo_config.top_k,
        }

        for problem_idx, raw_examples_batch in enumerate(eval_dataloader):
            # For APPS, data_loader.load_eval_data returns raw examples, not tokenized tensors in a batch.
            # So raw_examples_batch is a list of dicts. We process one problem at a time for pass@k.
            # Each dict in raw_examples_batch represents a single problem (prompt, tests).
            for example in raw_examples_batch:
                problem_prompt_text = example['prompt'] # This key might vary depending on data_loader output
                unit_tests = example['test_cases'] # This key might vary depending on data_loader output
                
                if not unit_tests:
                    logger.warning(f"Problem {problem_idx} has no unit tests. Skipping.")
                    continue

                encoded_prompt = self.tokenizer_wrapper.encode(
                    problem_prompt_text,
                    max_length=self.config.ppo_config.max_prompt_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors="pt"
                )
                prompt_ids = encoded_prompt['input_ids'].to(self.device)
                attention_mask = encoded_prompt['attention_mask'].to(self.device)

                problem_pass_status: List[bool] = []
                generated_codes_data = self._generate_responses(
                    {'prompt_ids': prompt_ids, 'attention_mask': attention_mask},
                    self.policy_model,
                    **sampling_params,
                    num_samples=max_k_to_generate # Generate up to max_k samples
                )
                
                for _, _, code_texts in generated_codes_data:
                    generated_code = code_texts[0] # Each item in generated_codes_data is for one batch element.
                    
                    # Execute generated code
                    n_pass, n_fail, error_type = self.code_executor.execute_code(generated_code, unit_tests)
                    
                    # Check if all tests passed successfully
                    passed_all_tests = (error_type == 'success' and n_pass == len(unit_tests))
                    problem_pass_status.append(passed_all_tests)

                all_results.append(problem_pass_status)
        
        # Calculate pass@k
        pass_at_k_scores: Dict[str, float] = {}
        for k in pass_at_k_metrics:
            if max_k_to_generate < k:
                logger.warning(f"Cannot calculate pass@{k} as only {max_k_to_generate} samples were generated per problem.")
                continue

            num_problems = len(all_results)
            if num_problems == 0:
                pass_at_k_scores[f"pass@{k}"] = 0.0
                continue

            # Unbiased estimator for pass@k
            # P_k = 1 - product((N-c_i-j)/(N-j)) for j=0 to k-1
            # where N is total samples generated per problem, c_i is number of passing samples for problem i
            
            total_pass_at_k = 0.0
            for problem_pass_status in all_results:
                num_passing_samples_for_problem = sum(1 for p in problem_pass_status[:k] if p) # Only consider up to k samples
                num_generated_for_problem = len(problem_pass_status[:k]) # Actual number of samples considered for this problem for this k
                
                if num_generated_for_problem < k:
                    # If we generated fewer than k samples, then we can only evaluate based on what we have.
                    # This scenario should ideally not happen if max_k_to_generate is set correctly.
                    # If it does, we assume it's a failure for the missing samples.
                    # Or a simpler interpretation: if any of the `num_generated_for_problem` pass, then it passes for this problem.
                    # The more standard way handles N as the *total* samples you _could_ draw from.
                    # For a strict "unbiased estimator", we generated N samples, and check k-subsets.
                    # Here, we have `max_k_to_generate` samples.
                    
                    # If num_passing_samples_for_problem >= 1, then at least one passed within the first 'k' attempts.
                    # This is simpler than the combinatorial formula if not enough samples are available.
                    # Let's stick to the combinatorial if N >= k
                    
                    # For practical implementation of pass@k (not the strict combinatorial formula if N<k):
                    # We check if any of the *available* k samples passed.
                    if num_passing_samples_for_problem >= 1:
                        total_pass_at_k += 1.0
                else:
                    # Unbiased estimator based on Chen et al. (2021)
                    # c_i: number of successful samples out of N_samples_per_problem
                    # N_samples_per_problem: max_k_to_generate
                    # k: pass@k value (e.g., 1, 5)
                    c_i = sum(1 for p in problem_pass_status if p) # total successful samples for this problem
                    N_samples_per_problem = len(problem_pass_status) # total generated samples for this problem (which is max_k_to_generate)
                    
                    if N_samples_per_problem < k: # Should not happen if max_k_to_generate is set properly
                        # Fallback for insufficient samples: assume failure
                        continue
                    
                    if c_i >= k: # If we have enough passing samples, P_k is 1
                        total_pass_at_k += 1.0
                    else:
                        # 1 - (combinations(N-c, k) / combinations(N, k))
                        # Using log-sum-exp for numerical stability
                        log_combinations_N_k = sum(math.log(i) for i in range(N_samples_per_problem - k + 1, N_samples_per_problem + 1)) \
                                             - sum(math.log(i) for i in range(1, k + 1))
                        
                        log_combinations_N_minus_c_k = sum(math.log(i) for i in range(N_samples_per_problem - c_i - k + 1, N_samples_per_problem - c_i + 1)) \
                                                      - sum(math.log(i) for i in range(1, k + 1))
                        
                        prob_all_fail = math.exp(log_combinations_N_minus_c_k - log_combinations_N_k)
                        total_pass_at_k += (1.0 - prob_all_fail)

            pass_at_k_scores[f"pass@{k}"] = total_pass_at_k / num_problems if num_problems > 0 else 0.0

        logger.info(f"Pass@k Evaluation Results for {task}: {pass_at_k_scores}")
        return pass_at_k_scores


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # --- Mock Config ---
    mock_config_yaml = """
    global:
      output_dir: "./outputs_eval_test"
      logging_dir: "./logs_eval_test"
      seed: 42
      precision: "float32"
      code_execution_timeout: 5 # For CodeExecutor

    model_configs:
      test_model:
        name: "gpt2"
        type: "causal_lm"
      codegemma_test:
        name: "gpt2" # Using gpt2 for mock CodeGemma
        type: "causal_lm"

    data_configs:
      base_data_path: "./data_eval_test"
      tldr_summarization:
        train_file: "tldr_train.jsonl"
        eval_file: "tldr_eval.jsonl"
        prompt_template: "POST Subreddit: {post}\nSummary:"
      apps_code_gen:
        train_file: "apps_train.jsonl"
        eval_file: "apps_eval.jsonl"
        prompt_template: "Problem: {problem}\nCode:"
      sft_data_ratio: 0.2
      rm_data_ratio: 0.4
      ppo_data_ratio: 0.4

    ppo_config:
      max_response_length: 50
      temperature_sampling: 0.7
      top_p: 0.9
      top_k: 0
      max_prompt_length: 256 # Added for evaluation

    evaluation_config:
      eval_batch_size: 4
      rm_eval_samples: 10 # Sample small for test
      gpt4_human_eval_samples: 5
      gpt4_model_name: "gpt-3.5-turbo-instruct" # Use a mock or cheap model for testing if API key exists
      gpt4_temperature: 0.0
      gpt4_prompts_dir: "./gpt4_prompts_test"
      human_eval_annotators_7b: 3
      human_eval_annotators_others: 1
      pass_at_k_metrics: [1, 2] # Test pass@1 and pass@2
      tldr_summarization_vanilla_ppo_responses_path: "./data_eval_test/tldr_vanilla_ppo_responses.jsonl"
    """
    mock_config = OmegaConf.create(mock_config_yaml)
    mock_config.resolved_model_id = "test_model" # Set resolved model ID for config logic


    # --- Mock Data and Files ---
    os.makedirs(mock_config.global.output_dir, exist_ok=True)
    os.makedirs(mock_config.global.logging_dir, exist_ok=True)
    os.makedirs(mock_config.data_configs.base_data_path, exist_ok=True)
    os.makedirs(mock_config.evaluation_config.gpt4_prompts_dir, exist_ok=True)

    # Create dummy SFT model output for RM init.
    # We won't actually save a full model, just a directory to simulate it.
    os.makedirs(os.path.join(mock_config.global.output_dir, "sft_model"), exist_ok=True)


    # Mock GPT-4 prompt templates
    with open(os.path.join(mock_config.evaluation_config.gpt4_prompts_dir, "tldr_summarization_gpt4_prompt.txt"), "w") as f:
        f.write("You will be given an article and two summaries (A and B). Pick the better one.\nArticle: {article}\nSummary A:{summaryA}\nSummary B:{summaryB}\nYour Choice (JSON): {\"choice\": \"E\"}")
    with open(os.path.join(mock_config.evaluation_config.gpt4_prompts_dir, "hh_rlhf_gpt4_prompt.txt"), "w") as f:
        f.write("Which response is more helpful? Query: {query}\nResponse A: {response_a}\nResponse B: {response_b}\nYour Choice (JSON): {\"choice\": \"A\"}")


    # Mock Data for TLDR
    tldr_eval_data = [
        {"post": "This is article 1. It's about cats.", "summary": "Cats are great.", "chosen_response": "Cats are great.", "rejected_response": "Dogs are loud."},
        {"post": "Article 2. About dogs.", "summary": "Dogs are loyal.", "chosen_response": "Dogs are loyal.", "rejected_response": "Cats are lazy."}
    ]
    with open(os.path.join(mock_config.data_configs.base_data_path, mock_config.data_configs.tldr_summarization.eval_file), "w") as f:
        for item in tldr_eval_data:
            f.write(json.dumps(item) + '\n')
    
    # Mock Baseline Responses file for TLDR
    tldr_vanilla_ppo_responses = [
        {"prompt_text": "POST Subreddit: This is article 1. It's about cats.\nSummary:", "response_text": "Vanilla PPO response for cats."},
        {"prompt_text": "POST Subreddit: Article 2. About dogs.\nSummary:", "response_text": "Vanilla PPO response for dogs."}
    ]
    with open(mock_config.evaluation_config.tldr_summarization_vanilla_ppo_responses_path, "w") as f:
        for item in tldr_vanilla_ppo_responses:
            f.write(json.dumps(item) + '\n')


    # Mock Data for APPS
    apps_eval_data = [
        {"prompt": "Write a function that adds two numbers.", "test_cases": ["assert solve(1,2)==3", "assert solve(-1,1)==0"]},
        {"prompt": "Write a function that subtracts two numbers.", "test_cases": ["assert solve(5,2)==3", "assert solve(1,1)==0"]}
    ]
    with open(os.path.join(mock_config.data_configs.base_data_path, mock_config.data_configs.apps_code_gen.eval_file), "w") as f:
        for item in apps_eval_data:
            f.write(json.dumps(item) + '\n')


    # --- Mock Models and TokenizerWrapper ---
    class MockTokenizerWrapper:
        def __init__(self, model_name="gpt2"):
            self.tokenizer = openai.tokenizer.GPT2TokenizerFast.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        def encode(self, text, add_special_tokens=True, max_length=None, truncation=True, padding='max_length', return_tensors="pt"):
            return self.tokenizer(text, add_special_tokens=add_special_tokens, max_length=max_length, truncation=truncation, padding=padding, return_tensors=return_tensors)
        def decode(self, token_ids, skip_special_tokens=True):
            return self.tokenizer.decode(token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids, skip_special_tokens=skip_special_tokens)

    mock_tokenizer_wrapper = MockTokenizerWrapper()

    class MockPolicyModel(PolicyModel):
        def __init__(self, model_name, config):
            # Bypass actual model loading for mock
            nn.Module.__init__(self)
            self.model_name = model_name
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.torch_dtype = torch.float32
            self.tokenizer = mock_tokenizer_wrapper.tokenizer # Use mock tokenizer
            self.eval()

        def generate(self, prompt_ids, attention_mask, max_new_tokens, temperature, top_p, top_k, do_sample=True, output_scores=False, return_dict_in_generate=True, **kwargs):
            # Simulate generation based on prompt content
            simulated_responses = []
            for i in range(prompt_ids.shape[0]):
                prompt_text = mock_tokenizer_wrapper.decode(prompt_ids[i], skip_special_tokens=True)
                if "cats" in prompt_text:
                    simulated_responses.append(prompt_text + " This is a MA-PPO cat response." * (max_new_tokens // 5))
                elif "dogs" in prompt_text:
                    simulated_responses.append(prompt_text + " This is a MA-PPO dog response." * (max_new_tokens // 5))
                elif "adds two numbers" in prompt_text:
                    simulated_responses.append(prompt_text + "\ndef solve(a, b):\n    return a + b\n")
                elif "subtracts two numbers" in prompt_text:
                    simulated_responses.append(prompt_text + "\ndef solve(a, b):\n    return a - b\n")
                else:
                    simulated_responses.append(prompt_text + " This is a generic MA-PPO response." * (max_new_tokens // 5))

            encoded_responses = mock_tokenizer_wrapper.encode(
                simulated_responses,
                add_special_tokens=False,
                padding='max_length',
                max_length=prompt_ids.shape[1] + max_new_tokens,
                truncation=True,
                return_tensors="pt"
            )
            # Log probs are simulated as zeros for simplicity in this mock
            log_probs = torch.zeros_like(encoded_responses['input_ids'], dtype=self.torch_dtype)
            
            # Mock generate output
            return openai.tokenizer.GPT2TokenizerFast.from_pretrained("gpt2")._generate_return_dict( # Use actual transformers class here
                sequences=encoded_responses['input_ids'],
                scores=None # Skip complex score generation for mock
            )

    class MockRewardModel(RewardModel):
        def __init__(self, model_name, config):
            nn.Module.__init__(self)
            self.model_name = model_name
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.torch_dtype = torch.float32
            self.tokenizer = mock_tokenizer_wrapper.tokenizer # Use mock tokenizer
            self.eval()

        def get_reward(self, prompt_ids, response_ids):
            # Simulate RM scores. Give higher score for MA-PPO
            scores = torch.rand(prompt_ids.shape[0], device=self.device, dtype=self.torch_dtype) * 2 - 1 # Scores between -1 and 1
            for i in range(prompt_ids.shape[0]):
                full_text = mock_tokenizer_wrapper.decode(response_ids[i], skip_special_tokens=True)
                if "MA-PPO" in full_text: # Assume MA-PPO responses are better
                    scores[i] = scores[i] + 0.5
            return scores

    # Mock SFT model for _get_baseline_responses_from_file fallback
    class MockSFTModel(SFTModel):
        def __init__(self, model_name, config):
            nn.Module.__init__(self)
            self.model_name = model_name
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.torch_dtype = torch.float32
            self.tokenizer = mock_tokenizer_wrapper.tokenizer # Use mock tokenizer
            self.eval()

        def generate(self, prompt_ids, attention_mask, max_new_tokens, temperature, top_p, top_k, do_sample=True, output_scores=False, return_dict_in_generate=True, **kwargs):
            simulated_responses = []
            for i in range(prompt_ids.shape[0]):
                prompt_text = mock_tokenizer_wrapper.decode(prompt_ids[i], skip_special_tokens=True)
                if "cats" in prompt_text:
                    simulated_responses.append(prompt_text + " This is a SFT cat response." * (max_new_tokens // 5))
                elif "dogs" in prompt_text:
                    simulated_responses.append(prompt_text + " This is a SFT dog response." * (max_new_tokens // 5))
                else:
                    simulated_responses.append(prompt_text + " This is a generic SFT response." * (max_new_tokens // 5))
            
            encoded_responses = mock_tokenizer_wrapper.encode(
                simulated_responses,
                add_special_tokens=False,
                padding='max_length',
                max_length=prompt_ids.shape[1] + max_new_tokens,
                truncation=True,
                return_tensors="pt"
            )
            log_probs = torch.zeros_like(encoded_responses['input_ids'], dtype=self.torch_dtype)
            
            return openai.tokenizer.GPT2TokenizerFast.from_pretrained("gpt2")._generate_return_dict(
                sequences=encoded_responses['input_ids'],
                scores=None
            )

    # Initialize mock models
    mock_policy_model = MockPolicyModel(mock_config.model_configs.test_model.name, mock_config)
    mock_reward_model = MockRewardModel(mock_config.model_configs.test_model.name, mock_config)
    mock_sft_model = MockSFTModel(mock_config.model_configs.test_model.name, mock_config)


    # --- Test DataLoader (re-initialize to account for mock_tokenizer_wrapper) ---
    mock_data_loader = DataLoader(mock_config, mock_tokenizer_wrapper)

    # --- Test CodeExecutor ---
    mock_code_executor = CodeExecutor(mock_config)


    # --- Instantiate EvaluationManager ---
    eval_manager = EvaluationManager(
        config=mock_config,
        data_loader=mock_data_loader,
        policy_model=mock_policy_model,
        reward_model=mock_reward_model,
        sft_model=mock_sft_model,
        tokenizer_wrapper=mock_tokenizer_wrapper,
        code_executor=mock_code_executor
    )

    # --- Run Tests ---
    print("\n##### Testing RM Scores #####")
    rm_results = eval_manager.evaluate_rm_scores(task="tldr_summarization")
    print(rm_results)
    assert "rm_score_mean" in rm_results and not math.isnan(rm_results["rm_score_mean"])


    print("\n##### Testing GPT-4 Evaluation #####")
    # Set a dummy API key for test to pass client initialization, even if actual calls mock
    os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    # Overwrite openai client with a mock for `chat.completions.create`
    class MockOpenAIClient:
        def chat(self):
            return self
        def completions(self):
            return self
        def create(self, **kwargs):
            # Simulate GPT-4 choosing A for MA-PPO (response A for first, B for second if randomized)
            # or simply alternating
            if "response_A" in kwargs["messages"][0]["content"] and "MA-PPO cat" in kwargs["messages"][0]["content"]:
                choice = "A"
            elif "response_B" in kwargs["messages"][0]["content"] and "MA-PPO dog" in kwargs["messages"][0]["content"]:
                choice = "B"
            else:
                choice = "E"
            
            mock_response = {
                "choices": [{"message": {"content": json.dumps({"choice": choice})}}]
            }
            # Use dot access for consistency with the real client
            class MockChoice:
                def __init__(self, message_content):
                    self.message = type('MockMessage', (object,), {'content': message_content})()
            class MockChatCompletion:
                def __init__(self, choices_data):
                    self.choices = [MockChoice(c['message']['content']) for c in choices_data]
            return MockChatCompletion(mock_response["choices"])

    eval_manager.openai_client = MockOpenAIClient()

    gpt4_results = eval_manager.evaluate_gpt4(task="tldr_summarization", baseline_responses_path=mock_config.evaluation_config.tldr_summarization_vanilla_ppo_responses_path)
    print(gpt4_results)
    assert gpt4_results["total_evaluations"] > 0


    print("\n##### Testing Human Evaluation Prep #####")
    human_eval_results = eval_manager.evaluate_human(task="tldr_summarization", baseline_responses_path=mock_config.evaluation_config.tldr_summarization_vanilla_ppo_responses_path)
    print(human_eval_results)
    assert os.path.exists(human_eval_results["data_file"])


    print("\n##### Testing Pass@k Evaluation #####")
    mock_config.resolved_model_id = "codegemma_test" # Change resolved model ID for APPS specific config access
    pass_at_k_results = eval_manager.evaluate_pass_at_k(task="apps_code_gen")
    print(pass_at_k_results)
    assert "pass@1" in pass_at_k_results and "pass@2" in pass_at_k_results


    # --- Cleanup mock files and directories ---
    import shutil
    shutil.rmtree(mock_config.global.output_dir, ignore_errors=True)
    shutil.rmtree(mock_config.global.logging_dir, ignore_errors=True)
    shutil.rmtree(mock_config.data_configs.base_data_path, ignore_errors=True)
    shutil.rmtree(mock_config.evaluation_config.gpt4_prompts_dir, ignore_errors=True)
    print("\nCleanup complete.")

    print("\nAll EvaluationManager tests passed!")

