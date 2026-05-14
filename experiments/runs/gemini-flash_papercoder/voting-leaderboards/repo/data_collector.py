import logging
import os
import random
from typing import List, Dict, Any, Optional

from config import Config
from llm_api_interface import LLMAPIInterface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Manages the acquisition of prompts from specified sources and the collection
    of responses from LLMs via the LLMAPIInterface.
    """

    def __init__(self, llm_api: LLMAPIInterface, config: Config):
        """
        Initializes the DataCollector with necessary dependencies.

        Args:
            llm_api (LLMAPIInterface): An instance of the LLMAPIInterface for querying models.
            config (Config): The configuration object containing experiment parameters.
        """
        self.llm_api = llm_api
        self.config = config
        random.seed(self.config.RANDOM_SEED)  # Set seed for reproducibility in sampling prompts

    def load_prompts(self, source_name: str, category: str, num_prompts: int) -> List[str]:
        """
        Loads a specified number of prompts from a designated source and category.
        Assumes prompt datasets are available as local text files.

        Args:
            source_name (str): The name of the prompt source (e.g., "LMSYS-Chat-1M").
            category (str): The specific category of prompts (e.g., "english_chat", "math").
            num_prompts (int): The desired number of prompts to load.

        Returns:
            List[str]: A list of selected prompts.
        """
        # Adopt a convention: data/prompts/{source_name_lower}/{category_lower}.txt
        # Example: data/prompts/lmsys-chat-1m/english_chat.txt
        # Replace spaces or special chars for path safety if source_name has them
        safe_source_name = source_name.lower().replace(" ", "-")
        safe_category_name = category.lower().replace(" ", "-")

        prompt_file_path = os.path.join(
            "data", "prompts", safe_source_name, f"{safe_category_name}.txt"
        )
        all_prompts: List[str] = []

        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line:  # Only add non-empty lines
                        all_prompts.append(stripped_line)
            logger.info(f"Loaded {len(all_prompts)} prompts from {prompt_file_path}.")
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_file_path}. Returning empty list.")
            return []
        except Exception as e:
            logger.error(f"Error reading prompt file {prompt_file_path}: {e}. Returning empty list.")
            return []

        if not all_prompts:
            logger.warning(f"No prompts found in {prompt_file_path}.")
            return []

        # Sample num_prompts from the loaded list
        if len(all_prompts) > num_prompts:
            return random.sample(all_prompts, num_prompts)
        else:
            logger.warning(f"Requested {num_prompts} prompts, but only {len(all_prompts)} available in {prompt_file_path}. Returning all available prompts.")
            return all_prompts

    def collect_identity_responses(
        self, models: List[Dict], prompts: List[str], num_queries_per_prompt: int
    ) -> List[Dict]:
        """
        Collects responses from each model for a set of identity-probing prompts,
        multiple times per prompt, to evaluate the identity-probing detector's accuracy.

        Args:
            models (List[Dict]): A list of model dictionaries, typically from config.MODEL_LIST.
            prompts (List[str]): A list of identity-probing prompt strings.
            num_queries_per_prompt (int): The number of times to query each model with each prompt.

        Returns:
            List[Dict]: A list of dictionaries, each containing 'model_id', 'prompt', and 'response'.
        """
        all_collected_data: List[Dict] = []
        total_queries = len(models) * len(prompts) * num_queries_per_prompt
        query_count = 0

        logger.info(f"Starting identity response collection for {len(models)} models, "
                    f"{len(prompts)} prompts, {num_queries_per_prompt} queries per prompt. "
                    f"Total queries expected: {total_queries}")

        for model_info in models:
            model_id = model_info.get('model_id')
            if not model_id:
                logger.warning(f"Skipping model due to missing 'model_id' in: {model_info}")
                continue

            for prompt_text in prompts:
                for i in range(num_queries_per_prompt):
                    query_count += 1
                    if query_count % 100 == 0 or query_count == total_queries:
                        logger.info(f"Progress: {query_count}/{total_queries} identity queries completed.")

                    response_text = self.llm_api.query_model(
                        model_id, prompt_text, self.config.DETECTOR_OUTPUT_TOKEN_LENGTH
                    )

                    if response_text is None:
                        logger.warning(f"Failed to get response for model '{model_id}' with prompt '{prompt_text}' (Query {i+1}). Skipping.")
                        continue

                    all_collected_data.append({
                        'model_id': model_id,
                        'prompt': prompt_text,
                        'response': response_text,
                        'query_idx': i  # Added for traceability if needed
                    })
        logger.info(f"Finished identity response collection. Collected {len(all_collected_data)} responses.")
        return all_collected_data

    def collect_training_responses(
        self, models: List[Dict], prompt_categories: List[str],
        num_prompts_per_category: int, num_responses_per_model: int
    ) -> List[Dict]:
        """
        Collects a large dataset of responses for the training-based detector,
        covering diverse prompt categories and multiple responses per model per prompt.

        Args:
            models (List[Dict]): A list of model dictionaries, typically from config.MODEL_LIST.
            prompt_categories (List[str]): A list of prompt category names (keys from config.PROMPT_SOURCES).
            num_prompts_per_category (int): The number of prompts to load for each category.
            num_responses_per_model (int): The number of responses to generate for each model
                                           for each selected prompt.

        Returns:
            List[Dict]: A list of dictionaries, each containing 'model_id', 'category',
                        'prompt', and 'response'.
        """
        all_collected_data: List[Dict] = []
        total_expected_prompts = len(prompt_categories) * num_prompts_per_category
        total_expected_responses = total_expected_prompts * len(models) * num_responses_per_model
        responses_count = 0
        
        logger.info(f"Starting training response collection for {len(models)} models, "
                    f"{len(prompt_categories)} categories, {num_prompts_per_category} prompts/category, "
                    f"{num_responses_per_model} responses/model/prompt. "
                    f"Total expected responses: {total_expected_responses}")

        for category in prompt_categories:
            source_name = self.config.PROMPT_SOURCES.get(category)
            if not source_name:
                logger.error(f"Source name not found for category '{category}'. Skipping.")
                continue

            category_prompts = self.load_prompts(source_name, category, num_prompts_per_category)
            if not category_prompts:
                logger.warning(f"No prompts loaded for category '{category}'. Skipping.")
                continue

            for prompt_text in category_prompts:
                for model_info in models:
                    model_id = model_info.get('model_id')
                    if not model_id:
                        logger.warning(f"Skipping model due to missing 'model_id' in: {model_info}")
                        continue

                    for i in range(num_responses_per_model):
                        responses_count += 1
                        if responses_count % 1000 == 0 or responses_count == total_expected_responses:
                            logger.info(f"Progress: {responses_count}/{total_expected_responses} training responses collected.")

                        response_text = self.llm_api.query_model(
                            model_id, prompt_text, self.config.DETECTOR_OUTPUT_TOKEN_LENGTH
                        )

                        if response_text is None:
                            logger.warning(f"Failed to get response for model '{model_id}' with prompt '{prompt_text}' (Attempt {i+1}). Skipping.")
                            continue

                        all_collected_data.append({
                            'model_id': model_id,
                            'category': category,
                            'prompt': prompt_text,
                            'response': response_text,
                            'response_idx': i # Added for traceability if needed
                        })
        logger.info(f"Finished training response collection. Collected {len(all_collected_data)} responses.")
        return all_collected_data

