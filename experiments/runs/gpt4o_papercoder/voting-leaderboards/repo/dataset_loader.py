# dataset_loader.py

import os
import json
import pandas as pd
from typing import List, Dict


class DatasetLoader:
    """Handles loading of prompts and model response data."""

    def __init__(self, config: dict) -> None:
        """
        Initializes DatasetLoader with configuration.

        Args:
            config (dict): Configuration dictionary with data file paths.
        """
        self.prompts_file = config.get("data", {}).get("prompts_file", "data/prompts.json")
        self.responses_dir = config.get("data", {}).get("responses_dir", "data/model_responses/")
        self.voting_logs_file = config.get("data", {}).get("voting_logs_file", "data/voting_logs.json")

        # Validate existence of required files and directories
        if not os.path.exists(self.prompts_file):
            raise FileNotFoundError(f"Prompts file not found at {self.prompts_file}")
        if not os.path.exists(self.responses_dir):
            raise FileNotFoundError(f"Responses directory not found at {self.responses_dir}")
        if not os.path.isdir(self.responses_dir):
            raise NotADirectoryError(f"Expected a directory at {self.responses_dir}")

    def load_prompt_data(self) -> List[str]:
        """
        Loads prompts from the JSON file and combines them into a single list.

        Returns:
            List[str]: A combined list of prompts across all categories.

        Raises:
            ValueError: If the prompts file contains invalid or unexpected data.
        """
        try:
            with open(self.prompts_file, "r", encoding="utf-8") as file:
                prompt_data = json.load(file)

            all_prompts = []
            for category, prompts in prompt_data.items():
                if not isinstance(prompts, list):
                    raise ValueError(f"Invalid format for prompts in category '{category}'. Expected a list.")
                all_prompts.extend(prompts)

            return all_prompts
        except Exception as e:
            raise ValueError(f"Failed to read or parse prompts file: {e}")

    def load_response_data(self) -> Dict[str, Dict[str, str]]:
        """
        Loads model responses from JSON files in the responses directory.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary where:
                - keys are model names (extracted from filenames),
                - values are dictionaries mapping prompts to responses.

        Raises:
            ValueError: If individual response files are invalid or not properly structured.
        """
        model_responses = {}

        try:
            # Iterate through each file in the responses directory
            for filename in os.listdir(self.responses_dir):
                file_path = os.path.join(self.responses_dir, filename)

                # Only process JSON files
                if not filename.endswith(".json"):
                    continue

                model_name = os.path.splitext(filename)[0]  # Extract model name from filename
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        responses = json.load(file)

                    # Validate that responses are in a dictionary format
                    if not isinstance(responses, dict):
                        raise ValueError(f"Invalid response data in '{filename}'. Expected a dictionary.")

                    model_responses[model_name] = responses
                except Exception as e:
                    raise ValueError(f"Failed to load or parse responses file '{file_path}': {e}")

            if not model_responses:
                raise ValueError(f"No valid response files found in directory: {self.responses_dir}")

            return model_responses
        except Exception as e:
            raise ValueError(f"Error while loading response data: {e}")
