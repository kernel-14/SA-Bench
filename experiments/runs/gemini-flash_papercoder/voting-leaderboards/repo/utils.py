import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
from typing import List, Dict, Tuple, Any, Optional

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

# Pre-defined keyword mapping for models based on common knowledge and config.yaml structure.
# This mapping is hardcoded to avoid circular dependencies with config.py,
# while adhering to the spirit of deriving keywords from model_id and company info.
# Keys are model family prefixes, values are lists of lowercase identifying keywords.
_MODEL_KEYWORD_MAPPING: Dict[str, List[str]] = {
    "claude": ["claude", "anthropic"],
    "gemini": ["gemini", "google"],
    "gpt": ["gpt", "openai"],
    "gemma": ["gemma", "google"],
    "llama": ["llama", "meta"],
    "mixtral": ["mixtral", "mistral", "mistral ai"], # "Mistral AI" is the company name
    "qwen": ["qwen", "alibaba"],
    # Add more mappings here if other model families are introduced
}


def calculate_accuracy(predictions: List[bool], labels: List[bool]) -> float:
    """
    Calculates the accuracy of binary predictions against true labels.

    Args:
        predictions (List[bool]): A list of boolean predictions.
        labels (List[bool]): A list of true boolean labels.

    Returns:
        float: The accuracy as a float between 0.0 and 1.0.

    Raises:
        ValueError: If the lengths of predictions and labels do not match.
    """
    if len(predictions) != len(labels):
        raise ValueError("Lengths of predictions and labels must be equal.")

    if not predictions:  # Handle empty lists
        return 0.0

    correct_predictions = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct_predictions / len(predictions)
    return accuracy


def preprocess_text(text: str) -> str:
    """
    Preprocesses text by converting to lowercase, tokenizing, and removing non-alphanumeric
    characters. The tokens are then rejoined into a single string.

    Args:
        text (str): The input text string.

    Returns:
        str: The preprocessed text string.
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string input

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Filter out tokens that are not alphanumeric (e.g., punctuation)
    # Keeping isalnum to retain words like "llama3" or "gpt4o" if they appear.
    # Note: If numbers alone are considered noise for BoW/TF-IDF, further filtering might be needed.
    # For "simple text features" and typical NLP practices, this is a reasonable approach.
    filtered_tokens = [word for word in tokens if word.isalnum()]

    # Rejoin the tokens into a single string
    preprocessed_text = " ".join(filtered_tokens)

    return preprocessed_text


def get_keyword_for_model(model_id: str) -> List[str]:
    """
    Provides a list of identifying keywords for a given model_id.
    These keywords are used by the identity-probing detector.

    Args:
        model_id (str): The identifier of the model (e.g., "claude-3-5-sonnet-20240620").

    Returns:
        List[str]: A list of lowercase keywords associated with the model.
                   Returns an empty list if no keywords are found for the model family.
    """
    if not isinstance(model_id, str):
        return []

    # Extract the model family prefix (e.g., "claude" from "claude-3-5-sonnet-20240620")
    model_family_prefix = model_id.split('-')[0].lower()

    return _MODEL_KEYWORD_MAPPING.get(model_family_prefix, [])


# Placeholder for table parsing, if needed (not directly used in current design but suggested in data structures)
def parse_table_data(table_html: str) -> Any:
    """
    A placeholder function for parsing HTML table data.
    This function is not detailed in the current plan's logic analysis,
    but is included to match the Data Structures & Interfaces.
    """
    # Implementation would typically involve BeautifulSoup or similar library
    # For now, it returns a placeholder or raises NotImplementedError
    raise NotImplementedError("parse_table_data is a placeholder and not implemented.")

