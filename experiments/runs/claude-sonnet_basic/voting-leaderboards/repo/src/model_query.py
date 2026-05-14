"""
Model querying module for collecting responses from various LLM APIs.
Supports OpenAI, Anthropic, Google AI Studio, and Together AI APIs.
"""

import os
import time
import logging
from typing import Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelQuerier(ABC):
    """Abstract base class for model queriers."""

    @abstractmethod
    def query(self, prompt: str, max_tokens: int = 512) -> str:
        """Query the model with a prompt and return the response."""
        pass


class OpenAIQuerier(ModelQuerier):
    """Querier for OpenAI models (GPT-3.5, GPT-4, GPT-4o, etc.)."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def query(self, prompt: str, max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class AnthropicQuerier(ModelQuerier):
    """Querier for Anthropic models (Claude)."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        self.model_name = model_name
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def query(self, prompt: str, max_tokens: int = 512) -> str:
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


class GoogleAIQuerier(ModelQuerier):
    """Querier for Google AI Studio models (Gemini)."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required: pip install google-generativeai"
            )

        self.model_name = model_name
        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def query(self, prompt: str, max_tokens: int = 512) -> str:
        import google.generativeai as genai

        generation_config = genai.GenerationConfig(max_output_tokens=max_tokens)
        response = self.model.generate_content(
            prompt, generation_config=generation_config
        )
        return response.text


class TogetherAIQuerier(ModelQuerier):
    """Querier for Together AI models (open-source models)."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        try:
            from together import Together
        except ImportError:
            raise ImportError("together package required: pip install together")

        self.model_name = model_name
        self.client = Together(api_key=api_key or os.environ.get("TOGETHER_API_KEY"))

    def query(self, prompt: str, max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def get_querier(model_name: str, query_method: str) -> ModelQuerier:
    """Factory function to get the appropriate querier for a model."""
    method_map = {
        "openai_api": OpenAIQuerier,
        "anthropic_api": AnthropicQuerier,
        "google_ai_studio_api": GoogleAIQuerier,
        "together_ai_api": TogetherAIQuerier,
    }

    if query_method not in method_map:
        raise ValueError(f"Unknown query method: {query_method}")

    return method_map[query_method](model_name)


def collect_responses(
    model_name: str,
    query_method: str,
    prompts: list,
    n_responses_per_prompt: int = 50,
    max_tokens: int = 512,
    retry_delay: float = 1.0,
    max_retries: int = 3,
) -> dict:
    """
    Collect multiple responses from a model for a list of prompts.

    Args:
        model_name: Name of the model to query
        query_method: API method to use
        prompts: List of prompts to query
        n_responses_per_prompt: Number of responses to collect per prompt
        max_tokens: Maximum tokens per response
        retry_delay: Delay between retries on failure
        max_retries: Maximum number of retries per query

    Returns:
        Dictionary mapping prompt -> list of responses
    """
    querier = get_querier(model_name, query_method)
    results = {}

    for prompt in prompts:
        responses = []
        for i in range(n_responses_per_prompt):
            for attempt in range(max_retries):
                try:
                    response = querier.query(prompt, max_tokens=max_tokens)
                    responses.append(response)
                    break
                except Exception as e:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for "
                        f"model {model_name}, prompt '{prompt[:50]}...': {e}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                    else:
                        logger.error(
                            f"All retries failed for model {model_name}"
                        )
                        responses.append("")  # Empty response on failure

        results[prompt] = responses
        logger.info(
            f"Collected {len(responses)} responses for model {model_name}, "
            f"prompt '{prompt[:50]}...'"
        )

    return results
