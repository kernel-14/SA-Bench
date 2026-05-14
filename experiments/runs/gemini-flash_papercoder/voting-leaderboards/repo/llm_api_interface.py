import logging
import requests
import anthropic
import google.generativeai
import openai
import os

from typing import Dict, List, Any, Optional
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMAPIInterface:
    """
    Provides an abstraction layer for interacting with various LLM APIs
    (Anthropic, Google AI Studio, OpenAI, Together AI).
    """

    def __init__(self, config: Config):
        """
        Initializes the LLMAPIInterface by setting up API clients for each supported
        LLM provider using the provided API keys from the Config object.

        Args:
            config (Config): The configuration object containing API keys and model details.
        """
        self.config = config
        self.api_keys = config.LLM_API_KEYS
        self.model_list = config.MODEL_LIST

        self.anthropic_client: Optional[anthropic.Anthropic] = None
        self.openai_client: Optional[openai.OpenAI] = None
        self.google_api_configured: bool = False # Google's API often configures globally
        self.together_ai_api_key: Optional[str] = None

        # Initialize Anthropic client
        anthropic_api_key = self.api_keys.get('ANTHROPIC_API_KEY')
        if anthropic_api_key and anthropic_api_key != "your_anthropic_api_key_here":
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                logger.info("Anthropic client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
        else:
            logger.warning("Anthropic API key not provided or is default placeholder. Anthropic models will not be queryable.")

        # Configure Google AI Studio
        google_api_key = self.api_keys.get('GOOGLE_API_KEY')
        if google_api_key and google_api_key != "your_google_api_key_here":
            try:
                google.generativeai.configure(api_key=google_api_key)
                self.google_api_configured = True
                logger.info("Google AI Studio configured.")
            except Exception as e:
                logger.error(f"Failed to configure Google AI Studio: {e}")
        else:
            logger.warning("Google API key not provided or is default placeholder. Google models will not be queryable.")

        # Initialize OpenAI client
        openai_api_key = self.api_keys.get('OPENAI_API_KEY')
        if openai_api_key and openai_api_key != "your_openai_api_key_here":
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("OpenAI API key not provided or is default placeholder. OpenAI models will not be queryable.")

        # Store Together AI API key
        together_ai_api_key = self.api_keys.get('TOGETHER_AI_API_KEY')
        if together_ai_api_key and together_ai_api_key != "your_together_ai_api_key_here":
            self.together_ai_api_key = together_ai_api_key
            logger.info("Together AI API key loaded.")
        else:
            logger.warning("Together AI API key not provided or is default placeholder. Together AI models will not be queryable.")

    def _get_model_config(self, model_id: str) -> Optional[Dict[str, str]]:
        """
        Helper method to retrieve the configuration dictionary for a given model_id.

        Args:
            model_id (str): The identifier of the model.

        Returns:
            Optional[Dict[str, str]]: The model's configuration dictionary if found, else None.
        """
        for model_info in self.model_list:
            if model_info.get('model_id') == model_id:
                return model_info
        return None

    def query_model(self, model_id: str, prompt: str, max_tokens: int) -> Optional[str]:
        """
        The main entry point for querying any supported LLM. It identifies the correct
        API method based on the model_id and dispatches the call.

        Args:
            model_id (str): The unique identifier of the model.
            prompt (str): The text prompt to send to the LLM.
            max_tokens (int): The maximum number of tokens expected in the LLM's response.

        Returns:
            Optional[str]: The generated text response, or None if the query fails.
        """
        model_config = self._get_model_config(model_id)
        if not model_config:
            logger.error(f"Model '{model_id}' not found in configuration's model_list.")
            return None

        query_method = model_config.get('query_method')
        if not query_method:
            logger.error(f"Query method not specified for model '{model_id}' in configuration.")
            return None

        try:
            if query_method == 'anthropic_api':
                return self.query_anthropic(model_id, prompt, max_tokens)
            elif query_method == 'google_ai_studio_api':
                return self.query_google_ai_studio(model_id, prompt, max_tokens)
            elif query_method == 'openai_text_generation_api':
                return self.query_openai(model_id, prompt, max_tokens)
            elif query_method == 'together_ai_inference_api':
                return self.query_together_ai(model_id, prompt, max_tokens)
            else:
                logger.error(f"Unknown query method '{query_method}' for model '{model_id}'.")
                return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while querying model '{model_id}': {e}")
            return None

    def query_anthropic(self, model_id: str, prompt: str, max_tokens: int) -> Optional[str]:
        """
        Queries Anthropic models using their API.

        Args:
            model_id (str): The identifier of the Anthropic model.
            prompt (str): The text prompt.
            max_tokens (int): Max tokens for the response.

        Returns:
            Optional[str]: Generated text or None on failure.
        """
        if not self.anthropic_client:
            logger.error(f"Anthropic client not initialized for model '{model_id}'.")
            return None

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.anthropic_client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=messages
            )
            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                return response.content[0].text
            logger.warning(f"Anthropic response for model '{model_id}' was empty or in unexpected format.")
            return None
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error for model '{model_id}': {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error during Anthropic query for model '{model_id}': {e}")
            return None

    def query_google_ai_studio(self, model_id: str, prompt: str, max_tokens: int) -> Optional[str]:
        """
        Queries Google AI Studio models (e.g., Gemini) using their API.

        Args:
            model_id (str): The identifier of the Google model.
            prompt (str): The text prompt.
            max_tokens (int): Max tokens for the response.

        Returns:
            Optional[str]: Generated text or None on failure.
        """
        if not self.google_api_configured:
            logger.error(f"Google AI Studio API not configured for model '{model_id}'.")
            return None

        try:
            # For Google Generative AI, model_id typically maps to the model name in the library
            model = google.generativeai.GenerativeModel(model_id)
            response = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens}
            )
            # Access response.text directly as per Google's API client design
            return response.text
        except google.api_core.exceptions.GoogleAPIError as e:
            logger.error(f"Google AI Studio API error for model '{model_id}': {e}")
            return None
        except google.generativeai.types.StopCandidateException as e:
            logger.warning(f"Google AI Studio model '{model_id}' stopped generation due to safety filters or other issues: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error during Google AI Studio query for model '{model_id}': {e}")
            return None

    def query_openai(self, model_id: str, prompt: str, max_tokens: int) -> Optional[str]:
        """
        Queries OpenAI models (e.g., GPT) using their API.

        Args:
            model_id (str): The identifier of the OpenAI model.
            prompt (str): The text prompt.
            max_tokens (int): Max tokens for the response.

        Returns:
            Optional[str]: Generated text or None on failure.
        """
        if not self.openai_client:
            logger.error(f"OpenAI client not initialized for model '{model_id}'.")
            return None

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.openai_client.chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=messages
            )
            if response.choices and len(response.choices) > 0 and response.choices[0].message:
                return response.choices[0].message.content
            logger.warning(f"OpenAI response for model '{model_id}' was empty or in unexpected format.")
            return None
        except openai.APIError as e:
            logger.error(f"OpenAI API error for model '{model_id}': {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error during OpenAI query for model '{model_id}': {e}")
            return None

    def query_together_ai(self, model_id: str, prompt: str, max_tokens: int) -> Optional[str]:
        """
        Queries models hosted via the Together AI Inference API.

        Args:
            model_id (str): The identifier of the Together AI model.
            prompt (str): The text prompt.
            max_tokens (int): Max tokens for the response.

        Returns:
            Optional[str]: Generated text or None on failure.
        """
        if not self.together_ai_api_key:
            logger.error(f"Together AI API key not available for model '{model_id}'.")
            return None

        TOGETHER_AI_API_URL = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.together_ai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(TOGETHER_AI_API_URL, headers=headers, json=payload)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            response_json = response.json()

            if response_json.get("choices") and len(response_json["choices"]) > 0 and response_json["choices"][0].get("message"):
                return response_json["choices"][0]["message"].get("content")
            logger.warning(f"Together AI response for model '{model_id}' was empty or in unexpected format: {response_json}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Together AI API request error for model '{model_id}': {e}")
            return None
        except ValueError as e: # JSON decoding error
            logger.error(f"Error decoding JSON response from Together AI for model '{model_id}': {e}. Response: {response.text}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error during Together AI query for model '{model_id}': {e}")
            return None

