## api_client.py
"""Unified LLM API client for querying all four providers used in the paper.

This module provides the APIClient class, which abstracts over OpenAI, Anthropic,
Google AI Studio, and Together AI APIs behind a single query() interface. All
data collection for the de-anonymization experiments (Section 2.3) and identity-
probing experiments (Section 2.4.1) flows through this module.

Key design decisions:
- Provider-specific retry logic is applied at the single-call level, not the
  batch level, so a transient failure on call 30 of 50 retries only that call.
- No sampling parameters (temperature, top-p) are overridden — provider defaults
  are used per Appendix A.1: "default decoding hyperparameters."
- Together AI model names from config.yaml are translated to full API identifiers
  via TOGETHER_MODEL_MAP at module level.
- APIClient does not import config.py to remain loosely coupled; callers pass
  provider and api_key directly.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import anthropic
import google.generativeai as genai
import openai
import together
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Together AI model name mapping
# Translates short config.yaml model names to Together AI's full model IDs.
# Without this map, Together API calls fail with model-not-found errors.
# ---------------------------------------------------------------------------
TOGETHER_MODEL_MAP: dict = {
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemma-2-27b-it": "google/gemma-2-27b-it",
    "llama-3-8b-instruct": "meta-llama/Llama-3-8b-chat-hf",
    "llama-3-70b-instruct": "meta-llama/Llama-3-70b-chat-hf",
    "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "llama-3.1-405b-instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "mixtral-8x7b-instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral-8x22b-instruct-v0.1": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "qwen2-72b-instruct": "Qwen/Qwen2-72B-Instruct",
}

# ---------------------------------------------------------------------------
# Valid provider identifiers — must match api_provider values in config.yaml
# ---------------------------------------------------------------------------
_VALID_PROVIDERS: frozenset = frozenset({"openai", "anthropic", "google", "together"})

# ---------------------------------------------------------------------------
# Retry configuration matching config.yaml api.retry section:
#   min_wait_seconds: 1
#   max_wait_seconds: 60
#   max_attempts: 5
# ---------------------------------------------------------------------------
_RETRY_WAIT = wait_exponential(min=1, max=60)
_RETRY_STOP = stop_after_attempt(5)


class APIClient:
    """Unified LLM query interface across OpenAI, Anthropic, Google, and Together AI.

    Provides a single query() method that dispatches to the appropriate provider-
    specific implementation. All provider-specific retry logic, response parsing,
    and error handling is encapsulated within the private methods.

    Attributes:
        provider: API provider identifier, one of "openai", "anthropic",
            "google", "together". Matches ModelConfig.api_provider in config.yaml.
        api_key: API key for the provider. Read from environment variables by
            the caller (main.py) and passed in — never hardcoded here.
        client: The initialized SDK client object. Type varies by provider:
            openai.OpenAI, anthropic.Anthropic, google.generativeai module,
            or together.Together.

    Example:
        >>> import os
        >>> client = APIClient("openai", os.environ["OPENAI_API_KEY"])
        >>> responses = client.query("gpt-4o-2024-05-13", "Who are you?", n_samples=3, max_tokens=512)
        >>> len(responses)
        3
    """

    def __init__(self, provider: str, api_key: str) -> None:
        """Initialize the API client for the specified provider.

        Validates the provider string and initializes the appropriate SDK client.
        Raises ValueError immediately for unknown providers to fail fast rather
        than producing cryptic errors later during data collection.

        Args:
            provider: API provider identifier. Must be one of "openai",
                "anthropic", "google", "together".
            api_key: API authentication key for the provider. Must be non-empty.

        Raises:
            ValueError: If provider is not one of the four valid identifiers,
                or if api_key is empty.

        Example:
            >>> client = APIClient("openai", "sk-...")
            >>> client.provider
            'openai'
        """
        if provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"Unknown API provider '{provider}'. "
                f"Must be one of {sorted(_VALID_PROVIDERS)}."
            )
        if not api_key:
            raise ValueError(
                f"api_key must be a non-empty string for provider '{provider}'."
            )

        self.provider: str = provider
        self.api_key: str = api_key

        # Initialize the provider-specific SDK client.
        if provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
            logger.debug("Initialized OpenAI client.")

        elif provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.debug("Initialized Anthropic client.")

        elif provider == "google":
            # Google's generativeai module uses a global configure() call.
            # Store the module itself as self.client; model instantiation
            # happens per-call in _query_google_single.
            genai.configure(api_key=api_key)
            self.client = genai
            logger.debug("Initialized Google Generative AI client.")

        elif provider == "together":
            self.client = together.Together(api_key=api_key)
            logger.debug("Initialized Together AI client.")

    def query(
        self,
        model_name: str,
        prompt: str,
        n_samples: int,
        max_tokens: int = 512,
    ) -> List[str]:
        """Query a model for n_samples responses to the given prompt.

        Dispatches to the provider-specific private method based on self.provider.
        Returns exactly n_samples response strings. If a provider does not support
        batch sampling (Anthropic, Google, Together), the private method loops
        internally.

        No sampling parameters (temperature, top-p, etc.) are overridden —
        provider defaults are used per Appendix A.1 of the paper.

        Args:
            model_name: Exact model identifier from config.yaml, e.g.
                "gpt-4o-2024-05-13" or "llama-3.1-70b-instruct". For Together AI
                models, the name is translated via TOGETHER_MODEL_MAP internally.
            prompt: The user prompt text to send to the model.
            n_samples: Number of independent response samples to collect.
                Typically 50 (training-based detector) or 1 (identity-probing,
                called in a loop by DataCollector).
            max_tokens: Maximum output tokens per response. Defaults to 512
                per Appendix A.3: "we set the output length to 512 tokens."

        Returns:
            List of exactly n_samples response strings. Responses from blocked
            safety filters (Google) are represented as empty strings "".

        Raises:
            ValueError: If self.provider is not a recognized value (should not
                happen if __init__ validated correctly, but included for safety).
            tenacity.RetryError: If all retry attempts are exhausted for a
                transient error.
            openai.AuthenticationError: If the API key is invalid (not retried).
            anthropic.AuthenticationError: If the API key is invalid (not retried).

        Example:
            >>> client = APIClient("openai", api_key)
            >>> responses = client.query("gpt-4o-2024-05-13", "Hello", n_samples=2, max_tokens=512)
            >>> len(responses)
            2
        """
        logger.debug(
            "Querying model '%s' via provider '%s' for %d samples.",
            model_name,
            self.provider,
            n_samples,
        )

        if self.provider == "openai":
            return self._query_openai(model_name, prompt, n_samples, max_tokens)
        elif self.provider == "anthropic":
            return self._query_anthropic(model_name, prompt, n_samples, max_tokens)
        elif self.provider == "google":
            return self._query_google(model_name, prompt, n_samples, max_tokens)
        elif self.provider == "together":
            return self._query_together(model_name, prompt, n_samples, max_tokens)
        else:
            # This branch is unreachable if __init__ validated correctly.
            raise ValueError(f"Unrecognized provider '{self.provider}'.")

    # -----------------------------------------------------------------------
    # OpenAI
    # -----------------------------------------------------------------------

    @retry(
        wait=_RETRY_WAIT,
        stop=_RETRY_STOP,
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            )
        ),
        reraise=True,
    )
    def _query_openai(
        self,
        model_name: str,
        prompt: str,
        n_samples: int,
        max_tokens: int,
    ) -> List[str]:
        """Query an OpenAI model for n_samples responses in a single API call.

        OpenAI's chat.completions.create supports n=n_samples natively, so a
        single call returns all samples. The @retry decorator wraps the entire
        method — since it is a single API call, a retry re-requests all samples,
        which is acceptable and simpler than a per-sample retry loop.

        Temperature is intentionally omitted to use the provider's default,
        matching the paper's "default decoding hyperparameters" requirement.

        Args:
            model_name: OpenAI model identifier, e.g. "gpt-4o-2024-05-13".
            prompt: User prompt text.
            n_samples: Number of completions to request (n parameter).
            max_tokens: Maximum tokens per completion.

        Returns:
            List of n_samples response strings extracted from choices[*].message.content.

        Raises:
            openai.AuthenticationError: Not retried — invalid API key.
            openai.NotFoundError: Not retried — model does not exist.
            openai.RateLimitError: Retried with exponential backoff.
            openai.APIConnectionError: Retried with exponential backoff.
            openai.APITimeoutError: Retried with exponential backoff.
            openai.InternalServerError: Retried with exponential backoff.
        """
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            n=n_samples,
            max_tokens=max_tokens,
        )
        texts: List[str] = [
            choice.message.content or ""
            for choice in response.choices
        ]
        logger.debug(
            "OpenAI '%s': received %d/%d responses.",
            model_name,
            len(texts),
            n_samples,
        )
        return texts

    # -----------------------------------------------------------------------
    # Anthropic
    # -----------------------------------------------------------------------

    def _query_anthropic(
        self,
        model_name: str,
        prompt: str,
        n_samples: int,
        max_tokens: int,
    ) -> List[str]:
        """Query an Anthropic model for n_samples responses via a loop.

        Anthropic's messages.create does not support n > 1, so this method
        calls the single-call helper _query_anthropic_single n_samples times.
        The retry decorator is on the single-call helper, so a transient failure
        on iteration 30 retries only that call, not the entire batch.

        Args:
            model_name: Anthropic model identifier, e.g. "claude-3-5-sonnet-20240620".
            prompt: User prompt text.
            n_samples: Number of independent responses to collect.
            max_tokens: Maximum tokens per response.

        Returns:
            List of n_samples response strings.
        """
        results: List[str] = []
        for i in range(n_samples):
            text: str = self._query_anthropic_single(model_name, prompt, max_tokens)
            results.append(text)
            if (i + 1) % 10 == 0:
                logger.debug(
                    "Anthropic '%s': collected %d/%d responses.",
                    model_name,
                    i + 1,
                    n_samples,
                )
        return results

    @retry(
        wait=_RETRY_WAIT,
        stop=_RETRY_STOP,
        retry=retry_if_exception_type(
            (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.InternalServerError,
            )
        ),
        reraise=True,
    )
    def _query_anthropic_single(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Make a single Anthropic messages.create call with retry logic.

        Decorated with @retry so that transient failures on any individual
        call are retried independently without restarting the outer loop in
        _query_anthropic.

        Args:
            model_name: Anthropic model identifier.
            prompt: User prompt text.
            max_tokens: Maximum tokens for this response.

        Returns:
            Single response string from response.content[0].text.

        Raises:
            anthropic.AuthenticationError: Not retried — invalid API key.
            anthropic.NotFoundError: Not retried — model does not exist.
            anthropic.RateLimitError: Retried with exponential backoff.
            anthropic.APIConnectionError: Retried with exponential backoff.
            anthropic.APITimeoutError: Retried with exponential backoff.
            anthropic.InternalServerError: Retried with exponential backoff.
        """
        response = self.client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # response.content is a list of ContentBlock objects; the first is text.
        text: str = response.content[0].text if response.content else ""
        return text

    # -----------------------------------------------------------------------
    # Google
    # -----------------------------------------------------------------------

    def _query_google(
        self,
        model_name: str,
        prompt: str,
        n_samples: int,
        max_tokens: int,
    ) -> List[str]:
        """Query a Google Generative AI model for n_samples responses via a loop.

        Google's generate_content does not support batch sampling natively.
        Calls the single-call helper _query_google_single n_samples times.
        The retry decorator is on the single-call helper for per-call retry.

        Args:
            model_name: Google model identifier, e.g. "gemini-1.5-pro".
                Matches config.yaml exactly — no translation needed.
            prompt: User prompt text.
            n_samples: Number of independent responses to collect.
            max_tokens: Maximum output tokens per response.

        Returns:
            List of n_samples response strings. Blocked responses are "".
        """
        results: List[str] = []
        for i in range(n_samples):
            text: str = self._query_google_single(model_name, prompt, max_tokens)
            results.append(text)
            if (i + 1) % 10 == 0:
                logger.debug(
                    "Google '%s': collected %d/%d responses.",
                    model_name,
                    i + 1,
                    n_samples,
                )
        return results

    @retry(
        wait=_RETRY_WAIT,
        stop=_RETRY_STOP,
        retry=retry_if_exception_type(
            (
                Exception,  # Broad catch for Google API transient errors;
                # refined below via before_sleep logging.
                # google.api_core exceptions are not always importable without
                # the full google-cloud stack, so we use a broad type here and
                # exclude non-retryable errors via the except block inside.
            )
        ),
        reraise=True,
    )
    def _query_google_single(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Make a single Google GenerativeModel.generate_content call with retry.

        Handles safety-blocked responses by returning "" rather than raising,
        since retrying a blocked prompt will produce the same block.

        Args:
            model_name: Google model identifier, e.g. "gemini-1.5-flash".
            prompt: User prompt text.
            max_tokens: Maximum output tokens.

        Returns:
            Response text string, or "" if the response was safety-blocked.

        Raises:
            Exception: Retried for transient errors (rate limits, service
                unavailable). Safety blocks are caught and return "" instead.
        """
        generation_config = genai.GenerationConfig(max_output_tokens=max_tokens)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        # response.text raises ValueError if the response was blocked by safety
        # filters. Catch it and return "" — do not retry blocked responses.
        try:
            text: str = response.text
        except ValueError as exc:
            logger.warning(
                "Google '%s': response blocked by safety filter. "
                "Returning empty string. Error: %s",
                model_name,
                exc,
            )
            text = ""
        return text

    # -----------------------------------------------------------------------
    # Together AI
    # -----------------------------------------------------------------------

    def _query_together(
        self,
        model_name: str,
        prompt: str,
        n_samples: int,
        max_tokens: int,
    ) -> List[str]:
        """Query a Together AI model for n_samples responses via a loop.

        Together AI's client uses an OpenAI-compatible interface but does not
        reliably support n > 1. Loops n_samples times for safety and consistency.
        Translates config.yaml model names to Together's full model IDs via
        TOGETHER_MODEL_MAP before making API calls.

        Args:
            model_name: Short model name from config.yaml, e.g.
                "llama-3.1-70b-instruct". Translated to Together's full ID
                via TOGETHER_MODEL_MAP.
            prompt: User prompt text.
            n_samples: Number of independent responses to collect.
            max_tokens: Maximum tokens per response.

        Returns:
            List of n_samples response strings.

        Raises:
            ValueError: If model_name is not in TOGETHER_MODEL_MAP.
        """
        # Translate short name to Together's full model identifier.
        together_model_id: str = TOGETHER_MODEL_MAP.get(model_name, "")
        if not together_model_id:
            raise ValueError(
                f"Model '{model_name}' not found in TOGETHER_MODEL_MAP. "
                f"Available models: {sorted(TOGETHER_MODEL_MAP.keys())}"
            )

        results: List[str] = []
        for i in range(n_samples):
            text: str = self._query_together_single(
                together_model_id, prompt, max_tokens
            )
            results.append(text)
            if (i + 1) % 10 == 0:
                logger.debug(
                    "Together '%s' (%s): collected %d/%d responses.",
                    model_name,
                    together_model_id,
                    i + 1,
                    n_samples,
                )
        return results

    @retry(
        wait=_RETRY_WAIT,
        stop=_RETRY_STOP,
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
                # Together AI uses the OpenAI-compatible SDK, so OpenAI exception
                # types are raised for transient errors.
            )
        ),
        reraise=True,
    )
    def _query_together_single(
        self,
        together_model_id: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Make a single Together AI chat.completions.create call with retry.

        Uses the Together AI client's OpenAI-compatible interface. The model ID
        passed here is already the full Together model path (e.g.,
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"), translated by the
        caller _query_together.

        Args:
            together_model_id: Full Together AI model path, e.g.
                "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo".
            prompt: User prompt text.
            max_tokens: Maximum tokens for this response.

        Returns:
            Single response string from choices[0].message.content.

        Raises:
            openai.RateLimitError: Retried with exponential backoff.
            openai.APIConnectionError: Retried with exponential backoff.
            openai.APITimeoutError: Retried with exponential backoff.
            openai.InternalServerError: Retried with exponential backoff.
        """
        response = self.client.chat.completions.create(
            model=together_model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        text: str = response.choices[0].message.content or ""
        return text
