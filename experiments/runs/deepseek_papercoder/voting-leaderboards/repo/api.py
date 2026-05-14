"""
api.py – Unified interface for querying multiple LLM providers (OpenAI, Anthropic,
Google Gemini, Together AI).

This module provides the :class:`ModelAPI` class, which abstracts away differences
in SDK syntax, authentication, and response extraction.  It uses the global
configuration (from :mod:`config`) to obtain API credentials and rate‑limiting
parameters, and implements automatic retries with exponential backoff.

Usage:
    from config import Config
    from api import ModelAPI

    cfg = Config("config.yaml")
    api = ModelAPI(cfg)
    answer = api.query("gpt-4o-mini-2024-07-18", "Hello!")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import anthropic
import google.generativeai as genai
import openai
import together

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

from config import Config

logger = logging.getLogger(__name__)


class ModelAPI:
    """
    Abstraction layer over OpenAI, Anthropic, Google Gemini, and Together AI LLMs.

    Instances are bound to a :class:`Config` that provides API key environment
    variable names, rate‑limiting parameters, and default generation settings.

    Attributes:
        config (Config): The application configuration.
        provider_map (Dict[str, str]): Mapping from full model name to provider
            identifier (``"openai"``, ``"anthropic"``, ``"google"``, ``"together"``).
    """

    # ------------------------------------------------------------------
    # Mapping from every model in the paper (Appendix A.1, Table 6) to its
    # provider.  This ensures deterministic routing without relying on prefix
    # heuristics that could silently produce wrong results.
    # ------------------------------------------------------------------
    _MODEL_PROVIDER_MAP: Dict[str, str] = {
        # Anthropic
        "claude-3-5-sonnet-20240620": "anthropic",
        "claude-3-haiku-20240307": "anthropic",
        # Google (accessed via Google AI Studio)
        "gemini-1.5-pro": "google",
        "gemini-1.5-flash": "google",
        # Together AI (serves open‑weight models)
        "gemma-2-2b-it": "together",
        "gemma-2-9b-it": "together",
        "gemma-2-27b-it": "together",
        "llama-3-8b-instruct": "together",
        "llama-3-70b-instruct": "together",
        "llama-3.1-8b-instruct": "together",
        "llama-3.1-70b-instruct": "together",
        "llama-3.1-405b-instruct": "together",
        "mixtral-8x7b-instruct-v0.1": "together",
        "mixtral-8x22b-instruct-v0.1": "together",
        "qwen2-72b-instruct": "together",
        # OpenAI
        "gpt-3.5-turbo": "openai",
        "gpt-4-0125-preview": "openai",
        "gpt-4-1106-preview": "openai",
        "gpt-4-turbo-2024-04-09": "openai",
        "gpt-4o-2024-05-13": "openai",
        "gpt-4o-2024-08-06": "openai",
        "gpt-4o-mini-2024-07-18": "openai",
        "chatgpt-4o-latest": "openai",  # often used as alias
    }

    # ------------------------------------------------------------------
    def __init__(self, config: Config) -> None:
        """
        Initialise the API wrapper.

        Client objects are created lazily on the first call to a provider,
        so missing credentials for unused providers do not prevent
        instantiation.

        Args:
            config: The application configuration.

        Raises:
            TypeError: If *config* is not an instance of :class:`Config`.
        """
        if not isinstance(config, Config):
            raise TypeError("config must be an instance of Config")

        self.config: Config = config
        # Lazy client storage: provider name -> SDK client object
        self._clients: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def query(
        self,
        model: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a single prompt to the specified model and return the generated text.

        The function automatically determines the correct provider, applies
        default generation parameters from the configuration when arguments are
        ``None``, and retries on transient errors with exponential backoff.

        Args:
            model: Full model identifier (e.g. ``"gpt-4o-mini-2024-07-18"``).
            prompt: The input text to be completed.
            max_tokens: Maximum number of tokens to generate.  If ``None``,
                the value ``config.max_output_tokens`` is used.
            temperature: Sampling temperature.  If ``None``, ``config.temperature``
                is used.  If that is also ``None``, the parameter is omitted so
                the model's default is applied.

        Returns:
            The generated text (stripped of leading/trailing whitespace).

        Raises:
            ValueError: If *model* is not recognised.
            RuntimeError: If all retry attempts are exhausted and the request
                still fails.
        """
        provider = self._get_provider(model)

        # Resolve generation parameters
        mt = max_tokens if max_tokens is not None else self.config.max_output_tokens
        temp = temperature if temperature is not None else self.config.temperature

        # Retry loop
        max_retries = self.config.api.get("max_retries", 3)
        base_wait = self.config.api.get("request_interval", 0.5)

        last_exception: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                client = self._get_client(provider)
                response_text = self._call_provider(
                    provider, client, model, prompt, mt, temp
                )
                logger.debug(
                    "Query succeeded: model=%s, tokens=%d, attempt=%d",
                    model, len(response_text.split()), attempt,
                )
                return response_text.strip()

            except Exception as exc:  # noqa: BLE001 – we want to catch any API error
                last_exception = exc
                wait_time = base_wait * (2 ** attempt)
                logger.warning(
                    "Query attempt %d/%d failed (model=%s). Retrying in %.1fs ... Error: %s",
                    attempt + 1, max_retries + 1, model, wait_time, exc,
                )
                if attempt < max_retries:
                    time.sleep(wait_time)
                else:
                    break

        # Exhausted retries
        raise RuntimeError(
            f"Failed to query model '{model}' after {max_retries + 1} attempts. "
            f"Last error: {last_exception}"
        ) from last_exception

    def batch_query(
        self,
        model: str,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        Query the model for a list of prompts sequentially.

        A configurable inter‑request delay (``config.api.request_interval``) is
        inserted between successive calls to avoid rate‑limiting.  If the
        ``tqdm`` library is available, a progress bar is displayed.

        Args:
            model: Same as for :meth:`query`.
            prompts: A list of input strings.
            max_tokens: Same as for :meth:`query`.
            temperature: Same as for :meth:`query`.

        Returns:
            A list of generated responses, in the same order as *prompts*.
            If any single query fails after retries, the exception propagates
            immediately.
        """
        results: List[str] = []
        # Determine generation defaults once
        mt = max_tokens if max_tokens is not None else self.config.max_output_tokens
        temp = temperature if temperature is not None else self.config.temperature

        iterator = tqdm(prompts, desc=f"Querying {model}") if tqdm else prompts
        for prompt in iterator:
            results.append(self.query(model, prompt, max_tokens=mt, temperature=temp))
            # Enforce inter‑request sleep (config.api.request_interval)
            time.sleep(self.config.api.get("request_interval", 0.5))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_provider(self, model: str) -> str:
        """Return the provider identifier for a given model name."""
        try:
            return self._MODEL_PROVIDER_MAP[model]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model: '{model}'. Add it to the MODEL_PROVIDER_MAP."
            ) from exc

    def _get_client(self, provider: str) -> Any:
        """Obtain (or create and cache) the SDK client for *provider*."""
        if provider not in self._clients:
            if provider == "openai":
                api_key = os.environ.get(self.config.api["openai_key_env"])
                if not api_key:
                    raise ValueError(
                        f"OpenAI API key not found. Set environment variable "
                        f"{self.config.api['openai_key_env']}"
                    )
                self._clients["openai"] = openai.OpenAI(api_key=api_key)

            elif provider == "anthropic":
                api_key = os.environ.get(self.config.api["anthropic_key_env"])
                if not api_key:
                    raise ValueError(
                        f"Anthropic API key not found. Set environment variable "
                        f"{self.config.api['anthropic_key_env']}"
                    )
                self._clients["anthropic"] = anthropic.Anthropic(api_key=api_key)

            elif provider == "google":
                api_key = os.environ.get(self.config.api["google_key_env"])
                if not api_key:
                    raise ValueError(
                        f"Google API key not found. Set environment variable "
                        f"{self.config.api['google_key_env']}"
                    )
                # The Google client is stateful, configure once.
                genai.configure(api_key=api_key)
                self._clients["google"] = genai

            elif provider == "together":
                api_key = os.environ.get(self.config.api["together_key_env"])
                if not api_key:
                    raise ValueError(
                        f"Together API key not found. Set environment variable "
                        f"{self.config.api['together_key_env']}"
                    )
                self._clients["together"] = together.Together(api_key=api_key)

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        return self._clients[provider]

    def _call_provider(
        self,
        provider: str,
        client: Any,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: Optional[float],
    ) -> str:
        """Execute the actual API call and extract the generated text."""
        if provider == "openai":
            # Prepare kwargs, omitting temperature if None
            openai_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                openai_kwargs["temperature"] = temperature

            response = client.chat.completions.create(**openai_kwargs)
            content = response.choices[0].message.content
            # content can be None in edge cases; treat as empty string
            return content if content is not None else ""

        elif provider == "anthropic":
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = client.messages.create(**kwargs)
            # Anthropic returns a list of ContentBlock; take the first text block
            for block in response.content:
                if block.type == "text":
                    return block.text
            # Fallback if no text block found (unlikely)
            return ""

        elif provider == "google":
            # Google SDK requires model name as 'models/<name>'
            google_model = model if model.startswith("models/") else f"models/{model}"
            # Generation config as dict
            gen_config = {}
            if temperature is not None:
                gen_config["temperature"] = temperature
            gen_config["max_output_tokens"] = max_tokens

            model_obj = client.GenerativeModel(google_model)
            response = model_obj.generate_content(
                prompt,
                generation_config=gen_config,
            )
            # response.text is a regular string (or raises ValueError)
            return response.text

        elif provider == "together":
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return content if content is not None else ""

        else:
            raise ValueError(f"Internal error: unknown provider '{provider}'")

