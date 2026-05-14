## utils/cache.py
"""Disk-based caching utility to avoid redundant API calls during data collection.

This module provides the Cache class, which wraps the diskcache library to
offer a simple key-value store backed by SQLite on disk. It is a foundational
utility with zero internal project dependencies.

The cache is used by DataCollector to persist API responses across runs,
enabling resumable data collection and cost control for the ~$440-$600
experiment budget described in Appendix A.3 of the paper.

Usage:
    from utils.cache import Cache

    cache = Cache(cache_dir="cache")
    key = cache.make_key("gpt-4o-2024-05-13", "Who are you?", "identity_probing")
    if not cache.exists(key):
        responses = api_client.query(...)
        cache.set(key, responses)
    else:
        responses = cache.get(key)
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import diskcache


class Cache:
    """Disk-backed key-value cache for storing LLM API responses.

    Uses the diskcache library (SQLite + WAL mode) for thread-safe, process-safe
    persistent storage. Keys are SHA256 hashes of (model, category, prompt)
    triples to handle arbitrarily long prompt strings without filesystem issues.

    Attributes:
        cache_dir: Path to the directory where the diskcache SQLite database
            and associated files are stored.

    Example:
        >>> cache = Cache(cache_dir="cache")
        >>> key = cache.make_key("gpt-4o-2024-05-13", "Who are you?", "identity")
        >>> cache.set(key, ["I am GPT-4o.", "I'm an AI assistant by OpenAI."])
        >>> cache.exists(key)
        True
        >>> cache.get(key)
        ["I am GPT-4o.", "I'm an AI assistant by OpenAI."]
    """

    def __init__(self, cache_dir: str = "cache") -> None:
        """Initialize the disk cache at the given directory.

        The diskcache.Cache constructor creates the directory and its SQLite
        database if they do not already exist, so no manual os.makedirs call
        is needed. The cache has no size limit, which is appropriate given the
        manageable total data volume (~1.76M string responses).

        Args:
            cache_dir: Path to the cache directory. Defaults to "cache" to
                match the cache_dir value in config.yaml.
        """
        self.cache_dir: str = cache_dir
        # diskcache.Cache is thread-safe and process-safe via SQLite WAL mode.
        # This matters when joblib.Parallel is active in TrainingBasedDetector.
        self._cache: diskcache.Cache = diskcache.Cache(cache_dir)

    def make_key(self, model: str, prompt: str, category: str) -> str:
        """Produce a deterministic, fixed-length cache key for a (model, prompt, category) triple.

        Constructs the canonical string '{model}||{category}||{prompt}' per the
        Shared Knowledge spec, then returns its SHA256 hex digest. The '||'
        separator prevents collisions between inputs that might otherwise
        concatenate identically. SHA256 produces a 64-character hex string,
        avoiding filesystem path-length issues with long prompts (e.g., the
        OFAC sanctions prompt from Appendix A.2 is several hundred characters).

        Args:
            model: Exact model name string, e.g. "claude-3-5-sonnet-20240620".
            prompt: The prompt text sent to the model. May be arbitrarily long.
            category: Prompt category name, e.g. "english", "math",
                "identity_probing". Used to namespace identity-probing queries
                separately from training-based detector queries.

        Returns:
            A 64-character lowercase hex string (SHA256 digest) that uniquely
            identifies the (model, prompt, category) combination with
            overwhelming probability.

        Example:
            >>> cache = Cache()
            >>> key = cache.make_key("gpt-4o-2024-05-13", "Who are you?", "identity_probing")
            >>> len(key)
            64
            >>> key == cache.make_key("gpt-4o-2024-05-13", "Who are you?", "identity_probing")
            True
        """
        # Order matches Shared Knowledge spec: '{model_name}||{category}||{prompt}'
        combined: str = f"{model}||{category}||{prompt}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def exists(self, key: str) -> bool:
        """Check whether a key is present in the cache.

        Uses diskcache's __contains__ for an O(1) SQLite index lookup.
        Called by DataCollector before every API call to decide whether to
        skip the request and use the cached value instead.

        Args:
            key: A cache key, typically produced by make_key().

        Returns:
            True if the key exists in the cache, False otherwise.

        Example:
            >>> cache = Cache()
            >>> key = cache.make_key("gpt-4o", "Hello", "english")
            >>> cache.exists(key)
            False
            >>> cache.set(key, ["Hello! How can I help?"])
            >>> cache.exists(key)
            True
        """
        return key in self._cache

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value by key.

        Returns None on a cache miss rather than raising an exception, allowing
        callers to use either the exists()-then-get() pattern or the
        get()-and-check-None pattern interchangeably.

        The stored values are typically List[str] (lists of API response
        strings), but the method accepts Any to keep the cache generic and
        reusable for other serializable Python objects.

        Args:
            key: A cache key, typically produced by make_key().

        Returns:
            The cached value (deserialized from pickle by diskcache) if the
            key exists, or None if the key is not in the cache.

        Example:
            >>> cache = Cache()
            >>> key = cache.make_key("gpt-4o", "Hello", "english")
            >>> cache.get(key) is None
            True
            >>> cache.set(key, ["Hi there!"])
            >>> cache.get(key)
            ['Hi there!']
        """
        return self._cache.get(key, default=None)

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache under the given key.

        Serializes value using pickle and writes atomically to disk via
        diskcache. Calling set() with an existing key silently overwrites the
        previous value. No TTL is applied — cached responses are permanent,
        which is correct because model API responses for a given prompt are
        stable enough to reuse across experiment runs.

        Args:
            key: A cache key, typically produced by make_key().
            value: The value to cache. Typically List[str] (API responses).
                Must be pickle-serializable.

        Returns:
            None.

        Example:
            >>> cache = Cache()
            >>> key = cache.make_key("gpt-4o", "Hello", "english")
            >>> cache.set(key, ["Hello! How can I help you today?"])
            >>> cache.get(key)
            ['Hello! How can I help you today?']
        """
        self._cache.set(key, value)
