"""Application configuration loading utilities for BrandLens."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping, MutableMapping, Optional

from dotenv import load_dotenv

from .core import (
    APIConfig,
    AppConfig,
    CacheConfig,
    CompressionMethod,
    ConfigurationError,
)

BOOLEAN_TRUE = {"1", "true", "t", "yes", "y", "on"}
BOOLEAN_FALSE = {"0", "false", "f", "no", "n", "off"}


def _load_env_file(env_file: Optional[str]) -> None:
    """Load variables from the provided `.env` file if it exists."""

    candidate = Path(env_file or ".env")
    if candidate.exists():
        # Do not override explicit environment variables.
        load_dotenv(candidate, override=False)


def _require(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ConfigurationError(
            f"Missing required configuration value: {key}",
            config_key=key,
        )
    return value


def _parse_int(env: Mapping[str, str], key: str) -> Optional[int]:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Configuration value for {key} must be an integer",
            config_key=key,
            expected_type="int",
            provided_value=raw,
        ) from exc


def _parse_float(env: Mapping[str, str], key: str) -> Optional[float]:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Configuration value for {key} must be a float",
            config_key=key,
            expected_type="float",
            provided_value=raw,
        ) from exc


def _parse_bool(env: Mapping[str, str], key: str) -> Optional[bool]:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return None
    value = raw.strip().lower()
    if value in BOOLEAN_TRUE:
        return True
    if value in BOOLEAN_FALSE:
        return False
    raise ConfigurationError(
        f"Configuration value for {key} must be a boolean",
        config_key=key,
        expected_type="bool",
        provided_value=raw,
    )


def _build_api_config(env: Mapping[str, str]) -> APIConfig:
    config_kwargs: MutableMapping[str, object] = {
        "gemini_api_key": _require(env, "GEMINI_API_KEY"),
        "tavily_api_key": _require(env, "TAVILY_API_KEY"),
        "gemini_model": _require(env, "GEMINI_MODEL"),
    }

    temperature = _parse_float(env, "GEMINI_TEMPERATURE")
    if temperature is not None:
        config_kwargs["gemini_temperature"] = temperature

    max_tokens = _parse_int(env, "GEMINI_MAX_TOKENS")
    if max_tokens is not None:
        config_kwargs["gemini_max_tokens"] = max_tokens

    search_depth = env.get("TAVILY_SEARCH_DEPTH")
    if search_depth:
        config_kwargs["tavily_search_depth"] = search_depth.strip().lower()

    include_raw = _parse_bool(env, "TAVILY_INCLUDE_RAW_CONTENT")
    if include_raw is not None:
        config_kwargs["tavily_include_raw_content"] = include_raw

    content_mode = env.get("TAVILY_CONTENT_MODE")
    if content_mode:
        config_kwargs["tavily_content_mode"] = content_mode.strip().lower()

    max_searches = _parse_int(env, "MAX_SEARCHES_PER_QUERY")
    if max_searches is not None:
        config_kwargs["max_searches_per_query"] = max_searches

    max_sources = _parse_int(env, "MAX_SOURCES_PER_SEARCH")
    if max_sources is None:
        max_sources = _parse_int(env, "TAVILY_MAX_RESULTS")
    if max_sources is not None:
        config_kwargs["max_sources_per_search"] = max_sources

    return APIConfig(**config_kwargs)


def _build_cache_config(env: Mapping[str, str]) -> CacheConfig:
    config_kwargs: MutableMapping[str, object] = {}

    cache_dir = env.get("CACHE_DIR")
    if cache_dir:
        config_kwargs["cache_dir"] = cache_dir.strip()

    ttl = _parse_int(env, "CACHE_TTL")
    if ttl is not None:
        config_kwargs["default_ttl_seconds"] = ttl

    max_mb = _parse_int(env, "CACHE_MAX_SIZE_MB")
    if max_mb is not None:
        config_kwargs["max_cache_size_mb"] = max_mb

    cleanup = _parse_int(env, "CACHE_CLEANUP_INTERVAL")
    if cleanup is not None:
        config_kwargs["cleanup_interval_seconds"] = cleanup

    return CacheConfig(**config_kwargs)


def _build_app_kwargs(env: Mapping[str, str]) -> MutableMapping[str, object]:
    kwargs: MutableMapping[str, object] = {}

    compression_tokens = _parse_int(env, "COMPRESSION_TARGET_TOKENS")
    if compression_tokens is not None:
        kwargs["compression_target_tokens"] = compression_tokens

    compression_method = env.get("COMPRESSION_METHOD")
    if compression_method:
        try:
            kwargs["compression_method"] = CompressionMethod(compression_method.strip().lower())
        except ValueError as exc:
            raise ConfigurationError(
                "Invalid compression method configured",
                config_key="COMPRESSION_METHOD",
                expected_type=f"one of {[m.value for m in CompressionMethod]}",
                provided_value=compression_method,
            ) from exc

    compression_target = _parse_float(env, "TOKEN_COMPRESSION_TARGET")
    if compression_target is not None:
        kwargs["target_compression_ratio"] = compression_target

    output_format = env.get("OUTPUT_FORMAT")
    if output_format:
        kwargs["output_format"] = output_format.strip().lower()

    include_debug = _parse_bool(env, "INCLUDE_DEBUG_INFO")
    if include_debug is None:
        include_debug = _parse_bool(env, "DEBUG")
    if include_debug is not None:
        kwargs["include_debug_info"] = include_debug

    log_level = env.get("LOG_LEVEL")
    if log_level:
        kwargs["log_level"] = log_level.strip().upper()

    log_file = env.get("LOG_FILE")
    if log_file:
        kwargs["log_file"] = log_file.strip()

    return kwargs


def validate_app_config(config: AppConfig) -> None:
    """Run additional validation that Pydantic does not cover."""

    placeholder_values = {
        "your_gemini_api_key_here",
        "your_tavily_api_key_here",
        "",
    }

    if config.api.gemini_api_key in placeholder_values:
        raise ConfigurationError(
            "Gemini API key is using a placeholder value",
            config_key="GEMINI_API_KEY",
            provided_value=config.api.gemini_api_key,
        )
    if config.api.tavily_api_key in placeholder_values:
        raise ConfigurationError(
            "Tavily API key is using a placeholder value",
            config_key="TAVILY_API_KEY",
            provided_value=config.api.tavily_api_key,
        )


def load_app_config(
    env: Optional[Mapping[str, str]] = None,
    *,
    env_file: Optional[str] = None,
) -> AppConfig:
    """Load an :class:`AppConfig` from environment variables."""

    if env is None:
        _load_env_file(env_file)
        env_mapping = dict(os.environ)
    else:
        env_mapping = dict(env)

    api_config = _build_api_config(env_mapping)
    cache_config = _build_cache_config(env_mapping)
    app_kwargs = _build_app_kwargs(env_mapping)

    config = AppConfig(api=api_config, cache=cache_config, **app_kwargs)
    validate_app_config(config)
    return config


@lru_cache()
def get_app_config(env_file: Optional[str] = None) -> AppConfig:
    """Cached configuration loader for use within the application."""

    return load_app_config(env_file=env_file)


def reload_app_config(env_file: Optional[str] = None) -> AppConfig:
    """Clear the cached configuration and reload from the environment."""

    get_app_config.cache_clear()
    return get_app_config(env_file=env_file)


__all__ = [
    "get_app_config",
    "load_app_config",
    "reload_app_config",
    "validate_app_config",
]
