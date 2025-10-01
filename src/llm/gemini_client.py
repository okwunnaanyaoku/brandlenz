"""Gemini API client utilities for BrandLens."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Sequence

import google.generativeai as genai
from tenacity import Retrying, RetryError, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..core.exceptions import GeminiAPIError
from ..core.models import APIConfig, LLMResponse, ModelName

LOGGER = logging.getLogger(__name__)


@dataclass
class GeminiClientSettings:
    """Runtime configuration for the Gemini client."""

    api_key: str
    model: str
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 32
    max_output_tokens: int = 2048
    candidate_count: int = 1
    request_timeout: float = 30.0
    max_attempts: int = 3

    @classmethod
    def from_api_config(cls, config: APIConfig) -> "GeminiClientSettings":
        return cls(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
            temperature=config.gemini_temperature,
            max_output_tokens=config.gemini_max_tokens,
        )


class GeminiStreamingResponse:
    """Iterator wrapper for streaming Gemini responses."""

    def __init__(
        self,
        raw_stream: Iterable[Any],
        *,
        finalizer: Callable[[List[str], Optional[Any], float], LLMResponse],
        started_at: float,
    ) -> None:
        self._raw_stream = raw_stream
        self._finalizer = finalizer
        self._started_at = started_at
        self._chunks: List[str] = []
        self._resolved_response: Optional[LLMResponse] = None
        self._resolved = False

    def __iter__(self) -> Generator[str, None, None]:
        for chunk in self._raw_stream:
            text = getattr(chunk, "text", None)
            if text:
                self._chunks.append(text)
                yield text
        self._resolve_if_needed(final_chunk=chunk)

    def _resolve_if_needed(self, final_chunk: Optional[Any] = None) -> None:
        if self._resolved:
            return
        elapsed_ms = (time.perf_counter() - self._started_at) * 1000
        self._resolved_response = self._finalizer(self._chunks, final_chunk, elapsed_ms)
        self._resolved = True

    def resolve(self) -> LLMResponse:
        if not self._resolved:
            # Drain the iterator to gather remaining chunks
            for _ in self:
                pass
        assert self._resolved_response is not None
        return self._resolved_response


class GeminiClient:
    """Thin wrapper around the Google Generative AI client with retries and metrics."""

    def __init__(self, settings: GeminiClientSettings) -> None:
        self._settings = settings
        genai.configure(api_key=settings.api_key)

        self._model = genai.GenerativeModel(
            model_name=settings.model
        )

    @property
    def settings(self) -> GeminiClientSettings:
        return self._settings

    def generate(
        self,
        prompt: Sequence[str] | str,
        *,
        system_instruction: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> LLMResponse | GeminiStreamingResponse:
        contents = self._normalise_prompt(prompt)
        generation_config = self._build_generation_config(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences,
        )

        def _call_model() -> Any:
            options: Dict[str, Any] = {
                "contents": contents,
                "generation_config": generation_config,
                "stream": stream,
            }
            # Note: system instructions are handled during model initialization in newer versions
            return self._model.generate_content(**options)

        start_time = time.perf_counter()
        try:
            response = self._retry_call(_call_model)
        except GeminiAPIError:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            raise self._wrap_error(exc, prompt=contents) from exc

        if stream:
            return GeminiStreamingResponse(
                response,
                finalizer=lambda chunks, final_chunk, elapsed_ms: self._build_llm_response(
                    final_chunk or response,
                    "".join(chunks),
                    elapsed_ms,
                ),
                started_at=start_time,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return self._build_llm_response(response, response_text=None, elapsed_ms=elapsed_ms)

    def _retry_call(self, func: Callable[[], Any]) -> Any:
        retryer = Retrying(
            stop=stop_after_attempt(self._settings.max_attempts),
            wait=wait_exponential(min=1, max=5),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        try:
            return retryer(func)
        except RetryError as exc:
            raise exc.last_attempt.exception() from exc
    def _build_generation_config(
        self,
        *,
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        stop_sequences: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        config = {
            "temperature": temperature if temperature is not None else self._settings.temperature,
            "top_p": self._settings.top_p,
            "top_k": self._settings.top_k,
            "max_output_tokens": max_output_tokens if max_output_tokens is not None else self._settings.max_output_tokens,
            "candidate_count": self._settings.candidate_count,
        }
        if stop_sequences:
            config["stop_sequences"] = list(stop_sequences)
        return config

    def _build_llm_response(
        self,
        raw_response: Any,
        response_text: Optional[str],
        elapsed_ms: float,
    ) -> LLMResponse:
        text = response_text
        if not text:
            text = getattr(raw_response, "text", "")
        usage = getattr(raw_response, "usage_metadata", None)
        if usage:
            prompt_tokens = int(getattr(usage, "prompt_token_count", 0))
            completion_tokens = int(getattr(usage, "candidates_token_count", 0))
            total_tokens = int(getattr(usage, "total_token_count", prompt_tokens + completion_tokens))
            # Calculate cached tokens from the difference
            cached_prompt_tokens = max(0, total_tokens - prompt_tokens - completion_tokens)
        else:
            prompt_tokens = completion_tokens = total_tokens = cached_prompt_tokens = 0

        try:
            response = LLMResponse(
                markdown_content=text or "",
                model=self._settings.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
                generation_time_ms=elapsed_ms,
                temperature=self._settings.temperature,
            )
        except ValueError as exc:
            raise self._wrap_error(exc, prompt=text) from exc

        LOGGER.info(
            "Gemini response generated",
            extra={
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "cached_prompt_tokens": response.cached_prompt_tokens,
                "total_tokens": response.total_tokens,
                "cost_usd": response.cost_usd,
            },
        )

        # Explain token counting if there are cached tokens
        if response.cached_prompt_tokens > 0:
            LOGGER.info(
                "Prompt caching active: cached tokens billed at 10% rate (90% savings)",
                extra={
                    "cached_tokens": response.cached_prompt_tokens,
                    "savings_usd": round(response.cached_prompt_tokens / 1000 * 0.00001875 * 0.9, 6)
                }
            )
        
        return response

    def _wrap_error(self, error: Exception, *, prompt: Sequence[str] | str) -> GeminiAPIError:
        status_code = getattr(error, "status_code", None)
        response_data = getattr(error, "response", None)
        if isinstance(response_data, dict):
            payload = response_data
        else:
            payload = {"error": str(response_data)} if response_data else None
        return GeminiAPIError(
            message=str(error),
            status_code=status_code,
            response_data=payload if isinstance(payload, dict) else None,
            model=self._settings.model,
            prompt_tokens=len(" ".join(self._normalise_prompt(prompt)).split()),
        )

    @staticmethod
    def _normalise_prompt(prompt: Sequence[str] | str) -> List[str]:
        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError("Prompt cannot be empty")
            return [prompt]
        parts = [part for part in prompt if part and part.strip()]
        if not parts:
            raise ValueError("Prompt cannot be empty")
        return parts


__all__ = ["GeminiClient", "GeminiClientSettings", "GeminiStreamingResponse"]
