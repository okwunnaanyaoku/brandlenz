"""Async Tavily client for BrandLens search integration."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..core import APIConfig, SearchDepth, TavilyAPIError

LOGGER = logging.getLogger(__name__)


class TavilyResponseMetadata(BaseModel):
    request_id: Optional[str] = None
    cost_usd: float = 0.0
    rate_limit_remaining: Optional[int] = None


class TavilySearchResult(BaseModel):
    """Single Tavily search hit."""

    url: str
    title: str
    content: Optional[str] = Field(default=None, description="Summary content or snippets")
    raw_content: Optional[str] = Field(default=None, description="Full cleaned HTML/markdown content")
    score: Optional[float] = None
    published_date: Optional[str] = None


class TavilySearchResponse(BaseModel):
    """High level response returned from a Tavily search."""

    query: str
    results: List[TavilySearchResult] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    total_results: Optional[int] = None
    metadata: TavilyResponseMetadata = Field(default_factory=TavilyResponseMetadata)


class TavilyContentResponse(BaseModel):
    """Response wrapper when requesting article content by URL."""

    url: str
    content: str
    metadata: TavilyResponseMetadata = Field(default_factory=TavilyResponseMetadata)
    source_search_query: Optional[str] = None


@dataclass
class TavilyClientSettings:
    """Runtime options for the Tavily client."""

    api_key: str
    base_url: str = "https://api.tavily.com"
    timeout: float = 30.0
    max_attempts: int = 3
    include_raw_content: bool = True
    enable_cache: bool = False

    @classmethod
    def from_api_config(cls, config: APIConfig, *, enable_cache: bool = False) -> "TavilyClientSettings":
        return cls(
            api_key=config.tavily_api_key,
            include_raw_content=config.tavily_include_raw_content,
            enable_cache=enable_cache,
        )


class TavilyClient:
    """Async wrapper around Tavily's REST API."""

    def __init__(
        self,
        settings: TavilyClientSettings,
        *,
        client: Optional[httpx.AsyncClient] = None,
        cache_enabled: Optional[bool] = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._owns_client = client is None
        self._cache_enabled = cache_enabled if cache_enabled is not None else settings.enable_cache
        self._cache: Dict[Tuple[str, str], Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    @classmethod
    def from_config(
        cls,
        config: APIConfig,
        *,
        client: Optional[httpx.AsyncClient] = None,
        enable_cache: bool = False,
    ) -> "TavilyClient":
        return cls(TavilyClientSettings.from_api_config(config, enable_cache=enable_cache), client=client)

    async def __aenter__(self) -> "TavilyClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.aclose()

    async def aclose(self) -> None:
        if self._client and self._owns_client:
            await self._client.aclose()
            self._client = None

    def clear_cache(self) -> None:
        self._cache.clear()

    async def search(
        self,
        query: str,
        *,
        depth: SearchDepth = SearchDepth.ADVANCED,
        max_results: int = 10,
        include_raw_content: Optional[Union[bool, str]] = None,
        chunks_per_source: int = 3,
    ) -> TavilySearchResponse:
        payload = {
            "query": query,
            "search_depth": depth.value,
            "max_results": max_results,
            "include_raw_content": (
                include_raw_content
                if include_raw_content is not None
                else self._settings.include_raw_content
            ),
            "chunks_per_source": chunks_per_source,
        }
        data, meta = await self._post_json("/search", payload)
        return TavilySearchResponse(
            query=data.get("query", query),
            results=[TavilySearchResult(**item) for item in data.get("results", [])],
            follow_up_questions=data.get("follow_up_questions", []) or [],
            total_results=data.get("total_results"),
            metadata=TavilyResponseMetadata(**meta),
        )

    async def get_content(self, url: str, *, source_query: Optional[str] = None) -> TavilyContentResponse:
        payload = {"url": url}
        data, meta = await self._post_json("/content", payload)
        return TavilyContentResponse(
            metadata=TavilyResponseMetadata(**meta),
            source_search_query=source_query,
            **data,
        )

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        await self._ensure_client()
        assert self._client is not None
        request_kwargs = {
            "json": payload,
            "timeout": self._settings.timeout,
        }
        headers = {"Authorization": f"Bearer {self._settings.api_key}"}

        cache_key: Optional[Tuple[str, str]] = None
        if self._cache_enabled:
            cache_key = (path, json.dumps(payload, sort_keys=True))
            if cache_key in self._cache:
                LOGGER.debug("Returning Tavily response from cache", extra={"path": path})
                cached_data, cached_meta = self._cache[cache_key]
                return deepcopy(cached_data), deepcopy(cached_meta)

        async for attempt in AsyncRetrying(
            reraise=True,
            retry=retry_if_exception_type(TavilyAPIError),
            wait=wait_exponential(min=1, max=5),
            stop=stop_after_attempt(self._settings.max_attempts),
        ):
            with attempt:
                try:
                    response = await self._client.post(
                        path,
                        headers=headers,
                        **request_kwargs,
                    )
                except httpx.RequestError as exc:
                    raise TavilyAPIError("Tavily request failed", status_code=None, cause=exc) from exc
                if response.status_code == 429:
                    meta = self._extract_metadata(response)
                    LOGGER.warning(
                        "Tavily rate limit hit",
                        extra={"path": path, **meta},
                    )
                    payload_dict: Dict[str, Any]
                    try:
                        payload_dict = response.json()
                    except ValueError:
                        payload_dict = {"error": response.text}
                    raise TavilyAPIError(
                        "Tavily rate limit hit",
                        status_code=response.status_code,
                        response_data=payload_dict,
                    )
                if response.status_code >= 400:
                    raise self._map_error(response)
                try:
                    data = response.json()
                except ValueError as exc:
                    raise TavilyAPIError("Tavily returned invalid JSON", status_code=response.status_code) from exc
                meta = self._extract_metadata(response)
                LOGGER.info(
                    "Tavily request succeeded",
                    extra={"path": path, **meta},
                )
                if self._cache_enabled and cache_key is not None:
                    self._cache[cache_key] = (deepcopy(data), deepcopy(meta))
                return data, meta
        raise TavilyAPIError("Tavily request failed", status_code=None)

    async def _ensure_client(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._settings.base_url)

    def _extract_metadata(self, response: httpx.Response) -> Dict[str, Any]:
        headers = response.headers

        def _get_float(key: str) -> float:
            value = headers.get(key)
            if not value:
                return 0.0
            try:
                return float(value)
            except ValueError:
                return 0.0

        def _get_int(key: str) -> Optional[int]:
            value = headers.get(key)
            if not value:
                return None
            try:
                return int(value)
            except ValueError:
                return None

        return {
            "request_id": headers.get("X-Request-Id"),
            "cost_usd": _get_float("X-Tavily-Cost"),
            "rate_limit_remaining": _get_int("X-RateLimit-Remaining"),
        }

    def _map_error(self, response: httpx.Response) -> TavilyAPIError:
        try:
            data = response.json()
        except ValueError:
            data = {"error": response.text}
        message = data.get("error") or data.get("message") or response.text
        return TavilyAPIError(
            message,
            status_code=response.status_code,
            response_data=data if isinstance(data, dict) else None,
        )
