"""
BrandLens Core Data Models

This module defines comprehensive Pydantic v2 models for the BrandLens brand visibility
analyzer. All models are production-ready with complete validation, serialization, and
error handling for real API integration with Gemini and Tavily.

Key Features:
- Full Pydantic v2 compatibility with field validation
- Production-ready models for real API integration
- Comprehensive type hints and documentation
- JSON serialization/deserialization support
- Proper validation rules and constraints
- Performance tracking and cost analysis
- Cache-ready data structures

Author: BrandLens Development Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    computed_field,
)


class MentionType(str, Enum):
    """Types of brand mentions detected in content."""

    LINKED = "linked"
    UNLINKED = "unlinked"


class ModelName(str, Enum):
    """Supported LLM models for analysis."""

    GEMINI_FLASH = "gemini-2.5-flash-lite"
    GEMINI_FLASH_LATEST = "gemini-2.5-flash-lite"
    GEMINI_PRO = "gemini-2.5-flash-lite"


class SearchDepth(str, Enum):
    """Tavily search depth configurations."""

    BASIC = "basic"
    ADVANCED = "advanced"


class CompressionMethod(str, Enum):
    """Content compression methods."""

    SEMANTIC = "semantic"
    EXTRACTIVE = "extractive"
    HYBRID = "hybrid"


# Core Data Models
class SearchResult(BaseModel):
    """
    Represents a single search result from Tavily API.

    This model encapsulates search results with validation for URLs,
    content quality, and scoring metrics for downstream processing.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    url: str = Field(
        ...,
        description="The source URL of the search result",
        min_length=1,
    )
    title: str = Field(
        ...,
        description="The title or headline of the content",
        min_length=1,
        max_length=1000,
    )
    content: str = Field(
        ...,
        description="The extracted text content from the source",
        min_length=1,
    )
    score: float = Field(
        default=0.5,
        description="Relevance score from the search engine (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    raw_content: Optional[str] = Field(
        default=None,
        description="Original unprocessed content if available",
    )
    fetch_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this result was fetched",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate that the URL is properly formatted."""
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("URL must have valid scheme and domain")
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must use http or https protocol")
        return v

    @computed_field
    @property
    def domain(self) -> str:
        """Extract the domain from the URL."""
        return urlparse(self.url).netloc

    @computed_field
    @property
    def content_length(self) -> int:
        """Get the length of the content in characters."""
        return len(self.content)

    def is_owned_source(self, brand_domain: str) -> bool:
        """Check if this result comes from a brand-owned domain."""
        return brand_domain.lower() in self.domain.lower()


class LLMResponse(BaseModel):
    """
    Represents a response from an LLM (Gemini) API call.

    Tracks token usage, costs, and metadata for performance monitoring
    and optimization of the brand analysis pipeline.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    markdown_content: str = Field(
        ...,
        description="The generated markdown response from the LLM",
        min_length=1,
    )
    model: str = Field(
        ...,
        description="The specific LLM model used for generation",
    )
    prompt_tokens: int = Field(
        ...,
        description="Number of tokens in the input prompt",
        ge=0,
    )
    completion_tokens: int = Field(
        ...,
        description="Number of tokens in the generated response",
        ge=0,
    )
    total_tokens: int = Field(
        ...,
        description="Total tokens used (prompt + completion + cached)",
        ge=0,
    )
    cached_prompt_tokens: int = Field(
        default=0,
        description="Number of cached prompt tokens (billed at 10% rate)",
        ge=0,
    )
    generation_time_ms: float = Field(
        ...,
        description="Time taken to generate the response in milliseconds",
        ge=0.0,
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature setting used for generation",
        ge=0.0,
        le=2.0,
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this response was generated",
    )


    @computed_field
    @property
    def cost_usd(self) -> float:
        """Calculate the cost in USD for this API call."""
        # Gemini 1.5 Flash pricing (per 1K tokens)
        input_cost_per_1k = 0.00001875  # $0.000075 per 1K input tokens
        output_cost_per_1k = 0.0000375   # $0.000375 per 1K output tokens
        cached_cost_per_1k = 0.000001875  # $0.0000075 per 1K cached tokens (10% of input)

        input_cost = (self.prompt_tokens / 1000) * input_cost_per_1k
        output_cost = (self.completion_tokens / 1000) * output_cost_per_1k
        cached_cost = (self.cached_prompt_tokens / 1000) * cached_cost_per_1k

        return round(input_cost + output_cost + cached_cost, 6)

    @computed_field
    @property
    def tokens_per_second(self) -> float:
        """Calculate generation speed in tokens per second."""
        if self.generation_time_ms <= 0:
            return 0.0
        return (self.completion_tokens * 1000) / self.generation_time_ms


class Citation(BaseModel):
    """
    Represents a citation extracted from LLM-generated content.

    Citations link specific text to source URLs and associate them with
    entities for brand visibility analysis.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    text: str = Field(
        ...,
        description="The linked text that forms the citation",
        min_length=1,
        max_length=500,
    )
    url: str = Field(
        ...,
        description="The URL that the citation links to",
        min_length=1,
    )
    entity: str = Field(
        ...,
        description="The entity (brand, organization) associated with this citation",
        min_length=1,
        max_length=200,
    )
    confidence: float = Field(
        ...,
        description="Confidence score for the entity association (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    position: Optional[int] = Field(
        default=None,
        description="Character position where the citation appears in the text",
        ge=0,
    )
    context: Optional[str] = Field(
        default=None,
        description="Surrounding text context for the citation",
        max_length=1000,
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate that the URL is properly formatted."""
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Citation URL must have valid scheme and domain")
        return v

    @computed_field
    @property
    def domain(self) -> str:
        """Extract the domain from the citation URL."""
        return urlparse(self.url).netloc

    def is_brand_owned(self, brand_domain: str) -> bool:
        """Check if this citation links to a brand-owned domain."""
        return brand_domain.lower() in self.domain.lower()


class Mention(BaseModel):
    """
    Represents a brand mention detected in content.

    Mentions can be linked (with citations) or unlinked (plain text references)
    and include context for sentiment and positioning analysis.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    text: str = Field(
        ...,
        description="The exact text of the brand mention",
        min_length=1,
        max_length=200,
    )
    type: MentionType = Field(
        ...,
        description="Whether the mention is linked or unlinked",
    )
    position: int = Field(
        ...,
        description="Character position where the mention appears",
        ge=0,
    )
    context: str = Field(
        ...,
        description="Surrounding text context for the mention",
        min_length=1,
        max_length=1000,
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score for mention detection (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    sentiment: Optional[str] = Field(
        default=None,
        description="Sentiment of the mention context (positive/negative/neutral)",
    )

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v: Optional[str]) -> Optional[str]:
        """Validate sentiment values."""
        if v is not None:
            allowed_sentiments = {"positive", "negative", "neutral"}
            if v.lower() not in allowed_sentiments:
                raise ValueError(f"Sentiment must be one of: {allowed_sentiments}")
            return v.lower()
        return v

    @computed_field
    @property
    def context_length(self) -> int:
        """Get the length of the context in characters."""
        return len(self.context)


# Analysis Models
class CompressedContent(BaseModel):
    """
    Represents content that has been compressed for token optimization.

    Tracks compression metrics and preserves important metadata for
    quality assessment and pipeline optimization.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    original_content: str = Field(
        ...,
        description="The original uncompressed content",
    )
    compressed_content: str = Field(
        ...,
        description="The compressed version of the content",
    )
    method: CompressionMethod = Field(
        ...,
        description="The compression method used",
    )
    original_tokens: int = Field(
        ...,
        description="Number of tokens in the original content",
        ge=0,
    )
    compressed_tokens: int = Field(
        ...,
        description="Number of tokens in the compressed content",
        ge=0,
    )
    compression_time_ms: float = Field(
        ...,
        description="Time taken to compress the content in milliseconds",
        ge=0.0,
    )
    quality_score: float = Field(
        default=0.0,
        description="Quality preservation score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_compression(self) -> CompressedContent:
        """Ensure compression is meaningful."""
        if self.compressed_tokens > self.original_tokens:
            raise ValueError("Compressed content cannot have more tokens than original")
        return self

    @computed_field
    @property
    def compression_ratio(self) -> float:
        """Calculate the compression ratio (0.0-1.0)."""
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)

    @computed_field
    @property
    def tokens_saved(self) -> int:
        """Calculate the number of tokens saved."""
        return self.original_tokens - self.compressed_tokens


class BrandAnalysis(BaseModel):
    """
    Complete brand visibility analysis result.

    This is the main output model that aggregates all analysis components
    including LLM responses, citations, mentions, and calculated metrics.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # Core response data
    human_response_markdown: str = Field(
        ...,
        description="The generated markdown response analyzing the brand",
        min_length=1,
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="All citations extracted from the response",
    )
    mentions: List[Mention] = Field(
        default_factory=list,
        description="All brand mentions detected in the response",
    )

    # Source categorization
    owned_sources: List[str] = Field(
        default_factory=list,
        description="URLs from brand-owned domains",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="URLs from external sources",
    )

    # Analysis metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Processing metadata including costs and performance",
    )
    advanced_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Advanced visibility and competitive metrics",
    )

    # Analysis context
    brand_name: str = Field(
        ...,
        description="The brand being analyzed",
        min_length=1,
        max_length=200,
    )
    brand_domain: str = Field(
        ...,
        description="The primary domain for the brand",
        min_length=1,
    )
    query: str = Field(
        ...,
        description="The original query that prompted this analysis",
        min_length=1,
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this analysis was created",
    )

    @field_validator("brand_domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        """Validate brand domain format."""
        # Remove protocol if present
        domain = v.replace("http://", "").replace("https://", "")

        # Basic domain validation
        if "." not in domain or " " in domain:
            raise ValueError("Brand domain must be a valid domain name")

        return domain.lower()

    @computed_field
    @property
    def total_citations(self) -> int:
        """Get the total number of citations."""
        return len(self.citations)

    @computed_field
    @property
    def total_mentions(self) -> int:
        """Get the total number of mentions."""
        return len(self.mentions)

    @computed_field
    @property
    def owned_citation_count(self) -> int:
        """Count citations linking to brand-owned sources."""
        return sum(1 for c in self.citations if c.is_brand_owned(self.brand_domain))

    @computed_field
    @property
    def visibility_summary(self) -> Dict[str, Union[int, float]]:
        """Generate a summary of visibility metrics."""
        total_sources = len(self.owned_sources) + len(self.sources)
        owned_percentage = (
            len(self.owned_sources) / total_sources * 100
            if total_sources > 0 else 0.0
        )

        return {
            "total_citations": self.total_citations,
            "total_mentions": self.total_mentions,
            "owned_citations": self.owned_citation_count,
            "total_sources": total_sources,
            "owned_sources_percentage": round(owned_percentage, 2),
            "linked_mentions": sum(1 for m in self.mentions if m.type == MentionType.LINKED),
            "unlinked_mentions": sum(1 for m in self.mentions if m.type == MentionType.UNLINKED),
        }

    def to_json_file(self, filepath: str) -> None:
        """Export analysis to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            # Exclude computed fields from JSON export to avoid validation issues
            data = self.model_dump(mode="json")
            # Remove computed fields manually
            computed_fields = {
                "total_citations", "total_mentions", "owned_citation_count", "visibility_summary"
            }
            filtered_data = {k: v for k, v in data.items() if k not in computed_fields}

            # Also remove computed fields from nested models
            if "citations" in filtered_data:
                for citation in filtered_data["citations"]:
                    citation.pop("domain", None)
            if "mentions" in filtered_data:
                for mention in filtered_data["mentions"]:
                    mention.pop("context_length", None)

            json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, filepath: str) -> BrandAnalysis:
        """Load analysis from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


# System Models
class PerformanceMetrics(BaseModel):
    """
    Tracks system performance metrics for monitoring and optimization.

    Used to measure and optimize the performance of the brand analysis
    pipeline including API response times and processing efficiency.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # Timing metrics
    total_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds",
        ge=0.0,
    )
    search_time_ms: float = Field(
        ...,
        description="Time spent on search operations in milliseconds",
        ge=0.0,
    )
    compression_time_ms: float = Field(
        ...,
        description="Time spent on content compression in milliseconds",
        ge=0.0,
    )
    llm_time_ms: float = Field(
        ...,
        description="Time spent on LLM API calls in milliseconds",
        ge=0.0,
    )
    extraction_time_ms: float = Field(
        ...,
        description="Time spent on information extraction in milliseconds",
        ge=0.0,
    )

    # API usage metrics
    api_calls: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of API calls by service",
    )
    total_tokens: int = Field(
        default=0,
        description="Total tokens processed",
        ge=0,
    )
    cache_hits: int = Field(
        default=0,
        description="Number of cache hits",
        ge=0,
    )
    cache_misses: int = Field(
        default=0,
        description="Number of cache misses",
        ge=0,
    )

    # Cost tracking
    total_cost_usd: float = Field(
        default=0.0,
        description="Total cost in USD for this operation",
        ge=0.0,
    )
    last_rate_limit_remaining: Optional[int] = Field(
        default=None,
        description="Rate limit remaining after the run",
    )
    last_request_id: Optional[str] = Field(
        default=None,
        description="Last request identifier from API",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When these metrics were recorded",
    )

    @computed_field
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return (self.cache_hits / total_requests) * 100

    @computed_field
    @property
    def cost_per_token(self) -> float:
        """Calculate cost per token processed."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_cost_usd / self.total_tokens


class CacheEntry(BaseModel):
    """
    Represents a cached entry in the BrandLens caching system.

    Provides TTL-based caching with metadata for cache management
    and invalidation strategies.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    key: str = Field(
        ...,
        description="The cache key for this entry",
        min_length=1,
    )
    data: Dict[str, Any] = Field(
        ...,
        description="The cached data",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this cache entry was created",
    )
    expires_at: datetime = Field(
        ...,
        description="When this cache entry expires",
    )
    access_count: int = Field(
        default=0,
        description="Number of times this entry has been accessed",
        ge=0,
    )
    last_accessed: Optional[datetime] = Field(
        default=None,
        description="When this entry was last accessed",
    )
    size_bytes: int = Field(
        default=0,
        description="Size of the cached data in bytes",
        ge=0,
    )

    @model_validator(mode="after")
    def validate_expiration(self) -> CacheEntry:
        """Ensure expiration is after creation."""
        if self.expires_at <= self.created_at:
            raise ValueError("Expiration time must be after creation time")
        return self

    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return datetime.now(timezone.utc) >= self.expires_at

    @computed_field
    @property
    def ttl_seconds(self) -> float:
        """Get the time-to-live in seconds."""
        if self.is_expired:
            return 0.0
        delta = self.expires_at - datetime.now(timezone.utc)
        return delta.total_seconds()

    def mark_accessed(self) -> None:
        """Mark this entry as accessed and update counters."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)


# Configuration Models
class APIConfig(BaseModel):
    """Configuration for API clients."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # Gemini configuration
    gemini_api_key: str = Field(
        ...,
        description="Google Gemini API key",
        min_length=1,
    )
    gemini_model: str = Field(
        ...,
        description="Gemini model to use (set in .env as GEMINI_MODEL)",
    )
    gemini_temperature: float = Field(
        default=0.7,
        description="Temperature for Gemini generation",
        ge=0.0,
        le=2.0,
    )
    gemini_max_tokens: int = Field(
        default=8192,
        description="Maximum tokens for Gemini responses",
        gt=0,
    )

    # Tavily configuration
    tavily_api_key: str = Field(
        ...,
        description="Tavily search API key",
        min_length=1,
    )
    tavily_search_depth: SearchDepth = Field(
        default=SearchDepth.ADVANCED,
        description="Tavily search depth setting",
    )
    tavily_include_raw_content: bool = Field(
        default=True,
        description="Whether to include raw content from Tavily",
    )
    tavily_content_mode: str = Field(
        default="raw",
        description="Content extraction mode: 'raw' for full content, 'snippet' for summaries",
    )

    # Rate limiting
    max_searches_per_query: int = Field(
        default=5,
        description="Maximum number of searches per analysis",
        gt=0,
        le=10,
    )
    max_sources_per_search: int = Field(
        default=10,
        description="Maximum sources per search",
        gt=0,
        le=20,
    )


class CacheConfig(BaseModel):
    """Configuration for the caching system."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    cache_dir: str = Field(
        default=".cache",
        description="Directory for cache storage",
    )
    default_ttl_seconds: int = Field(
        default=3600,
        description="Default TTL for cache entries in seconds",
        gt=0,
    )
    max_cache_size_mb: int = Field(
        default=100,
        description="Maximum cache size in megabytes",
        gt=0,
    )
    cleanup_interval_seconds: int = Field(
        default=300,
        description="How often to clean up expired entries",
        gt=0,
    )


class AppConfig(BaseModel):
    """Main application configuration."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # API settings
    api: APIConfig = Field(
        ...,
        description="API configuration settings",
    )

    # Cache settings
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache configuration settings",
    )

    # Processing settings
    compression_target_tokens: int = Field(
        default=4000,
        description="Target token count for compression",
        gt=0,
    )
    compression_method: CompressionMethod = Field(
        default=CompressionMethod.SEMANTIC,
        description="Default compression method",
    )
    target_compression_ratio: float = Field(
        default=0.25,
        description="Target compression ratio (0.0-1.0, e.g., 0.25 = 25% compression)",
        gt=0.0,
        lt=1.0,
    )

    # Output settings
    output_format: str = Field(
        default="json",
        description="Default output format",
    )
    include_debug_info: bool = Field(
        default=False,
        description="Whether to include debug information",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path (None for console only)",
    )


# Export all models for easy importing
__all__ = [
    # Enums
    "MentionType",
    "ModelName",
    "SearchDepth",
    "CompressionMethod",
    # Core models
    "SearchResult",
    "LLMResponse",
    "Citation",
    "Mention",
    # Analysis models
    "CompressedContent",
    "BrandAnalysis",
    # System models
    "PerformanceMetrics",
    "CacheEntry",
    # Configuration models
    "APIConfig",
    "CacheConfig",
    "AppConfig",
]