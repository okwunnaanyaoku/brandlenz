"""
BrandLens Token Counter Module

Advanced token counting and cost calculation system for LLM optimization.
This module provides accurate token counting using tiktoken, cost calculation
for Gemini models, and performance optimization features.

Key Features:
- Exact token counting using tiktoken library
- Real Gemini pricing integration with input/output differentiation
- Multi-model support for different Gemini variants
- Performance-optimized caching for repeated content
- Batch processing capabilities
- Memory-efficient for large content processing

Integration Points:
- Uses existing Pydantic models from src.core.models
- Integrates with budget management system
- Supports compression pipeline measurement
- Enables CLI budget enforcement

Author: BrandLens Development Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import tiktoken
from pydantic import BaseModel, Field, computed_field, field_validator

from ..core.models import ModelName


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TokenizationResult:
    """Result of tokenizing content with metadata."""

    content: str
    token_count: int
    tokens: List[int]
    encoding_name: str
    model_name: str
    processing_time_ms: float
    content_hash: str


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for LLM usage."""

    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    model_name: str
    calculated_at: datetime


class TokenCountingError(Exception):
    """Raised when token counting operations fail."""
    pass


class UnsupportedModelError(TokenCountingError):
    """Raised when an unsupported model is requested."""
    pass


class TokenCounter:
    """
    Advanced token counter with exact counting, cost calculation, and optimization features.

    This class provides production-ready token counting capabilities for BrandLens,
    including accurate tiktoken-based tokenization, real Gemini pricing, and
    performance optimization through caching and batch processing.

    Features:
    - Accurate token counting using tiktoken
    - Real Gemini 1.5 Flash pricing (updated pricing as of 2024)
    - Multi-model support for Gemini variants
    - LRU caching for repeated content
    - Batch processing for efficiency
    - Memory-optimized for large content

    Example:
        ```python
        counter = TokenCounter()

        # Count tokens for a single text
        result = counter.count_tokens("Hello, world!", ModelName.GEMINI_FLASH)
        print(f"Token count: {result.token_count}")

        # Calculate cost for input/output
        cost = counter.calculate_cost(
            input_tokens=100,
            output_tokens=50,
            model=ModelName.GEMINI_FLASH
        )
        print(f"Total cost: ${cost.total_cost_usd}")

        # Batch process multiple texts
        texts = ["Text 1", "Text 2", "Text 3"]
        results = counter.count_tokens_batch(texts, ModelName.GEMINI_FLASH)
        ```
    """

    # Current Gemini 1.5 Flash pricing (per 1K tokens) - Updated 2024
    PRICING = {
        ModelName.GEMINI_FLASH: {
            "input_per_1k": 0.000075,   # $0.075 per 1M tokens = $0.000075 per 1K input tokens
            "output_per_1k": 0.0003,    # $0.30 per 1M tokens = $0.0003 per 1K output tokens
        },
        ModelName.GEMINI_FLASH_LATEST: {
            "input_per_1k": 0.000075,   # Same as flash
            "output_per_1k": 0.0003,
        },
        ModelName.GEMINI_PRO: {
            "input_per_1k": 0.00125,    # $0.00125 per 1K input tokens
            "output_per_1k": 0.005,     # $0.005 per 1K output tokens
        },
    }

    # Model to encoding mapping (supports various Gemini model name formats)
    MODEL_ENCODINGS = {
        "gemini-2.5-flash": "cl100k_base",
        "gemini-2.5-flash-lite": "cl100k_base",
        "gemini-1.5-flash": "cl100k_base",
        "gemini-1.5-pro": "cl100k_base",
        "models/gemini-2.5-flash": "cl100k_base",
        "models/gemini-2.5-flash-lite": "cl100k_base",
        "models/gemini-1.5-flash": "cl100k_base",
        "models/gemini-1.5-pro": "cl100k_base",
    }

    def __init__(self, cache_size: int = 1000, enable_metrics: bool = True):
        """
        Initialize the TokenCounter with specified cache size.

        Args:
            cache_size: Maximum number of entries to cache for repeated content
            enable_metrics: Whether to track detailed performance metrics
        """
        self.cache_size = cache_size
        self.enable_metrics = enable_metrics
        self._encoding_cache: Dict[str, tiktoken.Encoding] = {}
        self._content_cache: Dict[str, TokenizationResult] = {}
        self._lock = threading.Lock()

        # Performance metrics
        self._metrics = {
            "total_calls": 0,
            "total_tokens_processed": 0,
            "total_processing_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_operations": 0,
            "average_tokens_per_call": 0.0,
            "average_processing_time_ms": 0.0
        } if enable_metrics else {}

        # Initialize commonly used encodings
        self._preload_encodings()

        logger.info(f"TokenCounter initialized with cache size: {cache_size}, metrics: {enable_metrics}")

    def _preload_encodings(self) -> None:
        """Preload commonly used encodings for better performance."""
        try:
            for encoding_name in set(self.MODEL_ENCODINGS.values()):
                self._encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
            logger.debug(f"Preloaded {len(self._encoding_cache)} encodings")
        except Exception as e:
            logger.warning(f"Failed to preload some encodings: {e}")

    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """
        Get the tiktoken encoding for a specific model.

        Args:
            model: The model to get encoding for

        Returns:
            The tiktoken encoding instance

        Raises:
            UnsupportedModelError: If the model is not supported
        """
        if model not in self.MODEL_ENCODINGS:
            raise UnsupportedModelError(f"Model {model} is not supported")

        encoding_name = self.MODEL_ENCODINGS[model]

        if encoding_name not in self._encoding_cache:
            try:
                self._encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                raise TokenCountingError(f"Failed to load encoding {encoding_name}: {e}")

        return self._encoding_cache[encoding_name]

    def _generate_content_hash(self, content: str, model: str) -> str:
        """Generate a hash for content and model combination for caching."""
        combined = f"{content}|{model}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    @lru_cache(maxsize=1000)
    def _tokenize_cached(self, content: str, model: str) -> Tuple[int, List[int], str]:
        """
        Cached tokenization method using LRU cache.

        Returns:
            Tuple of (token_count, tokens, encoding_name)
        """
        encoding = self._get_encoding(model)
        tokens = encoding.encode(content)
        return len(tokens), tokens, encoding.name

    def count_tokens(
        self,
        content: str,
        model: str = "gemini-2.5-flash",
        use_cache: bool = True
    ) -> TokenizationResult:
        """
        Count tokens in content for the specified model.

        Args:
            content: The text content to tokenize
            model: The model to use for tokenization
            use_cache: Whether to use caching for performance

        Returns:
            TokenizationResult with detailed tokenization information

        Raises:
            TokenCountingError: If tokenization fails
            UnsupportedModelError: If the model is not supported
        """
        start_time = time.perf_counter()

        try:
            # Generate content hash for caching
            content_hash = self._generate_content_hash(content, model) if use_cache else None

            # Check cache first if enabled
            if use_cache and content_hash in self._content_cache:
                cached_result = self._content_cache[content_hash]

                # Update metrics
                if self.enable_metrics:
                    with self._lock:
                        self._metrics["cache_hits"] += 1
                        self._metrics["total_calls"] += 1

                logger.debug(f"Cache hit for content hash: {content_hash[:8]}...")
                return cached_result

            # Perform tokenization
            token_count, tokens, encoding_name = self._tokenize_cached(content, model)

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            result = TokenizationResult(
                content=content,
                token_count=token_count,
                tokens=tokens,
                encoding_name=encoding_name,
                model_name=model,
                processing_time_ms=processing_time_ms,
                content_hash=content_hash or ""
            )

            # Cache the result if caching is enabled
            if use_cache and content_hash:
                # Implement simple LRU by removing oldest if cache is full
                if len(self._content_cache) >= self.cache_size:
                    # Remove the first item (oldest)
                    oldest_key = next(iter(self._content_cache))
                    del self._content_cache[oldest_key]

                self._content_cache[content_hash] = result

            # Update metrics
            if self.enable_metrics:
                with self._lock:
                    self._metrics["total_calls"] += 1
                    self._metrics["cache_misses"] += 1
                    self._metrics["total_tokens_processed"] += token_count
                    self._metrics["total_processing_time_ms"] += processing_time_ms

                    # Update averages
                    if self._metrics["total_calls"] > 0:
                        self._metrics["average_tokens_per_call"] = (
                            self._metrics["total_tokens_processed"] / self._metrics["total_calls"]
                        )
                        self._metrics["average_processing_time_ms"] = (
                            self._metrics["total_processing_time_ms"] / self._metrics["total_calls"]
                        )

            logger.debug(
                f"Tokenized {len(content)} chars -> {token_count} tokens "
                f"({model}) in {processing_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Token counting failed after {processing_time_ms:.2f}ms: {e}")
            raise TokenCountingError(f"Failed to count tokens: {e}") from e

    def count_tokens_batch(
        self,
        contents: List[str],
        model: str = "gemini-2.5-flash",
        use_cache: bool = True
    ) -> List[TokenizationResult]:
        """
        Count tokens for multiple content pieces efficiently.

        Args:
            contents: List of text content to tokenize
            model: The model to use for tokenization
            use_cache: Whether to use caching for performance

        Returns:
            List of TokenizationResult objects

        Raises:
            TokenCountingError: If batch processing fails
        """
        start_time = time.perf_counter()
        results = []

        try:
            for i, content in enumerate(contents):
                try:
                    result = self.count_tokens(content, model, use_cache)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to tokenize item {i}: {e}")
                    # Create a failed result
                    failed_result = TokenizationResult(
                        content=content,
                        token_count=0,
                        tokens=[],
                        encoding_name="",
                        model_name=model,
                        processing_time_ms=0.0,
                        content_hash=""
                    )
                    results.append(failed_result)

            total_time_ms = (time.perf_counter() - start_time) * 1000
            total_tokens = sum(r.token_count for r in results)

            # Update batch metrics
            if self.enable_metrics:
                with self._lock:
                    self._metrics["batch_operations"] += 1

            logger.info(
                f"Batch processed {len(contents)} items -> {total_tokens} total tokens "
                f"in {total_time_ms:.2f}ms"
            )

            return results

        except Exception as e:
            logger.error(f"Batch token counting failed: {e}")
            raise TokenCountingError(f"Batch processing failed: {e}") from e

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gemini-2.5-flash"
    ) -> CostBreakdown:
        """
        Calculate the cost breakdown for input and output tokens.

        Args:
            input_tokens: Number of input (prompt) tokens
            output_tokens: Number of output (completion) tokens
            model: The model used for pricing

        Returns:
            CostBreakdown with detailed cost information

        Raises:
            UnsupportedModelError: If the model pricing is not available
        """
        if model not in self.PRICING:
            raise UnsupportedModelError(f"Pricing not available for model {model}")

        pricing = self.PRICING[model]

        # Calculate costs
        input_cost_usd = (input_tokens / 1000) * pricing["input_per_1k"]
        output_cost_usd = (output_tokens / 1000) * pricing["output_per_1k"]
        total_cost_usd = input_cost_usd + output_cost_usd

        breakdown = CostBreakdown(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=round(input_cost_usd, 8),
            output_cost_usd=round(output_cost_usd, 8),
            total_cost_usd=round(total_cost_usd, 8),
            model_name=model,
            calculated_at=datetime.now(timezone.utc)
        )

        logger.debug(
            f"Cost calculated: {input_tokens} input + {output_tokens} output "
            f"= ${total_cost_usd:.8f} ({model})"
        )

        return breakdown

    def estimate_compression_savings(
        self,
        original_content: str,
        compressed_content: str,
        model: str = "gemini-2.5-flash"
    ) -> Dict[str, Union[int, float]]:
        """
        Estimate cost savings from content compression.

        Args:
            original_content: The original uncompressed content
            compressed_content: The compressed content
            model: The model to use for calculation

        Returns:
            Dictionary with compression metrics and cost savings
        """
        try:
            original_result = self.count_tokens(original_content, model)
            compressed_result = self.count_tokens(compressed_content, model)

            tokens_saved = original_result.token_count - compressed_result.token_count
            compression_ratio = 1.0 - (compressed_result.token_count / original_result.token_count) if original_result.token_count > 0 else 0.0

            # Calculate cost savings (assuming this is input content)
            original_cost = self.calculate_cost(original_result.token_count, 0, model)
            compressed_cost = self.calculate_cost(compressed_result.token_count, 0, model)
            cost_savings_usd = original_cost.total_cost_usd - compressed_cost.total_cost_usd

            return {
                "original_tokens": original_result.token_count,
                "compressed_tokens": compressed_result.token_count,
                "tokens_saved": tokens_saved,
                "compression_ratio": round(compression_ratio, 4),
                "compression_percentage": round(compression_ratio * 100, 2),
                "original_cost_usd": original_cost.total_cost_usd,
                "compressed_cost_usd": compressed_cost.total_cost_usd,
                "cost_savings_usd": round(cost_savings_usd, 8),
                "model": model
            }

        except Exception as e:
            logger.error(f"Failed to estimate compression savings: {e}")
            raise TokenCountingError(f"Compression estimation failed: {e}") from e

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about the token counting cache.

        Returns:
            Dictionary with cache statistics
        """
        # Get LRU cache info
        cache_info = self._tokenize_cached.cache_info()

        stats = {
            "content_cache_size": len(self._content_cache),
            "content_cache_max": self.cache_size,
            "encoding_cache_size": len(self._encoding_cache),
            "lru_cache_hits": cache_info.hits,
            "lru_cache_misses": cache_info.misses,
            "lru_cache_size": cache_info.currsize,
            "lru_cache_max": cache_info.maxsize,
            "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses) * 100 if (cache_info.hits + cache_info.misses) > 0 else 0.0
        }

        # Add performance metrics if enabled
        if self.enable_metrics:
            stats.update(self._metrics)

        return stats

    def clear_cache(self, reset_metrics: bool = False) -> None:
        """
        Clear all caches to free memory.

        Args:
            reset_metrics: Whether to also reset performance metrics
        """
        self._content_cache.clear()
        self._tokenize_cached.cache_clear()

        if reset_metrics and self.enable_metrics:
            with self._lock:
                for key in self._metrics:
                    self._metrics[key] = 0 if key != "average_processing_time_ms" and key != "average_tokens_per_call" else 0.0

        logger.info(f"All caches cleared{' and metrics reset' if reset_metrics else ''}")

    def get_supported_models(self) -> List[ModelName]:
        """
        Get list of supported models.

        Returns:
            List of supported ModelName enum values
        """
        return list(self.MODEL_ENCODINGS.keys())

    def validate_budget_constraint(
        self,
        content: str,
        max_tokens: int,
        model: str = "gemini-2.5-flash"
    ) -> Dict[str, Union[bool, int, str]]:
        """
        Validate if content fits within a token budget constraint.

        Args:
            content: The content to validate
            max_tokens: Maximum allowed tokens
            model: The model to use for counting

        Returns:
            Dictionary with validation results
        """
        try:
            result = self.count_tokens(content, model)
            is_valid = result.token_count <= max_tokens

            return {
                "is_valid": is_valid,
                "token_count": result.token_count,
                "max_tokens": max_tokens,
                "tokens_over": max(0, result.token_count - max_tokens),
                "tokens_remaining": max(0, max_tokens - result.token_count),
                "model": model,
                "content_length_chars": len(content)
            }

        except Exception as e:
            logger.error(f"Budget validation failed: {e}")
            return {
                "is_valid": False,
                "token_count": 0,
                "max_tokens": max_tokens,
                "tokens_over": 0,
                "tokens_remaining": max_tokens,
                "model": model,
                "content_length_chars": len(content),
                "error": str(e)
            }

    def estimate_budget_consumption(
        self,
        contents: List[str],
        model: str = "gemini-2.5-flash",
        budget_usd: float = 1.0
    ) -> Dict[str, Union[int, float, bool]]:
        """
        Estimate how much of a budget would be consumed by processing content.

        Args:
            contents: List of content to process
            model: Model to use for estimation
            budget_usd: Available budget in USD

        Returns:
            Dictionary with budget consumption analysis
        """
        try:
            total_input_tokens = 0
            processing_time_total = 0.0

            start_time = time.perf_counter()

            # Count tokens for all content
            for content in contents:
                result = self.count_tokens(content, model, use_cache=True)
                total_input_tokens += result.token_count

            processing_time_total = (time.perf_counter() - start_time) * 1000

            # Estimate output tokens (assuming 50% of input as reasonable estimate)
            estimated_output_tokens = int(total_input_tokens * 0.5)

            # Calculate costs
            cost_breakdown = self.calculate_cost(total_input_tokens, estimated_output_tokens, model)
            budget_remaining = budget_usd - cost_breakdown.total_cost_usd
            budget_utilization = (cost_breakdown.total_cost_usd / budget_usd) * 100 if budget_usd > 0 else 0.0

            return {
                "total_content_pieces": len(contents),
                "total_input_tokens": total_input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "total_estimated_tokens": total_input_tokens + estimated_output_tokens,
                "estimated_cost_usd": cost_breakdown.total_cost_usd,
                "budget_usd": budget_usd,
                "budget_remaining_usd": budget_remaining,
                "budget_utilization_percent": round(budget_utilization, 2),
                "within_budget": cost_breakdown.total_cost_usd <= budget_usd,
                "processing_time_ms": processing_time_total,
                "model": model,
                "cost_per_content_piece": cost_breakdown.total_cost_usd / len(contents) if contents else 0.0
            }

        except Exception as e:
            logger.error(f"Budget estimation failed: {e}")
            raise TokenCountingError(f"Budget estimation failed: {e}") from e

    def optimize_content_for_budget(
        self,
        contents: List[str],
        max_budget_usd: float,
        model: str = "gemini-2.5-flash",
        target_compression_ratio: float = 0.65
    ) -> Dict[str, Any]:
        """
        Analyze content and provide optimization recommendations for budget constraints.

        Args:
            contents: List of content to analyze
            max_budget_usd: Maximum budget available
            model: Model to use for analysis
            target_compression_ratio: Target compression ratio (65% as per requirements)

        Returns:
            Dictionary with optimization recommendations
        """
        try:
            # Initial budget estimation
            budget_analysis = self.estimate_budget_consumption(contents, model, max_budget_usd)

            recommendations = []
            optimization_needed = not budget_analysis["within_budget"]

            if optimization_needed:
                # Calculate how much we need to reduce
                cost_reduction_needed = budget_analysis["estimated_cost_usd"] - max_budget_usd
                token_reduction_needed = int(
                    (cost_reduction_needed / budget_analysis["estimated_cost_usd"]) *
                    budget_analysis["total_estimated_tokens"]
                )

                recommendations.append({
                    "type": "compression",
                    "description": f"Apply {target_compression_ratio*100}% compression to reduce tokens",
                    "target_token_reduction": token_reduction_needed,
                    "estimated_cost_savings": cost_reduction_needed
                })

                # Calculate post-compression budget
                compressed_tokens = int(budget_analysis["total_input_tokens"] * (1 - target_compression_ratio))
                compressed_output_tokens = int(compressed_tokens * 0.5)
                compressed_cost = self.calculate_cost(compressed_tokens, compressed_output_tokens, model)

                recommendations.append({
                    "type": "post_compression_estimate",
                    "description": "Estimated budget after compression",
                    "compressed_input_tokens": compressed_tokens,
                    "compressed_output_tokens": compressed_output_tokens,
                    "compressed_cost_usd": compressed_cost.total_cost_usd,
                    "within_budget_after_compression": compressed_cost.total_cost_usd <= max_budget_usd
                })

            return {
                "optimization_needed": optimization_needed,
                "current_budget_analysis": budget_analysis,
                "recommendations": recommendations,
                "target_compression_ratio": target_compression_ratio,
                "analyzed_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Content optimization analysis failed: {e}")
            raise TokenCountingError(f"Content optimization failed: {e}") from e

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary of the TokenCounter.

        Returns:
            Dictionary with performance metrics and insights
        """
        cache_stats = self.get_cache_stats()

        summary = {
            "tokenizer_info": {
                "cache_size": self.cache_size,
                "metrics_enabled": self.enable_metrics,
                "supported_models": [m.value for m in self.get_supported_models()],
                "loaded_encodings": list(self._encoding_cache.keys())
            },
            "performance_stats": cache_stats,
            "pricing_info": self.PRICING,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

        if self.enable_metrics and self._metrics["total_calls"] > 0:
            summary["insights"] = {
                "efficiency_score": min(100, cache_stats.get("hit_rate", 0) + 10),  # Bonus for having metrics
                "average_processing_speed": f"{cache_stats.get('average_processing_time_ms', 0):.2f}ms per call",
                "total_tokens_processed": cache_stats.get("total_tokens_processed", 0),
                "cost_per_token_usd": self.PRICING[ModelName.GEMINI_FLASH_LATEST]["input_per_1k"] / 1000
            }

        return summary


# Global thread-safe counter instance for convenience functions
_global_counter = None
_global_counter_lock = threading.Lock()


def _get_global_counter() -> TokenCounter:
    """Get or create the global TokenCounter instance."""
    global _global_counter
    with _global_counter_lock:
        if _global_counter is None:
            _global_counter = TokenCounter(cache_size=500, enable_metrics=False)
        return _global_counter


# Convenience functions for direct usage
def count_tokens(
    content: str,
    model: str = "gemini-2.5-flash"
) -> int:
    """
    Convenience function to quickly count tokens in content.

    Args:
        content: Text content to count
        model: Model to use for counting

    Returns:
        Number of tokens
    """
    counter = _get_global_counter()
    result = counter.count_tokens(content, model)
    return result.token_count


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gemini-2.5-flash"
) -> float:
    """
    Convenience function to calculate cost.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model to use for pricing

    Returns:
        Total cost in USD
    """
    counter = _get_global_counter()
    breakdown = counter.calculate_cost(input_tokens, output_tokens, model)
    return breakdown.total_cost_usd


def estimate_content_cost(
    content: str,
    model: str = "gemini-2.5-flash",
    output_ratio: float = 0.5
) -> float:
    """
    Convenience function to estimate total cost for content processing.

    Args:
        content: Text content to analyze
        model: Model to use for estimation
        output_ratio: Ratio of output tokens to input tokens (default 0.5)

    Returns:
        Estimated total cost in USD
    """
    counter = _get_global_counter()
    result = counter.count_tokens(content, model)
    estimated_output = int(result.token_count * output_ratio)
    breakdown = counter.calculate_cost(result.token_count, estimated_output, model)
    return breakdown.total_cost_usd


def validate_content_budget(
    content: str,
    budget_usd: float,
    model: str = "gemini-2.5-flash"
) -> bool:
    """
    Convenience function to check if content fits within budget.

    Args:
        content: Text content to validate
        budget_usd: Available budget in USD
        model: Model to use for validation

    Returns:
        True if content fits within budget, False otherwise
    """
    estimated_cost = estimate_content_cost(content, model)
    return estimated_cost <= budget_usd


# Export public interface
__all__ = [
    # Core classes
    "TokenCounter",
    "TokenizationResult",
    "CostBreakdown",
    # Exception classes
    "TokenCountingError",
    "UnsupportedModelError",
    # Convenience functions
    "count_tokens",
    "calculate_cost",
    "estimate_content_cost",
    "validate_content_budget",
]