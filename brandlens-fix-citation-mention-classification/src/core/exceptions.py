"""
BrandLens Custom Exceptions

This module defines a comprehensive exception hierarchy for the BrandLens brand visibility
analyzer. All exceptions are production-ready with proper error handling, context tracking,
and debugging information for real API integration with Gemini and Tavily.

Key Features:
- Hierarchical exception structure for precise error handling
- Context-aware error messages with debugging information
- Retry-friendly error classification for API resilience
- Support for error chaining and cause tracking
- Integration with monitoring and logging systems
- Production-ready error handling for real API integration

Exception Categories:
- API Integration Errors: Gemini, Tavily, rate limiting, authentication
- Data Processing Errors: Validation, compression, extraction, parsing
- System Errors: Cache, configuration, timeouts, insufficient data
- Business Logic Errors: Brand analysis, validation, search strategy

Author: BrandLens Development Team
Version: 1.0.0
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


class BrandLensError(Exception):
    """
    Base exception class for all BrandLens errors.

    Provides common functionality for error tracking, context preservation,
    and debugging information across the entire application.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        retryable: bool = False,
        user_message: Optional[str] = None,
    ) -> None:
        """
        Initialize a BrandLens error.

        Args:
            message: Technical error message for developers
            error_code: Unique error code for categorization
            context: Additional context information
            cause: The underlying exception that caused this error
            retryable: Whether this error might succeed on retry
            user_message: User-friendly error message
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause
        self.retryable = retryable
        self.user_message = user_message or message
        self.timestamp = datetime.now(timezone.utc)
        self.traceback_str = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "context": self.context,
            "retryable": self.retryable,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback_str,
        }

    def __str__(self) -> str:
        """String representation with context."""
        base_msg = f"{self.error_code}: {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        if self.cause:
            base_msg += f" (Caused by: {self.cause})"
        return base_msg


# API Integration Errors
class APIError(BrandLensError):
    """
    Base class for all API-related errors.

    Provides common functionality for tracking API response details,
    status codes, and service-specific error information.
    """

    def __init__(
        self,
        message: str,
        *,
        service: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        endpoint: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize an API error.

        Args:
            message: Error message
            service: Name of the API service (e.g., "gemini", "tavily")
            status_code: HTTP status code if applicable
            response_data: Raw response data from the API
            endpoint: API endpoint that failed
            request_id: Unique request identifier for tracking
            **kwargs: Additional arguments passed to BrandLensError
        """
        context = kwargs.pop("context", {})
        context.update({
            "service": service,
            "status_code": status_code,
            "endpoint": endpoint,
            "request_id": request_id,
        })
        if response_data:
            context["response_data"] = response_data

        super().__init__(message, context=context, **kwargs)
        self.service = service
        self.status_code = status_code
        self.response_data = response_data
        self.endpoint = endpoint
        self.request_id = request_id


class GeminiAPIError(APIError):
    """
    Gemini-specific API errors.

    Handles errors from Google's Gemini LLM API including quota issues,
    model unavailability, and content filtering problems.
    """

    def __init__(
        self,
        message: str,
        *,
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Gemini API error.

        Args:
            message: Error message
            model: Gemini model that failed (e.g., "gemini-1.5-flash")
            prompt_tokens: Number of tokens in the prompt
            **kwargs: Additional arguments passed to APIError
        """
        context = kwargs.pop("context", {})
        context.update({
            "model": model,
            "prompt_tokens": prompt_tokens,
        })

        super().__init__(
            message,
            service="gemini",
            context=context,
            **kwargs,
        )
        self.model = model
        self.prompt_tokens = prompt_tokens


class TavilyAPIError(APIError):
    """
    Tavily-specific API errors.

    Handles errors from Tavily search API including search depth issues,
    content retrieval problems, and search quota limitations.
    """

    def __init__(
        self,
        message: str,
        *,
        query: Optional[str] = None,
        search_depth: Optional[str] = None,
        max_results: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Tavily API error.

        Args:
            message: Error message
            query: Search query that failed
            search_depth: Search depth setting ("basic" or "advanced")
            max_results: Maximum results requested
            **kwargs: Additional arguments passed to APIError
        """
        context = kwargs.pop("context", {})
        context.update({
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
        })

        super().__init__(
            message,
            service="tavily",
            context=context,
            **kwargs,
        )
        self.query = query
        self.search_depth = search_depth
        self.max_results = max_results


class RateLimitError(APIError):
    """
    API rate limiting errors.

    Handles rate limit exceeded scenarios with retry timing information
    and quota consumption details.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: Optional[int] = None,
        quota_type: Optional[str] = None,
        current_usage: Optional[int] = None,
        quota_limit: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a rate limit error.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            quota_type: Type of quota exceeded (e.g., "requests", "tokens")
            current_usage: Current quota usage
            quota_limit: Maximum quota limit
            **kwargs: Additional arguments passed to APIError
        """
        context = kwargs.pop("context", {})
        context.update({
            "retry_after": retry_after,
            "quota_type": quota_type,
            "current_usage": current_usage,
            "quota_limit": quota_limit,
        })

        # Rate limit errors are typically retryable
        kwargs.setdefault("retryable", True)

        super().__init__(message, context=context, **kwargs)
        self.retry_after = retry_after
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.quota_limit = quota_limit


class QuotaExceededError(APIError):
    """
    API quota exceeded errors.

    Handles scenarios where API quotas are permanently exceeded
    and require manual intervention or billing updates.
    """

    def __init__(
        self,
        message: str,
        *,
        quota_type: Optional[str] = None,
        period: Optional[str] = None,
        reset_time: Optional[datetime] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a quota exceeded error.

        Args:
            message: Error message
            quota_type: Type of quota exceeded
            period: Quota period (e.g., "daily", "monthly")
            reset_time: When the quota resets
            **kwargs: Additional arguments passed to APIError
        """
        context = kwargs.pop("context", {})
        context.update({
            "quota_type": quota_type,
            "period": period,
            "reset_time": reset_time.isoformat() if reset_time else None,
        })

        # Quota exceeded errors are typically not retryable
        kwargs.setdefault("retryable", False)

        super().__init__(message, context=context, **kwargs)
        self.quota_type = quota_type
        self.period = period
        self.reset_time = reset_time


class AuthenticationError(APIError):
    """
    API authentication errors.

    Handles invalid API keys, expired tokens, and permission issues.
    """

    def __init__(
        self,
        message: str,
        *,
        auth_type: Optional[str] = None,
        key_masked: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize an authentication error.

        Args:
            message: Error message
            auth_type: Type of authentication (e.g., "api_key", "token")
            key_masked: Masked version of the key for debugging
            **kwargs: Additional arguments passed to APIError
        """
        context = kwargs.pop("context", {})
        context.update({
            "auth_type": auth_type,
            "key_masked": key_masked,
        })

        # Authentication errors are typically not retryable
        kwargs.setdefault("retryable", False)

        super().__init__(message, context=context, **kwargs)
        self.auth_type = auth_type
        self.key_masked = key_masked


# Data Processing Errors
class ValidationError(BrandLensError):
    """
    Data validation errors.

    Handles Pydantic validation failures, schema mismatches,
    and data integrity issues.
    """

    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Any = None,
        model: Optional[str] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a validation error.

        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            model: Pydantic model name
            validation_errors: List of Pydantic validation errors
            **kwargs: Additional arguments passed to BrandLensError
        """
        context = kwargs.pop("context", {})
        context.update({
            "field": field,
            "value": str(value) if value is not None else None,
            "model": model,
            "validation_errors": validation_errors,
        })

        super().__init__(message, context=context, **kwargs)
        self.field = field
        self.value = value
        self.model = model
        self.validation_errors = validation_errors or []


class CompressionError(BrandLensError):
    """
    Content compression errors.

    Handles token optimization failures, compression algorithm issues,
    and quality preservation problems.
    """

    def __init__(
        self,
        message: str,
        *,
        method: Optional[str] = None,
        original_tokens: Optional[int] = None,
        target_tokens: Optional[int] = None,
        content_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a compression error.

        Args:
            message: Error message
            method: Compression method used
            original_tokens: Original token count
            target_tokens: Target token count
            content_length: Content length in characters
            **kwargs: Additional arguments passed to BrandLensError
        """
        context = kwargs.pop("context", {})
        context.update({
            "method": method,
            "original_tokens": original_tokens,
            "target_tokens": target_tokens,
            "content_length": content_length,
        })

        super().__init__(message, context=context, **kwargs)
        self.method = method
        self.original_tokens = original_tokens
        self.target_tokens = target_tokens
        self.content_length = content_length


class ExtractionError(BrandLensError):
    """
    Information extraction errors.

    Handles failures in extracting citations, mentions, and metadata
    from LLM responses and search results.
    """

    def __init__(
        self,
        message: str,
        *,
        extraction_type: Optional[str] = None,
        content_preview: Optional[str] = None,
        pattern: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize an extraction error.

        Args:
            message: Error message
            extraction_type: Type of extraction (e.g., "citations", "mentions")
            content_preview: Preview of content that failed extraction
            pattern: Extraction pattern or regex that failed
            **kwargs: Additional arguments passed to BrandLensError
        """
        context = kwargs.pop("context", {})
        context.update({
            "extraction_type": extraction_type,
            "content_preview": content_preview[:200] if content_preview else None,
            "pattern": pattern,
        })

        super().__init__(message, context=context, **kwargs)
        self.extraction_type = extraction_type
        self.content_preview = content_preview
        self.pattern = pattern


class ParsingError(BrandLensError):
    """
    Response parsing errors.

    Handles failures in parsing API responses, JSON data,
    and structured content formats.
    """

    def __init__(
        self,
        message: str,
        *,
        parser_type: Optional[str] = None,
        raw_content: Optional[str] = None,
        expected_format: Optional[str] = None,
        parse_position: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a parsing error.

        Args:
            message: Error message
            parser_type: Type of parser (e.g., "json", "markdown", "xml")
            raw_content: Raw content that failed parsing
            expected_format: Expected content format
            parse_position: Position where parsing failed
            **kwargs: Additional arguments passed to BrandLensError
        """
        context = kwargs.pop("context", {})
        context.update({
            "parser_type": parser_type,
            "raw_content_preview": raw_content[:500] if raw_content else None,
            "expected_format": expected_format,
            "parse_position": parse_position,
        })

        super().__init__(message, context=context, **kwargs)
        self.parser_type = parser_type
        self.raw_content = raw_content
        self.expected_format = expected_format
        self.parse_position = parse_position


# System Errors
class CacheError(BrandLensError):
    """
    Caching system errors.

    Handles cache storage failures, retrieval issues,
    and cache corruption problems.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: Optional[str] = None,
        cache_key: Optional[str] = None,
        cache_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a cache error.

        Args:
            message: Error message
            operation: Cache operation that failed (e.g., "get", "set", "delete")
            cache_key: Cache key involved in the operation
            cache_path: File system path for cache storage
            **kwargs: Additional arguments passed to BrandLensError
        """
        context = kwargs.pop("context", {})
        context.update({
            "operation": operation,
            "cache_key": cache_key,
            "cache_path": cache_path,
        })

        super().__init__(message, context=context, **kwargs)
        self.operation = operation
        self.cache_key = cache_key
        self.cache_path = cache_path


class ConfigurationError(BrandLensError):
    """
    Configuration errors.

    Handles missing API keys, invalid configuration values,
    and environment setup issues.
    """

    def __init__(
        self,
        message: str,
        *,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        expected_type: Optional[str] = None,
        provided_value: Any = None,
        **kwargs,
    ) -> None:
        """
        Initialize a configuration error.

        Args:
            message: Error message
            config_key: Configuration key that's problematic
            config_file: Configuration file path
            expected_type: Expected value type
            provided_value: Actually provided value
            **kwargs: Additional arguments passed to BrandLensError
        """
        context = kwargs.pop("context", {})
        context.update({
            "config_key": config_key,
            "config_file": config_file,
            "expected_type": expected_type,
            "provided_value": str(provided_value) if provided_value is not None else None,
        })

        # Configuration errors are typically not retryable
        kwargs.setdefault("retryable", False)

        super().__init__(message, context=context, **kwargs)
        self.config_key = config_key
        self.config_file = config_file
        self.expected_type = expected_type
        self.provided_value = provided_value


class TimeoutError(BrandLensError):
    """
    Operation timeout errors.

    Handles API request timeouts, processing timeouts,
    and deadline exceeded scenarios.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a timeout error.

        Args:
            message: Error message
            operation: Operation that timed out
            timeout_seconds: Configured timeout duration
            elapsed_seconds: Actual elapsed time
            **kwargs: Additional arguments passed to BrandLensError
        """
        context = kwargs.pop("context", {})
        context.update({
            "operation": operation,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": elapsed_seconds,
        })

        # Timeout errors might be retryable depending on the cause
        kwargs.setdefault("retryable", True)

        super().__init__(message, context=context, **kwargs)
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


class InsufficientDataError(BrandLensError):
    """
    Insufficient data errors.

    Handles scenarios where there's not enough data for meaningful
    brand analysis or visibility calculations.
    """

    def __init__(
        self,
        message: str,
        *,
        data_type: Optional[str] = None,
        required_minimum: Optional[int] = None,
        actual_count: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize an insufficient data error.

        Args:
            message: Error message
            data_type: Type of data that's insufficient (e.g., "search_results", "citations")
            required_minimum: Minimum required count
            actual_count: Actual count available
            **kwargs: Additional arguments passed to BrandLensError
        """
        context = kwargs.pop("context", {})
        context.update({
            "data_type": data_type,
            "required_minimum": required_minimum,
            "actual_count": actual_count,
        })

        super().__init__(message, context=context, **kwargs)
        self.data_type = data_type
        self.required_minimum = required_minimum
        self.actual_count = actual_count


# Business Logic Errors
class BrandAnalysisError(BrandLensError):
    """
    Brand analysis pipeline errors.

    Handles failures in the core brand visibility analysis process,
    metric calculations, and report generation.
    """

    def __init__(
        self,
        message: str,
        *,
        analysis_stage: Optional[str] = None,
        brand_name: Optional[str] = None,
        query: Optional[str] = None,
        partial_results: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a brand analysis error.

        Args:
            message: Error message
            analysis_stage: Stage where analysis failed (e.g., "search", "compression", "extraction")
            brand_name: Brand being analyzed
            query: Analysis query
            partial_results: Any partial results that were generated
            **kwargs: Additional arguments passed to BrandLensError
        """
        context = kwargs.pop("context", {})
        context.update({
            "analysis_stage": analysis_stage,
            "brand_name": brand_name,
            "query": query,
            "has_partial_results": bool(partial_results),
        })

        super().__init__(message, context=context, **kwargs)
        self.analysis_stage = analysis_stage
        self.brand_name = brand_name
        self.query = query
        self.partial_results = partial_results or {}


class InvalidBrandError(BrandLensError):
    """
    Brand validation errors.

    Handles invalid brand names, malformed domains,
    and brand information validation failures.
    """

    def __init__(
        self,
        message: str,
        *,
        brand_name: Optional[str] = None,
        brand_domain: Optional[str] = None,
        validation_issue: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize an invalid brand error.

        Args:
            message: Error message
            brand_name: Invalid brand name
            brand_domain: Invalid brand domain
            validation_issue: Specific validation issue
            **kwargs: Additional arguments passed to BrandLensError
        """
        context = kwargs.pop("context", {})
        context.update({
            "brand_name": brand_name,
            "brand_domain": brand_domain,
            "validation_issue": validation_issue,
        })

        # Brand validation errors are typically not retryable
        kwargs.setdefault("retryable", False)

        super().__init__(message, context=context, **kwargs)
        self.brand_name = brand_name
        self.brand_domain = brand_domain
        self.validation_issue = validation_issue


class SearchStrategyError(BrandLensError):
    """
    Search strategy errors.

    Handles failures in search query generation, strategy selection,
    and search optimization problems.
    """

    def __init__(
        self,
        message: str,
        *,
        strategy: Optional[str] = None,
        query: Optional[str] = None,
        search_depth: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a search strategy error.

        Args:
            message: Error message
            strategy: Search strategy that failed
            query: Search query
            search_depth: Search depth setting
            **kwargs: Additional arguments passed to BrandLensError
        """
        context = kwargs.pop("context", {})
        context.update({
            "strategy": strategy,
            "query": query,
            "search_depth": search_depth,
        })

        super().__init__(message, context=context, **kwargs)
        self.strategy = strategy
        self.query = query
        self.search_depth = search_depth


# Utility Functions
def classify_error_for_retry(error: Exception) -> bool:
    """
    Classify whether an error should be retried.

    Args:
        error: Exception to classify

    Returns:
        True if the error might succeed on retry, False otherwise
    """
    if isinstance(error, BrandLensError):
        return error.retryable

    # Default retry logic for non-BrandLens errors
    retryable_types = (
        ConnectionError,
        TimeoutError,
        OSError,
    )

    return isinstance(error, retryable_types)


def get_user_friendly_message(error: Exception) -> str:
    """
    Get a user-friendly error message.

    Args:
        error: Exception to get message for

    Returns:
        User-friendly error message
    """
    if isinstance(error, BrandLensError):
        return error.user_message

    # Default user-friendly messages for common errors
    error_messages = {
        ConnectionError: "Unable to connect to the service. Please check your internet connection.",
        TimeoutError: "The operation took too long to complete. Please try again.",
        PermissionError: "Permission denied. Please check your access rights.",
        FileNotFoundError: "Required file not found. Please check your configuration.",
    }

    return error_messages.get(type(error), str(error))


def create_error_context(
    operation: str,
    **additional_context: Any,
) -> Dict[str, Any]:
    """
    Create standardized error context.

    Args:
        operation: Operation being performed
        **additional_context: Additional context information

    Returns:
        Standardized error context dictionary
    """
    context = {
        "operation": operation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_traceback": traceback.format_exc(),
    }
    context.update(additional_context)
    return context


# Export all exceptions for easy importing
__all__ = [
    # Base exception
    "BrandLensError",
    # API Integration Errors
    "APIError",
    "GeminiAPIError",
    "TavilyAPIError",
    "RateLimitError",
    "QuotaExceededError",
    "AuthenticationError",
    # Data Processing Errors
    "ValidationError",
    "CompressionError",
    "ExtractionError",
    "ParsingError",
    # System Errors
    "CacheError",
    "ConfigurationError",
    "TimeoutError",
    "InsufficientDataError",
    # Business Logic Errors
    "BrandAnalysisError",
    "InvalidBrandError",
    "SearchStrategyError",
    # Utility Functions
    "classify_error_for_retry",
    "get_user_friendly_message",
    "create_error_context",
]