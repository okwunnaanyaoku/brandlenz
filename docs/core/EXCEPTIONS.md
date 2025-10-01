# Exception System Documentation

## Overview

BrandLens implements a comprehensive exception hierarchy designed for production-ready error handling with real API integration. The system provides detailed error context, retry logic classification, and user-friendly error messages for robust error management.

**File Reference**: `src/core/exceptions.py`

## Exception Hierarchy

```
BrandLensError (Base)
├── APIError
│   ├── GeminiAPIError
│   ├── TavilyAPIError
│   ├── RateLimitError
│   ├── QuotaExceededError
│   └── AuthenticationError
├── ValidationError
├── CompressionError
├── ExtractionError
├── ParsingError
├── CacheError
├── ConfigurationError
├── TimeoutError
├── InsufficientDataError
├── BrandAnalysisError
├── InvalidBrandError
└── SearchStrategyError
```

## Base Exception Classes

### BrandLensError

The base exception class for all BrandLens errors with comprehensive error tracking.

**Location**: `src/core/exceptions.py:33-95`

```python
class BrandLensError(Exception):
    """Base exception class for all BrandLens errors."""
```

**Key Features**:
- **Context Preservation**: Rich context information for debugging
- **Retry Classification**: Automatic retry recommendation
- **User-Friendly Messages**: Separate technical and user messages
- **Timestamp Tracking**: Automatic error occurrence timing
- **Cause Chaining**: Support for underlying exception tracking
- **Serialization**: JSON-serializable for logging and monitoring

**Constructor Parameters**:
```python
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
```

**Core Attributes**:
- `message: str` - Technical error message for developers
- `error_code: str` - Unique error code for categorization
- `context: Dict[str, Any]` - Additional context information
- `cause: Optional[Exception]` - Underlying exception
- `retryable: bool` - Whether retry might succeed
- `user_message: str` - User-friendly error message
- `timestamp: datetime` - Error occurrence time (UTC)
- `traceback_str: str` - Captured stack trace

**Methods**:

#### to_dict()
```python
def to_dict(self) -> Dict[str, Any]:
    """Convert exception to dictionary for logging/serialization."""
```

**Returns**: Complete error information as dictionary including:
- Error type and code
- Technical and user messages
- Context information
- Retry recommendation
- Timestamp and cause details
- Stack trace information

#### __str__()
Enhanced string representation with context and cause information.

**Example Usage**:
```python
try:
    # Some operation
    pass
except Exception as e:
    raise BrandLensError(
        "Operation failed during processing",
        error_code="PROCESSING_FAILURE",
        context={"operation": "brand_analysis", "brand": "Apple"},
        cause=e,
        retryable=True,
        user_message="Analysis temporarily unavailable. Please try again."
    )
```

## API Integration Exceptions

### APIError

Base class for all external API-related errors with service-specific tracking.

**Location**: `src/core/exceptions.py:98-145`

**Service Tracking**:
- `service: str` - API service name ("gemini", "tavily")
- `status_code: int` - HTTP status code if applicable
- `endpoint: str` - API endpoint that failed
- `request_id: str` - Unique request identifier
- `response_data: Dict` - Raw API response data

### GeminiAPIError

Google Gemini LLM API specific errors.

**Location**: `src/core/exceptions.py:147-186`

**Additional Context**:
- `model: str` - Gemini model name (e.g., "gemini-1.5-flash")
- `prompt_tokens: int` - Token count in the prompt

**Common Scenarios**:
```python
# Model unavailable
raise GeminiAPIError(
    "Model temporarily unavailable",
    model="gemini-1.5-flash",
    status_code=503,
    retryable=True
)

# Content filtering
raise GeminiAPIError(
    "Content filtered by safety systems",
    model="gemini-1.5-flash",
    prompt_tokens=150,
    retryable=False,
    user_message="Content cannot be processed due to safety policies"
)

# Token limit exceeded
raise GeminiAPIError(
    "Prompt exceeds model token limit",
    model="gemini-1.5-flash",
    prompt_tokens=10000,
    retryable=False
)
```

### TavilyAPIError

Tavily search API specific errors.

**Location**: `src/core/exceptions.py:188-231`

**Additional Context**:
- `query: str` - Search query that failed
- `search_depth: str` - Search depth setting ("basic" or "advanced")
- `max_results: int` - Maximum results requested

**Common Scenarios**:
```python
# Search quota exceeded
raise TavilyAPIError(
    "Daily search quota exceeded",
    query="Apple iPhone features",
    search_depth="advanced",
    max_results=10,
    status_code=429,
    retryable=False
)

# Invalid search depth
raise TavilyAPIError(
    "Unsupported search depth requested",
    query="Tesla models",
    search_depth="premium",
    retryable=False
)
```

### RateLimitError

API rate limiting errors with retry timing information.

**Location**: `src/core/exceptions.py:233-278`

**Rate Limit Context**:
- `retry_after: int` - Seconds to wait before retrying
- `quota_type: str` - Type of quota ("requests", "tokens")
- `current_usage: int` - Current quota usage
- `quota_limit: int` - Maximum quota limit

**Default**: `retryable=True`

**Usage Example**:
```python
raise RateLimitError(
    "Request rate limit exceeded",
    service="gemini",
    retry_after=60,
    quota_type="requests",
    current_usage=100,
    quota_limit=100,
    user_message="Service temporarily busy. Please try again in a minute."
)
```

### QuotaExceededError

API quota permanently exceeded requiring manual intervention.

**Location**: `src/core/exceptions.py:280-321`

**Quota Context**:
- `quota_type: str` - Type of quota exceeded
- `period: str` - Quota period ("daily", "monthly")
- `reset_time: datetime` - When quota resets

**Default**: `retryable=False`

### AuthenticationError

API authentication and authorization errors.

**Location**: `src/core/exceptions.py:323-359`

**Auth Context**:
- `auth_type: str` - Authentication type ("api_key", "token")
- `key_masked: str` - Masked key for debugging (e.g., "AIza****")

**Default**: `retryable=False`

**Security Note**: Never logs actual API keys, only masked versions for debugging.

## Data Processing Exceptions

### ValidationError

Pydantic validation and data integrity errors.

**Location**: `src/core/exceptions.py:362-404`

**Validation Context**:
- `field: str` - Field that failed validation
- `value: Any` - Invalid value (converted to string)
- `model: str` - Pydantic model name
- `validation_errors: List[Dict]` - Detailed Pydantic errors

**Usage Example**:
```python
raise ValidationError(
    "Brand domain validation failed",
    field="brand_domain",
    value="invalid-domain",
    model="BrandAnalysis",
    validation_errors=pydantic_errors
)
```

### CompressionError

Token optimization and content compression errors.

**Location**: `src/core/exceptions.py:406-448`

**Compression Context**:
- `method: str` - Compression method used
- `original_tokens: int` - Original token count
- `target_tokens: int` - Target token count
- `content_length: int` - Content length in characters

**Common Scenarios**:
```python
# Insufficient compression
raise CompressionError(
    "Unable to achieve target compression ratio",
    method="semantic",
    original_tokens=5000,
    target_tokens=1000,
    content_length=25000
)

# Quality preservation failure
raise CompressionError(
    "Compression would degrade quality below threshold",
    method="semantic",
    original_tokens=2000,
    target_tokens=500
)
```

### ExtractionError

Information extraction failures (citations, mentions, entities).

**Location**: `src/core/exceptions.py:450-488`

**Extraction Context**:
- `extraction_type: str` - Type of extraction ("citations", "mentions", "entities")
- `content_preview: str` - Preview of problematic content (truncated to 200 chars)
- `pattern: str` - Extraction pattern or regex that failed

### ParsingError

Response parsing and format conversion errors.

**Location**: `src/core/exceptions.py:490-532`

**Parsing Context**:
- `parser_type: str` - Parser type ("json", "markdown", "xml")
- `raw_content: str` - Raw content that failed (preview only)
- `expected_format: str` - Expected content format
- `parse_position: int` - Character position where parsing failed

## System Exceptions

### CacheError

Caching system storage and retrieval errors.

**Location**: `src/core/exceptions.py:535-573`

**Cache Context**:
- `operation: str` - Cache operation ("get", "set", "delete")
- `cache_key: str` - Cache key involved
- `cache_path: str` - File system path for cache

### ConfigurationError

Configuration and environment setup errors.

**Location**: `src/core/exceptions.py:575-620`

**Configuration Context**:
- `config_key: str` - Problematic configuration key
- `config_file: str` - Configuration file path
- `expected_type: str` - Expected value type
- `provided_value: Any` - Actually provided value

**Default**: `retryable=False`

**Common Cases**:
```python
# Missing API key
raise ConfigurationError(
    "GEMINI_API_KEY is required but not set",
    config_key="GEMINI_API_KEY",
    expected_type="string",
    user_message="Please set your Gemini API key in the .env file"
)

# Invalid compression ratio
raise ConfigurationError(
    "compression_ratio must be between 0.1 and 0.9",
    config_key="TOKEN_COMPRESSION_TARGET",
    expected_type="float",
    provided_value="1.5"
)
```

### TimeoutError

Operation timeout and deadline exceeded errors.

**Location**: `src/core/exceptions.py:622-663`

**Timeout Context**:
- `operation: str` - Operation that timed out
- `timeout_seconds: float` - Configured timeout duration
- `elapsed_seconds: float` - Actual elapsed time

**Default**: `retryable=True`

### InsufficientDataError

Insufficient data for meaningful analysis errors.

**Location**: `src/core/exceptions.py:665-703`

**Data Context**:
- `data_type: str` - Type of insufficient data
- `required_minimum: int` - Minimum required count
- `actual_count: int` - Actual count available

**Usage Example**:
```python
raise InsufficientDataError(
    "Not enough search results for analysis",
    data_type="search_results",
    required_minimum=3,
    actual_count=1,
    user_message="Please try a different query or increase search limits"
)
```

## Business Logic Exceptions

### BrandAnalysisError

Core brand visibility analysis pipeline errors.

**Location**: `src/core/exceptions.py:706-748`

**Analysis Context**:
- `analysis_stage: str` - Stage where failure occurred
- `brand_name: str` - Brand being analyzed
- `query: str` - Analysis query
- `partial_results: Dict` - Any partial results generated

**Analysis Stages**:
- "search" - Web search operations
- "compression" - Content optimization
- "llm_processing" - Language model analysis
- "extraction" - Information extraction
- "metrics_calculation" - Visibility metrics

### InvalidBrandError

Brand validation and format errors.

**Location**: `src/core/exceptions.py:750-791`

**Brand Context**:
- `brand_name: str` - Invalid brand name
- `brand_domain: str` - Invalid brand domain
- `validation_issue: str` - Specific validation problem

**Default**: `retryable=False`

### SearchStrategyError

Search query generation and strategy errors.

**Location**: `src/core/exceptions.py:793-831`

**Strategy Context**:
- `strategy: str` - Search strategy that failed
- `query: str` - Generated search query
- `search_depth: str` - Search depth setting

## Utility Functions

### classify_error_for_retry()

Determines if an error should be retried.

**Location**: `src/core/exceptions.py:834-854`

```python
def classify_error_for_retry(error: Exception) -> bool:
    """Classify whether an error should be retried."""
```

**Logic**:
1. BrandLens errors: Use `error.retryable` attribute
2. Standard errors: Classify based on type
3. Retryable types: `ConnectionError`, `TimeoutError`, `OSError`

**Usage Example**:
```python
try:
    # Operation that might fail
    result = api_call()
except Exception as e:
    if classify_error_for_retry(e):
        # Implement retry logic
        retry_operation()
    else:
        # Log and abort
        logger.error(f"Non-retryable error: {e}")
        raise
```

### get_user_friendly_message()

Extracts user-appropriate error messages.

**Location**: `src/core/exceptions.py:857-878`

```python
def get_user_friendly_message(error: Exception) -> str:
    """Get a user-friendly error message."""
```

**Fallback Messages**:
- `ConnectionError`: "Unable to connect to the service..."
- `TimeoutError`: "The operation took too long..."
- `PermissionError`: "Permission denied..."
- `FileNotFoundError`: "Required file not found..."

### create_error_context()

Creates standardized error context information.

**Location**: `src/core/exceptions.py:881-901`

```python
def create_error_context(
    operation: str,
    **additional_context: Any,
) -> Dict[str, Any]:
    """Create standardized error context."""
```

**Standard Context**:
- `operation`: Operation being performed
- `timestamp`: Error occurrence time
- `python_traceback`: Stack trace information

## Error Handling Patterns

### Retry Pattern with Classification

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=classify_error_for_retry
)
async def resilient_api_call():
    try:
        return await api_operation()
    except GeminiAPIError as e:
        if e.retryable:
            raise  # Will be retried
        else:
            # Log non-retryable error and re-raise
            logger.error(f"Non-retryable Gemini error: {e}")
            raise
```

### Context-Rich Error Creation

```python
def process_brand_analysis(brand_name: str, query: str):
    try:
        # Analysis operations
        return perform_analysis(brand_name, query)
    except Exception as e:
        context = create_error_context(
            operation="brand_analysis",
            brand_name=brand_name,
            query=query,
            analysis_stage="processing"
        )

        raise BrandAnalysisError(
            f"Brand analysis failed for {brand_name}",
            analysis_stage="processing",
            brand_name=brand_name,
            query=query,
            context=context,
            cause=e,
            retryable=classify_error_for_retry(e)
        )
```

### User-Friendly Error Display

```python
def handle_cli_error(error: Exception):
    """Handle errors in CLI with user-friendly messages."""
    user_message = get_user_friendly_message(error)

    if isinstance(error, BrandLensError):
        # Rich error information available
        console.print(f"[red]Error:[/red] {user_message}")

        if error.retryable:
            console.print("[yellow]Tip:[/yellow] This error might resolve if you try again.")

        # Log technical details separately
        logger.error(f"Technical details: {error.to_dict()}")
    else:
        # Simple error handling
        console.print(f"[red]Error:[/red] {user_message}")
        logger.error(f"Unexpected error: {error}", exc_info=True)
```

### Error Logging Integration

```python
import structlog

logger = structlog.get_logger()

def log_brandlens_error(error: BrandLensError):
    """Log BrandLens error with structured information."""
    logger.error(
        "BrandLens operation failed",
        error_type=error.__class__.__name__,
        error_code=error.error_code,
        message=error.message,
        retryable=error.retryable,
        context=error.context,
        timestamp=error.timestamp.isoformat(),
        cause=str(error.cause) if error.cause else None
    )
```

## Testing Exception Handling

### Exception Testing Utilities

```python
import pytest

def test_gemini_api_error_creation():
    """Test GeminiAPIError creation and attributes."""
    error = GeminiAPIError(
        "Model unavailable",
        model="gemini-1.5-flash",
        status_code=503,
        retryable=True
    )

    assert error.service == "gemini"
    assert error.model == "gemini-1.5-flash"
    assert error.status_code == 503
    assert error.retryable is True
    assert "gemini-1.5-flash" in str(error)

def test_error_serialization():
    """Test error dictionary serialization."""
    error = BrandLensError(
        "Test error",
        error_code="TEST_ERROR",
        context={"test": "value"}
    )

    error_dict = error.to_dict()
    assert error_dict["error_type"] == "BrandLensError"
    assert error_dict["error_code"] == "TEST_ERROR"
    assert error_dict["context"]["test"] == "value"

def test_retry_classification():
    """Test error retry classification."""
    # Retryable errors
    timeout_error = TimeoutError("Operation timed out", retryable=True)
    assert classify_error_for_retry(timeout_error) is True

    # Non-retryable errors
    config_error = ConfigurationError("Invalid config", retryable=False)
    assert classify_error_for_retry(config_error) is False
```

### Mock Error Scenarios

```python
@pytest.fixture
def mock_gemini_error():
    """Mock Gemini API error for testing."""
    return GeminiAPIError(
        "Content filtering triggered",
        model="gemini-1.5-flash",
        prompt_tokens=150,
        status_code=400,
        retryable=False,
        user_message="Content cannot be processed"
    )

def test_error_handling_with_mock(mock_gemini_error):
    """Test error handling with mocked errors."""
    with pytest.raises(GeminiAPIError) as exc_info:
        raise mock_gemini_error

    error = exc_info.value
    assert error.model == "gemini-1.5-flash"
    assert not error.retryable
```

## Production Monitoring

### Error Metrics Collection

```python
from prometheus_client import Counter, Histogram

# Error counters by type
error_counter = Counter(
    'brandlens_errors_total',
    'Total BrandLens errors',
    ['error_type', 'retryable', 'service']
)

# Error handling duration
error_handling_duration = Histogram(
    'brandlens_error_handling_seconds',
    'Time spent handling errors'
)

def track_error_metrics(error: BrandLensError):
    """Track error metrics for monitoring."""
    error_counter.labels(
        error_type=error.__class__.__name__,
        retryable=str(error.retryable).lower(),
        service=getattr(error, 'service', 'unknown')
    ).inc()
```

### Health Check Integration

```python
def system_health_check():
    """Check system health based on recent errors."""
    recent_errors = get_recent_errors(minutes=5)

    critical_errors = [
        e for e in recent_errors
        if isinstance(e, (ConfigurationError, AuthenticationError))
    ]

    if critical_errors:
        return {
            "status": "unhealthy",
            "reason": "Critical configuration or authentication errors",
            "error_count": len(critical_errors)
        }

    return {"status": "healthy"}
```

---

**Exception System Version**: 1.0
**Last Updated**: 2025-01-29
**Error Handling Standard**: Production-ready
**Monitoring Integration**: Prometheus-compatible