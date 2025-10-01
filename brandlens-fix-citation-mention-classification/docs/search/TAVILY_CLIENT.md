# Tavily Client Documentation

## Overview

The TavilyClient provides async integration with Tavily's search API, featuring comprehensive cost tracking, intelligent caching, retry logic, and error handling. It's designed for production use with real API integration and budget management.

**File Reference**: `src/search/tavily_client.py`

## Core Architecture

### TavilyClient Class

**Location**: `src/search/tavily_client.py:74-259`

```python
class TavilyClient:
    """Async wrapper around Tavily's REST API."""
```

**Purpose**: Provides robust, production-ready integration with Tavily search API including cost tracking, caching, rate limiting, and error handling.

**Key Features**:
- Asynchronous HTTP client with connection pooling
- Intelligent response caching with configurable TTL
- Comprehensive retry logic with exponential backoff
- Real-time cost tracking and budget management
- Rate limit handling with proper backoff
- Request/response metadata extraction
- Error mapping and classification

## Client Initialization

### Constructor

**Location**: `src/search/tavily_client.py:77-88`

```python
def __init__(
    self,
    settings: TavilyClientSettings,
    *,
    client: Optional[httpx.AsyncClient] = None,
    cache_enabled: Optional[bool] = None,
) -> None:
```

**Parameters**:
- `settings`: Client configuration including API key, timeouts, retry settings
- `client`: Optional pre-configured httpx client (for dependency injection)
- `cache_enabled`: Override cache setting from configuration

**Internal State**:
- `_settings`: Configuration object
- `_client`: httpx AsyncClient instance
- `_owns_client`: Whether client manages the httpx instance lifecycle
- `_cache_enabled`: Cache configuration flag
- `_cache`: In-memory cache dictionary for responses

### Factory Method

**Location**: `src/search/tavily_client.py:90-98`

```python
@classmethod
def from_config(
    cls,
    config: APIConfig,
    *,
    client: Optional[httpx.AsyncClient] = None,
    enable_cache: bool = False,
) -> "TavilyClient":
```

**Purpose**: Creates TavilyClient from application configuration.

**Usage Example**:
```python
from src.config import load_app_config

config = load_app_config()
async with TavilyClient.from_config(config.api, enable_cache=True) as client:
    results = await client.search("Apple iPhone features")
```

## Context Manager Protocol

### Async Context Manager

**Entry**: `src/search/tavily_client.py:100-102`
**Exit**: `src/search/tavily_client.py:104-105`

```python
async def __aenter__(self) -> "TavilyClient":
    await self._ensure_client()
    return self

async def __aexit__(self, exc_type, exc, tb) -> None:
    await self.aclose()
```

**Benefits**:
- Automatic HTTP client lifecycle management
- Guaranteed resource cleanup
- Exception-safe connection handling

**Resource Management**:

```python
async def aclose(self) -> None:
    if self._client and self._owns_client:
        await self._client.aclose()
        self._client = None
```

## Core API Methods

### Search Operation

**Location**: `src/search/tavily_client.py:115-140`

```python
async def search(
    self,
    query: str,
    *,
    depth: SearchDepth = SearchDepth.ADVANCED,
    max_results: int = 10,
    include_raw_content: Optional[bool] = None,
) -> TavilySearchResponse:
```

**Parameters**:
- `query`: Search query string
- `depth`: Search depth level (BASIC or ADVANCED)
- `max_results`: Maximum number of results to return
- `include_raw_content`: Whether to include full page content

**Returns**: `TavilySearchResponse` with search results and metadata

**Features**:
- Automatic payload construction and validation
- Response parsing and model validation
- Cost and performance metadata extraction
- Error handling and retry logic

**Usage Example**:
```python
# Basic search
response = await client.search("Tesla electric vehicles")

# Advanced search with full content
response = await client.search(
    "Apple iPhone 15 features",
    depth=SearchDepth.ADVANCED,
    max_results=5,
    include_raw_content=True
)

# Access results
for result in response.results:
    print(f"Title: {result.title}")
    print(f"URL: {result.url}")
    print(f"Content: {result.content[:200]}...")
```

### Content Retrieval

**Location**: `src/search/tavily_client.py:142-149`

```python
async def get_content(
    self,
    url: str,
    *,
    source_query: Optional[str] = None
) -> TavilyContentResponse:
```

**Purpose**: Retrieves full content from a specific URL with Tavily's content extraction.

**Parameters**:
- `url`: Target URL for content extraction
- `source_query`: Optional query context for relevance scoring

**Features**:
- Clean content extraction without ads/navigation
- Structured content parsing
- Relevance scoring when query provided

## Caching System

### Cache Implementation

**Cache Structure**:
```python
_cache: Dict[Tuple[str, str], Tuple[Dict[str, Any], Dict[str, Any]]]
```

**Cache Key**: Tuple of (API endpoint path, JSON-serialized payload)

**Cache Management**:

```python
def clear_cache(self) -> None:
    """Clear all cached responses."""
    self._cache.clear()
```

**Cache Logic**:
1. **Cache Check**: Before API call, check if identical request cached
2. **Cache Hit**: Return deep copy of cached response and metadata
3. **Cache Miss**: Make API call, cache response if caching enabled
4. **Cache Storage**: Store deep copies to prevent mutation

**Cache Benefits**:
- Reduces API costs for repeated queries
- Improves response times for cached requests
- Prevents rate limit exhaustion
- Enables offline development/testing

**Usage Example**:
```python
# Enable caching
client = TavilyClient.from_config(config, enable_cache=True)

# First call - hits API
response1 = await client.search("Apple iPhone")  # API call

# Second identical call - hits cache
response2 = await client.search("Apple iPhone")  # Cached response

# Clear cache if needed
client.clear_cache()
```

## HTTP Request Management

### Core Request Method

**Location**: `src/search/tavily_client.py:151-216`

```python
async def _post_json(
    self,
    path: str,
    payload: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
```

**Purpose**: Handles all HTTP communication with comprehensive error handling and retry logic.

**Request Flow**:
1. **Client Initialization**: Ensure HTTP client is ready
2. **Cache Check**: Check for cached response
3. **Request Construction**: Build headers and payload
4. **Retry Logic**: Execute with exponential backoff
5. **Response Processing**: Parse and validate response
6. **Metadata Extraction**: Extract cost and rate limit info
7. **Cache Storage**: Store successful responses
8. **Error Handling**: Map HTTP errors to domain exceptions

**Authentication**:
```python
headers = {"Authorization": f"Bearer {self._settings.api_key}"}
```

**Retry Configuration**:
```python
async for attempt in AsyncRetrying(
    reraise=True,
    retry=retry_if_exception_type(TavilyAPIError),
    wait=wait_exponential(min=1, max=5),
    stop=stop_after_attempt(self._settings.max_attempts),
):
```

**Error Scenarios Handled**:
- Network connectivity issues
- Rate limiting (429 responses)
- Server errors (5xx responses)
- Invalid JSON responses
- Authentication failures

### HTTP Client Management

**Location**: `src/search/tavily_client.py:218-220`

```python
async def _ensure_client(self) -> None:
    if self._client is None:
        self._client = httpx.AsyncClient(base_url=self._settings.base_url)
```

**Features**:
- Lazy HTTP client initialization
- Base URL configuration
- Connection pooling and reuse
- Timeout configuration
- SSL verification

## Metadata Extraction

### Response Metadata

**Location**: `src/search/tavily_client.py:222-247`

```python
def _extract_metadata(self, response: httpx.Response) -> Dict[str, Any]:
```

**Extracted Metadata**:

#### Cost Tracking
- `X-Tavily-Cost`: API call cost in USD
- Parsed as float with fallback to 0.0
- Used for budget management and cost reporting

#### Rate Limiting
- `X-RateLimit-Remaining`: Remaining requests in current window
- Used for proactive rate limit management
- Enables intelligent request pacing

#### Request Tracking
- `X-Request-Id`: Unique request identifier
- Used for debugging and support
- Enables request correlation across logs

**Parsing Logic**:
```python
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
```

## Error Handling

### Error Mapping

**Location**: `src/search/tavily_client.py:249-259`

```python
def _map_error(self, response: httpx.Response) -> TavilyAPIError:
```

**Error Processing**:
1. **JSON Parsing**: Attempt to parse error response as JSON
2. **Message Extraction**: Extract error message from multiple possible fields
3. **Context Preservation**: Include HTTP status code and response data
4. **Exception Creation**: Create appropriate TavilyAPIError

**Error Message Priority**:
1. `error` field in JSON response
2. `message` field in JSON response
3. Raw HTTP response text

**Error Classifications**:

#### Rate Limiting (429)
```python
if response.status_code == 429:
    raise TavilyAPIError(
        "Tavily rate limit hit",
        status_code=response.status_code,
        response_data=payload_dict,
    )
```

#### Client Errors (4xx)
- Authentication failures (401)
- Invalid request format (400)
- Not found errors (404)

#### Server Errors (5xx)
- Internal server errors (500)
- Service unavailable (503)
- Gateway timeouts (504)

### Network Error Handling

**Network Exception Mapping**:
```python
except httpx.RequestError as exc:
    raise TavilyAPIError("Tavily request failed", status_code=None, cause=exc) from exc
```

**Handled Network Issues**:
- Connection timeouts
- DNS resolution failures
- SSL/TLS errors
- Connection refused
- Network unreachable

## Configuration

### TavilyClientSettings

**Configuration Fields**:
- `api_key`: Tavily API authentication key
- `base_url`: API endpoint base URL
- `timeout`: Request timeout in seconds
- `max_attempts`: Maximum retry attempts
- `include_raw_content`: Default content inclusion setting
- `enable_cache`: Cache configuration

**Configuration Example**:
```python
settings = TavilyClientSettings(
    api_key="tvly-your-api-key-here",
    base_url="https://api.tavily.com",
    timeout=30.0,
    max_attempts=3,
    include_raw_content=True,
    enable_cache=True
)
```

## Response Models

### TavilySearchResponse

**Structure**:
```python
@dataclass
class TavilySearchResponse:
    query: str
    results: List[TavilySearchResult]
    follow_up_questions: List[str]
    total_results: Optional[int]
    metadata: TavilyResponseMetadata
```

### TavilySearchResult

**Structure**:
```python
@dataclass
class TavilySearchResult:
    url: str
    title: str
    content: str
    score: float
    published_date: Optional[str]
```

### TavilyResponseMetadata

**Structure**:
```python
@dataclass
class TavilyResponseMetadata:
    request_id: Optional[str]
    cost_usd: float
    rate_limit_remaining: Optional[int]
```

## Usage Patterns

### Basic Search Pattern

```python
from src.search.tavily_client import TavilyClient
from src.config import load_app_config

async def basic_search():
    config = load_app_config()

    async with TavilyClient.from_config(config.api) as client:
        response = await client.search("Tesla Model 3 features")

        print(f"Query: {response.query}")
        print(f"Total results: {response.total_results}")
        print(f"Cost: ${response.metadata.cost_usd:.4f}")

        for result in response.results:
            print(f"- {result.title}")
            print(f"  {result.url}")
            print(f"  Score: {result.score}")
```

### Cached Search Pattern

```python
async def cached_search_pattern():
    config = load_app_config()

    # Enable caching for cost savings
    async with TavilyClient.from_config(
        config.api,
        enable_cache=True
    ) as client:

        # Multiple searches with potential cache hits
        queries = [
            "Apple iPhone 15 features",
            "Tesla Model Y specifications",
            "Apple iPhone 15 features",  # Cache hit
        ]

        total_cost = 0.0
        for query in queries:
            response = await client.search(query)
            total_cost += response.metadata.cost_usd
            print(f"Query: {query} - Cost: ${response.metadata.cost_usd:.4f}")

        print(f"Total cost: ${total_cost:.4f}")
```

### Error Handling Pattern

```python
from src.core.exceptions import TavilyAPIError, RateLimitError

async def robust_search_pattern():
    config = load_app_config()

    async with TavilyClient.from_config(config.api) as client:
        try:
            response = await client.search("complex query")
            return response

        except RateLimitError as e:
            print(f"Rate limited. Wait {e.retry_after} seconds")
            # Implement backoff logic

        except TavilyAPIError as e:
            if e.status_code == 401:
                print("Authentication failed. Check API key")
            elif e.status_code >= 500:
                print("Server error. Retry later")
            else:
                print(f"API error: {e}")

        except Exception as e:
            print(f"Unexpected error: {e}")
            # Log and handle appropriately
```

### Budget-Aware Search Pattern

```python
async def budget_aware_search():
    config = load_app_config()
    max_budget = 0.05  # $0.05 budget
    current_cost = 0.0

    async with TavilyClient.from_config(config.api) as client:
        queries = ["query1", "query2", "query3"]

        for query in queries:
            if current_cost >= max_budget:
                print("Budget exceeded, stopping searches")
                break

            response = await client.search(query)
            current_cost += response.metadata.cost_usd

            print(f"Query: {query}")
            print(f"Cost: ${response.metadata.cost_usd:.4f}")
            print(f"Remaining budget: ${max_budget - current_cost:.4f}")
```

## Performance Considerations

### Connection Pooling

- HTTP client reuses connections across requests
- Reduces connection establishment overhead
- Improves throughput for multiple requests

### Caching Strategy

- In-memory cache for development and testing
- Deep copying prevents cache pollution
- Cache key includes full request context
- Manual cache clearing for testing

### Rate Limit Management

- Proactive rate limit checking via headers
- Exponential backoff on rate limit errors
- Request pacing based on remaining quota

### Memory Management

- Async context manager ensures cleanup
- HTTP client properly closed on exit
- Cache can be manually cleared if needed

## Testing Support

### Mock Client Pattern

```python
from unittest.mock import AsyncMock

def test_search_operation():
    mock_client = AsyncMock(spec=TavilyClient)
    mock_response = TavilySearchResponse(
        query="test query",
        results=[],
        follow_up_questions=[],
        total_results=0,
        metadata=TavilyResponseMetadata(
            request_id="test-123",
            cost_usd=0.001,
            rate_limit_remaining=99
        )
    )
    mock_client.search.return_value = mock_response

    # Test code using mock_client
```

### Integration Testing

```python
import pytest

@pytest.mark.api
async def test_real_tavily_search():
    """Integration test with real Tavily API."""
    config = load_test_config()

    async with TavilyClient.from_config(config.api) as client:
        response = await client.search("test query", max_results=1)

        assert response.query == "test query"
        assert len(response.results) <= 1
        assert response.metadata.cost_usd >= 0
```

## Monitoring and Logging

### Request Logging

```python
# Successful requests
LOGGER.info(
    "Tavily request succeeded",
    extra={"path": path, "request_id": meta["request_id"], "cost_usd": meta["cost_usd"]}
)

# Rate limiting
LOGGER.warning(
    "Tavily rate limit hit",
    extra={"path": path, "rate_limit_remaining": meta["rate_limit_remaining"]}
)
```

### Metrics Collection

```python
# Cost tracking
total_cost = sum(response.metadata.cost_usd for response in responses)

# Request counting
request_count = len(responses)

# Cache hit rate
cache_hits = sum(1 for response in responses if response.cached)
cache_hit_rate = cache_hits / len(responses) if responses else 0
```

---

**Client Version**: 1.0
**Last Updated**: 2025-01-29
**API Compatibility**: Tavily REST API v1
**Python Compatibility**: 3.11+