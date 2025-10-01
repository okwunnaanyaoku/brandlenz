# Core Data Models Documentation

## Overview

This document provides comprehensive documentation for all data models in BrandLens. These Pydantic-based models define the structure and validation for all data flowing through the system.

**File Reference**: `src/core/models.py`

## Primary Models

### BrandAnalysis

The main output model containing complete brand visibility analysis results.

**Location**: `src/core/models.py:446-593`

```python
class BrandAnalysis(BaseModel):
```

**Purpose**: Aggregates all analysis components including LLM responses, citations, mentions, and calculated metrics into a single comprehensive result.

**Key Fields**:

#### Core Response Data
- `human_response_markdown: str` - The generated markdown response analyzing the brand
  - **Validation**: Minimum length 1 character
  - **Usage**: Primary user-facing analysis content

- `citations: List[Citation]` - All citations extracted from the response
  - **Default**: Empty list
  - **Purpose**: References to external sources supporting claims

- `mentions: List[Mention]` - All brand mentions detected in the response
  - **Default**: Empty list
  - **Purpose**: Track brand presence and context

#### Source Categorization
- `owned_sources: List[str]` - URLs from brand-owned domains
  - **Purpose**: Track brand's own content presence
  - **Validation**: Must be valid URLs

- `sources: List[str]` - URLs from external sources
  - **Purpose**: Track third-party content references

#### Analysis Context
- `brand_name: str` - The brand being analyzed
  - **Validation**: 1-200 characters
  - **Required**: Yes

- `brand_domain: str` - The primary domain for the brand
  - **Validation**: Must be valid domain format
  - **Processing**: Automatically normalized (removes protocol, converts to lowercase)

- `query: str` - The original query that prompted this analysis
  - **Validation**: Minimum 1 character
  - **Purpose**: Preserves analysis context

- `created_at: datetime` - When this analysis was created
  - **Default**: Current UTC time
  - **Format**: ISO 8601 with timezone

#### Metadata
- `metadata: Dict[str, Any]` - Processing metadata including costs and performance
  - **Contains**: Response times, token counts, API costs, compression ratios
  - **Purpose**: Technical analysis metrics

- `advanced_metrics: Dict[str, Any]` - Advanced visibility and competitive metrics
  - **Contains**: Share of voice, sentiment analysis, competitive positioning
  - **Purpose**: Business intelligence metrics

**Computed Properties**:

```python
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
```

**Methods**:
- `to_json_file(filepath: str)` - Export analysis to JSON file
- `from_json_file(filepath: str)` - Load analysis from JSON file
- `validate_domain(domain: str)` - Domain format validation

### Citation

Represents a citation found in LLM response linking to external sources.

**Location**: `src/core/models.py:309-390`

```python
class Citation(BaseModel):
```

**Purpose**: Captures references to external sources with validation and metadata.

**Fields**:
- `text: str` - Citation marker text (e.g., "[1]", "[Apple]")
  - **Validation**: 1-50 characters
  - **Examples**: "[1]", "[a]", "[Apple Inc]"

- `url: str` - Resolved URL for the citation
  - **Validation**: Must be valid HTTP/HTTPS URL
  - **Processing**: Automatically normalized and validated

- `entity: str` - Associated brand or entity name
  - **Validation**: 1-200 characters
  - **Purpose**: Links citation to specific entities

- `confidence: float` - Confidence score for the citation match
  - **Range**: 0.0 to 1.0
  - **Purpose**: Quality assessment of extraction

- `context: str` - Surrounding text context (up to 200 characters)
  - **Purpose**: Provides context for citation relevance
  - **Validation**: Maximum 500 characters

**Computed Properties**:
```python
@computed_field
@property
def domain(self) -> str:
    """Extract domain from URL."""
```

**Methods**:
- `is_brand_owned(brand_domain: str) -> bool` - Check if citation links to brand's domain
- `normalize_url(url: str) -> str` - URL normalization and validation

### Mention

Represents a brand mention found in content with context and confidence scoring.

**Location**: `src/core/models.py:231-308`

```python
class Mention(BaseModel):
```

**Purpose**: Tracks brand references with fuzzy matching and confidence assessment.

**Fields**:
- `text: str` - Actual mention text as found
  - **Validation**: 1-200 characters
  - **Examples**: "Apple", "Apple's", "AAPL"

- `type: MentionType` - Whether mention is LINKED or UNLINKED
  - **LINKED**: Associated with a citation
  - **UNLINKED**: Standalone reference
  - **Purpose**: Distinguish sourced vs unsourced mentions

- `position: int` - Character position in source text
  - **Validation**: Non-negative integer
  - **Purpose**: Locate mention within content

- `context: str` - Surrounding context (up to 100 characters)
  - **Purpose**: Semantic context for relevance assessment
  - **Validation**: Maximum 300 characters

- `confidence: float` - Match confidence score
  - **Range**: 0.0 to 1.0
  - **Purpose**: Quality assessment of mention detection

**Computed Properties**:
```python
@computed_field
@property
def context_length(self) -> int:
    """Get length of context string."""
```

**Validation Rules**:
- Position must be non-negative
- Confidence must be between 0.0 and 1.0
- Text cannot be empty

## Supporting Models

### PerformanceMetrics

Tracks system performance and resource utilization.

**Location**: `src/core/models.py:595-628`

**Fields**:
- `total_time_ms: int` - Total processing time in milliseconds
- `search_time_ms: int` - Time spent on search operations
- `llm_time_ms: int` - Time spent on LLM processing
- `extraction_time_ms: int` - Time spent on information extraction
- `total_tokens: int` - Total tokens processed
- `total_cost_usd: float` - Total API costs in USD
- `compression_ratio: float` - Content compression achieved (0.0-1.0)
- `quality_score: float` - Overall quality assessment (0.0-1.0)

### SearchResult

Represents results from web search operations.

**Location**: `src/core/models.py:155-183`

**Fields**:
- `url: str` - Source URL
- `title: str` - Page title
- `content: str` - Extracted content
- `score: float` - Relevance score
- `published_date: Optional[datetime]` - Publication date if available

### LLMResponse

Captures structured response from language model.

**Location**: `src/core/models.py:185-229`

**Fields**:
- `content: str` - Generated response content
- `model: str` - Model identifier used
- `tokens_used: int` - Token consumption
- `cost_usd: float` - API cost for this response
- `response_time_ms: int` - Processing time
- `metadata: Dict[str, Any]` - Additional model-specific data

### CompressedContent

Result of content optimization operations.

**Location**: `src/core/models.py:391-445`

**Fields**:
- `original_content: str` - Original uncompressed content
- `compressed_content: str` - Optimized content
- `compression_ratio: float` - Ratio achieved (0.0-1.0)
- `quality_score: float` - Quality preservation score
- `method: CompressionMethod` - Algorithm used
- `chunks_selected: int` - Number of content chunks retained
- `chunks_total: int` - Total chunks available

## Configuration Models

### AppConfig

Main application configuration container.

**Location**: `src/core/models.py:729-755`

**Fields**:
- `api: APIConfig` - External API settings
- `cache: CacheConfig` - Caching configuration
- `compression_ratio: float` - Default compression target
- `compression_target_tokens: int` - Target token count after compression

### APIConfig

External API configuration and credentials.

**Location**: `src/core/models.py:669-706`

**Fields**:
- `gemini_api_key: str` - Google Gemini API key (sensitive)
- `tavily_api_key: str` - Tavily Search API key (sensitive)
- `gemini_model: str` - Gemini model identifier
- `gemini_max_tokens: int` - Maximum tokens per request
- `gemini_temperature: float` - Model temperature setting (0.0-2.0)
- `tavily_search_depth: SearchDepth` - Search depth level
- `tavily_max_results: int` - Maximum search results

### CacheConfig

Caching system configuration.

**Location**: `src/core/models.py:708-727`

**Fields**:
- `cache_dir: str` - Cache directory path
- `cache_ttl: int` - Time to live in seconds
- `cache_enabled: bool` - Enable/disable caching
- `max_cache_size: int` - Maximum cache entries

## Enumeration Types

### MentionType

Categorizes mention linkage status.

**Location**: `src/core/models.py:25-29`

**Values**:
- `LINKED` - Mention associated with citation
- `UNLINKED` - Standalone mention without citation

### ModelName

Supported language model identifiers.

**Location**: `src/core/models.py:32-37`

**Values**:
- `GEMINI_FLASH` - Gemini 1.5 Flash (cost-optimized)
- `GEMINI_PRO` - Gemini 1.5 Pro (performance-optimized)

### SearchDepth

Search operation depth levels.

**Location**: `src/core/models.py:40-45`

**Values**:
- `BASIC` - Basic search depth
- `ADVANCED` - Advanced search with additional processing

### CompressionMethod

Content compression algorithm types.

**Location**: `src/core/models.py:48-53`

**Values**:
- `SEMANTIC` - Semantic-based compression
- `TOKEN_LIMIT` - Token-count-based compression
- `HYBRID` - Combined approach

## Validation Patterns

### URL Validation

All URL fields use comprehensive validation:
```python
@field_validator("url")
@classmethod
def validate_url(cls, v: str) -> str:
    """Validate and normalize URLs."""
    # Must start with http:// or https://
    # Must have valid domain structure
    # Automatically adds https:// if missing
```

### Domain Validation

Brand domains are normalized and validated:
```python
@field_validator("brand_domain")
@classmethod
def validate_domain(cls, v: str) -> str:
    """Validate domain format."""
    # Removes protocol prefixes
    # Validates domain structure
    # Converts to lowercase
```

### Confidence Score Validation

All confidence scores are constrained:
```python
confidence: float = Field(
    ...,
    ge=0.0,
    le=1.0,
    description="Confidence score between 0.0 and 1.0"
)
```

## Usage Patterns

### Creating Analysis Results

```python
from src.core.models import BrandAnalysis, Citation, Mention

# Create analysis with required fields
analysis = BrandAnalysis(
    human_response_markdown="Analysis content...",
    brand_name="Apple",
    brand_domain="apple.com",
    query="What are the latest iPhone features?",
    citations=[
        Citation(
            text="[1]",
            url="https://apple.com/newsroom/...",
            entity="Apple",
            confidence=0.95,
            context="Apple announced new features..."
        )
    ],
    mentions=[
        Mention(
            text="Apple",
            type=MentionType.LINKED,
            position=145,
            context="...latest Apple innovations...",
            confidence=1.0
        )
    ]
)

# Access computed properties
print(f"Total citations: {analysis.total_citations}")
print(f"Visibility summary: {analysis.visibility_summary}")

# Export to JSON
analysis.to_json_file("analysis_results.json")
```

### Model Validation

```python
from pydantic import ValidationError

try:
    analysis = BrandAnalysis(
        human_response_markdown="",  # Invalid: empty string
        brand_name="Apple",
        brand_domain="invalid-domain",  # Invalid: missing TLD
        query="test"
    )
except ValidationError as e:
    print(f"Validation errors: {e}")
```

### Configuration Loading

```python
from src.core.models import AppConfig, APIConfig, CacheConfig

config = AppConfig(
    api=APIConfig(
        gemini_api_key="your-key-here",
        tavily_api_key="your-key-here",
        gemini_model="gemini-1.5-flash"
    ),
    cache=CacheConfig(
        cache_dir="/tmp/brandlens-cache",
        cache_enabled=True
    ),
    compression_ratio=0.25
)
```

## Error Handling

### Common Validation Errors

1. **Invalid URLs**: Must be well-formed HTTP/HTTPS URLs
2. **Empty Required Fields**: Brand name, query cannot be empty
3. **Invalid Confidence Scores**: Must be between 0.0 and 1.0
4. **Invalid Domain Format**: Must be valid domain structure
5. **Field Length Violations**: Text fields have maximum lengths

### Error Messages

The models provide descriptive error messages:
```python
# Example validation error
{
    "type": "string_too_short",
    "loc": ("human_response_markdown",),
    "msg": "String should have at least 1 character",
    "input": ""
}
```

## Performance Considerations

### Model Serialization

- JSON serialization excludes computed fields to avoid circular dependencies
- Large content fields can impact serialization performance
- Use streaming for large datasets

### Memory Usage

- Citation and mention lists can grow large for comprehensive analyses
- Consider pagination for UI display
- Metadata dictionaries are flexible but can consume memory

### Validation Overhead

- URL validation involves network-like parsing
- Domain validation includes format checking
- Confidence score validation is lightweight

## Testing Support

### Model Factories

```python
def create_test_analysis() -> BrandAnalysis:
    """Factory for test analysis objects."""
    return BrandAnalysis(
        human_response_markdown="Test analysis content",
        brand_name="TestBrand",
        brand_domain="testbrand.com",
        query="Test query"
    )

def create_test_citation() -> Citation:
    """Factory for test citation objects."""
    return Citation(
        text="[1]",
        url="https://example.com",
        entity="TestBrand",
        confidence=0.9,
        context="Test context"
    )
```

### Validation Testing

```python
import pytest
from pydantic import ValidationError

def test_citation_validation():
    """Test citation field validation."""
    with pytest.raises(ValidationError):
        Citation(
            text="",  # Invalid: empty
            url="invalid-url",  # Invalid: not HTTP/HTTPS
            entity="Brand",
            confidence=1.5,  # Invalid: > 1.0
            context="Context"
        )
```

---

**Model Version**: 1.0
**Last Updated**: 2025-01-29
**Pydantic Version**: 2.0+
**Python Compatibility**: 3.11+