# Configuration System Documentation

## Overview

BrandLens uses a comprehensive configuration system based on environment variables with Pydantic validation. The system supports multiple configuration sources with clear precedence rules and extensive validation.

**File Reference**: `src/config.py`

## Configuration Architecture

### Configuration Flow

```
Environment Variables → .env File → Pydantic Validation → AppConfig Object
```

### Configuration Sources (Precedence Order)

1. **Command-line environment variables** (highest priority)
2. **System environment variables**
3. **`.env` file variables**
4. **Default values** (lowest priority)

### Main Configuration Function

**Location**: `src/config.py:225-244`

```python
def load_app_config(
    env: Optional[Mapping[str, str]] = None,
    *,
    env_file: Optional[str] = None,
) -> AppConfig:
    """Load an AppConfig from environment variables."""
```

**Parameters**:
- `env`: Optional environment mapping (for testing)
- `env_file`: Optional path to .env file (defaults to `.env` in project root)

**Returns**: Validated `AppConfig` object

**Process**:
1. Load .env file if specified
2. Build API configuration from environment
3. Build cache configuration from environment
4. Build application-level configuration
5. Validate complete configuration
6. Return validated config object

## Configuration Models

### AppConfig (Main Configuration Container)

**Location**: `src/core/models.py:729-755`

```python
class AppConfig(BaseModel):
    api: APIConfig
    cache: CacheConfig
    compression_ratio: float = Field(default=0.25, ge=0.1, le=0.9)
    compression_target_tokens: int = Field(default=2000, ge=500, le=10000)
```

**Fields**:
- `api`: External API configuration (Gemini, Tavily)
- `cache`: Caching system configuration
- `compression_ratio`: Target compression ratio (10%-90%)
- `compression_target_tokens`: Target token count after compression

### APIConfig (External API Settings)

**Location**: `src/core/models.py:669-706`

```python
class APIConfig(BaseModel):
```

**Sensitive Fields**:
- `gemini_api_key: str` - Google Gemini API key
- `tavily_api_key: str` - Tavily Search API key

**Gemini Configuration**:
- `gemini_model: str` - Model identifier (default: "models/gemini-2.5-flash")
- `gemini_max_tokens: int` - Maximum tokens per request (default: 8192)
- `gemini_temperature: float` - Model temperature 0.0-2.0 (default: 0.7)
- `gemini_max_retries: int` - Retry attempts (default: 3)

**Tavily Configuration**:
- `tavily_search_depth: SearchDepth` - Search depth level (default: ADVANCED)
- `tavily_include_raw_content: bool` - Include full content (default: True)
- `tavily_max_results: int` - Maximum results per search (default: 10)

### CacheConfig (Caching System Settings)

**Location**: `src/core/models.py:708-727`

```python
class CacheConfig(BaseModel):
```

**Fields**:
- `cache_dir: str` - Cache directory path (default: ".cache")
- `cache_ttl: int` - Time to live in seconds (default: 3600)
- `cache_enabled: bool` - Enable/disable caching (default: True)
- `max_cache_size: int` - Maximum cache entries (default: 1000)

## Environment Variables Reference

### Required Variables

```bash
# CRITICAL: These must be provided
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### Gemini Configuration Variables

```bash
# Model Settings
GEMINI_MODEL=models/gemini-2.5-flash           # Model to use
GEMINI_MAX_TOKENS=8192                         # Max tokens per request
GEMINI_TEMPERATURE=0.7                         # Model temperature (0.0-2.0)
GEMINI_MAX_RETRIES=3                          # Retry attempts

# Rate Limiting
GEMINI_RATE_LIMIT=60                          # Requests per minute
```

### Tavily Configuration Variables

```bash
# Search Settings
TAVILY_SEARCH_DEPTH=advanced                  # basic|advanced
TAVILY_INCLUDE_RAW_CONTENT=true              # Include full content
TAVILY_MAX_RESULTS=10                        # Max results per search

# Rate Limiting
TAVILY_RATE_LIMIT=100                        # Requests per minute
```

### Application Configuration Variables

```bash
# Performance Settings
TOKEN_COMPRESSION_TARGET=0.25                # Compression ratio (0.1-0.9)
MAX_SEARCHES_PER_QUERY=5                    # Max searches per analysis
MAX_SOURCES_PER_SEARCH=10                   # Max sources per search
ASYNC_TIMEOUT=30                            # Async operation timeout (seconds)

# Budget Management
MAX_COST_USD=1.0                            # Maximum cost per analysis
COST_ALERT_THRESHOLD=0.10                   # Cost alerting threshold
```

### Cache Configuration Variables

```bash
# Cache Settings
CACHE_DIR=.cache                            # Cache directory path
CACHE_TTL=3600                              # Time to live (seconds)
CACHE_ENABLED=true                          # Enable caching
MAX_CACHE_SIZE=1000                         # Maximum cached entries
```

### Logging Configuration Variables

```bash
# Logging Settings
LOG_LEVEL=INFO                              # CRITICAL|ERROR|WARNING|INFO|DEBUG
LOG_FILE=brandlens.log                      # Log file path
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Performance Monitoring
ENABLE_PERFORMANCE_TRACKING=true            # Enable performance metrics
ENABLE_COST_TRACKING=true                   # Enable cost tracking
```

### Security Configuration Variables

```bash
# Security Settings
VALIDATE_SSL=true                           # Validate SSL certificates
API_TIMEOUT=30                              # API request timeout (seconds)
MAX_CONTENT_LENGTH=1048576                  # Max content size (bytes)
```

### Development Configuration Variables

```bash
# Development Settings
DEBUG=false                                 # Enable debug mode
ENABLE_API_MOCKING=false                   # Enable API mocking (always false in production)

# Testing Settings
TEST_API_CALLS=false                       # Enable test API calls
TEST_TIMEOUT=60                            # Test timeout (seconds)
TEST_CACHE_DIR=.test_cache                 # Test cache directory
```

## Configuration Validation

### Validation Process

**Location**: `src/config.py:191-223`

```python
def validate_app_config(config: AppConfig) -> None:
    """Validate application configuration."""
```

**Validation Steps**:
1. **API Key Validation**: Ensures keys are present and properly formatted
2. **Model Validation**: Validates Gemini model names and parameters
3. **Range Validation**: Ensures numeric values are within acceptable ranges
4. **Directory Validation**: Validates cache directory accessibility
5. **Cross-field Validation**: Ensures configuration consistency

### Validation Rules

#### API Keys
- Must be non-empty strings
- Must match expected format patterns
- Gemini keys: Must start with "AIzaSy" pattern
- Tavily keys: Must start with "tvly-" pattern

#### Numeric Ranges
- `compression_ratio`: 0.1 to 0.9 (10% to 90%)
- `gemini_temperature`: 0.0 to 2.0
- `gemini_max_tokens`: 1 to 32768
- `cache_ttl`: 1 to 86400 (1 second to 1 day)

#### File Paths
- `cache_dir`: Must be writable directory or creatable
- `log_file`: Must be in writable directory
- All paths support environment variable expansion

### Configuration Errors

**Exception Type**: `ConfigurationError`

**Common Errors**:
1. **Missing API Keys**: Required keys not provided
2. **Invalid Ranges**: Numeric values outside acceptable ranges
3. **Invalid Paths**: Inaccessible directories or files
4. **Invalid Models**: Unrecognized Gemini model names
5. **Type Mismatches**: Wrong data types for configuration values

**Error Messages**:
```python
# Example error messages
"GEMINI_API_KEY is required but not set"
"compression_ratio must be between 0.1 and 0.9"
"cache_dir '/invalid/path' is not writable"
"gemini_model 'invalid-model' is not supported"
```

## Configuration Builder Functions

### Internal Builder Functions

**Location**: `src/config.py:66-189`

#### _build_api_config()

```python
def _build_api_config(env: Mapping[str, str]) -> APIConfig:
    """Build API configuration from environment variables."""
```

**Process**:
1. Extract API keys with validation
2. Build Gemini configuration with defaults
3. Build Tavily configuration with defaults
4. Create and validate APIConfig object

#### _build_cache_config()

```python
def _build_cache_config(env: Mapping[str, str]) -> CacheConfig:
    """Build cache configuration from environment variables."""
```

**Process**:
1. Extract cache directory with expansion
2. Parse TTL and size limits
3. Handle boolean cache_enabled flag
4. Create and validate CacheConfig object

#### _build_app_kwargs()

```python
def _build_app_kwargs(env: Mapping[str, str]) -> Dict[str, Any]:
    """Build application-level configuration arguments."""
```

**Process**:
1. Parse compression settings
2. Extract performance parameters
3. Handle optional advanced settings
4. Return dictionary for AppConfig constructor

## Configuration Usage Patterns

### Basic Configuration Loading

```python
from src.config import load_app_config

# Load with default .env file
config = load_app_config()

# Load with specific .env file
config = load_app_config(env_file="/path/to/custom.env")

# Load with explicit environment
custom_env = {
    "GEMINI_API_KEY": "test-key",
    "TAVILY_API_KEY": "test-key"
}
config = load_app_config(env=custom_env)
```

### Accessing Configuration Values

```python
# API configuration
print(f"Gemini model: {config.api.gemini_model}")
print(f"Max tokens: {config.api.gemini_max_tokens}")
print(f"Search depth: {config.api.tavily_search_depth}")

# Cache configuration
print(f"Cache directory: {config.cache.cache_dir}")
print(f"Cache TTL: {config.cache.cache_ttl}")
print(f"Cache enabled: {config.cache.cache_enabled}")

# Application configuration
print(f"Compression ratio: {config.compression_ratio}")
print(f"Target tokens: {config.compression_target_tokens}")
```

### Configuration Validation

```python
from src.config import load_app_config
from src.core.exceptions import ConfigurationError

try:
    config = load_app_config()
    print("Configuration loaded successfully")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle configuration problems
```

## Environment File Format

### .env File Structure

```bash
# =============================================================================
# BrandLens Configuration File
# =============================================================================

# REQUIRED: API Keys
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Gemini Settings
GEMINI_MODEL=models/gemini-2.5-flash
GEMINI_MAX_TOKENS=8192
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_RETRIES=3

# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================

# Tavily Settings
TAVILY_SEARCH_DEPTH=advanced
TAVILY_INCLUDE_RAW_CONTENT=true
TAVILY_MAX_RESULTS=10

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

# Token Optimization
TOKEN_COMPRESSION_TARGET=0.25
MAX_SEARCHES_PER_QUERY=3
MAX_SOURCES_PER_SEARCH=5

# Timeouts and Limits
ASYNC_TIMEOUT=30
API_TIMEOUT=30
MAX_CONTENT_LENGTH=1048576

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

CACHE_DIR=.cache
CACHE_TTL=3600
CACHE_ENABLED=true
MAX_CACHE_SIZE=1000

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL=INFO
LOG_FILE=brandlens.log
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

ENABLE_PERFORMANCE_TRACKING=true
ENABLE_COST_TRACKING=true
COST_ALERT_THRESHOLD=0.10
```

### Environment File Loading

**Location**: `src/config.py:33-64`

```python
def _load_env_file(env_file: Optional[str] = None) -> None:
    """Load environment variables from .env file."""
```

**Process**:
1. Determine .env file path (default: project root)
2. Check file existence and readability
3. Parse key-value pairs with validation
4. Load into environment with existing variable precedence
5. Handle encoding and special characters

**Supported Formats**:
- Key-value pairs: `KEY=value`
- Comments: `# This is a comment`
- Empty lines: Ignored
- Quoted values: `KEY="value with spaces"`
- Variable expansion: `KEY=${OTHER_KEY}/suffix`

## Testing Configuration

### Test Configuration Utilities

```python
def create_test_config(**overrides) -> AppConfig:
    """Create test configuration with overrides."""
    base_env = {
        "GEMINI_API_KEY": "test-gemini-key",
        "TAVILY_API_KEY": "test-tavily-key",
        "CACHE_ENABLED": "false",  # Disable caching for tests
        "LOG_LEVEL": "DEBUG"
    }
    base_env.update(overrides)
    return load_app_config(env=base_env)

def test_api_config():
    """Test API configuration loading."""
    config = create_test_config(
        GEMINI_MODEL="custom-model",
        GEMINI_MAX_TOKENS="4096"
    )
    assert config.api.gemini_model == "custom-model"
    assert config.api.gemini_max_tokens == 4096
```

### Mock Configuration for Testing

```python
import pytest
from unittest.mock import patch

@pytest.fixture
def mock_config():
    """Provide mock configuration for tests."""
    with patch('src.config.load_app_config') as mock_load:
        mock_config = create_test_config()
        mock_load.return_value = mock_config
        yield mock_config
```

## Configuration Security

### Sensitive Data Handling

1. **API Keys**: Never logged or displayed in error messages
2. **Environment Isolation**: Test and production configs separated
3. **File Permissions**: .env files should have restricted permissions (600)
4. **Version Control**: .env files excluded from git via .gitignore

### Security Best Practices

```bash
# Set secure file permissions
chmod 600 .env

# Validate .env file is not in version control
echo ".env" >> .gitignore

# Use environment variable validation
# API keys validated for format but never logged
```

### Production Security

```python
# Production configuration checklist:
- API keys from secure environment (not files)
- SSL validation enabled
- Appropriate timeouts configured
- Rate limiting properly set
- Cache in secure directory
- Logging excludes sensitive data
```

## Troubleshooting Configuration

### Common Issues

#### Missing API Keys
**Symptom**: `ConfigurationError: GEMINI_API_KEY is required`
**Solution**: Set API keys in environment or .env file

#### Invalid .env File
**Symptom**: Configuration values not loading from .env
**Solution**: Check file format, permissions, and location

#### Path Issues
**Symptom**: `ConfigurationError: cache_dir '/path' is not writable`
**Solution**: Ensure directory exists and has proper permissions

#### Type Conversion Errors
**Symptom**: `ValidationError: Input should be a valid integer`
**Solution**: Check environment variable format (quotes, spaces)

### Debug Configuration

```python
# Enable debug logging for configuration
import os
os.environ["LOG_LEVEL"] = "DEBUG"

from src.config import load_app_config
config = load_app_config()  # Will show detailed loading information
```

### Configuration Validation CLI

```bash
# Validate current configuration
python -m src validate-config

# Show current configuration (without secrets)
python -m src info
```

---

**Configuration System Version**: 1.0
**Last Updated**: 2025-01-29
**Environment Format**: .env (dotenv)
**Validation Framework**: Pydantic 2.0+