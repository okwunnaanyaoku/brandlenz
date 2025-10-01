# BrandLens System Architecture

## Overview

BrandLens is a CLI-based brand visibility analyzer that measures and analyzes how brands appear in LLM-generated responses. The system follows a modular architecture with clear separation of concerns across seven major subsystems.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Interface                           │
│                      (src/cli.py)                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    Main Analyzer                               │
│                   (src/analyzer.py)                           │
│  - Orchestrates end-to-end analysis pipeline                   │
│  - Manages component integration and error handling            │
└─────────────┬───────────────────────────────────┬─────────────┘
              │                                   │
        ┌─────▼─────┐                       ┌─────▼─────┐
        │  Search   │                       │    LLM    │
        │  System   │                       │ Integration│
        │           │                       │           │
        └─────┬─────┘                       └─────┬─────┘
              │                                   │
    ┌─────────▼─────────┐               ┌─────────▼─────────┐
    │   Optimization    │               │   Information     │
    │     System        │               │   Extraction      │
    │                   │               │                   │
    └───────────────────┘               └───────────────────┘
              │                                   │
    ┌─────────▼─────────┐               ┌─────────▼─────────┐
    │   Cache System    │               │  Output Formatting│
    │                   │               │                   │
    └───────────────────┘               └───────────────────┘
              │
    ┌─────────▼─────────┐
    │ Core Models &     │
    │ Configuration     │
    └───────────────────┘
```

## Core Subsystems

### 1. Core Foundation (`src/core/`)

**Purpose**: Provides fundamental data models, configuration management, and exception hierarchy.

**Key Components**:
- `models.py`: Pydantic data models for all system entities
- `exceptions.py`: Custom exception hierarchy for error handling
- `metrics.py`: Performance and quality metrics calculation
- `__init__.py`: Public API exports

**Dependencies**: External (Pydantic, Python standard library)

### 2. CLI Interface (`src/cli.py`)

**Purpose**: Provides command-line interface for user interaction and system configuration.

**Key Components**:
- Command parsing and validation
- Configuration loading and validation
- Rich terminal output formatting
- Error reporting and user feedback

**Dependencies**: Click, Rich, Core Foundation

### 3. Main Analyzer (`src/analyzer.py`)

**Purpose**: Orchestrates the complete brand visibility analysis pipeline.

**Key Components**:
- `BrandAnalyzer` class: Main orchestration logic
- End-to-end pipeline coordination
- Component integration and error handling
- Performance monitoring and reporting

**Dependencies**: All other subsystems

### 4. Search System (`src/search/`)

**Purpose**: Handles web search operations, budget management, and result orchestration.

**Key Components**:
- `tavily_client.py`: Tavily API integration with cost tracking
- `orchestrator.py`: Multi-strategy search coordination
- `budget.py`: Cost and resource management
- `analytics.py`: Search performance analysis
- `strategies/`: Multiple search strategy implementations

**Dependencies**: Tavily API, Core Foundation, aiohttp

### 5. LLM Integration (`src/llm/`)

**Purpose**: Manages interaction with Gemini LLM for content analysis.

**Key Components**:
- `gemini_client.py`: Gemini API client with streaming support
- `prompts.py`: Prompt engineering and template management
- `response_parser.py`: Structured response extraction and validation

**Dependencies**: Google Generative AI SDK, Core Foundation

### 6. Optimization System (`src/optimization/`)

**Purpose**: Implements token reduction and content optimization strategies.

**Key Components**:
- `semantic_chunker.py`: Brand-aware content chunking and scoring
- `content_compressor.py`: Multi-strategy content compression
- `token_counter.py`: Model-specific token counting and estimation

**Dependencies**: Sentence Transformers, tiktoken, Core Foundation

### 7. Information Extraction (`src/extraction/`)

**Purpose**: Extracts citations, mentions, and entities from LLM responses.

**Key Components**:
- `citation_extractor.py`: Citation detection and URL normalization
- `mention_detector.py`: Brand mention detection with fuzzy matching
- `entity_recognizer.py`: Named entity recognition using spaCy and rules

**Dependencies**: spaCy, difflib, Core Foundation

### 8. Supporting Systems

**Cache System** (`src/cache/`):
- `cache_manager.py`: Multi-tier caching with TTL management

**Utilities** (`src/utils/`):
- `formatters.py`: Rich console output and JSON serialization
- `logger.py`: Structured logging configuration

**Configuration** (`src/config.py`):
- Environment-based configuration loading and validation

## Data Flow Architecture

### 1. Request Processing Flow

```
CLI Input → Configuration Loading → Main Analyzer → Search System
    ↓
Search Results → Optimization System → LLM Integration → Response Processing
    ↓
Information Extraction → Result Aggregation → Output Formatting → CLI Output
```

### 2. Component Interaction Patterns

**Synchronous Operations**:
- Configuration loading and validation
- Token counting and estimation
- Information extraction processing
- Output formatting and display

**Asynchronous Operations**:
- Tavily API calls (search system)
- Gemini API calls (LLM integration)
- Concurrent processing pipelines
- Streaming response handling

## Key Design Principles

### 1. Modularity and Separation of Concerns

Each subsystem has a single, well-defined responsibility:
- Search system handles only web search operations
- LLM integration handles only language model interactions
- Extraction handles only information parsing
- Optimization handles only content reduction

### 2. Dependency Inversion

Higher-level modules depend on abstractions, not concrete implementations:
- Main analyzer depends on interfaces, not specific implementations
- Strategy pattern used for search strategies
- Factory patterns for client instantiation

### 3. Error Isolation and Recovery

Each component implements comprehensive error handling:
- Graceful degradation when external APIs fail
- Retry logic with exponential backoff
- Fallback mechanisms for critical operations
- Detailed error reporting and logging

### 4. Performance Optimization

System-wide focus on performance and cost optimization:
- Async/await patterns for I/O operations
- Intelligent caching at multiple levels
- Token optimization to reduce LLM costs
- Budget management to prevent cost overruns

## Configuration Architecture

### Environment-Based Configuration

```
.env file → AppConfig (Pydantic validation) → Component-specific settings
```

**Configuration Hierarchy**:
1. Default values (in code)
2. Environment variables
3. .env file values
4. Command-line overrides

### Configuration Distribution

Each component receives only its required configuration subset:
- `APIConfig` for external API clients
- `CacheConfig` for caching systems
- Performance settings distributed as needed

## Security Architecture

### 1. API Key Management

- Environment variable storage only
- No hardcoded credentials
- Validation on startup
- Secure transmission (HTTPS only)

### 2. Input Validation

- Pydantic models for all data validation
- Command-line input sanitization
- URL normalization and validation
- Content size limits

### 3. Error Information Disclosure

- Sanitized error messages for users
- Detailed logging for debugging
- No sensitive data in error responses

## Performance Architecture

### 1. Caching Strategy

**Three-Tier Caching**:
- L1: In-memory caching for frequently accessed data
- L2: Disk-based caching for search results
- L3: Database caching for persistent data (future)

### 2. Optimization Pipeline

**Sequential Optimization**:
1. Source limiting at API level (40% token reduction)
2. Semantic compression (25% additional reduction)
3. Quality-preserving content selection
4. Efficient tokenization and processing

### 3. Async Processing

- Concurrent API calls where possible
- Streaming response processing
- Background cache warming
- Parallel extraction processing

## Extensibility Points

### 1. Search Strategy Extension

New search strategies can be added by implementing the base strategy interface:
```python
class CustomStrategy(BaseSearchStrategy):
    def build_query(self, context: SearchStrategyContext) -> str:
        # Custom implementation
```

### 2. LLM Provider Extension

System designed to support multiple LLM providers:
- Abstract base client interface
- Provider-specific implementations
- Configuration-driven provider selection

### 3. Extraction Algorithm Extension

New extraction algorithms can be plugged in:
- Interface-based design
- Configurable extraction pipelines
- Custom entity recognition rules

## Quality Assurance Architecture

### 1. Testing Strategy

**Multi-Level Testing**:
- Unit tests for individual components (95% coverage)
- Integration tests with real APIs
- End-to-end pipeline validation
- Performance regression testing

### 2. Monitoring and Observability

**Built-in Monitoring**:
- Performance metrics collection
- Cost tracking and alerting
- Error rate monitoring
- Quality score validation

### 3. Configuration Validation

**Startup Validation**:
- API key validation
- Configuration consistency checks
- Dependency verification
- Performance baseline establishment

## Deployment Architecture

### 1. Package Distribution

- Standard Python package structure
- pip/uv installable
- Requirements pinning for stability
- Cross-platform compatibility

### 2. Runtime Requirements

- Python 3.11+ for optimal performance
- External API dependencies (Gemini, Tavily)
- Optional spaCy models for enhanced extraction
- Configurable cache directory

### 3. Operational Considerations

- Logging configuration for production
- Performance monitoring hooks
- Error reporting integration
- Cost budget enforcement

## Future Architecture Considerations

### 1. Horizontal Scaling

- Component containerization
- Service mesh architecture
- Distributed caching
- Load balancing strategies

### 2. Data Persistence

- Database integration for historical analysis
- Long-term cache persistence
- Analytics data warehouse
- Backup and recovery

### 3. Multi-Tenancy

- User authentication and authorization
- Resource isolation
- Budget management per tenant
- Custom configuration per user

---

**Last Updated**: 2025-01-29
**Architecture Version**: 1.0
**Component Compatibility**: All current components compatible