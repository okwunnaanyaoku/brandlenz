# BrandLens - Claude Code Project Instructions

## Project Overview
**BrandLens** is a CLI-based brand visibility analyzer that measures and analyzes how brands appear in LLM-generated responses. This is a production-ready system with real API integration (no mocking) built for a technical interview assessment.

**Key Characteristics:**
- 100% real API integration (Gemini + Tavily)
- 65% token reduction through semantic compression
- 98% entity recognition accuracy
- <8s response time, <$0.05 per query
- Production-ready with comprehensive testing

## Development Guidelines

### Code Quality Standards
- **No mocking**: All APIs use real connections
- **Type hints**: 100% coverage with mypy --strict
- **Testing**: Unit, integration, and E2E with real APIs
- **Documentation**: Comprehensive docstrings and README
- **Error handling**: Graceful degradation and retry logic

### Architecture Principles
- Clean separation of concerns with `src/` structure
- Async-first design for performance
- Comprehensive caching (3-tier system)
- Real-time cost and performance tracking
- Modular design with clear interfaces

### API Integration Requirements
- **Gemini 1.5 Flash**: Primary LLM (cost-effective)
- **Tavily Search**: Content retrieval with extraction
- **Real connections only**: No mocks, stubs, or simulations
- **Rate limiting**: Respect API quotas
- **Cost optimization**: Track and minimize API usage

## Project Structure
```
brandlens/
├── src/                    # Main source code
│   ├── core/              # Business logic & models
│   ├── search/            # Tavily API integration
│   ├── llm/               # Gemini API integration
│   ├── optimization/      # Token compression
│   ├── extraction/        # Information extraction
│   ├── cache/             # Caching system
│   └── utils/             # Utilities
├── tests/                 # Test suite (all real APIs)
├── scripts/               # Utility scripts
├── examples/              # Usage examples
└── docs/                  # Documentation
```

## Key Commands

### Setup
```bash
# Install dependencies
make install

# Validate API connections
make validate-apis

# Run tests
make test                  # All tests
make test-unit            # Unit tests only
make test-integration     # Real API tests
```

### Development
```bash
# Run analysis
make run BRAND="Apple" URL="apple.com" QUESTION="What are the latest iPhone features?"

# Performance monitoring
make benchmark
make cost-analysis

# Code quality
make lint
make format
```

## Development Workflow

### Phase-by-Phase Implementation
1. **Foundation** (Day 1): Project setup, data models, CLI
2. **Search Integration** (Day 2): Tavily API with strategies
3. **Token Optimization** (Day 3): Semantic compression
4. **LLM Integration** (Day 4): Gemini API integration
5. **Information Extraction** (Day 5): Citations, mentions, entities
6. **Analytics** (Day 6): Metrics and competitive analysis
7. **Production Polish** (Day 7): Integration and optimization

### Sub-Agent Usage Strategy
- **system-architect**: For architectural decisions and component design
- **performance-optimizer**: For API optimization and compression algorithms
- **code-refactoring-specialist**: For code quality and technical debt
- **general-purpose**: For implementation tasks
- **hardcoded-data-detector**: Final security scan

### Testing Strategy
- **Unit tests**: Fast, isolated component testing
- **Integration tests**: Real API calls with `@pytest.mark.api`
- **E2E tests**: Complete pipeline validation
- **Performance tests**: Response time and cost validation

## Environment Configuration

### Required API Keys
```bash
# Copy .env.example to .env and configure:
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### Performance Targets
- Response time: <8 seconds
- Cost per query: <$0.05
- Token reduction: 65%
- Extraction accuracy: 95%
- Cache hit rate: 70%

## Implementation Notes

### Critical Requirements
- **No mocking policy**: Every API integration must use real services
- **Cost tracking**: Monitor and report API usage costs
- **Error resilience**: Graceful handling of API failures
- **Performance monitoring**: Real-time metrics collection

### Code Patterns
- Async/await for all I/O operations
- Pydantic models for data validation
- Rich console for beautiful CLI output
- Comprehensive logging with structured data
- Retry logic with exponential backoff

### Quality Gates
- All tests must pass (including real API tests)
- Code coverage >90%
- No linting errors (flake8, mypy)
- Performance requirements met
- Cost requirements met

## Project Tracking
- **Active tracker**: `PROJECT_TRACKER.md` (updated after each task)
- **Progress metrics**: Tests, performance, cost, quality scores
- **Daily updates**: Achievements, challenges, next steps
- **Risk tracking**: Blockers and mitigation strategies

## Documentation Requirements
- **README.md**: Complete usage guide with real examples
- **API.md**: API documentation and integration guides
- **METRICS.md**: Explanation of visibility metrics
- **ARCHITECTURE.md**: Technical architecture decisions

## Success Criteria
- ✅ All 7 phases complete
- ✅ Real API integration working
- ✅ Performance targets met
- ✅ Cost targets met
- ✅ Production-ready code quality
- ✅ Comprehensive test coverage
- ✅ Clear documentation with examples

## Important Reminders
- Always use real APIs - no shortcuts
- Track costs and performance continuously
- Update PROJECT_TRACKER.md after each major task
- Use appropriate sub-agents for specialized tasks
- Maintain high code quality throughout
- Test everything with real data
- Document decisions and learnings

---
*This file serves as the definitive guide for developing BrandLens with Claude Code*