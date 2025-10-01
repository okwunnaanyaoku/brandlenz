# Phase 2 - Tavily Search Integration (Task 2.1)

## Objective
Build an async Tavily API client that supports search and content retrieval for downstream brand analysis.

## Components
- src/search/orchestrator.py
  - Strategy-driven orchestration using Tavily client
  - Aggregates strategy responses and logs metrics
- src/search/tavily_client.py
  - Async client wrapper (httpx.AsyncClient)
  - Methods: search(query, depth, max_results), get_content(url)
  - Retry/backoff via tenacity
  - Request/response models (Pydantic) for Tavily payloads
  - Metrics hooks for API usage tracking
- src/search/__init__.py
  - Export Tavily client class and helper functions
- src/search/strategies/
  - Strategy base class + Factual/Comparative/Exploratory/Brand implementations
  - Query classifier to pick strategies via heuristics with keyword scoring
- src/search/analytics.py
  - Summarize SearchRunSummary objects into strategy metrics (cost, domains, results)
- Config
  - Reuse APIConfig fields (API key, search depth, raw content toggle)
  - Add rate limit handling (MAX_SEARCHES_PER_QUERY, MAX_SOURCES_PER_SEARCH)

## Tests
- CLI: `python -m src search "<query>" --max-searches 5 --max-cost 1.0 --enable-cache`
  - Budget-aware run that prints strategy table and remaining budget
- tests/unit/search/test_tavily_client.py
- tests/unit/search/test_tavily_client.py
- tests/unit/search/test_strategies.py
- tests/unit/search/test_analytics.py
  - Mock httpx responses
  - Validate request payloads and error handling
  - Retry scenarios (rate limit, transient failures)
  - Raw content toggle behaviour
- tests/integration/search/test_tavily_client.py
  - Optional real API smoke (marked @pytest.mark.api)

## Timeline
- Estimate: 3 hours (1.5h client implementation, 1h unit tests, 0.5h integration setup)
- Sub-Agent: general-purpose for client code, code-refactoring-specialist post-implementation review

## Risks
- API rate limits (mitigate with backoff and caching)
- Large payloads when raw content enabled (consider streaming or chunking)
- Credential errors (provide clear ConfigurationError mapping)

## Next Actions
1. Scaffold client module with async httpx session management.
2. Define Pydantic models aligned with Tavily response schema.
3. Implement search and content methods with retries and error mapping.
4. Write unit tests mocking httpx.
5. Add optional integration smoke test.
6. Update tracker and assign refactor review upon completion.
