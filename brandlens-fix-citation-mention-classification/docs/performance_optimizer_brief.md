## Performance Optimizer Brief (Phase 2)

### Scope
- Review Tavily client + orchestrator for API efficiency: retry strategy, rate limit handling, cost metrics.
- Identify opportunities for caching (reuse search results) and request batching when multiple strategies hit similar queries.
- Evaluate timeouts/backoff tuning against real API latency.

### Inputs
- Code: src/search/tavily_client.py, src/search/orchestrator.py
- Tests: unit + optional smoke tests (`pytest -m api ...`)
- Tracker: cost/rate data once smoke tests run

### Questions for Optimizer
1. Are retry/backoff settings aligned with Tavily quotas?
2. Should we introduce caching or de-duplication beyond current query dedupe?
3. How should we log/monitor cost over time (hook into PerformanceMetrics)?

### Next Steps
- After running smoke tests, provide optimizer with metrics snapshot (cost, rate limit, request IDs).
- Schedule performance-optimizer review (Phase 2 plan) to propose tuning.


### Smoke Test Metrics (2025-01-28)
- Tavily cost per search: ~$0.001 (latest brandlens news)
- Tavily content retrieval: skipped when 404 (fallback documented)
- Rate limit remaining: ~95 requests after smoke run
- Total smoke spend: ~$0.0015 (2 calls)


### Optimizer Recommendations (2025-01-28)
- Tune Tavily retry strategy to stop after HTTP 4xx (rate limit) with cooldown logging.
- Introduce optional in-memory caching for identical strategy queries within a run.
- Emit cost/rate metrics via PerformanceMetrics to feed project dashboards.
- Increase timeout to 45s for content endpoint; wrap with graceful skip fallback.
- Capture request IDs in logs to aid incident correlation.


### Implemented (2025-01-28)
- Added optional in-memory cache for Tavily client per run.
- Logging request id, cost, and rate limit on every Tavily call.
- Search orchestrator aggregates cost and rate-limit metadata.
- Smoke tests updated to handle 404 gracefully.


### Budget Management
- Implemented BudgetManager hook in orchestrator with cost tracking.
- Need CLI flag to enable caching/budget enforcement per run.
- Follow up after Task 2.3 to verify spend stays within limits.
