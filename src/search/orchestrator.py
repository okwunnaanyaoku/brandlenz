"""Search orchestration utilities combining strategies and Tavily client."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from . import TavilyClient, TavilySearchResponse
from .budget import BudgetManager, BudgetState
from .strategies import (
    BrandFocusedStrategy,
    ComparativeSearchStrategy,
    ExploratorySearchStrategy,
    FactualSearchStrategy,
    SearchParameters,
    SearchStrategy,
    SearchStrategyContext,
    classify_query,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class SearchResultBundle:
    """Aggregated search response annotated with the strategy that produced it."""

    strategy: SearchStrategy
    parameters: SearchParameters
    response: TavilySearchResponse


@dataclass
class SearchRunSummary:
    bundles: List[SearchResultBundle]
    total_cost_usd: float
    api_calls: int
    last_rate_limit_remaining: Optional[int]
    last_request_id: Optional[str]
    budget_state: Optional[BudgetState] = None


class SearchOrchestrator:
    """High-level orchestrator that chooses strategies and queries Tavily."""

    def __init__(
        self,
        client: TavilyClient,
        *,
        strategies: Optional[Iterable[SearchStrategy]] = None,
        budget_manager: Optional[BudgetManager] = None,
    ) -> None:
        self._client = client
        self._strategies = list(strategies) if strategies else []
        self._budget = budget_manager

    def register_strategy(self, strategy: SearchStrategy) -> None:
        self._strategies.append(strategy)

    async def run(
        self,
        context: SearchStrategyContext,
        *,
        include_classifier: bool = True,
        max_sources: Optional[int] = None,
    ) -> SearchRunSummary:
        bundles: List[SearchResultBundle] = []

        strategy_list = list(self._strategies)
        if include_classifier:
            strategy_list.insert(0, classify_query(context))

        seen_keys = set()
        seen_urls = set()  # Track unique sources collected
        total_cost = 0.0
        last_rate_limit: Optional[int] = None
        last_request_id: Optional[str] = None

        # Use first strategy (simplified - only Factual strategy now)
        if not strategy_list:
            raise ValueError("No strategies configured")

        strategy = strategy_list[0]
        params = strategy.build(context)

        # Keep making calls until we reach max_sources or budget limit
        while True:
            # Check if we've collected enough sources
            if max_sources is not None:
                remaining_sources = max_sources - len(seen_urls)
                if remaining_sources <= 0:
                    LOGGER.info(
                        "Source limit reached; stopping search execution",
                        extra={"collected_sources": len(seen_urls), "max_sources": max_sources}
                    )
                    break
                # Request exactly what we still need
                effective_max_results = remaining_sources
            else:
                # No max_sources specified - use a sensible default
                effective_max_results = params.max_results if params.max_results else 10

            # Check budget before making the call
            if self._budget is not None and self._budget.should_stop():
                LOGGER.info("Budget limits reached; stopping search execution")
                break

            LOGGER.info(
                "Running strategy",
                extra={
                    "strategy": strategy.name,
                    "query": params.query,
                    "max_results": effective_max_results,
                    "collected_sources": len(seen_urls),
                    "call_number": len(bundles) + 1
                }
            )

            response = await self._client.search(
                params.query,
                depth=params.depth,
                max_results=effective_max_results,
                include_raw_content=params.include_raw_content,
                chunks_per_source=params.chunks_per_source,
            )
            bundles.append(SearchResultBundle(strategy=strategy, parameters=params, response=response))
            total_cost += response.metadata.cost_usd
            if response.metadata.rate_limit_remaining is not None:
                last_rate_limit = response.metadata.rate_limit_remaining
            if response.metadata.request_id:
                last_request_id = response.metadata.request_id

            # Track unique URLs from this response
            urls_before = len(seen_urls)
            for result in response.results:
                seen_urls.add(result.url)
            urls_added = len(seen_urls) - urls_before

            # If Tavily returned 0 new unique sources, no point in continuing
            if urls_added == 0:
                LOGGER.info(
                    "No new sources returned; stopping search execution",
                    extra={"collected_sources": len(seen_urls)}
                )
                break

            # Update budget after the call
            if self._budget is not None:
                self._budget.record_call(response.metadata.cost_usd)

        budget_state = None
        if self._budget is not None:
            state = self._budget.state
            budget_state = BudgetState(total_searches=state.total_searches, total_cost_usd=state.total_cost_usd)

        return SearchRunSummary(
            bundles=bundles,
            total_cost_usd=round(total_cost, 6),
            api_calls=len(bundles),
            last_rate_limit_remaining=last_rate_limit,
            last_request_id=last_request_id,
            budget_state=budget_state,
        )
