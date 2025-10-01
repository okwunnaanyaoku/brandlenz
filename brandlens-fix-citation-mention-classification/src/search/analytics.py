"""Analytics helpers for evaluating search strategy performance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

from .budget import BudgetState

from .orchestrator import SearchResultBundle, SearchRunSummary


@dataclass
class StrategyMetrics:
    strategy: str
    total_results: int
    unique_domains: int
    total_cost_usd: float
    average_results_per_call: float


@dataclass
class AnalyticsReport:
    metrics: List[StrategyMetrics]
    total_cost_usd: float
    api_calls: int
    last_rate_limit_remaining: int | None
    budget_state: Optional[BudgetState]


def _domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or url


def summarize_strategy_results(summary: SearchRunSummary) -> AnalyticsReport:
    domain_map: Dict[str, set[str]] = {}
    result_count: Dict[str, int] = {}
    cost_map: Dict[str, float] = {}

    for bundle in summary.bundles:
        name = bundle.strategy.name
        result_count.setdefault(name, 0)
        domain_map.setdefault(name, set())
        cost_map.setdefault(name, 0.0)

        for result in bundle.response.results:
            domain_map[name].add(_domain(result.url))
            result_count[name] += 1

        cost_map[name] += bundle.response.metadata.cost_usd

    metrics: List[StrategyMetrics] = []
    for name in result_count:
        calls = sum(1 for bundle in summary.bundles if bundle.strategy.name == name)
        metrics.append(
            StrategyMetrics(
                strategy=name,
                total_results=result_count[name],
                unique_domains=len(domain_map[name]),
                total_cost_usd=round(cost_map[name], 6),
                average_results_per_call=result_count[name] / calls if calls else 0.0,
            )
        )

    return AnalyticsReport(
        metrics=metrics,
        total_cost_usd=summary.total_cost_usd,
        api_calls=summary.api_calls,
        last_rate_limit_remaining=summary.last_rate_limit_remaining,
        budget_state=summary.budget_state,
    )

