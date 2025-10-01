"""Search utilities for BrandLens."""

from .tavily_client import (
    TavilyClient,
    TavilyClientSettings,
    TavilyContentResponse,
    TavilySearchResponse,
    TavilySearchResult,
)
from .budget import BudgetLimits, BudgetManager, BudgetState
from .analytics import AnalyticsReport, StrategyMetrics, summarize_strategy_results
from .orchestrator import SearchOrchestrator, SearchResultBundle
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

__all__ = [
    "TavilyClient",
    "TavilyClientSettings",
    "TavilySearchResponse",
    "TavilySearchResult",
    "TavilyContentResponse",
    "BudgetLimits",
    "BudgetManager",
    "BudgetState",
    "AnalyticsReport",
    "StrategyMetrics",
    "summarize_strategy_results",
    "SearchOrchestrator",
    "SearchResultBundle",
    "SearchStrategy",
    "SearchStrategyContext",
    "SearchParameters",
    "FactualSearchStrategy",
    "ComparativeSearchStrategy",
    "ExploratorySearchStrategy",
    "BrandFocusedStrategy",
    "classify_query",
]
