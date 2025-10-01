"""Exploratory search strategy for trend discovery."""

from __future__ import annotations

from .base import SearchParameters, SearchStrategy, SearchStrategyContext
from ...core import SearchDepth

EXPLORATORY_KEYWORDS = ["trend", "insights", "analysis", "overview", "guide"]


class ExploratorySearchStrategy(SearchStrategy):
    name = "exploratory"
    description = "Discover broad insights, guides, and trend pieces."

    def build(self, context: SearchStrategyContext) -> SearchParameters:
        query = f"{context.query} {' '.join(EXPLORATORY_KEYWORDS)}"
        return SearchParameters(
            query=query.strip(),
            depth=SearchDepth.BASIC,
            max_results=12,
            include_raw_content=False,
            tags=["exploratory", "trends"],
        )
