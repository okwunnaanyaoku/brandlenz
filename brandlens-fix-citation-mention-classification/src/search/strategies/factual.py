"""Factual search strategy focusing on official sources and announcements."""

from __future__ import annotations

from .base import SearchParameters, SearchStrategy, SearchStrategyContext
from ...core import SearchDepth

FACTUAL_KEYWORDS = ["official", "announcement", "press release", "statement"]


class FactualSearchStrategy(SearchStrategy):
    name = "factual"
    description = "Surface official statements and high-authority sources."

    def build(self, context: SearchStrategyContext) -> SearchParameters:
        query_parts = [context.query, " OR ".join(FACTUAL_KEYWORDS)]
        if context.brand_name:
            query_parts.append(f'"{context.brand_name}"')
        query = " ".join(filter(None, query_parts))
        return SearchParameters(
            query=query,
            depth=SearchDepth.ADVANCED,
            max_results=None,  # Will be set by orchestrator based on user's max_sources
            include_raw_content="markdown",
            tags=["factual", "official"],
        )
