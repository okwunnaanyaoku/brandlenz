"""Comparative search strategy for competitor analysis."""

from __future__ import annotations

from .base import SearchParameters, SearchStrategy, SearchStrategyContext
from ...core import SearchDepth

COMPARATIVE_KEYWORDS = ["vs", "comparison", "alternatives", "competitor"]


class ComparativeSearchStrategy(SearchStrategy):
    name = "comparative"
    description = "Highlight competitor comparisons and alternatives."

    def build(self, context: SearchStrategyContext) -> SearchParameters:
        competitor_clause = " OR ".join(COMPARATIVE_KEYWORDS)
        query = f"{context.query} ({competitor_clause})"
        if context.brand_name:
            query += f" {context.brand_name}" 
        return SearchParameters(
            query=query.strip(),
            depth=SearchDepth.ADVANCED,
            max_results=12,
            include_raw_content=True,
            tags=["comparative", "competitor"],
        )
