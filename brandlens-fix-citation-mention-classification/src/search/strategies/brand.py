"""Brand monitoring search strategy."""

from __future__ import annotations

from .base import SearchParameters, SearchStrategy, SearchStrategyContext
from ...core import SearchDepth


class BrandFocusedStrategy(SearchStrategy):
    name = "brand-monitoring"
    description = "Monitor brand reputation, sentiment, and competitive mentions."

    def build(self, context: SearchStrategyContext) -> SearchParameters:
        brand = context.brand_name or context.brand_domain or "the brand"
        query = f"{brand} reputation sentiment {context.query}".strip()
        return SearchParameters(
            query=query,
            depth=SearchDepth.BASIC,
            max_results=15,
            include_raw_content=True,
            tags=["brand", "sentiment"],
        )
