"""News-focused search strategy."""

from __future__ import annotations

from .base import SearchParameters, SearchStrategy, SearchStrategyContext
from ...core import SearchDepth


class NewsSearchStrategy(SearchStrategy):
    name = "news"
    description = "Track latest press coverage and announcements."

    def build(self, context: SearchStrategyContext) -> SearchParameters:
        brand_segment = ""
        if context.brand_name:
            brand_segment = f" \"{context.brand_name}\""
        query = f"latest news{brand_segment} {context.query}".strip()
        return SearchParameters(
            query=query,
            depth=SearchDepth.ADVANCED,
            max_results=8,
            include_raw_content=False,
            tags=["news", "time-sensitive"],
        )
