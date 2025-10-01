"""Technical deep-dive search strategy."""

from __future__ import annotations

from .base import SearchParameters, SearchStrategy, SearchStrategyContext
from ...core import SearchDepth


TECH_KEYWORDS = ["API", "integration", "architecture", "SDK", "technical whitepaper"]


class TechnicalSearchStrategy(SearchStrategy):
    name = "technical"
    description = "Surface technical integrations, documentation, and expert commentary."

    def build(self, context: SearchStrategyContext) -> SearchParameters:
        keyword_phrase = " ".join(TECH_KEYWORDS)
        domain_clause = f" site:{context.brand_domain}" if context.brand_domain else ""
        query = f"{context.query} {keyword_phrase}{domain_clause}".strip()
        return SearchParameters(
            query=query,
            depth=SearchDepth.ADVANCED,
            max_results=12,
            include_raw_content=True,
            tags=["technical", "deep-dive"],
        )
