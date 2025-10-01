"""Social listening search strategy."""

from __future__ import annotations

from .base import SearchParameters, SearchStrategy, SearchStrategyContext
from ...core import SearchDepth

SOCIAL_KEYWORDS = ["Twitter", "Reddit", "LinkedIn", "TikTok", "social media", "conversation"]


class SocialListeningStrategy(SearchStrategy):
    name = "social-listening"
    description = "Capture social media chatter and community discussions."

    def build(self, context: SearchStrategyContext) -> SearchParameters:
        keyword_phrase = " ".join(SOCIAL_KEYWORDS)
        query = f"{context.query} {keyword_phrase}".strip()
        return SearchParameters(
            query=query,
            depth=SearchDepth.BASIC,
            max_results=12,
            include_raw_content=False,
            tags=["social", "community"],
        )
