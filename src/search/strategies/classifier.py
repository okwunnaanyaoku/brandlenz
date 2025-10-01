"""Query classification helpers for selecting search strategies."""

from __future__ import annotations

from typing import Dict, Tuple, Type

from .base import SearchStrategy, SearchStrategyContext
from .brand import BrandFocusedStrategy
from .factual import FactualSearchStrategy
from .comparative import ComparativeSearchStrategy
from .exploratory import ExploratorySearchStrategy
from .social import SocialListeningStrategy

_KEYWORD_MAP: Dict[Type[SearchStrategy], Tuple[str, ...]] = {
    FactualSearchStrategy: (
        "official",
        "announcement",
        "press release",
        "earnings",
        "launch",
    ),
    ComparativeSearchStrategy: (
        "versus",
        "vs",
        "comparison",
        "alternative",
        "competitor",
    ),
    ExploratorySearchStrategy: (
        "trend",
        "insight",
        "overview",
        "guide",
        "analysis",
    ),
    SocialListeningStrategy: (
        "social",
        "reddit",
        "twitter",
        "tiktok",
        "community",
        "influencer",
    ),
    BrandFocusedStrategy: (
        "reputation",
        "sentiment",
        "brand",
        "monitor",
        "perception",
    ),
}

_DEFAULT_STRATEGY = FactualSearchStrategy


def classify_query(context: SearchStrategyContext) -> SearchStrategy:
    """Pick the most appropriate strategy based on the query context."""

    text = (context.query or "").lower()

    # Brand-specific queries default to monitoring.
    if context.brand_name and any(term in text for term in (context.brand_name.lower(), "brand")):
        return BrandFocusedStrategy()

    matches = []
    for strategy_class, keywords in _KEYWORD_MAP.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score:
            matches.append((score, strategy_class))

    if not matches:
        # Fall back to news coverage when no obvious signals are present.
        return _DEFAULT_STRATEGY()

    matches.sort(reverse=True, key=lambda item: item[0])
    return matches[0][1]()
