"""Search strategy abstractions for BrandLens."""

from __future__ import annotations

from .base import SearchStrategy, SearchStrategyContext, SearchParameters
from .factual import FactualSearchStrategy
from .comparative import ComparativeSearchStrategy
from .exploratory import ExploratorySearchStrategy
from .brand import BrandFocusedStrategy
from .news import NewsSearchStrategy  # legacy
from .technical import TechnicalSearchStrategy  # legacy
from .social import SocialListeningStrategy  # legacy
from .classifier import classify_query

__all__ = [
    "SearchStrategy",
    "SearchStrategyContext",
    "SearchParameters",
    "FactualSearchStrategy",
    "ComparativeSearchStrategy",
    "ExploratorySearchStrategy",
    "BrandFocusedStrategy",
    "NewsSearchStrategy",
    "TechnicalSearchStrategy",
    "SocialListeningStrategy",
    "classify_query",
]
