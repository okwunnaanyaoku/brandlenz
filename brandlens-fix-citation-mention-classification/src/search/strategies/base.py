"""Base classes for search strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

from ...core import APIConfig, SearchDepth


@dataclass
class SearchStrategyContext:
    """Context shared across search strategies."""

    query: str
    brand_name: Optional[str] = None
    brand_domain: Optional[str] = None
    api_config: Optional[APIConfig] = None


@dataclass
class SearchParameters:
    """Normalized Tavily search parameters produced by a strategy."""

    query: str
    depth: SearchDepth = SearchDepth.ADVANCED
    max_results: Optional[int] = None  # Set by orchestrator based on user's max_sources
    include_raw_content: Union[bool, str] = "markdown"
    chunks_per_source: int = 3
    tags: List[str] = field(default_factory=list)


class SearchStrategy:
    """Base strategy abstraction."""

    name: str = "base"
    description: str = "Generic search strategy"

    def build(self, context: SearchStrategyContext) -> SearchParameters:
        """Return search parameters for the provided context."""

        raise NotImplementedError
