# coding: ascii
"""Analytics metrics calculator for BrandLens.""" 

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from src.extraction.citation_extractor import ExtractedCitation
from src.extraction.entity_recognizer import RecognizedEntity
from src.extraction.mention_detector import Mention

_POSITIVE_TERMS = {
    "growth",
    "increase",
    "improved",
    "lead",
    "strong",
    "positive",
    "opportunity",
    "gain",
}
_NEGATIVE_TERMS = {
    "decline",
    "decrease",
    "drop",
    "risk",
    "concern",
    "weak",
    "negative",
    "issue",
}


@dataclass
class MetricsCalculator:
    """Compute BrandLens visibility and competitive metrics."""

    brand_name: str
    citations: List[ExtractedCitation]
    mentions: List[Mention]
    entities: List[RecognizedEntity]
    competitors: List[str]

    def __init__(
        self,
        *,
        brand_name: str,
        citations: Optional[Iterable[ExtractedCitation]] = None,
        mentions: Optional[Iterable[Mention]] = None,
        entities: Optional[Iterable[RecognizedEntity]] = None,
        competitors: Optional[Iterable[str]] = None,
    ) -> None:
        if not brand_name or not brand_name.strip():
            raise ValueError("brand_name is required")

        self.brand_name = brand_name.strip()
        self.brand_name_lower = self.brand_name.lower()
        self.citations = list(citations or [])
        self.mentions = list(mentions or [])
        self.entities = list(entities or [])
        self.competitors = [comp.strip() for comp in competitors or [] if comp and comp.strip()]
        self.competitors_lower = [comp.lower() for comp in self.competitors]

    def calculate_visibility_score(self) -> float:
        """Blend citation, mention, and entity signals into a 0-1 visibility score."""

        citation_signal = len(self.citations) * 3
        mention_signal = len(self.mentions) * 2
        brand_entity_signal = sum(
            1 for entity in self.entities if entity.text.lower() == self.brand_name_lower
        )

        raw_score = citation_signal + mention_signal + brand_entity_signal
        if raw_score <= 0:
            return 0.0

        normalized = raw_score / (raw_score + 10)
        return round(min(1.0, normalized), 4)

    def calculate_position_adjusted_score(self) -> float:
        """Weight mentions by how early they appear in the text."""

        if not self.mentions:
            return 0.0

        weights = [1 / (1 + mention.start) for mention in self.mentions]
        score = sum(weights) / len(self.mentions)
        scaled = min(1.0, score * 50)
        return round(scaled, 4)

    def calculate_share_of_voice(self) -> float:
        """Return the proportion of BrandLens mentions versus competitors."""

        brand_hits = sum(
            1
            for mention in self.mentions
            if mention.matched_variant.lower().startswith(self.brand_name_lower)
        )
        competitor_hits = sum(
            1
            for mention in self.mentions
            if mention.matched_variant.lower() in self.competitors_lower
        )
        total = brand_hits + competitor_hits
        if total == 0:
            return 0.0
        return round(brand_hits / total, 4)

    def calculate_sentiment_indicators(self) -> Dict[str, float]:
        """Return sentiment counts based on keyword heuristics.""" 

        positive = 0
        negative = 0
        for mention in self.mentions:
            context = mention.context.lower()
            if any(term in context for term in _POSITIVE_TERMS):
                positive += 1
            if any(term in context for term in _NEGATIVE_TERMS):
                negative += 1

        neutral = max(0, len(self.mentions) - positive - negative)
        total = max(1, len(self.mentions))
        overall = (positive - negative) / total
        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "overall_score": round(overall, 4),
        }

    def analyze_competitive_landscape(self) -> Dict[str, object]:
        """Summarize competitor presence based on recognized entities."""

        counter = Counter()
        for entity in self.entities:
            text_lower = entity.text.lower()
            if text_lower == self.brand_name_lower:
                continue
            if text_lower in self.competitors_lower:
                counter[text_lower] += 1

        top_competitors = [name for name, _count in counter.most_common(3)]
        return {
            "brand_mentions": self._count_brand_entities(),
            "competitor_mentions": dict(counter),
            "top_competitors": top_competitors,
        }

    def _count_brand_entities(self) -> int:
        return sum(
            1
            for entity in self.entities
            if entity.text.lower() == self.brand_name_lower
        )
