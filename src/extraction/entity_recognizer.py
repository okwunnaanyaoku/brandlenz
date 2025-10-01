# coding: ascii
"""Entity recognition utilities for BrandLens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


class EntityRecognitionError(Exception):
    """Raised when entity recognition input is invalid."""


@dataclass
class RecognizedEntity:
    """Structured named entity extracted from free text."""

    text: str
    start: int
    end: int
    label: str
    score: float
    source: str


class EntityRecognizer:
    """Combine spaCy NER with rule-based brand detection."""

    def __init__(
        self,
        *,
        nlp: Optional[object] = None,
        brand_terms: Optional[Iterable[str]] = None,
        brand_score: float = 0.95,
    ) -> None:
        self._nlp = nlp
        self._brand_terms = _normalise_terms(brand_terms)
        self._brand_score = brand_score

    def recognize(
        self,
        text: str,
        *,
        brands: Optional[Iterable[str]] = None,
    ) -> List[RecognizedEntity]:
        if not text or not text.strip():
            raise EntityRecognitionError("Text content is empty")

        content = text
        results: List[RecognizedEntity] = []

        occupied: List[Tuple[int, int]] = []

        if self._nlp is not None:
            doc = self._nlp(content)
            ents = getattr(doc, "ents", [])
            for ent in ents:
                start = getattr(ent, "start_char", None)
                end = getattr(ent, "end_char", None)
                span_text = getattr(ent, "text", None)
                if start is None or end is None or span_text is None:
                    continue
                label = getattr(ent, "label_", "UNKNOWN")
                score = _extract_score(ent)
                if _overlaps_range(start, end, occupied):
                    continue
                occupied.append((start, end))
                results.append(
                    RecognizedEntity(
                        text=span_text,
                        start=start,
                        end=end,
                        label=label,
                        score=score,
                        source="spacy",
                    )
                )

        combined_terms = set(self._brand_terms)
        combined_terms.update(_normalise_terms(brands))

        for term in sorted(combined_terms, key=len, reverse=True):
            if not term:
                continue
            for match in re.finditer(rf"\b{re.escape(term)}\b", content, re.IGNORECASE):
                start, end = match.start(), match.end()
                if _overlaps_range(start, end, occupied):
                    continue
                occupied.append((start, end))
                results.append(
                    RecognizedEntity(
                        text=content[start:end],
                        start=start,
                        end=end,
                        label="BRAND",
                        score=self._brand_score,
                        source="rule",
                    )
                )

        results.sort(key=lambda item: item.start)
        return results


def _normalise_terms(terms: Optional[Iterable[str]]) -> List[str]:
    if not terms:
        return []
    seen = []
    for term in terms:
        if not term:
            continue
        lowered = term.strip().lower()
        if lowered and lowered not in seen:
            seen.append(lowered)
    return seen


def _overlaps_range(start: int, end: int, occupied: Sequence[Tuple[int, int]]) -> bool:
    for existing_start, existing_end in occupied:
        if not (end <= existing_start or start >= existing_end):
            return True
    return False


def _extract_score(ent: object) -> float:
    for attr in ("kprob", "score", "confidence"):
        value = getattr(ent, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.9


__all__ = [
    "RecognizedEntity",
    "EntityRecognitionError",
    "EntityRecognizer",
]
