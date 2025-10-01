# coding: ascii
"""Mention detection utilities for BrandLens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Tuple


class MentionDetectionError(Exception):
    """Raised when mention detection input is invalid."""


@dataclass
class Mention:
    """Structured representation of a detected brand mention."""

    text: str
    start: int
    end: int
    matched_variant: str
    score: float
    context: str


_VARIANT_SUFFIXES = [
    "",
    " inc",
    " inc.",
    " incorporated",
    " ltd",
    " ltd.",
]
_POSSESSIVE_SUFFIXES = ["'s"]
_WORD_PATTERN = re.compile(r"\b\w[\w'-]*\b", re.UNICODE)


def generate_variants(brand_name: str, extra_variants: Optional[Iterable[str]] = None) -> List[str]:
    """Generate canonical variants for a brand name."""

    if not brand_name or not brand_name.strip():
        raise MentionDetectionError("Brand name cannot be empty")

    base = brand_name.strip()
    variants = []

    lower_base = base.lower()
    for suffix in _VARIANT_SUFFIXES:
        variant = f"{lower_base}{suffix}".strip()
        if variant not in variants:
            variants.append(variant)

    for pos in _POSSESSIVE_SUFFIXES:
        variant = f"{lower_base}{pos}"
        if variant not in variants:
            variants.append(variant)

    if extra_variants:
        for item in extra_variants:
            if item and item.strip():
                lowered = item.strip().lower()
                if lowered not in variants:
                    variants.append(lowered)

    variants = [variant for variant in variants if variant]
    variants.sort(key=len, reverse=True)
    return variants

def detect_mentions(
    text: str,
    brand_name: str,
    *,
    extra_variants: Optional[Iterable[str]] = None,
    fuzzy_threshold: float = 0.86,
    context_window: int = 80,
) -> List[Mention]:
    """Detect mentions of a brand, including fuzzy matches for typos."""

    if not text or not text.strip():
        raise MentionDetectionError("Text content is empty")

    variants = generate_variants(brand_name, extra_variants=extra_variants)
    lowered_text = text.lower()

    mentions: List[Mention] = []
    occupied: List[Tuple[int, int]] = []

    for variant in variants:
        pattern = re.compile(rf"\b{re.escape(variant)}\b", re.IGNORECASE)
        for match in pattern.finditer(lowered_text):
            start, end = match.start(), match.end()
            if _overlaps(start, end, occupied):
                continue
            original_text = text[start:end]
            context = _context_window(text, start, end, context_window)
            mentions.append(
                Mention(
                    text=original_text,
                    start=start,
                    end=end,
                    matched_variant=variant,
                    score=1.0,
                    context=context,
                )
            )
            occupied.append((start, end))

    fuzzy_mentions = _fuzzy_mentions(
        text=text,
        brand=brand_name,
        occupied=occupied,
        threshold=fuzzy_threshold,
        context_window=context_window,
    )
    mentions.extend(fuzzy_mentions)

    mentions.sort(key=lambda item: item.start)
    return mentions


def _overlaps(start: int, end: int, ranges: List[Tuple[int, int]]) -> bool:
    for range_start, range_end in ranges:
        if not (end <= range_start or start >= range_end):
            return True
    return False


def _fuzzy_mentions(
    *,
    text: str,
    brand: str,
    occupied: List[Tuple[int, int]],
    threshold: float,
    context_window: int,
) -> List[Mention]:
    mentions: List[Mention] = []
    normalized_brand = brand.strip()
    if not normalized_brand:
        return mentions

    brand_token_count = len(normalized_brand.split())
    for match in _WORD_PATTERN.finditer(text):
        token_start = match.start()
        window_tokens = [match.group()]
        window_end = match.end()

        token_iter = _WORD_PATTERN.finditer(text, match.end())
        for _ in range(brand_token_count - 1):
            try:
                next_token = next(token_iter)
            except StopIteration:
                break
            window_tokens.append(next_token.group())
            window_end = next_token.end()

        candidate = " ".join(window_tokens).strip()
        if not candidate:
            continue

        start = token_start
        end = window_end
        if _overlaps(start, end, occupied):
            continue

        score = SequenceMatcher(None, candidate.lower(), normalized_brand.lower()).ratio()
        if score >= threshold:
            context = _context_window(text, start, end, context_window)
            mentions.append(
                Mention(
                    text=text[start:end],
                    start=start,
                    end=end,
                    matched_variant=normalized_brand.lower(),
                    score=score,
                    context=context,
                )
            )
            occupied.append((start, end))

    return mentions


def _context_window(text: str, start: int, end: int, window: int) -> str:
    left = max(start - window, 0)
    right = min(end + window, len(text))
    snippet = text[left:right].replace("\n", " ")
    return snippet.strip()


__all__ = [
    "Mention",
    "MentionDetectionError",
    "detect_mentions",
    "generate_variants",
]
