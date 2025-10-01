# coding: ascii
"""Markdown citation extraction utilities."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class CitationExtractionError(Exception):
    """Raised when citation extraction fails or encounters invalid input."""


@dataclass
class ExtractedCitation:
    """Structured representation of a citation pulled from markdown."""

    text: str
    url: str
    matched_entities: List[str]
    context_snippet: Optional[str] = None


_LINK_PATTERN = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<url>[^)]+)\)")
_REFERENCE_PATTERN = re.compile(r"\[(?P<number>\d+)\]\s*(?P<text>[^-\n]+)\s*-\s*(?P<url>https?://[^\s\n]+)")
_CITATION_USAGE_PATTERN = re.compile(r"\[([0-9, ]+)\]")  # Matches [1], [1, 2], [1, 2, 3] etc
_INVALID_TRAILING_CHARS = {'.', ',', ';', ')'}


def extract_citations(
    markdown: str,
    *,
    entities: Optional[Iterable[str]] = None,
    window: int = 160,
) -> List[ExtractedCitation]:
    """Extract citations from markdown and associate known entities."""

    if not markdown or not markdown.strip():
        raise CitationExtractionError("Markdown content is empty")

    text = markdown.strip()
    entities_list = [entity.strip() for entity in (entities or []) if entity and entity.strip()]

    # Step 1: Build citation number to URL mapping from References section
    reference_url_map = _build_reference_map(text)

    if not reference_url_map:
        # Fallback to direct link extraction if no references section
        return _extract_direct_links(text, entities_list, window)

    # Step 2: Find citation usages in text and extract contextual text
    # First, identify the References section to exclude citations from it
    references_start = _find_references_section_start(text)

    citations: List[ExtractedCitation] = []
    citation_usages = list(_CITATION_USAGE_PATTERN.finditer(text))

    for usage_match in citation_usages:
        citation_nums_str = usage_match.group(1)
        citation_numbers = [int(num.strip()) for num in citation_nums_str.split(',') if num.strip().isdigit()]

        # Skip citations found in the References section
        start = usage_match.start()
        end = usage_match.end()

        if references_start is not None and start >= references_start:
            continue  # Skip citations in References section

        context = _context_window(text, start, end, window)

        # Create citations for each referenced number
        for citation_num in citation_numbers:
            if citation_num in reference_url_map:
                url = reference_url_map[citation_num]
                # Use citation marker as text (e.g., "[1]")
                citation_marker = f"[{citation_num}]"
                matched_entities = _match_entities(context, context, entities_list)

                citations.append(
                    ExtractedCitation(
                        text=citation_marker,
                        url=url,
                        matched_entities=matched_entities,
                        context_snippet=context,
                    )
                )

    # Deduplicate citations before returning
    deduplicated_citations = _deduplicate_citations(citations)

    # Validate: warn if citations are referenced but not found in References section
    _validate_citations(deduplicated_citations, text)

    return deduplicated_citations


def _deduplicate_citations(citations: List[ExtractedCitation]) -> List[ExtractedCitation]:
    """Remove duplicate citations and merge citations with same URL."""
    if not citations:
        return citations

    # Group citations by URL
    url_groups = {}
    for citation in citations:
        url = citation.url
        if url not in url_groups:
            url_groups[url] = []
        url_groups[url].append(citation)

    deduplicated = []
    for url, citation_group in url_groups.items():
        if len(citation_group) == 1:
            # Single citation for this URL - keep as is
            deduplicated.append(citation_group[0])
        else:
            # Multiple citations for same URL - merge them
            merged_citation = _merge_citations(citation_group)
            deduplicated.append(merged_citation)

    return deduplicated


def _merge_citations(citations: List[ExtractedCitation]) -> ExtractedCitation:
    """Merge multiple citations for the same URL into a single citation."""
    if not citations:
        raise ValueError("Cannot merge empty citation list")

    if len(citations) == 1:
        return citations[0]

    # Use the first citation as base
    base_citation = citations[0]

    # Collect all unique text segments
    text_segments = []
    all_entities = set()
    contexts = []

    for citation in citations:
        # Add unique text segments
        if citation.text not in text_segments:
            text_segments.append(citation.text)

        # Combine entities
        all_entities.update(citation.matched_entities)

        # Collect context snippets
        if citation.context_snippet:
            contexts.append(citation.context_snippet)

    # Choose the most meaningful text (longest unique segment)
    merged_text = max(text_segments, key=len) if text_segments else base_citation.text

    # Combine context snippets (take first non-empty one)
    merged_context = next((ctx for ctx in contexts if ctx and ctx.strip()), base_citation.context_snippet)

    return ExtractedCitation(
        text=merged_text,
        url=base_citation.url,
        matched_entities=list(all_entities),
        context_snippet=merged_context
    )


def _validate_citations(citations: List[ExtractedCitation], markdown: str) -> None:
    """Log warnings for referenced but undefined citations."""
    # Find all citation usages in markdown
    all_cited_nums = set()
    for match in _CITATION_USAGE_PATTERN.finditer(markdown):
        nums = [int(n.strip()) for n in match.group(1).split(',') if n.strip().isdigit()]
        all_cited_nums.update(nums)
    
    # Find which citations were extracted
    extracted_nums = set()
    for cit in citations:
        match = re.match(r'\[(\d+)\]', cit.text)
        if match:
            extracted_nums.add(int(match.group(1)))
    
    # Log missing citations
    missing = all_cited_nums - extracted_nums
    if missing:
        logger.warning(f"Citations referenced but not found in References section: {sorted(missing)}")


def _build_reference_map(text: str) -> dict[int, str]:
    """Build a mapping from citation numbers to URLs from the References section."""
    reference_map = {}
    reference_matches = list(_REFERENCE_PATTERN.finditer(text))

    for match in reference_matches:
        try:
            citation_num = int(match.group("number"))
            raw_url = match.group("url").strip()
            normalized_url = _normalize_url(raw_url)
            if normalized_url:
                reference_map[citation_num] = normalized_url
        except (ValueError, AttributeError):
            continue

    return reference_map


def _extract_direct_links(text: str, entities_list: List[str], window: int) -> List[ExtractedCitation]:
    """Fallback extraction for direct markdown links."""
    citations = []
    link_matches = list(_LINK_PATTERN.finditer(text))

    for match in link_matches:
        link_text = match.group("text").strip()
        raw_url = match.group("url").strip()
        normalized_url = _normalize_url(raw_url)

        if normalized_url:
            start = match.start()
            end = match.end()
            context = _context_window(text, start, end, window)
            matched_entities = _match_entities(link_text, context, entities_list)

            citations.append(
                ExtractedCitation(
                    text=link_text,
                    url=normalized_url,
                    matched_entities=matched_entities,
                    context_snippet=context,
                )
            )

    return citations


def _extract_citation_context(text: str, citation_start: int, citation_end: int) -> str:
    """Extract meaningful context text around a citation."""
    # Find the broader paragraph or section that contains the citation
    # Look backwards for paragraph start (double newline, section header, or start of text)
    paragraph_start = citation_start
    for i in range(citation_start - 1, max(0, citation_start - 500), -1):
        if i > 0 and text[i-1:i+1] == '\n\n':  # Paragraph break
            paragraph_start = i + 1
            break
        elif text[i:i+2] == '##':  # Section header
            paragraph_start = i
            break
        elif i == 0:
            paragraph_start = 0
            break

    # Look forwards for paragraph end
    paragraph_end = citation_end
    for i in range(citation_end, min(len(text), citation_end + 500)):
        if i < len(text) - 1 and text[i:i+2] == '\n\n':  # Paragraph break
            paragraph_end = i
            break
        elif i < len(text) - 2 and text[i:i+2] == '##':  # Next section header
            paragraph_end = i
            break
        elif i == len(text) - 1:
            paragraph_end = len(text)
            break

    # Extract the paragraph/section
    paragraph = text[paragraph_start:paragraph_end].strip()

    # Remove citation markers for cleaner display
    cleaned_paragraph = re.sub(r'\[[\d\s,]+\]', '', paragraph).strip()
    cleaned_paragraph = re.sub(r'\s+', ' ', cleaned_paragraph)

    # Extract the most relevant sentence or claim from the paragraph
    # Split into sentences and find the one closest to the citation
    sentences = re.split(r'[.!?]\s+', cleaned_paragraph)

    if not sentences:
        return cleaned_paragraph[:200] + "..." if len(cleaned_paragraph) > 200 else cleaned_paragraph

    # Find the sentence that would contain the citation (by position)
    citation_relative_pos = citation_start - paragraph_start

    # Calculate approximate sentence positions
    best_sentence = sentences[0]  # fallback
    best_score = float('inf')

    char_count = 0
    for sentence in sentences:
        if not sentence.strip():
            continue

        sentence_start = char_count
        sentence_end = char_count + len(sentence)

        # Distance from citation position to this sentence
        distance = abs(citation_relative_pos - (sentence_start + sentence_end) // 2)

        # Prefer longer, more informative sentences
        informativeness = len(sentence) + (50 if any(keyword in sentence.lower()
                                                    for keyword in ['apple', 'iphone', 'ios', 'features', 'intelligence']) else 0)

        # Combined score (lower is better)
        score = distance - informativeness

        if score < best_score and len(sentence.strip()) > 20:
            best_score = score
            best_sentence = sentence.strip()

        char_count = sentence_end + 2  # Account for '. ' separator

    # If the best sentence is too long, truncate intelligently
    if len(best_sentence) > 150:
        # Try to find a good breaking point
        words = best_sentence.split()
        if len(words) > 20:
            # Take first ~15 words and add ellipsis
            best_sentence = ' '.join(words[:15]) + "..."

    return best_sentence


@lru_cache(maxsize=1000)
def _normalize_url(url: str) -> Optional[str]:
    cleaned = url.strip()
    while cleaned and cleaned[-1] in _INVALID_TRAILING_CHARS:
        cleaned = cleaned[:-1]

    if not cleaned:
        return None

    parsed = urlparse(cleaned)
    if not parsed.scheme:
        cleaned = f"https://{cleaned}"
        parsed = urlparse(cleaned)
    if not parsed.netloc or " " in parsed.netloc or "." not in parsed.netloc:
        return None
    return cleaned


def _match_entities(link_text: str, context: str, entities: List[str]) -> List[str]:
    if not entities:
        return []

    link_lower = link_text.lower()
    context_lower = context.lower()

    matches: List[str] = []
    for entity in entities:
        entity_lower = entity.lower()
        if entity_lower in link_lower or entity_lower in context_lower:
            if entity not in matches:
                matches.append(entity)
    return matches


def _context_window(text: str, start: int, end: int, window: int) -> str:
    left = max(start - window, 0)
    right = min(end + window, len(text))
    snippet = text[left:right].replace("\n", " ")
    return snippet.strip()


def _find_references_section_start(text: str) -> Optional[int]:
    """Find the start position of the References section to exclude citations from it."""
    import re

    # Look for "## References" heading (case insensitive)
    references_pattern = re.compile(r'^##\s+References?\s*$', re.MULTILINE | re.IGNORECASE)
    match = references_pattern.search(text)

    if match:
        return match.start()

    # Also check for "# References" (single hash)
    references_pattern_alt = re.compile(r'^#\s+References?\s*$', re.MULTILINE | re.IGNORECASE)
    match_alt = references_pattern_alt.search(text)

    if match_alt:
        return match_alt.start()

    return None


__all__ = [
    "CitationExtractionError",
    "ExtractedCitation",
    "extract_citations",
]
