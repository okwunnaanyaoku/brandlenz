# coding: ascii
"""Response parsing utilities for BrandLens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


class ResponseParseError(Exception):
    """Raised when a Gemini response cannot be parsed into structured data."""


@dataclass
class ParsedReference:
    index: int
    title: str
    url: str


@dataclass
class SectionContent:
    heading: str
    text: str
    citations: List[int]


@dataclass
class ParsedResponse:
    title: Optional[str]
    executive_summary: SectionContent
    sections: List[SectionContent]
    references: List[ParsedReference]


_HEADING_PATTERN = re.compile(r"^##\s+(?P<heading>.+)$", re.MULTILINE)
_REFERENCE_LINE_PATTERN = re.compile(r"^\[(?P<index>\d+)]\s+(?P<title>.+?)\s+-\s+(?P<url>https?://\S+)$")
_CITATION_PATTERN = re.compile(r"\[([0-9, ]+)\]")
_TITLE_PATTERN = re.compile(r"^#\s+(?P<title>.+)$", re.MULTILINE)


def parse_response(markdown: str) -> ParsedResponse:
    """Parse the Markdown response returned by Gemini into structured data."""

    if not markdown or not markdown.strip():
        raise ResponseParseError("Response content is empty")

    content = markdown.strip()
    title_match = _TITLE_PATTERN.search(content)
    title = title_match.group("title").strip() if title_match else None

    all_sections = _extract_sections(content)
    if not all_sections:
        raise ResponseParseError("No section headings found in response")

    section_map = {section.heading.lower(): section for section in all_sections}

    exec_section = section_map.get("executive summary")
    if exec_section is None:
        raise ResponseParseError("Executive Summary section is required")

    references_section = section_map.get("references")
    if references_section is None:
        raise ResponseParseError("References section is required")

    references = _parse_references(references_section.text)
    if not references:
        raise ResponseParseError("References section is empty")

    reference_map: Dict[int, ParsedReference] = {ref.index: ref for ref in references}

    body_sections = [
        section
        for section in all_sections
        if section.heading.lower() not in {"executive summary", "references"}
    ]

    executive_summary = _attach_citations(exec_section)
    parsed_sections = [_attach_citations(section) for section in body_sections]

    _validate_citations([executive_summary] + parsed_sections, reference_map.values())

    return ParsedResponse(
        title=title,
        executive_summary=executive_summary,
        sections=parsed_sections,
        references=references,
    )


def _extract_sections(markdown: str) -> List[SectionContent]:
    matches = list(_HEADING_PATTERN.finditer(markdown))
    sections: List[SectionContent] = []
    for idx, match in enumerate(matches):
        heading = match.group("heading").strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        text = markdown[start:end].strip()
        sections.append(SectionContent(heading=heading, text=text, citations=[]))
    return sections


def _parse_references(raw_text: str) -> List[ParsedReference]:
    references: List[ParsedReference] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _REFERENCE_LINE_PATTERN.match(stripped)
        if not match:
            raise ResponseParseError(f"Invalid reference line: {stripped}")
        index = int(match.group("index"))
        title = match.group("title").strip()
        url = match.group("url").strip()
        references.append(ParsedReference(index=index, title=title, url=url))
    return references


def _attach_citations(section: SectionContent) -> SectionContent:
    citations = []
    for match in _CITATION_PATTERN.findall(section.text):
        # Handle comma-separated citations like "1, 2, 3"
        citation_numbers = [int(num.strip()) for num in match.split(',') if num.strip().isdigit()]
        citations.extend(citation_numbers)
    return SectionContent(heading=section.heading, text=section.text, citations=citations)


def _validate_citations(sections: Iterable[SectionContent], references: Iterable[ParsedReference]) -> None:
    reference_map = {ref.index: ref for ref in references}

    unknown: List[int] = []
    referenced = set()
    for section in sections:
        for citation in section.citations:
            if citation not in reference_map:
                unknown.append(citation)
            else:
                referenced.add(citation)

    if unknown:
        unique = sorted(set(unknown))
        raise ResponseParseError(
            "Unknown citation references: " + ", ".join(str(num) for num in unique)
        )

    missing = sorted(num for num in reference_map if num not in referenced)
    if missing:
        raise ResponseParseError(
            "References not cited in body: " + ", ".join(str(num) for num in missing)
        )


def _extract_title(content: str) -> Optional[str]:
    """Extract title from markdown content."""
    title_match = _TITLE_PATTERN.search(content)
    if title_match:
        title = title_match.group("title").strip()
        return title if title else None
    return None


__all__ = [
    "ResponseParseError",
    "ParsedReference",
    "SectionContent",
    "ParsedResponse",
    "parse_response",
    "REQUIRED_SECTIONS"
]
