# Extraction Components Overview

## Citation Extractor (`src/extraction/citation_extractor.py`)
- Scans Markdown for links matching `[text](url)` and normalises each URL (trims trailing punctuation, requires a host, adds `https://` when missing).
- Builds a 160-character context window around each link, flattening newlines so entity searches see the surrounding sentence.
- `_match_entities` keeps entities in the order supplied. It checks both the anchor text and the surrounding context; when a brand name appears in either area the entity is attached to that citation exactly once.

## Mention Detector (`src/extraction/mention_detector.py`)
- `generate_variants` expands a brand into lowercase variants (Inc/Ltd suffixes, possessive forms, optional extras) and sorts them by length so specific forms match before fallbacks.
- `detect_mentions` executes two passes:
  - Exact matches via regex for each variant, avoiding overlaps.
  - Fuzzy matches using `SequenceMatcher` sliding over token windows; matches above the threshold become `Mention` objects with score, span, and context snippet.
- Results are returned in document order as `Mention` dataclasses.

## Entity Recognizer (`src/extraction/entity_recognizer.py`)
- Accepts an optional spaCy model; when present its spans are recorded first (source=`"spacy"`) with the score extracted from common attributes (`score`, `kprob`, etc.).
- A rule-based pass then tags provided brand/competitor terms (source=`"rule"`) while avoiding overlaps with spaCy spans.
- Returns a chronologically ordered list of `RecognizedEntity` items containing text, offsets, label, score, and provenance.
