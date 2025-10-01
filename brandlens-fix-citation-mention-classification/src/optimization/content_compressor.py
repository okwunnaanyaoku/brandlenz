"""Content compression utilities built on semantic chunking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..core.exceptions import ValidationError
from ..core.models import CompressedContent, ModelName
from .semantic_chunker import (
    ChunkingResult,
    SelectionResult,
    SemanticChunker,
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkImportance:
    """Lightweight view of chunk importance for reporting/metrics."""

    chunk_id: str
    score: float
    contains_brand: bool
    selected: bool


@dataclass
class CompressionMetrics:
    """Aggregated metrics describing a compression run."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    tokens_saved: int
    quality_score: float
    coherence_score: float
    brand_preservation: float
    citation_preservation: float
    selection_time_ms: float
    over_compression_corrected: bool
    compression_goal_met: bool
    target_ratio: float
    top_chunks: List[ChunkImportance]

    @property
    def compression_percentage(self) -> float:
        """Return compression ratio expressed as a percentage."""
        return self.compression_ratio * 100.0


class ContentCompressor:
    """High-level orchestrator that turns chunk selections into compressed content."""

    def __init__(
        self,
        *,
        brand_names: Optional[Sequence[str]] = None,
        competitor_names: Optional[Sequence[str]] = None,
        target_compression_ratio: float = 0.65,
        fallback_tolerance: float = 0.05,
        adaptive_ratio_adjustment: bool = True,
        chunker: Optional[SemanticChunker] = None,
    ) -> None:
        if chunker is not None:
            self._chunker = chunker
            self.target_compression_ratio = chunker.target_compression_ratio
        else:
            self._chunker = SemanticChunker(
                brand_names=list(brand_names) if brand_names else None,
                competitor_names=list(competitor_names) if competitor_names else None,
                target_compression_ratio=target_compression_ratio,
            )
            self.target_compression_ratio = target_compression_ratio

        if not 0.0 < self.target_compression_ratio < 1.0:
            raise ValidationError("Target compression ratio must be between 0 and 1")

        self.fallback_tolerance = fallback_tolerance
        self.adaptive_ratio_adjustment = adaptive_ratio_adjustment
        self._compression_history: List[Tuple[int, float, float]] = []  # (tokens, target, actual) for learning

    @property
    def chunker(self) -> SemanticChunker:
        """Expose the underlying semantic chunker for advanced use cases."""
        return self._chunker

    def compress(
        self,
        content: str,
        *,
        model: ModelName = ModelName.GEMINI_FLASH_LATEST,
        target_ratio: Optional[float] = None,
    ) -> Tuple[CompressedContent, CompressionMetrics]:
        """Compress content and return both the payload and metrics."""
        ratio = target_ratio if target_ratio is not None else self.target_compression_ratio
        if not 0.0 < ratio < 1.0:
            raise ValidationError("Compression ratio must be between 0 and 1")

        # Apply adaptive ratio adjustment based on content size and history
        adjusted_ratio = self._adjust_compression_ratio(content, ratio) if self.adaptive_ratio_adjustment else ratio

        chunking_result = self._chunker.chunk_content(content, model)
        selection_result = self._chunker.select_chunks(
            chunking_result.chunks,
            compression_ratio=adjusted_ratio,
        )

        adjusted_selection, fallback_applied = self._apply_overcompression_fallback(
            selection_result,
            chunking_result,
            ratio,
        )

        compressed = self._chunker.create_compressed_content(
            original_content=content,
            selection_result=adjusted_selection,
            model=model,
        )

        metrics = self._build_metrics(
            chunking_result,
            adjusted_selection,
            compressed,
            ratio,
            fallback_applied,
        )

        # Track compression history for adaptive learning
        if self.adaptive_ratio_adjustment:
            self._compression_history.append((
                chunking_result.total_tokens,
                ratio,
                adjusted_selection.compression_ratio
            ))
            # Keep only recent history (last 20 compressions)
            if len(self._compression_history) > 20:
                self._compression_history = self._compression_history[-20:]

        logger.debug(
            "Compression complete",
            extra={
                "original_tokens": metrics.original_tokens,
                "compressed_tokens": metrics.compressed_tokens,
                "compression_ratio": metrics.compression_ratio,
                "quality_score": metrics.quality_score,
                "adjusted_ratio": adjusted_ratio,
                "fallback_applied": fallback_applied,
            },
        )

        return compressed, metrics

    def _adjust_compression_ratio(self, content: str, target_ratio: float) -> float:
        """
        Adaptively adjust compression ratio based on content characteristics and history.

        Args:
            content: Content to be compressed
            target_ratio: Desired compression ratio

        Returns:
            Adjusted compression ratio optimized for content characteristics
        """
        if not self._compression_history:
            return target_ratio

        # Estimate content tokens (rough approximation)
        estimated_tokens = len(content.split()) * 1.3  # ~1.3 tokens per word average

        # Find similar historical compressions (within 20% token count)
        similar_compressions = [
            (target, actual) for tokens, target, actual in self._compression_history
            if abs(tokens - estimated_tokens) / max(tokens, estimated_tokens) < 0.2
        ]

        if not similar_compressions:
            return target_ratio

        # Calculate adjustment based on historical performance
        adjustments = []
        for hist_target, hist_actual in similar_compressions:
            if abs(hist_target - target_ratio) < 0.1:  # Similar target ratios
                # If we consistently under-compress, adjust target higher
                adjustment = hist_actual - hist_target
                adjustments.append(adjustment)

        if adjustments:
            avg_adjustment = sum(adjustments) / len(adjustments)
            # Apply conservative adjustment (max 10% change)
            adjustment_factor = max(-0.1, min(0.1, avg_adjustment))
            adjusted_ratio = target_ratio + adjustment_factor

            # Ensure ratio stays within valid bounds
            adjusted_ratio = max(0.05, min(0.95, adjusted_ratio))

            if abs(adjusted_ratio - target_ratio) > 0.01:  # Only log significant adjustments
                logger.debug(
                    "Adaptive ratio adjustment applied",
                    extra={
                        "original_ratio": target_ratio,
                        "adjusted_ratio": adjusted_ratio,
                        "adjustment": adjustment_factor,
                        "historical_samples": len(adjustments)
                    }
                )

            return adjusted_ratio

        return target_ratio

    def _apply_overcompression_fallback(
        self,
        selection_result: SelectionResult,
        chunking_result: ChunkingResult,
        target_ratio: float,
    ) -> Tuple[SelectionResult, bool]:
        """Keep adding high-value chunks until compression is within tolerance."""
        total_tokens = chunking_result.total_tokens
        if total_tokens <= 0:
            return selection_result, False

        max_ratio = min(0.99, target_ratio + self.fallback_tolerance)
        if selection_result.compression_ratio <= max_ratio:
            return selection_result, False

        logger.debug(
            "Over-compression detected",
            extra={
                "actual_ratio": selection_result.compression_ratio,
                "target_ratio": target_ratio,
                "tolerance": self.fallback_tolerance,
            },
        )

        selected = list(selection_result.selected_chunks)
        selected_ids = {chunk.metadata.chunk_id for chunk in selected}
        rejected = [
            chunk
            for chunk in chunking_result.chunks
            if chunk.metadata.chunk_id not in selected_ids
        ]
        rejected.sort(key=lambda c: c.final_score, reverse=True)

        selected_tokens = sum(chunk.metadata.token_count for chunk in selected)

        for chunk in rejected:
            selected.append(chunk)
            chunk.selected = True
            selected_ids.add(chunk.metadata.chunk_id)
            selected_tokens += chunk.metadata.token_count

            current_ratio = (total_tokens - selected_tokens) / total_tokens
            if current_ratio <= max_ratio:
                break

        final_selected = sorted(selected, key=lambda c: c.metadata.start_pos)
        final_selected_ids = {chunk.metadata.chunk_id for chunk in final_selected}
        final_rejected = [
            chunk
            for chunk in chunking_result.chunks
            if chunk.metadata.chunk_id not in final_selected_ids
        ]
        for chunk in final_rejected:
            chunk.selected = False

        selected_tokens = sum(chunk.metadata.token_count for chunk in final_selected)
        rejected_tokens = total_tokens - selected_tokens
        final_ratio = rejected_tokens / total_tokens if total_tokens else 0.0

        adjusted_result = SelectionResult(
            selected_chunks=final_selected,
            rejected_chunks=final_rejected,
            selected_tokens=selected_tokens,
            rejected_tokens=rejected_tokens,
            compression_ratio=final_ratio,
            target_tokens=selection_result.target_tokens,
            selection_time_ms=selection_result.selection_time_ms,
        )

        logger.debug(
            "Fallback applied",
            extra={"adjusted_ratio": adjusted_result.compression_ratio},
        )

        return adjusted_result, True

    def _build_metrics(
        self,
        chunking_result: ChunkingResult,
        selection_result: SelectionResult,
        compressed: CompressedContent,
        target_ratio: float,
        fallback_applied: bool,
    ) -> CompressionMetrics:
        original_tokens = chunking_result.total_tokens
        compressed_tokens = selection_result.selected_tokens
        tokens_saved = original_tokens - compressed_tokens

        total_paragraphs = (
            max(chunk.metadata.paragraph_index for chunk in chunking_result.chunks) + 1
            if chunking_result.chunks
            else 0
        )
        selected_paragraphs = {
            chunk.metadata.paragraph_index for chunk in selection_result.selected_chunks
        }
        coherence_score = (
            len(selected_paragraphs) / max(1, total_paragraphs)
            if total_paragraphs
            else 1.0
        )

        brand_total = sum(
            1 for chunk in chunking_result.chunks if chunk.metadata.contains_brand_mention
        )
        brand_preserved = sum(
            1 for chunk in selection_result.selected_chunks if chunk.metadata.contains_brand_mention
        )
        brand_preservation = (
            brand_preserved / brand_total if brand_total else 1.0
        )

        citation_total = sum(
            1 for chunk in chunking_result.chunks if chunk.metadata.contains_citation
        )
        citation_preserved = sum(
            1 for chunk in selection_result.selected_chunks if chunk.metadata.contains_citation
        )
        citation_preservation = (
            citation_preserved / citation_total if citation_total else 1.0
        )

        selected_ids = {chunk.metadata.chunk_id for chunk in selection_result.selected_chunks}
        top_chunks = sorted(
            chunking_result.chunks,
            key=lambda chunk: chunk.final_score,
            reverse=True,
        )[:5]
        importance_view = [
            ChunkImportance(
                chunk_id=chunk.metadata.chunk_id,
                score=chunk.final_score,
                contains_brand=chunk.metadata.contains_brand_mention,
                selected=chunk.metadata.chunk_id in selected_ids,
            )
            for chunk in top_chunks
        ]

        goal_threshold = max(0.0, target_ratio - self.fallback_tolerance)
        max_possible_ratio = 0.0
        if chunking_result.chunks and original_tokens > 0:
            min_chunk_tokens = min(
                chunk.metadata.token_count for chunk in chunking_result.chunks
            )
            max_possible_ratio = (original_tokens - min_chunk_tokens) / original_tokens

        compression_goal_met = selection_result.compression_ratio >= goal_threshold
        if not compression_goal_met and max_possible_ratio < goal_threshold:
            compression_goal_met = True

        return CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=selection_result.compression_ratio,
            tokens_saved=tokens_saved,
            quality_score=compressed.quality_score,
            coherence_score=coherence_score,
            brand_preservation=brand_preservation,
            citation_preservation=citation_preservation,
            selection_time_ms=selection_result.selection_time_ms,
            over_compression_corrected=fallback_applied,
            compression_goal_met=compression_goal_met,
            target_ratio=target_ratio,
            top_chunks=importance_view,
        )


__all__ = [
    "ChunkImportance",
    "CompressionMetrics",
    "ContentCompressor",
]
