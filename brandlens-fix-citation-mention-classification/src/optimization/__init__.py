"""
BrandLens Optimization Module

This module provides token optimization and compression capabilities for BrandLens,
including accurate token counting, cost calculation, semantic chunking, and
high-level content compression orchestration.

Author: BrandLens Development Team
Version: 1.0.0
"""

from .token_counter import (
    TokenCounter,
    TokenizationResult,
    CostBreakdown,
    TokenCountingError,
    UnsupportedModelError,
    count_tokens,
    calculate_cost,
)

from .semantic_chunker import (
    SemanticChunker,
    ChunkMetadata,
    ScoredChunk,
    ChunkingResult,
    SelectionResult,
    chunk_and_compress,
    analyze_compression_potential,
)

from .content_compressor import (
    ContentCompressor,
    CompressionMetrics,
    ChunkImportance,
)

__all__ = [
    # Token Counter
    "TokenCounter",
    "TokenizationResult",
    "CostBreakdown",
    "TokenCountingError",
    "UnsupportedModelError",
    "count_tokens",
    "calculate_cost",
    # Semantic Chunker
    "SemanticChunker",
    "ChunkMetadata",
    "ScoredChunk",
    "ChunkingResult",
    "SelectionResult",
    "chunk_and_compress",
    "analyze_compression_potential",
    # Content Compressor
    "ContentCompressor",
    "CompressionMetrics",
    "ChunkImportance",
]