"""
Text chunking utilities optimized for RAG applications
Provides various chunking strategies with memory efficiency focus
"""

from typing import List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)

class ChunkingStrategy:
    """Base class for chunking strategies"""

    def chunk(self, text: str) -> List[str]:
        """Override in subclasses"""
        raise NotImplementedError

class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with overlap - optimized for 8GB systems"""

    def __init__(self, size: int = 512, overlap: int = 77, separator: str = "\n\n"):
        self.size = size
        self.overlap = overlap  
        self.separator = separator

    def chunk(self, text: str) -> List[str]:
        """Chunk text into fixed-size pieces with overlap"""
        # First try to split on separator
        if self.separator in text:
            sections = text.split(self.separator)
            chunks = []
            current_chunk = ""

            for section in sections:
                # If adding this section would exceed size, save current chunk
                if len(current_chunk) + len(section) > self.size and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap from previous
                    if self.overlap > 0:
                        overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                        current_chunk = overlap_text + " " + section
                    else:
                        current_chunk = section
                else:
                    current_chunk += (self.separator if current_chunk else "") + section

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            return chunks
        else:
            # Fallback to character-based chunking
            return self._character_chunking(text)

    def _character_chunking(self, text: str) -> List[str]:
        """Character-based chunking with overlap"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk.strip())

            if end >= len(text):
                break

            start = end - self.overlap

        return chunks

class SemanticChunking(ChunkingStrategy):
    """Semantic chunking based on sentence boundaries - more CPU intensive"""

    def __init__(self, max_size: int = 512, min_size: int = 100):
        self.max_size = max_size
        self.min_size = min_size

    def chunk(self, text: str) -> List[str]:
        """Chunk text based on sentence boundaries"""
        # Simple sentence splitting (avoiding NLTK for memory efficiency)
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed max_size
            if len(current_chunk) + len(sentence) > self.max_size:
                if len(current_chunk) >= self.min_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Current chunk too small, add sentence anyway
                    current_chunk += " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk.strip() and len(current_chunk) >= self.min_size:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        # Basic sentence splitting regex
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

class ChunkingUtils:
    """Utility functions for chunking operations"""

    @staticmethod
    def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
        """Rough token estimation for text"""
        return int(len(text) / chars_per_token)

    @staticmethod
    def optimize_chunk_size(target_tokens: int, chars_per_token: float = 4.0) -> int:
        """Convert target tokens to character count"""
        return int(target_tokens * chars_per_token)

    @staticmethod
    def analyze_chunks(chunks: List[str]) -> Dict[str, Any]:
        """Analyze chunk statistics"""
        if not chunks:
            return {"error": "No chunks to analyze"}

        lengths = [len(chunk) for chunk in chunks]
        token_estimates = [ChunkingUtils.estimate_tokens(chunk) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_tokens": sum(token_estimates) / len(token_estimates),
            "total_characters": sum(lengths),
            "estimated_total_tokens": sum(token_estimates)
        }

    @staticmethod
    def get_chunking_strategy(strategy_name: str, **kwargs) -> ChunkingStrategy:
        """Factory method for chunking strategies"""
        if strategy_name == "fixed":
            return FixedSizeChunking(**kwargs)
        elif strategy_name == "semantic":  
            return SemanticChunking(**kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}")
