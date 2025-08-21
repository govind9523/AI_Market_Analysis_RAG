"""
Document loader with optimized chunking for 8GB systems
Handles text processing and chunking with minimal memory overhead
"""

from typing import List, Dict, Any
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Lightweight document loader and processor"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 77):
        """
        Initialize document loader

        Args:
            chunk_size: Target chunk size in characters (optimized for all-MiniLM-L6-v2)
            chunk_overlap: Overlap between chunks (15% of chunk_size)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_text_file(self, filepath: str) -> str:
        """Load text from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            logger.info(f"Loaded {len(content)} characters from {filepath}")
            return content

        except Exception as e:
            logger.error(f"Failed to load file {filepath}: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove extra blank lines
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text using fixed-size chunking with overlap
        Optimized for memory efficiency on 8GB systems

        Args:
            text: Input text to chunk

        Returns:
            List of document chunks with metadata
        """
        try:
            # Clean text first
            text = self.clean_text(text)

            chunks = []
            start = 0
            chunk_id = 0

            while start < len(text):
                # Calculate end position
                end = start + self.chunk_size

                # If this is not the last chunk, try to break at sentence boundary
                if end < len(text):
                    # Look for sentence ending in the last 20% of chunk
                    search_start = max(start + int(self.chunk_size * 0.8), start)
                    search_text = text[search_start:end + 50]  # Look a bit ahead

                    # Find sentence boundaries
                    sentence_endings = []
                    for i, char in enumerate(search_text):
                        if char in '.!?' and i > 0:
                            # Check if this is likely end of sentence (followed by space/newline)
                            if i + 1 < len(search_text) and search_text[i + 1] in ' \n\t':
                                sentence_endings.append(search_start + i + 1)

                    if sentence_endings:
                        # Use the last sentence ending within reasonable range
                        end = sentence_endings[-1]

                # Extract chunk
                chunk_text = text[start:end].strip()

                if chunk_text:  # Only add non-empty chunks
                    chunks.append({
                        'content': chunk_text,
                        'metadata': {
                            'chunk_id': chunk_id,
                            'start_pos': start,
                            'end_pos': end,
                            'length': len(chunk_text),
                            'overlap': self.chunk_overlap if chunk_id > 0 else 0
                        }
                    })
                    chunk_id += 1

                # Move start position with overlap
                if end >= len(text):
                    break

                start = max(end - self.chunk_overlap, start + 1)

            logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")

            # Log chunk statistics
            chunk_lengths = [len(chunk['content']) for chunk in chunks]
            if chunk_lengths:
                avg_length = sum(chunk_lengths) / len(chunk_lengths)
                logger.info(f"Average chunk length: {avg_length:.1f} characters")
                logger.info(f"Chunk range: {min(chunk_lengths)}-{max(chunk_lengths)} characters")

            return chunks

        except Exception as e:
            logger.error(f"Text chunking failed: {e}")
            raise

    def load_and_chunk(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load file and chunk it in one step

        Args:
            filepath: Path to text file

        Returns:
            List of document chunks ready for embedding
        """
        try:
            # Load text
            text = self.load_text_file(filepath)

            # Add file metadata to all chunks
            file_metadata = {
                'source_file': str(filepath),
                'file_size': len(text),
                'processing_timestamp': None  # Could add timestamp if needed
            }

            # Chunk text
            chunks = self.chunk_text(text)

            # Add file metadata to each chunk
            for chunk in chunks:
                chunk['metadata'].update(file_metadata)

            return chunks

        except Exception as e:
            logger.error(f"Load and chunk failed: {e}")
            raise

    def estimate_memory_usage(self, text_length: int) -> Dict[str, float]:
        """
        Estimate memory usage for chunking given text

        Args:
            text_length: Length of text in characters

        Returns:
            Dictionary with memory estimates in MB
        """
        # Rough estimates based on typical usage
        num_chunks = (text_length // self.chunk_size) + 1

        # Each chunk stores text + metadata (rough estimates)
        bytes_per_chunk = self.chunk_size * 1.5  # Text + overhead
        total_chunk_memory = (num_chunks * bytes_per_chunk) / 1024 / 1024

        return {
            "estimated_chunks": num_chunks,
            "chunk_memory_mb": round(total_chunk_memory, 2),
            "original_text_mb": round(text_length / 1024 / 1024, 2),
            "total_memory_mb": round((text_length + num_chunks * bytes_per_chunk) / 1024 / 1024, 2)
        }
