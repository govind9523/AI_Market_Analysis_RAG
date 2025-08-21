"""
Lightweight embedding model wrapper using all-MiniLM-L6-v2
Optimized for 8GB RAM constraints
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model

        Args:
            model_name: HuggingFace model name (optimized: all-MiniLM-L6-v2)
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name

        # Load model with CPU-only inference for memory efficiency
        self.model = SentenceTransformer(model_name)
        self.model.eval()  # Set to evaluation mode

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: List of text strings

        Returns:
            numpy array of embeddings
        """
        try:
            # Process in batches for memory efficiency on 8GB systems
            batch_size = 32 if len(texts) > 32 else len(texts)

            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization for better search
            )

            return embeddings

        except Exception as e:
            logger.error(f"Embedding encoding failed: {e}")
            raise

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text efficiently"""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
