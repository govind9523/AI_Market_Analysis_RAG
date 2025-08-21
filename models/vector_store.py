"""
FAISS-based vector store optimized for CPU inference
Memory-efficient implementation for 8GB systems
"""

import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store for document embeddings"""

    def __init__(self, embedding_model, index_type: str = "flat"):
        """
        Initialize vector store

        Args:
            embedding_model: Embedding model instance
            index_type: FAISS index type (flat for CPU efficiency)
        """
        self.embedding_model = embedding_model
        self.dimension = embedding_model.dimension
        self.index_type = index_type

        # Initialize FAISS index (CPU-only for memory efficiency)
        if index_type == "flat":
            # L2 distance, good for normalized embeddings
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors
        else:
            # Could add IVF or other indexes for larger datasets
            self.index = faiss.IndexFlatL2(self.dimension)

        # Store document chunks and metadata
        self.documents = []
        self.metadata = []

        logger.info(f"Initialized FAISS {index_type} index with dimension {self.dimension}")

    async def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to vector store

        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
        """
        try:
            logger.info(f"Adding {len(documents)} documents to vector store...")

            # Extract text content
            texts = [doc['content'] for doc in documents]

            # Generate embeddings in batches (memory efficient)
            embeddings = self.embedding_model.encode(texts)

            # Add to FAISS index
            self.index.add(embeddings.astype(np.float32))

            # Store documents and metadata
            self.documents.extend(texts)
            self.metadata.extend([doc.get('metadata', {}) for doc in documents])

            logger.info(f"Successfully added {len(documents)} documents")
            logger.info(f"Total documents in index: {self.index.ntotal}")

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of search results with scores and metadata
        """
        try:
            # Encode query
            query_embedding = self.embedding_model.encode_single(query)
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)

            # Search FAISS index
            scores, indices = self.index.search(query_vector, top_k)

            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0:  # Valid result
                    results.append({
                        'content': self.documents[idx],
                        'score': float(score),
                        'metadata': self.metadata[idx],
                        'rank': i + 1
                    })

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def save(self, filepath: str):
        """Save vector store to disk"""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")

            # Save documents and metadata
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata,
                    'dimension': self.dimension,
                    'index_type': self.index_type
                }, f)

            logger.info(f"Vector store saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise

    def load(self, filepath: str):
        """Load vector store from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")

            # Load documents and metadata
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
                self.dimension = data['dimension']
                self.index_type = data['index_type']

            logger.info(f"Vector store loaded from {filepath}")
            logger.info(f"Loaded {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise

    @property
    def size(self) -> int:
        """Get number of documents in store"""
        return len(self.documents)
