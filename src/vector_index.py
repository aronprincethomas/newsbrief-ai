import faiss
import numpy as np
from pathlib import Path


class VectorIndex:
    def __init__(self, embedding_dim):
        """
        Initialize FAISS index.
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_embeddings(self, embeddings):
        """
        Add embeddings to the FAISS index.
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        self.index.add(embeddings.astype("float32"))

    def search(self, query_embedding, top_k=5):
        """
        Search for nearest neighbors.
        """
        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = self.index.search(
            query_embedding.astype("float32"),
            top_k
        )

        return distances, indices

    def save(self, path: Path):
        """Persist the FAISS index to disk."""
        faiss.write_index(self.index, str(path))

    def load(self, path: Path):
        """Load a previously saved FAISS index from disk."""
        self.index = faiss.read_index(str(path))
