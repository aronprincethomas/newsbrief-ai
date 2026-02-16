from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        """
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts, batch_size=32):
        """
        Generate embeddings for a list of texts.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
