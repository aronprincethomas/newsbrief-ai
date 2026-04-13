from pathlib import Path

import pandas as pd

from src.data_loader import load_news_data
from src.text_chunker import chunk_articles
from src.embedding_generator import EmbeddingGenerator
from src.vector_index import VectorIndex
from src.summarizer import NewsSummarizer
from src.cache_manager import (
    cache_is_valid,
    save_hash,
    INDEX_PATH,
    CHUNKS_PATH,
    CACHE_DIR,
)


class QueryEngine:

    def __init__(self, data_path):
        data_path = Path(data_path)
        self.embedder = EmbeddingGenerator()

        if cache_is_valid(data_path):
            # ---- FAST PATH: load from disk ----
            print("✅ Loading cached FAISS index...")
            self.chunked_df = pd.read_parquet(CHUNKS_PATH)
            embedding_dim = self.embedder.model.get_sentence_embedding_dimension()
            self.index = VectorIndex(embedding_dim)
            self.index.load(INDEX_PATH)

        else:
            # ---- SLOW PATH: build and cache ----
            print("⚙️  Building index (first run or data changed)...")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            df = load_news_data(data_path)
            self.chunked_df = chunk_articles(df)

            embeddings = self.embedder.generate_embeddings(
                self.chunked_df["content"].tolist()
            )

            self.index = VectorIndex(embeddings.shape[1])
            self.index.add_embeddings(embeddings)

            # Persist to disk
            self.index.save(INDEX_PATH)
            self.chunked_df.to_parquet(CHUNKS_PATH, index=False)
            save_hash(data_path)

            print("✅ Index cached. Future startups will be instant.")

        self.summarizer = NewsSummarizer()

    def ask(self, query, top_k=3):

        query_embedding = self.embedder.generate_embeddings([query])

        distances, indices = self.index.search(query_embedding, top_k=top_k)

        THRESHOLD = 1.2

        if distances[0][0] > THRESHOLD:
            return {"summary": "No relevant information found in dataset.", "sources": []}

        chunk_summaries = []
        seen_article_ids = set()
        sources = []

        for i in indices[0]:
            row = self.chunked_df.iloc[i]
            text = row["content"]
            summary = self.summarizer.summarize(text)
            chunk_summaries.append(summary)

            article_id = row["article_id"]
            if article_id not in seen_article_ids:
                seen_article_ids.add(article_id)
                date = row["date"]
                sources.append({
                    "title": row["title"],
                    "source_url": row["source_url"],
                    "category": row["category"],
                    "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                })

        combined_summary = " ".join(chunk_summaries)

        final_summary = self.summarizer.summarize(
            combined_summary,
            max_length=120,
            min_length=40
        )

        return {"summary": final_summary, "sources": sources}
