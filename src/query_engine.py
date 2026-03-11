from src.data_loader import load_news_data
from src.text_chunker import chunk_articles
from src.embedding_generator import EmbeddingGenerator
from src.vector_index import VectorIndex
from src.summarizer import NewsSummarizer


class QueryEngine:

    def __init__(self, data_path):

        self.df = load_news_data(data_path)
        self.chunked_df = chunk_articles(self.df)

        self.embedder = EmbeddingGenerator()

        self.embeddings = self.embedder.generate_embeddings(
            self.chunked_df["content"].tolist()
        )

        embedding_dim = self.embeddings.shape[1]

        self.index = VectorIndex(embedding_dim)
        self.index.add_embeddings(self.embeddings)

        self.summarizer = NewsSummarizer()

    def ask(self, query, top_k=3):

        query_embedding = self.embedder.generate_embeddings([query])

        distances, indices = self.index.search(query_embedding, top_k=top_k)

        THRESHOLD = 1.2

        if distances[0][0] > THRESHOLD:
            return "No relevant information found in dataset."

        chunk_summaries = []

        for i in indices[0]:
            text = self.chunked_df.iloc[i]["content"]
            summary = self.summarizer.summarize(text)
            chunk_summaries.append(summary)

        combined_summary = " ".join(chunk_summaries)

        final_summary = self.summarizer.summarize(
            combined_summary,
            max_length=120,
            min_length=40
        )

        return final_summary
