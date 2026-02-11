import pandas as pd

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_articles(df, chunk_size=300, overlap=50):
    chunked_rows = []

    for idx, row in df.iterrows():
        chunks = chunk_text(
            row["content"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for i, chunk in enumerate(chunks):
            chunked_rows.append({
                "article_id": idx,
                "chunk_id": f"{idx}_{i}",
                "title": row["title"],
                "content": chunk,
                "category": row["category"],
                "date": row["date"],
                "source_url": row["source_url"]
            })

    return pd.DataFrame(chunked_rows)
