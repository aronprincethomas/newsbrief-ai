import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = {
    "title",
    "content",
    "category",
    "date",
    "source_url"
}

def load_news_data(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=["content"])
    df = df[df["content"].str.len() > 50]

    df["title"] = df["title"].fillna("").str.strip()
    df["content"] = df["content"].str.strip()
    df["category"] = df["category"].fillna("General").str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df
