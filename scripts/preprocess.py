import pandas as pd
import re
from dateutil import parser

RAW_PATH = "data/raw/news_dataset.csv"
OUT_PATH = "data/processed/news_clean.csv"

def clean_content(text):
    if pd.isna(text):
        return ""

    # Remove location headers like "PARIS (Reuters) -"
    text = re.sub(r"^[A-Z\s]+?\(Reuters\)\s+-\s+", "", text)

    # Remove reporting/editing credits
    text = re.sub(r"Reporting by.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Writing by.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Editing by.*", "", text, flags=re.IGNORECASE)

    # Remove newsletter promos
    text = re.sub(r"Sign up for.*", "", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_date(date_str):
    try:
        return parser.parse(date_str).strftime("%Y-%m-%d")
    except Exception:
        return ""


def main():
    df = pd.read_csv(RAW_PATH)

    # Clean column names (safety)
    df.columns = df.columns.str.strip()

    df_out = pd.DataFrame({
        "title": df["title"],
        "content": df["article"].apply(clean_content),  # ✅ FIX
        "category": df["section"].fillna("General"),
        "date": df["date"].apply(normalize_date),
        "source_url": df["url"]
    })

    # Drop empty or very short articles
    df_out = df_out[df_out["content"].str.len() > 100]

    df_out.to_csv(OUT_PATH, index=False)
    print(f"✅ Clean dataset saved to {OUT_PATH}")



if __name__ == "__main__":
    main()
