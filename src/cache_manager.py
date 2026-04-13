import hashlib
from pathlib import Path

CACHE_DIR = Path("data/cache")
INDEX_PATH = CACHE_DIR / "faiss.index"
CHUNKS_PATH = CACHE_DIR / "chunked_df.parquet"
HASH_PATH = CACHE_DIR / "dataset.hash"


def compute_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file for change detection."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_is_valid(data_path: Path) -> bool:
    """
    Returns True if all cache files exist AND were built from
    the same version of the CSV (hash match).
    """
    if not all([INDEX_PATH.exists(), CHUNKS_PATH.exists(), HASH_PATH.exists()]):
        return False
    current_hash = compute_hash(data_path)
    saved_hash = HASH_PATH.read_text().strip()
    return current_hash == saved_hash


def save_hash(data_path: Path):
    """Persist the CSV hash so future runs can validate the cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    HASH_PATH.write_text(compute_hash(data_path))


def clear_cache():
    """Force a full index rebuild on next startup by removing cache files."""
    for f in [INDEX_PATH, CHUNKS_PATH, HASH_PATH]:
        f.unlink(missing_ok=True)
    print("🗑️  Cache cleared. Index will rebuild on next startup.")
