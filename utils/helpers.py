"""
utils/helpers.py — Shared utility functions for text processing and timing.
"""

import time
import unicodedata
import re
from typing import List
from contextlib import contextmanager
from utils.logger import get_logger

logger = get_logger("utils.helpers")


@contextmanager
def timer(stage_name: str):
    """Context manager that logs elapsed time for a pipeline stage."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"[TIMER] {stage_name} completed in {elapsed:.2f}s")


def normalize_text(text: str) -> str:
    """
    Apply text normalization:
    - lowercase
    - strip whitespace
    - normalize unicode (NFC)
    - remove excessive newlines
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r'\n{2,}', '\n', text)       # collapse multiple newlines
    text = re.sub(r'[ \t]+', ' ', text)         # collapse spaces/tabs
    text = text.strip()
    return text


def word_count(text: str) -> int:
    """Return number of whitespace-delimited tokens."""
    return len(text.split())


def extract_domain(url: str) -> str:
    """
    Extract the base domain from a URL for source weighting.
    e.g. 'https://www.reuters.com/article/...' → 'reuters.com'
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        netloc = parsed.netloc
        # strip 'www.' prefix
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def sliding_window_chunks(words: List[str], chunk_size: int, overlap: int) -> List[str]:
    """
    Split a list of words into overlapping chunks.

    Args:
        words:      List of word tokens
        chunk_size: Target words per chunk
        overlap:    Number of words shared between adjacent chunks

    Returns:
        List of chunk strings
    """
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = 1
    i = 0
    while i < len(words):
        chunk = words[i: i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks


def cosine_similarity(vec_a, vec_b) -> float:
    """Compute cosine similarity between two numpy vectors."""
    import numpy as np
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
