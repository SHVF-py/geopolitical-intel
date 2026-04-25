"""
deduplication/deduplicator.py

Stage 3: Three-pass Deduplication (PRD §5).

Pass 1 — URL deduplication    : exact URL match (cross-day via persistent store)
Pass 2 — Content hash dedup   : SHA-256(normalized_text) match
Pass 3 — Semantic dedup       : cosine_similarity(emb_a, emb_b) > 0.85 → duplicate

Embeddings for semantic dedup are computed here using the embedding model
(all-MiniLM-L6-v2) on the full clean_text (NOT chunked — chunking is for
clustering; here we need a single compact vector for fast dedup comparison).

Results are persisted to the embedding store for cross-day dedup.

Input:  List[CleanArticle]
Output: List[CleanArticle]   (duplicates removed)
"""

import hashlib
import numpy as np
from typing import List, Dict, Set

from config import config
from utils.logger import get_logger
from utils.helpers import cosine_similarity, timer
from storage.embedding_store import (
    load_seen_urls,
    save_seen_urls,
    load_seen_hashes,
    save_seen_hashes,
    load_seen_embeddings,
    save_new_embeddings,
)

logger = get_logger("deduplication.deduplicator")

# ── Lazy-loaded embedding model ──────────────────────────────────────────────
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.debug(f"[DEDUP] Embedding model loaded: {config.EMBEDDING_MODEL}")
    return _embed_model


def _embed_text(text: str) -> np.ndarray:
    """Embed a single text string for semantic dedup comparison."""
    model = _get_embed_model()
    return model.encode(text, convert_to_numpy=True, show_progress_bar=False)


# ─────────────────────────────────────────────
# Pass 1 — URL Deduplication
# ─────────────────────────────────────────────

def _dedup_by_url(
    articles: List[Dict],
    seen_urls: Set[str],
) -> tuple[List[Dict], Set[str]]:
    """
    Remove articles whose URL has already been seen (cross-day).
    Returns (kept_articles, newly_seen_urls).
    """
    kept = []
    new_urls: Set[str] = set()
    removed = 0

    for article in articles:
        url = article["url"]
        if url in seen_urls or url in new_urls:
            removed += 1
            logger.debug(f"[DEDUP-URL] Duplicate URL: {url[:80]}")
        else:
            new_urls.add(url)
            kept.append(article)

    logger.info(f"[DEDUP] Pass 1 (URL): removed {removed}, kept {len(kept)}")
    return kept, new_urls


# ─────────────────────────────────────────────
# Pass 2 — Content Hash Deduplication
# ─────────────────────────────────────────────

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dedup_by_hash(
    articles: List[Dict],
    seen_hashes: Set[str],
) -> tuple[List[Dict], Set[str]]:
    """
    Remove articles whose SHA-256 content hash has been seen.
    Returns (kept_articles, new_hashes).
    """
    kept = []
    new_hashes: Set[str] = set()
    removed = 0

    for article in articles:
        h = _content_hash(article["clean_text"])
        if h in seen_hashes or h in new_hashes:
            removed += 1
            logger.debug(f"[DEDUP-HASH] Duplicate content: {article['url'][:80]}")
        else:
            new_hashes.add(h)
            kept.append(article)

    logger.info(f"[DEDUP] Pass 2 (Hash): removed {removed}, kept {len(kept)}")
    return kept, new_hashes


# ─────────────────────────────────────────────
# Pass 3 — Semantic Deduplication
# ─────────────────────────────────────────────

def _dedup_by_semantics(
    articles: List[Dict],
    seen_embeddings: List[np.ndarray],
) -> tuple[List[Dict], List[np.ndarray]]:
    """
    Remove articles whose embedding is within cosine distance of DEDUP_SIMILARITY_THRESHOLD
    to any already-seen embedding.

    Returns (kept_articles, new_embeddings_for_kept).
    """
    threshold = config.DEDUP_SIMILARITY_THRESHOLD
    kept = []
    new_embeddings: List[np.ndarray] = []
    removed = 0

    # Build combined list of seen + new for incremental comparison
    all_seen = list(seen_embeddings)  # copy

    for article in articles:
        emb = _embed_text(article["clean_text"])

        is_duplicate = False
        for seen_emb in all_seen:
            sim = cosine_similarity(emb, seen_emb)
            if sim > threshold:
                is_duplicate = True
                logger.debug(
                    f"[DEDUP-SEM] Duplicate (sim={sim:.3f}): {article['url'][:80]}"
                )
                break

        if is_duplicate:
            removed += 1
        else:
            # Store embedding on article for potential reuse in embedding stage
            article["_dedup_embedding"] = emb
            kept.append(article)
            new_embeddings.append(emb)
            all_seen.append(emb)  # guard against duplicates within this batch

    logger.info(f"[DEDUP] Pass 3 (Semantic): removed {removed}, kept {len(kept)}")
    return kept, new_embeddings


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def deduplicate(articles: List[Dict]) -> List[Dict]:
    """
    Run all three deduplication passes on CleanArticle list.
    Persists new seen URLs, hashes, and embeddings to storage.

    Returns deduplicated list of CleanArticle dicts.
    """
    with timer("Deduplication"):
        # Load cross-day state
        seen_urls       = load_seen_urls()
        seen_hashes     = load_seen_hashes()
        seen_embeddings = load_seen_embeddings()

        logger.info(
            f"[DEDUP] Loaded cross-day state: "
            f"{len(seen_urls)} URLs, "
            f"{len(seen_hashes)} hashes, "
            f"{len(seen_embeddings)} embeddings"
        )

        # Pass 1: URL
        articles, new_urls = _dedup_by_url(articles, seen_urls)

        # Pass 2: Hash
        articles, new_hashes = _dedup_by_hash(articles, seen_hashes)

        # Pass 3: Semantic
        articles, new_embeddings = _dedup_by_semantics(articles, seen_embeddings)

        # Persist new state
        save_seen_urls(new_urls)
        save_seen_hashes(new_hashes)
        save_new_embeddings(new_embeddings)

    logger.info(f"[DEDUP] Final output: {len(articles)} unique articles")

    if config.DEBUG:
        for a in articles[:3]:
            print(f"  [DEBUG DEDUP] {a['source_domain']} | {a['url'][:70]}")

    return articles
