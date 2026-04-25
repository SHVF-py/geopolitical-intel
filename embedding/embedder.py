"""
embedding/embedder.py

Stage 5: Chunking + Embedding (PRD §7 — CRITICAL DESIGN).

Design:
    Problem: Full-article embedding → semantic dilution / noisy clustering.
    Solution: Sliding Window Chunking then embed each chunk independently.

    chunk_size    = 250 words
    chunk_overlap = 50 words

    For each article:
        - Split clean_text into overlapping word-chunks
        - Embed each chunk with all-MiniLM-L6-v2
        - Store ALL chunk embeddings on the article
        - article_embedding is resolved DURING clustering (max similarity to centroid)

    This preserves factual density and improves clustering quality.

Input:  List[CleanArticle]
Output: List[EmbeddedArticle]

EmbeddedArticle = CleanArticle + {
    "chunks":           List[str],
    "chunk_embeddings": List[np.ndarray],
    "article_embedding": np.ndarray   # placeholder — set during clustering
}
"""

import numpy as np
from typing import List, Dict

from config import config
from utils.logger import get_logger
from utils.helpers import sliding_window_chunks, timer

logger = get_logger("embedding.embedder")

_embed_model = None


def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[EMBED] Loading model: {config.EMBEDDING_MODEL}")
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("[EMBED] Embedding model ready")
    return _embed_model


def _chunk_article(article: Dict) -> List[str]:
    """
    Split article's clean_text into overlapping word-chunks.
    Returns list of chunk strings.
    """
    words = article["clean_text"].split()
    return sliding_window_chunks(words, config.CHUNK_SIZE, config.CHUNK_OVERLAP)


def _embed_chunks(chunks: List[str]) -> List[np.ndarray]:
    """Embed a list of chunk strings. Returns list of numpy vectors."""
    model = _get_model()
    if not chunks:
        return []
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return list(embeddings)


def embed_articles(articles: List[Dict]) -> List[Dict]:
    """
    Chunk and embed all articles.

    Sets on each article:
        chunks:           List[str]
        chunk_embeddings: List[np.ndarray]
        article_embedding: np.ndarray  (mean of chunk embeddings as initial placeholder;
                                        will be replaced with max-sim chunk during clustering)

    Returns list of EmbeddedArticle dicts.
    """
    embedded = []

    with timer("Chunking + Embedding"):
        for article in articles:
            try:
                # Reuse embedding computed during semantic dedup if available
                # (avoids re-embedding the same text for single-chunk articles)
                chunks = _chunk_article(article)

                # Reuse the embedding already computed during semantic dedup
                # for single-chunk articles to avoid redundant model inference.
                dedup_emb = article.pop("_dedup_embedding", None)
                if dedup_emb is not None and len(chunks) == 1:
                    chunk_embeddings = [dedup_emb]
                else:
                    chunk_embeddings = _embed_chunks(chunks)

                if not chunk_embeddings:
                    logger.warning(
                        f"[EMBED] No chunks produced for {article.get('url','')[:80]}. Skipping."
                    )
                    continue

                # Initial article_embedding = mean of chunks (placeholder)
                # This will be replaced in the clustering stage with the
                # max-similarity chunk to the cluster centroid.
                article_embedding = np.mean(chunk_embeddings, axis=0)

                article["chunks"]            = chunks
                article["chunk_embeddings"]  = chunk_embeddings
                article["article_embedding"] = article_embedding

                embedded.append(article)

                if config.DEBUG:
                    print(
                        f"  [DEBUG EMBED] {len(chunks)} chunks | "
                        f"emb_dim={article_embedding.shape[0]} | "
                        f"{article.get('url','')[:60]}"
                    )

            except Exception as e:
                logger.error(
                    f"[EMBED] Failed to embed {article.get('url','')[:80]}: {e}"
                )
                # Skip article rather than crash pipeline
                continue

    logger.info(
        f"[EMBED] Input: {len(articles)} | "
        f"Embedded: {len(embedded)} | "
        f"chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}"
    )
    return embedded
