"""
summarization/fact_selector.py

Summarization Step 10.4: Fact Selection — Token Budget Control (PRD §10.4).

Constraints:
    min_sentences = MIN_SUMMARY_SENTENCES  (7)
    max_sentences = MAX_SUMMARY_SENTENCES  (12)

Selection Priority:
    1. Frequency across articles  (sentences that appear in more articles rank higher)
    2. Centrality score           (TextRank-style centrality over the merged pool)

Input:
    deduplicated_sentences: List[str]   — merged, deduped sentence pool
    cluster_centroid:       np.ndarray  — for centrality computation
    original_per_article:   List[List[str]]  — for frequency scoring

Output: List[str]  (7–12 selected fact sentences)
"""

import numpy as np
from typing import List

from config import config
from utils.logger import get_logger
from utils.helpers import cosine_similarity

logger = get_logger("summarization.fact_selector")

_embed_model = None


def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embed_model


def _frequency_score(sentence: str, per_article_sentences: List[List[str]]) -> float:
    """
    Score a sentence by how many articles contained a similar sentence.
    Exact-match on normalized lowercase text — sufficient for extractive sentences.
    """
    norm = sentence.lower().strip()
    count = sum(
        1 for article_sents in per_article_sentences
        if any(s.lower().strip() == norm for s in article_sents)
    )
    return count / max(len(per_article_sentences), 1)


def _centrality_scores(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
    Centrality = cosine similarity of each sentence embedding to the cluster centroid.
    High centrality → sentence is representative of the cluster topic.
    """
    scores = np.array([
        cosine_similarity(emb, centroid) for emb in embeddings
    ])
    return scores


def select_facts(
    deduplicated_sentences: List[str],
    cluster_centroid: np.ndarray,
    original_per_article: List[List[str]],
) -> List[str]:
    """
    Select 7–12 sentences from the deduplicated pool.

    Args:
        deduplicated_sentences: Merged + deduped sentence pool.
        cluster_centroid:       Centroid embedding of the cluster.
        original_per_article:   Sentences per article (for frequency scoring).

    Returns:
        Selected fact sentences (7–12 items).
    """
    sentences = deduplicated_sentences

    if len(sentences) <= config.MIN_SUMMARY_SENTENCES:
        logger.debug(
            f"[FACT_SELECT] Pool size {len(sentences)} <= min {config.MIN_SUMMARY_SENTENCES}. "
            f"Returning all."
        )
        return sentences

    if len(sentences) <= config.MAX_SUMMARY_SENTENCES:
        logger.debug(
            f"[FACT_SELECT] Pool size {len(sentences)} within budget. Returning all."
        )
        return sentences

    # Need to select MAX_SUMMARY_SENTENCES from a larger pool
    try:
        model = _get_model()
        embeddings = model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
    except Exception as e:
        logger.error(
            f"[FACT_SELECT] Embedding failed: {e}. "
            f"Returning first {config.MAX_SUMMARY_SENTENCES} sentences."
        )
        return sentences[:config.MAX_SUMMARY_SENTENCES]

    # Score each sentence: frequency (priority 1) + centrality (priority 2)
    freq_scores       = np.array([_frequency_score(s, original_per_article) for s in sentences])
    centrality_scores = _centrality_scores(embeddings, cluster_centroid)

    # Normalise each to [0, 1]
    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)

    combined = 0.6 * _norm(freq_scores) + 0.4 * _norm(centrality_scores)

    # Select top MAX_SUMMARY_SENTENCES by combined score,
    # then sort by original index to preserve reading order.
    top_indices = sorted(
        np.argsort(combined)[-config.MAX_SUMMARY_SENTENCES:].tolist()
    )

    selected = [sentences[i] for i in top_indices]

    logger.info(
        f"[FACT_SELECT] Pool: {len(sentences)} | "
        f"Selected: {len(selected)} "
        f"(min={config.MIN_SUMMARY_SENTENCES}, max={config.MAX_SUMMARY_SENTENCES})"
    )

    if config.DEBUG:
        for s in selected:
            print(f"  [DEBUG FACT] {s[:100]}")

    return selected
