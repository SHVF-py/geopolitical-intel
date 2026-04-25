"""
summarization/sentence_dedup.py

Summarization Step 10.3: Merge + Deduplicate Extracted Sentences (PRD §10.3).

Process:
    1. Combine all extracted sentences from all representative articles
    2. Remove sentences where cosine_similarity > SENTENCE_SIM_THRESHOLD (0.85)
       (keep the first occurrence — preserves order and quality)

Input:  List[List[str]]  (extracted sentences per article)
Output: List[str]        (merged, deduplicated sentence list)
"""

import numpy as np
from typing import List

from config import config
from utils.logger import get_logger
from utils.helpers import cosine_similarity

logger = get_logger("summarization.sentence_dedup")

_embed_model = None


def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embed_model


def merge_and_dedup_sentences(sentences_per_article: List[List[str]]) -> List[str]:
    """
    Merge sentences from all articles into one pool, then remove near-duplicates.

    Args:
        sentences_per_article: Each inner list is extracted sentences from one article.

    Returns:
        Deduplicated list of sentences, preserving order of first occurrence.
    """
    # Flatten
    all_sentences: List[str] = []
    for article_sentences in sentences_per_article:
        all_sentences.extend(article_sentences)

    if not all_sentences:
        return []

    if len(all_sentences) == 1:
        return all_sentences

    try:
        model = _get_model()
        embeddings = model.encode(
            all_sentences,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
    except Exception as e:
        logger.error(f"[SENT_DEDUP] Embedding failed: {e}. Returning merged without dedup.")
        return all_sentences

    threshold = config.SENTENCE_SIM_THRESHOLD
    kept_sentences: List[str]     = []
    kept_embeddings: List[np.ndarray] = []
    removed = 0

    for i, (sentence, emb) in enumerate(zip(all_sentences, embeddings)):
        is_duplicate = False

        for kept_emb in kept_embeddings:
            sim = cosine_similarity(emb, kept_emb)
            if sim > threshold:
                is_duplicate = True
                logger.debug(
                    f"[SENT_DEDUP] Removed near-duplicate (sim={sim:.3f}): "
                    f"{sentence[:60]}"
                )
                break

        if not is_duplicate:
            kept_sentences.append(sentence)
            kept_embeddings.append(emb)
        else:
            removed += 1

    logger.info(
        f"[SENT_DEDUP] Input: {len(all_sentences)} | "
        f"Removed: {removed} | "
        f"Output: {len(kept_sentences)}"
    )
    return kept_sentences
