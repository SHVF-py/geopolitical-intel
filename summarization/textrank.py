"""
summarization/textrank.py

Summarization Step 10.2: Per-Article Extractive Compression via TextRank
(PRD §10.2).

Method: TextRank (NOT LLM)

Steps per article:
    1. Sentence tokenize  (from article["sentences"])
    2. Embed sentences    (MiniLM — same model used elsewhere)
    3. Build similarity graph (sentence × sentence cosine similarity matrix)
    4. Run TextRank       (power iteration on similarity matrix)
    5. Select top TEXTRANK_TOP_SENTENCES sentences

Input:  Single EmbeddedArticle dict
Output: List[str]  (top extracted sentences, in original order)
"""

import numpy as np
from typing import List, Dict

from config import config
from utils.logger import get_logger
from utils.helpers import cosine_similarity

logger = get_logger("summarization.textrank")

_embed_model = None


def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embed_model


def _embed_sentences(sentences: List[str]) -> np.ndarray:
    """Returns (N, D) embedding matrix for a list of sentences."""
    model = _get_model()
    return model.encode(
        sentences,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
    )


def _build_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix using vectorized numpy ops.
    Returns (N, N) matrix with zeros on diagonal.
    """
    # Normalize rows to unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms
    # All-pairs cosine similarity via matrix multiply
    sim_matrix = normed @ normed.T
    np.fill_diagonal(sim_matrix, 0.0)
    return sim_matrix


def _textrank_scores(sim_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100) -> np.ndarray:
    """
    Run TextRank power iteration on the similarity matrix.

    Returns array of sentence scores.
    """
    n = sim_matrix.shape[0]
    if n == 1:
        return np.array([1.0])

    # Row-normalize (transition probabilities)
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for fully-disconnected rows
    row_sums[row_sums == 0] = 1.0
    transition = sim_matrix / row_sums

    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * transition.T.dot(scores)
        if np.allclose(scores, new_scores, atol=1e-6):
            break
        scores = new_scores

    return scores


def extract_top_sentences(article: Dict) -> List[str]:
    """
    Run TextRank on a single article's sentences.

    Returns top TEXTRANK_TOP_SENTENCES sentences in their original order
    (preserving logical flow).
    """
    sentences = article.get("sentences", [])

    # Filter out very short sentences (likely navigation artefacts)
    sentences = [s for s in sentences if len(s.split()) >= 5]

    if not sentences:
        logger.warning(
            f"[TEXTRANK] No valid sentences in {article.get('url','')[:80]}"
        )
        return []

    k = min(config.TEXTRANK_TOP_SENTENCES, len(sentences))

    if len(sentences) == 1:
        return sentences

    try:
        embeddings  = _embed_sentences(sentences)
        sim_matrix  = _build_similarity_matrix(embeddings)
        scores      = _textrank_scores(sim_matrix)

        # Select top-k by score
        top_indices = sorted(
            np.argsort(scores)[-k:].tolist()
        )  # sort by original index to preserve order

        extracted = [sentences[i] for i in top_indices]

        logger.debug(
            f"[TEXTRANK] {article.get('url','')[:60]} → "
            f"{len(extracted)}/{len(sentences)} sentences extracted"
        )
        return extracted

    except Exception as e:
        logger.error(
            f"[TEXTRANK] Failed for {article.get('url','')[:80]}: {e}. "
            f"Returning first {k} sentences."
        )
        return sentences[:k]
