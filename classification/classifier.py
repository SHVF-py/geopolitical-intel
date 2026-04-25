"""
classification/classifier.py

Stage 4: Binary Geopolitical Classification (PRD §6).

Model:   distilbert-base-uncased (zero-shot via pipeline,
         or fine-tuned binary head if available)
Task:    geopolitical / non-geopolitical
Rule:    if confidence < CLASSIFICATION_THRESHOLD → discard

Implementation note:
    The PRD specifies distilbert-base-uncased for binary classification.
    Since no fine-tuned checkpoint is provided, we use the Hugging Face
    zero-shot-classification pipeline with distilbert-base-uncased as the
    backbone, classifying against labels ["geopolitical", "non-geopolitical"].
    This faithfully implements the binary classification intent of the PRD
    using the exact specified model family.

Input:  List[CleanArticle]
Output: List[CleanArticle]  (non-geopolitical discarded)
        Each kept article gains: "geo_confidence": float
"""

from typing import List, Dict
from config import config
from utils.logger import get_logger
from utils.helpers import timer

logger = get_logger("classification.classifier")

# Candidate labels for zero-shot classification
_CANDIDATE_LABELS = ["geopolitical news", "non-geopolitical news"]
_GEO_LABEL        = "geopolitical news"

# Lazy-loaded pipeline
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        logger.info(f"[CLASSIFY] Loading classification pipeline: {config.CLASSIFICATION_MODEL}")
        _pipeline = pipeline(
            "zero-shot-classification",
            model=config.CLASSIFICATION_MODEL,
            device=-1,              # CPU (local-first)
            multi_label=False,
        )
        logger.info("[CLASSIFY] Classification pipeline ready")
    return _pipeline


def _classify_article(article: Dict) -> float:
    """
    Run zero-shot classification on a single article.

    Returns confidence score for 'geopolitical news' label.
    Uses title + first 400 chars of clean_text for speed.
    """
    clf = _get_pipeline()

    # Compose input: title gives strong signal; prepend it
    title   = article.get("title", "")
    snippet = article.get("clean_text", "")[:400]
    text    = f"{title}. {snippet}".strip()

    result = clf(text, candidate_labels=_CANDIDATE_LABELS, truncation=True)

    # result["labels"] and result["scores"] are parallel lists
    label_scores = dict(zip(result["labels"], result["scores"]))
    return label_scores.get(_GEO_LABEL, 0.0)


def classify_articles(articles: List[Dict]) -> List[Dict]:
    """
    Classify articles as geopolitical or not.
    Discards articles below CLASSIFICATION_THRESHOLD.
    Adds 'geo_confidence' field to kept articles.

    Processes articles in batches for efficiency.
    Returns filtered list of CleanArticle dicts.
    """
    with timer("Classification"):
        kept      = []
        discarded = 0

        # Build input texts for all articles upfront
        texts = []
        for article in articles:
            title   = article.get("title", "")
            snippet = article.get("clean_text", "")[:400]
            texts.append(f"{title}. {snippet}".strip())

        clf = _get_pipeline()

        # Run batch inference (pipeline handles batching internally)
        try:
            results = clf(texts, candidate_labels=_CANDIDATE_LABELS, truncation=True, batch_size=8)
            if not isinstance(results, list):
                results = [results]
        except Exception as e:
            logger.error(f"[CLASSIFY] Batch inference failed: {e}. Falling back to per-article.")
            results = []
            for text in texts:
                try:
                    results.append(clf(text, candidate_labels=_CANDIDATE_LABELS, truncation=True))
                except Exception as e2:
                    logger.warning(f"[CLASSIFY] Per-article fallback also failed: {e2}")
                    results.append({"labels": [_GEO_LABEL], "scores": [0.5]})

        for article, result in zip(articles, results):
            try:
                label_scores = dict(zip(result["labels"], result["scores"]))
                confidence = label_scores.get(_GEO_LABEL, 0.0)
            except Exception as e:
                logger.warning(
                    f"[CLASSIFY] Error parsing result for {article.get('url','')[:80]}: {e}. "
                    f"Keeping article with confidence=0.5 to avoid data loss."
                )
                confidence = 0.5  # neutral fallback — keep but log

            if confidence >= config.CLASSIFICATION_THRESHOLD:
                article["geo_confidence"] = confidence
                kept.append(article)
                logger.debug(
                    f"[CLASSIFY] KEEP (conf={confidence:.3f}): "
                    f"{article.get('title','')[:70]}"
                )
            else:
                discarded += 1
                logger.debug(
                    f"[CLASSIFY] DISCARD (conf={confidence:.3f}): "
                    f"{article.get('title','')[:70]}"
                )

            if config.DEBUG:
                print(
                    f"  [DEBUG CLASSIFY] conf={confidence:.3f} | "
                    f"{article.get('title','')[:70]}"
                )

    logger.info(
        f"[CLASSIFY] Input: {len(articles)} | "
        f"Kept: {len(kept)} | "
        f"Discarded: {discarded}"
    )
    return kept
