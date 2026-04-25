"""
main.py — Geopolitical News Intelligence System — Pipeline Orchestrator

Runs the full end-to-end pipeline:
    RSS → Preprocessing → Deduplication → Classification
    → Embedding → Clustering → Ranking
    → Summarization → Email Delivery

Failure handling (PRD §14):
    Case 1 — RSS failure:         retry (handled inside rss_fetcher)
    Case 2 — No articles:         send insufficient data notice
    Case 3 — Clustering failure:  fallback to top-N individual article summaries
    Case 4 — LLM failure:         send bullet-only summaries (handled in formatter)

Usage:
    python main.py
    python main.py --debug    (sets DEBUG=True at runtime)
"""

import sys
import traceback
from datetime import datetime

# ── Allow top-level imports without installing as package ────────────────────
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Parse CLI flags before importing config ──────────────────────────────────
if "--debug" in sys.argv:
    os.environ["GEO_DEBUG"] = "1"

import config.config as _cfg
if os.environ.get("GEO_DEBUG") == "1":
    _cfg.DEBUG = True

# ── Imports ──────────────────────────────────────────────────────────────────
from config import config
from utils.logger import get_logger
from utils.helpers import timer

from storage.embedding_store import purge_old_data

from ingestion.rss_fetcher       import fetch_all_feeds
from preprocessing.preprocessor  import preprocess_articles
from deduplication.deduplicator  import deduplicate
from classification.classifier   import classify_articles
from embedding.embedder          import embed_articles
from clustering.clusterer        import cluster_articles
from ranking.ranker              import rank_and_select
from summarization.summarizer    import summarize_all
from mailer.formatter             import format_email
from mailer.sender                import (
    send_email,
    send_error_alert,
    send_insufficient_data_notice,
)

logger = get_logger("main")


# ─────────────────────────────────────────────
# Clustering failure fallback (PRD §14 Case 3)
# ─────────────────────────────────────────────

def _clustering_fallback(articles: list) -> list:
    """
    When clustering fails entirely, treat each of the top-N articles
    as its own single-article cluster (PRD §14 Case 3).
    """
    logger.warning(
        "[MAIN] Clustering fallback: using top articles as individual events."
    )

    # Sort by geo_confidence descending; take top TOP_N_EVENTS
    sorted_articles = sorted(
        articles,
        key=lambda a: a.get("geo_confidence", 0.5),
        reverse=True
    )[:config.TOP_N_EVENTS]

    import numpy as np
    clusters = []
    for i, article in enumerate(sorted_articles):
        emb = article.get("article_embedding", np.zeros(384))
        clusters.append({
            "cluster_id":    f"fallback_{i}",
            "articles":      [article],
            "centroid":      emb,
            "size":          1,
            "score":         article.get("geo_confidence", 0.5),
            "recency_score": 0.0,
            "source_diversity": 1,
        })
    return clusters


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline() -> bool:
    """
    Execute the full pipeline. Returns True on success, False on critical failure.
    """
    run_start = datetime.now()
    logger.info("=" * 60)
    logger.info(f"[MAIN] Pipeline started at {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"[MAIN] DEBUG={config.DEBUG}")
    logger.info("=" * 60)

    try:
        # ── Storage cleanup (TTL) ─────────────────────────────────────
        purge_old_data()

        # ── Stage 1: RSS Ingestion ────────────────────────────────────
        raw_articles = fetch_all_feeds()
        logger.info(f"[MAIN] Stage 1 complete: {len(raw_articles)} raw articles")

        # ── Stage 2: Preprocessing ────────────────────────────────────
        clean_articles = preprocess_articles(raw_articles)
        logger.info(f"[MAIN] Stage 2 complete: {len(clean_articles)} clean articles")

        # ── Case 2: No articles after preprocessing ───────────────────
        if not clean_articles:
            logger.error("[MAIN] No articles after preprocessing. Sending notice.")
            send_insufficient_data_notice()
            return False

        # ── Stage 3: Deduplication ────────────────────────────────────
        deduped = deduplicate(clean_articles)
        logger.info(f"[MAIN] Stage 3 complete: {len(deduped)} unique articles")

        if not deduped:
            logger.error("[MAIN] No articles after deduplication. Sending notice.")
            send_insufficient_data_notice()
            return False

        # ── Stage 4: Classification ───────────────────────────────────
        geo_articles = classify_articles(deduped)
        logger.info(f"[MAIN] Stage 4 complete: {len(geo_articles)} geopolitical articles")

        if not geo_articles:
            logger.error("[MAIN] No geopolitical articles found. Sending notice.")
            send_insufficient_data_notice()
            return False

        # ── Stage 5: Embedding ────────────────────────────────────────
        embedded = embed_articles(geo_articles)
        logger.info(f"[MAIN] Stage 5 complete: {len(embedded)} embedded articles")

        if not embedded:
            logger.error("[MAIN] No articles after embedding. Sending notice.")
            send_insufficient_data_notice()
            return False

        # ── Stage 6: Clustering ───────────────────────────────────────
        clustering_ok = True
        try:
            clusters = cluster_articles(embedded)
            logger.info(f"[MAIN] Stage 6 complete: {len(clusters)} clusters")
        except Exception as e:
            logger.error(f"[MAIN] Clustering failed: {e}\n{traceback.format_exc()}")
            clusters     = _clustering_fallback(embedded)
            clustering_ok = False

        if not clusters:
            logger.error("[MAIN] No clusters formed. Sending notice.")
            send_insufficient_data_notice()
            return False

        # ── Stage 7: Ranking ──────────────────────────────────────────
        if clustering_ok:
            top_clusters = rank_and_select(clusters)
        else:
            # Fallback clusters already scored — take as-is
            top_clusters = clusters[:config.TOP_N_EVENTS]

        logger.info(f"[MAIN] Stage 7 complete: {len(top_clusters)} top clusters selected")

        # ── Stage 8: Summarization ────────────────────────────────────
        events = summarize_all(top_clusters)
        logger.info(f"[MAIN] Stage 8 complete: {len(events)} summarized events")

        if not events:
            logger.error("[MAIN] Summarization produced no events. Sending notice.")
            send_insufficient_data_notice()
            return False

        # ── Stage 9: Email Formatting ─────────────────────────────────
        subject, body = format_email(events)
        logger.info(f"[MAIN] Stage 9 complete: email formatted ({len(body)} chars)")

        if config.DEBUG:
            print("\n" + "=" * 72)
            print("[DEBUG] EMAIL PREVIEW")
            print("=" * 72)
            print(f"Subject: {subject}")
            print(body)
            print("=" * 72)

        # ── Stage 10: Email Delivery ──────────────────────────────────
        success = send_email(subject, body)
        if success:
            logger.info("[MAIN] Stage 10 complete: email delivered successfully")
        else:
            logger.error("[MAIN] Email delivery failed.")

        elapsed = (datetime.now() - run_start).total_seconds()
        logger.info(
            f"[MAIN] Pipeline finished in {elapsed:.1f}s | "
            f"success={success}"
        )
        logger.info("=" * 60)

        return success

    except Exception as e:
        full_trace = traceback.format_exc()
        logger.error(f"[MAIN] Unhandled pipeline error: {e}\n{full_trace}")

        try:
            send_error_alert(
                error_description=str(e),
                stack_trace=full_trace,
            )
        except Exception as alert_err:
            logger.error(f"[MAIN] Could not send error alert: {alert_err}")

        return False


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ok = run_pipeline()
    sys.exit(0 if ok else 1)
