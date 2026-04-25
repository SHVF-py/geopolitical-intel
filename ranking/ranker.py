"""
ranking/ranker.py

Stage 7: Event Ranking — Top N Selection (PRD §9).

Formula:
    score = 0.5 * cluster_size
          + 0.3 * recency_score
          + 0.2 * source_diversity

Components:
    cluster_size:    number of articles in the cluster
    recency_score:   1 / (hours_since_most_recent_publish + 1)
    source_diversity: number of unique source domains in cluster

Output: Top 6 RankedCluster dicts

RankedCluster = Cluster + {
    "score":            float,
    "recency_score":    float,
    "source_diversity": int
}
"""

from datetime import datetime, timezone
from typing import List, Dict

from config import config
from utils.logger import get_logger
from utils.helpers import timer

logger = get_logger("ranking.ranker")


def _recency_score(articles: List[Dict]) -> float:
    """
    Compute recency score using the most recently published article in cluster.
    recency = 1 / (hours_since_publish + 1)
    """
    now = datetime.now(tz=timezone.utc)
    most_recent = None

    for article in articles:
        pub = article.get("published")
        if pub is None:
            continue
        # Ensure timezone-aware
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        if most_recent is None or pub > most_recent:
            most_recent = pub

    if most_recent is None:
        return 0.0

    hours_since = (now - most_recent).total_seconds() / 3600.0
    return 1.0 / (hours_since + 1.0)


def _source_diversity(articles: List[Dict]) -> int:
    """Count unique source domains in the cluster."""
    return len({a.get("source_domain", "") for a in articles if a.get("source_domain")})


def _score_cluster(cluster: Dict) -> Dict:
    """
    Compute ranking score for a single cluster.
    Adds score, recency_score, source_diversity fields.
    Returns updated cluster dict.
    """
    articles = cluster["articles"]
    size     = cluster["size"]

    recency   = _recency_score(articles)
    diversity = _source_diversity(articles)

    # Normalize size to [0, 1] range relative to a reasonable max (e.g. 20)
    # The PRD formula weights raw cluster_size directly, so we use it as-is
    # but cap to avoid one giant cluster dominating unfairly.
    # We keep this faithful to the PRD formula (no normalization imposed).
    score = (
        config.RANK_WEIGHT_SIZE      * size
        + config.RANK_WEIGHT_RECENCY * recency
        + config.RANK_WEIGHT_DIVERSITY * diversity
    )

    cluster["score"]            = score
    cluster["recency_score"]    = recency
    cluster["source_diversity"] = diversity
    return cluster


def rank_and_select(clusters: List[Dict]) -> List[Dict]:
    """
    Score all clusters and return the top TOP_N_EVENTS by score.

    Returns sorted list of top RankedCluster dicts (descending score).
    HARD RULE: always returns exactly TOP_N_EVENTS if enough clusters exist.
    """
    with timer("Ranking"):
        if not clusters:
            logger.warning("[RANK] No clusters to rank.")
            return []

        scored = [_score_cluster(c) for c in clusters]
        scored.sort(key=lambda c: c["score"], reverse=True)

        top_n = scored[:config.TOP_N_EVENTS]

        for i, c in enumerate(top_n, 1):
            logger.info(
                f"[RANK] #{i} cluster_id={c['cluster_id']} | "
                f"size={c['size']} | "
                f"recency={c['recency_score']:.4f} | "
                f"diversity={c['source_diversity']} | "
                f"score={c['score']:.4f}"
            )

            if config.DEBUG:
                print(
                    f"  [DEBUG RANK] #{i} score={c['score']:.4f} "
                    f"size={c['size']} "
                    f"titles={[a['title'][:40] for a in c['articles'][:2]]}"
                )

    logger.info(
        f"[RANK] Total clusters: {len(clusters)} | Selected: {len(top_n)}"
    )
    return top_n
