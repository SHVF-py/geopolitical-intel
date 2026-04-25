"""
clustering/clusterer.py

Stage 6: Clustering — Event Formation (PRD §8).

Algorithm: HDBSCAN
Input:     Article embeddings (article_embedding per article)
Output:    List[Cluster]

During clustering, article_embedding is resolved as:
    max similarity(chunk_embeddings, cluster_centroid)
This is an iterative process — we first cluster with mean embeddings,
compute centroids, then refine article_embeddings to the best-matching chunk.

Cluster = {
    "cluster_id":  int,
    "articles":    List[EmbeddedArticle],
    "centroid":    np.ndarray,
    "size":        int
}

Fallback Logic (HARD RULE: always output Top N events):
    Primary:  min_cluster_size = 3
    Fallback: include size-2 clusters if clusters < TOP_N
    Final:    include high-confidence singles if still insufficient
"""

import numpy as np
from typing import List, Dict
from collections import defaultdict

from config import config
from utils.logger import get_logger
from utils.helpers import cosine_similarity, timer

logger = get_logger("clustering.clusterer")


# ─────────────────────────────────────────────
# Embedding resolution (max-sim to centroid)
# ─────────────────────────────────────────────

def _resolve_article_embedding(article: Dict, centroid: np.ndarray) -> np.ndarray:
    """
    Replace article_embedding with the chunk embedding that has
    maximum cosine similarity to the cluster centroid.
    This is the PRD's max similarity strategy.
    """
    chunk_embeddings = article.get("chunk_embeddings", [])
    if not chunk_embeddings:
        return article["article_embedding"]

    best_emb  = chunk_embeddings[0]
    best_sim  = cosine_similarity(chunk_embeddings[0], centroid)

    for emb in chunk_embeddings[1:]:
        sim = cosine_similarity(emb, centroid)
        if sim > best_sim:
            best_sim = sim
            best_emb = emb

    return best_emb


def _compute_centroid(articles: List[Dict]) -> np.ndarray:
    embeddings = np.array([a["article_embedding"] for a in articles])
    return np.mean(embeddings, axis=0)


# ─────────────────────────────────────────────
# HDBSCAN Clustering
# ─────────────────────────────────────────────

def _run_hdbscan(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """
    Run HDBSCAN and return cluster label array.
    Label -1 = noise (unclustered).
    """
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="cosine",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings)
    return labels


def _build_clusters(articles: List[Dict], labels: np.ndarray) -> List[Dict]:
    """
    Group articles by cluster label.
    Noise articles (label == -1) are excluded from returned clusters
    but stored separately for fallback use.

    Returns list of Cluster dicts (excludes noise cluster).
    """
    cluster_map: Dict[int, List[Dict]] = defaultdict(list)

    for article, label in zip(articles, labels):
        cluster_map[int(label)].append(article)

    clusters = []
    for label, members in cluster_map.items():
        if label == -1:
            continue  # noise — handled separately

        # Initial centroid with mean embeddings
        centroid = _compute_centroid(members)

        # Refine each article's embedding to max-sim chunk
        for article in members:
            article["article_embedding"] = _resolve_article_embedding(article, centroid)

        # Recompute centroid after refinement
        centroid = _compute_centroid(members)

        cluster = {
            "cluster_id": label,
            "articles":   members,
            "centroid":   centroid,
            "size":       len(members),
        }
        clusters.append(cluster)

    return clusters


# ─────────────────────────────────────────────
# Fallback: size-2 clusters
# ─────────────────────────────────────────────

def _cluster_size_two(noise_articles: List[Dict]) -> List[Dict]:
    """
    Greedily form clusters of size 2 from noise articles
    by pairing the two most similar articles repeatedly.
    Returns list of Cluster dicts.
    """
    if len(noise_articles) < 2:
        return []

    from itertools import combinations

    used = set()
    pairs = []

    # Build similarity scores for all pairs
    pair_sims = []
    for i, j in combinations(range(len(noise_articles)), 2):
        sim = cosine_similarity(
            noise_articles[i]["article_embedding"],
            noise_articles[j]["article_embedding"]
        )
        pair_sims.append((sim, i, j))

    pair_sims.sort(reverse=True)

    for sim, i, j in pair_sims:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        members = [noise_articles[i], noise_articles[j]]
        centroid = _compute_centroid(members)
        pairs.append({
            "cluster_id": f"pair_{i}_{j}",
            "articles":   members,
            "centroid":   centroid,
            "size":       2,
        })

    return pairs


# ─────────────────────────────────────────────
# Fallback: high-confidence single articles
# ─────────────────────────────────────────────

def _single_article_clusters(articles: List[Dict], used_urls: set) -> List[Dict]:
    """
    Wrap remaining individual articles as single-article clusters.
    Sorted by geo_confidence descending to prefer higher-confidence articles.
    """
    remaining = [a for a in articles if a["url"] not in used_urls]
    remaining.sort(key=lambda a: a.get("geo_confidence", 0.5), reverse=True)

    clusters = []
    for i, article in enumerate(remaining):
        clusters.append({
            "cluster_id": f"single_{i}",
            "articles":   [article],
            "centroid":   article["article_embedding"],
            "size":       1,
        })
    return clusters


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def cluster_articles(articles: List[Dict]) -> List[Dict]:
    """
    Cluster embedded articles into event groups using HDBSCAN.

    Implements the full fallback chain to ensure >= TOP_N clusters are returned.

    Returns list of Cluster dicts.
    """
    if not articles:
        logger.warning("[CLUSTER] No articles to cluster.")
        return []

    with timer("Clustering"):
        embeddings = np.array([a["article_embedding"] for a in articles])

        # ── Primary clustering ────────────────────────────────────────
        try:
            labels = _run_hdbscan(embeddings, min_cluster_size=config.MIN_CLUSTER_SIZE)
        except Exception as e:
            logger.error(f"[CLUSTER] HDBSCAN failed: {e}")
            # Clustering failure fallback: treat all as noise → singles
            labels = np.full(len(articles), -1, dtype=int)

        clusters = _build_clusters(articles, labels)
        noise_articles = [a for a, l in zip(articles, labels) if l == -1]

        n_clusters = len(clusters)
        logger.info(
            f"[CLUSTER] HDBSCAN: {n_clusters} clusters | "
            f"{len(noise_articles)} noise articles"
        )

        if config.DEBUG:
            for c in clusters:
                print(
                    f"  [DEBUG CLUSTER] id={c['cluster_id']} size={c['size']} | "
                    f"titles={[a['title'][:40] for a in c['articles'][:2]]}"
                )

        # ── Fallback 1: size-2 clusters from noise ─────────────────
        if n_clusters < config.TOP_N_EVENTS and noise_articles:
            logger.info(
                f"[CLUSTER] Fewer than {config.TOP_N_EVENTS} clusters. "
                f"Attempting size-2 fallback from {len(noise_articles)} noise articles."
            )
            pair_clusters = _cluster_size_two(noise_articles)
            clusters.extend(pair_clusters)
            logger.info(
                f"[CLUSTER] After size-2 fallback: {len(clusters)} clusters"
            )

            # Update noise after pairing
            paired_urls = {
                a["url"]
                for c in pair_clusters
                for a in c["articles"]
            }
            noise_articles = [a for a in noise_articles if a["url"] not in paired_urls]

        # ── Fallback 2: high-confidence singles ─────────────────────
        if len(clusters) < config.TOP_N_EVENTS:
            logger.info(
                f"[CLUSTER] Still < {config.TOP_N_EVENTS} clusters. "
                f"Adding high-confidence single articles."
            )
            used_urls = {a["url"] for c in clusters for a in c["articles"]}
            single_clusters = _single_article_clusters(articles, used_urls)
            clusters.extend(single_clusters)
            logger.info(
                f"[CLUSTER] After singles fallback: {len(clusters)} clusters"
            )

    logger.info(f"[CLUSTER] Total clusters formed: {len(clusters)}")
    return clusters
