"""
summarization/mmr.py

Summarization Step 10.1: Representative Article Selection via MMR
(Maximal Marginal Relevance) — PRD §10.1.

Goal:
    Select k articles from a cluster that are both:
    - Relevant (close to cluster centroid)
    - Diverse (dissimilar to already-selected articles)

Parameters:
    k      = MMR_K      (3–5, from config)
    lambda = MMR_LAMBDA (0.7, relevance weight)

MMR formula:
    score(d) = lambda * sim(d, centroid)
             - (1 - lambda) * max_{s in selected} sim(d, s)

Input:  Cluster dict (has centroid + articles with article_embedding)
Output: List of up to k EmbeddedArticle dicts (most informative + diverse)
"""

import numpy as np
from typing import List, Dict

from config import config
from utils.logger import get_logger
from utils.helpers import cosine_similarity

logger = get_logger("summarization.mmr")


def select_representative_articles(cluster: Dict) -> List[Dict]:
    """
    Run MMR on the cluster's articles to select up to MMR_K representative ones.

    Args:
        cluster: A RankedCluster dict with 'articles' and 'centroid'.

    Returns:
        List of selected EmbeddedArticle dicts (len <= MMR_K).
    """
    articles = cluster["articles"]
    centroid = cluster["centroid"]
    k        = min(config.MMR_K, len(articles))
    lam      = config.MMR_LAMBDA

    if len(articles) <= k:
        logger.debug(
            f"[MMR] Cluster {cluster['cluster_id']}: "
            f"{len(articles)} articles <= k={k}, returning all."
        )
        return articles

    # Pre-compute similarity of each article to the centroid
    centroid_sims = [
        cosine_similarity(a["article_embedding"], centroid)
        for a in articles
    ]

    selected_indices = []
    candidate_indices = list(range(len(articles)))

    for _ in range(k):
        best_idx   = None
        best_score = float("-inf")

        for idx in candidate_indices:
            relevance = lam * centroid_sims[idx]

            if selected_indices:
                # Penalise similarity to already-selected articles
                max_sim_to_selected = max(
                    cosine_similarity(
                        articles[idx]["article_embedding"],
                        articles[s]["article_embedding"]
                    )
                    for s in selected_indices
                )
                diversity_penalty = (1 - lam) * max_sim_to_selected
            else:
                diversity_penalty = 0.0

            score = relevance - diversity_penalty

            if score > best_score:
                best_score = score
                best_idx   = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    selected = [articles[i] for i in selected_indices]

    logger.debug(
        f"[MMR] Cluster {cluster['cluster_id']}: "
        f"selected {len(selected)}/{len(articles)} articles via MMR"
    )
    return selected
