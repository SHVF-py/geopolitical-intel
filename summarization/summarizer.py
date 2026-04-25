"""
summarization/summarizer.py

Summarization Pipeline Orchestrator (PRD §10).

Runs the complete hybrid summarization chain per cluster:
    10.1  MMR article selection
    10.2  TextRank per-article extraction
    10.3  Merge + sentence deduplication
    10.4  Fact selection (token budget)
    10.5  Qwen abstractive paragraph
    10.6  Qwen title generation

Input:  List[RankedCluster]
Output: List[SummarizedEvent]

SummarizedEvent = {
    "title":        str,
    "paragraph":    str or None,
    "bullet_facts": List[str],
    "sources":      List[str]    (≥2 URLs; from all cluster articles)
}
"""

from typing import List, Dict, Optional

from config import config
from utils.logger import get_logger
from utils.helpers import timer

from summarization.mmr           import select_representative_articles
from summarization.textrank      import extract_top_sentences
from summarization.sentence_dedup import merge_and_dedup_sentences
from summarization.fact_selector  import select_facts
from summarization.abstractive    import generate_abstractive_summary
from summarization.title_generator import generate_title

logger = get_logger("summarization.summarizer")


def _collect_sources(cluster: Dict) -> List[str]:
    """
    Collect unique source URLs from all articles in the cluster.
    Returns at least all URLs (caller should enforce ≥2 rule before emailing).
    """
    seen  = set()
    urls  = []
    for article in cluster["articles"]:
        url = article.get("url", "")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def summarize_cluster(cluster: Dict) -> Optional[Dict]:
    """
    Run the full summarization pipeline for one cluster.

    Returns a SummarizedEvent dict, or None if the cluster cannot
    produce a valid summary (e.g. no sentences extracted).
    """
    cluster_id = cluster["cluster_id"]
    logger.info(
        f"[SUMMARIZE] Processing cluster {cluster_id} "
        f"(size={cluster['size']})"
    )

    # 10.1 — MMR article selection
    # Cap input articles to MAX_ARTICLES_PER_CLUSTER before MMR to bound
    # memory and compute (config value was defined but never enforced).
    candidate_articles = cluster["articles"][:config.MAX_ARTICLES_PER_CLUSTER]
    cluster_for_mmr = {**cluster, "articles": candidate_articles}
    representative = select_representative_articles(cluster_for_mmr)
    logger.debug(f"[SUMMARIZE] MMR selected {len(representative)} articles")

    # 10.2 — TextRank per-article extraction
    per_article_sentences: List[List[str]] = []
    for article in representative:
        extracted = extract_top_sentences(article)
        per_article_sentences.append(extracted)
        logger.debug(
            f"[SUMMARIZE] TextRank: {len(extracted)} sentences from "
            f"{article.get('url','')[:60]}"
        )

    if not any(per_article_sentences):
        logger.warning(
            f"[SUMMARIZE] No sentences extracted for cluster {cluster_id}. Skipping."
        )
        return None

    # 10.3 — Merge + deduplicate sentences
    merged = merge_and_dedup_sentences(per_article_sentences)

    if not merged:
        logger.warning(
            f"[SUMMARIZE] No sentences after dedup for cluster {cluster_id}. Skipping."
        )
        return None

    # 10.4 — Fact selection (7–12 sentences)
    facts = select_facts(
        deduplicated_sentences=merged,
        cluster_centroid=cluster["centroid"],
        original_per_article=per_article_sentences,
    )

    if not facts:
        logger.warning(
            f"[SUMMARIZE] Fact selection returned empty for cluster {cluster_id}."
        )
        return None

    # 10.5 — Abstractive paragraph (Qwen)
    rep_titles = [a.get("title", "") for a in representative]
    paragraph  = generate_abstractive_summary(facts, rep_titles)
    # paragraph may be None → email formatter uses bullet fallback

    # 10.6 — Title generation (Qwen)
    title = generate_title(representative, facts)

    # Collect sources
    sources = _collect_sources(cluster)

    event = {
        "title":        title,
        "paragraph":    paragraph,
        "bullet_facts": facts,
        "sources":      sources,
    }

    logger.info(
        f"[SUMMARIZE] Cluster {cluster_id} → "
        f"title='{title[:60]}' | "
        f"facts={len(facts)} | "
        f"sources={len(sources)} | "
        f"has_paragraph={paragraph is not None}"
    )

    if config.DEBUG:
        print(f"\n  [DEBUG EVENT] {title}")
        print(f"  Paragraph: {paragraph}")
        for f in facts:
            print(f"    - {f[:100]}")

    return event


def summarize_all(clusters: List[Dict]) -> List[Dict]:
    """
    Summarize all ranked clusters.

    Returns list of SummarizedEvent dicts.
    Clusters that fail summarization are skipped (logged as errors).
    """
    events = []

    with timer("Summarization"):
        for cluster in clusters:
            try:
                event = summarize_cluster(cluster)
                if event is not None:
                    events.append(event)
            except Exception as e:
                logger.error(
                    f"[SUMMARIZE] Unhandled error on cluster "
                    f"{cluster.get('cluster_id','?')}: {e}"
                )
                # Continue to next cluster — don't crash the pipeline

    logger.info(
        f"[SUMMARIZE] Completed: {len(events)}/{len(clusters)} events summarized"
    )

    if len(events) == 0:
        logger.error("[SUMMARIZE] No events could be summarized!")

    return events
