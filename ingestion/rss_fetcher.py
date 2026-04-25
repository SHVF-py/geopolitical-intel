"""
ingestion/rss_fetcher.py

Stage 1: RSS Ingestion.

Responsibilities:
  - Fetch all configured RSS feeds
  - Parse entries into RawArticle dicts
  - Retry on failure (3 attempts)
  - Fall back to last cached fetch on persistent failure
  - Log fetch counts per feed

Output schema:
  RawArticle = {
      "url":           str,
      "title":         str,
      "published":     datetime,
      "source_domain": str,
      "raw_html":      str   # raw body/summary from feed
  }
"""

import time
import feedparser
from datetime import datetime, timezone
from typing import List, Dict, Optional
from email.utils import parsedate_to_datetime

from config import config
from utils.logger import get_logger
from utils.helpers import extract_domain, timer
from storage.embedding_store import save_rss_cache, load_rss_cache

logger = get_logger("ingestion.rss_fetcher")


# ─────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────

RawArticle = Dict  # typed via docstring above


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _parse_date(entry) -> datetime:
    """
    Parse publication date from a feedparser entry.
    Falls back to current UTC time if unavailable.
    """
    for attr in ("published", "updated", "created"):
        raw = getattr(entry, attr, None)
        if raw:
            try:
                return parsedate_to_datetime(raw).astimezone(timezone.utc)
            except Exception:
                pass
    # feedparser also provides published_parsed (struct_time)
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            import calendar
            ts = calendar.timegm(entry.published_parsed)
            return datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(tz=timezone.utc)


def _entry_to_raw_article(entry, source_domain: str) -> Optional[RawArticle]:
    """Convert a feedparser entry to a RawArticle dict."""
    url = getattr(entry, "link", None)
    if not url:
        return None

    title = getattr(entry, "title", "") or ""

    # Prefer content over summary for richer text
    raw_html = ""
    if hasattr(entry, "content") and entry.content:
        raw_html = entry.content[0].get("value", "") or ""
    if not raw_html:
        raw_html = getattr(entry, "summary", "") or ""

    published = _parse_date(entry)

    return {
        "url":           url.strip(),
        "title":         title.strip(),
        "published":     published,
        "source_domain": source_domain,
        "raw_html":      raw_html,
    }


def _fetch_feed_with_retry(feed_url: str) -> List[RawArticle]:
    """
    Fetch a single RSS feed with up to RSS_RETRY_COUNT retries.
    On total failure, fall back to cached entries.

    Returns list of RawArticle dicts.
    """
    domain = extract_domain(feed_url)
    last_error = None

    for attempt in range(1, config.RSS_RETRY_COUNT + 1):
        try:
            logger.debug(f"[RSS] Fetching {feed_url} (attempt {attempt})")
            parsed = feedparser.parse(
                feed_url,
                agent="GeopoliticalIntelBot/1.0",
                request_headers={"Connection": "close"},
            )

            # feedparser does not raise on HTTP errors — check bozo flag
            if parsed.bozo and not parsed.entries:
                raise ValueError(f"Feed parse error: {parsed.bozo_exception}")

            articles = []
            for entry in parsed.entries:
                article = _entry_to_raw_article(entry, domain)
                if article:
                    articles.append(article)

            logger.info(f"[RSS] {domain}: fetched {len(articles)} articles")

            # Cache successful fetch (store serialisable version)
            serialisable = [
                {**a, "published": a["published"].isoformat()}
                for a in articles
            ]
            save_rss_cache(feed_url, serialisable)

            return articles

        except Exception as e:
            last_error = e
            logger.warning(
                f"[RSS] Attempt {attempt}/{config.RSS_RETRY_COUNT} failed for "
                f"{feed_url}: {e}"
            )
            if attempt < config.RSS_RETRY_COUNT:
                time.sleep(config.RSS_RETRY_DELAY)

    # ── Fallback: last cached fetch ──────────────────────────────────────
    logger.error(f"[RSS] All retries failed for {feed_url}. Falling back to cache.")
    cached = load_rss_cache(feed_url)
    if cached:
        logger.info(f"[RSS] Loaded {len(cached)} cached articles for {feed_url}")
        articles = []
        for item in cached:
            try:
                item["published"] = datetime.fromisoformat(item["published"])
            except Exception:
                item["published"] = datetime.now(tz=timezone.utc)
            articles.append(item)
        return articles

    logger.error(f"[RSS] No cache available for {feed_url}. Skipping feed.")
    return []


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def fetch_all_feeds() -> List[RawArticle]:
    """
    Fetch all RSS feeds defined in config.RSS_FEEDS.

    Returns:
        Flat list of RawArticle dicts from all feeds.
    """
    all_articles: List[RawArticle] = []

    with timer("RSS Ingestion"):
        for feed_url in config.RSS_FEEDS:
            try:
                articles = _fetch_feed_with_retry(feed_url)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"[RSS] Unexpected error for {feed_url}: {e}")

    logger.info(f"[RSS] Total raw articles fetched: {len(all_articles)}")

    if config.DEBUG:
        for a in all_articles[:3]:
            print(f"  [DEBUG RSS] {a['source_domain']} | {a['title'][:80]}")

    return all_articles
