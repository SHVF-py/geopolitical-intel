"""
storage/embedding_store.py

Local persistent embedding + metadata cache with 3-day TTL.
Stores data organized by date under STORAGE_DIR.
Enables cross-day deduplication.

Storage layout:
    storage/embeddings/
        YYYY-MM-DD/
            urls.json          → set of seen URLs
            hashes.json        → set of seen SHA-256 hashes
            embeddings.npy     → (N, D) array of semantic dedup embeddings
            rss_cache/
                <feed_hash>.json  → cached raw feed entries
"""

import os
import json
import shutil
import numpy as np
from datetime import datetime, timedelta
from typing import Set, List, Optional
from utils.logger import get_logger
from config import config

logger = get_logger("storage.embedding_store")


def _date_str(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d")


def _day_dir(date_str: str) -> str:
    return os.path.join(config.STORAGE_DIR, date_str)


def _ensure_day_dir(date_str: str) -> str:
    d = _day_dir(date_str)
    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────
# TTL CLEANUP
# ─────────────────────────────────────────────

def purge_old_data():
    """Delete storage directories older than STORAGE_TTL_DAYS."""
    if not os.path.isdir(config.STORAGE_DIR):
        return
    cutoff = datetime.now() - timedelta(days=config.STORAGE_TTL_DAYS)
    for entry in os.listdir(config.STORAGE_DIR):
        entry_path = os.path.join(config.STORAGE_DIR, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            entry_dt = datetime.strptime(entry, "%Y-%m-%d")
            if entry_dt < cutoff:
                shutil.rmtree(entry_path)
                logger.info(f"[STORAGE] Purged old data directory: {entry}")
        except ValueError:
            pass  # skip non-date directories


# ─────────────────────────────────────────────
# SEEN URLs (cross-day dedup)
# ─────────────────────────────────────────────

def _collect_seen_urls_across_days() -> Set[str]:
    seen = set()
    if not os.path.isdir(config.STORAGE_DIR):
        return seen
    for entry in os.listdir(config.STORAGE_DIR):
        urls_path = os.path.join(config.STORAGE_DIR, entry, "urls.json")
        if os.path.isfile(urls_path):
            try:
                with open(urls_path, "r", encoding="utf-8") as f:
                    seen.update(json.load(f))
            except Exception as e:
                logger.warning(f"[STORAGE] Could not load urls from {urls_path}: {e}")
    return seen


def load_seen_urls() -> Set[str]:
    return _collect_seen_urls_across_days()


def save_seen_urls(urls: Set[str], date_str: Optional[str] = None):
    if date_str is None:
        date_str = _date_str()
    d = _ensure_day_dir(date_str)
    path = os.path.join(d, "urls.json")
    # Merge with existing
    existing = set()
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = set(json.load(f))
        except Exception:
            pass
    merged = existing | urls
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(merged), f)


# ─────────────────────────────────────────────
# SEEN HASHES (cross-day dedup)
# ─────────────────────────────────────────────

def _collect_seen_hashes_across_days() -> Set[str]:
    seen = set()
    if not os.path.isdir(config.STORAGE_DIR):
        return seen
    for entry in os.listdir(config.STORAGE_DIR):
        path = os.path.join(config.STORAGE_DIR, entry, "hashes.json")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    seen.update(json.load(f))
            except Exception as e:
                logger.warning(f"[STORAGE] Could not load hashes from {path}: {e}")
    return seen


def load_seen_hashes() -> Set[str]:
    return _collect_seen_hashes_across_days()


def save_seen_hashes(hashes: Set[str], date_str: Optional[str] = None):
    if date_str is None:
        date_str = _date_str()
    d = _ensure_day_dir(date_str)
    path = os.path.join(d, "hashes.json")
    existing = set()
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = set(json.load(f))
        except Exception:
            pass
    merged = existing | hashes
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(merged), f)


# ─────────────────────────────────────────────
# SEMANTIC DEDUP EMBEDDINGS (cross-day)
# ─────────────────────────────────────────────

def load_seen_embeddings() -> List[np.ndarray]:
    """Load all stored embeddings across the TTL window."""
    all_embeddings = []
    if not os.path.isdir(config.STORAGE_DIR):
        return all_embeddings
    for entry in os.listdir(config.STORAGE_DIR):
        path = os.path.join(config.STORAGE_DIR, entry, "embeddings.npy")
        if os.path.isfile(path):
            try:
                arr = np.load(path)
                if arr.ndim == 2:
                    all_embeddings.extend(arr)
                logger.debug(f"[STORAGE] Loaded {len(arr)} embeddings from {entry}")
            except Exception as e:
                logger.warning(f"[STORAGE] Could not load embeddings from {path}: {e}")
    return all_embeddings


def save_new_embeddings(embeddings: List[np.ndarray], date_str: Optional[str] = None):
    """Append new embeddings to today's storage file."""
    if not embeddings:
        return
    if date_str is None:
        date_str = _date_str()
    d = _ensure_day_dir(date_str)
    path = os.path.join(d, "embeddings.npy")
    new_arr = np.array(embeddings)
    if os.path.isfile(path):
        try:
            existing = np.load(path)
            new_arr = np.vstack([existing, new_arr])
        except Exception:
            pass
    np.save(path, new_arr)
    logger.debug(f"[STORAGE] Saved {len(embeddings)} new embeddings to {date_str}")


# ─────────────────────────────────────────────
# RSS CACHE
# ─────────────────────────────────────────────

def save_rss_cache(feed_url: str, entries: list, date_str: Optional[str] = None):
    """Cache raw RSS entries keyed by feed URL hash."""
    import hashlib
    if date_str is None:
        date_str = _date_str()
    cache_dir = os.path.join(_ensure_day_dir(date_str), "rss_cache")
    os.makedirs(cache_dir, exist_ok=True)
    feed_key = hashlib.md5(feed_url.encode()).hexdigest()
    path = os.path.join(cache_dir, f"{feed_key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"url": feed_url, "entries": entries}, f, default=str)


def load_rss_cache(feed_url: str, date_str: Optional[str] = None) -> Optional[list]:
    """Load cached RSS entries for a feed URL (same day only)."""
    import hashlib
    if date_str is None:
        date_str = _date_str()
    feed_key = hashlib.md5(feed_url.encode()).hexdigest()
    path = os.path.join(_day_dir(date_str), "rss_cache", f"{feed_key}.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("entries", [])
        except Exception as e:
            logger.warning(f"[STORAGE] Could not load RSS cache for {feed_url}: {e}")
    return None
