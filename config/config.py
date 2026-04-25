"""
config.py — Centralized configuration for the Geopolitical News Intelligence System.
All tunable parameters live here. DO NOT hardcode values in pipeline modules.
"""

import os

# Base directory: absolute path to the project root (where config.py lives is config/)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# DEBUG
# ─────────────────────────────────────────────
DEBUG = False  # Set True to print intermediate outputs, clusters, selected sentences

# ─────────────────────────────────────────────
# EMAIL
# ─────────────────────────────────────────────
# Credentials are read from environment variables when set (for GitHub Actions security).
# Fallback to hardcoded values for local use.
EMAIL_SENDER   = os.environ.get("EMAIL_SENDER",   "batmanintj@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "lmgp levd payc vvpn")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "sshafs2004@gmail.com")
SMTP_SERVER    = "smtp.gmail.com"
SMTP_PORT      = 587

# ─────────────────────────────────────────────
# RSS FEED SOURCES
# ─────────────────────────────────────────────
RSS_FEEDS = [
    # Global / General News (High Priority)
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.reuters.com/rssFeed/worldNews",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://feeds.washingtonpost.com/rss/world",
    "https://www.theguardian.com/world/rss",
    # Geopolitics-Focused / Analysis
    "https://www.politico.com/rss/politics08.xml",
    "https://www.politico.com/rss/world-news.xml",
    "https://asiatimes.com/feed/",
    "https://www.geopoliticalmonitor.com/feed/",
    # Economy / Policy
    "https://www.economist.com/the-world-this-week/rss.xml",
    "https://www.ft.com/world?format=rss",
    "https://www.ft.com/global-economy?format=rss",
    # Defense / Security
    "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml",
    "https://www.stripes.com/arc/outboundfeeds/rss/?outputType=xml",
    # Regional Coverage
    "https://www.scmp.com/rss/91/feed",
    "https://www.japantimes.co.jp/feed/",
    "https://www.hindustantimes.com/feeds/rss/world-news/rssfeed.xml",
    # Middle East
    "https://www.middleeasteye.net/rss",
    # Wire Services
    "https://apnews.com/rss",
]

# ─────────────────────────────────────────────
# SOURCE WEIGHTS (for title generation priority)
# ─────────────────────────────────────────────
SOURCE_WEIGHTS = {
    "reuters.com":            1.0,
    "apnews.com":             1.0,
    "bbc.co.uk":              0.95,
    "nytimes.com":            0.9,
    "economist.com":          0.9,
    "aljazeera.com":          0.85,
    "politico.com":           0.85,
    "asiatimes.com":          0.7,
    "geopoliticalmonitor.com":0.7,
}

# Trusted sources for title generation (PRD §10.6)
TRUSTED_TITLE_SOURCES = [
    "reuters.com",
    "bbc.co.uk",
    "economist.com",
    "aljazeera.com",
    "politico.com",
    "asiatimes.com",
    "geopoliticalmonitor.com",
]

# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────
EMBEDDING_MODEL        = "all-MiniLM-L6-v2"
CLASSIFICATION_MODEL   = "typeform/distilbert-base-uncased-mnli"  # NLI-fine-tuned distilbert; supports zero-shot-classification
LLM_PROVIDER           = "ollama"
LLM_MODEL              = "qwen:4b"
LLM_ENDPOINT           = "http://localhost:11434"

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
MIN_WORD_COUNT = 100  # Discard articles with fewer words

# ─────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────
DEDUP_SIMILARITY_THRESHOLD = 0.85  # Cosine similarity for semantic dedup

# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────
CLASSIFICATION_THRESHOLD = 0.5  # Min confidence to keep article as geopolitical

# ─────────────────────────────────────────────
# CHUNKING + EMBEDDING
# ─────────────────────────────────────────────
CHUNK_SIZE    = 250  # words per chunk
CHUNK_OVERLAP = 50   # overlapping words between consecutive chunks

# ─────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────
MIN_CLUSTER_SIZE = 3  # HDBSCAN primary min_cluster_size

# ─────────────────────────────────────────────
# RANKING
# ─────────────────────────────────────────────
TOP_N_EVENTS           = 6
RANK_WEIGHT_SIZE       = 0.5
RANK_WEIGHT_RECENCY    = 0.3
RANK_WEIGHT_DIVERSITY  = 0.2

# ─────────────────────────────────────────────
# MMR (Representative Article Selection)
# ─────────────────────────────────────────────
MMR_K      = 5    # max articles to select per cluster (3–5)
MMR_LAMBDA = 0.7  # trade-off: relevance vs diversity

# ─────────────────────────────────────────────
# SUMMARIZATION
# ─────────────────────────────────────────────
TEXTRANK_TOP_SENTENCES     = 5    # sentences extracted per article by TextRank
SENTENCE_SIM_THRESHOLD     = 0.85 # cosine threshold for sentence dedup
MIN_SUMMARY_SENTENCES      = 7
MAX_SUMMARY_SENTENCES      = 12
MAX_ARTICLES_PER_CLUSTER   = 5
QWEN_MAX_TOKENS            = 512
QWEN_PARAGRAPH_MAX_WORDS   = 120
TITLE_MAX_WORDS            = 12

# ─────────────────────────────────────────────
# STORAGE (Embedding + metadata cache)
# ─────────────────────────────────────────────
STORAGE_DIR      = os.path.join(_BASE_DIR, "storage", "embeddings")
STORAGE_TTL_DAYS = 3   # delete data older than 3 days

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_DIR = os.path.join(_BASE_DIR, "logs")

# ─────────────────────────────────────────────
# SCHEDULING
# ─────────────────────────────────────────────
# Run at 7 AM and 7 PM local time (configured in GitHub Actions)
SCHEDULE_CRON = "0 7,19 * * *"

# ─────────────────────────────────────────────
# RSS FETCH RETRY
# ─────────────────────────────────────────────
RSS_RETRY_COUNT    = 3
RSS_RETRY_DELAY    = 5  # seconds between retries
RSS_CACHE_DIR      = os.path.join(_BASE_DIR, "storage", "rss_cache")
RSS_FETCH_TIMEOUT  = 15  # seconds
