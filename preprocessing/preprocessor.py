"""
preprocessing/preprocessor.py

Stage 2: Preprocessing Pipeline.

Responsibilities (per PRD §4):
  4.1  HTML cleaning — remove <script>, <style> tags, extract visible text
  4.2  Boilerplate removal — trafilatura (primary), newspaper3k (fallback)
  4.3  Text normalization — lowercase, strip whitespace, normalize unicode,
                            remove excessive newlines
  4.4  Sentence segmentation — spaCy (primary), nltk.sent_tokenize (fallback)
  4.5  Length filtering — discard articles with word_count < MIN_WORD_COUNT

Input:  List[RawArticle]
Output: List[CleanArticle]

CleanArticle = {
    "url":           str,
    "title":         str,
    "published":     datetime,
    "source_domain": str,
    "clean_text":    str,         # normalized, boilerplate-free full text
    "sentences":     List[str],   # sentence-tokenized version of clean_text
    "word_count":    int
}
"""

import re
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

from config import config
from utils.logger import get_logger
from utils.helpers import normalize_text, word_count, timer

logger = get_logger("preprocessing.preprocessor")

# ── Lazy-loaded NLP tools ────────────────────────────────────────────────────
_spacy_nlp = None
_nltk_ready = False


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser", "lemmatizer"])
            _spacy_nlp.enable_pipe("senter")   # sentence boundary detection only
            logger.debug("[PREPROCESS] spaCy loaded (en_core_web_sm)")
        except Exception as e:
            logger.warning(f"[PREPROCESS] spaCy unavailable: {e}. Will use nltk fallback.")
            _spacy_nlp = "unavailable"
    return _spacy_nlp if _spacy_nlp != "unavailable" else None


def _ensure_nltk():
    global _nltk_ready
    if not _nltk_ready:
        try:
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            _nltk_ready = True
        except Exception as e:
            logger.warning(f"[PREPROCESS] nltk setup failed: {e}")


# ─────────────────────────────────────────────
# 4.1 HTML Cleaning
# ─────────────────────────────────────────────

def _clean_html(raw_html: str) -> str:
    """
    Remove <script>, <style> tags; extract visible text.
    Uses BeautifulSoup.
    """
    if not raw_html:
        return ""
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=" ")
    except Exception as e:
        logger.warning(f"[PREPROCESS] HTML cleaning failed: {e}")
        # Naive fallback: strip all angle-bracket tags
        return re.sub(r"<[^>]+>", " ", raw_html)


# ─────────────────────────────────────────────
# 4.2 Boilerplate Removal
# ─────────────────────────────────────────────

def _remove_boilerplate_trafilatura(url: str, raw_html: str) -> Optional[str]:
    """Primary: trafilatura for boilerplate removal."""
    try:
        import trafilatura
        result = trafilatura.extract(
            raw_html,
            url=url,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return result
    except Exception as e:
        logger.debug(f"[PREPROCESS] trafilatura failed for {url}: {e}")
        return None


def _remove_boilerplate_newspaper(url: str, raw_html: str) -> Optional[str]:
    """Fallback: newspaper3k for boilerplate removal."""
    try:
        from newspaper import Article
        article = Article(url)
        article.set_html(raw_html)
        article.parse()
        return article.text
    except Exception as e:
        logger.debug(f"[PREPROCESS] newspaper3k failed for {url}: {e}")
        return None


def _extract_clean_body(url: str, raw_html: str, title: str) -> str:
    """
    Attempt boilerplate removal in priority order:
    1. trafilatura
    2. newspaper3k
    3. HTML-stripped text
    4. title as last resort (will be filtered out by length check)
    """
    # Try trafilatura first
    text = _remove_boilerplate_trafilatura(url, raw_html)
    if text and word_count(text) >= 30:
        return text

    # Try newspaper3k
    text = _remove_boilerplate_newspaper(url, raw_html)
    if text and word_count(text) >= 30:
        return text

    # Fall back to raw HTML stripping
    text = _clean_html(raw_html)
    if text and word_count(text) >= 30:
        return text

    # Last resort: return title (will be discarded by word_count filter)
    return title


# ─────────────────────────────────────────────
# 4.3 Text Normalization
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Apply normalization per PRD §4.3."""
    return normalize_text(text)


# ─────────────────────────────────────────────
# 4.4 Sentence Segmentation
# ─────────────────────────────────────────────

def _segment_sentences(text: str) -> List[str]:
    """
    Segment text into sentences.
    Primary: spaCy; Fallback: nltk.sent_tokenize.
    Returns list of non-empty sentence strings.
    """
    nlp = _get_spacy()
    if nlp is not None:
        try:
            doc = nlp(text[:1_000_000])  # spaCy has a default limit guard
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if sentences:
                return sentences
        except Exception as e:
            logger.warning(f"[PREPROCESS] spaCy segmentation failed: {e}")

    # nltk fallback
    _ensure_nltk()
    try:
        from nltk.tokenize import sent_tokenize
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        return sentences
    except Exception as e:
        logger.warning(f"[PREPROCESS] nltk sent_tokenize failed: {e}")

    # Crude fallback: split on period-space
    return [s.strip() for s in re.split(r'\.\s+', text) if s.strip()]


# ─────────────────────────────────────────────
# 4.5 Length Filtering
# ─────────────────────────────────────────────

def _passes_length_filter(text: str) -> bool:
    return word_count(text) >= config.MIN_WORD_COUNT


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def preprocess_articles(raw_articles: List[Dict]) -> List[Dict]:
    """
    Run the full preprocessing pipeline on a list of RawArticle dicts.

    Returns a list of CleanArticle dicts.
    Discards articles that do not pass the length filter.
    """
    clean_articles = []
    discarded_short = 0

    with timer("Preprocessing"):
        for raw in raw_articles:
            url    = raw.get("url", "")
            title  = raw.get("title", "")
            raw_html = raw.get("raw_html", "")

            # 4.2 Boilerplate removal (includes implicit 4.1 HTML cleaning)
            body = _extract_clean_body(url, raw_html, title)

            # 4.3 Normalize
            clean = _normalize(body)

            # 4.5 Length filter
            wc = word_count(clean)
            if wc < config.MIN_WORD_COUNT:
                discarded_short += 1
                logger.debug(
                    f"[PREPROCESS] Discarded (too short, {wc} words): {url[:80]}"
                )
                continue

            # 4.4 Sentence segmentation
            sentences = _segment_sentences(clean)

            article: Dict = {
                "url":           url,
                "title":         title,
                "published":     raw.get("published"),
                "source_domain": raw.get("source_domain", ""),
                "clean_text":    clean,
                "sentences":     sentences,
                "word_count":    wc,
            }
            clean_articles.append(article)

            if config.DEBUG:
                print(
                    f"  [DEBUG PREPROCESS] {url[:60]} | "
                    f"{wc} words | {len(sentences)} sentences"
                )

    logger.info(
        f"[PREPROCESS] Input: {len(raw_articles)} | "
        f"Output: {len(clean_articles)} | "
        f"Discarded (short): {discarded_short}"
    )
    return clean_articles
