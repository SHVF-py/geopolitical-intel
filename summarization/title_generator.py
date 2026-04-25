"""
summarization/title_generator.py

Summarization Step 10.6: Title Generation via Qwen (PRD §10.6).

Input:
    - Trusted source titles (filtered by SOURCE_WEIGHTS priority)
    - Bullet facts (the selected extractive sentences)

Trusted Sources Priority (for title candidates):
    Reuters, BBC, The Economist, Al Jazeera, POLITICO, Asia Times,
    Geopolitical Monitor

Constraints:
    ≤ 12 words
    No speculation
    Event-focused

Failure fallback:
    Use the highest-weighted source title directly (truncated to 12 words).

Input:  List[EmbeddedArticle]  (representative articles in cluster)
        List[str]              (selected fact sentences)
Output: str                    (generated event title, ≤ 12 words)
"""

import json
import urllib.request
from typing import List, Dict, Optional

from config import config
from utils.logger import get_logger
from utils.helpers import extract_domain

logger = get_logger("summarization.title_generator")

_TITLE_PROMPT_TEMPLATE = """\
You are a news headline editor. Generate a concise, factual event title.

STRICT RULES:
- Maximum {max_words} words
- No speculation or opinion
- Event-focused (what happened, not analysis)
- Do not use quotes
- Write as a news headline

Context — Trusted Source Headlines:
{trusted_titles}

Key Facts:
{bullets}

Write only the title (no explanation, no punctuation at the end):"""


def _get_trusted_titles(articles: List[Dict]) -> List[str]:
    """
    Extract titles from trusted sources, sorted by SOURCE_WEIGHTS descending.
    Returns up to 5 trusted titles.
    """
    weighted = []
    for article in articles:
        domain = extract_domain(article.get("url", ""))
        weight = config.SOURCE_WEIGHTS.get(domain, 0.0)
        if domain in config.TRUSTED_TITLE_SOURCES or weight > 0:
            weighted.append((weight, article.get("title", "")))

    # Sort by weight descending
    weighted.sort(key=lambda x: x[0], reverse=True)
    return [title for _, title in weighted[:5] if title]


def _call_ollama_title(prompt: str) -> Optional[str]:
    """Call Ollama for title generation. Returns raw text or None."""
    url     = f"{config.LLM_ENDPOINT}/api/generate"
    payload = {
        "model":  config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 50,      # titles are short
            "temperature": 0.1,
            "top_p":       0.9,
            "stop":        ["\n"],  # stop at newline — title is one line
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip()
    except Exception as e:
        logger.error(f"[TITLE_GEN] Ollama call failed: {e}")
        return None


def _fallback_title(articles: List[Dict]) -> str:
    """
    Use the highest-weighted source title as fallback.
    Truncated to TITLE_MAX_WORDS.
    """
    trusted = _get_trusted_titles(articles)
    if trusted:
        title = trusted[0]
    elif articles:
        title = articles[0].get("title", "Geopolitical Event")
    else:
        title = "Geopolitical Event"

    words = title.split()
    if len(words) > config.TITLE_MAX_WORDS:
        title = " ".join(words[:config.TITLE_MAX_WORDS])
    return title


def _clean_title(raw: str) -> str:
    """Strip quotes, trailing punctuation, newlines from generated title."""
    title = raw.strip().strip('"\'').rstrip(".,:;!?").strip()
    # Enforce word limit
    words = title.split()
    if len(words) > config.TITLE_MAX_WORDS:
        title = " ".join(words[:config.TITLE_MAX_WORDS])
    return title


def generate_title(
    representative_articles: List[Dict],
    fact_sentences: List[str],
) -> str:
    """
    Generate an event title using Qwen with trusted source titles + bullet facts.

    Args:
        representative_articles: MMR-selected articles (for trusted source titles).
        fact_sentences:          Selected extractive facts.

    Returns:
        Event title string (≤ 12 words).
    """
    trusted_titles = _get_trusted_titles(representative_articles)

    if not trusted_titles and not fact_sentences:
        return _fallback_title(representative_articles)

    trusted_str = "\n".join(f"- {t}" for t in trusted_titles) if trusted_titles else "(none available)"
    bullets_str = "\n".join(f"- {s}" for s in fact_sentences[:5])  # first 5 facts

    prompt = _TITLE_PROMPT_TEMPLATE.format(
        max_words=config.TITLE_MAX_WORDS,
        trusted_titles=trusted_str,
        bullets=bullets_str,
    )

    raw_title = _call_ollama_title(prompt)

    if raw_title:
        title = _clean_title(raw_title)
        if title and len(title.split()) >= 2:
            logger.info(f"[TITLE_GEN] Generated: '{title}'")
            if config.DEBUG:
                print(f"  [DEBUG TITLE] '{title}'")
            return title

    # Fallback
    fallback = _fallback_title(representative_articles)
    logger.warning(f"[TITLE_GEN] Using fallback title: '{fallback}'")
    return fallback
