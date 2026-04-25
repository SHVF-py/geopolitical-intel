"""
summarization/abstractive.py

Summarization Step 10.5: Abstractive Summary via Qwen 3.5 4B (PRD §10.5).

Qwen is locally hosted via Ollama.
LLM_ENDPOINT = http://localhost:11434

CRITICAL CONSTRAINT:
    Input to Qwen = ONLY deduplicated bullet facts (NOT raw article text).
    Prompt strictly instructs model:
        - Do NOT introduce new information
        - Only use the provided facts
        - Keep it factual and neutral
        - Max 120 words

Failure fallback:
    If Qwen call fails, return None — the email formatter will use
    bullet points only (as specified in PRD §14 Case 4).

Input:  List[str]  (selected fact sentences, 7–12 items)
        List[str]  (representative article titles, for context)
Output: str        (abstractive paragraph, ≤120 words) OR None on failure
"""

import json
import urllib.request
from typing import List, Optional

from config import config
from utils.logger import get_logger

logger = get_logger("summarization.abstractive")

_ABSTRACTIVE_PROMPT_TEMPLATE = """\
You are given verified bullet points extracted from multiple news sources about a single geopolitical event.

Write a concise paragraph summarizing the event.

STRICT RULES:
- Do NOT introduce any new information not present in the bullet points below
- Only use the provided facts
- Keep it strictly factual and neutral in tone
- Maximum {max_words} words in your paragraph
- Do not add opinions, predictions, or speculation
- Write in third person, past or present tense as appropriate

Bullet Points:
{bullets}

Write the summary paragraph now:"""


def _call_ollama(prompt: str) -> Optional[str]:
    """
    Call Ollama's /api/generate endpoint with the Qwen model.
    Returns the generated text or None on failure.
    """
    url     = f"{config.LLM_ENDPOINT}/api/generate"
    payload = {
        "model":  config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": config.QWEN_MAX_TOKENS,
            "temperature": 0.1,   # low temperature for factual grounding
            "top_p":       0.9,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip()
    except Exception as e:
        logger.error(f"[ABSTRACTIVE] Ollama call failed: {e}")
        return None


def _trim_to_word_limit(text: str, max_words: int) -> str:
    """Hard-trim output to max_words if Qwen overshoots."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def generate_abstractive_summary(
    fact_sentences: List[str],
    representative_titles: List[str],
) -> Optional[str]:
    """
    Generate an abstractive paragraph from extracted bullet facts using Qwen.

    Args:
        fact_sentences:        The 7–12 selected extractive facts.
        representative_titles: Titles of the MMR-selected articles (for context).

    Returns:
        Paragraph string (≤120 words) or None if Qwen is unavailable.
    """
    if not fact_sentences:
        logger.warning("[ABSTRACTIVE] No fact sentences provided. Skipping.")
        return None

    # Format bullet points
    bullets = "\n".join(f"- {s}" for s in fact_sentences)

    prompt = _ABSTRACTIVE_PROMPT_TEMPLATE.format(
        max_words=config.QWEN_PARAGRAPH_MAX_WORDS,
        bullets=bullets,
    )

    logger.debug(f"[ABSTRACTIVE] Calling Qwen with {len(fact_sentences)} bullet facts.")

    result = _call_ollama(prompt)

    if result is None:
        logger.error("[ABSTRACTIVE] Qwen generation failed. Will use bullet fallback.")
        return None

    # Enforce word limit
    result = _trim_to_word_limit(result, config.QWEN_PARAGRAPH_MAX_WORDS)

    logger.info(
        f"[ABSTRACTIVE] Generated paragraph: {len(result.split())} words"
    )

    if config.DEBUG:
        print(f"  [DEBUG ABSTRACTIVE] {result}")

    return result
