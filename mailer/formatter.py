"""
email/formatter.py

Stage 9: Email Formatting (PRD §2 — STRICT FORMAT).

Each email section MUST follow this exact structure:

    Event: <Generated Title>

    <Abstractive Summary Paragraph>

    Extractive Facts:
    - Fact 1
    - Fact 2
    ...
    - Fact N

    Sources:
    - URL 1
    - URL 2
    - URL 3

HARD CONSTRAINTS (enforced here):
    - Each event must have ≥ 2 sources  (events with < 2 are skipped)
    - Max 6 events per email
    - If paragraph is None (Qwen failed) → omit paragraph section
    - Bullet points: extractive (not modified)

Input:  List[SummarizedEvent]
Output: (subject: str, body: str)  — plain text email components
"""

from datetime import datetime
from typing import List, Dict, Tuple

from config import config
from utils.logger import get_logger

logger = get_logger("email.formatter")

_DIVIDER = "-" * 72


def _format_event_block(event: Dict, index: int) -> str:
    """Format a single SummarizedEvent as a plain-text block."""
    title        = event.get("title", "Geopolitical Event")
    paragraph    = event.get("paragraph")
    bullet_facts = event.get("bullet_facts", [])
    sources      = event.get("sources", [])

    lines = []

    # Header — matches PRD §2 required format
    lines.append(f"Event {index}: {title}")
    lines.append(_DIVIDER)

    # Abstractive paragraph (if available)
    if paragraph:
        lines.append("")
        lines.append(paragraph)

    # Extractive facts
    lines.append("")
    lines.append("Extractive Facts:")
    for fact in bullet_facts:
        lines.append(f"  - {fact}")

    # Sources
    lines.append("")
    lines.append("Sources:")
    for url in sources:
        lines.append(f"  - {url}")

    lines.append("")
    return "\n".join(lines)


def _email_header(n_events: int) -> str:
    now = datetime.now().strftime("%A, %d %B %Y — %H:%M")
    lines = [
        "=" * 72,
        "  GEOPOLITICAL INTELLIGENCE BRIEFING",
        f"  {now}",
        f"  Top {n_events} Events",
        "=" * 72,
        "",
    ]
    return "\n".join(lines)


def _email_footer() -> str:
    return (
        "\n" + _DIVIDER + "\n"
        "This briefing was generated automatically by the Geopolitical "
        "News Intelligence System.\n"
        "Sources: global RSS feeds. Summaries are grounded in extracted facts only.\n"
    )


def format_email(events: List[Dict]) -> Tuple[str, str]:
    """
    Format a list of SummarizedEvents into a plain-text email.

    Enforces:
        - Skip events with < 2 sources
        - Limit to TOP_N_EVENTS (6)

    Returns:
        (subject, body) — both plain text strings.
    """
    # Filter: must have ≥ 2 sources
    valid_events = []
    for event in events:
        sources = event.get("sources", [])
        if len(sources) >= 2:
            valid_events.append(event)
        else:
            logger.warning(
                f"[FORMAT] Skipping event '{event.get('title','?')[:60]}' "
                f"— only {len(sources)} source(s) (need ≥ 2)"
            )

    # Cap at TOP_N_EVENTS
    valid_events = valid_events[:config.TOP_N_EVENTS]

    if not valid_events:
        subject = "Geopolitical Intelligence Briefing — Insufficient Data"
        body    = (
            "Insufficient data for today's briefing.\n"
            "The pipeline could not produce events with ≥ 2 sources.\n"
            "Please check the logs for details."
        )
        logger.warning("[FORMAT] No valid events to include in email.")
        return subject, body

    n = len(valid_events)
    now_str = datetime.now().strftime("%d %b %Y %H:%M")
    subject = f"Geopolitical Briefing — {n} Events — {now_str}"

    body_parts = [_email_header(n)]
    for i, event in enumerate(valid_events, 1):
        body_parts.append(_format_event_block(event, i))

    body_parts.append(_email_footer())
    body = "\n".join(body_parts)

    logger.info(f"[FORMAT] Email formatted: {n} events, {len(body)} characters")
    return subject, body
