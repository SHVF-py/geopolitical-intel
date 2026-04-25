"""
email/sender.py

Stage 10: Email Delivery via Gmail SMTP (PRD §11).

Method: Gmail SMTP (TLS, port 587)
Format: Plain text (MVP)

Failure handling:
    - On send failure: log full stack trace
    - On pipeline error: send error alert email to developer

Input:  (subject: str, body: str)
Output: bool — True if sent successfully
"""

import smtplib
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional

from config import config
from utils.logger import get_logger

logger = get_logger("email.sender")


def _build_message(
    subject: str,
    body: str,
    sender: str,
    receiver: str,
) -> MIMEMultipart:
    """Construct a MIME plain-text email message."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = receiver
    msg.attach(MIMEText(body, "plain", "utf-8"))
    return msg


def send_email(subject: str, body: str) -> bool:
    """
    Send the formatted briefing email via Gmail SMTP.

    Returns True on success, False on failure.
    On failure: logs full stack trace.
    """
    try:
        msg = _build_message(
            subject=subject,
            body=body,
            sender=config.EMAIL_SENDER,
            receiver=config.EMAIL_RECEIVER,
        )

        with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
            smtp.sendmail(
                config.EMAIL_SENDER,
                config.EMAIL_RECEIVER,
                msg.as_string(),
            )

        logger.info(
            f"[SENDER] Email sent successfully to {config.EMAIL_RECEIVER} "
            f"| Subject: '{subject}'"
        )
        return True

    except Exception as e:
        logger.error(
            f"[SENDER] Failed to send email: {e}\n"
            f"{traceback.format_exc()}"
        )
        return False


def send_error_alert(error_description: str, stack_trace: str = "") -> None:
    """
    Send a developer alert email when the pipeline encounters a critical failure.
    Defined in PRD §14 Alerting.
    """
    now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[ALERT] Geopolitical Pipeline Error — {now}"
    body    = (
        f"Geopolitical News Intelligence System — Pipeline Error\n"
        f"Timestamp: {now}\n\n"
        f"Error Description:\n{error_description}\n\n"
    )
    if stack_trace:
        body += f"Stack Trace:\n{stack_trace}\n"

    try:
        msg = _build_message(
            subject=subject,
            body=body,
            sender=config.EMAIL_SENDER,
            receiver=config.EMAIL_SENDER,   # alert goes to developer (sender account)
        )

        with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
            smtp.sendmail(
                config.EMAIL_SENDER,
                config.EMAIL_SENDER,
                msg.as_string(),
            )

        logger.info(f"[SENDER] Error alert sent to developer.")

    except Exception as e:
        logger.error(f"[SENDER] Could not send error alert: {e}")


def send_insufficient_data_notice() -> None:
    """
    Send PRD §14 Case 2 message: 'Insufficient data for today's briefing'.
    """
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"Geopolitical Briefing — Insufficient Data — {now}"
    body    = (
        f"Insufficient data for today's briefing.\n"
        f"Timestamp: {now}\n\n"
        "The pipeline could not gather enough articles to form events. "
        "Please check the logs for details."
    )
    send_email(subject, body)
