"""
utils/logger.py — Structured per-stage logging to logs/YYYY-MM-DD.log
"""

import logging
import os
from datetime import datetime
from config import config


def get_logger(stage_name: str) -> logging.Logger:
    """
    Returns a logger namespaced to a pipeline stage.
    All loggers write to the same daily log file AND to stdout.
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)

    log_filename = os.path.join(
        config.LOG_DIR,
        datetime.now().strftime("%Y-%m-%d") + ".log"
    )

    logger = logging.getLogger(stage_name)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler — always write
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if config.DEBUG else logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
