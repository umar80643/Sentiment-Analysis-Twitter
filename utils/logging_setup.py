"""Central logging configuration used by every entry point.

Replaces the project's former print()-based output with proper INFO/WARNING/
ERROR logging that also persists to logs/training.log for later inspection.
"""
from __future__ import annotations

import logging

from config import LOG_DIR


def configure_logging(level: int = logging.INFO, log_file: str = "training.log") -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        # Idempotent: safe to call more than once (e.g. from tests) without
        # duplicating log lines.
        return
    root.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_DIR / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
