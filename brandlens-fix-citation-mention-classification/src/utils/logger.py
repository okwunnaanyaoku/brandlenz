"""Logging utilities for the BrandLens application."""

from __future__ import annotations

import logging
from typing import Optional

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_configured = False


def _resolve_level(level: str) -> int:
    numeric_level = logging.getLevelName(level.upper())
    if isinstance(numeric_level, str):
        raise ValueError(f"Unknown log level: {level}")
    return int(numeric_level)


def configure_logging(
    *,
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    force: bool = False,
) -> None:
    """Configure global logging handlers.

    Args:
        level: Logging level name (e.g. "INFO", "DEBUG").
        log_file: Optional path to a file that should receive logs in addition to stdout.
        log_format: Logging format string applied to all handlers.
        force: When ``True`` reconfigure logging even if it was already set up.
    """

    global _configured
    if _configured and not force:
        return

    handlers = []
    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=_resolve_level(level),
        handlers=handlers,
        force=True,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with the BrandLens defaults."""

    if not _configured:
        configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger", "DEFAULT_LOG_FORMAT"]
