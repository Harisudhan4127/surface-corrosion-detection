"""
logger.py — Centralised logging setup (console + rotating file handler)
"""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_dir: str = "logs", level=logging.INFO) -> logging.Logger:
    """
    Returns a logger that writes to both console and a rotating file.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (5 MB × 3 backups)
    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
