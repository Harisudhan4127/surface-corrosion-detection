"""src/utils/logger.py"""
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_dir: str = "logs", level=logging.INFO) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = RotatingFileHandler(Path(log_dir) / f"{name}.log",
                              maxBytes=5*1024*1024, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
