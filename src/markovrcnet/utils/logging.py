import logging
import sys
from typing import Optional

formatter = logging.Formatter(
    "%(asctime)s [%(name)s] %(message)s",
    "%Y-%m-%d %H:%M:%S"
)

def _build_logger(name, logfile):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


logger_a = _build_logger("MiF", "MiF.log")
logger_b = _build_logger("MatrixLoader", "matrix.log")
logger_c = _build_logger("re_mcl", "re_mcl.log")


def get_logger(context: str) -> logging.Logger:
    context = context.lower()
    if context in ("mif", "mifdi", "distance", "similarity"):
        return logger_a
    elif context in ("matrix", "loader", "io"):
        return logger_b
    elif context in ("mcl", "re_mcl", "rmcl"):
        return logger_c
    else:
        raise ValueError(f"Unknown logging context: {context}")


def resolve_logger(logger: Optional[logging.Logger], context: str) -> logging.Logger:
    return logger if logger is not None else get_logger(context)
