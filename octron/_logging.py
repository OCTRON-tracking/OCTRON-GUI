"""Central logging configuration for OCTRON.

All modules import `from loguru import logger` and use it directly.
Call `setup_logging()` once at startup (from the GUI entry point or CLI
callback). The CLI branch can pass `debug=True` to enable DEBUG-level
output.
"""

import sys

from loguru import logger
from tqdm import tqdm

from octron._version import version as octron_version

_LOG_FORMAT_NORMAL = (
    "<cyan>{time:HH:mm:ss}</cyan> | <level>{level: <8}</level> | {message}"
)

_LOG_FORMAT_DEBUG = (
    "<cyan>{time:HH:mm:ss.SSS}</cyan> | "
    "<level>{level: <8}</level> | "
    "<dim>{name}:{line}</dim> | "
    "{message}"
)


def _tqdm_safe_sink(message) -> None:
    """Write a log record without corrupting active tqdm progress bars.

    ``tqdm`` renders its bars with a carriage return on ``sys.stderr``.
    A plain write to the same stream (a log line, a ``print``) does not
    clear the bar first, so on refresh the bar "staircases" onto new
    lines. Routing loguru through :meth:`tqdm.tqdm.write` clears any live
    bar, emits the already-formatted (colorized, newline-terminated)
    record, then redraws the bar on the next tick.
    """
    tqdm.write(message, end="", file=sys.stderr)


def setup_logging(debug: bool = False) -> None:
    """Configure loguru for the OCTRON application.

    Parameters
    ----------
    debug:
        When True, set log level to DEBUG and include source location in every
        message.  When False (default), use INFO level with a compact format.

    This function is idempotent — calling it multiple times simply resets the
    handler to the current settings.

    """
    logger.remove()  # drop default stderr handler

    level = "DEBUG" if debug else "INFO"
    fmt = _LOG_FORMAT_DEBUG if debug else _LOG_FORMAT_NORMAL

    # Route logs through tqdm.write so log lines never fracture an active
    # progress bar (the bar is cleared, the record is written, then the bar
    # is redrawn). colorize=True is kept so colours still render.
    logger.add(
        _tqdm_safe_sink,
        level=level,
        format=fmt,
        colorize=True,
        enqueue=False,
    )

    if debug:
        logger.debug("Debug logging enabled")


def print_welcome() -> None:
    """Print a short welcome banner to stderr via loguru."""
    logger.info("=" * 48)
    logger.info(f"  OCTRON  v{octron_version}")
    logger.info("  Segmentation & tracking for animal behavior")
    logger.info("=" * 48)
