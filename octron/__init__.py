import importlib.metadata
import warnings
from importlib.metadata import version

from pydantic import PydanticDeprecationWarning

from octron._logging import setup_logging

# Configure loguru with OCTRON's compact log format as soon as the package is
# imported. NOTE: some dependencies (e.g. boxmot) call logger.remove() /
# logger.add() with their own verbose format at their own import time. Since
# those dependencies are only pulled in lazily (see __getattr__ below), this
# initial call does not survive past the first such import — setup_logging()
# is called again after each lazy import for that reason.
setup_logging()


def _install_warning_filters():
    """Suppress known dependency deprecations that OCTRON cannot fix directly.

    napari/npe2/psygnal currently define Pydantic v2 models that still use
    deprecated APIs (``json_encoders``, extra ``Field`` kwargs such as
    ``min_ver``/``hide_docs``, ``allow_mutation``, etc.). These are raised
    from pydantic's own internals (not from a stable, filterable module
    path) with wording that varies per deprecated API, so filter on the
    ``PydanticDeprecationWarning`` category itself rather than matching
    individual message strings — OCTRON's own pydantic models (see
    sam_octron/object_organizer.py) already use current, non-deprecated
    APIs, so this cannot mask an actionable warning from OCTRON's own code.
    """
    warnings.filterwarnings(
        "ignore",
        category=PydanticDeprecationWarning,
    )


_install_warning_filters()


class _suppress_known_dependency_warnings:
    """Context manager for lazy imports that trigger known dependency warnings."""

    def __enter__(self):
        self._catch = warnings.catch_warnings()
        self._catch.__enter__()
        _install_warning_filters()

    def __exit__(self, exc_type, exc, tb):
        return self._catch.__exit__(exc_type, exc, tb)


try:
    __version__ = version("octron")
except importlib.metadata.PackageNotFoundError:
    __version__ = "no version"


__all__ = (
    "octron_widget",
    "octron_reader",
    "AnalysisOctron",
    "AnalysisResults",
    "ANNOT_results",
)


def __getattr__(name):
    # Some dependency imports mutate the warnings filter stack. Re-apply the
    # OCTRON-specific filters immediately before lazy heavy imports (napari,
    # psygnal, etc.) so users do not see known dependency deprecations when
    # doing `from octron import AnalysisResults` or similar.
    _install_warning_filters()
    if name == "octron_widget":
        with _suppress_known_dependency_warnings():
            from .main import octron_widget
        setup_logging()
        return octron_widget
    if name == "octron_reader":
        with _suppress_known_dependency_warnings():
            from .reader import octron_reader
        setup_logging()
        return octron_reader
    if name == "AnalysisOctron":
        with _suppress_known_dependency_warnings():
            from .analysis_octron.analysis_octron import AnalysisOctron
        setup_logging()
        return AnalysisOctron
    if name == "AnalysisResults":
        with _suppress_known_dependency_warnings():
            from .analysis_octron.helpers.analysis_results import (
                AnalysisResults,
            )
        setup_logging()
        return AnalysisResults
    if name == "ANNOT_results":
        with _suppress_known_dependency_warnings():
            from .analysis_octron.helpers.sam2_results import ANNOT_results
        setup_logging()
        return ANNOT_results
    raise AttributeError(f"module 'octron' has no attribute {name!r}")
