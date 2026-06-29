import importlib.metadata
import warnings
from importlib.metadata import version


def _install_warning_filters():
    """Suppress known dependency deprecations that OCTRON cannot fix directly.

    napari/psygnal currently define Pydantic v2 models that still use
    ``json_encoders``. Pydantic emits the warning from inside its own schema
    generation code when those dependency models are created. Keep the filter
    narrow by matching the exact warning text so unrelated deprecations remain
    visible.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*`json_encoders` is deprecated.*",
        category=DeprecationWarning,
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
    "YOLO_octron",
    "YOLO_results",
    "ANNOT_results",
)


def __getattr__(name):
    # Some dependency imports mutate the warnings filter stack. Re-apply the
    # OCTRON-specific filters immediately before lazy heavy imports (napari,
    # psygnal, etc.) so users do not see known dependency deprecations when
    # doing `from octron import YOLO_results` or similar.
    _install_warning_filters()
    if name == "octron_widget":
        with _suppress_known_dependency_warnings():
            from .main import octron_widget
        return octron_widget
    if name == "octron_reader":
        with _suppress_known_dependency_warnings():
            from .reader import octron_reader
        return octron_reader
    if name == "YOLO_octron":
        with _suppress_known_dependency_warnings():
            from .yolo_octron.yolo_octron import YOLO_octron
        return YOLO_octron
    if name == "YOLO_results":
        with _suppress_known_dependency_warnings():
            from .yolo_octron.helpers.yolo_results import YOLO_results
        return YOLO_results
    if name == "ANNOT_results":
        with _suppress_known_dependency_warnings():
            from .yolo_octron.helpers.sam2_results import ANNOT_results
        return ANNOT_results
    raise AttributeError(f"module 'octron' has no attribute {name!r}")
