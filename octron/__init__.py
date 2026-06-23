import importlib.metadata
import warnings
from importlib.metadata import version


def _install_warning_filters():
    """Suppress known dependency deprecations that OCTRON cannot fix directly.

    napari/psygnal currently define Pydantic v2 models that still use
    ``json_encoders``. Pydantic emits the warning from inside its own schema
    generation code when those dependency models are created. Keep the filter
    narrow so unrelated deprecations remain visible.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*`json_encoders` is deprecated.*",
        category=DeprecationWarning,
        module=r"pydantic\..*",
    )


_install_warning_filters()

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
    if name == "octron_widget":
        from .main import octron_widget
        return octron_widget
    if name == "octron_reader":
        from .reader import octron_reader
        return octron_reader
    if name == "YOLO_octron":
        from .yolo_octron.yolo_octron import YOLO_octron
        return YOLO_octron
    if name == "YOLO_results":
        from .yolo_octron.helpers.yolo_results import YOLO_results
        return YOLO_results
    if name == "ANNOT_results":
        from .yolo_octron.helpers.sam2_results import ANNOT_results
        return ANNOT_results
    raise AttributeError(f"module 'octron' has no attribute {name!r}")
