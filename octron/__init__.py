import importlib.metadata
from importlib.metadata import version

try:
    __version__ = version("octron")
except importlib.metadata.PackageNotFoundError:
    __version__ = "no version"


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


__all__ = (
    "octron_widget",
    "octron_reader",
    "YOLO_octron",
    "YOLO_results",
)
