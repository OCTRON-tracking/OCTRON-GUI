import importlib.metadata
from importlib.metadata import version

try:
    __version__ = version("octron")
except importlib.metadata.PackageNotFoundError:
    __version__ = "no version"  

from .main import octron_widget
from .reader import octron_reader
from .yolo_octron.yolo_octron import YOLO_octron
from .yolo_octron.helpers.yolo_results import YOLO_results
from .yolo_octron.helpers.sam2_results import ANNOT_results

__all__ = (
    "octron_widget",
    "octron_reader",
    "YOLO_octron",
    "YOLO_results",
)
