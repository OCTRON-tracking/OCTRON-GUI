"""OCTRON prediction cleaner widget.

Dockable napari widget for merging over-segmented tracks/masks (produced
when a real animal briefly loses tracking and BoxMOT assigns it a new
track ID) back into coherent timelines. See ``cleanup_gui_elements.py``
for how the widgets are constructed (verbatim from the .ui-generated code)
and ``cleanup_handler.py`` for the actual join/fuse/revert logic.
"""

import os
from pathlib import Path

from loguru import logger
from qtpy.QtWidgets import QWidget

from octron.cleanup_octron.cleanup_gui_elements import (
    octron_prediction_cleaner_gui_elements,
)
from octron.cleanup_octron.cleanup_handler import CleanerHandler


class octron_prediction_cleaner_widget(QWidget):
    """Main prediction cleaner widget class."""

    def __init__(self, viewer=None, parent=None, analysis_results=None, save_dir=None):
        """Initialize the prediction cleaner widget.

        Parameters
        ----------
        viewer : napari.viewer.Viewer, optional
            The napari viewer this widget is docked into. napari calls
            widget factories as ``WidgetClass(viewer)``, so this must be
            the first positional parameter for the napari.yaml ``widgets``
            contribution (manual Plugins-menu invocation) to work; the
            auto-dock call site in ``load_predictions()`` also passes it
            explicitly.
        parent : QWidget, optional
            Qt parent widget (rarely set explicitly; napari manages this
            via ``add_dock_widget``).
        analysis_results : AnalysisResults, optional
            Already-loaded results for the prediction folder currently
            shown in ``viewer`` (passed by ``load_predictions()``). None
            when the widget is opened manually from the Plugins menu
            without a prediction folder in view yet -- the widget then
            just shows its empty/disabled default state.
        save_dir : str or Path, optional
            Path to the prediction folder ``analysis_results`` was loaded
            from. Required (together with ``analysis_results``) for any
            join/fuse/revert/reset operation, since those write to disk.

        """
        super().__init__(parent)
        self._viewer = viewer
        # octron/cleanup_octron/cleanup_widget.py -> parent.parent == octron/
        # (base_path is used to resolve icons under octron/qt_gui/)
        self.base_path = Path(os.path.abspath(__file__)).parent.parent

        # Build the UI (see cleanup_gui_elements.py; generated from
        # octron/qt_gui/octron_prediction_cleaner.ui)
        self.gui = octron_prediction_cleaner_gui_elements(
            self, base_path=self.base_path
        )

        # Business logic lives in the handler (mirrors AnalysisHandler's
        # relationship to the main octron_widget).
        self.cleaner_handler = CleanerHandler(
            self, analysis_results=analysis_results, save_dir=save_dir
        )
        self.cleaner_handler.connect_signals()
        self.cleaner_handler.refresh_from_results()
        logger.debug("Prediction cleaner widget initialized")
