"""OCTRON prediction cleaner widget.

Dockable napari widget for merging over-segmented tracks/masks (produced
when a real animal briefly loses tracking and BoxMOT assigns it a new
track ID) back into coherent timelines.

This is currently a visible shell only: the UI is built and displayed,
but no button/table logic is wired up yet. See ``cleanup_gui_elements.py``
for how the widgets are constructed (verbatim from the .ui-generated code).
"""

import os
from pathlib import Path

from loguru import logger
from qtpy.QtWidgets import QWidget

from octron.cleanup_octron.cleanup_gui_elements import (
    octron_prediction_cleaner_gui_elements,
)


class octron_prediction_cleaner_widget(QWidget):
    """Main prediction cleaner widget class."""

    def __init__(self, viewer=None, parent=None):
        """Initialize the prediction cleaner widget (UI shell, no logic).

        Parameters
        ----------
        viewer : napari.viewer.Viewer, optional
            The napari viewer this widget is docked into. napari calls
            widget factories as ``WidgetClass(viewer)``, so this must be
            the first positional parameter for the napari.yaml ``widgets``
            contribution (manual Plugins-menu invocation) to work; the
            auto-dock call site in ``load_predictions()`` also passes it
            explicitly. Not used yet (UI shell only).
        parent : QWidget, optional
            Qt parent widget (rarely set explicitly; napari manages this
            via ``add_dock_widget``).

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
        logger.debug("Prediction cleaner widget shell initialized")
