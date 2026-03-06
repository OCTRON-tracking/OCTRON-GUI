"""
Dialog for configuring which scikit-image region properties
are extracted during YOLO segmentation prediction.

Modeled after BoxmotTrackerConfigDialog in tracking/tracker_config_ui.py.
"""

import re
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QCheckBox, QFrame, QScrollArea,
    QWidget, QGroupBox, QSizePolicy
)
from qtpy.QtCore import Qt

from octron.yolo_octron.constants import ALL_REGION_PROPERTIES, DEFAULT_REGION_PROPERTIES

# Hardcoded original defaults (used by "Reset Defaults" button).
_ORIGINAL_DEFAULTS = ('area',)


class RegionPropertiesDialog(QDialog):
    """Modal dialog that displays all scikit-image region property options
    in a two-column layout with a checkbox next to each one."""

    def __init__(self, parent=None, current_selection=None):
        """
        Parameters
        ----------
        parent : QWidget
            Parent widget.
        current_selection : set or tuple or list, optional
            Currently enabled property names.  Defaults to DEFAULT_REGION_PROPERTIES.
        """
        super().__init__(parent)

        if current_selection is None:
            current_selection = set(DEFAULT_REGION_PROPERTIES)
        else:
            current_selection = set(current_selection)

        self.current_selection = current_selection
        self.checkboxes: dict[str, QCheckBox] = {}  # prop_name -> QCheckBox

        self.setWindowModality(Qt.WindowModality.WindowModal)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _setup_ui(self):
        self.setWindowTitle("Region Properties")
        self.setFixedWidth(450)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMinimumHeight(400)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Scroll area --------------------------------------------------
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(5)

        # Build a group box per category
        for category, properties in ALL_REGION_PROPERTIES.items():
            group = QGroupBox(category)
            grid = QGridLayout(group)
            grid.setHorizontalSpacing(15)
            grid.setVerticalSpacing(4)

            for idx, prop_name in enumerate(properties):
                cb = QCheckBox(prop_name)
                cb.setChecked(prop_name in self.current_selection)
                self.checkboxes[prop_name] = cb
                row = idx // 2
                col = idx % 2
                grid.addWidget(cb, row, col)

            scroll_layout.addWidget(group)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        # Select All / Deselect All ------------------------------------
        toggle_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        toggle_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        toggle_layout.addWidget(deselect_all_btn)

        toggle_layout.addStretch()
        main_layout.addLayout(toggle_layout)

        # Separator ----------------------------------------------------
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Bottom buttons -----------------------------------------------
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        button_layout.addWidget(save_btn)

        main_layout.addLayout(button_layout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _select_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(True)

    def _deselect_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(False)

    def _reset_defaults(self):
        for name, cb in self.checkboxes.items():
            cb.setChecked(name in _ORIGINAL_DEFAULTS)

    def get_selected_properties(self) -> tuple:
        """Return a tuple of currently checked property names."""
        return tuple(name for name, cb in self.checkboxes.items() if cb.isChecked())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @staticmethod
    def _write_defaults_to_constants(selected: tuple):
        """Rewrite DEFAULT_REGION_PROPERTIES in constants.py on disk."""
        import octron.yolo_octron.constants as _mod
        path = _mod.__file__

        with open(path, 'r') as f:
            content = f.read()

        # Build the replacement tuple string
        if selected:
            items = '\n'.join(f"    '{p}'," for p in selected)
            new_block = f"DEFAULT_REGION_PROPERTIES = (\n{items}\n)"
        else:
            new_block = "DEFAULT_REGION_PROPERTIES = ()"

        # Replace the existing DEFAULT_REGION_PROPERTIES block
        content = re.sub(
            r'DEFAULT_REGION_PROPERTIES\s*=\s*\(.*?\)',
            new_block,
            content,
            count=1,
            flags=re.DOTALL,
        )

        with open(path, 'w') as f:
            f.write(content)

        # Reload so the running process picks up the change
        import importlib
        importlib.reload(_mod)

    def _save(self):
        selected = self.get_selected_properties()
        self._write_defaults_to_constants(selected)
        self.accept()

    # ------------------------------------------------------------------
    # Public accessor
    # ------------------------------------------------------------------
    def get_config(self):
        """Return the selected properties (mirrors BoxmotTrackerConfigDialog API)."""
        return self.get_selected_properties()


def open_region_properties_dialog(parent, current_selection=None):
    """Opens a modal dialog to configure region properties.

    Returns the selected properties as a tuple, or None if cancelled.
    """
    dialog = RegionPropertiesDialog(parent, current_selection)
    result = dialog.exec_()

    if result == QDialog.Accepted:
        return dialog.get_selected_properties()
    return None
