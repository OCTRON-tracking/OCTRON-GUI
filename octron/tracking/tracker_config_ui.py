import copy

import yaml
from boxmot.appearance.reid import config
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

trained_reid_models = config.TRAINED_URLS

# Define the set of base tracker params (those are passed from all boxmot trackers during init)
BASEPARAMS = [
    "det_thresh",
    "max_age",
    "max_obs",
    "min_hits",
    "iou_threshold",
    "per_class",
    "nr_classes",
    "asso_func",
    "is_obb",
]


class BoxmotTrackerConfigDialog(QDialog):
    def __init__(
        self,
        parent=None,
        tracker_id=None,
        tracker_config=None,
        config_path=None,
    ):
        """Dialog for configuring tracker parameters

        Parameters
        ----------
        parent: Parent widget
        tracker_id: ID of the tracker being configured
        tracker_config: Dict containing the tracker configuration
        config_path: Path to the YAML file where config will be saved

        """
        super().__init__(parent)

        self.tracker_id = tracker_id
        self.tracker_config = copy.deepcopy(
            tracker_config
        )  # Make a copy to avoid modifying original until save
        self.config_path = config_path
        self.original_config = copy.deepcopy(
            tracker_config
        )  # Keep original for reset

        self.parameter_widgets = {}  # Store widgets for accessing later
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setup_ui()

    def setup_ui(self):
        """Set up the dialog UI"""
        # Set window properties
        tracker_name = self.tracker_config[self.tracker_id]["name"]
        self.setWindowTitle(f"Configure {tracker_name}")

        self.setFixedWidth(450)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # Set minimum height but allow resizing in that direction
        window_height = self.tracker_config[self.tracker_id]["win_height"]
        self.setMinimumHeight(window_height)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            10, 10, 10, 10
        )  # Tighter margins for narrow dialog

        # Create scroll area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # Disable both horizontal and vertical scrollbars
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(5)  # Reduce spacing to fit narrow dialog

        # Get parameters
        parameters = self.tracker_config[self.tracker_id]["parameters"]
        is_reid = self.tracker_config[self.tracker_id]["is_reid"]

        # Create parameter groups
        base_params_group = QGroupBox("Base Parameters")
        base_params_layout = QFormLayout(base_params_group)

        specific_params_group = QGroupBox(
            f"{tracker_name} - specific Parameters"
        )
        specific_params_layout = QFormLayout(specific_params_group)

        # Process parameters
        base_params = BASEPARAMS

        # Add parameters to appropriate groups
        for param_name, param_config in parameters.items():
            if param_config.get("disabled", False):
                continue  # Skip disabled parameters

            # Create appropriate widget based on gui_element
            widget = self.create_widget_for_param(param_config)

            # Set tooltip with docstring
            if "docstring" in param_config:
                widget.setToolTip(param_config["docstring"])

            # Store widget for later access
            self.parameter_widgets[param_name] = widget

            # Add to appropriate layout
            if param_name in base_params:
                base_params_layout.addRow(QLabel(param_name), widget)
            else:
                specific_params_layout.addRow(QLabel(param_name), widget)

        if is_reid:
            # Create a group box for ReID model selection
            reid_group = QGroupBox("ReID model selection")
            reid_layout = QFormLayout(reid_group)
            reid_model_combo = QComboBox()

            # Add available ReID models to the combo box
            for model_name in trained_reid_models:
                reid_model_combo.addItem(model_name)

            reid_model_combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToContents
            )

            # Set current model from config
            current_reid_model = self.tracker_config[self.tracker_id][
                "reid_model"
            ]
            if current_reid_model in trained_reid_models:
                reid_model_combo.setCurrentText(current_reid_model)

            # Add to layout
            reid_layout.addRow(QLabel("ReID Model:"), reid_model_combo)
            # Store widget with special key for later access in save_config
            self.parameter_widgets["__reid_model"] = reid_model_combo
            scroll_layout.addWidget(reid_group)

        # Add groups to scroll layout
        scroll_layout.addWidget(base_params_group)
        if len(specific_params_layout) > 0:
            scroll_layout.addWidget(specific_params_group)

        # Add stretch to push everything to the top
        scroll_layout.addStretch()

        # Set scroll widget
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Button layout
        button_layout = QHBoxLayout()

        # Reset button
        reset_btn = QPushButton("Reset Defaults")
        reset_btn.clicked.connect(self.reset_defaults)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()

        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        # Save button
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self.save_config)
        button_layout.addWidget(save_btn)

        main_layout.addLayout(button_layout)

    def create_widget_for_param(self, param_config):
        """Create appropriate widget based on gui_element type"""
        gui_element = param_config.get("gui_element", "")
        current_value = param_config.get("current_value")

        if gui_element == "spin_box":
            widget = QSpinBox()
            if "range" in param_config:
                widget.setMinimum(param_config["range"][0])
                widget.setMaximum(param_config["range"][1])
            widget.setValue(current_value)

        elif gui_element == "double_spin_box":
            widget = QDoubleSpinBox()
            widget.setDecimals(2)
            if "range" in param_config:
                widget.setMinimum(param_config["range"][0])
                widget.setMaximum(param_config["range"][1])
            widget.setValue(current_value)
            widget.setSingleStep(0.01)

        elif gui_element == "checkbox":
            widget = QCheckBox()
            widget.setChecked(current_value)

        elif gui_element == "combo_box":
            widget = QComboBox()
            if "options" in param_config:
                # Add items first
                widget.addItems(param_config["options"])
                if current_value in param_config["options"]:
                    widget.setCurrentText(current_value)
                widget.setSizeAdjustPolicy(
                    QComboBox.SizeAdjustPolicy.AdjustToContents
                )
        else:
            # Default to label if unknown element type
            widget = QLabel(str(current_value))

        return widget

    def reset_defaults(self):
        """Reset all parameters to default values"""
        parameters = self.tracker_config[self.tracker_id]["parameters"]

        for param_name, widget in self.parameter_widgets.items():
            # Handle special case for ReID model
            if param_name == "__reid_model":
                # Get original ReID model from the original config
                original_reid_model = self.original_config[self.tracker_id][
                    "reid_model"
                ]
                if original_reid_model in trained_reid_models:
                    widget.setCurrentText(original_reid_model)
                continue

            # Regular parameters
            param_config = parameters[param_name]
            default_value = param_config.get("default_value")

            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(default_value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(default_value)
            elif isinstance(widget, QComboBox):
                if (
                    "options" in param_config
                    and default_value in param_config["options"]
                ):
                    widget.setCurrentText(default_value)

    def save_config(self):
        """Save the configuration"""
        parameters = self.tracker_config[self.tracker_id]["parameters"]

        # Update config with widget values
        for param_name, widget in self.parameter_widgets.items():
            # Handle special case for ReID model
            if param_name == "__reid_model":
                self.tracker_config[self.tracker_id]["reid_model"] = (
                    widget.currentText()
                )
                continue
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                parameters[param_name]["current_value"] = widget.value()
            elif isinstance(widget, QCheckBox):
                parameters[param_name]["current_value"] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                parameters[param_name]["current_value"] = widget.currentText()

        # Save to file if path provided
        if self.config_path:
            with open(self.config_path, "w") as f:
                yaml.dump(self.tracker_config, f, default_flow_style=False)

        self.accept()

    def get_config(self):
        """Return the updated config"""
        return self.tracker_config


def open_boxmot_tracker_config_dialog(
    parent, tracker_id, tracker_config, config_path
):
    """Opens a modal dialog to configure BoxMOT tracker parameters"""
    dialog = BoxmotTrackerConfigDialog(
        parent, tracker_id, tracker_config, config_path
    )
    result = dialog.exec_()

    if result == QDialog.Accepted:
        # Config was saved, you can get the updated config if needed
        updated_config = dialog.get_config()
        return updated_config
    else:
        # User canceled, no changes were saved
        return None
