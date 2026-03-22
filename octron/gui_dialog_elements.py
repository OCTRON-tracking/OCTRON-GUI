# This file collects custom input dialogs for the main OCTRON application, 
# such as little pop-ups where users can input additional information.

from qtpy.QtWidgets import (
    QDialog,
    QWidget,
    QLineEdit,
    QPushButton,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QTextEdit,
    QFileDialog,
)
from octron.cotracker_octron.helpers.skeleton_definition import SkeletonDefinition

class add_new_label_dialog(QDialog):
    """
    Allows user to add a new label name to the list 
    of labels in the octron GUI.
    
    
    """
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setWindowTitle("Create new label")
        self.label_name = QLineEdit()
        self.label_name.setObjectName(u"label_name")
        # self.label_name.setMinimumSize(QSize(60, 25))
        # self.label_name.setMaximumSize(QSize(60, 25))
        self.label_name.setInputMask(u"")
        self.label_name.setText(u"")
        self.label_name.setMaxLength(100)
        
        
        self.add_btn = QPushButton("Add")
        self.cancel_btn = QPushButton("Cancel")

        layout = QGridLayout()
        layout.addWidget(QLabel("Label name:"), 0, 0)
        layout.addWidget(self.label_name, 0, 1)
        layout.addWidget(self.add_btn, 1, 0)
        layout.addWidget(self.cancel_btn, 1, 1)
        self.setLayout(layout)

        self.add_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        
class remove_label_dialog(QDialog):
    """
    A dialog that shows a list of  current label_names 
    and allows the user to click on an entry to remove it.
    """
    def __init__(self, parent: QWidget, items: list):
        """
        Parameters
        ----------
        parent : QWidget
            That is the octron main GUI 
        items : list
            A list of current label names
        
        
        """
        super().__init__(parent)
        self.setWindowTitle("Remove label")
        self.resize(300, 200)
        
        # Create a list widget and add items if provided
        self.list_widget = QListWidget()
        if items:
            self.list_widget.addItems(items)
        
        # Create buttons
        self.remove_btn = QPushButton("Remove")
        self.cancel_btn = QPushButton("Cancel")
        
        # Layout setup
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("Select a label to remove:"))
        main_layout.addWidget(self.list_widget)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.remove_btn)
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        self.remove_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

class remove_video_dialog(QDialog):
    """
    A dialog that shows a list of  current videos for prediction, 
    and allows the user to click on an entry to remove it.
    """
    def __init__(self, parent: QWidget, items: list):
        """
        Parameters
        ----------
        parent : QWidget
            That is the octron main GUI 
        items : list
            A list of current label names
        
        
        """
        super().__init__(parent)
        self.setWindowTitle("Remove video(s) from list")
        self.resize(300, 200)
        
        # Create a list widget and add items if provided
        self.list_widget = QListWidget()
        if items:
            self.list_widget.addItems(items)
        
        # Create buttons
        self.remove_btn = QPushButton("Remove")
        self.cancel_btn = QPushButton("Cancel")
        
        # Layout setup
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("Select one or multiple videos to remove:"))
        main_layout.addWidget(self.list_widget)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.remove_btn)
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
        
        self.remove_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)


class skeleton_setup_dialog(QDialog):
    """
    Dialog for defining a skeleton (keypoint names + optional connectivity).

    The user can either type keypoint names (one per line) or load them
    from a YAML file. Connectivity is specified as comma-separated pairs
    of keypoint indices, e.g. "0-1, 0-2, 1-3".
    """
    def __init__(self, parent: QWidget, existing_names: list = None):
        """
        Parameters
        ----------
        parent : QWidget
            The octron main GUI
        existing_names : list, optional
            Pre-fill with existing keypoint names (e.g. when editing)
        """
        super().__init__(parent)
        self.setWindowTitle("Setup skeleton")
        self.resize(350, 400)

        main_layout = QVBoxLayout()

        # Keypoint names text area
        main_layout.addWidget(QLabel("Keypoint names (one per line):"))
        self.keypoint_text = QTextEdit()
        self.keypoint_text.setPlaceholderText("nose\nleft_ear\nright_ear\ntail_base")
        if existing_names:
            self.keypoint_text.setText("\n".join(existing_names))
        main_layout.addWidget(self.keypoint_text)

        # Load from YAML button
        self.load_yaml_btn = QPushButton("Load from YAML")
        self.load_yaml_btn.clicked.connect(self._load_yaml)
        main_layout.addWidget(self.load_yaml_btn)

        # Connectivity input
        main_layout.addWidget(QLabel("Connectivity (optional, e.g. 0-1, 0-2, 1-3):"))
        self.connectivity_edit = QLineEdit()
        self.connectivity_edit.setPlaceholderText("0-1, 0-2, 1-3")
        main_layout.addWidget(self.connectivity_edit)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def _load_yaml(self):
        """Open a file dialog to load a skeleton YAML file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load skeleton YAML", "", "YAML Files (*.yaml *.yml)"
        )
        if not file_path:
            return

        try:
            skeleton = SkeletonDefinition.load_yaml(file_path)
        except Exception as e:
            self.keypoint_text.setText(f"Error loading YAML: {e}")
            return

        self.keypoint_text.setText("\n".join(skeleton.keypoint_names))
        if skeleton.connectivity:
            pairs = [f"{i}-{j}" for i, j in skeleton.connectivity]
            self.connectivity_edit.setText(", ".join(pairs))

    def get_keypoint_names(self) -> list:
        """Return the list of keypoint names from the text area."""
        text = self.keypoint_text.toPlainText().strip()
        if not text:
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

    def get_connectivity(self) -> list:
        """Parse connectivity string into list of (int, int) tuples."""
        text = self.connectivity_edit.text().strip()
        if not text:
            return None
        pairs = []
        for part in text.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    i, j = part.split("-")
                    pairs.append((int(i.strip()), int(j.strip())))
                except ValueError:
                    continue
        return pairs if pairs else None
        