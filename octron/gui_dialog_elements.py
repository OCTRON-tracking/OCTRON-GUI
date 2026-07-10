# This file collects custom input dialogs for the main OCTRON application,
# such as little pop-ups where users can input additional information.

from qtpy.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class add_new_label_dialog(QDialog):
    """Allows user to add a new label name to the list
    of labels in the octron GUI.


    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setWindowTitle("Create new label")
        self.label_name = QLineEdit()
        self.label_name.setObjectName("label_name")
        # self.label_name.setMinimumSize(QSize(60, 25))
        # self.label_name.setMaximumSize(QSize(60, 25))
        self.label_name.setInputMask("")
        self.label_name.setText("")
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
    """A dialog that shows a list of  current label_names
    and allows the user to click on an entry to remove it.
    """

    def __init__(self, parent: QWidget, items: list):
        """Parameters
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
    """A dialog that shows a list of  current videos for prediction,
    and allows the user to click on an entry to remove it.
    """

    def __init__(self, parent: QWidget, items: list):
        """Parameters
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
        main_layout.addWidget(
            QLabel("Select one or multiple videos to remove:")
        )
        main_layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.remove_btn)
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        self.remove_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
