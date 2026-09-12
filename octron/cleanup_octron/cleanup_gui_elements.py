"""Prediction cleaner GUI elements.

This mirrors ``octron/gui_elements.py::octron_gui_elements`` for the main
widget: ``setupUi()`` below is pasted verbatim from
``octron/qt_gui/octron_prediction_cleaner_corrected.py``, which is itself
generated from ``octron/qt_gui/octron_prediction_cleaner.ui`` via

    uic -g python octron_prediction_cleaner.ui > octron_prediction_cleaner.py
    python conversion_code_cleaner.py   # -> octron_prediction_cleaner_corrected.py

To update the UI: edit the .ui file in Qt Designer, regenerate, then
replace the ``setupUi()`` body below with the fresh corrected output.
Do NOT hand-edit the ``setupUi()`` body itself -- keep it a straight
copy-paste so re-generation stays a mechanical, low-risk operation.
"""

import pathlib

from qtpy.QtCore import QCoreApplication, QRect, QSize, Qt
from qtpy.QtGui import QCursor, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)


class octron_prediction_cleaner_gui_elements(QWidget):
    """Build and hold all Qt widgets for the prediction cleaner window."""

    def __init__(self, parent: QWidget, base_path: pathlib.Path):
        """Initialize the GUI elements for the prediction cleaner widget.

        Parameters
        ----------
        parent : QWidget
            The parent widget for the GUI elements.
        base_path : pathlib.Path
            The base path for loading resources (e.g., icons, images).

        """
        super().__init__(parent)
        self.cleaner = parent

        # Initialize the GUI elements
        self.setupUi(base_path)

    ###### GUI SETUP CODE FROM QT DESIGNER ##############################
    def setupUi(self, base_path):
        if not self.cleaner.objectName():
            self.cleaner.setObjectName(u"self")
        self.cleaner.setEnabled(True)
        self.cleaner.resize(270, 630)
        self.cleaner.setMinimumSize(QSize(270, 630))
        self.cleaner.setMaximumSize(QSize(270, 630))
        self.cleaner.setCursor(QCursor(Qt.ArrowCursor))
        self.cleaner.setWindowOpacity(1.000000000000000)
        self.cleaner.verticalLayoutWidget = QWidget(self)
        self.cleaner.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.cleaner.verticalLayoutWidget.setGeometry(QRect(10, 10, 254, 611))
        self.cleaner.mainLayout = QVBoxLayout(self.cleaner.verticalLayoutWidget)
        self.cleaner.mainLayout.setSpacing(20)
        self.cleaner.mainLayout.setObjectName(u"mainLayout")
        self.cleaner.mainLayout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.cleaner.mainLayout.setContentsMargins(0, 5, 0, 0)
        self.cleaner.octron_cleaner_ui_logo = QLabel(self.cleaner.verticalLayoutWidget)
        self.cleaner.octron_cleaner_ui_logo.setObjectName(u"octron_cleaner_ui_logo")
        self.cleaner.octron_cleaner_ui_logo.setEnabled(True)
        self.cleaner.octron_cleaner_ui_logo.setMinimumSize(QSize(250, 50))
        self.cleaner.octron_cleaner_ui_logo.setMaximumSize(QSize(250, 50))
        self.cleaner.octron_cleaner_ui_logo.setBaseSize(QSize(0, 0))
        self.cleaner.octron_cleaner_ui_logo.setLineWidth(0)
        self.cleaner.octron_cleaner_ui_logo.setPixmap(QPixmap(f"{base_path}/qt_gui/octron_prediction_cleaner.svg"))
        self.cleaner.octron_cleaner_ui_logo.setScaledContents(False)
        self.cleaner.octron_cleaner_ui_logo.setAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.cleaner.mainLayout.addWidget(self.cleaner.octron_cleaner_ui_logo)

        self.cleaner.main_vertical_layout = QVBoxLayout()
        self.cleaner.main_vertical_layout.setSpacing(0)
        self.cleaner.main_vertical_layout.setObjectName(u"main_vertical_layout")
        self.cleaner.source_trace_groupbox = QGroupBox(self.cleaner.verticalLayoutWidget)
        self.cleaner.source_trace_groupbox.setObjectName(u"source_trace_groupbox")
        self.cleaner.source_trace_groupbox.setMinimumSize(QSize(250, 100))
        self.cleaner.source_trace_groupbox.setMaximumSize(QSize(250, 100))
        self.cleaner.verticalLayoutWidget_4 = QWidget(self.cleaner.source_trace_groupbox)
        self.cleaner.verticalLayoutWidget_4.setObjectName(u"verticalLayoutWidget_4")
        self.cleaner.verticalLayoutWidget_4.setGeometry(QRect(9, 30, 231, 61))
        self.cleaner.verticalLayout_5 = QVBoxLayout(self.cleaner.verticalLayoutWidget_4)
        self.cleaner.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.cleaner.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.cleaner.source_layers_comboBox = QComboBox(self.cleaner.verticalLayoutWidget_4)
        self.cleaner.source_layers_comboBox.setObjectName(u"source_layers_comboBox")
        self.cleaner.source_layers_comboBox.setMinimumSize(QSize(230, 25))
        self.cleaner.source_layers_comboBox.setMaximumSize(QSize(230, 25))

        self.cleaner.verticalLayout_5.addWidget(self.cleaner.source_layers_comboBox, 0, Qt.AlignmentFlag.AlignHCenter)

        self.cleaner.horizontalLayout_3 = QHBoxLayout()
        self.cleaner.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.cleaner.tracks_label = QLabel(self.cleaner.verticalLayoutWidget_4)
        self.cleaner.tracks_label.setObjectName(u"tracks_label")
        self.cleaner.tracks_label.setMinimumSize(QSize(60, 25))
        self.cleaner.tracks_label.setMaximumSize(QSize(60, 25))

        self.cleaner.horizontalLayout_3.addWidget(self.cleaner.tracks_label, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.cleaner.total_no_tracks_label = QLabel(self.cleaner.verticalLayoutWidget_4)
        self.cleaner.total_no_tracks_label.setObjectName(u"total_no_tracks_label")
        self.cleaner.total_no_tracks_label.setMinimumSize(QSize(60, 25))
        self.cleaner.total_no_tracks_label.setMaximumSize(QSize(60, 25))

        self.cleaner.horizontalLayout_3.addWidget(self.cleaner.total_no_tracks_label, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.cleaner.showall_layers_pushButton = QPushButton(self.cleaner.verticalLayoutWidget_4)
        self.cleaner.showall_layers_pushButton.setObjectName(u"showall_layers_pushButton")
        self.cleaner.showall_layers_pushButton.setMinimumSize(QSize(80, 25))
        self.cleaner.showall_layers_pushButton.setMaximumSize(QSize(80, 25))

        self.cleaner.horizontalLayout_3.addWidget(self.cleaner.showall_layers_pushButton, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.cleaner.verticalLayout_5.addLayout(self.cleaner.horizontalLayout_3)


        self.cleaner.main_vertical_layout.addWidget(self.cleaner.source_trace_groupbox)

        self.cleaner.tuning_params_groupbox = QGroupBox(self.cleaner.verticalLayoutWidget)
        self.cleaner.tuning_params_groupbox.setObjectName(u"tuning_params_groupbox")
        self.cleaner.tuning_params_groupbox.setMinimumSize(QSize(250, 120))
        self.cleaner.tuning_params_groupbox.setMaximumSize(QSize(250, 120))
        self.cleaner.verticalLayoutWidget_2 = QWidget(self.cleaner.tuning_params_groupbox)
        self.cleaner.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.cleaner.verticalLayoutWidget_2.setGeometry(QRect(10, 30, 231, 37))
        self.cleaner.max_gap_frames_horizontal_layout = QHBoxLayout(self.cleaner.verticalLayoutWidget_2)
        self.cleaner.max_gap_frames_horizontal_layout.setObjectName(u"max_gap_frames_horizontal_layout")
        self.cleaner.max_gap_frames_horizontal_layout.setContentsMargins(0, 0, 0, 0)
        self.cleaner.max_gap_time_label = QLabel(self.cleaner.verticalLayoutWidget_2)
        self.cleaner.max_gap_time_label.setObjectName(u"max_gap_time_label")
        self.cleaner.max_gap_time_label.setMinimumSize(QSize(90, 35))
        self.cleaner.max_gap_time_label.setMaximumSize(QSize(100, 35))
        self.cleaner.max_gap_time_label.setWordWrap(True)

        self.cleaner.max_gap_frames_horizontal_layout.addWidget(self.cleaner.max_gap_time_label)

        self.cleaner.max_gap_time_spinbox = QSpinBox(self.cleaner.verticalLayoutWidget_2)
        self.cleaner.max_gap_time_spinbox.setObjectName(u"max_gap_time_spinbox")
        self.cleaner.max_gap_time_spinbox.setMinimumSize(QSize(120, 25))
        self.cleaner.max_gap_time_spinbox.setMaximumSize(QSize(120, 25))
        self.cleaner.max_gap_time_spinbox.setMaximum(99999)
        self.cleaner.max_gap_time_spinbox.setValue(10)

        self.cleaner.max_gap_frames_horizontal_layout.addWidget(self.cleaner.max_gap_time_spinbox)

        self.cleaner.verticalLayoutWidget_3 = QWidget(self.cleaner.tuning_params_groupbox)
        self.cleaner.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.cleaner.verticalLayoutWidget_3.setGeometry(QRect(10, 70, 231, 37))
        self.cleaner.max_gap_space_horizontal_layout = QHBoxLayout(self.cleaner.verticalLayoutWidget_3)
        self.cleaner.max_gap_space_horizontal_layout.setObjectName(u"max_gap_space_horizontal_layout")
        self.cleaner.max_gap_space_horizontal_layout.setContentsMargins(0, 0, 0, 0)
        self.cleaner.max_gap_space_label = QLabel(self.cleaner.verticalLayoutWidget_3)
        self.cleaner.max_gap_space_label.setObjectName(u"max_gap_space_label")
        self.cleaner.max_gap_space_label.setMinimumSize(QSize(90, 35))
        self.cleaner.max_gap_space_label.setMaximumSize(QSize(100, 35))
        self.cleaner.max_gap_space_label.setWordWrap(True)

        self.cleaner.max_gap_space_horizontal_layout.addWidget(self.cleaner.max_gap_space_label)

        self.cleaner.max_gap_space_spinbox = QSpinBox(self.cleaner.verticalLayoutWidget_3)
        self.cleaner.max_gap_space_spinbox.setObjectName(u"max_gap_space_spinbox")
        self.cleaner.max_gap_space_spinbox.setMinimumSize(QSize(120, 25))
        self.cleaner.max_gap_space_spinbox.setMaximumSize(QSize(120, 25))
        self.cleaner.max_gap_space_spinbox.setMaximum(99999)
        self.cleaner.max_gap_space_spinbox.setValue(250)

        self.cleaner.max_gap_space_horizontal_layout.addWidget(self.cleaner.max_gap_space_spinbox)


        self.cleaner.main_vertical_layout.addWidget(self.cleaner.tuning_params_groupbox)

        self.cleaner.tuning_params_groupbox_3 = QGroupBox(self.cleaner.verticalLayoutWidget)
        self.cleaner.tuning_params_groupbox_3.setObjectName(u"tuning_params_groupbox_3")
        self.cleaner.tuning_params_groupbox_3.setMinimumSize(QSize(250, 200))
        self.cleaner.tuning_params_groupbox_3.setMaximumSize(QSize(250, 200))
        self.cleaner.verticalLayoutWidget_5 = QWidget(self.cleaner.tuning_params_groupbox_3)
        self.cleaner.verticalLayoutWidget_5.setObjectName(u"verticalLayoutWidget_5")
        self.cleaner.verticalLayoutWidget_5.setGeometry(QRect(9, 29, 233, 161))
        self.cleaner.verticalLayout = QVBoxLayout(self.cleaner.verticalLayoutWidget_5)
        self.cleaner.verticalLayout.setObjectName(u"verticalLayout")
        self.cleaner.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.cleaner.possible_joins_table = QTableView(self.cleaner.verticalLayoutWidget_5)
        self.cleaner.possible_joins_table.setObjectName(u"possible_joins_table")
        self.cleaner.possible_joins_table.setMinimumSize(QSize(230, 120))
        self.cleaner.possible_joins_table.setMaximumSize(QSize(230, 120))
        self.cleaner.possible_joins_table.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.cleaner.possible_joins_table.setAutoFillBackground(False)
        self.cleaner.possible_joins_table.setStyleSheet(u"QTableView {\n"
"    background-color: transparent;\n"
"    gridline-color: transparent;\n"
"}\n"
"QTableView::item {\n"
"    background-color: transparent;\n"
"    border: none;\n"
"    border-bottom: 1px solid #555;\n"
"}\n"
"QHeaderView {\n"
"    background-color: transparent;\n"
"}\n"
"QHeaderView::section {\n"
"    background-color: transparent;\n"
"    border: none;\n"
"    border-bottom: 1px solid #555;\n"
"    padding: 2px 4px;\n"
"}\n"
"QTableCornerButton::section {\n"
"    background-color: transparent;\n"
"}")
        self.cleaner.possible_joins_table.setFrameShape(QFrame.Shape.NoFrame)
        self.cleaner.possible_joins_table.setFrameShadow(QFrame.Shadow.Plain)
        self.cleaner.possible_joins_table.setLineWidth(1)
        self.cleaner.possible_joins_table.setEditTriggers(QAbstractItemView.EditTrigger.AnyKeyPressed|QAbstractItemView.EditTrigger.EditKeyPressed|QAbstractItemView.EditTrigger.SelectedClicked)
        self.cleaner.possible_joins_table.setProperty("showDropIndicator", False)
        self.cleaner.possible_joins_table.setDragDropOverwriteMode(False)
        self.cleaner.possible_joins_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.cleaner.possible_joins_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.cleaner.possible_joins_table.setGridStyle(Qt.PenStyle.SolidLine)
        self.cleaner.possible_joins_table.setSortingEnabled(False)
        self.cleaner.possible_joins_table.setWordWrap(False)
        self.cleaner.possible_joins_table.setCornerButtonEnabled(False)
        self.cleaner.possible_joins_table.horizontalHeader().setCascadingSectionResizes(True)
        self.cleaner.possible_joins_table.horizontalHeader().setMinimumSectionSize(85)
        self.cleaner.possible_joins_table.horizontalHeader().setDefaultSectionSize(85)
        self.cleaner.possible_joins_table.horizontalHeader().setHighlightSections(True)
        self.cleaner.possible_joins_table.horizontalHeader().setStretchLastSection(True)
        self.cleaner.possible_joins_table.verticalHeader().setVisible(False)
        self.cleaner.possible_joins_table.verticalHeader().setMinimumSectionSize(20)
        self.cleaner.possible_joins_table.verticalHeader().setDefaultSectionSize(20)
        self.cleaner.possible_joins_table.verticalHeader().setHighlightSections(False)

        self.cleaner.verticalLayout.addWidget(self.cleaner.possible_joins_table, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.cleaner.coverage_layout = QHBoxLayout()
        self.cleaner.coverage_layout.setObjectName(u"coverage_layout")
        self.cleaner.coverage_label = QLabel(self.cleaner.verticalLayoutWidget_5)
        self.cleaner.coverage_label.setObjectName(u"coverage_label")
        self.cleaner.coverage_label.setMinimumSize(QSize(65, 25))
        self.cleaner.coverage_label.setMaximumSize(QSize(65, 25))

        self.cleaner.coverage_layout.addWidget(self.cleaner.coverage_label)

        self.cleaner.coverage_percent = QLabel(self.cleaner.verticalLayoutWidget_5)
        self.cleaner.coverage_percent.setObjectName(u"coverage_percent")
        self.cleaner.coverage_percent.setMinimumSize(QSize(40, 25))
        self.cleaner.coverage_percent.setMaximumSize(QSize(40, 25))
        self.cleaner.coverage_percent.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.cleaner.coverage_layout.addWidget(self.cleaner.coverage_percent)

        self.cleaner.increase_label = QLabel(self.cleaner.verticalLayoutWidget_5)
        self.cleaner.increase_label.setObjectName(u"increase_label")
        self.cleaner.increase_label.setMinimumSize(QSize(60, 25))
        self.cleaner.increase_label.setMaximumSize(QSize(50, 25))
        self.cleaner.increase_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.cleaner.coverage_layout.addWidget(self.cleaner.increase_label)

        self.cleaner.increase_percent_label = QLabel(self.cleaner.verticalLayoutWidget_5)
        self.cleaner.increase_percent_label.setObjectName(u"increase_percent_label")
        self.cleaner.increase_percent_label.setMinimumSize(QSize(40, 25))
        self.cleaner.increase_percent_label.setMaximumSize(QSize(40, 25))
        self.cleaner.increase_percent_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.cleaner.coverage_layout.addWidget(self.cleaner.increase_percent_label)


        self.cleaner.verticalLayout.addLayout(self.cleaner.coverage_layout)


        self.cleaner.main_vertical_layout.addWidget(self.cleaner.tuning_params_groupbox_3)

        self.cleaner.tuning_params_groupbox_2 = QGroupBox(self.cleaner.verticalLayoutWidget)
        self.cleaner.tuning_params_groupbox_2.setObjectName(u"tuning_params_groupbox_2")
        self.cleaner.tuning_params_groupbox_2.setMinimumSize(QSize(250, 60))
        self.cleaner.tuning_params_groupbox_2.setMaximumSize(QSize(250, 60))
        self.cleaner.horizontalLayoutWidget = QWidget(self.cleaner.tuning_params_groupbox_2)
        self.cleaner.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.cleaner.horizontalLayoutWidget.setGeometry(QRect(10, 20, 231, 31))
        self.cleaner.control_buttons_layout = QHBoxLayout(self.cleaner.horizontalLayoutWidget)
        self.cleaner.control_buttons_layout.setObjectName(u"control_buttons_layout")
        self.cleaner.control_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.cleaner.reset_btn = QPushButton(self.cleaner.horizontalLayoutWidget)
        self.cleaner.reset_btn.setObjectName(u"reset_btn")
        self.cleaner.reset_btn.setEnabled(False)
        self.cleaner.reset_btn.setMinimumSize(QSize(70, 25))
        self.cleaner.reset_btn.setMaximumSize(QSize(70, 25))

        self.cleaner.control_buttons_layout.addWidget(self.cleaner.reset_btn)

        self.cleaner.revert1_btn = QPushButton(self.cleaner.horizontalLayoutWidget)
        self.cleaner.revert1_btn.setObjectName(u"revert1_btn")
        self.cleaner.revert1_btn.setEnabled(False)
        self.cleaner.revert1_btn.setMinimumSize(QSize(70, 25))
        self.cleaner.revert1_btn.setMaximumSize(QSize(70, 25))

        self.cleaner.control_buttons_layout.addWidget(self.cleaner.revert1_btn)

        self.cleaner.save_btn = QPushButton(self.cleaner.horizontalLayoutWidget)
        self.cleaner.save_btn.setObjectName(u"save_btn")
        self.cleaner.save_btn.setEnabled(False)
        self.cleaner.save_btn.setMinimumSize(QSize(70, 25))
        self.cleaner.save_btn.setMaximumSize(QSize(70, 25))

        self.cleaner.control_buttons_layout.addWidget(self.cleaner.save_btn)


        self.cleaner.main_vertical_layout.addWidget(self.cleaner.tuning_params_groupbox_2)


        self.cleaner.mainLayout.addLayout(self.cleaner.main_vertical_layout)

    # setupUi
        self.cleaner.setWindowTitle(QCoreApplication.translate("self", u"octron_gui", None))
        self.cleaner.octron_cleaner_ui_logo.setText("")
        self.cleaner.source_trace_groupbox.setTitle(QCoreApplication.translate("self", u"Source tracking layer", None))
        self.cleaner.tracks_label.setText(QCoreApplication.translate("self", u"# Tracks:", None))
        self.cleaner.total_no_tracks_label.setText(QCoreApplication.translate("self", u"0", None))
        self.cleaner.showall_layers_pushButton.setText(QCoreApplication.translate("self", u"Show all", None))
        self.cleaner.tuning_params_groupbox.setTitle(QCoreApplication.translate("self", u"Tuning", None))
        self.cleaner.max_gap_time_label.setText(QCoreApplication.translate("self", u"Max. gap in time (frames)", None))
        self.cleaner.max_gap_time_spinbox.setSuffix("")
        self.cleaner.max_gap_space_label.setText(QCoreApplication.translate("self", u"Max. gap in space (px)", None))
        self.cleaner.max_gap_space_spinbox.setSuffix("")
        self.cleaner.tuning_params_groupbox_3.setTitle(QCoreApplication.translate("self", u"Possible tracking joins", None))
        self.cleaner.coverage_label.setText(QCoreApplication.translate("self", u"Coverage:", None))
        self.cleaner.coverage_percent.setText(QCoreApplication.translate("self", u"%", None))
        self.cleaner.increase_label.setText(QCoreApplication.translate("self", u"Increase:", None))
        self.cleaner.increase_percent_label.setText(QCoreApplication.translate("self", u"%", None))
        self.cleaner.tuning_params_groupbox_2.setTitle("")
        self.cleaner.reset_btn.setText(QCoreApplication.translate("self", u"\u29bf Reset", None))
        self.cleaner.revert1_btn.setText(QCoreApplication.translate("self", u"\u2190 Revert", None))
        self.cleaner.save_btn.setText(QCoreApplication.translate("self", u"\u21f2 Save", None))
    # retranslateUi
