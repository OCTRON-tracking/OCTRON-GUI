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
            self.cleaner.setObjectName("self")
        self.cleaner.setEnabled(True)
        self.cleaner.resize(270, 600)
        self.cleaner.setMinimumSize(QSize(270, 300))
        self.cleaner.setMaximumSize(QSize(270, 600))
        self.cleaner.setCursor(QCursor(Qt.ArrowCursor))
        self.cleaner.setWindowOpacity(1.000000000000000)
        self.cleaner.verticalLayoutWidget = QWidget(self)
        self.cleaner.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.cleaner.verticalLayoutWidget.setGeometry(QRect(10, 10, 254, 581))
        self.cleaner.mainLayout = QVBoxLayout(self.cleaner.verticalLayoutWidget)
        self.cleaner.mainLayout.setSpacing(20)
        self.cleaner.mainLayout.setObjectName("mainLayout")
        self.cleaner.mainLayout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.cleaner.mainLayout.setContentsMargins(0, 10, 0, 0)
        self.cleaner.octron_cleaner_ui_logo = QLabel(self.cleaner.verticalLayoutWidget)
        self.cleaner.octron_cleaner_ui_logo.setObjectName("octron_cleaner_ui_logo")
        self.cleaner.octron_cleaner_ui_logo.setEnabled(True)
        self.cleaner.octron_cleaner_ui_logo.setMinimumSize(QSize(250, 70))
        self.cleaner.octron_cleaner_ui_logo.setMaximumSize(QSize(250, 70))
        self.cleaner.octron_cleaner_ui_logo.setBaseSize(QSize(0, 0))
        self.cleaner.octron_cleaner_ui_logo.setLineWidth(0)
        self.cleaner.octron_cleaner_ui_logo.setPixmap(QPixmap(f"{base_path}/qt_gui/octron_prediction_cleaner.svg"))
        self.cleaner.octron_cleaner_ui_logo.setScaledContents(False)
        self.cleaner.octron_cleaner_ui_logo.setAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.cleaner.mainLayout.addWidget(self.cleaner.octron_cleaner_ui_logo)

        self.cleaner.main_vertical_layout = QVBoxLayout()
        self.cleaner.main_vertical_layout.setObjectName("main_vertical_layout")
        self.cleaner.source_trace_groupbox = QGroupBox(self.cleaner.verticalLayoutWidget)
        self.cleaner.source_trace_groupbox.setObjectName("source_trace_groupbox")
        self.cleaner.source_trace_groupbox.setMinimumSize(QSize(250, 70))
        self.cleaner.source_trace_groupbox.setMaximumSize(QSize(250, 70))
        self.cleaner.source_trace_list = QComboBox(self.cleaner.source_trace_groupbox)
        self.cleaner.source_trace_list.setObjectName("source_trace_list")
        self.cleaner.source_trace_list.setGeometry(QRect(10, 30, 230, 25))
        self.cleaner.source_trace_list.setMinimumSize(QSize(230, 25))
        self.cleaner.source_trace_list.setMaximumSize(QSize(230, 25))
        self.cleaner.source_trace_list.setMaxVisibleItems(15)

        self.cleaner.main_vertical_layout.addWidget(self.cleaner.source_trace_groupbox)

        self.cleaner.tuning_params_groupbox = QGroupBox(self.cleaner.verticalLayoutWidget)
        self.cleaner.tuning_params_groupbox.setObjectName("tuning_params_groupbox")
        self.cleaner.tuning_params_groupbox.setMinimumSize(QSize(250, 110))
        self.cleaner.tuning_params_groupbox.setMaximumSize(QSize(250, 110))
        self.cleaner.verticalLayoutWidget_2 = QWidget(self.cleaner.tuning_params_groupbox)
        self.cleaner.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.cleaner.verticalLayoutWidget_2.setGeometry(QRect(20, 30, 92, 70))
        self.cleaner.max_gap_time_layout = QVBoxLayout(self.cleaner.verticalLayoutWidget_2)
        self.cleaner.max_gap_time_layout.setObjectName("max_gap_time_layout")
        self.cleaner.max_gap_time_layout.setContentsMargins(0, 0, 0, 0)
        self.cleaner.max_gap_time_label = QLabel(self.cleaner.verticalLayoutWidget_2)
        self.cleaner.max_gap_time_label.setObjectName("max_gap_time_label")
        self.cleaner.max_gap_time_label.setMinimumSize(QSize(90, 35))
        self.cleaner.max_gap_time_label.setMaximumSize(QSize(90, 35))
        self.cleaner.max_gap_time_label.setWordWrap(True)

        self.cleaner.max_gap_time_layout.addWidget(self.cleaner.max_gap_time_label)

        self.cleaner.max_gap_time_spinbox = QSpinBox(self.cleaner.verticalLayoutWidget_2)
        self.cleaner.max_gap_time_spinbox.setObjectName("max_gap_time_spinbox")
        self.cleaner.max_gap_time_spinbox.setMinimumSize(QSize(90, 25))
        self.cleaner.max_gap_time_spinbox.setMaximumSize(QSize(90, 25))
        self.cleaner.max_gap_time_spinbox.setMaximum(99999)
        self.cleaner.max_gap_time_spinbox.setValue(10)

        self.cleaner.max_gap_time_layout.addWidget(self.cleaner.max_gap_time_spinbox)

        self.cleaner.verticalLayoutWidget_3 = QWidget(self.cleaner.tuning_params_groupbox)
        self.cleaner.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.cleaner.verticalLayoutWidget_3.setGeometry(QRect(140, 30, 92, 70))
        self.cleaner.max_gap_dist_layout = QVBoxLayout(self.cleaner.verticalLayoutWidget_3)
        self.cleaner.max_gap_dist_layout.setObjectName("max_gap_dist_layout")
        self.cleaner.max_gap_dist_layout.setContentsMargins(0, 0, 0, 0)
        self.cleaner.max_gap_space_label = QLabel(self.cleaner.verticalLayoutWidget_3)
        self.cleaner.max_gap_space_label.setObjectName("max_gap_space_label")
        self.cleaner.max_gap_space_label.setMinimumSize(QSize(90, 35))
        self.cleaner.max_gap_space_label.setMaximumSize(QSize(90, 35))
        self.cleaner.max_gap_space_label.setWordWrap(True)

        self.cleaner.max_gap_dist_layout.addWidget(self.cleaner.max_gap_space_label)

        self.cleaner.max_gap_space_spinbox = QSpinBox(self.cleaner.verticalLayoutWidget_3)
        self.cleaner.max_gap_space_spinbox.setObjectName("max_gap_space_spinbox")
        self.cleaner.max_gap_space_spinbox.setMinimumSize(QSize(90, 25))
        self.cleaner.max_gap_space_spinbox.setMaximumSize(QSize(90, 25))
        self.cleaner.max_gap_space_spinbox.setMaximum(99999)
        self.cleaner.max_gap_space_spinbox.setValue(250)

        self.cleaner.max_gap_dist_layout.addWidget(self.cleaner.max_gap_space_spinbox)


        self.cleaner.main_vertical_layout.addWidget(self.cleaner.tuning_params_groupbox)

        self.cleaner.alignment_target_groupbox = QGroupBox(self.cleaner.verticalLayoutWidget)
        self.cleaner.alignment_target_groupbox.setObjectName("alignment_target_groupbox")
        self.cleaner.alignment_target_groupbox.setMinimumSize(QSize(250, 300))
        self.cleaner.alignment_target_groupbox.setMaximumSize(QSize(250, 280))
        self.cleaner.horizontalLayoutWidget_3 = QWidget(self.cleaner.alignment_target_groupbox)
        self.cleaner.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.cleaner.horizontalLayoutWidget_3.setGeometry(QRect(10, 200, 231, 31))
        self.cleaner.coverage_layout = QHBoxLayout(self.cleaner.horizontalLayoutWidget_3)
        self.cleaner.coverage_layout.setObjectName("coverage_layout")
        self.cleaner.coverage_layout.setContentsMargins(0, 0, 0, 0)
        self.cleaner.coverage_after_join_label = QLabel(self.cleaner.horizontalLayoutWidget_3)
        self.cleaner.coverage_after_join_label.setObjectName("coverage_after_join_label")
        self.cleaner.coverage_after_join_label.setMinimumSize(QSize(140, 25))
        self.cleaner.coverage_after_join_label.setMaximumSize(QSize(140, 25))

        self.cleaner.coverage_layout.addWidget(self.cleaner.coverage_after_join_label)

        self.cleaner.coverage_percent_label = QLabel(self.cleaner.horizontalLayoutWidget_3)
        self.cleaner.coverage_percent_label.setObjectName("coverage_percent_label")
        self.cleaner.coverage_percent_label.setMinimumSize(QSize(60, 25))
        self.cleaner.coverage_percent_label.setMaximumSize(QSize(60, 25))
        self.cleaner.coverage_percent_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.cleaner.coverage_layout.addWidget(self.cleaner.coverage_percent_label)

        self.cleaner.horizontalLayoutWidget_4 = QWidget(self.cleaner.alignment_target_groupbox)
        self.cleaner.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.cleaner.horizontalLayoutWidget_4.setGeometry(QRect(10, 240, 231, 25))
        self.cleaner.reset_revert_bake_layout = QHBoxLayout(self.cleaner.horizontalLayoutWidget_4)
        self.cleaner.reset_revert_bake_layout.setObjectName("reset_revert_bake_layout")
        self.cleaner.reset_revert_bake_layout.setContentsMargins(0, 0, 0, 0)
        self.cleaner.reset_btn = QPushButton(self.cleaner.horizontalLayoutWidget_4)
        self.cleaner.reset_btn.setObjectName("reset_btn")
        self.cleaner.reset_btn.setMinimumSize(QSize(65, 25))
        self.cleaner.reset_btn.setMaximumSize(QSize(65, 25))

        self.cleaner.reset_revert_bake_layout.addWidget(self.cleaner.reset_btn)

        self.cleaner.revert_btn = QPushButton(self.cleaner.horizontalLayoutWidget_4)
        self.cleaner.revert_btn.setObjectName("revert_btn")
        self.cleaner.revert_btn.setMinimumSize(QSize(80, 25))
        self.cleaner.revert_btn.setMaximumSize(QSize(80, 25))

        self.cleaner.reset_revert_bake_layout.addWidget(self.cleaner.revert_btn)

        self.cleaner.bake_btn = QPushButton(self.cleaner.horizontalLayoutWidget_4)
        self.cleaner.bake_btn.setObjectName("bake_btn")
        self.cleaner.bake_btn.setMinimumSize(QSize(56, 25))
        self.cleaner.bake_btn.setMaximumSize(QSize(56, 25))

        self.cleaner.reset_revert_bake_layout.addWidget(self.cleaner.bake_btn)

        self.cleaner.alignment_target_table = QTableView(self.cleaner.alignment_target_groupbox)
        self.cleaner.alignment_target_table.setObjectName("alignment_target_table")
        self.cleaner.alignment_target_table.setGeometry(QRect(10, 40, 230, 140))
        self.cleaner.alignment_target_table.setMinimumSize(QSize(230, 140))
        self.cleaner.alignment_target_table.setMaximumSize(QSize(230, 140))
        self.cleaner.alignment_target_table.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.cleaner.alignment_target_table.setAutoFillBackground(False)
        self.cleaner.alignment_target_table.setStyleSheet("QTableView {\n"
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
        self.cleaner.alignment_target_table.setFrameShape(QFrame.Shape.NoFrame)
        self.cleaner.alignment_target_table.setFrameShadow(QFrame.Shadow.Plain)
        self.cleaner.alignment_target_table.setLineWidth(1)
        self.cleaner.alignment_target_table.setEditTriggers(QAbstractItemView.EditTrigger.AnyKeyPressed|QAbstractItemView.EditTrigger.EditKeyPressed|QAbstractItemView.EditTrigger.SelectedClicked)
        self.cleaner.alignment_target_table.setProperty("showDropIndicator", False)
        self.cleaner.alignment_target_table.setDragDropOverwriteMode(False)
        self.cleaner.alignment_target_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.cleaner.alignment_target_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.cleaner.alignment_target_table.setGridStyle(Qt.PenStyle.SolidLine)
        self.cleaner.alignment_target_table.setSortingEnabled(False)
        self.cleaner.alignment_target_table.setWordWrap(False)
        self.cleaner.alignment_target_table.setCornerButtonEnabled(False)
        self.cleaner.alignment_target_table.horizontalHeader().setCascadingSectionResizes(True)
        self.cleaner.alignment_target_table.horizontalHeader().setMinimumSectionSize(85)
        self.cleaner.alignment_target_table.horizontalHeader().setDefaultSectionSize(85)
        self.cleaner.alignment_target_table.horizontalHeader().setHighlightSections(True)
        self.cleaner.alignment_target_table.horizontalHeader().setStretchLastSection(True)
        self.cleaner.alignment_target_table.verticalHeader().setVisible(False)
        self.cleaner.alignment_target_table.verticalHeader().setMinimumSectionSize(20)
        self.cleaner.alignment_target_table.verticalHeader().setDefaultSectionSize(20)
        self.cleaner.alignment_target_table.verticalHeader().setHighlightSections(False)

        self.cleaner.main_vertical_layout.addWidget(self.cleaner.alignment_target_groupbox)


        self.cleaner.mainLayout.addLayout(self.cleaner.main_vertical_layout)

    # setupUi
        self.cleaner.setWindowTitle(QCoreApplication.translate("self", "octron_gui", None))
        self.cleaner.octron_cleaner_ui_logo.setText("")
        self.cleaner.source_trace_groupbox.setTitle(QCoreApplication.translate("self", "Source trace", None))
        self.cleaner.tuning_params_groupbox.setTitle(QCoreApplication.translate("self", "Tuning parameters", None))
        self.cleaner.max_gap_time_label.setText(QCoreApplication.translate("self", "Maximum gap in time", None))
        self.cleaner.max_gap_time_spinbox.setSuffix(QCoreApplication.translate("self", " frames", None))
        self.cleaner.max_gap_space_label.setText(QCoreApplication.translate("self", "Maximum gap in space", None))
        self.cleaner.max_gap_space_spinbox.setSuffix(QCoreApplication.translate("self", " px", None))
        self.cleaner.alignment_target_groupbox.setTitle(QCoreApplication.translate("self", "Alignment targets", None))
        self.cleaner.coverage_after_join_label.setText(QCoreApplication.translate("self", "Coverage after join:", None))
        self.cleaner.coverage_percent_label.setText(QCoreApplication.translate("self", " %", None))
        self.cleaner.reset_btn.setText(QCoreApplication.translate("self", "\u21a4 Reset", None))
        self.cleaner.revert_btn.setText(QCoreApplication.translate("self", "Revert step", None))
        self.cleaner.bake_btn.setText(QCoreApplication.translate("self", "\u2714 Bake!", None))
    # retranslateUi
