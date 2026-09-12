# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'octron_prediction_cleaner.ui'
##
## Created by: Qt User Interface Compiler version 5.15.18
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from qtpy.QtCore import *  # type: ignore
from qtpy.QtGui import *  # type: ignore
from qtpy.QtWidgets import *  # type: ignore


class Ui_octron_prediction_cleaner(object):
    def setupUi(self, base_path):
        if not self.cleaner.objectName():
            self.cleaner.setObjectName(u"self")
        self.cleaner.setEnabled(True)
        self.cleaner.resize(270, 600)
        self.cleaner.setMinimumSize(QSize(270, 300))
        self.cleaner.setMaximumSize(QSize(270, 600))
        self.cleaner.setCursor(QCursor(Qt.ArrowCursor))
        self.cleaner.setWindowOpacity(1.000000000000000)
        self.cleaner.verticalLayoutWidget = QWidget(self)
        self.cleaner.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.cleaner.verticalLayoutWidget.setGeometry(QRect(10, 10, 254, 581))
        self.cleaner.mainLayout = QVBoxLayout(self.cleaner.verticalLayoutWidget)
        self.cleaner.mainLayout.setSpacing(20)
        self.cleaner.mainLayout.setObjectName(u"mainLayout")
        self.cleaner.mainLayout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.cleaner.mainLayout.setContentsMargins(0, 10, 0, 0)
        self.cleaner.octron_cleaner_ui_logo = QLabel(self.cleaner.verticalLayoutWidget)
        self.cleaner.octron_cleaner_ui_logo.setObjectName(u"octron_cleaner_ui_logo")
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
        self.cleaner.main_vertical_layout.setObjectName(u"main_vertical_layout")
        self.cleaner.source_trace_groupbox = QGroupBox(self.cleaner.verticalLayoutWidget)
        self.cleaner.source_trace_groupbox.setObjectName(u"source_trace_groupbox")
        self.cleaner.source_trace_groupbox.setMinimumSize(QSize(250, 70))
        self.cleaner.source_trace_groupbox.setMaximumSize(QSize(250, 70))
        self.cleaner.source_trace_list = QComboBox(self.cleaner.source_trace_groupbox)
        self.cleaner.source_trace_list.setObjectName(u"source_trace_list")
        self.cleaner.source_trace_list.setGeometry(QRect(10, 30, 230, 25))
        self.cleaner.source_trace_list.setMinimumSize(QSize(230, 25))
        self.cleaner.source_trace_list.setMaximumSize(QSize(230, 25))
        self.cleaner.source_trace_list.setMaxVisibleItems(15)

        self.cleaner.main_vertical_layout.addWidget(self.cleaner.source_trace_groupbox)

        self.cleaner.tuning_params_groupbox = QGroupBox(self.cleaner.verticalLayoutWidget)
        self.cleaner.tuning_params_groupbox.setObjectName(u"tuning_params_groupbox")
        self.cleaner.tuning_params_groupbox.setMinimumSize(QSize(250, 110))
        self.cleaner.tuning_params_groupbox.setMaximumSize(QSize(250, 110))
        self.cleaner.verticalLayoutWidget_2 = QWidget(self.cleaner.tuning_params_groupbox)
        self.cleaner.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.cleaner.verticalLayoutWidget_2.setGeometry(QRect(20, 30, 92, 70))
        self.cleaner.max_gap_time_layout = QVBoxLayout(self.cleaner.verticalLayoutWidget_2)
        self.cleaner.max_gap_time_layout.setObjectName(u"max_gap_time_layout")
        self.cleaner.max_gap_time_layout.setContentsMargins(0, 0, 0, 0)
        self.cleaner.max_gap_time_label = QLabel(self.cleaner.verticalLayoutWidget_2)
        self.cleaner.max_gap_time_label.setObjectName(u"max_gap_time_label")
        self.cleaner.max_gap_time_label.setMinimumSize(QSize(90, 35))
        self.cleaner.max_gap_time_label.setMaximumSize(QSize(90, 35))
        self.cleaner.max_gap_time_label.setWordWrap(True)

        self.cleaner.max_gap_time_layout.addWidget(self.cleaner.max_gap_time_label)

        self.cleaner.max_gap_time_spinbox = QSpinBox(self.cleaner.verticalLayoutWidget_2)
        self.cleaner.max_gap_time_spinbox.setObjectName(u"max_gap_time_spinbox")
        self.cleaner.max_gap_time_spinbox.setMinimumSize(QSize(90, 25))
        self.cleaner.max_gap_time_spinbox.setMaximumSize(QSize(90, 25))
        self.cleaner.max_gap_time_spinbox.setMaximum(99999)
        self.cleaner.max_gap_time_spinbox.setValue(10)

        self.cleaner.max_gap_time_layout.addWidget(self.cleaner.max_gap_time_spinbox)

        self.cleaner.verticalLayoutWidget_3 = QWidget(self.cleaner.tuning_params_groupbox)
        self.cleaner.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.cleaner.verticalLayoutWidget_3.setGeometry(QRect(140, 30, 92, 70))
        self.cleaner.max_gap_dist_layout = QVBoxLayout(self.cleaner.verticalLayoutWidget_3)
        self.cleaner.max_gap_dist_layout.setObjectName(u"max_gap_dist_layout")
        self.cleaner.max_gap_dist_layout.setContentsMargins(0, 0, 0, 0)
        self.cleaner.max_gap_space_label = QLabel(self.cleaner.verticalLayoutWidget_3)
        self.cleaner.max_gap_space_label.setObjectName(u"max_gap_space_label")
        self.cleaner.max_gap_space_label.setMinimumSize(QSize(90, 35))
        self.cleaner.max_gap_space_label.setMaximumSize(QSize(90, 35))
        self.cleaner.max_gap_space_label.setWordWrap(True)

        self.cleaner.max_gap_dist_layout.addWidget(self.cleaner.max_gap_space_label)

        self.cleaner.max_gap_space_spinbox = QSpinBox(self.cleaner.verticalLayoutWidget_3)
        self.cleaner.max_gap_space_spinbox.setObjectName(u"max_gap_space_spinbox")
        self.cleaner.max_gap_space_spinbox.setMinimumSize(QSize(90, 25))
        self.cleaner.max_gap_space_spinbox.setMaximumSize(QSize(90, 25))
        self.cleaner.max_gap_space_spinbox.setMaximum(99999)
        self.cleaner.max_gap_space_spinbox.setValue(250)

        self.cleaner.max_gap_dist_layout.addWidget(self.cleaner.max_gap_space_spinbox)


        self.cleaner.main_vertical_layout.addWidget(self.cleaner.tuning_params_groupbox)

        self.cleaner.alignment_target_groupbox = QGroupBox(self.cleaner.verticalLayoutWidget)
        self.cleaner.alignment_target_groupbox.setObjectName(u"alignment_target_groupbox")
        self.cleaner.alignment_target_groupbox.setMinimumSize(QSize(250, 300))
        self.cleaner.alignment_target_groupbox.setMaximumSize(QSize(250, 280))
        self.cleaner.horizontalLayoutWidget_3 = QWidget(self.cleaner.alignment_target_groupbox)
        self.cleaner.horizontalLayoutWidget_3.setObjectName(u"horizontalLayoutWidget_3")
        self.cleaner.horizontalLayoutWidget_3.setGeometry(QRect(10, 200, 231, 31))
        self.cleaner.coverage_layout = QHBoxLayout(self.cleaner.horizontalLayoutWidget_3)
        self.cleaner.coverage_layout.setObjectName(u"coverage_layout")
        self.cleaner.coverage_layout.setContentsMargins(0, 0, 0, 0)
        self.cleaner.coverage_after_join_label = QLabel(self.cleaner.horizontalLayoutWidget_3)
        self.cleaner.coverage_after_join_label.setObjectName(u"coverage_after_join_label")
        self.cleaner.coverage_after_join_label.setMinimumSize(QSize(140, 25))
        self.cleaner.coverage_after_join_label.setMaximumSize(QSize(140, 25))

        self.cleaner.coverage_layout.addWidget(self.cleaner.coverage_after_join_label)

        self.cleaner.coverage_percent_label = QLabel(self.cleaner.horizontalLayoutWidget_3)
        self.cleaner.coverage_percent_label.setObjectName(u"coverage_percent_label")
        self.cleaner.coverage_percent_label.setMinimumSize(QSize(60, 25))
        self.cleaner.coverage_percent_label.setMaximumSize(QSize(60, 25))
        self.cleaner.coverage_percent_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.cleaner.coverage_layout.addWidget(self.cleaner.coverage_percent_label)

        self.cleaner.horizontalLayoutWidget_4 = QWidget(self.cleaner.alignment_target_groupbox)
        self.cleaner.horizontalLayoutWidget_4.setObjectName(u"horizontalLayoutWidget_4")
        self.cleaner.horizontalLayoutWidget_4.setGeometry(QRect(10, 240, 231, 25))
        self.cleaner.reset_revert_bake_layout = QHBoxLayout(self.cleaner.horizontalLayoutWidget_4)
        self.cleaner.reset_revert_bake_layout.setObjectName(u"reset_revert_bake_layout")
        self.cleaner.reset_revert_bake_layout.setContentsMargins(0, 0, 0, 0)
        self.cleaner.reset_btn = QPushButton(self.cleaner.horizontalLayoutWidget_4)
        self.cleaner.reset_btn.setObjectName(u"reset_btn")
        self.cleaner.reset_btn.setMinimumSize(QSize(65, 25))
        self.cleaner.reset_btn.setMaximumSize(QSize(65, 25))

        self.cleaner.reset_revert_bake_layout.addWidget(self.cleaner.reset_btn)

        self.cleaner.revert_btn = QPushButton(self.cleaner.horizontalLayoutWidget_4)
        self.cleaner.revert_btn.setObjectName(u"revert_btn")
        self.cleaner.revert_btn.setMinimumSize(QSize(80, 25))
        self.cleaner.revert_btn.setMaximumSize(QSize(80, 25))

        self.cleaner.reset_revert_bake_layout.addWidget(self.cleaner.revert_btn)

        self.cleaner.bake_btn = QPushButton(self.cleaner.horizontalLayoutWidget_4)
        self.cleaner.bake_btn.setObjectName(u"bake_btn")
        self.cleaner.bake_btn.setMinimumSize(QSize(56, 25))
        self.cleaner.bake_btn.setMaximumSize(QSize(56, 25))

        self.cleaner.reset_revert_bake_layout.addWidget(self.cleaner.bake_btn)

        self.cleaner.alignment_target_table = QTableView(self.cleaner.alignment_target_groupbox)
        self.cleaner.alignment_target_table.setObjectName(u"alignment_target_table")
        self.cleaner.alignment_target_table.setGeometry(QRect(10, 40, 230, 140))
        self.cleaner.alignment_target_table.setMinimumSize(QSize(230, 140))
        self.cleaner.alignment_target_table.setMaximumSize(QSize(230, 140))
        self.cleaner.alignment_target_table.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.cleaner.alignment_target_table.setAutoFillBackground(False)
        self.cleaner.alignment_target_table.setStyleSheet(u"QTableView {\n"
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
        self.cleaner.setWindowTitle(QCoreApplication.translate("self", u"octron_gui", None))
        self.cleaner.octron_cleaner_ui_logo.setText("")
        self.cleaner.source_trace_groupbox.setTitle(QCoreApplication.translate("self", u"Source trace", None))
        self.cleaner.tuning_params_groupbox.setTitle(QCoreApplication.translate("self", u"Tuning parameters", None))
        self.cleaner.max_gap_time_label.setText(QCoreApplication.translate("self", u"Maximum gap in time", None))
        self.cleaner.max_gap_time_spinbox.setSuffix(QCoreApplication.translate("self", u" frames", None))
        self.cleaner.max_gap_space_label.setText(QCoreApplication.translate("self", u"Maximum gap in space", None))
        self.cleaner.max_gap_space_spinbox.setSuffix(QCoreApplication.translate("self", u" px", None))
        self.cleaner.alignment_target_groupbox.setTitle(QCoreApplication.translate("self", u"Alignment targets", None))
        self.cleaner.coverage_after_join_label.setText(QCoreApplication.translate("self", u"Coverage after join:", None))
        self.cleaner.coverage_percent_label.setText(QCoreApplication.translate("self", u" %", None))
        self.cleaner.reset_btn.setText(QCoreApplication.translate("self", u"\u21a4 Reset", None))
        self.cleaner.revert_btn.setText(QCoreApplication.translate("self", u"Revert step", None))
        self.cleaner.bake_btn.setText(QCoreApplication.translate("self", u"\u2714 Bake!", None))
    # retranslateUi

