# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'octron_prediction_cleaner.ui'
##
## Created by: Qt User Interface Compiler version 5.15.18
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *  # type: ignore
from PySide2.QtGui import *  # type: ignore
from PySide2.QtWidgets import *  # type: ignore


class Ui_octron_prediction_cleaner(object):
    def setupUi(self, octron_prediction_cleaner):
        if not octron_prediction_cleaner.objectName():
            octron_prediction_cleaner.setObjectName(u"octron_prediction_cleaner")
        octron_prediction_cleaner.setEnabled(True)
        octron_prediction_cleaner.resize(270, 600)
        octron_prediction_cleaner.setMinimumSize(QSize(270, 300))
        octron_prediction_cleaner.setMaximumSize(QSize(270, 600))
        octron_prediction_cleaner.setCursor(QCursor(Qt.ArrowCursor))
        octron_prediction_cleaner.setWindowOpacity(1.000000000000000)
        self.verticalLayoutWidget = QWidget(octron_prediction_cleaner)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(10, 10, 254, 581))
        self.mainLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.mainLayout.setSpacing(20)
        self.mainLayout.setObjectName(u"mainLayout")
        self.mainLayout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.mainLayout.setContentsMargins(0, 10, 0, 0)
        self.octron_cleaner_ui_logo = QLabel(self.verticalLayoutWidget)
        self.octron_cleaner_ui_logo.setObjectName(u"octron_cleaner_ui_logo")
        self.octron_cleaner_ui_logo.setEnabled(True)
        self.octron_cleaner_ui_logo.setMinimumSize(QSize(250, 70))
        self.octron_cleaner_ui_logo.setMaximumSize(QSize(250, 70))
        self.octron_cleaner_ui_logo.setBaseSize(QSize(0, 0))
        self.octron_cleaner_ui_logo.setLineWidth(0)
        self.octron_cleaner_ui_logo.setPixmap(QPixmap(u"octron_prediction_cleaner.svg"))
        self.octron_cleaner_ui_logo.setScaledContents(False)
        self.octron_cleaner_ui_logo.setAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.mainLayout.addWidget(self.octron_cleaner_ui_logo)

        self.main_vertical_layout = QVBoxLayout()
        self.main_vertical_layout.setObjectName(u"main_vertical_layout")
        self.source_trace_groupbox = QGroupBox(self.verticalLayoutWidget)
        self.source_trace_groupbox.setObjectName(u"source_trace_groupbox")
        self.source_trace_groupbox.setMinimumSize(QSize(250, 70))
        self.source_trace_groupbox.setMaximumSize(QSize(250, 70))
        self.source_trace_list = QComboBox(self.source_trace_groupbox)
        self.source_trace_list.setObjectName(u"source_trace_list")
        self.source_trace_list.setGeometry(QRect(10, 30, 230, 25))
        self.source_trace_list.setMinimumSize(QSize(230, 25))
        self.source_trace_list.setMaximumSize(QSize(230, 25))
        self.source_trace_list.setMaxVisibleItems(15)

        self.main_vertical_layout.addWidget(self.source_trace_groupbox)

        self.tuning_params_groupbox = QGroupBox(self.verticalLayoutWidget)
        self.tuning_params_groupbox.setObjectName(u"tuning_params_groupbox")
        self.tuning_params_groupbox.setMinimumSize(QSize(250, 110))
        self.tuning_params_groupbox.setMaximumSize(QSize(250, 110))
        self.verticalLayoutWidget_2 = QWidget(self.tuning_params_groupbox)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(20, 30, 92, 70))
        self.max_gap_time_layout = QVBoxLayout(self.verticalLayoutWidget_2)
        self.max_gap_time_layout.setObjectName(u"max_gap_time_layout")
        self.max_gap_time_layout.setContentsMargins(0, 0, 0, 0)
        self.max_gap_time_label = QLabel(self.verticalLayoutWidget_2)
        self.max_gap_time_label.setObjectName(u"max_gap_time_label")
        self.max_gap_time_label.setMinimumSize(QSize(90, 35))
        self.max_gap_time_label.setMaximumSize(QSize(90, 35))
        self.max_gap_time_label.setWordWrap(True)

        self.max_gap_time_layout.addWidget(self.max_gap_time_label)

        self.max_gap_time_spinbox = QSpinBox(self.verticalLayoutWidget_2)
        self.max_gap_time_spinbox.setObjectName(u"max_gap_time_spinbox")
        self.max_gap_time_spinbox.setMinimumSize(QSize(90, 25))
        self.max_gap_time_spinbox.setMaximumSize(QSize(90, 25))
        self.max_gap_time_spinbox.setMaximum(99999)
        self.max_gap_time_spinbox.setValue(10)

        self.max_gap_time_layout.addWidget(self.max_gap_time_spinbox)

        self.verticalLayoutWidget_3 = QWidget(self.tuning_params_groupbox)
        self.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(140, 30, 92, 70))
        self.max_gap_dist_layout = QVBoxLayout(self.verticalLayoutWidget_3)
        self.max_gap_dist_layout.setObjectName(u"max_gap_dist_layout")
        self.max_gap_dist_layout.setContentsMargins(0, 0, 0, 0)
        self.max_gap_space_label = QLabel(self.verticalLayoutWidget_3)
        self.max_gap_space_label.setObjectName(u"max_gap_space_label")
        self.max_gap_space_label.setMinimumSize(QSize(90, 35))
        self.max_gap_space_label.setMaximumSize(QSize(90, 35))
        self.max_gap_space_label.setWordWrap(True)

        self.max_gap_dist_layout.addWidget(self.max_gap_space_label)

        self.max_gap_space_spinbox = QSpinBox(self.verticalLayoutWidget_3)
        self.max_gap_space_spinbox.setObjectName(u"max_gap_space_spinbox")
        self.max_gap_space_spinbox.setMinimumSize(QSize(90, 25))
        self.max_gap_space_spinbox.setMaximumSize(QSize(90, 25))
        self.max_gap_space_spinbox.setMaximum(99999)
        self.max_gap_space_spinbox.setValue(250)

        self.max_gap_dist_layout.addWidget(self.max_gap_space_spinbox)


        self.main_vertical_layout.addWidget(self.tuning_params_groupbox)

        self.alignment_target_groupbox = QGroupBox(self.verticalLayoutWidget)
        self.alignment_target_groupbox.setObjectName(u"alignment_target_groupbox")
        self.alignment_target_groupbox.setMinimumSize(QSize(250, 300))
        self.alignment_target_groupbox.setMaximumSize(QSize(250, 280))
        self.horizontalLayoutWidget_3 = QWidget(self.alignment_target_groupbox)
        self.horizontalLayoutWidget_3.setObjectName(u"horizontalLayoutWidget_3")
        self.horizontalLayoutWidget_3.setGeometry(QRect(10, 200, 231, 31))
        self.coverage_layout = QHBoxLayout(self.horizontalLayoutWidget_3)
        self.coverage_layout.setObjectName(u"coverage_layout")
        self.coverage_layout.setContentsMargins(0, 0, 0, 0)
        self.coverage_after_join_label = QLabel(self.horizontalLayoutWidget_3)
        self.coverage_after_join_label.setObjectName(u"coverage_after_join_label")
        self.coverage_after_join_label.setMinimumSize(QSize(140, 25))
        self.coverage_after_join_label.setMaximumSize(QSize(140, 25))

        self.coverage_layout.addWidget(self.coverage_after_join_label)

        self.coverage_percent_label = QLabel(self.horizontalLayoutWidget_3)
        self.coverage_percent_label.setObjectName(u"coverage_percent_label")
        self.coverage_percent_label.setMinimumSize(QSize(60, 25))
        self.coverage_percent_label.setMaximumSize(QSize(60, 25))
        self.coverage_percent_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.coverage_layout.addWidget(self.coverage_percent_label)

        self.horizontalLayoutWidget_4 = QWidget(self.alignment_target_groupbox)
        self.horizontalLayoutWidget_4.setObjectName(u"horizontalLayoutWidget_4")
        self.horizontalLayoutWidget_4.setGeometry(QRect(10, 240, 231, 25))
        self.reset_revert_bake_layout = QHBoxLayout(self.horizontalLayoutWidget_4)
        self.reset_revert_bake_layout.setObjectName(u"reset_revert_bake_layout")
        self.reset_revert_bake_layout.setContentsMargins(0, 0, 0, 0)
        self.reset_btn = QPushButton(self.horizontalLayoutWidget_4)
        self.reset_btn.setObjectName(u"reset_btn")
        self.reset_btn.setMinimumSize(QSize(65, 25))
        self.reset_btn.setMaximumSize(QSize(65, 25))

        self.reset_revert_bake_layout.addWidget(self.reset_btn)

        self.revert_btn = QPushButton(self.horizontalLayoutWidget_4)
        self.revert_btn.setObjectName(u"revert_btn")
        self.revert_btn.setMinimumSize(QSize(80, 25))
        self.revert_btn.setMaximumSize(QSize(80, 25))

        self.reset_revert_bake_layout.addWidget(self.revert_btn)

        self.bake_btn = QPushButton(self.horizontalLayoutWidget_4)
        self.bake_btn.setObjectName(u"bake_btn")
        self.bake_btn.setMinimumSize(QSize(56, 25))
        self.bake_btn.setMaximumSize(QSize(56, 25))

        self.reset_revert_bake_layout.addWidget(self.bake_btn)

        self.alignment_target_table = QTableView(self.alignment_target_groupbox)
        self.alignment_target_table.setObjectName(u"alignment_target_table")
        self.alignment_target_table.setGeometry(QRect(10, 40, 230, 140))
        self.alignment_target_table.setMinimumSize(QSize(230, 140))
        self.alignment_target_table.setMaximumSize(QSize(230, 140))
        self.alignment_target_table.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.alignment_target_table.setAutoFillBackground(False)
        self.alignment_target_table.setStyleSheet(u"QTableView {\n"
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
        self.alignment_target_table.setFrameShape(QFrame.Shape.NoFrame)
        self.alignment_target_table.setFrameShadow(QFrame.Shadow.Plain)
        self.alignment_target_table.setLineWidth(1)
        self.alignment_target_table.setEditTriggers(QAbstractItemView.EditTrigger.AnyKeyPressed|QAbstractItemView.EditTrigger.EditKeyPressed|QAbstractItemView.EditTrigger.SelectedClicked)
        self.alignment_target_table.setProperty("showDropIndicator", False)
        self.alignment_target_table.setDragDropOverwriteMode(False)
        self.alignment_target_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.alignment_target_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.alignment_target_table.setGridStyle(Qt.PenStyle.SolidLine)
        self.alignment_target_table.setSortingEnabled(False)
        self.alignment_target_table.setWordWrap(False)
        self.alignment_target_table.setCornerButtonEnabled(False)
        self.alignment_target_table.horizontalHeader().setCascadingSectionResizes(True)
        self.alignment_target_table.horizontalHeader().setMinimumSectionSize(85)
        self.alignment_target_table.horizontalHeader().setDefaultSectionSize(85)
        self.alignment_target_table.horizontalHeader().setHighlightSections(True)
        self.alignment_target_table.horizontalHeader().setStretchLastSection(True)
        self.alignment_target_table.verticalHeader().setVisible(False)
        self.alignment_target_table.verticalHeader().setMinimumSectionSize(20)
        self.alignment_target_table.verticalHeader().setDefaultSectionSize(20)
        self.alignment_target_table.verticalHeader().setHighlightSections(False)

        self.main_vertical_layout.addWidget(self.alignment_target_groupbox)


        self.mainLayout.addLayout(self.main_vertical_layout)


        self.retranslateUi(octron_prediction_cleaner)

        QMetaObject.connectSlotsByName(octron_prediction_cleaner)
    # setupUi

    def retranslateUi(self, octron_prediction_cleaner):
        octron_prediction_cleaner.setWindowTitle(QCoreApplication.translate("octron_prediction_cleaner", u"octron_gui", None))
        self.octron_cleaner_ui_logo.setText("")
        self.source_trace_groupbox.setTitle(QCoreApplication.translate("octron_prediction_cleaner", u"Source trace", None))
        self.tuning_params_groupbox.setTitle(QCoreApplication.translate("octron_prediction_cleaner", u"Tuning parameters", None))
        self.max_gap_time_label.setText(QCoreApplication.translate("octron_prediction_cleaner", u"Maximum gap in time", None))
        self.max_gap_time_spinbox.setSuffix(QCoreApplication.translate("octron_prediction_cleaner", u" frames", None))
        self.max_gap_space_label.setText(QCoreApplication.translate("octron_prediction_cleaner", u"Maximum gap in space", None))
        self.max_gap_space_spinbox.setSuffix(QCoreApplication.translate("octron_prediction_cleaner", u" px", None))
        self.alignment_target_groupbox.setTitle(QCoreApplication.translate("octron_prediction_cleaner", u"Alignment targets", None))
        self.coverage_after_join_label.setText(QCoreApplication.translate("octron_prediction_cleaner", u"Coverage after join:", None))
        self.coverage_percent_label.setText(QCoreApplication.translate("octron_prediction_cleaner", u" %", None))
        self.reset_btn.setText(QCoreApplication.translate("octron_prediction_cleaner", u"\u21a4 Reset", None))
        self.revert_btn.setText(QCoreApplication.translate("octron_prediction_cleaner", u"Revert step", None))
        self.bake_btn.setText(QCoreApplication.translate("octron_prediction_cleaner", u"\u2714 Bake!", None))
    # retranslateUi

