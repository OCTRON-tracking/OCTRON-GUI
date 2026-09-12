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
        octron_prediction_cleaner.resize(270, 630)
        octron_prediction_cleaner.setMinimumSize(QSize(270, 630))
        octron_prediction_cleaner.setMaximumSize(QSize(270, 630))
        octron_prediction_cleaner.setCursor(QCursor(Qt.ArrowCursor))
        octron_prediction_cleaner.setWindowOpacity(1.000000000000000)
        self.verticalLayoutWidget = QWidget(octron_prediction_cleaner)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(10, 10, 254, 611))
        self.mainLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.mainLayout.setSpacing(20)
        self.mainLayout.setObjectName(u"mainLayout")
        self.mainLayout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.mainLayout.setContentsMargins(0, 5, 0, 0)
        self.octron_cleaner_ui_logo = QLabel(self.verticalLayoutWidget)
        self.octron_cleaner_ui_logo.setObjectName(u"octron_cleaner_ui_logo")
        self.octron_cleaner_ui_logo.setEnabled(True)
        self.octron_cleaner_ui_logo.setMinimumSize(QSize(250, 50))
        self.octron_cleaner_ui_logo.setMaximumSize(QSize(250, 50))
        self.octron_cleaner_ui_logo.setBaseSize(QSize(0, 0))
        self.octron_cleaner_ui_logo.setLineWidth(0)
        self.octron_cleaner_ui_logo.setPixmap(QPixmap(u"octron_prediction_cleaner.svg"))
        self.octron_cleaner_ui_logo.setScaledContents(False)
        self.octron_cleaner_ui_logo.setAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.mainLayout.addWidget(self.octron_cleaner_ui_logo)

        self.main_vertical_layout = QVBoxLayout()
        self.main_vertical_layout.setSpacing(0)
        self.main_vertical_layout.setObjectName(u"main_vertical_layout")
        self.source_trace_groupbox = QGroupBox(self.verticalLayoutWidget)
        self.source_trace_groupbox.setObjectName(u"source_trace_groupbox")
        self.source_trace_groupbox.setMinimumSize(QSize(250, 100))
        self.source_trace_groupbox.setMaximumSize(QSize(250, 100))
        self.verticalLayoutWidget_4 = QWidget(self.source_trace_groupbox)
        self.verticalLayoutWidget_4.setObjectName(u"verticalLayoutWidget_4")
        self.verticalLayoutWidget_4.setGeometry(QRect(9, 30, 231, 61))
        self.verticalLayout_5 = QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.source_layers_comboBox = QComboBox(self.verticalLayoutWidget_4)
        self.source_layers_comboBox.setObjectName(u"source_layers_comboBox")
        self.source_layers_comboBox.setMinimumSize(QSize(230, 25))
        self.source_layers_comboBox.setMaximumSize(QSize(230, 25))

        self.verticalLayout_5.addWidget(self.source_layers_comboBox, 0, Qt.AlignmentFlag.AlignHCenter)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.tracks_label = QLabel(self.verticalLayoutWidget_4)
        self.tracks_label.setObjectName(u"tracks_label")
        self.tracks_label.setMinimumSize(QSize(60, 25))
        self.tracks_label.setMaximumSize(QSize(60, 25))

        self.horizontalLayout_3.addWidget(self.tracks_label, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.total_no_tracks_label = QLabel(self.verticalLayoutWidget_4)
        self.total_no_tracks_label.setObjectName(u"total_no_tracks_label")
        self.total_no_tracks_label.setMinimumSize(QSize(60, 25))
        self.total_no_tracks_label.setMaximumSize(QSize(60, 25))

        self.horizontalLayout_3.addWidget(self.total_no_tracks_label, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.showall_layers_pushButton = QPushButton(self.verticalLayoutWidget_4)
        self.showall_layers_pushButton.setObjectName(u"showall_layers_pushButton")
        self.showall_layers_pushButton.setMinimumSize(QSize(80, 25))
        self.showall_layers_pushButton.setMaximumSize(QSize(80, 25))

        self.horizontalLayout_3.addWidget(self.showall_layers_pushButton, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.verticalLayout_5.addLayout(self.horizontalLayout_3)


        self.main_vertical_layout.addWidget(self.source_trace_groupbox)

        self.tuning_params_groupbox = QGroupBox(self.verticalLayoutWidget)
        self.tuning_params_groupbox.setObjectName(u"tuning_params_groupbox")
        self.tuning_params_groupbox.setMinimumSize(QSize(250, 120))
        self.tuning_params_groupbox.setMaximumSize(QSize(250, 120))
        self.verticalLayoutWidget_2 = QWidget(self.tuning_params_groupbox)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(10, 30, 231, 37))
        self.max_gap_frames_horizontal_layout = QHBoxLayout(self.verticalLayoutWidget_2)
        self.max_gap_frames_horizontal_layout.setObjectName(u"max_gap_frames_horizontal_layout")
        self.max_gap_frames_horizontal_layout.setContentsMargins(0, 0, 0, 0)
        self.max_gap_time_label = QLabel(self.verticalLayoutWidget_2)
        self.max_gap_time_label.setObjectName(u"max_gap_time_label")
        self.max_gap_time_label.setMinimumSize(QSize(90, 35))
        self.max_gap_time_label.setMaximumSize(QSize(100, 35))
        self.max_gap_time_label.setWordWrap(True)

        self.max_gap_frames_horizontal_layout.addWidget(self.max_gap_time_label)

        self.max_gap_time_spinbox = QSpinBox(self.verticalLayoutWidget_2)
        self.max_gap_time_spinbox.setObjectName(u"max_gap_time_spinbox")
        self.max_gap_time_spinbox.setMinimumSize(QSize(120, 25))
        self.max_gap_time_spinbox.setMaximumSize(QSize(120, 25))
        self.max_gap_time_spinbox.setMaximum(99999)
        self.max_gap_time_spinbox.setValue(10)

        self.max_gap_frames_horizontal_layout.addWidget(self.max_gap_time_spinbox)

        self.verticalLayoutWidget_3 = QWidget(self.tuning_params_groupbox)
        self.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(10, 70, 231, 37))
        self.max_gap_space_horizontal_layout = QHBoxLayout(self.verticalLayoutWidget_3)
        self.max_gap_space_horizontal_layout.setObjectName(u"max_gap_space_horizontal_layout")
        self.max_gap_space_horizontal_layout.setContentsMargins(0, 0, 0, 0)
        self.max_gap_space_label = QLabel(self.verticalLayoutWidget_3)
        self.max_gap_space_label.setObjectName(u"max_gap_space_label")
        self.max_gap_space_label.setMinimumSize(QSize(90, 35))
        self.max_gap_space_label.setMaximumSize(QSize(100, 35))
        self.max_gap_space_label.setWordWrap(True)

        self.max_gap_space_horizontal_layout.addWidget(self.max_gap_space_label)

        self.max_gap_space_spinbox = QSpinBox(self.verticalLayoutWidget_3)
        self.max_gap_space_spinbox.setObjectName(u"max_gap_space_spinbox")
        self.max_gap_space_spinbox.setMinimumSize(QSize(120, 25))
        self.max_gap_space_spinbox.setMaximumSize(QSize(120, 25))
        self.max_gap_space_spinbox.setMaximum(99999)
        self.max_gap_space_spinbox.setValue(250)

        self.max_gap_space_horizontal_layout.addWidget(self.max_gap_space_spinbox)


        self.main_vertical_layout.addWidget(self.tuning_params_groupbox)

        self.tuning_params_groupbox_3 = QGroupBox(self.verticalLayoutWidget)
        self.tuning_params_groupbox_3.setObjectName(u"tuning_params_groupbox_3")
        self.tuning_params_groupbox_3.setMinimumSize(QSize(250, 200))
        self.tuning_params_groupbox_3.setMaximumSize(QSize(250, 200))
        self.verticalLayoutWidget_5 = QWidget(self.tuning_params_groupbox_3)
        self.verticalLayoutWidget_5.setObjectName(u"verticalLayoutWidget_5")
        self.verticalLayoutWidget_5.setGeometry(QRect(9, 29, 232, 159))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.possible_joins_table = QTableView(self.verticalLayoutWidget_5)
        self.possible_joins_table.setObjectName(u"possible_joins_table")
        self.possible_joins_table.setMinimumSize(QSize(230, 120))
        self.possible_joins_table.setMaximumSize(QSize(230, 120))
        self.possible_joins_table.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.possible_joins_table.setAutoFillBackground(False)
        self.possible_joins_table.setStyleSheet(u"QTableView {\n"
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
        self.possible_joins_table.setFrameShape(QFrame.Shape.NoFrame)
        self.possible_joins_table.setFrameShadow(QFrame.Shadow.Plain)
        self.possible_joins_table.setLineWidth(1)
        self.possible_joins_table.setEditTriggers(QAbstractItemView.EditTrigger.AnyKeyPressed|QAbstractItemView.EditTrigger.EditKeyPressed|QAbstractItemView.EditTrigger.SelectedClicked)
        self.possible_joins_table.setProperty("showDropIndicator", False)
        self.possible_joins_table.setDragDropOverwriteMode(False)
        self.possible_joins_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.possible_joins_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.possible_joins_table.setGridStyle(Qt.PenStyle.SolidLine)
        self.possible_joins_table.setSortingEnabled(False)
        self.possible_joins_table.setWordWrap(False)
        self.possible_joins_table.setCornerButtonEnabled(False)
        self.possible_joins_table.horizontalHeader().setCascadingSectionResizes(True)
        self.possible_joins_table.horizontalHeader().setMinimumSectionSize(85)
        self.possible_joins_table.horizontalHeader().setDefaultSectionSize(85)
        self.possible_joins_table.horizontalHeader().setHighlightSections(True)
        self.possible_joins_table.horizontalHeader().setStretchLastSection(True)
        self.possible_joins_table.verticalHeader().setVisible(False)
        self.possible_joins_table.verticalHeader().setMinimumSectionSize(20)
        self.possible_joins_table.verticalHeader().setDefaultSectionSize(20)
        self.possible_joins_table.verticalHeader().setHighlightSections(False)

        self.verticalLayout.addWidget(self.possible_joins_table, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignTop)

        self.coverage_layout = QHBoxLayout()
        self.coverage_layout.setObjectName(u"coverage_layout")
        self.coverage_label = QLabel(self.verticalLayoutWidget_5)
        self.coverage_label.setObjectName(u"coverage_label")
        self.coverage_label.setMinimumSize(QSize(40, 25))
        self.coverage_label.setMaximumSize(QSize(40, 25))

        self.coverage_layout.addWidget(self.coverage_label)

        self.coverage_percent_label = QLabel(self.verticalLayoutWidget_5)
        self.coverage_percent_label.setObjectName(u"coverage_percent_label")
        self.coverage_percent_label.setMinimumSize(QSize(50, 25))
        self.coverage_percent_label.setMaximumSize(QSize(50, 25))
        self.coverage_percent_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.coverage_layout.addWidget(self.coverage_percent_label)

        self.increase_label = QLabel(self.verticalLayoutWidget_5)
        self.increase_label.setObjectName(u"increase_label")
        self.increase_label.setMinimumSize(QSize(60, 25))
        self.increase_label.setMaximumSize(QSize(50, 25))
        self.increase_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.coverage_layout.addWidget(self.increase_label)

        self.increase_percent_label = QLabel(self.verticalLayoutWidget_5)
        self.increase_percent_label.setObjectName(u"increase_percent_label")
        self.increase_percent_label.setMinimumSize(QSize(50, 25))
        self.increase_percent_label.setMaximumSize(QSize(50, 25))
        self.increase_percent_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.coverage_layout.addWidget(self.increase_percent_label)


        self.verticalLayout.addLayout(self.coverage_layout)


        self.main_vertical_layout.addWidget(self.tuning_params_groupbox_3)

        self.tuning_params_groupbox_2 = QGroupBox(self.verticalLayoutWidget)
        self.tuning_params_groupbox_2.setObjectName(u"tuning_params_groupbox_2")
        self.tuning_params_groupbox_2.setMinimumSize(QSize(250, 60))
        self.tuning_params_groupbox_2.setMaximumSize(QSize(250, 60))
        self.horizontalLayoutWidget = QWidget(self.tuning_params_groupbox_2)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(10, 20, 231, 31))
        self.control_buttons_layout = QHBoxLayout(self.horizontalLayoutWidget)
        self.control_buttons_layout.setObjectName(u"control_buttons_layout")
        self.control_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.reset_btn = QPushButton(self.horizontalLayoutWidget)
        self.reset_btn.setObjectName(u"reset_btn")
        self.reset_btn.setEnabled(False)
        self.reset_btn.setMinimumSize(QSize(70, 25))
        self.reset_btn.setMaximumSize(QSize(70, 25))

        self.control_buttons_layout.addWidget(self.reset_btn)

        self.revert1_btn = QPushButton(self.horizontalLayoutWidget)
        self.revert1_btn.setObjectName(u"revert1_btn")
        self.revert1_btn.setEnabled(False)
        self.revert1_btn.setMinimumSize(QSize(70, 25))
        self.revert1_btn.setMaximumSize(QSize(70, 25))

        self.control_buttons_layout.addWidget(self.revert1_btn)

        self.save_btn = QPushButton(self.horizontalLayoutWidget)
        self.save_btn.setObjectName(u"save_btn")
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumSize(QSize(70, 25))
        self.save_btn.setMaximumSize(QSize(70, 25))

        self.control_buttons_layout.addWidget(self.save_btn)


        self.main_vertical_layout.addWidget(self.tuning_params_groupbox_2)


        self.mainLayout.addLayout(self.main_vertical_layout)


        self.retranslateUi(octron_prediction_cleaner)

        QMetaObject.connectSlotsByName(octron_prediction_cleaner)
    # setupUi

    def retranslateUi(self, octron_prediction_cleaner):
        octron_prediction_cleaner.setWindowTitle(QCoreApplication.translate("octron_prediction_cleaner", u"octron_gui", None))
        self.octron_cleaner_ui_logo.setText("")
        self.source_trace_groupbox.setTitle(QCoreApplication.translate("octron_prediction_cleaner", u"Source tracking layer", None))
        self.tracks_label.setText(QCoreApplication.translate("octron_prediction_cleaner", u"# Tracks:", None))
        self.total_no_tracks_label.setText(QCoreApplication.translate("octron_prediction_cleaner", u"0", None))
        self.showall_layers_pushButton.setText(QCoreApplication.translate("octron_prediction_cleaner", u"Show all", None))
        self.tuning_params_groupbox.setTitle(QCoreApplication.translate("octron_prediction_cleaner", u"Tuning", None))
        self.max_gap_time_label.setText(QCoreApplication.translate("octron_prediction_cleaner", u"Max. gap in time (frames)", None))
        self.max_gap_time_spinbox.setSuffix("")
        self.max_gap_space_label.setText(QCoreApplication.translate("octron_prediction_cleaner", u"Max. gap in space (px)", None))
        self.max_gap_space_spinbox.setSuffix("")
        self.tuning_params_groupbox_3.setTitle(QCoreApplication.translate("octron_prediction_cleaner", u"Possible tracking joins", None))
        self.coverage_label.setText(QCoreApplication.translate("octron_prediction_cleaner", u"Cov:", None))
        self.coverage_percent_label.setText(QCoreApplication.translate("octron_prediction_cleaner", u"%", None))
        self.increase_label.setText(QCoreApplication.translate("octron_prediction_cleaner", u"Increase:", None))
        self.increase_percent_label.setText(QCoreApplication.translate("octron_prediction_cleaner", u"%", None))
        self.tuning_params_groupbox_2.setTitle("")
        self.reset_btn.setText(QCoreApplication.translate("octron_prediction_cleaner", u"\u29bf Reset", None))
        self.revert1_btn.setText(QCoreApplication.translate("octron_prediction_cleaner", u"\u2190 Revert", None))
        self.save_btn.setText(QCoreApplication.translate("octron_prediction_cleaner", u"\u21f2 Save", None))
    # retranslateUi

