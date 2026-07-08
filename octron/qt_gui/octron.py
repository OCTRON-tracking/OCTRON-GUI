################################################################################
## Form generated from reading UI file 'octron.ui'
##
## Created by: Qt User Interface Compiler version 5.15.18
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *  # type: ignore
from PySide2.QtGui import *  # type: ignore
from PySide2.QtWidgets import *  # type: ignore


class Ui_octron_widgetui:
    def setupUi(self, octron_widgetui):
        if not octron_widgetui.objectName():
            octron_widgetui.setObjectName("octron_widgetui")
        octron_widgetui.setEnabled(True)
        octron_widgetui.resize(410, 750)
        octron_widgetui.setMinimumSize(QSize(410, 750))
        octron_widgetui.setMaximumSize(QSize(410, 750))
        octron_widgetui.setCursor(QCursor(Qt.ArrowCursor))
        octron_widgetui.setWindowOpacity(1.000000000000000)
        self.verticalLayoutWidget = QWidget(octron_widgetui)
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(0, 0, 412, 751))
        self.mainLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.mainLayout.setSpacing(20)
        self.mainLayout.setObjectName("mainLayout")
        self.mainLayout.setSizeConstraint(
            QLayout.SizeConstraint.SetNoConstraint
        )
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.octron_logo = QLabel(self.verticalLayoutWidget)
        self.octron_logo.setObjectName("octron_logo")
        self.octron_logo.setEnabled(True)
        self.octron_logo.setMinimumSize(QSize(410, 120))
        self.octron_logo.setMaximumSize(QSize(410, 120))
        self.octron_logo.setBaseSize(QSize(0, 0))
        self.octron_logo.setLineWidth(0)
        self.octron_logo.setPixmap(QPixmap("octron_logo.svg"))
        self.octron_logo.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )

        self.mainLayout.addWidget(
            self.octron_logo,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self.main_toolbox = QToolBox(self.verticalLayoutWidget)
        self.main_toolbox.setObjectName("main_toolbox")
        self.main_toolbox.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.main_toolbox.sizePolicy().hasHeightForWidth()
        )
        self.main_toolbox.setSizePolicy(sizePolicy)
        self.main_toolbox.setMinimumSize(QSize(410, 600))
        self.main_toolbox.setMaximumSize(QSize(410, 600))
        self.main_toolbox.setCursor(QCursor(Qt.ArrowCursor))
        self.main_toolbox.setFrameShape(QFrame.Shape.NoFrame)
        self.main_toolbox.setFrameShadow(QFrame.Shadow.Plain)
        self.main_toolbox.setLineWidth(0)
        self.main_toolbox.setMidLineWidth(0)
        self.project_tab = QWidget()
        self.project_tab.setObjectName("project_tab")
        self.project_tab.setGeometry(QRect(0, 0, 410, 464))
        sizePolicy1 = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(
            self.project_tab.sizePolicy().hasHeightForWidth()
        )
        self.project_tab.setSizePolicy(sizePolicy1)
        self.verticalLayoutWidget_3 = QWidget(self.project_tab)
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(0, 0, 402, 461))
        self.project_vertical_layout = QVBoxLayout(self.verticalLayoutWidget_3)
        self.project_vertical_layout.setSpacing(20)
        self.project_vertical_layout.setObjectName("project_vertical_layout")
        self.project_vertical_layout.setContentsMargins(0, 0, 0, 15)
        self.folder_sect_groupbox = QGroupBox(self.verticalLayoutWidget_3)
        self.folder_sect_groupbox.setObjectName("folder_sect_groupbox")
        sizePolicy2 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(
            self.folder_sect_groupbox.sizePolicy().hasHeightForWidth()
        )
        self.folder_sect_groupbox.setSizePolicy(sizePolicy2)
        self.folder_sect_groupbox.setMinimumSize(QSize(400, 95))
        self.folder_sect_groupbox.setMaximumSize(QSize(400, 95))
        self.horizontalLayout_11 = QHBoxLayout(self.folder_sect_groupbox)
        self.horizontalLayout_11.setSpacing(20)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.horizontalLayout_11.setContentsMargins(9, 12, 9, 12)
        self.create_project_btn = QPushButton(self.folder_sect_groupbox)
        self.create_project_btn.setObjectName("create_project_btn")
        self.create_project_btn.setMinimumSize(QSize(100, 25))
        self.create_project_btn.setMaximumSize(QSize(100, 25))

        self.horizontalLayout_11.addWidget(
            self.create_project_btn,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.project_folder_path_label = QLabel(self.folder_sect_groupbox)
        self.project_folder_path_label.setObjectName(
            "project_folder_path_label"
        )
        self.project_folder_path_label.setEnabled(False)
        self.project_folder_path_label.setMinimumSize(QSize(255, 53))
        self.project_folder_path_label.setMaximumSize(QSize(255, 53))
        self.project_folder_path_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTrailing
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.project_folder_path_label.setWordWrap(True)
        self.project_folder_path_label.setMargin(0)

        self.horizontalLayout_11.addWidget(self.project_folder_path_label)

        self.project_vertical_layout.addWidget(
            self.folder_sect_groupbox,
            0,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
        )

        self.project_video_drop_groupbox = QGroupBox(
            self.verticalLayoutWidget_3
        )
        self.project_video_drop_groupbox.setObjectName(
            "project_video_drop_groupbox"
        )
        sizePolicy2.setHeightForWidth(
            self.project_video_drop_groupbox.sizePolicy().hasHeightForWidth()
        )
        self.project_video_drop_groupbox.setSizePolicy(sizePolicy2)
        self.project_video_drop_groupbox.setMinimumSize(QSize(400, 100))
        self.project_video_drop_groupbox.setMaximumSize(QSize(400, 100))
        self.horizontalLayout = QHBoxLayout(self.project_video_drop_groupbox)
        self.horizontalLayout.setSpacing(9)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout.setContentsMargins(9, 12, 9, 12)
        self.video_file_drop_widget = QWidget(self.project_video_drop_groupbox)
        self.video_file_drop_widget.setObjectName("video_file_drop_widget")
        self.video_file_drop_widget.setMinimumSize(QSize(380, 60))
        self.video_file_drop_widget.setMaximumSize(QSize(380, 60))

        self.horizontalLayout.addWidget(self.video_file_drop_widget)

        self.project_vertical_layout.addWidget(
            self.project_video_drop_groupbox
        )

        self.project_existing_data_groupbox = QGroupBox(
            self.verticalLayoutWidget_3
        )
        self.project_existing_data_groupbox.setObjectName(
            "project_existing_data_groupbox"
        )
        self.project_existing_data_groupbox.setEnabled(False)
        sizePolicy2.setHeightForWidth(
            self.project_existing_data_groupbox.sizePolicy().hasHeightForWidth()
        )
        self.project_existing_data_groupbox.setSizePolicy(sizePolicy2)
        self.project_existing_data_groupbox.setMinimumSize(QSize(400, 220))
        self.project_existing_data_groupbox.setMaximumSize(QSize(400, 220))
        self.horizontalLayout_9 = QHBoxLayout(
            self.project_existing_data_groupbox
        )
        self.horizontalLayout_9.setSpacing(20)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(9, 12, 9, 12)
        self.existing_data_table = QTableView(
            self.project_existing_data_groupbox
        )
        self.existing_data_table.setObjectName("existing_data_table")
        self.existing_data_table.setMinimumSize(QSize(380, 180))
        self.existing_data_table.setMaximumSize(QSize(380, 180))
        self.existing_data_table.setEditTriggers(
            QAbstractItemView.EditTrigger.AnyKeyPressed
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.existing_data_table.setProperty("showDropIndicator", False)
        self.existing_data_table.setDragDropOverwriteMode(False)
        self.existing_data_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.existing_data_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.existing_data_table.setGridStyle(Qt.PenStyle.SolidLine)
        self.existing_data_table.setSortingEnabled(False)
        self.existing_data_table.setWordWrap(False)
        self.existing_data_table.setCornerButtonEnabled(False)
        self.existing_data_table.horizontalHeader().setCascadingSectionResizes(
            True
        )
        self.existing_data_table.horizontalHeader().setMinimumSectionSize(85)
        self.existing_data_table.horizontalHeader().setDefaultSectionSize(85)
        self.existing_data_table.horizontalHeader().setHighlightSections(False)
        self.existing_data_table.verticalHeader().setVisible(False)
        self.existing_data_table.verticalHeader().setMinimumSectionSize(20)
        self.existing_data_table.verticalHeader().setDefaultSectionSize(20)
        self.existing_data_table.verticalHeader().setHighlightSections(False)

        self.horizontalLayout_9.addWidget(
            self.existing_data_table,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self.project_vertical_layout.addWidget(
            self.project_existing_data_groupbox, 0, Qt.AlignmentFlag.AlignTop
        )

        icon = QIcon()
        icon.addFile(
            "icons/noun-project-7158867.svg", QSize(), QIcon.Normal, QIcon.Off
        )
        self.main_toolbox.addItem(self.project_tab, icon, "Manage project")
        self.annotate_tab = QWidget()
        self.annotate_tab.setObjectName("annotate_tab")
        self.annotate_tab.setGeometry(QRect(0, 0, 405, 464))
        sizePolicy1.setHeightForWidth(
            self.annotate_tab.sizePolicy().hasHeightForWidth()
        )
        self.annotate_tab.setSizePolicy(sizePolicy1)
        self.annotate_tab.setMaximumSize(QSize(405, 700))
        self.verticalLayoutWidget_2 = QWidget(self.annotate_tab)
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(0, 0, 402, 394))
        self.annotate_vertical_layout = QVBoxLayout(
            self.verticalLayoutWidget_2
        )
        self.annotate_vertical_layout.setSpacing(20)
        self.annotate_vertical_layout.setObjectName("annotate_vertical_layout")
        self.annotate_vertical_layout.setContentsMargins(0, 0, 0, 15)
        self.model_select_groupbox = QGroupBox(self.verticalLayoutWidget_2)
        self.model_select_groupbox.setObjectName("model_select_groupbox")
        sizePolicy3 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(
            self.model_select_groupbox.sizePolicy().hasHeightForWidth()
        )
        self.model_select_groupbox.setSizePolicy(sizePolicy3)
        self.model_select_groupbox.setMinimumSize(QSize(400, 80))
        self.model_select_groupbox.setMaximumSize(QSize(400, 80))
        self.layoutWidget = QWidget(self.model_select_groupbox)
        self.layoutWidget.setObjectName("layoutWidget")
        self.layoutWidget.setGeometry(QRect(11, 30, 380, 37))
        self.model_select_grid_layout = QGridLayout(self.layoutWidget)
        self.model_select_grid_layout.setObjectName("model_select_grid_layout")
        self.model_select_grid_layout.setHorizontalSpacing(10)
        self.model_select_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.threshold_label = QLabel(self.layoutWidget)
        self.threshold_label.setObjectName("threshold_label")
        self.threshold_label.setMinimumSize(QSize(50, 25))
        self.threshold_label.setMaximumSize(QSize(50, 25))
        self.threshold_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTrailing
            | Qt.AlignmentFlag.AlignVCenter
        )

        self.model_select_grid_layout.addWidget(
            self.threshold_label,
            0,
            1,
            1,
            1,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        self.load_sam_model_btn = QPushButton(self.layoutWidget)
        self.load_sam_model_btn.setObjectName("load_sam_model_btn")
        self.load_sam_model_btn.setMinimumSize(QSize(88, 25))
        self.load_sam_model_btn.setMaximumSize(QSize(88, 25))

        self.model_select_grid_layout.addWidget(
            self.load_sam_model_btn,
            0,
            4,
            1,
            1,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        self.sam3detect_thresh = QLineEdit(self.layoutWidget)
        self.sam3detect_thresh.setObjectName("sam3detect_thresh")
        self.sam3detect_thresh.setMinimumSize(QSize(39, 25))
        self.sam3detect_thresh.setMaximumSize(QSize(39, 25))
        self.sam3detect_thresh.setInputMask("")
        self.sam3detect_thresh.setText("")
        self.sam3detect_thresh.setMaxLength(100)

        self.model_select_grid_layout.addWidget(
            self.sam3detect_thresh,
            0,
            2,
            1,
            1,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        self.sam_model_list = QComboBox(self.layoutWidget)
        self.sam_model_list.addItem("")
        self.sam_model_list.setObjectName("sam_model_list")
        self.sam_model_list.setMinimumSize(QSize(120, 25))
        self.sam_model_list.setMaximumSize(QSize(120, 25))

        self.model_select_grid_layout.addWidget(
            self.sam_model_list,
            0,
            0,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.feed_input_to_predictor_btn = QPushButton(self.layoutWidget)
        self.feed_input_to_predictor_btn.setObjectName(
            "feed_input_to_predictor_btn"
        )
        self.feed_input_to_predictor_btn.setEnabled(False)
        sizePolicy4 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(
            self.feed_input_to_predictor_btn.sizePolicy().hasHeightForWidth()
        )
        self.feed_input_to_predictor_btn.setSizePolicy(sizePolicy4)
        self.feed_input_to_predictor_btn.setMinimumSize(QSize(45, 25))
        self.feed_input_to_predictor_btn.setMaximumSize(QSize(45, 25))
        self.feed_input_to_predictor_btn.setBaseSize(QSize(15, 25))

        self.model_select_grid_layout.addWidget(
            self.feed_input_to_predictor_btn,
            0,
            3,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.annotate_vertical_layout.addWidget(
            self.model_select_groupbox, 0, Qt.AlignmentFlag.AlignTop
        )

        self.annotate_layer_create_groupbox = QGroupBox(
            self.verticalLayoutWidget_2
        )
        self.annotate_layer_create_groupbox.setObjectName(
            "annotate_layer_create_groupbox"
        )
        sizePolicy3.setHeightForWidth(
            self.annotate_layer_create_groupbox.sizePolicy().hasHeightForWidth()
        )
        self.annotate_layer_create_groupbox.setSizePolicy(sizePolicy3)
        self.annotate_layer_create_groupbox.setMinimumSize(QSize(400, 100))
        self.annotate_layer_create_groupbox.setMaximumSize(QSize(400, 100))
        self.layoutWidget1 = QWidget(self.annotate_layer_create_groupbox)
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(9, 30, 381, 64))
        self.label_manager_grid_layout = QGridLayout(self.layoutWidget1)
        self.label_manager_grid_layout.setObjectName(
            "label_manager_grid_layout"
        )
        self.label_manager_grid_layout.setHorizontalSpacing(10)
        self.label_manager_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_type_combobox = QComboBox(self.layoutWidget1)
        self.layer_type_combobox.addItem("")
        self.layer_type_combobox.addItem("")
        self.layer_type_combobox.addItem("")
        self.layer_type_combobox.addItem("")
        self.layer_type_combobox.setObjectName("layer_type_combobox")
        self.layer_type_combobox.setMinimumSize(QSize(110, 25))
        self.layer_type_combobox.setMaximumSize(QSize(110, 25))
        self.layer_type_combobox.setMaxCount(15)
        self.layer_type_combobox.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.layer_type_combobox.setIconSize(QSize(14, 14))
        self.layer_type_combobox.setFrame(False)

        self.label_manager_grid_layout.addWidget(
            self.layer_type_combobox, 0, 0, 1, 1, Qt.AlignmentFlag.AlignLeft
        )

        self.label_list_combobox = QComboBox(self.layoutWidget1)
        self.label_list_combobox.addItem("")
        self.label_list_combobox.addItem("")
        self.label_list_combobox.addItem("")
        self.label_list_combobox.setObjectName("label_list_combobox")
        self.label_list_combobox.setMinimumSize(QSize(110, 25))
        self.label_list_combobox.setMaximumSize(QSize(110, 25))
        self.label_list_combobox.setEditable(False)
        self.label_list_combobox.setMaxVisibleItems(30)
        self.label_list_combobox.setMaxCount(30)
        self.label_list_combobox.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.label_list_combobox.setIconSize(QSize(14, 14))
        self.label_list_combobox.setFrame(False)

        self.label_manager_grid_layout.addWidget(
            self.label_list_combobox, 0, 1, 1, 1
        )

        self.label_suffix_lineedit = QLineEdit(self.layoutWidget1)
        self.label_suffix_lineedit.setObjectName("label_suffix_lineedit")
        self.label_suffix_lineedit.setMinimumSize(QSize(60, 25))
        self.label_suffix_lineedit.setMaximumSize(QSize(60, 25))
        self.label_suffix_lineedit.setInputMask("")
        self.label_suffix_lineedit.setText("")
        self.label_suffix_lineedit.setMaxLength(100)

        self.label_manager_grid_layout.addWidget(
            self.label_suffix_lineedit, 0, 2, 1, 1
        )

        self.create_annotation_layer_btn = QPushButton(self.layoutWidget1)
        self.create_annotation_layer_btn.setObjectName(
            "create_annotation_layer_btn"
        )
        self.create_annotation_layer_btn.setMinimumSize(QSize(70, 25))
        self.create_annotation_layer_btn.setMaximumSize(QSize(70, 25))

        self.label_manager_grid_layout.addWidget(
            self.create_annotation_layer_btn,
            0,
            3,
            1,
            1,
            Qt.AlignmentFlag.AlignRight,
        )

        self.create_projection_layer_btn = QPushButton(self.layoutWidget1)
        self.create_projection_layer_btn.setObjectName(
            "create_projection_layer_btn"
        )
        self.create_projection_layer_btn.setMinimumSize(QSize(110, 25))
        self.create_projection_layer_btn.setMaximumSize(QSize(110, 25))

        self.label_manager_grid_layout.addWidget(
            self.create_projection_layer_btn,
            1,
            0,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.hard_reset_layer_btn = QPushButton(self.layoutWidget1)
        self.hard_reset_layer_btn.setObjectName("hard_reset_layer_btn")
        self.hard_reset_layer_btn.setMinimumSize(QSize(70, 25))
        self.hard_reset_layer_btn.setMaximumSize(QSize(70, 25))
        self.hard_reset_layer_btn.setAutoRepeatInterval(2000)

        self.label_manager_grid_layout.addWidget(
            self.hard_reset_layer_btn, 1, 3, 1, 1, Qt.AlignmentFlag.AlignRight
        )

        self.annotate_vertical_layout.addWidget(
            self.annotate_layer_create_groupbox
        )

        self.annotate_layer_timeline_groupbox = QGroupBox(
            self.verticalLayoutWidget_2
        )
        self.annotate_layer_timeline_groupbox.setObjectName(
            "annotate_layer_timeline_groupbox"
        )
        sizePolicy3.setHeightForWidth(
            self.annotate_layer_timeline_groupbox.sizePolicy().hasHeightForWidth()
        )
        self.annotate_layer_timeline_groupbox.setSizePolicy(sizePolicy3)
        self.annotate_layer_timeline_groupbox.setMinimumSize(QSize(400, 70))
        self.annotate_layer_timeline_groupbox.setMaximumSize(QSize(400, 70))
        self.annotation_jump_previous_btn = QPushButton(
            self.annotate_layer_timeline_groupbox
        )
        self.annotation_jump_previous_btn.setObjectName(
            "annotation_jump_previous_btn"
        )
        self.annotation_jump_previous_btn.setGeometry(QRect(30, 33, 150, 25))
        self.annotation_jump_previous_btn.setMinimumSize(QSize(150, 25))
        self.annotation_jump_previous_btn.setMaximumSize(QSize(70, 25))
        self.annotation_jump_next_btn = QPushButton(
            self.annotate_layer_timeline_groupbox
        )
        self.annotation_jump_next_btn.setObjectName("annotation_jump_next_btn")
        self.annotation_jump_next_btn.setGeometry(QRect(218, 33, 150, 25))
        self.annotation_jump_next_btn.setMinimumSize(QSize(150, 25))
        self.annotation_jump_next_btn.setMaximumSize(QSize(70, 25))

        self.annotate_vertical_layout.addWidget(
            self.annotate_layer_timeline_groupbox
        )

        self.annotate_layer_predict_groupbox = QGroupBox(
            self.verticalLayoutWidget_2
        )
        self.annotate_layer_predict_groupbox.setObjectName(
            "annotate_layer_predict_groupbox"
        )
        sizePolicy3.setHeightForWidth(
            self.annotate_layer_predict_groupbox.sizePolicy().hasHeightForWidth()
        )
        self.annotate_layer_predict_groupbox.setSizePolicy(sizePolicy3)
        self.annotate_layer_predict_groupbox.setMinimumSize(QSize(400, 70))
        self.annotate_layer_predict_groupbox.setMaximumSize(QSize(400, 70))
        self.horizontalLayout_2 = QHBoxLayout(
            self.annotate_layer_predict_groupbox
        )
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(9, 12, 9, 12)
        self.batch_predict_progressbar = QProgressBar(
            self.annotate_layer_predict_groupbox
        )
        self.batch_predict_progressbar.setObjectName(
            "batch_predict_progressbar"
        )
        sizePolicy5 = QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
        )
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(
            self.batch_predict_progressbar.sizePolicy().hasHeightForWidth()
        )
        self.batch_predict_progressbar.setSizePolicy(sizePolicy5)
        self.batch_predict_progressbar.setMinimumSize(QSize(130, 25))
        self.batch_predict_progressbar.setMaximumSize(QSize(130, 25))
        self.batch_predict_progressbar.setMaximum(20)
        self.batch_predict_progressbar.setValue(0)

        self.horizontalLayout_2.addWidget(
            self.batch_predict_progressbar,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.skip_label = QLabel(self.annotate_layer_predict_groupbox)
        self.skip_label.setObjectName("skip_label")
        self.skip_label.setMinimumSize(QSize(30, 25))
        self.skip_label.setMaximumSize(QSize(30, 25))
        self.skip_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTrailing
            | Qt.AlignmentFlag.AlignVCenter
        )

        self.horizontalLayout_2.addWidget(
            self.skip_label, 0, Qt.AlignmentFlag.AlignRight
        )

        self.skip_frames_spinbox = QSpinBox(
            self.annotate_layer_predict_groupbox
        )
        self.skip_frames_spinbox.setObjectName("skip_frames_spinbox")
        sizePolicy4.setHeightForWidth(
            self.skip_frames_spinbox.sizePolicy().hasHeightForWidth()
        )
        self.skip_frames_spinbox.setSizePolicy(sizePolicy4)
        self.skip_frames_spinbox.setMinimumSize(QSize(35, 25))
        self.skip_frames_spinbox.setMaximumSize(QSize(35, 25))
        self.skip_frames_spinbox.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTrailing
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.skip_frames_spinbox.setMaximum(200)

        self.horizontalLayout_2.addWidget(self.skip_frames_spinbox)

        self.predict_next_oneframe_btn = QPushButton(
            self.annotate_layer_predict_groupbox
        )
        self.predict_next_oneframe_btn.setObjectName(
            "predict_next_oneframe_btn"
        )
        self.predict_next_oneframe_btn.setEnabled(False)
        sizePolicy4.setHeightForWidth(
            self.predict_next_oneframe_btn.sizePolicy().hasHeightForWidth()
        )
        self.predict_next_oneframe_btn.setSizePolicy(sizePolicy4)
        self.predict_next_oneframe_btn.setMinimumSize(QSize(20, 25))
        self.predict_next_oneframe_btn.setMaximumSize(QSize(20, 25))
        self.predict_next_oneframe_btn.setBaseSize(QSize(15, 25))

        self.horizontalLayout_2.addWidget(self.predict_next_oneframe_btn)

        self.predict_next_batch_btn = QPushButton(
            self.annotate_layer_predict_groupbox
        )
        self.predict_next_batch_btn.setObjectName("predict_next_batch_btn")
        self.predict_next_batch_btn.setEnabled(False)
        self.predict_next_batch_btn.setMinimumSize(QSize(80, 25))
        self.predict_next_batch_btn.setMaximumSize(QSize(80, 25))

        self.horizontalLayout_2.addWidget(
            self.predict_next_batch_btn, 0, Qt.AlignmentFlag.AlignVCenter
        )

        self.annotate_vertical_layout.addWidget(
            self.annotate_layer_predict_groupbox
        )

        icon1 = QIcon()
        icon1.addFile(
            "icons/noun-copywriting-7158879.svg",
            QSize(),
            QIcon.Normal,
            QIcon.Off,
        )
        self.main_toolbox.addItem(
            self.annotate_tab, icon1, "Generate annotation data"
        )
        self.train_tab = QWidget()
        self.train_tab.setObjectName("train_tab")
        self.train_tab.setGeometry(QRect(0, 0, 410, 464))
        sizePolicy1.setHeightForWidth(
            self.train_tab.sizePolicy().hasHeightForWidth()
        )
        self.train_tab.setSizePolicy(sizePolicy1)
        self.verticalLayoutWidget_4 = QWidget(self.train_tab)
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayoutWidget_4.setGeometry(QRect(0, 0, 402, 475))
        self.train_vertical_layout = QVBoxLayout(self.verticalLayoutWidget_4)
        self.train_vertical_layout.setSpacing(20)
        self.train_vertical_layout.setObjectName("train_vertical_layout")
        self.train_vertical_layout.setContentsMargins(0, 0, 0, 10)
        self.segmentation_bbox_decision_groupbox = QGroupBox(
            self.verticalLayoutWidget_4
        )
        self.segmentation_bbox_decision_groupbox.setObjectName(
            "segmentation_bbox_decision_groupbox"
        )
        self.segmentation_bbox_decision_groupbox.setMinimumSize(QSize(400, 65))
        self.segmentation_bbox_decision_groupbox.setMaximumSize(QSize(400, 65))
        self.segmentation_bbox_decision_groupbox.setBaseSize(QSize(400, 60))
        self.layoutWidget2 = QWidget(self.segmentation_bbox_decision_groupbox)
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.layoutWidget2.setGeometry(QRect(12, 30, 383, 21))
        self.gridLayout = QGridLayout(self.layoutWidget2)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.segmentation_radiobutton = QRadioButton(self.layoutWidget2)
        self.segmentation_radiobutton.setObjectName("segmentation_radiobutton")
        self.segmentation_radiobutton.setChecked(True)

        self.gridLayout.addWidget(
            self.segmentation_radiobutton,
            0,
            0,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.detection_radiobutton = QRadioButton(self.layoutWidget2)
        self.detection_radiobutton.setObjectName("detection_radiobutton")

        self.gridLayout.addWidget(
            self.detection_radiobutton, 0, 1, 1, 1, Qt.AlignmentFlag.AlignLeft
        )

        self.train_vertical_layout.addWidget(
            self.segmentation_bbox_decision_groupbox,
            0,
            Qt.AlignmentFlag.AlignTop,
        )

        self.train_generate_groupbox = QGroupBox(self.verticalLayoutWidget_4)
        self.train_generate_groupbox.setObjectName("train_generate_groupbox")
        self.train_generate_groupbox.setEnabled(False)
        sizePolicy4.setHeightForWidth(
            self.train_generate_groupbox.sizePolicy().hasHeightForWidth()
        )
        self.train_generate_groupbox.setSizePolicy(sizePolicy4)
        self.train_generate_groupbox.setMinimumSize(QSize(400, 160))
        self.train_generate_groupbox.setMaximumSize(QSize(400, 140))
        self.train_generate_groupbox.setAlignment(
            Qt.AlignmentFlag.AlignLeading
            | Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.layoutWidget_4 = QWidget(self.train_generate_groupbox)
        self.layoutWidget_4.setObjectName("layoutWidget_4")
        self.layoutWidget_4.setGeometry(QRect(10, 30, 281, 29))
        self.train_progress_A_horizontalLayout = QHBoxLayout(
            self.layoutWidget_4
        )
        self.train_progress_A_horizontalLayout.setObjectName(
            "train_progress_A_horizontalLayout"
        )
        self.train_progress_A_horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.train_polygons_overall_progressbar = QProgressBar(
            self.layoutWidget_4
        )
        self.train_polygons_overall_progressbar.setObjectName(
            "train_polygons_overall_progressbar"
        )
        self.train_polygons_overall_progressbar.setEnabled(False)
        sizePolicy5.setHeightForWidth(
            self.train_polygons_overall_progressbar.sizePolicy().hasHeightForWidth()
        )
        self.train_polygons_overall_progressbar.setSizePolicy(sizePolicy5)
        self.train_polygons_overall_progressbar.setMinimumSize(QSize(50, 25))
        self.train_polygons_overall_progressbar.setMaximumSize(QSize(50, 25))
        self.train_polygons_overall_progressbar.setMaximum(20)
        self.train_polygons_overall_progressbar.setValue(0)

        self.train_progress_A_horizontalLayout.addWidget(
            self.train_polygons_overall_progressbar
        )

        self.train_polygons_frames_progressbar = QProgressBar(
            self.layoutWidget_4
        )
        self.train_polygons_frames_progressbar.setObjectName(
            "train_polygons_frames_progressbar"
        )
        self.train_polygons_frames_progressbar.setEnabled(False)
        sizePolicy5.setHeightForWidth(
            self.train_polygons_frames_progressbar.sizePolicy().hasHeightForWidth()
        )
        self.train_polygons_frames_progressbar.setSizePolicy(sizePolicy5)
        self.train_polygons_frames_progressbar.setMinimumSize(QSize(100, 25))
        self.train_polygons_frames_progressbar.setMaximumSize(QSize(100, 25))
        self.train_polygons_frames_progressbar.setMaximum(20)
        self.train_polygons_frames_progressbar.setValue(0)

        self.train_progress_A_horizontalLayout.addWidget(
            self.train_polygons_frames_progressbar
        )

        self.train_polygons_label = QLabel(self.layoutWidget_4)
        self.train_polygons_label.setObjectName("train_polygons_label")
        self.train_polygons_label.setEnabled(False)
        self.train_polygons_label.setMinimumSize(QSize(0, 25))
        self.train_polygons_label.setMaximumSize(QSize(16777215, 25))

        self.train_progress_A_horizontalLayout.addWidget(
            self.train_polygons_label
        )

        self.layoutWidget_5 = QWidget(self.train_generate_groupbox)
        self.layoutWidget_5.setObjectName("layoutWidget_5")
        self.layoutWidget_5.setGeometry(QRect(10, 60, 281, 29))
        self.train_progress_B_horizontalLayout = QHBoxLayout(
            self.layoutWidget_5
        )
        self.train_progress_B_horizontalLayout.setObjectName(
            "train_progress_B_horizontalLayout"
        )
        self.train_progress_B_horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.train_export_overall_progressbar = QProgressBar(
            self.layoutWidget_5
        )
        self.train_export_overall_progressbar.setObjectName(
            "train_export_overall_progressbar"
        )
        self.train_export_overall_progressbar.setEnabled(False)
        sizePolicy5.setHeightForWidth(
            self.train_export_overall_progressbar.sizePolicy().hasHeightForWidth()
        )
        self.train_export_overall_progressbar.setSizePolicy(sizePolicy5)
        self.train_export_overall_progressbar.setMinimumSize(QSize(50, 25))
        self.train_export_overall_progressbar.setMaximumSize(QSize(50, 25))
        self.train_export_overall_progressbar.setMaximum(20)
        self.train_export_overall_progressbar.setValue(0)

        self.train_progress_B_horizontalLayout.addWidget(
            self.train_export_overall_progressbar
        )

        self.train_export_frames_progressbar = QProgressBar(
            self.layoutWidget_5
        )
        self.train_export_frames_progressbar.setObjectName(
            "train_export_frames_progressbar"
        )
        self.train_export_frames_progressbar.setEnabled(False)
        sizePolicy5.setHeightForWidth(
            self.train_export_frames_progressbar.sizePolicy().hasHeightForWidth()
        )
        self.train_export_frames_progressbar.setSizePolicy(sizePolicy5)
        self.train_export_frames_progressbar.setMinimumSize(QSize(100, 25))
        self.train_export_frames_progressbar.setMaximumSize(QSize(100, 25))
        self.train_export_frames_progressbar.setMaximum(20)
        self.train_export_frames_progressbar.setValue(0)

        self.train_progress_B_horizontalLayout.addWidget(
            self.train_export_frames_progressbar
        )

        self.train_export_label = QLabel(self.layoutWidget_5)
        self.train_export_label.setObjectName("train_export_label")
        self.train_export_label.setEnabled(False)
        self.train_export_label.setMinimumSize(QSize(0, 25))
        self.train_export_label.setMaximumSize(QSize(16777215, 25))

        self.train_progress_B_horizontalLayout.addWidget(
            self.train_export_label
        )

        self.layoutWidget_6 = QWidget(self.train_generate_groupbox)
        self.layoutWidget_6.setObjectName("layoutWidget_6")
        self.layoutWidget_6.setGeometry(QRect(300, 30, 90, 81))
        self.train_checkboxes_verticalLayout = QVBoxLayout(self.layoutWidget_6)
        self.train_checkboxes_verticalLayout.setSpacing(10)
        self.train_checkboxes_verticalLayout.setObjectName(
            "train_checkboxes_verticalLayout"
        )
        self.train_checkboxes_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.train_prune_checkBox = QCheckBox(self.layoutWidget_6)
        self.train_prune_checkBox.setObjectName("train_prune_checkBox")
        self.train_prune_checkBox.setEnabled(False)
        self.train_prune_checkBox.setMinimumSize(QSize(90, 25))
        self.train_prune_checkBox.setMaximumSize(QSize(90, 25))
        self.train_prune_checkBox.setChecked(False)

        self.train_checkboxes_verticalLayout.addWidget(
            self.train_prune_checkBox
        )

        self.train_data_watershed_checkBox = QCheckBox(self.layoutWidget_6)
        self.train_data_watershed_checkBox.setObjectName(
            "train_data_watershed_checkBox"
        )
        self.train_data_watershed_checkBox.setEnabled(False)
        self.train_data_watershed_checkBox.setMinimumSize(QSize(90, 25))
        self.train_data_watershed_checkBox.setMaximumSize(QSize(90, 25))
        self.train_data_watershed_checkBox.setChecked(False)

        self.train_checkboxes_verticalLayout.addWidget(
            self.train_data_watershed_checkBox
        )

        self.train_data_overwrite_checkBox = QCheckBox(self.layoutWidget_6)
        self.train_data_overwrite_checkBox.setObjectName(
            "train_data_overwrite_checkBox"
        )
        self.train_data_overwrite_checkBox.setEnabled(False)
        self.train_data_overwrite_checkBox.setMinimumSize(QSize(90, 25))
        self.train_data_overwrite_checkBox.setMaximumSize(QSize(90, 25))
        self.train_data_overwrite_checkBox.setChecked(True)

        self.train_checkboxes_verticalLayout.addWidget(
            self.train_data_overwrite_checkBox
        )

        self.layoutWidget_7 = QWidget(self.train_generate_groupbox)
        self.layoutWidget_7.setObjectName("layoutWidget_7")
        self.layoutWidget_7.setGeometry(QRect(10, 120, 381, 37))
        self.train_folder_btn_horizontalLayout = QHBoxLayout(
            self.layoutWidget_7
        )
        self.train_folder_btn_horizontalLayout.setObjectName(
            "train_folder_btn_horizontalLayout"
        )
        self.train_folder_btn_horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.training_data_folder_label = QLabel(self.layoutWidget_7)
        self.training_data_folder_label.setObjectName(
            "training_data_folder_label"
        )
        self.training_data_folder_label.setEnabled(False)
        self.training_data_folder_label.setMinimumSize(QSize(275, 25))
        self.training_data_folder_label.setMaximumSize(QSize(275, 25))

        self.train_folder_btn_horizontalLayout.addWidget(
            self.training_data_folder_label
        )

        self.generate_training_data_btn = QPushButton(self.layoutWidget_7)
        self.generate_training_data_btn.setObjectName(
            "generate_training_data_btn"
        )
        sizePolicy4.setHeightForWidth(
            self.generate_training_data_btn.sizePolicy().hasHeightForWidth()
        )
        self.generate_training_data_btn.setSizePolicy(sizePolicy4)
        self.generate_training_data_btn.setMinimumSize(QSize(90, 25))
        self.generate_training_data_btn.setMaximumSize(QSize(90, 25))

        self.train_folder_btn_horizontalLayout.addWidget(
            self.generate_training_data_btn,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        self.train_vertical_layout.addWidget(
            self.train_generate_groupbox, 0, Qt.AlignmentFlag.AlignTop
        )

        self.train_train_groupbox = QGroupBox(self.verticalLayoutWidget_4)
        self.train_train_groupbox.setObjectName("train_train_groupbox")
        self.train_train_groupbox.setEnabled(False)
        sizePolicy4.setHeightForWidth(
            self.train_train_groupbox.sizePolicy().hasHeightForWidth()
        )
        self.train_train_groupbox.setSizePolicy(sizePolicy4)
        self.train_train_groupbox.setMinimumSize(QSize(400, 185))
        self.train_train_groupbox.setMaximumSize(QSize(400, 185))
        self.train_train_groupbox.setAlignment(
            Qt.AlignmentFlag.AlignLeading
            | Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.layoutWidget3 = QWidget(self.train_train_groupbox)
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.layoutWidget3.setGeometry(QRect(10, 70, 201, 62))
        self.train_grid_layout = QGridLayout(self.layoutWidget3)
        self.train_grid_layout.setObjectName("train_grid_layout")
        self.train_grid_layout.setContentsMargins(0, 0, 10, 0)
        self.num_epochs_label = QLabel(self.layoutWidget3)
        self.num_epochs_label.setObjectName("num_epochs_label")
        sizePolicy6 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(
            self.num_epochs_label.sizePolicy().hasHeightForWidth()
        )
        self.num_epochs_label.setSizePolicy(sizePolicy6)
        self.num_epochs_label.setMinimumSize(QSize(100, 0))
        self.num_epochs_label.setMaximumSize(QSize(100, 25))

        self.train_grid_layout.addWidget(
            self.num_epochs_label,
            0,
            0,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.save_period_label = QLabel(self.layoutWidget3)
        self.save_period_label.setObjectName("save_period_label")
        sizePolicy6.setHeightForWidth(
            self.save_period_label.sizePolicy().hasHeightForWidth()
        )
        self.save_period_label.setSizePolicy(sizePolicy6)
        self.save_period_label.setMinimumSize(QSize(100, 0))
        self.save_period_label.setMaximumSize(QSize(100, 25))

        self.train_grid_layout.addWidget(
            self.save_period_label,
            1,
            0,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.num_epochs_input = QSpinBox(self.layoutWidget3)
        self.num_epochs_input.setObjectName("num_epochs_input")
        sizePolicy4.setHeightForWidth(
            self.num_epochs_input.sizePolicy().hasHeightForWidth()
        )
        self.num_epochs_input.setSizePolicy(sizePolicy4)
        self.num_epochs_input.setMinimumSize(QSize(80, 25))
        self.num_epochs_input.setMaximumSize(QSize(80, 25))
        self.num_epochs_input.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTrailing
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.num_epochs_input.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.PlusMinus
        )
        self.num_epochs_input.setMinimum(1)
        self.num_epochs_input.setMaximum(900)
        self.num_epochs_input.setSingleStep(10)
        self.num_epochs_input.setValue(250)

        self.train_grid_layout.addWidget(
            self.num_epochs_input, 0, 1, 1, 1, Qt.AlignmentFlag.AlignRight
        )

        self.save_period_input = QSpinBox(self.layoutWidget3)
        self.save_period_input.setObjectName("save_period_input")
        sizePolicy4.setHeightForWidth(
            self.save_period_input.sizePolicy().hasHeightForWidth()
        )
        self.save_period_input.setSizePolicy(sizePolicy4)
        self.save_period_input.setMinimumSize(QSize(80, 25))
        self.save_period_input.setMaximumSize(QSize(80, 25))
        self.save_period_input.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTrailing
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.save_period_input.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.PlusMinus
        )
        self.save_period_input.setMinimum(2)
        self.save_period_input.setMaximum(100)
        self.save_period_input.setSingleStep(15)
        self.save_period_input.setValue(50)

        self.train_grid_layout.addWidget(
            self.save_period_input, 1, 1, 1, 1, Qt.AlignmentFlag.AlignRight
        )

        self.layoutWidget4 = QWidget(self.train_train_groupbox)
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.layoutWidget4.setGeometry(QRect(300, 30, 90, 81))
        self.train_verticalLayout = QVBoxLayout(self.layoutWidget4)
        self.train_verticalLayout.setSpacing(10)
        self.train_verticalLayout.setObjectName("train_verticalLayout")
        self.train_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.train_resume_checkBox = QCheckBox(self.layoutWidget4)
        self.train_resume_checkBox.setObjectName("train_resume_checkBox")
        self.train_resume_checkBox.setEnabled(False)
        self.train_resume_checkBox.setMinimumSize(QSize(90, 25))
        self.train_resume_checkBox.setMaximumSize(QSize(90, 25))

        self.train_verticalLayout.addWidget(
            self.train_resume_checkBox,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self.train_training_overwrite_checkBox = QCheckBox(self.layoutWidget4)
        self.train_training_overwrite_checkBox.setObjectName(
            "train_training_overwrite_checkBox"
        )
        self.train_training_overwrite_checkBox.setEnabled(False)
        self.train_training_overwrite_checkBox.setMinimumSize(QSize(90, 25))
        self.train_training_overwrite_checkBox.setMaximumSize(QSize(90, 25))
        self.train_training_overwrite_checkBox.setChecked(True)

        self.train_verticalLayout.addWidget(
            self.train_training_overwrite_checkBox,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self.launch_tensorboard_checkBox = QCheckBox(self.layoutWidget4)
        self.launch_tensorboard_checkBox.setObjectName(
            "launch_tensorboard_checkBox"
        )
        self.launch_tensorboard_checkBox.setEnabled(False)
        self.launch_tensorboard_checkBox.setMinimumSize(QSize(90, 25))
        self.launch_tensorboard_checkBox.setMaximumSize(QSize(90, 25))
        self.launch_tensorboard_checkBox.setChecked(True)

        self.train_verticalLayout.addWidget(self.launch_tensorboard_checkBox)

        self.layoutWidget5 = QWidget(self.train_train_groupbox)
        self.layoutWidget5.setObjectName("layoutWidget5")
        self.layoutWidget5.setGeometry(QRect(10, 30, 281, 31))
        self.model_choose_horizontalLayout = QHBoxLayout(self.layoutWidget5)
        self.model_choose_horizontalLayout.setObjectName(
            "model_choose_horizontalLayout"
        )
        self.model_choose_horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.yolomodel_list = QComboBox(self.layoutWidget5)
        self.yolomodel_list.addItem("")
        self.yolomodel_list.setObjectName("yolomodel_list")
        self.yolomodel_list.setMinimumSize(QSize(150, 25))
        self.yolomodel_list.setMaximumSize(QSize(150, 25))

        self.model_choose_horizontalLayout.addWidget(
            self.yolomodel_list,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.yoloimagesize_list = QComboBox(self.layoutWidget5)
        self.yoloimagesize_list.addItem("")
        self.yoloimagesize_list.addItem("")
        self.yoloimagesize_list.addItem("")
        self.yoloimagesize_list.setObjectName("yoloimagesize_list")
        self.yoloimagesize_list.setMinimumSize(QSize(100, 25))
        self.yoloimagesize_list.setMaximumSize(QSize(100, 25))
        self.yoloimagesize_list.setEditable(True)

        self.model_choose_horizontalLayout.addWidget(
            self.yoloimagesize_list,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.layoutWidget6 = QWidget(self.train_train_groupbox)
        self.layoutWidget6.setObjectName("layoutWidget6")
        self.layoutWidget6.setGeometry(QRect(10, 140, 380, 37))
        self.epochs_horizontalLayout = QHBoxLayout(self.layoutWidget6)
        self.epochs_horizontalLayout.setObjectName("epochs_horizontalLayout")
        self.epochs_horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.train_epochs_progressbar = QProgressBar(self.layoutWidget6)
        self.train_epochs_progressbar.setObjectName("train_epochs_progressbar")
        self.train_epochs_progressbar.setEnabled(False)
        sizePolicy5.setHeightForWidth(
            self.train_epochs_progressbar.sizePolicy().hasHeightForWidth()
        )
        self.train_epochs_progressbar.setSizePolicy(sizePolicy5)
        self.train_epochs_progressbar.setMinimumSize(QSize(120, 25))
        self.train_epochs_progressbar.setMaximumSize(QSize(120, 25))
        self.train_epochs_progressbar.setMaximum(20)
        self.train_epochs_progressbar.setValue(0)

        self.epochs_horizontalLayout.addWidget(
            self.train_epochs_progressbar,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.train_finishtime_label = QLabel(self.layoutWidget6)
        self.train_finishtime_label.setObjectName("train_finishtime_label")
        self.train_finishtime_label.setEnabled(False)
        self.train_finishtime_label.setMinimumSize(QSize(150, 25))
        self.train_finishtime_label.setMaximumSize(QSize(150, 25))

        self.epochs_horizontalLayout.addWidget(
            self.train_finishtime_label,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.start_stop_training_btn = QPushButton(self.layoutWidget6)
        self.start_stop_training_btn.setObjectName("start_stop_training_btn")
        sizePolicy4.setHeightForWidth(
            self.start_stop_training_btn.sizePolicy().hasHeightForWidth()
        )
        self.start_stop_training_btn.setSizePolicy(sizePolicy4)
        self.start_stop_training_btn.setMinimumSize(QSize(90, 25))
        self.start_stop_training_btn.setMaximumSize(QSize(90, 25))

        self.epochs_horizontalLayout.addWidget(
            self.start_stop_training_btn,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        self.train_vertical_layout.addWidget(
            self.train_train_groupbox, 0, Qt.AlignmentFlag.AlignTop
        )

        icon2 = QIcon()
        icon2.addFile(
            "icons/noun-rocket-7158872.svg", QSize(), QIcon.Normal, QIcon.Off
        )
        self.main_toolbox.addItem(self.train_tab, icon2, "Train model")
        self.predict_tab = QWidget()
        self.predict_tab.setObjectName("predict_tab")
        self.predict_tab.setGeometry(QRect(0, 0, 410, 464))
        sizePolicy1.setHeightForWidth(
            self.predict_tab.sizePolicy().hasHeightForWidth()
        )
        self.predict_tab.setSizePolicy(sizePolicy1)
        self.verticalLayoutWidget_5 = QWidget(self.predict_tab)
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayoutWidget_5.setGeometry(QRect(0, 0, 402, 409))
        self.predict_verticalLayout = QVBoxLayout(self.verticalLayoutWidget_5)
        self.predict_verticalLayout.setSpacing(20)
        self.predict_verticalLayout.setObjectName("predict_verticalLayout")
        self.predict_verticalLayout.setContentsMargins(0, 0, 0, 10)
        self.predict_video_drop_groupbox = QGroupBox(
            self.verticalLayoutWidget_5
        )
        self.predict_video_drop_groupbox.setObjectName(
            "predict_video_drop_groupbox"
        )
        self.predict_video_drop_groupbox.setEnabled(True)
        sizePolicy2.setHeightForWidth(
            self.predict_video_drop_groupbox.sizePolicy().hasHeightForWidth()
        )
        self.predict_video_drop_groupbox.setSizePolicy(sizePolicy2)
        self.predict_video_drop_groupbox.setMinimumSize(QSize(400, 100))
        self.predict_video_drop_groupbox.setMaximumSize(QSize(400, 100))
        self.horizontalLayout_3 = QHBoxLayout(self.predict_video_drop_groupbox)
        self.horizontalLayout_3.setSpacing(20)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(9, 9, 9, 9)
        self.predict_video_drop_widget = QWidget(
            self.predict_video_drop_groupbox
        )
        self.predict_video_drop_widget.setObjectName(
            "predict_video_drop_widget"
        )
        self.predict_video_drop_widget.setMinimumSize(QSize(380, 60))
        self.predict_video_drop_widget.setMaximumSize(QSize(380, 60))

        self.horizontalLayout_3.addWidget(self.predict_video_drop_widget)

        self.predict_verticalLayout.addWidget(self.predict_video_drop_groupbox)

        self.predict_video_predict_groupbox = QGroupBox(
            self.verticalLayoutWidget_5
        )
        self.predict_video_predict_groupbox.setObjectName(
            "predict_video_predict_groupbox"
        )
        self.predict_video_predict_groupbox.setEnabled(True)
        sizePolicy2.setHeightForWidth(
            self.predict_video_predict_groupbox.sizePolicy().hasHeightForWidth()
        )
        self.predict_video_predict_groupbox.setSizePolicy(sizePolicy2)
        self.predict_video_predict_groupbox.setMinimumSize(QSize(400, 280))
        self.predict_video_predict_groupbox.setMaximumSize(QSize(400, 280))
        self.layoutWidget_2 = QWidget(self.predict_video_predict_groupbox)
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.layoutWidget_2.setGeometry(QRect(10, 210, 382, 29))
        self.predict_progress_bar_layout = QHBoxLayout(self.layoutWidget_2)
        self.predict_progress_bar_layout.setObjectName(
            "predict_progress_bar_layout"
        )
        self.predict_progress_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.predict_overall_progressbar = QProgressBar(self.layoutWidget_2)
        self.predict_overall_progressbar.setObjectName(
            "predict_overall_progressbar"
        )
        self.predict_overall_progressbar.setEnabled(False)
        sizePolicy5.setHeightForWidth(
            self.predict_overall_progressbar.sizePolicy().hasHeightForWidth()
        )
        self.predict_overall_progressbar.setSizePolicy(sizePolicy5)
        self.predict_overall_progressbar.setMinimumSize(QSize(50, 25))
        self.predict_overall_progressbar.setMaximumSize(QSize(50, 25))
        self.predict_overall_progressbar.setMaximum(20)
        self.predict_overall_progressbar.setValue(0)

        self.predict_progress_bar_layout.addWidget(
            self.predict_overall_progressbar,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.predict_current_video_progressbar = QProgressBar(
            self.layoutWidget_2
        )
        self.predict_current_video_progressbar.setObjectName(
            "predict_current_video_progressbar"
        )
        self.predict_current_video_progressbar.setEnabled(False)
        sizePolicy5.setHeightForWidth(
            self.predict_current_video_progressbar.sizePolicy().hasHeightForWidth()
        )
        self.predict_current_video_progressbar.setSizePolicy(sizePolicy5)
        self.predict_current_video_progressbar.setMinimumSize(QSize(120, 25))
        self.predict_current_video_progressbar.setMaximumSize(QSize(120, 25))
        self.predict_current_video_progressbar.setMaximum(20)
        self.predict_current_video_progressbar.setValue(0)

        self.predict_progress_bar_layout.addWidget(
            self.predict_current_video_progressbar,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.predict_current_videoname_label = QLabel(self.layoutWidget_2)
        self.predict_current_videoname_label.setObjectName(
            "predict_current_videoname_label"
        )
        self.predict_current_videoname_label.setEnabled(False)
        self.predict_current_videoname_label.setMinimumSize(QSize(188, 25))
        self.predict_current_videoname_label.setMaximumSize(QSize(188, 25))

        self.predict_progress_bar_layout.addWidget(
            self.predict_current_videoname_label
        )

        self.layoutWidget_3 = QWidget(self.predict_video_predict_groupbox)
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.layoutWidget_3.setGeometry(QRect(10, 240, 381, 37))
        self.predict_finish_time_layout = QHBoxLayout(self.layoutWidget_3)
        self.predict_finish_time_layout.setObjectName(
            "predict_finish_time_layout"
        )
        self.predict_finish_time_layout.setContentsMargins(0, 0, 1, 0)
        self.predict_finish_time_label = QLabel(self.layoutWidget_3)
        self.predict_finish_time_label.setObjectName(
            "predict_finish_time_label"
        )
        self.predict_finish_time_label.setEnabled(False)
        self.predict_finish_time_label.setMinimumSize(QSize(0, 25))
        self.predict_finish_time_label.setMaximumSize(QSize(16777215, 25))

        self.predict_finish_time_layout.addWidget(
            self.predict_finish_time_label
        )

        self.predict_start_btn = QPushButton(self.layoutWidget_3)
        self.predict_start_btn.setObjectName("predict_start_btn")
        self.predict_start_btn.setMinimumSize(QSize(90, 25))
        self.predict_start_btn.setMaximumSize(QSize(90, 25))

        self.predict_finish_time_layout.addWidget(self.predict_start_btn)

        self.layoutWidget7 = QWidget(self.predict_video_predict_groupbox)
        self.layoutWidget7.setObjectName("layoutWidget7")
        self.layoutWidget7.setGeometry(QRect(10, 140, 381, 61))
        self.predict_options_grid_layout = QGridLayout(self.layoutWidget7)
        self.predict_options_grid_layout.setObjectName(
            "predict_options_grid_layout"
        )
        self.predict_options_grid_layout.setHorizontalSpacing(5)
        self.predict_options_grid_layout.setVerticalSpacing(0)
        self.predict_options_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.prediction_mask_opening_label = QLabel(self.layoutWidget7)
        self.prediction_mask_opening_label.setObjectName(
            "prediction_mask_opening_label"
        )
        sizePolicy6.setHeightForWidth(
            self.prediction_mask_opening_label.sizePolicy().hasHeightForWidth()
        )
        self.prediction_mask_opening_label.setSizePolicy(sizePolicy6)
        self.prediction_mask_opening_label.setMinimumSize(QSize(75, 0))
        self.prediction_mask_opening_label.setMaximumSize(QSize(40, 25))

        self.predict_options_grid_layout.addWidget(
            self.prediction_mask_opening_label,
            0,
            0,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.predict_mask_opening_spinbox = QDoubleSpinBox(self.layoutWidget7)
        self.predict_mask_opening_spinbox.setObjectName(
            "predict_mask_opening_spinbox"
        )
        self.predict_mask_opening_spinbox.setMinimumSize(QSize(70, 25))
        self.predict_mask_opening_spinbox.setMaximumSize(QSize(70, 25))
        self.predict_mask_opening_spinbox.setDecimals(1)
        self.predict_mask_opening_spinbox.setMaximum(5.000000000000000)
        self.predict_mask_opening_spinbox.setSingleStep(0.250000000000000)
        self.predict_mask_opening_spinbox.setValue(0.000000000000000)

        self.predict_options_grid_layout.addWidget(
            self.predict_mask_opening_spinbox, 0, 1, 1, 1
        )

        self.prediction_iou_label = QLabel(self.layoutWidget7)
        self.prediction_iou_label.setObjectName("prediction_iou_label")
        sizePolicy6.setHeightForWidth(
            self.prediction_iou_label.sizePolicy().hasHeightForWidth()
        )
        self.prediction_iou_label.setSizePolicy(sizePolicy6)
        self.prediction_iou_label.setMinimumSize(QSize(75, 0))
        self.prediction_iou_label.setMaximumSize(QSize(75, 25))
        self.prediction_iou_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTrailing
            | Qt.AlignmentFlag.AlignVCenter
        )

        self.predict_options_grid_layout.addWidget(
            self.prediction_iou_label, 0, 2, 1, 1, Qt.AlignmentFlag.AlignRight
        )

        self.predict_iou_thresh_spinbox = QDoubleSpinBox(self.layoutWidget7)
        self.predict_iou_thresh_spinbox.setObjectName(
            "predict_iou_thresh_spinbox"
        )
        self.predict_iou_thresh_spinbox.setMinimumSize(QSize(70, 25))
        self.predict_iou_thresh_spinbox.setMaximumSize(QSize(70, 25))
        self.predict_iou_thresh_spinbox.setMaximum(1.000000000000000)
        self.predict_iou_thresh_spinbox.setSingleStep(0.100000000000000)
        self.predict_iou_thresh_spinbox.setValue(0.300000000000000)

        self.predict_options_grid_layout.addWidget(
            self.predict_iou_thresh_spinbox, 0, 3, 1, 1
        )

        self.prediction_conf_thresh_label = QLabel(self.layoutWidget7)
        self.prediction_conf_thresh_label.setObjectName(
            "prediction_conf_thresh_label"
        )
        sizePolicy6.setHeightForWidth(
            self.prediction_conf_thresh_label.sizePolicy().hasHeightForWidth()
        )
        self.prediction_conf_thresh_label.setSizePolicy(sizePolicy6)
        self.prediction_conf_thresh_label.setMinimumSize(QSize(75, 0))
        self.prediction_conf_thresh_label.setMaximumSize(QSize(40, 25))

        self.predict_options_grid_layout.addWidget(
            self.prediction_conf_thresh_label,
            1,
            0,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.predict_conf_thresh_spinbox = QDoubleSpinBox(self.layoutWidget7)
        self.predict_conf_thresh_spinbox.setObjectName(
            "predict_conf_thresh_spinbox"
        )
        self.predict_conf_thresh_spinbox.setMinimumSize(QSize(70, 25))
        self.predict_conf_thresh_spinbox.setMaximumSize(QSize(70, 25))
        self.predict_conf_thresh_spinbox.setMaximum(1.000000000000000)
        self.predict_conf_thresh_spinbox.setSingleStep(0.050000000000000)
        self.predict_conf_thresh_spinbox.setValue(0.500000000000000)

        self.predict_options_grid_layout.addWidget(
            self.predict_conf_thresh_spinbox, 1, 1, 1, 1
        )

        self.prediction_skip_label = QLabel(self.layoutWidget7)
        self.prediction_skip_label.setObjectName("prediction_skip_label")
        sizePolicy6.setHeightForWidth(
            self.prediction_skip_label.sizePolicy().hasHeightForWidth()
        )
        self.prediction_skip_label.setSizePolicy(sizePolicy6)
        self.prediction_skip_label.setMinimumSize(QSize(75, 25))
        self.prediction_skip_label.setMaximumSize(QSize(75, 25))
        self.prediction_skip_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignTrailing
            | Qt.AlignmentFlag.AlignVCenter
        )

        self.predict_options_grid_layout.addWidget(
            self.prediction_skip_label, 1, 2, 1, 1, Qt.AlignmentFlag.AlignRight
        )

        self.skip_frames_analysis_spinBox = QSpinBox(self.layoutWidget7)
        self.skip_frames_analysis_spinBox.setObjectName(
            "skip_frames_analysis_spinBox"
        )
        self.skip_frames_analysis_spinBox.setMinimumSize(QSize(70, 25))
        self.skip_frames_analysis_spinBox.setMaximumSize(QSize(70, 25))
        self.skip_frames_analysis_spinBox.setMaximum(1000)

        self.predict_options_grid_layout.addWidget(
            self.skip_frames_analysis_spinBox, 1, 3, 1, 1
        )

        self.layoutWidget8 = QWidget(self.predict_video_predict_groupbox)
        self.layoutWidget8.setObjectName("layoutWidget8")
        self.layoutWidget8.setGeometry(QRect(10, 31, 381, 101))
        self.predict_grid_layout = QGridLayout(self.layoutWidget8)
        self.predict_grid_layout.setObjectName("predict_grid_layout")
        self.predict_grid_layout.setHorizontalSpacing(5)
        self.predict_grid_layout.setVerticalSpacing(0)
        self.predict_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.detailed_extraction_checkBox = QCheckBox(self.layoutWidget8)
        self.detailed_extraction_checkBox.setObjectName(
            "detailed_extraction_checkBox"
        )
        self.detailed_extraction_checkBox.setEnabled(True)
        self.detailed_extraction_checkBox.setMinimumSize(QSize(90, 25))
        self.detailed_extraction_checkBox.setMaximumSize(QSize(100, 25))
        self.detailed_extraction_checkBox.setChecked(False)

        self.predict_grid_layout.addWidget(
            self.detailed_extraction_checkBox,
            2,
            3,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.tune_tracker_btn = QPushButton(self.layoutWidget8)
        self.tune_tracker_btn.setObjectName("tune_tracker_btn")
        self.tune_tracker_btn.setEnabled(False)
        sizePolicy4.setHeightForWidth(
            self.tune_tracker_btn.sizePolicy().hasHeightForWidth()
        )
        self.tune_tracker_btn.setSizePolicy(sizePolicy4)
        self.tune_tracker_btn.setMinimumSize(QSize(50, 25))
        self.tune_tracker_btn.setMaximumSize(QSize(50, 25))
        self.tune_tracker_btn.setBaseSize(QSize(50, 25))

        self.predict_grid_layout.addWidget(
            self.tune_tracker_btn,
            0,
            2,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.yolomodel_tracker_list = QComboBox(self.layoutWidget8)
        self.yolomodel_tracker_list.addItem("")
        self.yolomodel_tracker_list.setObjectName("yolomodel_tracker_list")
        self.yolomodel_tracker_list.setMinimumSize(QSize(110, 25))
        self.yolomodel_tracker_list.setMaximumSize(QSize(110, 25))
        self.yolomodel_tracker_list.setMaxCount(20)
        self.yolomodel_tracker_list.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.yolomodel_tracker_list.setMinimumContentsLength(20)

        self.predict_grid_layout.addWidget(
            self.yolomodel_tracker_list,
            0,
            1,
            1,
            1,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        self.overwrite_prediction_checkBox = QCheckBox(self.layoutWidget8)
        self.overwrite_prediction_checkBox.setObjectName(
            "overwrite_prediction_checkBox"
        )
        self.overwrite_prediction_checkBox.setEnabled(True)
        self.overwrite_prediction_checkBox.setMinimumSize(QSize(90, 25))
        self.overwrite_prediction_checkBox.setMaximumSize(QSize(100, 25))
        self.overwrite_prediction_checkBox.setChecked(False)

        self.predict_grid_layout.addWidget(
            self.overwrite_prediction_checkBox,
            3,
            3,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.open_when_finish_checkBox = QCheckBox(self.layoutWidget8)
        self.open_when_finish_checkBox.setObjectName(
            "open_when_finish_checkBox"
        )
        self.open_when_finish_checkBox.setMinimumSize(QSize(90, 25))
        self.open_when_finish_checkBox.setMaximumSize(QSize(100, 25))
        self.open_when_finish_checkBox.setChecked(True)

        self.predict_grid_layout.addWidget(
            self.open_when_finish_checkBox,
            0,
            3,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.videos_for_prediction_list = QComboBox(self.layoutWidget8)
        self.videos_for_prediction_list.addItem("")
        self.videos_for_prediction_list.addItem("")
        self.videos_for_prediction_list.setObjectName(
            "videos_for_prediction_list"
        )
        self.videos_for_prediction_list.setMinimumSize(QSize(280, 25))
        self.videos_for_prediction_list.setMaximumSize(QSize(280, 25))
        self.videos_for_prediction_list.setEditable(False)
        self.videos_for_prediction_list.setMaxVisibleItems(15)
        self.videos_for_prediction_list.setMaxCount(1000)
        self.videos_for_prediction_list.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.videos_for_prediction_list.setIconSize(QSize(14, 14))
        self.videos_for_prediction_list.setFrame(False)

        self.predict_grid_layout.addWidget(
            self.videos_for_prediction_list,
            3,
            0,
            1,
            3,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self.yolomodel_trained_list = QComboBox(self.layoutWidget8)
        self.yolomodel_trained_list.addItem("")
        self.yolomodel_trained_list.setObjectName("yolomodel_trained_list")
        self.yolomodel_trained_list.setEnabled(True)
        self.yolomodel_trained_list.setMinimumSize(QSize(110, 25))
        self.yolomodel_trained_list.setMaximumSize(QSize(110, 25))

        self.predict_grid_layout.addWidget(
            self.yolomodel_trained_list,
            0,
            0,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.single_subject_checkBox = QCheckBox(self.layoutWidget8)
        self.single_subject_checkBox.setObjectName("single_subject_checkBox")
        self.single_subject_checkBox.setEnabled(True)
        self.single_subject_checkBox.setMinimumSize(QSize(90, 25))
        self.single_subject_checkBox.setMaximumSize(QSize(100, 25))
        self.single_subject_checkBox.setChecked(False)

        self.predict_grid_layout.addWidget(
            self.single_subject_checkBox,
            1,
            3,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.predict_verticalLayout.addWidget(
            self.predict_video_predict_groupbox
        )

        icon3 = QIcon()
        icon3.addFile(
            "icons/noun-conversion-7158876.svg",
            QSize(),
            QIcon.Normal,
            QIcon.Off,
        )
        self.main_toolbox.addItem(
            self.predict_tab, icon3, "Analyze (new) videos"
        )

        self.mainLayout.addWidget(self.main_toolbox)

        self.main_toolbox.raise_()
        self.octron_logo.raise_()

        self.retranslateUi(octron_widgetui)

        self.main_toolbox.setCurrentIndex(0)
        self.main_toolbox.layout().setSpacing(10)

        QMetaObject.connectSlotsByName(octron_widgetui)

    # setupUi

    def retranslateUi(self, octron_widgetui):
        octron_widgetui.setWindowTitle(
            QCoreApplication.translate("octron_widgetui", "octron_gui", None)
        )
        self.octron_logo.setText("")
        self.folder_sect_groupbox.setTitle(
            QCoreApplication.translate(
                "octron_widgetui", "Project folder", None
            )
        )
        self.create_project_btn.setText(
            QCoreApplication.translate(
                "octron_widgetui", "\u2295 Choose", None
            )
        )
        self.project_folder_path_label.setText(
            QCoreApplication.translate(
                "octron_widgetui", "Project folder path", None
            )
        )
        self.project_video_drop_groupbox.setTitle(
            QCoreApplication.translate(
                "octron_widgetui", "Add new video file", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.video_file_drop_widget.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui", "Drag and drop one .mp4 file here", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.project_existing_data_groupbox.setTitle(
            QCoreApplication.translate(
                "octron_widgetui", "Existing data", None
            )
        )
        self.main_toolbox.setItemText(
            self.main_toolbox.indexOf(self.project_tab),
            QCoreApplication.translate(
                "octron_widgetui", "Manage project", None
            ),
        )
        # if QT_CONFIG(tooltip)
        self.main_toolbox.setItemToolTip(
            self.main_toolbox.indexOf(self.project_tab),
            QCoreApplication.translate(
                "octron_widgetui",
                "Create new octron projects or load existing ones",
                None,
            ),
        )
        # endif // QT_CONFIG(tooltip)
        self.model_select_groupbox.setTitle(
            QCoreApplication.translate(
                "octron_widgetui", "Model selection", None
            )
        )
        self.threshold_label.setText(
            QCoreApplication.translate("octron_widgetui", "Thresh.", None)
        )
        self.load_sam_model_btn.setText(
            QCoreApplication.translate("octron_widgetui", "Load model", None)
        )
        # if QT_CONFIG(tooltip)
        self.sam3detect_thresh.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "SAM3 multi only: Detection threshold for objects (0-1). Default is 0.5.",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.sam3detect_thresh.setPlaceholderText(
            QCoreApplication.translate("octron_widgetui", "0.5", None)
        )
        self.sam_model_list.setItemText(
            0, QCoreApplication.translate("octron_widgetui", "Model", None)
        )

        # if QT_CONFIG(tooltip)
        self.sam_model_list.setToolTip(
            QCoreApplication.translate("octron_widgetui", "SAM models", None)
        )
        # endif // QT_CONFIG(tooltip)
        self.sam_model_list.setCurrentText(
            QCoreApplication.translate("octron_widgetui", "Model", None)
        )
        # if QT_CONFIG(tooltip)
        self.feed_input_to_predictor_btn.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Predict with SAM3 multi on bounding boxes you drew.",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.feed_input_to_predictor_btn.setText("")
        self.annotate_layer_create_groupbox.setTitle(
            QCoreApplication.translate(
                "octron_widgetui", "Label manager", None
            )
        )
        self.layer_type_combobox.setItemText(
            0, QCoreApplication.translate("octron_widgetui", "Type ... ", None)
        )
        self.layer_type_combobox.setItemText(
            1, QCoreApplication.translate("octron_widgetui", "Shapes", None)
        )
        self.layer_type_combobox.setItemText(
            2, QCoreApplication.translate("octron_widgetui", "Points", None)
        )
        self.layer_type_combobox.setItemText(
            3, QCoreApplication.translate("octron_widgetui", "Anchors", None)
        )

        # if QT_CONFIG(tooltip)
        self.layer_type_combobox.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui", "Layer type to be created", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.layer_type_combobox.setCurrentText(
            QCoreApplication.translate("octron_widgetui", "Type ... ", None)
        )
        self.label_list_combobox.setItemText(
            0,
            QCoreApplication.translate("octron_widgetui", "Label ... ", None),
        )
        self.label_list_combobox.setItemText(
            1,
            QCoreApplication.translate(
                "octron_widgetui", "\u2295 Create", None
            ),
        )
        self.label_list_combobox.setItemText(
            2,
            QCoreApplication.translate(
                "octron_widgetui", "\u2296 Remove", None
            ),
        )

        # if QT_CONFIG(tooltip)
        self.label_list_combobox.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui", "Select, add or remove labels", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.label_list_combobox.setCurrentText(
            QCoreApplication.translate("octron_widgetui", "Label ... ", None)
        )
        # if QT_CONFIG(tooltip)
        self.label_suffix_lineedit.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "The suffix disambiguates label layers from each other\n"
                "that have the same label name.\n"
                "For example:\n"
                "The label could be octo and suffix 1 for the first octopus,\n"
                "and octo and suffix 2 for the second octo ",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.label_suffix_lineedit.setPlaceholderText(
            QCoreApplication.translate("octron_widgetui", "Suffix", None)
        )
        self.create_annotation_layer_btn.setText(
            QCoreApplication.translate(
                "octron_widgetui", "\u2295 Create", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.create_projection_layer_btn.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Create an average projection out of all segmented images for the current label",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(statustip)
        self.create_projection_layer_btn.setStatusTip("")
        # endif // QT_CONFIG(statustip)
        self.create_projection_layer_btn.setText(
            QCoreApplication.translate(
                "octron_widgetui", "Visualize all", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.hard_reset_layer_btn.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Hard reset of the predictor.\n"
                "Use this if prediction is not working well for you.\n"
                "This will delete the masks on the current frame but no other (already annotated) frame.",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.hard_reset_layer_btn.setText(
            QCoreApplication.translate("octron_widgetui", "\u3004 Reset", None)
        )
        self.annotate_layer_timeline_groupbox.setTitle(
            QCoreApplication.translate(
                "octron_widgetui", "Timeline control", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.annotation_jump_previous_btn.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui", "Jump to last annotated frame", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.annotation_jump_previous_btn.setText(
            QCoreApplication.translate(
                "octron_widgetui", "\u226a Jump to previous", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.annotation_jump_next_btn.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui", "Jump to next annotated frame", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.annotation_jump_next_btn.setText(
            QCoreApplication.translate(
                "octron_widgetui", "Jump to next \u226b", None
            )
        )
        self.annotate_layer_predict_groupbox.setTitle(
            QCoreApplication.translate(
                "octron_widgetui", "Batch prediction", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.batch_predict_progressbar.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "<html><head/><body><p>Batch predict progress bar</p></body></html>",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.batch_predict_progressbar.setFormat(
            QCoreApplication.translate("octron_widgetui", "%p%", None)
        )
        self.skip_label.setText(
            QCoreApplication.translate("octron_widgetui", "Skip", None)
        )
        # if QT_CONFIG(tooltip)
        self.skip_frames_spinbox.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "How many frames should be skipped during batch prediction?",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.skip_frames_spinbox.setSuffix("")
        self.skip_frames_spinbox.setPrefix("")
        # if QT_CONFIG(tooltip)
        self.predict_next_oneframe_btn.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui", "Predict next frame", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.predict_next_oneframe_btn.setText("")
        # if QT_CONFIG(tooltip)
        self.predict_next_batch_btn.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui", "Predict batch of next frames", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.predict_next_batch_btn.setText("")
        self.main_toolbox.setItemText(
            self.main_toolbox.indexOf(self.annotate_tab),
            QCoreApplication.translate(
                "octron_widgetui", "Generate annotation data", None
            ),
        )
        # if QT_CONFIG(tooltip)
        self.main_toolbox.setItemToolTip(
            self.main_toolbox.indexOf(self.annotate_tab),
            QCoreApplication.translate(
                "octron_widgetui",
                "Create annotation data for training your custom models",
                None,
            ),
        )
        # endif // QT_CONFIG(tooltip)
        self.segmentation_bbox_decision_groupbox.setTitle(
            QCoreApplication.translate(
                "octron_widgetui", "Segmentation or Detection?", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.segmentation_radiobutton.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Click this, if you want to train full segmentation models\n"
                "that yield both masks and bounding boxes.",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.segmentation_radiobutton.setText(
            QCoreApplication.translate(
                "octron_widgetui", "Masks and bboxes (Segmentation)", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.detection_radiobutton.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Click this, if you want to train (lightweight) detection models only\n"
                "that yield bboxes, but NO masks.",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.detection_radiobutton.setText(
            QCoreApplication.translate(
                "octron_widgetui", "Bboxes (Detection)", None
            )
        )
        self.train_generate_groupbox.setTitle(
            QCoreApplication.translate(
                "octron_widgetui", "Generate training data", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.train_polygons_overall_progressbar.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "<html><head/><body><p>Batch predict progress bar</p></body></html>",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.train_polygons_overall_progressbar.setFormat(
            QCoreApplication.translate("octron_widgetui", "%p%", None)
        )
        # if QT_CONFIG(tooltip)
        self.train_polygons_frames_progressbar.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "<html><head/><body><p>Batch predict progress bar</p></body></html>",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.train_polygons_frames_progressbar.setFormat(
            QCoreApplication.translate("octron_widgetui", "%p%", None)
        )
        self.train_polygons_label.setText(
            QCoreApplication.translate("octron_widgetui", "label", None)
        )
        # if QT_CONFIG(tooltip)
        self.train_export_overall_progressbar.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "<html><head/><body><p>Batch predict progress bar</p></body></html>",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.train_export_overall_progressbar.setFormat(
            QCoreApplication.translate("octron_widgetui", "%p%", None)
        )
        # if QT_CONFIG(tooltip)
        self.train_export_frames_progressbar.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "<html><head/><body><p>Batch predict progress bar</p></body></html>",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.train_export_frames_progressbar.setFormat(
            QCoreApplication.translate("octron_widgetui", "%p%", None)
        )
        self.train_export_label.setText(
            QCoreApplication.translate(
                "octron_widgetui", "label and split", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.train_prune_checkBox.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Exclude frames in which only a subset of all\n"
                "labels is present.",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.train_prune_checkBox.setText(
            QCoreApplication.translate("octron_widgetui", "Prune", None)
        )
        # if QT_CONFIG(tooltip)
        self.train_data_watershed_checkBox.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Enable watershed on mask data. This helps to separate masks that \n"
                "are on the same layer and carry the same label assignment,\n"
                "but should be separate entities in the training data.",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.train_data_watershed_checkBox.setText(
            QCoreApplication.translate("octron_widgetui", "Watershed", None)
        )
        self.train_data_overwrite_checkBox.setText(
            QCoreApplication.translate("octron_widgetui", "Overwrite", None)
        )
        self.training_data_folder_label.setText("")
        self.generate_training_data_btn.setText(
            QCoreApplication.translate("octron_widgetui", "Generate", None)
        )
        self.train_train_groupbox.setTitle(
            QCoreApplication.translate("octron_widgetui", "Train", None)
        )
        self.num_epochs_label.setText(
            QCoreApplication.translate("octron_widgetui", "Epochs", None)
        )
        self.save_period_label.setText(
            QCoreApplication.translate("octron_widgetui", "Save period", None)
        )
        # if QT_CONFIG(tooltip)
        self.num_epochs_input.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "How many epochs in total\n"
                "should be trained?\n"
                "Recommended are at least ~50.",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(tooltip)
        self.save_period_input.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "After how many epochs should\n"
                "(intermediary) output models be saved?",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.train_resume_checkBox.setText(
            QCoreApplication.translate("octron_widgetui", "Resume", None)
        )
        self.train_training_overwrite_checkBox.setText(
            QCoreApplication.translate("octron_widgetui", "Overwrite", None)
        )
        # if QT_CONFIG(tooltip)
        self.launch_tensorboard_checkBox.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Start tensorboard (open browser window)",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.launch_tensorboard_checkBox.setText(
            QCoreApplication.translate("octron_widgetui", "Tensorbrd", None)
        )
        self.yolomodel_list.setItemText(
            0,
            QCoreApplication.translate(
                "octron_widgetui", "Choose model ...", None
            ),
        )

        # if QT_CONFIG(tooltip)
        self.yolomodel_list.setToolTip(
            QCoreApplication.translate("octron_widgetui", "YOLO models", None)
        )
        # endif // QT_CONFIG(tooltip)
        self.yoloimagesize_list.setItemText(
            0, QCoreApplication.translate("octron_widgetui", "Img. size", None)
        )
        self.yoloimagesize_list.setItemText(
            1, QCoreApplication.translate("octron_widgetui", "640", None)
        )
        self.yoloimagesize_list.setItemText(
            2, QCoreApplication.translate("octron_widgetui", "1024", None)
        )

        # if QT_CONFIG(tooltip)
        self.yoloimagesize_list.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Image size used for training within YOLO",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(tooltip)
        self.train_epochs_progressbar.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "<html><head/><body><p>Batch predict progress bar</p></body></html>",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.train_epochs_progressbar.setFormat(
            QCoreApplication.translate("octron_widgetui", "%p%", None)
        )
        # if QT_CONFIG(tooltip)
        self.train_finishtime_label.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui", "Approximate time of training finish", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.train_finishtime_label.setText(
            QCoreApplication.translate("octron_widgetui", "Finish time", None)
        )
        self.start_stop_training_btn.setText(
            QCoreApplication.translate("octron_widgetui", "Start", None)
        )
        self.main_toolbox.setItemText(
            self.main_toolbox.indexOf(self.train_tab),
            QCoreApplication.translate("octron_widgetui", "Train model", None),
        )
        # if QT_CONFIG(tooltip)
        self.main_toolbox.setItemToolTip(
            self.main_toolbox.indexOf(self.train_tab),
            QCoreApplication.translate(
                "octron_widgetui",
                "Train a new or existing model with generated training data",
                None,
            ),
        )
        # endif // QT_CONFIG(tooltip)
        self.predict_video_drop_groupbox.setTitle(
            QCoreApplication.translate(
                "octron_widgetui", "Add video files", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.predict_video_drop_widget.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui", "Drag and drop .mp4 files here", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.predict_video_predict_groupbox.setTitle(
            QCoreApplication.translate(
                "octron_widgetui", "Create predictions from videos", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.predict_overall_progressbar.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "<html><head/><body><p>Batch predict progress bar</p></body></html>",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.predict_overall_progressbar.setFormat(
            QCoreApplication.translate("octron_widgetui", "%p%", None)
        )
        # if QT_CONFIG(tooltip)
        self.predict_current_video_progressbar.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "<html><head/><body><p>Batch predict progress bar</p></body></html>",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.predict_current_video_progressbar.setFormat(
            QCoreApplication.translate("octron_widgetui", "%p%", None)
        )
        self.predict_current_videoname_label.setText(
            QCoreApplication.translate("octron_widgetui", "video name", None)
        )
        self.predict_finish_time_label.setText(
            QCoreApplication.translate(
                "octron_widgetui", "Current video finishes:", None
            )
        )
        # if QT_CONFIG(tooltip)
        self.predict_start_btn.setToolTip(
            QCoreApplication.translate("octron_widgetui", "Yeah!", None)
        )
        # endif // QT_CONFIG(tooltip)
        self.predict_start_btn.setText(
            QCoreApplication.translate("octron_widgetui", "Let's go!", None)
        )
        # if QT_CONFIG(tooltip)
        self.prediction_mask_opening_label.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Perform morphological opening of predicted masks?\n"
                "Opens regions when > 0.0.\n"
                "This gets rid of some noise in the detected regions, but slows down analysis quite a bit.",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.prediction_mask_opening_label.setText(
            QCoreApplication.translate("octron_widgetui", "Opening", None)
        )
        # if QT_CONFIG(tooltip)
        self.prediction_iou_label.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Intersection over union. This threshold determines how much overlap between bounding boxes\n"
                "is allowed before they are considered to be detecting the same object.\n"
                "At IOU=0 all detected objects > conf. thresh\n"
                "of one label will be fused into one mask.",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.prediction_iou_label.setText(
            QCoreApplication.translate("octron_widgetui", "IOU", None)
        )
        # if QT_CONFIG(tooltip)
        self.prediction_conf_thresh_label.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Confidence threshold for accepting prediction results as real",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.prediction_conf_thresh_label.setText(
            QCoreApplication.translate("octron_widgetui", "Confidence", None)
        )
        # if QT_CONFIG(tooltip)
        self.prediction_skip_label.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Skip frames in videos? 0: All frames are analyzed, >0: This many frames are being skipped. ",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.prediction_skip_label.setText(
            QCoreApplication.translate("octron_widgetui", "Skip frames", None)
        )
        # if QT_CONFIG(tooltip)
        self.detailed_extraction_checkBox.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Extract more info per region than just its coordinates?\n"
                "Click this if you want properties like area, eccentricity, solidity etc. to be determined for each region.\n"
                "This slows down analysis, but gives you much more info per tracked region.",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.detailed_extraction_checkBox.setText(
            QCoreApplication.translate("octron_widgetui", "Detailed", None)
        )
        # if QT_CONFIG(tooltip)
        self.tune_tracker_btn.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui", "Tune parameters of selected tracker", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.tune_tracker_btn.setText("")
        self.yolomodel_tracker_list.setItemText(
            0,
            QCoreApplication.translate("octron_widgetui", "Tracker ...", None),
        )

        # if QT_CONFIG(tooltip)
        self.yolomodel_tracker_list.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui", "BoxMOT trackers", None
            )
        )
        # endif // QT_CONFIG(tooltip)
        # if QT_CONFIG(tooltip)
        self.overwrite_prediction_checkBox.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Overwrite previous analysis results? ",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.overwrite_prediction_checkBox.setText(
            QCoreApplication.translate("octron_widgetui", "Overwrite", None)
        )
        # if QT_CONFIG(tooltip)
        self.open_when_finish_checkBox.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Open the results in new napari window when finished",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.open_when_finish_checkBox.setText(
            QCoreApplication.translate("octron_widgetui", "View result", None)
        )
        self.videos_for_prediction_list.setItemText(
            0,
            QCoreApplication.translate(
                "octron_widgetui", "List of videos to be analyzed ...", None
            ),
        )
        self.videos_for_prediction_list.setItemText(
            1,
            QCoreApplication.translate(
                "octron_widgetui", "\u2296 Remove", None
            ),
        )

        # if QT_CONFIG(tooltip)
        self.videos_for_prediction_list.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "List of videos that will be analyzed, in order of processing.\n"
                'Click on "Remove" in dropdown to remove videos from list.',
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.videos_for_prediction_list.setCurrentText(
            QCoreApplication.translate(
                "octron_widgetui", "List of videos to be analyzed ...", None
            )
        )
        self.yolomodel_trained_list.setItemText(
            0, QCoreApplication.translate("octron_widgetui", "Model ...", None)
        )

        # if QT_CONFIG(tooltip)
        self.yolomodel_trained_list.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "OCTRON user trained models that are found in the project path",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.yolomodel_trained_list.setCurrentText(
            QCoreApplication.translate("octron_widgetui", "Model ...", None)
        )
        # if QT_CONFIG(tooltip)
        self.single_subject_checkBox.setToolTip(
            QCoreApplication.translate(
                "octron_widgetui",
                "Click this if you expect only one subject to be tracked per label.\n"
                "This prevents artificial splitting of tracks.",
                None,
            )
        )
        # endif // QT_CONFIG(tooltip)
        self.single_subject_checkBox.setText(
            QCoreApplication.translate("octron_widgetui", "1 subject", None)
        )
        self.main_toolbox.setItemText(
            self.main_toolbox.indexOf(self.predict_tab),
            QCoreApplication.translate(
                "octron_widgetui", "Analyze (new) videos", None
            ),
        )
        # if QT_CONFIG(tooltip)
        self.main_toolbox.setItemToolTip(
            self.main_toolbox.indexOf(self.predict_tab),
            QCoreApplication.translate(
                "octron_widgetui",
                "Use trained models to run predictions on new videos",
                None,
            ),
        )


# endif // QT_CONFIG(tooltip)
# retranslateUi
