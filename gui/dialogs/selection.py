import sqlite3
from typing import List, Any, Tuple

import pyqtgraph as pg
from PyQt5 import QtCore, uic
from PyQt5.QtCore import QItemSelectionModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QDialog, QCheckBox, QProgressBar

import Paths
from database.connections import Requester
from definitions.icons import Icon
from Util import create_image_item_list_from
from loader import Loader

pg.setConfigOptions(imageAxisOrder='row-major')


class ExperimentSelectionDialog(QDialog):
    """
    Class to enable selection of experiments
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = None
        self.requester = Requester()
        self.check_boxes = []
        self.active_channels = {}
        self.sel_exp = ""
        self.initialize_ui()

    def initialize_ui(self) -> None:
        """
        Method to initialize the ui

        :return: None
        """
        self.setWindowTitle("Experiment Selection")
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.ui = uic.loadUi(Paths.ui_experiment_selection_dial, self)
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint)
        # Load available experiments
        exps = self.requester.get_all_experiments()
        # Add experiments to combo box
        for experiment in [x[0] for x in exps]:
            self.ui.cbx_exp.addItem(experiment)
        self.on_experiment_selection_change(exps[0][0])
        self.ui.cbx_exp.currentTextChanged.connect(self.on_experiment_selection_change)

    def on_experiment_selection_change(self, current_text) -> None:
        """
        Method to react to a changed experiment selection

        :return: None
        """
        # Get the selected experiment
        exp = current_text
        # Load available channels
        channels = self.requester.get_channels_for_experiment(exp)
        # Get main channel
        main = self.requester.get_main_channel_for_experiment(exp)
        # Clean up channels
        channels = [x[0] for x in channels if x[0] != main]
        self.clear_vbox()
        self.active_channels.clear()
        # Define new VBoxLayout
        for channel in channels:
            # Define checkbox
            cbx_temp = QCheckBox(channel)
            cbx_temp.setStyleSheet("QCheckBox {color: white}")
            cbx_temp.setChecked(True)
            self.ui.vb_channels.addWidget(
                cbx_temp
            )
            self.active_channels[channel] = True
            cbx_temp.stateChanged.connect(self.on_checkbox_change)
            self.check_boxes.append(cbx_temp)
        self.sel_exp = exp

    def on_checkbox_change(self) -> None:
        """
        Method to react to selection changes for checkboxes

        :return: None
        """
        # Get the checkbox whose state was changed
        cbx = self.sender()
        # Change stored information
        self.active_channels[cbx.text()] = cbx.isChecked()

    def clear_vbox(self) -> None:
        """
        Method to remove all checkboxes from the dialog

        :return: None
        """
        for item in self.check_boxes:
            self.ui.vb_channels.removeWidget(item)
        self.check_boxes.clear()


class ImageSelectionDialog(QDialog):
    """
    Dialog to select images from a list
    """

    def __init__(self, images: List[str] = (), selected_images: List[str] = (), *args: Any, **kwargs: Any):
        """
        :param images: A list of paths leading to the images
        :param selected_images: A list of md5 hashes for selected images
        :param args: Positional Arguments
        :param kwargs: Keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.images: List[str] = sorted(images)
        self.selected_images = selected_images
        self.img_model = None
        self.ui = None
        self.prg_bar = None
        # Define timer for lazy image loading
        self.update_timer = None
        # Create index number for loading
        self.loading_index = 0
        self.initialize_ui()

    def initialize_ui(self) -> None:
        """
        Method to initialize the UI

        :return: None
        """
        self.ui = uic.loadUi(Paths.ui_img_sel_dial, self)
        self.prg_bar: QProgressBar = self.ui.prg_bar
        self.img_model = QStandardItemModel()
        self.ui.lv_images.setModel(self.img_model)
        self.ui.lv_images.setIconSize(Icon.get_icon_size("LIST_ITEM"))
        self.update_timer = Loader(self.images, feedback=self.load_images, processing=create_image_item_list_from)
        self.prg_bar.setValue(0)
        self.prg_bar.setMaximum(100)
        self.ui.lv_images.setEnabled(False)
        self.ui.buttonBox.setEnabled(False)
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowTitle("Image Selection Dialog")
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint)

    def load_images(self, items: List[QStandardItem]) -> None:
        """
        Method to load the images given by the list paths

        :param items: QStandardItems to add to the images list
        :return: None
        """
        for img in items:
            self.img_model.appendRow(img)
        self.prg_bar.setValue(self.update_timer.percentage * 100)
        if not items:
            # Enable image list
            self.ui.lv_images.setEnabled(True)
            self.ui.buttonBox.setEnabled(True)
            # Select images that are marked as selected
            for row in range(self.img_model.rowCount()):
                item = self.img_model.item(row)
                # Select marked images
                if item.data()["key"] in self.selected_images:
                    # Create index
                    index = self.img_model.createIndex(row, 0)
                    # Select image
                    self.ui.lv_images.selectionModel().select(index, QItemSelectionModel.Select)

    def get_selected_images(self) -> Tuple[List[str], List[str]]:
        """
        Method to get the selected images as items

        :return: A lis tof all selected images
        """
        data = [], []
        # Get selected indices
        indices = self.ui.lv_images.selectionModel().selectedIndexes()
        for index in indices:
            # Get item
            item = self.img_model.item(index.row())
            data[0].append(item.data()["key"])
            data[1].append(item.data()["path"])
        return data
