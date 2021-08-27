import json
import sqlite3
from typing import List, Any, Tuple, Dict, Union

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, uic
from PyQt5.QtCore import QItemSelection, QItemSelectionModel, Qt, QRectF
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QResizeEvent, QKeyEvent
from PyQt5.QtWidgets import QDialog, QMessageBox, QInputDialog, QCheckBox, QVBoxLayout, QFrame, QScrollArea, QWidget, \
    QLabel, QHBoxLayout, QSizePolicy, QGraphicsEllipseItem, QGraphicsLineItem, \
    QGraphicsItem, QProgressBar

from core.Detector import Detector
from core.roi.ROIHandler import ROIHandler
from gui import Paths, Util
from gui.Definitions import Icon
from gui.GraphicsItems import EditorView
from gui.Plots import PoissonPlotWidget
from gui.Util import create_image_item_list_from
from gui.loader import Loader
from gui.settings.Settings import SettingsShowWidget, SettingsSlider, SettingsDial, SettingsSpinner, \
    SettingsDecimalSpinner, SettingsText, SettingsComboBox, SettingsCheckBox

pg.setConfigOptions(imageAxisOrder='row-major')


class AnalysisSettingsDialog(QDialog):

    def __init__(self, all_=False, settings=None,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = None
        self.all = all_
        self.settings = settings
        self.initialize_ui()

    def get_data(self) -> Dict[str, Union[List, bool]]:
        """
        Method to get the data from the interface

        :return: A dictionary containing the data
        """
        return {
            "type": abs(self.ui.type_btn_group.checkedId()) - 2,
            "re-analyse": self.cbx_reanalyse.isChecked(),
            "add_to_experiment": self.ui.cbx_experiment.isChecked(),
            "experiment_details": {
                "name": self.ui.le_name.text(),
                "details": self.ui.pte_details.toPlainText(),
                "notes": self.ui.pte_notes.toPlainText()
            },
            "names": [
                self.ui.le_one.text(),
                self.ui.le_two.text(),
                self.ui.le_three.text(),
                self.ui.le_four.text(),
                self.ui.le_five.text()
            ],
            "activated": [
                self.ui.cbx_one.isChecked(),
                self.ui.cbx_two.isChecked(),
                self.ui.cbx_three.isChecked(),
                self.ui.cbx_four.isChecked(),
                self.ui.cbx_five.isChecked()
            ],
            "main": abs(self.ui.main_channel_btn_group.checkedId()) - 2
        }

    def initialize_ui(self) -> None:
        """
        Method to initialize the ui

        :return: None
        """
        # Load UI definition
        self.ui = uic.loadUi(Paths.ui_analysis_settings_dial, self)
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowTitle("Analysis Settings")
        # Check if single image analysis or multi image analysis is performed
        if not self.all:
            self.ui.cbx_reanalyse.hide()
            self.ui.lbl_reanalyse.hide()
        # Bind experiment text boxes to experiment checkbox
        self.ui.cbx_experiment.toggled.connect(self.ui.le_name.setEnabled)
        self.ui.cbx_experiment.toggled.connect(self.ui.pte_details.setEnabled)
        self.ui.cbx_experiment.toggled.connect(self.ui.pte_notes.setEnabled)
        channels = [
            self.ui.cbx_one,
            self.ui.cbx_two,
            self.ui.cbx_three,
            self.ui.cbx_four,
            self.ui.cbx_five
        ]
        channel_names = [
            self.ui.le_one,
            self.ui.le_two,
            self.ui.le_three,
            self.ui.le_four,
            self.ui.le_five
        ]
        channel_main = [
            self.ui.rbtn_one,
            self.ui.rbtn_two,
            self.ui.rbtn_three,
            self.ui.rbtn_four,
            self.ui.rbtn_five
        ]
        names = self.settings["names"].split(";")
        for name in range(len(names)):
            channels[name].setChecked(True)
            channel_names[name].setEnabled(True)
            channel_names[name].setText(names[name])
        channel_main[self.settings["main_channel"]].setChecked(True)
        # Bind checkboxes for individual channels to the corresponding text edit
        self.ui.cbx_one.toggled.connect(self.ui.le_one.setEnabled)
        self.ui.cbx_one.toggled.connect(self.ui.rbtn_one.setEnabled)
        self.ui.cbx_two.toggled.connect(self.ui.le_two.setEnabled)
        self.ui.cbx_two.toggled.connect(self.ui.rbtn_two.setEnabled)
        self.ui.cbx_three.toggled.connect(self.ui.le_three.setEnabled)
        self.ui.cbx_three.toggled.connect(self.ui.rbtn_three.setEnabled)
        self.ui.cbx_four.toggled.connect(self.ui.le_four.setEnabled)
        self.ui.cbx_four.toggled.connect(self.ui.rbtn_four.setEnabled)
        self.ui.cbx_five.toggled.connect(self.ui.le_five.setEnabled)
        self.ui.cbx_five.toggled.connect(self.ui.rbtn_five.setEnabled)


class ExperimentDialog(QDialog):

    def __init__(self, data: Dict[str, List[str]] = None, *args, **kwargs):
        """
        :param data: Dict containing the keys and paths of the available images
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.data = data
        self.images = []
        self.img_model = None
        self.exp_model = None
        self.ui = None
        # Image Loader for lazy loading
        self.update_timer = None
        self.initialize_ui()
        # Create connection to database
        self.connection = sqlite3.connect(Paths.database)
        self.cursor = self.connection.cursor()
        self.load_experiments()

    def initialize_ui(self):
        self.ui = uic.loadUi(Paths.ui_exp_dial, self)
        # Define models for used lists
        self.img_model = QStandardItemModel(self.ui.lv_images)
        self.exp_model = QStandardItemModel(self.ui.lv_experiments)
        self.ui.lv_images.setModel(self.img_model)
        self.ui.lv_experiments.setModel(self.exp_model)
        self.ui.lv_images.setIconSize(Icon.get_icon_size("LIST_ITEM"))
        self.ui.lv_experiments.setIconSize(Icon.get_icon_size("LIST_ITEM"))
        # Connect add btn to dialog
        self.ui.btn_add_group.clicked.connect(self.open_group_dialog)
        self.ui.btn_add.clicked.connect(self.add_experiment)
        self.ui.btn_images_add.clicked.connect(self.add_images_to_experiment)
        self.ui.btn_images_remove.clicked.connect(self.remove_images_from_experiment)
        self.ui.btn_images_clear.clicked.connect(self.remove_all_images_from_experiment)
        self.ui.lv_experiments.selectionModel().selectionChanged.connect(self.on_exp_selection_change)
        # Set window title and icon
        self.setWindowTitle("Experiment Dialog")
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint)

    def on_image_selection_change(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        """
        Method to react to changes in the image selection

        :param selected: The selected items
        :param deselected: The deselected items
        :return: None
        """
        if selected:
            self.ui.btn_images_remove.setEnabled(True)
        else:
            self.ui.btn_images_remove.setEnabled(False)

    def add_experiment(self) -> None:
        """
        Method to add a new experiment

        :return: None
        """
        dial = QInputDialog()
        dial.setWindowTitle("Add new Experiment...")
        dial.setWindowIcon(Icon.get_icon("LOGO"))
        dial.setStyleSheet(open("inputbox.css", "r").read())
        name, ok = QInputDialog.getText(dial, "Experiment Dialog", "Enter experiment name: ")
        if ok:
            add_item = QStandardItem()
            text = f"{name}\nNo Details\nGroups: No groups"
            add_item.setText(text)
            add_item.setData(
                {"name": name,
                 "details": "",
                 "notes": "",
                 "groups": {},
                 "keys": [],
                 "image_paths": []}
            )
            add_item.setIcon(Icon.get_icon("CLIPBOARD"))
            self.exp_model.appendRow(add_item)

    def add_images_to_experiment(self) -> None:
        """
        Method to add the selected images to the selected experiment

        :return: None
        """
        # Get selected images
        keys, paths = self.open_image_selection_dialog()
        if keys and paths:
            # Get selected experiment
            selected_exp = self.exp_model.item(self.ui.lv_experiments.selectionModel().selectedIndexes()[0].row())
            data = selected_exp.data()
            # Clear img_model
            self.img_model.clear()
            data["keys"] = keys
            data["image_paths"] = paths
            selected_exp.setData(data)
            self.enable_experiment_buttons(False)
            if len(keys) > 25:
                self.update_timer = Loader(paths,
                                           feedback=self.add_image_items,
                                           processing=create_image_item_list_from)
            else:
                # Get items
                items = create_image_item_list_from(paths)
                for item in items:
                    self.img_model.appendRow(item)
                self.enable_experiment_buttons()

    def accepted(self):
        # Change the information for the last selected experiment
        sel = self.ui.lv_experiments.selectionModel().selectedIndexes()
        if sel:
            item = self.exp_model.itemFromIndex(sel[0])
            data = item.data()
            data["name"] = self.ui.le_name.text()
            data["details"] = self.ui.te_details.toPlainText()
            data["notes"] = self.ui.te_notes.toPlainText()
            item.setData(data)
        # Reset the information of all images
        for key in self.data["keys"]:
            self.cursor.execute(
                "UPDATE images SET experiment=? WHERE md5=?",
                (None, key,)
            )
        for ind in range(self.exp_model.rowCount()):
            item = self.exp_model.item(ind)
            data = item.data()
            # Add experiment to database
            self.cursor.execute(
                "REPLACE INTO experiments VALUES (?, ?, ?)",
                (data["name"], data["details"], data["notes"])
            )
            # Update group data
            for group, values in data["groups"].items():
                for img in values:
                    self.cursor.execute(
                        "REPLACE INTO groups VALUES (?, ?, ?)",
                        (img, data["name"], group)
                    )
            # Update data for images
            for key in data["keys"]:
                self.cursor.execute(
                    "UPDATE images SET experiment=? WHERE md5=?",
                    (data["name"], key)
                )
        self.connection.commit()

    def remove_images_from_experiment(self) -> None:
        """
        Method to remove the selected images from the selected experiment

        :return: None
        """
        # Get selected experiment
        exp = self.exp_model.itemFromIndex(self.ui.lv_experiments.selectionModel().selectedIndexes()[0])
        # Get selected images
        sel_images = self.ui.lv_images.selectionModel().selectedIndexes()
        for index in sel_images:
            # Get key stored in item
            item_data = self.img_model.itemFromIndex(index).data()
            # Remove item from keys
            exp_data = exp.data()
            exp_keys = exp_data["keys"].remove(item_data["key"])
            exp_data["keys"] = exp_keys
            exp.setData(exp_data)
            # Remove item from model
            self.img_model.removeRows(index.row(), 1)

    def remove_all_images_from_experiment(self) -> None:
        """
        Method to remove all assigned images from the selected experiment

        :return: None
        """
        # Remove keys from experiment data
        exp = self.exp_model.itemFromIndex(self.ui.lv_experiments.selectionModel().selectedIndexes()[0])
        exp_data = exp.data()
        exp_data["keys"] = []
        exp.setData(exp_data)
        # Clear image model
        self.img_model.clear()

    def open_image_selection_dialog(self) -> Tuple[List[str], List[str]]:
        """
        Method to open the image selection dialog

        :return: None
        """
        # Get hashes of images in img_model
        image_hashs = []
        for row in range(self.img_model.rowCount()):
            # Get item
            img_item = self.img_model.item(row)
            image_hashs.append(img_item.data()["key"])
        sel_dialog = ImageSelectionDialog(images=self.data["paths"],
                                          selected_images=image_hashs)
        code = sel_dialog.exec()
        if code == QDialog.Accepted:
            return sel_dialog.get_selected_images()
        else:
            return (None, None)

    def load_experiments(self) -> None:
        """
        Method to load all existings experiments

        :return: None
        """
        exps = self.cursor.execute(
            "SELECT * FROM experiments"
        ).fetchall()
        # Iterate over all experiments
        for exp in exps:
            imgs = [x[0] for x in self.cursor.execute("SELECT md5 FROM images WHERE experiment = ?",
                                                      (exp[0],)).fetchall()]
            img_items = []
            # Check if all the necessary images are loaded
            if all(elem in self.data["keys"] for elem in imgs):
                name = exp[0]
                details = exp[1]
                notes = exp[2]
                groups = {}
                group_str = ""
                # Get the paths corresponding to the saved keys
                img_paths = [self.data["paths"][y] for y in [self.data["keys"].index(x) for x in imgs]]
                for key in imgs:
                    group = self.cursor.execute(
                        "SELECT name FROM groups WHERE image=?",
                        (key,)
                    ).fetchall()
                    if group:
                        if group[0][0] in groups:
                            groups[group[0][0]].append(key)
                        else:
                            groups[group[0][0]] = [key]
                for group in groups.keys():
                    group_str += f"{group}({len(groups[group])}) "

                add_item = QStandardItem()
                text = f"{name}\n{details[:47]}...\nGroups: {group_str}"
                add_item.setText(text)
                add_item.setData(
                    {
                        "name": name,
                        "details": details,
                        "notes": notes,
                        "groups": groups,
                        "keys": imgs,
                        "image_paths": img_paths
                    }
                )
                add_item.setIcon(Icon.get_icon("CLIPBOARD"))
                self.exp_model.appendRow(add_item)

    def enable_experiment_buttons(self, enable: bool = True) -> None:
        """
        Method to enable the buttons relevant for experiment changes

        :param enable: The bool to pass to the setEnabled function of the buttons
        :return: None
        """
        self.ui.lv_experiments.setEnabled(enable)
        self.ui.btn_add.setEnabled(enable)
        self.ui.btn_remove.setEnabled(enable)
        self.ui.btn_images_add.setEnabled(enable)
        self.ui.btn_images_clear.setEnabled(enable)
        self.ui.btn_add_group.setEnabled(enable)

    def on_exp_selection_change(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        """
        Function to react to changed experiment selection

        :param selected: The selected item
        :param deselected: The deselected item
        :return: None
        """
        # Get selected experiment
        selected = selected.indexes()
        deselected = deselected.indexes()
        self.enable_experiment_buttons(False)
        # Store the current data to the deselected item
        if deselected:
            item = self.exp_model.item(deselected[0].row())
            name = self.ui.le_name.text()
            details = self.ui.te_details.toPlainText()
            notes = self.ui.te_notes.toPlainText()
            keys = []
            img_paths = []
            # Iterate over all images
            for row in range(self.img_model.rowCount()):
                # Get item
                img_item = self.img_model.item(row)
                # Get item data and append key
                keys.append(img_item.data()["key"])
                img_paths.append(img_item.data()["path"])
            groups = item.data()["groups"]
            item.setData(
                {
                    "name": name,
                    "details": details,
                    "notes": notes,
                    "groups": groups,
                    "keys": keys,
                    "image_paths": img_paths
                }
            )
            group_str = ""
            for group in groups.keys():
                group_str += f"{group}({len(groups[group])}) "
            text = f"{name}\n{details[:47]}...\nGroups: {group_str}"
            item.setText(text)
        # Clear the image list
        self.img_model.clear()
        if selected:
            data = self.exp_model.item(selected[0].row()).data()
            # Insert data into textfields
            self.ui.le_name.setText(data["name"])
            self.ui.te_details.setPlainText(data["details"])
            self.ui.te_notes.setPlainText(data["notes"])
            # Enable text inputs for change
            self.ui.le_name.setEnabled(True)
            self.ui.te_details.setEnabled(True)
            self.ui.te_notes.setEnabled(True)
            groups_str = ""
            for name, keys in data["groups"].items():
                groups_str += f"{name} ({len(keys)})"
            self.ui.le_groups.setText(groups_str)
            if len(data["image_paths"]) > 25:
                self.update_timer = Loader(data["image_paths"],
                                           feedback=self.add_image_items,
                                           processing=create_image_item_list_from)
            else:
                # Add the saved image items to img_model
                items = Util.create_image_item_list_from(data["image_paths"])
                for item in items:
                    self.img_model.appendRow(item)
                # Enable buttons for input
                self.ui.btn_images_add.setEnabled(True)
                self.ui.btn_remove.setEnabled(True)
        else:
            # Clear everything if selection was cleared
            self.ui.le_name.clear()
            self.ui.te_details.clear()
            self.ui.te_notes.clear()
            self.ui.le_groups.clear()
            # Disable text inputs until an experiment was selected
            self.ui.le_name.setEnabled(False)
            self.ui.te_details.setEnabled(False)
            self.ui.te_notes.setEnabled(False)
            # Disable buttons to prevent unnecessary input
            self.ui.btn_add_images.setEnabled(False)
            self.ui.btn_remove_image.setEnabled(False)
            self.ui.btn_images_clear.setEnabled(False)
            self.ui.btn_remove.setEnabled(True)
        self.enable_experiment_buttons(True)

    def add_image_items(self, items: List[QStandardItem]) -> None:
        """
        Method to add items to the image list

        :param items: The items to add
        :return: None
        """
        for item in items:
            self.img_model.appendRow(item)
        self.ui.prg_images.setValue(self.update_timer.percentage * 100)
        if not items:
            # Enable buttons for input
            self.enable_experiment_buttons()

    def open_group_dialog(self):
        # Get the selected experiment
        exp_index = self.ui.lv_experiments.selectionModel().selectedIndexes()
        exp = self.exp_model.itemFromIndex(exp_index[0])
        exp_data = exp.data()
        group_dial = GroupDialog(data=exp_data)
        code = group_dial.exec()
        if code == QDialog.Accepted:
            groups = {}
            for row in range(group_dial.group_model.rowCount()):
                item = group_dial.group_model.item(row)
                data = item.data()
                groups[data["name"]] = data["keys"]
            exp_data["groups"] = groups
            exp.setData(exp_data)
            group_str = ""
            for group in groups.keys():
                group_str += f"{group}({len(groups[group])}) "
            exp.setText(
                f"{exp_data['name']}\n{exp_data['details'][:47]}...\nGroups: {group_str}"
            )
            self.ui.le_groups.setText(group_str)


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
        # TODO
        """
        # Convert image paths to QStandardItems
        items = Util.create_image_item_list_from(self.images, indicate_progress=False)
        for img in items:
            self.img_model.appendRow(img)
        # Select images that are marked as selected
        for row in range(self.img_model.rowCount()):
            item = self.img_model.item(row)
            # Select marked images
            if item.data()["key"] in self.selected_images:
                # Create index
                index = self.img_model.createIndex(row, 0)
                # Select image
                self.ui.lv_images.selectionModel().select(index, QItemSelectionModel.Select)
        """
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


class GroupDialog(QDialog):

    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.ui = None
        self.img_model = None
        self.group_model = None
        self.update_timer = None
        self.initialize_ui()
        self.load_groups()

    def initialize_ui(self) -> None:
        """
        Method to initialize the UI

        :return: None
        """
        self.ui = uic.loadUi(Paths.ui_exp_dial_group_dial, self)
        self.img_model = QStandardItemModel(self.ui.lv_images)
        self.group_model = QStandardItemModel(self.ui.lv_groups)
        self.prg_bar = self.ui.prg_images
        self.ui.lv_images.setModel(self.img_model)
        self.ui.lv_groups.setModel(self.group_model)
        self.ui.lv_images.setIconSize(Icon.get_icon_size("LIST_ITEM"))
        self.ui.lv_groups.setIconSize(Icon.get_icon_size("LIST_ITEM"))
        # Connect UI to functionality
        self.ui.btn_add.clicked.connect(self.add_group)
        self.ui.btn_remove.clicked.connect(self.remove_group)
        self.ui.lv_groups.selectionModel().selectionChanged.connect(self.on_group_selection_change)
        self.ui.btn_add_images.clicked.connect(self.add_images_to_group)
        self.ui.btn_remove_image.clicked.connect(self.remove_selected_image)
        self.setWindowTitle("Group Dialog")
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint)

    def load_groups(self) -> None:
        """
        Method to load existing groups from the database

        :return: None
        """
        for group, keys in self.data["groups"].items():
            group_item = QStandardItem()
            data = {
                "name": group,
                "exp": self.data["name"],
                "keys": keys
            }
            group_item.setText(f"{data['name']}:\nImages: {len(keys)}")
            group_item.setData(data)
            self.group_model.appendRow(group_item)

    def add_images_to_group(self) -> None:
        """
        Method to add images to the selected group

        :return: None
        """
        # Get selected images
        keys, paths = self.open_image_selection_dialog()
        if keys is not None and paths is not None:
            # Get selected group
            index = self.ui.lv_groups.selectionModel().selectedIndexes()[0]
            # Change group data
            item = self.group_model.itemFromIndex(index)
            data = item.data()
            data["keys"] = keys
            item.setData(data)
            item.setText(
                f"{data['name']}:\nImages: {len(keys)}"
            )
            self.img_model.clear()
            if len(paths) > 25:
                self.update_timer = Loader(paths, feedback=self.add_image_items,
                                           processing=create_image_item_list_from)
                self.setEnabled(False)
            else:
                # Create image items
                items = Util.create_image_item_list_from(paths, indicate_progress=False)
                for item in items:
                    self.img_model.appendRow(item)

    def add_image_items(self, items: List[QStandardItem]) -> None:
        """
        Method to load the images given by the list paths

        :param items: QStandardItems to add to the images list
        :return: None
        """
        for img in items:
            self.img_model.appendRow(img)
        self.prg_bar.setValue(self.update_timer.percentage * 100)
        if not items:
            self.setEnabled(True)

    def open_image_selection_dialog(self) -> Tuple[List[str], List[str]]:
        """
        Method to open a dialog to add images to the selected group

        :return: None
        """
        # Get selected group
        index = self.ui.lv_groups.selectionModel().selectedIndexes()[0]
        item = self.group_model.itemFromIndex(index)
        sel_dialog = ImageSelectionDialog(images=self.data["image_paths"],
                                          selected_images=item.data()["keys"])
        code = sel_dialog.exec()
        if code == QDialog.Accepted:
            return sel_dialog.get_selected_images()
        else:
            return (None, None)

    def remove_selected_image(self) -> None:
        """
        Methodt oremove the selected image from the selected group

        :return: None
        """
        # Get selected image
        indices = self.ui.lv_images.selectionModel().selectedIndexes()
        img = self.img_model.itemFromIndex(indices[0])
        key = img.data()["key"]
        self.img_model.removeRows(indices[0].row())
        # Get selected group
        indices = self.ui.lv_groups.selectionModel().selectedIndexes()
        group = self.group_model.itemFromIndex(indices[0])
        group_data = group.data()
        group_data["keys"].remove(key)
        group.setData(group_data)
        group.setText(
            f"{group.data['name']}:\nImages: {len(group.data()['keys'])}"
        )

    def clear_images(self) -> None:
        """
        Method to remove all image from the selected group

        :return: None
        """
        # Get selected group
        indices = self.ui.lv_groups.selectionModel().selectedIndexes()
        group = self.group_model.itemFromIndex(indices[0])
        group_data = group.data()
        name = group_data["name"]
        # Get the number of images
        img_num = self.img_model.rowCount()
        # Check if the user really wants to remove all images
        clk = QMessageBox.question(self, "Remove images from group",
                                   f"Do you really want to remove {img_num} images the group {name}?"
                                   " This action cannot be reversed!",
                                   QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if clk == QMessageBox.Yes:
            self.img_model.clear()
            group_data["keys"] = []
            group.setData(group_data)
            group.setText(
                f"{group.data['name']}:\nImages: 0"
            )

    def on_img_selection_change(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        """
        Method to react to the selection of images

        :param selected: The selected items
        :param deselected: The deselected items
        :return: None
        """
        selected = selected.indexes()
        if selected:
            self.ui.btn_images_remove.setEnabled(True)
        else:
            self.ui.btn_images_remove.setEnabled(False)

    def on_group_selection_change(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        """
        Function to react to changed experiment selection

        :param selected: The selected item
        :param deselected: The deselected item
        :return: None
        """
        self.ui.lv_images.setEnabled(False)
        # Delete images of the previously selected group
        self.img_model.clear()
        # Get selected experiment
        selected = selected.indexes()
        if selected:
            data = self.group_model.item(selected[0].row()).data()
            paths = [self.data["image_paths"][self.data["keys"].index(x)] for x in data["keys"]]
            for item in Util.create_image_item_list_from(paths, indicate_progress=False):
                self.img_model.appendRow(
                    item
                )
            self.ui.btn_add_images.setEnabled(True)
            self.ui.btn_remove.setEnabled(True)
        else:
            self.ui.btn_add_images.setEnabled(False)
            self.ui.btn_clear_images.setEnabled(False)
            self.ui.btn_remove_image.setEnabled(False)
            self.ui.btn_remove.setEnabled(False)
        self.ui.lv_images.setEnabled(True)

    def add_group(self) -> None:
        """
        Method to add a new group to the experiment

        :return: None
        """
        dial = QInputDialog()
        dial.setStyleSheet(open("inputbox.css", "r").read())
        dial.setWindowIcon(Icon.get_icon("LOGO"))
        name, ok = QInputDialog.getText(dial, "Group Dialog", "Enter the new group: ")
        if ok:
            # Create item to add to group list
            item = QStandardItem()
            item_data = {
                "name": name,
                "keys": [],
                "exp": self.data["name"]
            }
            item.setData(item_data)
            item.setText(f"{name}:\nImages: 0")
            self.group_model.appendRow(item)

    def remove_group(self) -> None:
        """
        Function to remove a group

        :return: None
        """
        # Create connection and cursor to db
        conn = sqlite3.connect(Paths.database)
        curs = conn.cursor()
        # Get selected group
        index = self.ui.lv_groups.selectionModel().selectedIndexes()[0].row()
        item = self.group_model.item(index)
        data = item.data()
        clk = QMessageBox.question(self, "Erase group",
                                    f"Do you really want to delete the group {data['name']}?"
                                    " This action cannot be reversed!",
                                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if clk == QMessageBox.Yes:
            # Remove list item
            self.group_model.removeRow(index)
            # Update images in database
            for key in data["keys"]:
                curs.execute(
                    "DELETE FROM groups WHERE image=?",
                    (key,)
                )
            conn.commit()


class ExperimentSelectionDialog(QDialog):
    """
    Class to enable selection of experiments
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = None
        self.connection = sqlite3.connect(Paths.database)
        self.cursor = self.connection.cursor()
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
        exps = self.cursor.execute(
            "SELECT * FROM experiments"
        ).fetchall()
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
        channels = self.cursor.execute(
            "SELECT DISTINCT name FROM channels WHERE md5 IN (SELECT image FROM groups WHERE experiment=?)",
            (exp,)
        ).fetchall()
        # Get main channel
        main = self.cursor.execute(
            "SELECT DISTINCT channel FROM roi WHERE image IN (SELECT image FROM groups WHERE experiment=?)"
            " AND associated IS NULL",
            (exp, )
        ).fetchall()[0][0]
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


class StatisticsDialog(QDialog):
    """
    Dialog to show statistical analysis of data
    """

    def __init__(self, experiment: str, active_channels: List[str], *args, **kwargs):
        """
        :param experiment: The experiment to show
        :param active_channels: The channels to analyse
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.ui = None
        self.experiment = experiment
        self.active_channels = active_channels
        self.connection: sqlite3.Connection = sqlite3.connect(Paths.database)
        self.cursor: sqlite3.Cursor = self.connection.cursor()
        self.current_channel = None
        self.current_group = None
        self.channels: List = []
        self.group_data: Dict = {}
        self.group_keys: Dict = {}
        self.initialize_ui()
        self.get_raw_data()
        self.create_plots()

    def initialize_ui(self) -> None:
        """
        Method to intialize the ui

        :return: None
        """
        self.ui = uic.loadUi(Paths.ui_stat_dial, self)
        self.ui.cbx_dist.addItems(["Poisson"])
        self.setWindowTitle(f"Statistics for {self.experiment}")
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint)

    def get_raw_data(self) -> Tuple[List, Dict]:
        """
        Method to retrieve the data about groups from the database

        :return: None
        """
        # Get the groups associated with the experiment
        groups_raw = self.cursor.execute(
            "SELECT * FROM groups WHERE experiment=?",
            (self.experiment,)
        )
        for raw in groups_raw:
            img = raw[0]
            name = raw[2]
            if name in self.group_keys:
                self.group_keys[name].append(img)
            else:
                self.group_keys[name] = [img]
        # Get the channels of the image
        channels = self.cursor.execute(
            "SELECT DISTINCT name, index_ FROM channels WHERE md5 IN (SELECT image FROM groups WHERE experiment=?)",
            (self.experiment,)
        ).fetchall()
        # Get main channel
        main = self.cursor.execute(
            "SELECT DISTINCT channel FROM roi WHERE associated IS NULL AND image"
            " IN (SELECT image FROM groups WHERE experiment=?)",
            (self.experiment,)
        ).fetchall()
        # Check if accross the images multiple main channels are given
        if len(main) > 1:
            return
        main = main[0][0]
        # Clean up channels
        self.channels = [x[0] for x in channels if x[0] != main]
        self.ui.cbx_channel.addItems(channels)
        # Select first channel as standard
        self.ui.cbx_group.addItems(self.group_keys.keys())
        # Select first group as standard
        self.ui.cbx_channel.setCurrentIndex(0)
        self.ui.cbx_group.setCurrentIndex(0)
        self.current_channel = self.channels[0]
        self.current_group = self.group_keys.keys()[0]



        # Create empty data lists
        for _ in self.channels:
            self.group_data.append([])
        # Get channel indices
        for group, imgs in groups.items():
            foci_per_nucleus = [[] for _ in range(len(channels))]
            # Iterate over the images of the group
            for key in imgs:
                # Get all nuclei for this image
                nuclei = self.cursor.execute(
                    "SELECT hash FROM roi WHERE image=? AND associated IS NULL",
                    (key,)
                ).fetchall()
                # Get the foci per individual nucleus
                for nuc in nuclei:
                    for channel in channels:
                        # Get channel of respective of the
                        foci_per_nucleus[channels.index(channel)].append(
                            self.cursor.execute(
                                "SELECT COUNT(*) FROM roi WHERE associated=? AND channel=?",
                                (nuc[0], channel)
                            ).fetchall()[0][0]
                        )
            group_data[group] = foci_per_nucleus
        return channels, group_data

    def get_data_for_group(self) -> Tuple[str, list]:
        """
        Method to get the data for a specific group and channel

        :return: A tuple containing the channel name and the data
        """
        # Get keys corresponding to currently
        keys = self.group_keys.get(self.current_group, [])
        # Get channel index
        index = self.channels.index(self.current_channel)

        # Iterate over all images of the group
        for key in keys:
            # Get all nuclei for this image
            nuclei = self.cursor.execute(
                "SELECT hash FROM roi WHERE image=? AND associated IS NULL",
                (key,)
            ).fetchall()
            # Get the foci per individual nucleus
            for nuc in nuclei:
                # Get channel of respective of the
                self.group_data[index].append(
                    self.cursor.execute(
                        "SELECT COUNT(*) FROM roi WHERE associated=? AND channel=?",
                        (nuc[0], self.current_channel)
                        ).fetchall()[0][0]
                    )
        group_data[group] = foci_per_nucleus



    def create_poisson_plots(self) -> QScrollArea:
        """
        Method to create plots for the value tab

        :param channels: A list of available channels
        :param group_data: The data to plot
        :return: The create plots inside a QScollArea
        """
        # Define scroll area for poi tab
        sa, layout = Util.create_scroll_area()
        for i in range(len(channels)):
            if self.active_channels[channels[i]]:
                check = False
                # Get the data for this channel
                data = {key: value[i] for key, value in group_data.items()}
                # Create Poisson Plot for channel
                for group, values in data.items():
                    # Create layout to add plot and labels
                    plot_layout = self.ui.vl_data
                    label_layout = self.ui.vl_text
                    poiss = PoissonPlotWidget(data=values, label=group)
                    poiss.setTitle(f"{group} - Comparison to Poisson Distribution")
                    # Create the line to add
                    line = QFrame()
                    line.setFrameShape(QFrame.HLine)
                    line.setFrameShadow(QFrame.Sunken)
                    # Add poisson plot
                    plot_layout.addWidget(poiss)
                    # Add additional information
                    label_layout.addWidget(QLabel(f"Values: {len(values)}"))
                    label_layout.addWidget(QLabel(f"Average: {np.average(values):.2f}"))
                    label_layout.addWidget(QLabel(f"Min.: {np.amin(values)}"))
                    label_layout.addWidget(QLabel(f"Max.: {np.amax(values)}"))
                    plot_layout.addLayout(label_layout)
                    if i == len(channels) - 1:
                        if not check:
                            layout.addWidget(line)
                            check = True
                    else:
                        layout.addWidget(line)
                    layout.addLayout(plot_layout)
        return sa

    def create_plots(self) -> None:
        """
        Method to fill the dialog

        :return: None
        """
        channels, group_data = self.get_data_for_group()
        self.ui.vl_poisson.addWidget(self.create_poisson_plots())


class ImgDialog(QDialog):
    MARKERS = [
        pg.mkPen(color="r", width=3),  # Red
        pg.mkPen(color="g", width=3),  # Green
        pg.mkPen(color="b", width=3),  # Blue
        pg.mkPen(color="c", width=3),  # Cyan
        pg.mkPen(color="m", width=3),  # Magenta
        pg.mkPen(color="y", width=3),  # Yellow
        pg.mkPen(color="k", width=3),  # Black
        pg.mkPen(color="w", width=3),  # White
        pg.mkPen(color=(0, 0, 0, 0))  # Invisible
    ]

    def __init__(self, image: np.ndarray, handler: ROIHandler, parent: QWidget = None):
        super(ImgDialog, self).__init__(parent)
        self.orig = image.copy()
        self.image = image
        self.handler = handler
        self.ui = uic.loadUi(Paths.ui_result_image_dialog, self)
        self.graph_widget = pg.GraphicsView()
        self.plot_item = pg.PlotItem()
        self.view = self.plot_item.getViewBox()
        self.view.setAspectLocked(True)
        self.view.invertY(True)
        self.graph_widget.setCentralWidget(self.plot_item)
        self.img_item = pg.ImageItem()
        self.plot_item.addItem(self.img_item)
        self.nuc_pen = pg.mkPen(color="FFD700", width=3, style=QtCore.Qt.DashLine)
        self.maj_ind = None
        self.items = []
        self.initialize_ui()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.set_current_image()

    def initialize_ui(self) -> None:
        self.setWindowTitle("Result Dialog")
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setToolTip("Dashed yellow line: Detected ellipsis with major and minor axis\nMagenta: Main axis")
        for ident in self.handler.idents:
            # Add an selection item to combobox
            self.ui.cbx_channels.addItem(ident)
            # Add list to items to store QGraphicItems
            self.items.append([])
        # Create QGraphicsItems for display
        for roi in self.handler.rois:
            ind = self.handler.idents.index(roi.ident)
            dims = roi.calculate_dimensions()
            if roi.main:
                params = roi.calculate_ellipse_parameters()
                cy, cx = params["center"][0], params["center"][1]
                d1 = params["minor_axis"]
                d2 = params["major_axis"]
                angle = params["angle"]
                ovx, ovy = params["orientation"]
                ellipse = QGraphicsEllipseItem(0, 0, d2 * 2, d1 * 2)
                ellipse.setData(0, self.nuc_pen)
                ellipse.setData(1, roi.main)
                # Rotate the ellipse according to the angle
                ellipse.setTransformOriginPoint(ellipse.sceneBoundingRect().center())
                ellipse.setRotation(angle)
                ellipse.setPos(cx - d2, cy - d1)
                # Draw main axis
                main_axis = QGraphicsLineItem(cx - ovx * 25, cy - ovy * 25,
                                              cx + ovx * 25, cy + ovy * 25)
                main_axis.setData(0, ImgDialog.MARKERS[4])
                # Draw major axis
                major_axis = QGraphicsLineItem(-d2, 0, d2, 0)
                major_axis.setData(0, self.nuc_pen)
                major_axis.setPos(cx, cy)
                major_axis.setRotation(angle)
                self.items[ind].append(major_axis)
                # Draw minor axis
                minor_axis = QGraphicsLineItem(-d1, 0, d1, 0)
                minor_axis.setData(0, self.nuc_pen)
                minor_axis.setPos(cx, cy)
                minor_axis.setRotation(angle + 90)
                self.items[ind].append(major_axis)
                self.items[ind].append(main_axis)
                self.items[ind].append(minor_axis)
                self.plot_item.addItem(main_axis)
                self.plot_item.addItem(major_axis)
                self.plot_item.addItem(minor_axis)
            else:
                c = dims["minX"], dims["minY"]
                d2 = dims["height"]
                d1 = dims["width"]
                ellipse = QGraphicsEllipseItem(c[0], c[1], d1, d2)
                ellipse.setData(0, self.MARKERS[ind])
                ellipse.setData(1, roi.main)
            self.items[ind].append(ellipse)
            self.plot_item.addItem(ellipse)
        # Add information
        assmap = Detector.create_association_map(self.handler)
        self.ui.channel_selector.addWidget(QLabel(f"Nuclei: {len(assmap)}"))
        empty = [key for key, val in assmap.items() if len(val) == 0]
        self.ui.channel_selector.addWidget(QLabel(f"Thereof empty: {len(empty)}"))
        roinum = {}
        # Add foci related labels
        for roi in self.handler:
            if not roi.main:
                if roi.ident not in roinum:
                    roinum[roi.ident] = {roi.associated: 1}
                elif roi.associated in roinum[roi.ident]:
                    roinum[roi.ident][roi.associated] += 1
                else:
                    roinum[roi.ident][roi.associated] = 1
        for key in roinum.keys():
            roinum[key].update({x: 0 for x in empty})
        for ident in self.handler.idents:
            if ident != self.handler.main:
                number = [x for _, x in roinum.get(ident, {"": []}).items()]
                std = np.std(number)
                number = np.average(number)
                self.ui.channel_selector.addWidget(QLabel(f"Foci/Nucleus ({ident}): {number:.2f} ± {std:.2f}"))
        self.ui.channel_selector.addItem(QtGui.QSpacerItem(20, 40,
                                                           QtGui.QSizePolicy.Minimum,
                                                           QtGui.QSizePolicy.Expanding))
        self.ui.cbx_channels.addItem("Composite")
        self.ui.cbx_channels.setCurrentText("Composite")
        self.ui.cbx_channels.currentIndexChanged.connect(self.on_channel_selection_change)
        self.ui.cbx_nuclei.stateChanged.connect(
            lambda: self.set_current_image()
        )
        self.ui.navbar.addWidget(self.graph_widget)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super(ImgDialog, self).resizeEvent(event)
        self.set_current_image()

    def on_channel_selection_change(self) -> None:
        if self.ui.cbx_channels.currentIndex() < len(self.handler.idents):
            self.image = self.orig[..., self.ui.cbx_channels.currentIndex()]
        else:
            self.image = self.orig
        self.set_current_image()

    def set_current_image(self) -> None:
        draw_nuclei = self.ui.cbx_nuclei.isChecked()
        cur_ind = self.ui.cbx_channels.currentIndex()
        # Iterate over all stored items
        for i in range(len(self.items)):
            # Get the item
            items = self.items[i]
            for item in items:
                # Select pen for the item
                if i == cur_ind or cur_ind == len(self.handler.idents) or item.data(1) and draw_nuclei:
                    pen = item.data(0)
                else:
                    pen = self.MARKERS[-1]
                item.setPen(pen)
        # Draw the background channel image
        self.img_item.setImage(self.image)


class SettingsDialog(QDialog):
    """
    Class to display a settings window, dynamically generated from a JSON file
    """

    def __init__(self, parent: QWidget = None):
        super(SettingsDialog, self).__init__(parent)
        self.data = {}
        self.changed = {}
        self.ui = None
        self.json = None
        self.url = None
        self._initialize_ui()

    def _initialize_ui(self) -> None:
        self.ui = uic.loadUi(Paths.ui_settings_dial, self)
        self.setWindowTitle("Settings Dialog")
        self.setWindowIcon(Icon.get_icon("LOGO"))

    def initialize_from_file(self, url: str) -> None:
        """
        Method to initialize the settings window from a JSON file

        :param url: The URL leading to the JSON
        :return: None
        """
        if not url.lower().endswith(".json"):
            raise ValueError("Only JSON files can be loaded!")
        self.url = url
        with open(url) as json_file:
            j_dat = json.load(json_file)
            self.json = j_dat
            for section, p in j_dat.items():
                self.add_menu_point(section, p)

    def add_section(self, section: str) -> None:
        """
        Method to add a section to the settings

        :param section: The name of the section
        :return: None
        """
        try:
            self.data[section]
        except KeyError:
            self.data[section] = {}
            tab = QScrollArea()
            tab.setWidgetResizable(True)
            kernel = QWidget()
            kernel.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Expanding
            )
            layout = QVBoxLayout()
            kernel.setLayout(layout)
            layout.setObjectName("base")
            tab.setWidget(kernel)
            self.ui.settings.addTab(tab, section)

    def add_menu_point(self, section: str, menupoint: Dict[str, Union[str, float, int]]) -> None:
        """
        Method to add a menu point to the settings section

        :param section: The name of the section
        :param menupoint: The menupoint to add
        :return: None
        """
        self.add_section(section)
        for ind in range(self.ui.settings.count()):
            if self.ui.settings.tabText(ind) == section:
                tab = self.ui.settings.widget(ind)
                base = tab.findChildren(QtGui.QVBoxLayout, "base")
                for mp in menupoint:
                    t = mp["type"].lower()
                    p = None
                    self.data[section][mp["id"]] = mp["value"]
                    if t == "show":
                        # TODO
                        p = SettingsShowWidget(
                            mp["widget"]
                        )
                    elif t == "slider":
                        p = SettingsSlider(
                            _id=mp["id"],
                            title=mp["title"],
                            desc=mp["desc"],
                            min_val=mp["values"]["min"],
                            max_val=mp["values"]["max"],
                            step=mp["values"]["step"],
                            value=mp["value"],
                            unit=mp["values"]["unit"],
                            parent=self,
                            callback=self.menupoint_changed
                        )
                    elif t == "dial":
                        p = SettingsDial(
                            _id=mp["id"],
                            title=mp["title"],
                            desc=mp["desc"],
                            min_val=mp["values"]["min"],
                            max_val=mp["values"]["max"],
                            step=mp["values"]["step"],
                            value=mp["value"],
                            unit=mp["values"]["unit"],
                            parent=self,
                            callback=self.menupoint_changed
                        )
                    elif t == "spin":
                        p = SettingsSpinner(
                            _id=mp["id"],
                            title=mp["title"],
                            desc=mp["desc"],
                            min_val=mp["values"]["min"],
                            max_val=mp["values"]["max"],
                            step=mp["values"]["step"],
                            value=mp["value"],
                            prefix=mp["values"]["prefix"],
                            suffix=mp["values"]["suffix"],
                            parent=self,
                            callback=self.menupoint_changed
                        )
                    elif t == "decspin":
                        p = SettingsDecimalSpinner(
                            _id=mp["id"],
                            title=mp["title"],
                            desc=mp["desc"],
                            min_val=mp["values"]["min"],
                            max_val=mp["values"]["max"],
                            step=mp["values"]["step"],
                            value=mp["value"],
                            decimals=mp["values"]["decimals"],
                            prefix=mp["values"]["prefix"],
                            suffix=mp["values"]["suffix"],
                            parent=self,
                            callback=self.menupoint_changed
                        )
                    elif t == "text":
                        p = SettingsText(
                            _id=mp["id"],
                            title=mp["title"],
                            desc=mp["desc"],
                            value=mp["value"],
                            parent=self,
                            callback=self.menupoint_changed
                        )
                    elif t == "combo":
                        dat = mp["values"].split(",")
                        p = SettingsComboBox(
                            _id=mp["id"],
                            title=mp["title"],
                            desc=mp["desc"],
                            data=dat,
                            value=mp["value"],
                            parent=self,
                            callback=self.menupoint_changed
                        )
                    elif t == "check":
                        p = SettingsCheckBox(
                            _id=mp["id"],
                            title=mp["title"],
                            desc=mp["desc"],
                            value=mp["value"],
                            tristate=mp["values"]["tristate"],
                            parent=self,
                            callback=self.menupoint_changed
                        )
                    base[0].addWidget(p)
                base[0].addStretch()

    def menupoint_changed(self, _id: str = None, value: Union[str, int, float] = None) -> None:
        """
        Method to detect value changes of the settings widgets

        :param _id: The id of the widget as str
        :param value: The value of the widget. Types depends on widget type
        :return: None
        """
        self.changed[_id] = value
        self.data[_id] = value

    def save_menu_settings(self) -> None:
        """
        Method to save the changes of the settings back to the defining JSON file

        :return: None
        :raises: RuntimeError if no JSON was loaded
        """
        if self.json is not None:
            if self.changed:
                # Update the saved JSON data
                for section, p in self.json.items():
                    for ind in range(len(p)):
                        try:
                            p[ind]["value"] = self.changed[p[ind]["id"]][0]
                        except KeyError:
                            pass
                # Dump JSON data back to file
                with open(self.url, 'w') as file:
                    json.dump(self.json, file)
        else:
            raise RuntimeError("Settings not initialized!")


class CategorizationDialog(QDialog):
    """
    Class to create a dialog to categorize singular images
    """

    def __init__(self, image: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = None
        self.image = image

    def initialize_ui(self) -> None:
        """
        Method to initialize the ui

        :return: None
        """
        self.ui = uic.loadUi(Paths.ui_class_dial, self)
        self.setWindowTitle(f"Classification of {self.cur_img['file_name']}")
        self.setWindowIcon(Icon.get_icon("LOGO"))


class Editor(QDialog):

    __slots__ = [
        "ui",
        "editor",
        "image",
        "roi",
        "temp_items"
    ]

    def __init__(self, image: np.ndarray, roi: ROIHandler):
        """
        Constructor

        :param image: The image to edit
        :param roi: The detected roi
        """
        super(Editor, self).__init__()
        self.ui = None
        self.editor = None
        self.image = image
        self.roi = roi
        self.temp_items = []
        self.initialize_ui()

    def accept(self) -> None:
        self.editor.apply_all_changes()
        super().accept()

    def initialize_ui(self) -> None:
        """
        Method to initialize the ui of this widget

        :return: None
        """
        self.ui = uic.loadUi(Paths.ui_editor_dial, self)
        self.editor = EditorView(self.image, self.roi, self)
        self.ui.view.addWidget(self.editor)
        # Add icons to buttons
        self.ui.btn_view.setIcon(Icon.get_icon("EYE"))
        self.ui.btn_add.setIcon(Icon.get_icon("PLUS_CIRCLE"))
        self.ui.btn_edit.setIcon(Icon.get_icon("EDIT"))
        self.ui.btn_show.setIcon(Icon.get_icon("CIRCLE"))
        self.ui.btn_coords.setIcon(Icon.get_icon("MOUSE"))
        self.ui.btn_preview.setIcon(Icon.get_icon("EYE"))
        self.ui.btn_accept.setIcon(Icon.get_icon("CHECK"))
        self.ui.btng_mode.idClicked.connect(self.set_mode)
        self.ui.btn_coords.toggled.connect(
            lambda: self.editor.track_mouse_position(self.ui.btn_coords.isChecked())
        )
        self.ui.btn_coords.toggled.connect(
            lambda: self.set_status(f"Coordinate Tracking: {self.ui.btn_coords.isChecked()}")
        )
        # Fill combobox with available channels
        for ident in self.roi.idents:
            self.ui.cbx_channel.addItem(ident)
        self.ui.cbx_channel.addItem("Composite")
        self.ui.cbx_channel.setCurrentText("Composite")
        self.ui.cbx_channel.currentIndexChanged.connect(
            lambda: self.editor.show_channel(self.ui.cbx_channel.currentIndex())
        )
        # React to Draw Ellipsis Button toggle
        self.ui.btn_show.toggled.connect(
            lambda: self.editor.draw_additional_items(self.ui.btn_show.isChecked())
        )
        self.ui.btng_mode.idToggled.connect(self.change_mode)
        # Setup editing boxes
        sy, sx, _ = self.image.shape
        self.ui.spb_x.setMinimum(0)
        self.ui.spb_x.setMaximum(sx - 1)
        self.ui.spb_x.valueChanged.connect(self.ui.btn_accept.setEnabled)
        self.ui.spb_x.valueChanged.connect(self.ui.btn_preview.setEnabled)
        self.ui.spb_y.setMinimum(0)
        self.ui.spb_y.setMaximum(sy - 1)
        self.ui.spb_y.valueChanged.connect(self.ui.btn_accept.setEnabled)
        self.ui.spb_y.valueChanged.connect(self.ui.btn_preview.setEnabled)
        self.ui.spb_width.setMinimum(0)
        self.ui.spb_width.setMaximum(sx)
        self.ui.spb_width.valueChanged.connect(self.ui.btn_accept.setEnabled)
        self.ui.spb_width.valueChanged.connect(self.ui.btn_preview.setEnabled)
        self.ui.spb_height.setMinimum(0)
        self.ui.spb_height.setMaximum(sy)
        self.ui.spb_height.valueChanged.connect(self.ui.btn_accept.setEnabled)
        self.ui.spb_height.valueChanged.connect(self.ui.btn_preview.setEnabled)
        self.ui.spb_angle.valueChanged.connect(self.ui.btn_accept.setEnabled)
        self.ui.spb_angle.valueChanged.connect(self.ui.btn_preview.setEnabled)
        self.ui.btn_preview.clicked.connect(self.set_changes)
        self.ui.btn_accept.clicked.connect(self.set_changes)

    def set_changes(self) -> None:
        """
        Method to make changes to existing item

        :return: None
        """
        # Get the info if this should be a preview or permanent
        preview = self.sender() == self.ui.btn_preview
        # Define QRect to adjust position of item
        x, y = self.ui.spb_x.value(), self.ui.spb_y.value(),
        width, height = self.ui.spb_width.value(), self.ui.spb_height.value()
        rect = QRectF(x - width / 2, y - height / 2, width, height)
        angle = self.ui.spb_angle.value()
        self.editor.set_changes(rect, angle, preview)

    def preview_changes(self) -> None:
        """
        Method to preview the changes made during editing

        :return: None
        """
        # Define QRect to adjust position of item
        rect = QRectF(self.ui.spb_x.value(), self.ui.spb_y.value(),
                      self.ui.spb_width.value(), self.ui.spb_height.value())
        angle = self.ui.spb_angle.value()
        self.editor.set_changes(rect, angle)

    def setup_editing(self, item: QGraphicsItem) -> None:
        """
        Method to display the information of the selected item

        :param item: The item to retrieve the information from
        :return: None
        """
        self.ui.spb_x.setValue(item.center[0])
        self.ui.spb_y.setValue(item.center[1])
        self.ui.spb_width.setValue(item.width)
        self.ui.spb_height.setValue(item.height)
        self.ui.spb_angle.setValue(item.angle)
        self.ui.btn_preview.setEnabled(False)
        self.ui.btn_accept.setEnabled(False)
        self.enable_editing_widgets(True)

    def enable_editing_widgets(self, enable: bool = True) -> None:
        """
        Method to enable the widgets necessary for editing

        :param enable: Boolean decider
        :return: None
        """
        self.ui.spb_x.setEnabled(enable)
        self.ui.spb_y.setEnabled(enable)
        self.ui.spb_width.setEnabled(enable)
        self.ui.spb_height.setEnabled(enable)
        self.ui.spb_angle.setEnabled(enable)
        if not enable:
            self.ui.btn_preview.setEnabled(enable)
            self.ui.btn_accept.setEnabled(enable)
            self.ui.spb_x.setValue(0)
            self.ui.spb_y.setValue(0)
            self.ui.spb_width.setValue(0)
            self.ui.spb_height.setValue(0)
            self.ui.spb_angle.setValue(0)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        super().keyPressEvent(event)
        if event.key() == Qt.Key_1:
            self.set_mode(0)
        elif event.key() == Qt.Key_2:
            self.set_mode(1)
        elif event.key() == Qt.Key_3:
            self.set_mode(2)
        elif event.key() == Qt.Key_4:
            self.ui.btn_edit_nuclei.setChecked(True)
        elif event.key() == Qt.Key_5:
            self.ui.btn_edit_foci.setChecked(True)
        elif event.key() == Qt.Key_6:
            self.ui.btn_coords.setChecked(not self.ui.btn_coords.isChecked())
        elif event.key() == Qt.Key_7:
            self.ui.btn_show.setChecked(not self.ui.btn_show.isChecked())

    def set_mode(self, mode: int) -> None:
        """
        Method to change the displayed mode
        
        :param mode: The mode to select
        :return: None
        """
        self.enable_editing_widgets(False)
        if mode == 0:
            self.ui.btn_view.setChecked(True)
        elif mode == 1:
            self.ui.btn_add.setChecked(True)
        elif mode == 2:
            self.ui.btn_edit.setChecked(True)

    def change_mode(self, id_, checked) -> None:
        """
        Method to change the editor mode

        :return: None
        """
        if checked:
            self.editor.change_mode(abs(id_) - 3)

    def set_status(self, status: str) -> None:
        """
        Method to display a status in the status bar

        :param status: The status to display
        :return: None
        """
        self.ui.lbl_status.setText(status)

