import copy
import json
import math
import sqlite3
import matplotlib.pyplot as plt
from typing import List, Any, Tuple, Dict, Union, Iterable

import numpy as np
import pyqtgraph as pg
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtGui, QtCore, uic
from PyQt5.QtCore import QItemSelection, QItemSelectionModel, QSize, Qt, QRectF
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QIcon, QPixmap, QColor, QResizeEvent, QKeyEvent, QMouseEvent, \
    QBrush, QPen, QPainter, QWheelEvent
from PyQt5.QtWidgets import QDialog, QMessageBox, QInputDialog, QCheckBox, QVBoxLayout, QFrame, QScrollArea, QWidget, \
    QLabel, QHBoxLayout, QSizePolicy, QGraphicsView, QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsLineItem, \
    QStyleOptionGraphicsItem, QGraphicsItem, QGraphicsScene, QGraphicsRectItem, QGraphicsSceneHoverEvent
from skimage.draw import ellipse

from core.Detector import Detector
from core.roi import AreaAnalysis
from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler
from gui import Paths, Util
from gui.Definitions import Icon
from gui.Plots import BoxPlotWidget, PoissonPlotWidget
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
        Methpd to add the selected images to the selected experiment

        :return: None
        """
        # Get selected images
        keys, paths = self.open_image_selection_dialog()
        if keys is not None and paths is not None:
            # Get selected experiment
            selected_exp = self.exp_model.item(self.ui.lv_experiments.selectionModel().selectedIndexes()[0].row())
            data = selected_exp.data()
            # Clear img_model
            self.img_model.clear()
            # Get items
            items = Util.create_image_item_list_from(paths)
            for item in items:
                self.img_model.appendRow(item)
            data["keys"] = keys
            data["image_paths"] = paths
            selected_exp.setData(data)

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
            groups_str = ""
            for name, keys in data["groups"].items():
                groups_str += f"{name} ({len(keys)})"
            self.ui.le_groups.setText(groups_str)
            # Add the saved image items to img_model
            items = Util.create_image_item_list_from(data["image_paths"])
            for item in items:
                self.img_model.appendRow(item)
            self.ui.btn_images_add.setEnabled(True)
            self.ui.btn_remove.setEnabled(True)
        else:
            # Clear everything if selection was cleared
            self.ui.le_name.clear()
            self.ui.te_details.clear()
            self.ui.te_notes.clear()
            self.ui.le_groups.clear()
            self.ui.btn_add_images.setEnabled(False)
            self.ui.btn_remove_image.setEnabled(False)
            self.ui.btn_images_clear.setEnabled(False)
            self.ui.btn_remove.setEnabled(True)

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
        self.images: List[str] = images
        self.selected_images = selected_images
        self.img_model = None
        self.ui = None
        self.initialize_ui()

    def initialize_ui(self) -> None:
        """
        Method to initialize the UI

        :return: None
        """
        self.ui = uic.loadUi(Paths.ui_img_sel_dial, self)
        self.img_model = QStandardItemModel()
        self.ui.lv_images.setModel(self.img_model)
        self.ui.lv_images.setIconSize(Icon.get_icon_size("LIST_ITEM"))
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
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowTitle("Image Selection Dialog")
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint)

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
            # Create image items
            items = Util.create_image_item_list_from(paths, indicate_progress=False)
            for item in items:
                self.img_model.appendRow(item)

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
            self.ui.btn_images_clear.setEnabled(False)
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
        self.connection = sqlite3.connect(Paths.database)
        self.cursor = self.connection.cursor()
        self.initialize_ui()
        self.create_plots()

    def initialize_ui(self) -> None:
        """
        Method to intialize the ui

        :return: None
        """
        self.ui = uic.loadUi(Paths.ui_stat_dial, self)
        self.setWindowTitle(f"Statistics for {self.experiment}")
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint)

    def get_group_data(self) -> Tuple[List, Dict]:
        """
        Method to retrieve the data about groups from the database

        :return: None
        """
        # Get the groups associated with the experiment
        groups_raw = self.cursor.execute(
            "SELECT * FROM groups WHERE experiment=?",
            (self.experiment,)
        )
        # Get the individual groups and corresponding images
        groups = {}
        for raw in groups_raw:
            img = raw[0]
            name = raw[2]
            if name in groups:
                groups[name].append(img)
            else:
                groups[name] = [img]
        # Get number of Nuclei per group
        group_data = {}
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
        channels = [x[0] for x in channels if x[0] != main]
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

    def create_value_plots(self, channels: List[str], group_data: Dict[str, List]) -> QScrollArea:
        """
        Method to create plots for the value tab

        :param channels: A list of available channels
        :param group_data: The data to plot
        :return: The create plots inside a QScollArea
        """
        # Define scroll area for val tab
        sa, layout = Util.create_scroll_area()
        # Create plots
        for i in range(len(channels)):
            if not self.active_channels[channels[i]]:
                continue
            # Get the data for this channel
            data = {key: value[i] for key, value in group_data.items()}
            # Create PlotWidget for channel
            d = list(data.values())
            g = list(data.keys())
            # Create layout to add plot and labels
            plot_layout = QHBoxLayout()
            label_layout = QHBoxLayout()

            pw = BoxPlotWidget(data=d, groups=g)
            pw.setTitle(f"{channels[i]} Analysis")
            pw.laxis.setLabel("Foci/Nucleus")
            plot_layout.addWidget(pw)
            # Create
            # Create the line to add
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            # Get plotting data of BoxPlot
            p_data = pw.p_data

            # Iterate over groups
            for j in range(len(g)):
                group_layout = QVBoxLayout()
                group_layout.addWidget(QLabel(
                    f"<strong>Group: {g[j]}</strong>"
                ))
                group_layout.addWidget(QLabel(
                    f"Values (w/o Outliers): {p_data[j]['number']}"
                ))
                group_layout.addWidget(QLabel(
                    f"Average: {p_data[j]['average']:.2f}"
                ))
                group_layout.addWidget(QLabel(
                    f"Median: {p_data[j]['median']}"
                ))
                group_layout.addWidget(QLabel(
                    f"IQR: {p_data[j]['iqr']}"
                ))
                group_layout.addWidget(QLabel(
                    f"Outliers: {len(p_data[j]['outliers'])}"
                ))
                group_layout.addStretch(1)
                label_layout.addLayout(group_layout)
            plot_layout.addLayout(label_layout)
            layout.addLayout(plot_layout)
            # layout_main_val.addWidget(sa)
            if i < len(channels) - 1:
                layout.addWidget(line)
        return sa

    def create_poisson_plots(self, channels: List[str], group_data: Dict[str, List]) -> QScrollArea:
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
                    plot_layout = QHBoxLayout()
                    label_layout = QVBoxLayout()
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
        channels, group_data = self.get_group_data()
        self.ui.vl_poisson.addWidget(self.create_poisson_plots(channels, group_data))
        self.ui.vl_values.addWidget(self.create_value_plots(channels, group_data))


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


class EditorView(pg.GraphicsView):
    COLORS = [
        QColor(255, 50, 0),  # Red
        QColor(50, 255, 0),  # Green
        QColor(255, 255, 0),  # Yellow
        QColor(255, 0, 255),  # Magenta
        QColor(0, 255, 255),  # Cyan
    ]

    def __init__(self, image: np.ndarray, roi: ROIHandler, parent: QDialog):
        super(EditorView, self).__init__()
        self.parent = parent
        self.mode = -1
        self.image = image
        self.active_channel = None
        self.roi = roi
        self.main_channel = roi.idents.index(roi.main)
        self.plot_item = pg.PlotItem()
        self.view = self.plot_item.getViewBox()
        self.view.setAspectLocked(True)
        self.view.invertY(True)
        self.pos_track = True
        self.img_item = pg.ImageItem()
        self.plot_item.addItem(self.img_item)
        self.plot_vb = self.plot_item.vb
        # Set proxy to detect mouse movement
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=45, slot=self.mouse_moved)
        self.mpos = None
        # Activate mouse tracking for widget
        self.setMouseTracking(True)
        self.setCentralWidget(self.plot_item)
        self.draw_additional = True
        # List of existing items
        self.items = self.draw_roi()
        # List for newly created items
        self.temp_items = []
        # List for items that should be removed
        self.delete = []
        self.item_changes = {}
        self.selected_item = None
        self.shift_down = False
        self.saved_values = None
        self.show_channel(3)

    def set_changes(self, rect: QRectF, angle: float, preview: bool = True) -> None:
        """
        Method to apply the changes made by editing

        :param rect: The new bounding box of the currently active item
        :param angle: The angle of the currently active item
        :param preview: When true, the item will save its original orientation and size

        :return: None
        """
        self.selected_item.update_data(rect, angle, preview)
        # Mark Original ident as delete
        if not preview:
            self.delete.append(self.selected_item.roi_ident)

    def draw_additional_items(self, state: bool = True) -> None:
        """
        Method to signal if additional items besides nuclei and foci should be drawn

        :param state: Boolean decider
        :return: None
        """
        self.draw_additional = state
        ROIDrawer.draw_additional_items(self.items, self.draw_additional)

    def show_channel(self, channel: int) -> None:
        """
        Method to show the specified channel of the image

        :param channel: The index of the channel
        :return: None
        """
        if self.selected_item:
            self.selected_item.enable_editing(False)
            self.selected_item = None
            self.parent.enable_editing_widgets(False)
        self.active_channel = channel
        if channel == self.image.shape[2]:
            self.img_item.setImage(self.image)
        else:
            self.img_item.setImage(self.image[..., channel])
        ROIDrawer.change_channel(self.items, channel, self.draw_additional)

    def change_mode(self, mode: int = 0) -> None:
        """
        Method to change the edit mode

        :param mode: 0 for add new, 1 for edit
        :return: None
        """
        self.mode = mode
        if self.selected_item:
            self.selected_item.enable_editing(False)
            self.selected_item = None

    def track_mouse_position(self, state: bool = True) -> None:
        """
        Enables mouse coordinate tracking

        :param state: Boolean decider
        :return: None
        """
        self.pos_track = state

    def draw_roi(self) -> Iterable[QGraphicsEllipseItem]:
        """
        Method to draw the roi

        :return: None
        """
        return ROIDrawer.draw_roi(self.plot_item, self.roi)

    def get_roi_index(self, roi) -> int:
        """
        Method to get the channel index for the given ROI

        :param roi: The ROI
        :return: The channel index as int
        """
        return self.roi.idents.index(roi.ident)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        super().keyPressEvent(event)
        if event.key() == Qt.Key_Shift:
            self.shift_down = True
        if event.key() == Qt.Key_Delete:
            item = self.selected_item
            # Remove item from scene
            item.remove_from_view(self)
            # Add item to deletion list to remove it from the database
            self.delete.append(item.roi_ident)
        elif event.key() == Qt.Key_1:
            self.change_mode(-1)
        elif event.key() == Qt.Key_2:
            self.change_mode(0)
        elif event.key() == Qt.Key_3:
            self.change_mode(1)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Shift:
            self.shift_down = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if self.mode == 0 and self.active_channel != 3 and event.button() == Qt.LeftButton and self.shift_down:
            # Get click position
            pos = self.mpos
            if self.active_channel == self.main_channel:
                item = NucleusItem(round(pos.x()) - 75, round(pos.y()) - 38, 150, 76,  round(pos.x()), round(pos.y()),
                                   0, (0, 0), self.main_channel, -1)
                item.set_pens(
                    pg.mkPen(color="FFD700", width=3, style=QtCore.Qt.DashLine),
                    pg.mkPen(color="FFD700", width=3, style=QtCore.Qt.DashLine),
                    ROIDrawer.MARKERS[-1]
                )
                item.add_to_view(self.plot_item)
                self.parent.set_mode(2)
                self.change_mode(1)
                self.selected_item = item
                item.enable_editing(True)
                self.parent.setup_editing(item)
            else:
                item = FocusItem(round(pos.x() - 3.5), round(pos.y() - 3.5), 7, 7, self.active_channel, -1)
                item.set_pen(
                    ROIDrawer.MARKERS[self.active_channel],
                    ROIDrawer.MARKERS[-1]
                )
            item.changed = True
            self.items.append(item)
            self.temp_items.append(item)
            # Add item to view
            item.add_to_view(self.plot_item)
        if self.mode == 1 and self.active_channel != 3 and event.button() == Qt.LeftButton:
            items = [x for x in self.scene().items(self.mapToScene(event.pos()))
                     if isinstance(x, NucleusItem) or isinstance(x, FocusItem)]
            items = [x for x in items if x.channel_index == self.active_channel]
            if items:
                if self.selected_item:
                    self.selected_item.enable_editing(False)
                self.selected_item = items[-1]
                self.selected_item.enable_editing(True)
                self.parent.setup_editing(self.selected_item)

    def mouse_moved(self, event: QMouseEvent) -> None:
        if self.pos_track:
            pos = event[0]
            if self.plot_item.sceneBoundingRect().contains(pos):
                coord = self.plot_vb.mapSceneToView(pos)
                self.mpos = coord
                self.parent.set_status(f"X: {coord.x():.2f} Y: {coord.y():.2f}")

    def apply_all_changes(self) -> None:
        """
        Method to apply all made changes and save them to the database

        :return: None
        """
        # Create list of changed items to ignore during map creation
        ignore = [x.roi_ident for x in self.items if x.changed]
        ignore.extend(self.delete)
        # Create a hash asso map
        maps = self.roi.create_hash_association_maps((self.image.shape[0], self.image.shape[1]), ignore)
        # Create connection to database
        conn = sqlite3.connect(Paths.database)
        # Create cursor to do stuff
        curs = conn.cursor()
        # Create list of tuples for sqlite3
        self.delete = [(x, ) for x in self.delete]
        # Create list for items which will be unassociated due to data changes
        unassociated = []
        # Delete items marked for it
        self.delete_unassociated_roi(curs, self.delete)
        # Get all associated roi
        for roi in self.delete:
            roi_hash = curs.execute("SELECT hash FROM roi WHERE associated=?", roi).fetchall()
            if roi_hash:
                unassociated.extend([x[0] for x in roi_hash])
        curs.executemany("UPDATE OR IGNORE roi SET associated=NULL WHERE associated=?", self.delete)
        for item in self.items:
            if item.changed:
                # Check if item was added
                if item.roi_ident != -1:
                    # Delete item from database
                    self.delete_item_from_database(curs, item.roi_ident)
                    if isinstance(item, NucleusItem):
                        # Get hash list of associated foci
                        hashes = [x[0] for x in curs.execute("SELECT hash FROM roi WHERE associated=?",
                                                             (item.roi_ident,)).fetchall()]
                        unassociated.extend(hashes)
                        curs.execute("UPDATE roi SET associated=? WHERE associated=?",
                                     (None, item.roi_ident))
                    else:
                        unassociated.append(item.roi_ident)
                # Get coordinates corresponding to the item
                rr, cc = ellipse(item.center[1], item.center[0], item.height / 2, item.width / 2,
                                 self.image.shape, np.deg2rad(-item.angle))
                # Get encoded area for item
                rle = self.encode_new_roi(rr, cc, maps[item.channel_index])
                # Create new ROI instance
                roi = ROI(channel=self.roi.idents[item.channel_index],
                          main=isinstance(item, NucleusItem), auto=False)
                roi.set_area(rle)
                roihash = hash(roi)
                # Foci need to be associated
                if isinstance(item, FocusItem):
                    unassociated.append(roihash)
                self.replace_placeholder(maps[item.channel_index], roihash)
                self.write_item_to_database(curs, item, roi, rle, self.image, self.roi.ident)
                # Add ROI to ROIHandler
                self.roi.rois.append(roi)
        associations = self.create_associations(self.roi.idents.index(self.roi.main), maps, unassociated)
        # Clean unassociated list
        unassociated = [(x, ) for x in unassociated if x not in associations.keys()]
        self.delete_unassociated_roi(curs, unassociated)
        # Create new associations
        for focus, nucleus in associations.items():
            curs.execute("UPDATE roi SET associated=? WHERE hash=?",
                         (int(nucleus), int(focus)))
        # Delete erased and unassociated ROI from ROIHandler
        self.delete.extend(unassociated)
        self.roi.rois = [x for x in self.roi.rois if hash(x) not in self.delete]
        conn.commit()

    @staticmethod
    def write_item_to_database(curs: sqlite3.Cursor, item, roi: ROI,
                               rle: List[List[int]], image: np.ndarray,
                               image_id: int) -> None:
        """
        Method to write the specified item to the database

        :param curs: Cursor pointing to the database
        :param item: The item to write to the database
        :param roi: The ROI associated with the item
        :param rle: The run length encoded area of this item
        :param image: The image from which the roi is derived
        :param image_id: The id of the image
        :return: None
        """
        # Calculate statistics
        stats = roi.calculate_statistics(image[..., item.channel_index])
        ellp = roi.calculate_ellipse_parameters()
        # Prepare data for SQL statement
        rle = [(hash(roi), x[0], x[1], x[2]) for x in rle]
        # Write item to database
        curs.execute("INSERT INTO roi VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (hash(roi), image_id, False, roi.ident,
                      f"({item.center[1]:.0f}, {item.center[0]:.0f})", item.edit_rect.width,
                      item.edit_rect.height, None))
        curs.executemany("INSERT INTO points VALUES (?, ?, ?, ?)",
                         rle)
        curs.execute("INSERT INTO statistics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (hash(roi), image_id, stats["area"], stats["intensity average"],
                      stats["intensity median"], stats["intensity maximum"], stats["intensity minimum"],
                      stats["intensity std"], ellp["eccentricity"], ellp["roundness"],
                      f"({item.center[1]:.0f}, {item.center[0]:.0f})", item.width / 2, item.height / 2,
                      item.angle, ellp["area"], str(ellp["orientation"]), ellp["shape_match"]))

    @staticmethod
    def delete_item_from_database(curs: sqlite3.Cursor, roihash: int) -> None:
        """
        Method to delete the item, specified by its hash, from the database

        :param curs: Cursor pointing to the database
        :param roihash: The hash of the item
        :return: None
        """
        curs.execute("DELETE FROM roi WHERE hash=?", (roihash,))
        curs.execute("DELETE FROM points WHERE hash=?", (roihash,))
        curs.execute("DELETE FROM statistics WHERE hash=?", (roihash,))

    @staticmethod
    def create_associations(main: int, maps: List[np.ndarray],
                            unassociated: Iterable[int]) -> Dict:
        """
        Method to create associations dictionary to associate nuclei with foci

        :param main: Index of the main channel
        :param maps:
        :param unassociated:
        :return: Dictionary containing the associations
        """
        # Create new associations
        associations = {}
        for c in range(len(maps)):
            if c != main:
                for y in range(maps[0].shape[0]):
                    for x in range(maps[0].shape[1]):
                        if maps[c][y][x] and maps[main][y][x] and maps[c][y][x] in unassociated:
                            associations[maps[c][y][x]] = maps[main][y][x]
        return associations

    @staticmethod
    def delete_unassociated_roi(curs: sqlite3.Cursor, unassociated: Iterable[Tuple[int]]) -> None:
        """
        Method to delete unassociated roi from the database

        :param curs: Cursor pointing to the database
        :param unassociated: List of hashes from unassociated roi, prepared for executemany
        :return: None
        """
        curs.executemany("DELETE FROM roi WHERE hash=?",
                         unassociated)
        curs.executemany("DELETE FROM points WHERE hash=?",
                         unassociated)
        curs.executemany("DELETE FROM statistics WHERE hash=?",
                         unassociated)

    @staticmethod
    def replace_placeholder(map_: np.ndarray, roihash: int, placeholder: int = -1) -> None:
        """
        Method to replace a placeholder in the given map

        :param map_: The map
        :param roihash: The hash to replace the placeholder with
        :param placeholder: The placeholder to replace
        :return: None
        """
        shape = map_.shape
        # Replace placeholder with real hash
        for y in range(shape[0]):
            for x in range(shape[1]):
                if map_[y][x] == placeholder:
                    map_[y][x] = roihash

    @staticmethod
    def encode_new_roi(rr: List[int], cc: List[int],
                       map_: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Method to run length encode newly created roi

        :param rr: The row indices
        :param cc: The corresponding column indices
        :param map_: The corresponding map for this roi
        :return: The run length encoded area of the given roi
        """
        # Get encoded area for item
        rle = []
        # Get unique rows
        rows = np.unique(rr)
        # Iterate over unique rows
        for row in rows:
            rl = 0
            col = -1
            for index in range(len(rr)):
                if rr[index] == row and map_[rr[index]][cc[index]] == 0:
                    map_[rr[index]][cc[index]] = -1
                    rl += 1
                    if col == -1:
                        col = int(cc[index])
            rle.append((int(row), col, rl))
        return rle


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


class EditingRectangle(QGraphicsRectItem):

    __slots__ = [
        "width",
        "height",
        "center",
        "x",
        "y",
        "pen",
        "ipen",
        "color",
    ]

    def __init__(self, x, y, cx, cy, width, height):
        super().__init__(x, y, width, height)
        self.active = False
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.center = cx, cy
        self.ipen = None
        self.pen = None
        self.color = None
        self.initialize()

    def initialize(self):
        """
        Method to initialize this class

        :return:  None
        """
        self.pen = pg.mkPen(color="#bdff00", width=3, style=QtCore.Qt.DashLine)
        self.ipen = ROIDrawer.MARKERS[-1]
        self.setPen(self.pen)

    def activate(self, enable: bool = True) -> None:
        """
        Method to activate this item

        :param enable: Bool
        :return: None
        """
        if enable:
            self.setPen(self.pen)
        else:
            self.setPen(self.ipen)


class NucleusItem(QGraphicsEllipseItem):

    __slots__ = [
        "changed",
        "x",
        "y",
        "width",
        "height",
        "angle",
        "channel_index",
        "roi_ident",
        "orientation",
        "pen",
        "center",
        "indicators"
        "pen",
        "iapen",
        "ipen"
    ]

    def __init__(self, x: int, y: int, width: int, height: int, center_x: int, center_y: int,
                 angle: float, orientation: Tuple[float, float], index: int, roi_ident: int):
        super().__init__(x, y, width, height)
        self.changed = False
        self.x = x
        self.y = y
        self.angle = angle
        self.width = width
        self.height = height
        self.center = center_x, center_y
        self.orientation = orientation
        self.channel_index = index
        self.roi_ident = roi_ident
        self.indicators = []
        self.edit = False
        self.edit_rect: EditingRectangle = None
        self.pen: pg.mkPen = None
        self.ipen: pg.mkPen = None
        self.iapen: pg.mkPen = None
        self.view: EditorView = None
        self.initialize()

    def update_data(self, rect: QRectF, angle: float, keep_original: bool = True) -> None:
        """
        Method to update position and angle of this item

        :param rect: The new bounding rect of this item
        :param angle: The new angle of this item
        :param keep_original: If true, the position and angle before the change will be stored.
        Used for preview purposes
        :return: None
        """
        if not keep_original:
            self.x = rect.x()
            self.y = rect.y()
            self.width = rect.width()
            self.height = rect.height()
            self.center = rect.center().x(), rect.center().y()
            self.angle = angle
            self.changed = True
        self.setRotation(0)
        self.setRect(rect)
        # Update indicators to represent new params
        r1 = rect.height() / 2
        r2 = rect.width() / 2
        self.indicators[0].setLine(-r2, 0, r2, 0)
        self.indicators[1].setLine(-r1, 0, r1, 0)
        self.setTransformOriginPoint(rect.center())
        self.setRotation(angle)
        self.edit_rect.setRotation(0)
        self.edit_rect.setRect(rect)
        self.edit_rect.setTransformOriginPoint(rect.center())
        self.edit_rect.setRotation(angle)
        for indicator in self.indicators:
            indicator.setPos(self.boundingRect().center())

    def remove_from_view(self, view: EditorView) -> None:
        """
        Method to remove this item from the given view

        :param view: The view to remove the item from
        :return: None
        """
        view.scene().removeItem(self.edit_rect)
        view.scene().removeItem(self)

    def is_active(self, active: bool = True) -> None:
        """
        Method to set the activity of this item

        :param active: Bool
        :return: None
        """
        if not active:
            self.edit_rect.activate(active)
        self.setPen(self.iapen if not active else self.pen)

    def update_indicators(self, draw: bool = True) -> None:
        """
        Method update the drawing of indicators

        :param draw: Bool to indicate if the indicators should be drawn
        :return: None
        """
        for indicator in self.indicators:
            indicator.setPen(self.ipen if draw else self.iapen)

    def set_pens(self, pen: pg.mkPen, indicator_pen: pg.mkPen,
                 inactive_pen: pg.mkPen) -> None:
        """
        Method to set the pen to draw this item

        :param pen: The pen to draw this item when active
        :param indicator_pen: The pen to draw the indicators of this item with
        :param inactive_pen: The pen to use if this item is set to inactive
        :return: None
        """
        self.pen = pen
        self.ipen = indicator_pen
        self.iapen = inactive_pen
        self.setPen(self.pen)
        for indicator in self.indicators:
            indicator.setPen(self.ipen)

    def initialize(self) -> None:
        """
        Method to initialize the display of this item

        :return: None
        """
        op = self.sceneBoundingRect().center()
        self.setTransformOriginPoint(op)
        self.setRotation(self.angle)
        cx, cy = self.center
        r1, r2 = self.height / 2, self.width / 2
        # Draw major axis
        major_axis = QGraphicsLineItem(-r2, 0, r2, 0)
        major_axis.setPos(cx, cy)
        major_axis.setParentItem(self)
        # Draw minor axis
        minor_axis = QGraphicsLineItem(-r1, 0, r1, 0)
        minor_axis.setPos(cx, cy)
        minor_axis.setParentItem(self)
        minor_axis.setRotation(90)
        rect = EditingRectangle(self.x, self.y, self.center[0], self.center[1], self.width, self.height)
        rect.setTransformOriginPoint(rect.sceneBoundingRect().center())
        rect.setRotation(self.angle)
        self.indicators.extend([
            major_axis,
            minor_axis,
        ])
        self.edit_rect = rect
        self.edit_rect.activate(False)
        self.setEnabled(False)

    def add_to_view(self, view: EditorView) -> None:
        """
        Method to add this item and all associated items to the given view

        :param view: The view to add to
        :return: None
        """
        self.view = view
        view.addItem(self)

    def enable_editing(self, enable: bool = True) -> None:
        """
        Method to enable the editing of this item

        :param enable: Bool
        :return: None
        """
        if enable:
            self.setEnabled(enable)
            self.view.addItem(self.edit_rect)
            self.edit_rect.activate(enable)
        else:
            self.setEnabled(enable)
            self.view.removeItem(self.edit_rect)
            self.edit_rect.activate(enable)

    def __str__(self):
        return f"NucleusItem X:{self.x} Y:{self.y} W:{self.width} H:{self.height} C:{self.center}"


class FocusItem(QGraphicsEllipseItem):

    __slots__ = [
        "changed",
        "x",
        "y",
        "width",
        "height",
        "center",
        "angle"
        "edit_rect",
        "channel_index",
        "roi_ident",
        "pen",
        "ipen"
        "hover_color",
        "main_color",
        "sel_color"
    ]

    def __init__(self, x: int, y: int, width: int, height: float, index: int, roi_ident: int):
        super().__init__(x, y, width, height)
        self.changed = False
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.center = int(self.x + self.width / 2), int(self.y + self.height / 2)
        self.angle = 0
        self.channel_index = index
        self.roi_ident = roi_ident
        self.pen: pg.mkPen = None
        self.ipen: pg.mkPen = None
        self.main_color = None
        self.hover_color = None
        self.sel_color = None
        self.view = None
        self.edit_rect = None
        self.setEnabled(False)

    def update_data(self, rect: QRectF, angle: float, keep_original: bool = True) -> None:
        """
        Method to update position and angle of this item

        :param rect: The new bounding rect of this item
        :param angle: The new angle of this item
        :param keep_original: If true, the position and angle before the change will be stored.
        Used for preview purposes
        :return: None
        """
        if not keep_original:
            self.x = rect.x()
            self.y = rect.y()
            self.width = rect.width()
            self.height = rect.height()
            self.center = rect.center().x(), rect.center().y()
            self.angle = angle
            self.changed = True
            print(f"Changed: {self}")
        self.setRotation(0)
        self.setRect(rect)
        self.setTransformOriginPoint(rect.center())
        self.setRotation(angle)
        self.edit_rect.setRotation(0)
        self.edit_rect.setRect(rect)
        self.edit_rect.setTransformOriginPoint(rect.center())
        self.edit_rect.setRotation(angle)

    def remove_from_view(self, view: EditorView) -> None:
        """
        Method to remove this item from the given view

        :param view: The view to remove the item from
        :return: None
        """
        view.scene().removeItem(self.edit_rect)
        view.scene().removeItem(self)

    def is_active(self, active: bool = True) -> None:
        """
        Method to set the activity of this item

        :param active: Bool
        :return: None
        """
        self.setPen(self.pen if active else self.ipen)

    def update_indicators(self, draw: bool = True) -> None:
        """
        Dummy Method to be compatible with NucleusItem
        """
        pass

    def set_pen(self, pen: pg.mkPen, inactive_pen: pg.mkPen):
        self.pen = pen
        self.ipen = inactive_pen
        # Define needed colors
        self.main_color = pen.color()
        self.hover_color = self.main_color.lighter(100)
        self.sel_color = self.main_color.lighter(150)
        self.setPen(pen)

    def add_to_view(self, view: EditorView) -> None:
        """
        Method to add this item and all associated items to the given view

        :param view: View to add to
        :return: None
        """
        self.view = view
        view.addItem(self)
        rect = EditingRectangle(self.x, self.y, self.center[0], self.center[1], self.width, self.height)
        rect.activate(False)
        self.edit_rect = rect

    def enable_editing(self, enable: bool = True) -> None:
        """
        Method to enable the editing of this item

        :param enable: Bool
        :return: None
        """
        if enable:
            self.setEnabled(enable)
            self.view.addItem(self.edit_rect)
            self.edit_rect.activate(enable)
        else:
            self.setEnabled(enable)
            self.view.removeItem(self.edit_rect)
            self.edit_rect.activate(enable)

    def __str__(self):
        return f"FocusItem X:{self.x} Y:{self.y} W:{self.width} H:{self.height} C:{self.center}"


class ROIDrawer:

    __slots__ = ()

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

    @staticmethod
    def change_channel(items: Iterable[QGraphicsItem],
                       active_channel: int = 3,
                       draw_additional: bool = False) -> None:
        """
        Method to change the drawing of foci and nuclei according to the active channel

        :param items: The items that are drawn on the view
        :param active_channel: The active channel
        :param draw_additional: Parameter to draw items for additional information
        :return: None
        """
        for item in items:
            if item.channel_index != active_channel and active_channel != 3:
                if isinstance(item, NucleusItem) and draw_additional:
                    item.is_active(True)
                else:
                    item.is_active(False)
            else:
                item.is_active(True)
            item.update_indicators(draw_additional)

    @staticmethod
    def draw_roi(view: pg.PlotItem, handler) -> List[QGraphicsEllipseItem]:
        """
        Method to populate the given plot with the roi stored in the handler

        :param view: The PlotItem to populate
        :param handler: The ROIHandler
        :return: List of all created items
        """
        items = []
        for roi in handler:
            ind = handler.idents.index(roi.ident)
            if roi.main:
                items.append(ROIDrawer.draw_nucleus(view, roi, ind))
            else:
                items.append(ROIDrawer.draw_focus(view, roi, ind))
        return items

    @staticmethod
    def draw_focus(view: pg.PlotItem, roi: ROI, ind) -> QGraphicsEllipseItem:
        """
        Function to draw a focus onto the given view

        :param view: The view to draw on
        :param roi: The focus to draw
        :param ind: The index of the roi channel
        :return: None
        """
        dims = roi.calculate_dimensions()
        pen = ROIDrawer.MARKERS[ind]
        c = dims["minX"], dims["minY"]
        d2 = dims["height"]
        d1 = dims["width"]
        nucleus = FocusItem(c[0], c[1], d1, d2, ind, hash(roi))
        nucleus.set_pen(pen, ROIDrawer.MARKERS[-1])
        nucleus.add_to_view(view)
        return nucleus

    @staticmethod
    def draw_nucleus(view: pg.PlotItem, roi: ROI, ind) -> QGraphicsEllipseItem:
        """
        Function to draw a nucleus onto the given view

        :param view: The view to draw on
        :param roi: The nucleus to draw
        :param ind: The index of the roi channel
        :return: None
        """
        pen = pg.mkPen(color="FFD700", width=3, style=QtCore.Qt.DashLine)
        params = roi.calculate_ellipse_parameters()
        cy, cx = params["center"][0], params["center"][1]
        r1 = params["minor_axis"]
        r2 = params["major_axis"]
        angle = params["angle"]
        ovx, ovy = params["orientation"]
        focus = NucleusItem(cx - r2, cy - r1, r2 * 2, r1 * 2, cx, cy, angle, (ovx, ovy), ind, hash(roi))
        focus.set_pens(
            pen,
            pen,
            ROIDrawer.MARKERS[-1]
        )
        focus.is_active()
        focus.update_indicators()
        focus.add_to_view(view)
        return focus

    @staticmethod
    def draw_additional_items(items: List[QGraphicsItem], draw_additional: bool = True) -> None:
        """
        Method to activate the drawing of additional items

        :param items: The list of items to activate
        :param draw_additional: Bool
        :return: None
        """
        for item in items:
            if isinstance(item, NucleusItem):
                item.is_active(draw_additional)
            item.update_indicators(draw_additional)
