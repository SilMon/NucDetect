import sqlite3
import numpy as np
from typing import List, Any, Tuple, Dict, Iterable, Union

from PyQt5 import QtGui, QtCore, uic
from PyQt5.QtCore import QItemSelection, QItemSelectionModel
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QDialog, QMessageBox, QInputDialog, QCheckBox, QVBoxLayout, QFrame, QScrollArea, QWidget, \
    QLabel, QHBoxLayout

from gui import Paths, Util
from gui.Definitions import Icon
from gui.Plots import BoxPlotWidget, PoissonPlotWidget


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

    def open_image_selection_dialog(self) -> Tuple[List[str]]:
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
        # Get selected group
        index = self.ui.lv_images.selectionModel().selectedIndexes()[0].row()
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
                self.cursor.execute(
                    "DELETE FROM groups WHERE image=?",
                    (key,)
                )
            self.connection.commit()


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

        """
        # Define scroll area for val tab
        sa_main_val, layout_main_val = Util.create_scroll_area(layout_type=True)
        # Define scroll area for poi tab
        sa_main_poi, layout_main_poi = Util.create_scroll_area(layout_type=True)
        
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
            plot_layout_val = QHBoxLayout()
            label_layout_val = QVBoxLayout()

            pw = BoxPlotWidget(data=d, groups=g)
            pw.setTitle(f"{channels[i]} Analysis")
            pw.laxis.setLabel("Foci/Nucleus")
            plot_layout_val.addWidget(pw)
            # Create the line to add
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            # Get plotting data of BoxPlot
            p_data = pw.p_data
            # Iterate over groups
            for j in range(len(g)):
                label_layout_val.addWidget(QLabel(
                    f"<strong>Group: {g[j]}</strong>"
                ))
                label_layout_val.addWidget(QLabel(
                    f"Values (w/o Outliers): {p_data[j]['number']}"
                ))
                label_layout_val.addWidget(QLabel(
                    f"Average: {p_data[j]['average']:.2f}"
                ))
                label_layout_val.addWidget(QLabel(
                    f"Median: {p_data[j]['median']}"
                ))
                label_layout_val.addWidget(QLabel(
                    f"IQR: {p_data[j]['iqr']}"
                ))
                label_layout_val.addWidget(QLabel(
                    f"Outliers: {len(p_data[j]['outliers'])}"
                ))
            plot_layout_val.addLayout(label_layout_val)
            layout_main_val.addLayout(plot_layout_val)
            #layout_main_val.addWidget(sa)
            if i < len(channels) - 1:
                layout_main_val.addWidget(line)
            # Add bool to check if a line was already created
            check = False
            # Create Poisson Plot for channel
            for group, values in data.items():
                # Create layout to add plot and labels
                plot_layout_poi = QHBoxLayout()
                label_layout_poi = QVBoxLayout()
                poiss = PoissonPlotWidget(data=values, label=group)
                poiss.setTitle(f"{group} - Comparison to Poisson Distribution")
                # Create the line to add
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setFrameShadow(QFrame.Sunken)
                # Add poisson plot
                plot_layout_poi.addWidget(poiss)
                # Add additional information
                label_layout_poi.addWidget(QLabel(f"Values: {len(values)}"))
                label_layout_poi.addWidget(QLabel(f"Average: {np.average(values):.2f}"))
                label_layout_poi.addWidget(QLabel(f"Min.: {np.amin(values)}"))
                label_layout_poi.addWidget(QLabel(f"Max.: {np.amax(values)}"))
                layout_main_poi.addLayout(label_layout_val)

                if i == len(channels) - 1:
                    if not check:
                        layout_main_poi.addWidget(line)
                        check = True
                else:
                    self.ui.dist_par.addWidget(line)
                self.ui.dist_par.addStretch(1)
        """