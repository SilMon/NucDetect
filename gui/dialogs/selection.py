# pyright: reportAttributeAccessIssue=false
# ^ PyQt5's stubs nest enum members inside their enum class (Qt.ItemDataRole.DisplayRole)
# while the C++ runtime also exposes them flat on Qt, which is what this file uses. The
# code is correct PyQt5 and a rewrite to the scoped form was declined -- PyQt6 is not
# planned (Romano, 2026-08-13). Suppressed at FILE level only because every hit of this
# rule here is that stub artefact; measured, not assumed. Re-check with the rule enabled
# before adding attribute access to a non-Qt object in this file.
from functools import partial
from typing import Dict, List, Any, Tuple

from PyQt5 import QtCore, uic
from PyQt5.QtCore import QItemSelectionModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QDialog, QCheckBox, QProgressBar

from gui import Paths
from gui import Util
from gui.Util import create_image_item_list_from
from core.database.connections import Requester
from gui.definitions.icons import Icon
from gui.loader import Loader


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
        # Annotated Any deliberately -- see the comment on the same assignment in
        # gui/NucDetectAppQT.py: uic has no stubs, so the inferred type is "Unknown | None"
        self.ui: Any = uic.loadUi(Paths.ui_experiment_selection_dial, self)
        # AFTER loadUi, not before: loadUi applies the windowTitle stored in the .ui file, so a
        # title set ahead of it is discarded. ImageSelectionDialog below has always done it in
        # this order
        self.setWindowTitle("Experiment Selection")
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint)
        # Load available experiments
        exps = self.requester.get_all_experiments()
        # Add experiments to combo box
        for experiment in exps:
            self.ui.cbx_exp.addItem(experiment)
        # A fresh database has no experiments, and exps[0] made opening this dialog an IndexError
        # before the window was ever shown -- reachable from the statistics button on a new install
        if exps:
            self.on_experiment_selection_change(exps[0])
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
        channels = [x for x in channels if x != main]
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
            # The check box is bound into the connection rather than recovered with sender().
            # sender() returns Optional[QObject]: it is None whenever the slot is reached outside
            # signal delivery -- a direct call, a test, a singleShot wrapper -- and it carries no
            # text()/isChecked(), so the slot silently depended on being invoked from exactly one
            # connection. partial binds the widget at connect time, inside the loop, so each box
            # gets its own
            cbx_temp.stateChanged.connect(partial(self.on_checkbox_change, cbx_temp))
            self.check_boxes.append(cbx_temp)
        self.sel_exp = exp

    def on_checkbox_change(self, cbx: QCheckBox, state: int = 0) -> None:
        """
        Method to react to selection changes for checkboxes

        :param cbx: The check box whose state changed, bound at connection time
        :param state: The Qt check state, as emitted by stateChanged. Unused -- isChecked() is read
        from the box itself, so the slot behaves the same when called directly
        :return: None
        """
        # Change stored information
        self.active_channels[cbx.text()] = cbx.isChecked()

    def clear_vbox(self) -> None:
        """
        Method to remove all checkboxes from the dialog

        :return: None
        """
        # deleteLater, not just removeWidget: QLayout.removeWidget unparents the widget from the
        # LAYOUT only. The QCheckBox keeps its parent and stays visible at its last position, so
        # switching experiment stacked the old channel boxes on top of the new ones
        for item in self.check_boxes:
            self.ui.vb_channels.removeWidget(item)
            item.setParent(None)
            item.deleteLater()
        self.check_boxes.clear()

    def get_selected_experiment(self) -> str:
        """
        Method to get the experiment selected in this dialog

        :return: The name of the selected experiment
        """
        return self.sel_exp

    def get_active_channels(self) -> Dict[str, bool]:
        """
        Method to get the channels activated in this dialog

        :return: Dictionary mapping channel name to its activation state
        """
        return self.active_channels


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
        # No self.ui / self.prg_bar placeholders here: initialize_ui() runs at the end of this
        # constructor and assigns both, so a None seeded first is never observable -- and it
        # contradicted the annotations on the real assignments (self.prg_bar: QProgressBar)
        # No self.img_model placeholder either -- initialize_ui() assigns it unconditionally
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
        # Annotated Any deliberately -- see the comment on the same assignment in
        # gui/NucDetectAppQT.py: uic has no stubs, so the inferred type is "Unknown | None"
        self.ui: Any = uic.loadUi(Paths.ui_img_sel_dial, self)
        self.prg_bar: QProgressBar = self.ui.prg_bar
        self.img_model = QStandardItemModel()
        self.ui.lv_images.setModel(self.img_model)
        self.ui.lv_images.setIconSize(Icon.get_icon_size("LIST_ITEM"))
        self.update_timer = Loader(self.images, feedback=self.load_images, processing=create_image_item_list_from)
        self.prg_bar.setValue(0)
        self.prg_bar.setMaximum(100)
        self.ui.lv_images.setEnabled(False)
        self.ui.buttonBox.setEnabled(False)
        self.setStyleSheet(Util.load_stylesheet("main.css"))
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
        self.prg_bar.setValue(int(self.update_timer.percentage * 100))
        if not items:
            # Enable image list
            self.ui.lv_images.setEnabled(True)
            self.ui.buttonBox.setEnabled(True)
            # Select images that are marked as selected
            for row in range(self.img_model.rowCount()):
                item = self.img_model.item(row)
                # Select marked images
                if item.data()["key"] in self.selected_images:
                    # index(), not createIndex(): createIndex builds an index with no internal
                    # pointer, so itemFromIndex() and model.data() on it both return None. The
                    # selection itself works either way -- QItemSelectionModel matches by
                    # row/column -- but anything that later resolves the index to its item does not
                    index = self.img_model.index(row, 0)
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
