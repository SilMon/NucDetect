import json
import os
from typing import Dict, Union, List

from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QDialog, QWidget, QScrollArea, QSizePolicy, QVBoxLayout, QMessageBox

import gui.Paths as gpaths
from gui import Util
from core.logging_config import get_logger, reset_log_file
from gui.definitions.icons import Icon
from gui.settings.Widgets import SettingsSlider, SettingsDial, SettingsSpinner, SettingsDecimalSpinner, \
    SettingsText, SettingsComboBox, SettingsCheckBox

LOGGER = get_logger(__name__)


class AnalysisSettingsDialog(QDialog):

    def __init__(self, all_=False, settings=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if settings is None:
            # The default is kept for the signature's shape only -- initialize_ui indexes this
            # dictionary immediately, so a None reached the user as a bare TypeError
            raise ValueError("AnalysisSettingsDialog requires a settings dictionary")
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
            "main": self.get_main_channel_index(),
            "analysis_settings": {
                "method": self.get_detection_method(),
                "dots_per_micron": self.spbx_mmpd.value(),
                "use_smoothing": self.cbx_noise.isChecked(),
                "smoothing_method": self.cmbx_noise.currentText(),
                "use_background_reduction": self.cbx_bckg.isChecked(),
                "background_reduction_method": self.cmbx_bckg.currentText()
            }
        }

    def get_main_channel_index(self) -> int:
        """
        Method to get the index of the channel selected as main channel

        The ids are assigned explicitly in initialize_ui, so this is a plain lookup rather than
        arithmetic on Qt's auto-assigned negative ids

        :return: The index of the main channel, 0 if nothing is checked
        """
        index = self.ui.main_channel_btn_group.checkedId()
        if index < 0:
            # Unreachable while rbtn_three carries checked=true in the .ui, which is why this
            # reports rather than raises -- an unchecked group must not silently mean "last channel"
            LOGGER.warning("No main channel is selected -- falling back to the first channel")
            return 0
        return index

    def get_detection_method(self) -> str:
        """
        Method to get the selected detection method

        :return: The method as lowercase string, the combined method if nothing is checked
        """
        button = self.ui.detection_method_btn_group.checkedButton()
        if button is None:
            # Masked while rbtn_combined carries checked=true in the .ui; without the guard this is
            # an AttributeError in the middle of reading the dialog result
            LOGGER.warning("No detection method is selected -- falling back to the combined method")
            return "combined"
        return button.text().lower()

    def initialize_ui(self) -> None:
        """
        Method to initialize the ui

        :return: None
        """
        # Load UI definition
        self.ui = uic.loadUi(gpaths.ui_analysis_settings_dial, self)
        # Load css file
        self.ui.setStyleSheet(Util.load_stylesheet("main.css"))
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
        # Qt numbers the buttons -2, -3, ... in the order they appear in the .ui file, and the old
        # `abs(checkedId()) - 2` was correct only because that order happens to match the channel
        # order. Explicit ids move that contract out of the .ui and into the code
        for index, button in enumerate(channel_main):
            self.ui.main_channel_btn_group.setId(button, index)
        # Both values below come from a user-editable JSON file and index fixed-size widget lists
        names = self.settings["names"].split(";")
        if len(names) > len(channels):
            LOGGER.warning(f"{len(names)} channel names configured, but the dialog has "
                           f"{len(channels)} channels -- the surplus is ignored")
        for name in range(min(len(names), len(channels))):
            channels[name].setChecked(True)
            channel_names[name].setEnabled(True)
            channel_names[name].setText(names[name])
        main_channel = self.settings["main_channel"]
        if not 0 <= main_channel < len(channel_main):
            LOGGER.warning(f"Configured main channel {main_channel} is outside the "
                           f"0-{len(channel_main) - 1} range -- falling back to the first channel")
            main_channel = 0
        channel_main[main_channel].setChecked(True)
        # Fill the pre-processing combo boxes
        self.ui.cmbx_noise.addItems(("Gaussian", "average",
                                     "median", "Total Variation Denoising",
                                     "Bilateral Denoising", "Wavelet Denoising"))
        self.ui.cmbx_bckg.addItems(("White Top-Hat", "Unsharp Masking", "Butterworth-Filtering"))
        # Bind the checkboxes for the pre-processing methods
        self.ui.cbx_noise.toggled.connect(self.ui.cmbx_noise.setEnabled)
        self.ui.cbx_bckg.toggled.connect(self.ui.cmbx_bckg.setEnabled)
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

class SettingsDialog(QDialog):
    """
    Class to display a settings window, dynamically generated from a JSON file
    """

    def __init__(self, inserter, parent: QWidget = None):
        super(SettingsDialog, self).__init__(parent)
        self.data = {}
        self.changed = {}
        # The tab each section was built into, and the section each menu point belongs to. The
        # second is what lets menupoint_changed write into the same nested shape add_menu_point uses
        self.tabs = {}
        self.sections = {}
        self.ui = None
        self.json = None
        self.url = None
        self.inserter = inserter
        self._initialize_ui()

    def _initialize_ui(self) -> None:
        self.ui = uic.loadUi(gpaths.ui_settings_dial, self)
        # Load css file
        self.setWindowFlags(
            self.windowFlags() |
            QtCore.Qt.WindowSystemMenuHint |
            QtCore.Qt.WindowMinMaxButtonsHint |
            QtCore.Qt.Window
        )
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowTitle("Settings")
        self.setStyleSheet(Util.load_stylesheet("main.css"))
        self.setModal(True)
        self.ui.btn_reset_db.clicked.connect(self.reset_database)
        self.ui.btn_reset_an.clicked.connect(self.reset_analysis_data)
        self.ui.btn_reset_log.clicked.connect(self.reset_log_file)
        # TODO implement program settings and chosen presets

    def accept(self):
        # Update the database to reflect the changes made
        for key, value in self.changed.items():
            self.inserter.update_setting(key, value[0])
        self.inserter.commit()
        # Save the menu to JSON
        self.save_menu_settings()
        # super().accept(), not close(): close() makes exec() return Rejected, so no caller could
        # tell OK from Cancel while these writes had already happened
        super().accept()

    def show_warning_dialog(self, msg: str):
        """
        Method to show a warning dialog

        :param msg: The message to display
        :return: The code returned by the dialog
        """
        msbbox = QMessageBox()
        msbbox.setIcon(QMessageBox.Warning)
        msbbox.setWindowIcon(Icon.get_icon("LOGO"))
        msbbox.setStyleSheet(Util.load_stylesheet("messagebox.css"))
        msbbox.addButton(QMessageBox.Yes)
        msbbox.addButton(QMessageBox.No)
        # The first button added is the default one, so Return on a focused dialog erased the
        # database. Every caller of this method is destructive and irreversible
        msbbox.setDefaultButton(QMessageBox.No)
        msbbox.setWindowTitle("Warning: Permanent removal of stored data imminent")
        msbbox.setText(msg)
        return msbbox.exec()

    def reset_database(self) -> None:
        """
        Method to reset the database

        :return: None
        """
        if self.show_warning_dialog("This action will erase all saved data. Are you sure?") == QMessageBox.Yes:
            LOGGER.info("Database erased")
            self.inserter.reset_database()

    def reset_analysis_data(self) -> None:
        """
        Method to reset the analysis data

        :return: None
        """
        if self.show_warning_dialog("This action will erase all analysis data. Are you sure?") == QMessageBox.Yes:
            LOGGER.info("Analysis data erased")
            self.inserter.reset_analysis_data()

    def reset_log_file(self) -> None:
        """
        Method to reset the log file

        :return: None
        """
        if self.show_warning_dialog("This action will erase all saved logs. Are you sure?") == QMessageBox.Yes:
            # Truncating the file directly is not enough: the logger holds it open, which keeps the
            # file locked on Windows and leaves the write position past the new end of file
            reset_log_file(gpaths.log_path)
            LOGGER.info("Log file erased")

    def initialize_from_file(self, url: str) -> None:
        """
        Method to initialize the settings window from a JSON file

        :param url: The URL leading to the JSON
        :return: None
        """
        if not url.lower().endswith(".json"):
            raise ValueError("Only JSON files can be loaded!")
        self.url = url
        with open(url, encoding="utf-8") as json_file:
            j_dat = json.load(json_file)
            self.json = j_dat
            for section, p in j_dat.items():
                self.add_menu_point(section, p)

    def add_section(self, section: str) -> QScrollArea:
        """
        Method to add a section to the settings

        :param section: The name of the section
        :return: The tab holding the section, newly built or the one already present
        """
        if section not in self.data:
            self.data[section] = {}
            tab = QScrollArea()
            tab.setWidgetResizable(True)
            kernel = QWidget()
            kernel.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Expanding
            )
            kernel.setObjectName("SettingsTabWidget")
            layout = QVBoxLayout()
            kernel.setLayout(layout)
            layout.setObjectName("base")
            tab.setWidget(kernel)
            self.ui.settings.addTab(tab, section)
            self.tabs[section] = tab
        return self.tabs[section]

    def add_menu_point(self, section: str, menupoint: Dict[str, Union[str, float, int]]) -> None:
        """
        Method to add a menu point to the settings section

        :param section: The name of the section
        :param menupoint: The menupoint to add
        :return: None
        """
        # add_section hands back the tab it built or already had, so there is nothing to search for
        tab = self.add_section(section)
        base = tab.findChildren(QVBoxLayout, "base")
        for mp in menupoint:
            t = mp["type"].lower()
            p = None
            if t == "slider":
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
            else:
                # Without this the unmatched `p = None` reached addWidget() as a TypeError, half-way
                # through building the dialog. A typo in settings.json costs one menu point, not the
                # whole settings window
                LOGGER.error(f"Unknown menu point type '{mp['type']}' for setting "
                             f"'{mp['id']}' in section '{section}' -- the menu point is skipped")
                continue
            # Registered only once the widget exists, so a skipped menu point contributes no value
            # that nothing on screen can edit
            self.data[section][mp["id"]] = mp["value"]
            self.sections[mp["id"]] = section
            base[0].addWidget(p)
        base[0].addStretch()

    def menupoint_changed(self, _id: str = None, value: Union[str, int, float] = None) -> None:
        """
        Method to detect value changes of the settings widgets

        :param _id: The id of the widget as str
        :param value: The value of the widget, wrapped in a list by the widgets' signal
        :return: None
        """
        # self.changed keeps the signal's list shape -- accept() and save_menu_settings() both
        # index [0] out of it
        self.changed[_id] = value
        # self.data is nested per section, the shape add_menu_point builds. Writing self.data[_id]
        # here left two incompatible layouts in one dictionary, and a consumer walking it per
        # section never saw an edited value
        section = self.sections.get(_id)
        if section is None:
            LOGGER.warning(f"Change reported for unknown setting '{_id}' -- not recorded in the "
                           f"section data")
            return
        self.data[section][_id] = value[0]

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
                with open(self.url, 'w', encoding="utf-8") as file:
                    json.dump(self.json, file)
        else:
            raise RuntimeError("Settings not initialized!")
