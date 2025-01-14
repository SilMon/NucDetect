import json
import os
from typing import Dict, Union, List

import pyqtgraph as pg
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QDialog, QWidget, QScrollArea, QSizePolicy, QVBoxLayout, QMessageBox

import Paths
from definitions.icons import Icon
from settings.Settings import SettingsShowWidget, SettingsSlider, SettingsDial, SettingsSpinner, SettingsDecimalSpinner, \
    SettingsText, SettingsComboBox, SettingsCheckBox

pg.setConfigOptions(imageAxisOrder='row-major')


class AnalysisSettingsDialog(QDialog):

    def __init__(self, all_=False, settings=None, *args, **kwargs):
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
            "re-analyse": self.cbx_reanalyse.isChecked(),
            "use_pre-processing": self.cbx_preproc.isChecked(),
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
            "main": abs(self.ui.main_channel_btn_group.checkedId()) - 2,
            "analysis_settings": {
                "method": self.ui.detection_method_btn_group.checkedButton().text().lower(),
                "dots_per_micron": self.spbx_mmpd.value()
            }
        }

    def initialize_ui(self) -> None:
        """
        Method to initialize the ui

        :return: None
        """
        # Load UI definition
        self.ui = uic.loadUi(Paths.ui_analysis_settings_dial, self)
        # Load css file
        self.ui.setStyleSheet(open(os.path.join(Paths.css_dir, "main.css")).read())
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

class SettingsDialog(QDialog):
    """
    Class to display a settings window, dynamically generated from a JSON file
    """

    def __init__(self, inserter, parent: QWidget = None):
        super(SettingsDialog, self).__init__(parent)
        self.data = {}
        self.changed = {}
        self.ui = None
        self.json = None
        self.url = None
        self.inserter = inserter
        self._initialize_ui()

    def _initialize_ui(self) -> None:
        self.ui = uic.loadUi(Paths.ui_settings_dial, self)
        # Load css file
        self.setWindowFlags(
            self.windowFlags() |
            QtCore.Qt.WindowSystemMenuHint |
            QtCore.Qt.WindowMinMaxButtonsHint |
            QtCore.Qt.Window
        )
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowTitle("Settings")
        self.setStyleSheet(open(os.path.join(Paths.css_dir, "settings.css"), "r").read())
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
        self.close()

    def show_warning_dialog(self, msg: str):
        """
        Method to show a warning dialog

        :param msg: The message to display
        :return: The code returned by the dialog
        """
        msbbox = QMessageBox()
        msbbox.setIcon(QMessageBox.Warning)
        msbbox.setWindowIcon(Icon.get_icon("LOGO"))
        msbbox.setStyleSheet(open(os.path.join(Paths.css_dir, "messagebox.css"), "r").read())
        msbbox.addButton(QMessageBox.Yes)
        msbbox.addButton(QMessageBox.No)
        msbbox.setWindowTitle("Warning: Permanent removal of stored data imminent")
        msbbox.setText(msg)
        return msbbox.exec()

    def reset_database(self) -> None:
        """
        Method to reset the database

        :return: None
        """
        if self.show_warning_dialog("This action will erase all saved data. Are you sure?") == QMessageBox.Yes:
            print("Database erased")
            self.inserter.reset_database()

    def reset_analysis_data(self) -> None:
        """
        Method to reset the analysis data

        :return: None
        """
        if self.show_warning_dialog("This action will erase all analysis data. Are you sure?") == QMessageBox.Yes:
            print("Analysis data erased")
            self.inserter.reset_analysis_data()

    def reset_log_file(self) -> None:
        """
        Method to reset the log file

        :return: None
        """
        if self.show_warning_dialog("This action will erase all saved logs. Are you sure?") == QMessageBox.Yes:
            print("Log file erased")
            with open(Paths.log_path, "w") as log_file:
                log_file.write("")

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
            kernel.setObjectName("SettingsTabWidget")
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
                base = tab.findChildren(QVBoxLayout, "base")
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
        :param value: The value of the widget. Type depends on widget type
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
