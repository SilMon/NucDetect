from __future__ import annotations

import csv
import os
import re
import shutil
import sqlite3
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from threading import Thread
from typing import Union, Dict, Iterable, List, Tuple

import PyQt5
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5.QtCore import QSize, pyqtSignal, QItemSelectionModel, QSortFilterProxyModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QHeaderView, QDialog, QSplashScreen, QMessageBox

from core.Detector import Detector
from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler
from gui import Paths, Util
from gui.Definitions import Icon
from gui.Dialogs import ExperimentDialog, ExperimentSelectionDialog, StatisticsDialog, ImgDialog, SettingsDialog, \
    AnalysisSettingsDialog, Editor
from gui.Util import create_image_item_list_from
from gui.loader import Loader

PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, False)
PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, False)
pg.setConfigOptions(imageAxisOrder='row-major')


class NucDetect(QMainWindow):
    """
    Created on 11.02.2019
    @author: Romano Weiss
    """
    prg_signal = pyqtSignal(str, float, float, str)
    selec_signal = pyqtSignal(bool)
    add_signal = pyqtSignal(str)
    aa_signal = pyqtSignal(int, int)
    executor = Thread()

    def __init__(self):
        """
        Constructor of the main window
        """
        QMainWindow.__init__(self)
        # Create working directories
        self.create_required_dirs()
        # Connect to database
        self.connection = sqlite3.connect(Paths.database)
        self.cursor = self.connection.cursor()
        # Create tables if they do not exists
        self.create_tables(self.cursor)
        # Load the settings from database
        self.settings = self.load_settings()
        # Create detector for analysis
        self.detector = Detector()
        # Initialize needed variables
        self.reg_images = []
        # Contains the displayed table data
        self.data = None
        # Contains data for the associated experiment
        self.cur_exp = None
        # Contains data of the loaded image
        self.cur_img = None
        # Contains the associated roi for the loaded image
        self.roi_cache = None
        # A list of all loaded image files -> Used for reloading
        self.loaded_files = []
        # Dict to convert md5 image hashes to file names
        self.hash_to_name = {}
        # Timer responsible for lazy loading
        self.update_timer = None
        self.unsaved_changes = False
        # Setup UI
        self._setup_ui()
        self.icon = Icon.get_icon("LOGO")
        self.setWindowTitle("NucDetect")
        self.setWindowIcon(self.icon)
        self.showMaximized()

    @staticmethod
    def create_tables(cursor: sqlite3.Cursor) -> None:
        """
        Method to create the tables in the database

        :param cursor: Cursor for the database
        :return: None
        """
        # Create the tables
        cursor.executescript(
            '''
            BEGIN TRANSACTION;
            CREATE TABLE IF NOT EXISTS "categories" (
                            "image"	INTEGER,
                            "category"	TEXT,
                            PRIMARY KEY("image","category")
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS "channels" (
                            "md5"	INTEGER,
                            "index_"	INTEGER,
                            "name"	INTEGER,
                            "active"	INTEGER,
                            "main"	INTEGER,
                            PRIMARY KEY("md5","index_")
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS "experiments" (
                            "name"	TEXT,
                            "details"	TEXT,
                            "notes"	TEXT,
                            PRIMARY KEY("name")
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS "groups" (
                            "image"	INTEGER,
                            "experiment"	INTEGER,
                            "name"	TEXT,
                            PRIMARY KEY("image","experiment")
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS "images" (
                            "md5"	TEXT,
                            "datetime"	TEXT,
                            "channels"	INTEGER NOT NULL,
                            "width"	INTEGER NOT NULL,
                            "height"	INTEGER NOT NULL,
                            "x_res"	INTEGER,
                            "y_res"	INTEGER,
                            "unit"	INTEGER,
                            "analysed"	INTEGER NOT NULL,
                            "settings"	TEXT,
                            "experiment"	TEXT,
                            "modified" INTEGER NOT NULL,
                            PRIMARY KEY("md5")
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS "points" (
                            "hash"	INTEGER,
                            "row"	INTEGER,
                            "column"	INTEGER,
                            "width"	INTEGER,
                            PRIMARY KEY("hash","row","column")
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS "roi" (
                            "hash"	INTEGER,
                            "image"	INTEGER,
                            "auto"	INTEGER,
                            "channel"	TEXT,
                            "center"	TEXT,
                            "width"	INTEGER,
                            "height"	INTEGER,
                            "associated"	INTEGER,
                            PRIMARY KEY("hash","image")
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS "settings" (
                            "key_"	TEXT,
                            "value"	TEXT,
                            "type_" TEXT,
                            PRIMARY KEY("key_")
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS "statistics" (
                            "hash"	INTEGER,
                            "image"	INTEGER,
                            "area"	INTEGER,
                            "intensity_average"	INTEGER,
                            "intensity_median"	INTEGER,
                            "intensity_maximum"	INTEGER,
                            "intensity_minimum"	INTEGER,
                            "intensity_std"	INTEGER,
                            "eccentricity"	INTEGER,
                            "roundness"	INTEGER,
                            "ellipse_center"	TEXT,
                            "ellipse_major"	INTEGER,
                            "ellipse_minor"	INTEGER,
                            "ellipse_angle"	INTEGER,
                            "ellipse_area"	INTEGER,
                            "orientation_vector"	TEXT,
                            "ellipticity"	INTEGER,
                            PRIMARY KEY("hash","image")
            ) WITHOUT ROWID;
            COMMIT;
            '''
        )
        # Create the standard settings
        cursor.executescript(
            '''
            BEGIN TRANSACTION;
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("logging", 1, "bool");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("exp_std_name", "Default", "str");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("names", "Red;Green;Blue", "str");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("main_channel", 2, "int");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("logging", 1, "bool");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("ml_analysis", 0, "bool");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("quality_check", 1, "bool");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("blob_min_sigma", 1, "float");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("blob_max_sigma", 4, "float");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("blob_num_sigma", 10, "int");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("blob_threshold", 0.15, "float");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("thresh_iterations", 10, "int");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("thresh_mask_size", 7, "int");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("thresh_percent_hmax", 0.05, "float");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("thresh_local_thresh_mult", 8, "int");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("thresh_max_mult", 2, "int");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("canny_sigma", 3, "float");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("canny_low_thresh", 0.10, "float");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("canny_up_thresh", 0.2, "float");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("fcn_certainty_nuclei", 0.95, "float");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("fcn_certainty_foci", 0.80, "float");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("show_ellipsis", 1, "bool");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("track_mouse", 1, "bool");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("quality_min_nuc_size", 750, "int");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("quality_min_foc_size", 8, "int");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("quality_max_nuc_size", 30000, "int");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("quality_max_foc_size", 280, "int");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("quality_max_foc_overlap", 0.75, "float");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("size_factor", 1, "float");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("cutoff", 0.03, "float");
            INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("num_threads", 8, "int");
            COMMIT;
            '''
        )

    @staticmethod
    def create_required_dirs() -> None:
        """
        Method to create the working dirs of this program

        :return: None
        """
        if not os.path.isdir(Paths.thumb_path):
            os.makedirs(Paths.thumb_path)
        if not os.path.isdir(Paths.nuc_detect_dir):
            os.makedirs(Paths.nuc_detect_dir)
        if not os.path.isdir(Paths.result_path):
            os.makedirs(Paths.result_path)
        if not os.path.isdir(Paths.images_path):
            os.makedirs(Paths.images_path)
            shutil.copy2(os.path.join(os.pardir, "demo.tif"), os.path.join(Paths.images_path, "demo.tif"))

    def load_settings(self) -> Dict:
        """
        Method to load the saved Settings
        :return: None
        """
        self.cursor.execute(
            "SELECT * FROM settings"
        )
        settings = {}
        for row in self.cursor.fetchall():
            settings[row[0]] = self.convert_to_type(row[1], row[2])
        return settings

    @staticmethod
    def convert_to_type(value: str, type_: str) -> Union[int, float, str, bool]:
        """
        Method to convert the given value into its specified type

        :param value: The value as string
        :param type_: The type as string
        :return: The converted type
        """
        if type_ == "int":
            return int(value)
        elif type_ == "float":
            return float(value)
        elif type_ == "bool":
            return bool(value)
        else:
            return value

    def closeEvent(self, event) -> None:
        """
        Will be called if the program window closes

        :param event: The closing event
        :return: None
        """
        self.on_close()
        event.accept()

    def _setup_ui(self) -> None:
        """
        Method to initialize the UI of the main window

        :return: None
        """
        self.ui = uic.loadUi(Paths.ui_main, self)
        # Initialization of the image list
        self.img_list_model = QStandardItemModel(self.ui.list_images)
        self.ui.list_images.setModel(self.img_list_model)
        self.ui.list_images.selectionModel().selectionChanged.connect(self.on_image_selection_change)
        self.ui.list_images.setWordWrap(True)
        self.ui.list_images.setIconSize(QSize(75, 75))
        # Initialization of the result table
        self.res_table_model = QStandardItemModel(self.ui.table_results)
        self.res_table_model.setHorizontalHeaderLabels(["Image Identifier", "ROI Identifier", "Center[(y, x)]",
                                                        "Area [px]", "Ellipticity[%]", "Or. Angle [deg]",
                                                        "Maj. Axis", "Min. Axis"])
        self.res_table_sort_model = TableFilterModel(self)
        self.res_table_sort_model.setSourceModel(self.res_table_model)
        self.ui.table_results.setModel(self.res_table_sort_model)
        self.ui.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Addition of on click listeners
        self.ui.btn_load.clicked.connect(self._show_loading_dialog)
        self.ui.btn_experiments.clicked.connect(self.show_experiment_dialog)
        self.ui.btn_save.clicked.connect(self.save_results)
        self.ui.btn_analyse.clicked.connect(self.analyze)
        self.ui.btn_images.clicked.connect(self.show_result_image)
        self.ui.btn_statistics.clicked.connect(self.show_statistics)
        self.ui.btn_categories.clicked.connect(self.show_categorization)
        self.ui.btn_settings.clicked.connect(self.show_settings)
        self.ui.btn_modify.clicked.connect(self.show_modification_window)
        self.ui.btn_analyse_all.clicked.connect(self.analyze_all)
        self.ui.btn_delete_from_list.clicked.connect(self.remove_image_from_list)
        self.ui.btn_clear_list.clicked.connect(self.clear_image_list)
        self.ui.btn_reload.clicked.connect(self.reload)
        # Add button icons
        self.ui.btn_load.setIcon(Icon.get_icon("FOLDER_OPEN"))
        self.ui.btn_experiments.setIcon(Icon.get_icon("FLASK"))
        self.ui.btn_save.setIcon(Icon.get_icon("SAVE"))
        self.ui.btn_images.setIcon(Icon.get_icon("MICROSCOPE"))
        self.ui.btn_statistics.setIcon(Icon.get_icon("CHART_BAR"))
        self.ui.btn_categories.setIcon(Icon.get_icon("LIST_UL"))
        self.ui.btn_settings.setIcon(Icon.get_icon("COGS"))
        self.ui.btn_modify.setIcon(Icon.get_icon("TOOLS"))
        self.ui.btn_analyse.setIcon(Icon.get_icon("HAT_WIZARD_BLUE"))
        self.ui.btn_analyse_all.setIcon(Icon.get_icon("HAT_WIZARD_RED"))
        self.ui.btn_delete_from_list.setIcon(Icon.get_icon("TIMES"))
        self.ui.btn_clear_list.setIcon(Icon.get_icon("TRASH_ALT"))
        self.ui.btn_reload.setIcon(Icon.get_icon("SYNC"))
        # Create signal for thread-safe gui updates
        self.prg_signal.connect(self._set_progress)
        self.selec_signal.connect(self._select_next_image)
        self.add_signal.connect(self.add_item_to_list)
        self.add_images_from_folder(Paths.images_path)

    def reload(self) -> None:
        """
        Method to reload the images folder

        :return: None
        """
        self.add_images_from_folder(Paths.images_path, reload=True)

    def on_image_selection_change(self) -> None:
        """
        Will be called if a new image is selected

        :return: None
        """
        for index in self.ui.list_images.selectionModel().selectedIndexes():
            self.cur_img = self.img_list_model.item(index.row()).data()
        if self.cur_img:
            ana = self.cur_img["analysed"]
            if ana:
                exp = self.cursor.execute("SELECT experiment FROM images WHERE md5=?",
                                          (self.cur_img["key"],)).fetchall()
                experiment = None
                # Check if image is part of an experiment
                if exp[0][0]:
                    # Get number of attached images
                    num_imgs = self.cursor.execute("SELECT COUNT(md5) FROM images WHERE experiment=?",
                                                   exp[0]).fetchall()[0][0]
                    msg = QMessageBox()
                    msg.setStyleSheet(open("messagebox.css", "r").read())
                    msg.setWindowIcon(QtGui.QIcon('logo.png'))
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("Experiment attached!")
                    msg.setText("")
                    msg.setInformativeText(f"The selected image is assigned to the experiment {exp[0][0]}, "
                                           f"with {num_imgs} attached images. Loading it can take up"
                                           f" to approx. {num_imgs * 5 / 60:.2f} min ({num_imgs * 5} secs). ")
                    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                    msg.button(QMessageBox.Ok).setText("Load Experiment")
                    msg.button(QMessageBox.Cancel).setText("Load Image Data")
                    ok = msg.exec_()
                    if ok == QMessageBox.Ok:
                        experiment = exp[0][0]
                self.prg_signal.emit(f"Loading data from database for {self.cur_img['file_name']}",
                                     0, 100, "")
                thread = Thread(target=self.load_saved_data, args=(experiment, ))
                thread.start()
            else:
                self.ui.lbl_status.setText("Program ready")
                self.res_table_model.setRowCount(0)
                self.enable_buttons(False, ana_buttons=False)
        else:
            self.ui.btn_analyse.setEnabled(False)

    def load_saved_data(self, experiment: str = None) -> None:
        """
        Method to load saved data from the database

        :param experiment: Name of the experimental data to load. None if only image data should be loaded
        :return: None
        """
        # Disable Buttons and list during loading
        self.enable_buttons(state=False)
        self.ui.list_images.setEnabled(False)
        # Load saved data from databank
        self.roi_cache = self.load_rois_from_database(self.cur_img["key"])
        # Create the result table from loaded data
        self.create_result_table_from_list(self.roi_cache, experiment)
        # Re-enable buttons and list
        self.ui.list_images.setEnabled(True)
        self.enable_buttons()
        self.prg_signal.emit(f"Data loaded from database for {self.cur_img['file_name']}",
                             100, 100, "")

    def show_experiment_dialog(self) -> None:
        """
        Method to show the experiment dialog

        :return: None
        """
        exp_dialog = ExperimentDialog(data={
            "keys": [x[0] for x in self.reg_images],
            "paths": [x[1] for x in self.reg_images]
        })
        code = exp_dialog.exec()
        if code == QDialog.Accepted:
            exp_dialog.accepted()

    def _show_loading_dialog(self) -> None:
        """
        Method to show a file loading dialog, which allows the user to select images.

        :return: None
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        pardir = os.getcwd()
        imgdir = os.path.join(os.path.dirname(pardir),
                              r"images")
        file_name, _ = QFileDialog.getOpenFileName(self, "Load images..", imgdir,
                                                   "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)",
                                                   options=options)
        if file_name:
            self.add_item_to_list(
                Util.create_list_item(file_name)
            )

    def add_images_from_folder(self, url: str, reload: bool = False) -> None:
        """
        Method to load a whole folder of images

        :param url: The path of the folder
        :param reload: Indicator if a reload occurs
        :return: None
        """
        start = time.time()
        paths = []
        for t in os.walk(url):
            tpaths = [os.path.join(t[0], x) for x in t[2]]
            paths.extend([x for x in tpaths if x not in self.loaded_files])
        self.loaded_files.extend(paths)
        # Check if the list of available files is larger than 25
        if len(self.loaded_files) > 25 and not reload:
            self.setEnabled(False)
            self.update_timer = Loader(self.loaded_files, feedback=self.add_image_items,
                                       processing=create_image_item_list_from)
        elif len(self.loaded_files) > 25:
            self.setEnabled(False)
            self.update_timer = Loader(paths, feedback=self.add_image_items, processing=create_image_item_list_from)
        else:
            self.enable_buttons(True)
            items = Util.create_image_item_list_from(paths, indicate_progress=True)
            for item in items:
                self.add_item_to_list(item)
            print(f"Finished loading: {time.time() - start: .2f} secs\n")

    def add_image_items(self, items: List[QStandardItem]) -> None:
        """
        Method to add loaded image items to

        :param items: The items to add
        :return: None
        """
        # Indicate update in console
        print(f"{self.update_timer.percentage * 100:.2f}% of images added to image list")
        # Indicate update in progress bar
        self.prg_signal.emit(f"{self.update_timer.items_loaded} images added to image list",
                             self.update_timer.percentage * 100, 100, "")
        for item in items:
            self.add_item_to_list(item)
        if not items:
            self.setEnabled(True)

    def add_item_to_list(self, item: QStandardItem) -> None:
        """
        Utility method to add an item to the image list

        :param item: The item to add
        :return: None
        """
        if item is not None:
            path = item.data()["path"]
            key = item.data()["key"]
            name = item.data()["file_name"]
            self.hash_to_name[key] = name
            self.img_list_model.appendRow(item)
            self.reg_images.append((key, path))
            if not self.cursor.execute(
                    "SELECT * FROM images WHERE md5 = ?",
                    (key,)
            ).fetchall():
                d = Detector.get_image_data(path)
                self.cursor.execute(
                    "INSERT OR IGNORE INTO images VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (key, d["datetime"], d["channels"], d["width"], d["height"],
                     str(d["x_res"]), str(d["y_res"]), d["unit"], 0, -1, None, 0)
                )
            self.connection.commit()

    def remove_image_from_list(self) -> None:
        """
        Method to remove an loaded image from the file list.

        :return: None
        """
        cur_ind = self.ui.list_images.currentIndex()
        data = self.img_list_model.item(cur_ind.row()).data()
        self.reg_images.remove((data["key"], data["path"]))
        self.img_list_model.removeRow(cur_ind.row())

    def clear_image_list(self) -> None:
        """
        Method to clear the list of loaded images

        :return: None
        """
        self.img_list_model.clear()
        self.reg_images.clear()

    def analyze(self) -> None:
        """
        Method to analyze an loaded image

        :return: None
        """
        if not self.cur_img:
            self.selec_signal.emit(True)
        # Get settings for this analysis
        anal_sett_dial = AnalysisSettingsDialog(settings=self.settings)
        code = anal_sett_dial.exec()
        if code == QDialog.Accepted:
            settings = anal_sett_dial.get_data()
            an_sett = settings["analysis_settings"]
            settings["analysis_settings"].update({x: y for (x, y) in self.settings.items() if x not in an_sett})
        else:
            # If the dialog was rejected, abort analysis
            return
        self.res_table_model.setRowCount(0)
        self.prg_signal.emit(f"Analysing {self.cur_img['file_name']}",
                             0, 100, "")
        thread = Thread(target=self.analyze_image,
                        args=(self.cur_img["path"],
                              "Analysis finished in {} -- Program ready",
                              100, 100, settings))
        thread.start()

    def analyze_image(self, path: str, message: str,
                      percent: Union[int, float],
                      maxi: Union[int, float],
                      analysis_settings: Dict[str, Union[int, float, str]]) -> None:
        """
        Method to analyse the image given by path

        :param path: The path leading to the image
        :param message: The message to display above the progress bar
        :param percent: The value of the progress bar
        :param maxi: The maximum of the progress bar
        :param analysis_settings: The settings to apply to this analysis
        :return: None
        """
        self.enable_buttons(False)
        self.ui.list_images.setEnabled(False)
        start = time.time()
        self.prg_signal.emit("Starting analysis", 0, maxi, "")
        self.unsaved_changes = True
        self.prg_signal.emit("Analysing image", maxi * 0.05, maxi, "")
        data = self.detector.analyse_image(path, settings=analysis_settings)
        self.roi_cache = data["handler"]
        s0 = time.time()
        self.prg_signal.emit(f"Ellipse parameter calculation", maxi * 0.75, maxi, "")
        for roi in self.roi_cache:
            if roi.main:
                roi.calculate_ellipse_parameters()
        self.prg_signal.emit("Creating result table", maxi * 0.65, maxi, "")
        print(f"Calculation of ellipse parameters: {time.time() - s0:.4f}")
        self.create_result_table_from_list(data["handler"])
        print(f"Creation result table: {time.time() - s0:.4f} secs")
        self.prg_signal.emit("Checking database", maxi * 0.9, maxi, "")
        s1 = time.time()
        self.save_rois_to_database(data)
        print(f"Writing to database: {time.time() - s1:.4f} secs")
        self.prg_signal.emit(message.format(f"{time.time() - start:.2f} secs"),
                             percent, maxi, "")
        self.create_result_table_from_list(self.roi_cache)
        self.enable_buttons()
        self.ui.list_images.setEnabled(True)

    def analyze_all(self) -> None:
        """
        Method to perform concurrent batch analysis of registered images

        :return: None
        """
        self.enable_buttons(False)
        self.ui.list_images.setEnabled(False)
        self.unsaved_changes = True
        # Get settings for this analysis
        anal_sett_dial = AnalysisSettingsDialog(all_=True, settings=self.settings)
        code = anal_sett_dial.exec()
        if code == QDialog.Accepted:
            settings = anal_sett_dial.get_data()
            an_sett = settings["analysis_settings"]
            settings["analysis_settings"].update({x: y for (x, y) in self.settings.items() if x not in an_sett})
        else:
            # If the dialog was rejected, abort analysis
            return
        thread = Thread(target=self._analyze_all, args=(settings, ))
        thread.start()

    def _analyze_all(self, settings: Dict[str, Union[int, float, str, Iterable]], batch_size: int = 20) -> None:
        """
        Method to perform concurrent batch analysis of registered images

        :param settings: The settings for this analysis, e.g. channel names, active channels ect.
        :param batch_size: The number of images that are loaded parallel
        :return: None
        """
        start_time = time.time()
        max_workers = 1 if settings["type"] else settings["analysis_settings"]["num_threads"]
        with ProcessPoolExecutor(max_workers=max_workers) as e:
            self.res_table_model.setRowCount(0)
            self.res_table_model.setColumnCount(2)
            self.res_table_model.setHorizontalHeaderLabels(["Image Name", "Image Hash", "Number of ROI"])
            logstate = settings["analysis_settings"]["logging"]
            settings["analysis_settings"]["logging"] = False
            self.prg_signal.emit("Starting multi image analysis", 0, 100, "")
            paths = []
            for ind in range(self.img_list_model.rowCount()):
                data = self.img_list_model.item(ind).data()
                if not bool(data["analysed"]) or settings["re-analyse"]:
                    paths.append(data["path"])
            ind = 1
            cur_batch = 1
            curind = 0
            # Define needed batch variables
            start = batch_size + 1 if batch_size < len(paths) else len(paths) + 1
            stop = len(paths) + batch_size if len(paths) > batch_size else len(paths) + 2
            step = batch_size
            # Clear results table

            # Iterate over all images in batches
            for b in range(start, stop, step):
                s2 = time.time()
                tpaths = paths[curind:b if b < len(paths) else len(paths)]
                t_setts = [settings for _ in range(len(paths))]
                res = e.map(self.detector.analyse_image, tpaths, t_setts)
                maxi = len(paths)

                for r in res:
                    self.prg_signal.emit(f"Analysed images: {ind}/{maxi}",
                                         ind, maxi, "")
                    self.save_rois_to_database(r, all_=True)
                    self.res_table_model.appendRow(
                        [QStandardItem(self.hash_to_name[r["handler"].ident]),
                        QStandardItem(r["handler"].ident),
                         QStandardItem(str(len(r["handler"])))]
                    )
                    ind += 1
                print(f"Analysed batch {cur_batch} in {time.time() - s2:.3f} secs\t"
                      f"Total: {time.time() - start_time:.3f} secs")
                curind = b
                cur_batch += 1
            self.enable_buttons()
            self.ui.list_images.setEnabled(True)
            settings["analysis_settings"]["logging"] = logstate
            self.prg_signal.emit("Analysis finished -- Program ready",
                                 100,
                                 100, "")
            # Change the status of list items to reflect that they were analysed
            for ind in range(self.img_list_model.rowCount()):
                item = self.img_list_model.item(ind)
                data = item.data()
                data["analysed"] = True
                item.setData(data)
            self.selec_signal.emit(True)
        print(f"Total analysis time: {time.time() - start_time:.3f} secs")

    @staticmethod
    def save_rois_to_database(data: Dict[str, Union[ROIHandler, np.ndarray, Dict[str, str]]],
                              all_: bool = False) -> None:
        """
        Method to save the data stored in the ROIHandler rois to the database

        :param data: The data dict returned by the Detector class
        :param all_: Deactivates printing to console
        :return: None
        """
        con = sqlite3.connect(Paths.database)
        curs = con.cursor()
        key = data["id"]
        # Delete existing analysis data if image was already analysed
        if curs.execute(
                "SELECT analysed FROM images WHERE md5 = ?",
                (key,)
        ).fetchall()[0][0]:
            for h in curs.execute(
                    "SELECT hash FROM roi WHERE image = ?",
                    (key,)
            ).fetchall():
                # Delete saved points
                curs.execute(
                    "DELETE FROM points where hash = ?",
                    (h[0],)
                )
                # Delete saved statistics
                curs.execute(
                    "DELETE FROM statistics where hash = ?",
                    (h[0], )
                )
            curs.execute(
                "DELETE FROM roi WHERE image = ?",
                (key,)
            )
        # Check if image should be added to experiment
        if data["add to experiment"]:
            exp_data = data["experiment details"]
            # Add new experiment
            curs.execute(
                "INSERT OR IGNORE INTO experiments VALUES (?, ?, ?)",
                (exp_data["name"], exp_data["details"], exp_data["notes"])
            )
            # Add image to standard group
            curs.execute(
                "INSERT OR IGNORE INTO groups VALUES(?, ?, ?)",
                (key, exp_data["name"], "Standard")
            )
            # Update experiment column in images table
            curs.execute(
                "UPDATE images SET experiment = ? WHERE md5 = ?",
                (exp_data["name"], key)
            )
        # Update channel info
        for ind in range(len(data["names"])):
            active = data["active channels"][ind]
            name = data["names"][ind]
            main = True if data["main channel"] == ind else False
            curs.execute(
                "INSERT OR IGNORE INTO channels VALUES (?, ?, ?, ?, ?)",
                (key, ind, name, active, main)
            )
        # Save data for detected ROI
        roidat = []
        pdat = []
        elldat = []
        # Collect data
        for roi in data["handler"].rois:
            dim = roi.calculate_dimensions()
            ellp = roi.calculate_ellipse_parameters()
            # Get the channel of the roi
            stats = roi.calculate_statistics(data["channels"][data["handler"].idents.index(roi.ident)])
            asso = hash(roi.associated) if roi.associated is not None else None
            roidat.append((hash(roi), key, True, roi.ident, str(dim["center"]), dim["width"], dim["height"], asso))
            for p in roi.area:
                pdat.append((hash(roi), p[0], p[1], p[2]))
            elldat.append(
                (hash(roi), key, stats["area"], stats["intensity average"], stats["intensity median"],
                 stats["intensity maximum"], stats["intensity minimum"], stats["intensity std"],
                 ellp["eccentricity"], ellp["roundness"], str(ellp["center"]), ellp["major_axis"], ellp["minor_axis"],
                 ellp["angle"], ellp["area"], str(ellp["orientation"]), ellp["shape_match"])
            )
        # Save data to database
        curs.executemany(
            "INSERT OR IGNORE INTO roi VALUES (?, ?, ?, ?, ?, ?, ?,?)",
            roidat
        )
        curs.executemany(
            "INSERT OR IGNORE INTO points VALUES (?, ?, ?, ?)",
            pdat
        )
        curs.executemany(
            "INSERT OR IGNORE INTO statistics VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            elldat
        )
        curs.execute(
            "UPDATE images SET analysed = ? WHERE md5 = ?",
            (True, key)
        )
        con.commit()
        con.close()
        if not all_:
            print("ROI saved to database")

    def load_rois_from_database(self, md5: str) -> ROIHandler:
        """
        Method to load all rois associated with this image

        :param md5: The md5 hash of the image
        :return: A ROIHandler containing all roi
        """
        self.prg_signal.emit(f"Loading data",
                             0, 100, "")
        con = sqlite3.connect(Paths.database)
        crs = con.cursor()
        rois = ROIHandler(ident=md5)
        entries = crs.execute(
            "SELECT * FROM roi WHERE image = ?",
            (md5,)
        ).fetchall()
        names = crs.execute(
            "SELECT * FROM channels WHERE md5 = ?",
            (md5,)
        ).fetchall()
        for name in names:
            rois.idents.insert(name[1], name[2])
        main_ = []
        sec = []
        statkeys = ("area", "intensity average",
                    "intensity median", "intensity maximum",
                    "intensity minimum", "intensity std", "eccentricity", "roundness")
        ellkeys = ("center", "major_axis", "minor_axis", "angle",
                   "orientation", "area", "shape_match")
        ind = 1
        max = len(entries)
        for entry in entries:
            self.prg_signal.emit(f"Loading ROI:  {ind}/{max}",
                                 ind, max, "")
            temproi = ROI(channel=entry[3], main=entry[7] is None,
                          auto=bool(entry[2]), associated=entry[7])
            stats = crs.execute(
                "SELECT * FROM statistics WHERE hash = ?",
                (entry[0],)
            ).fetchall()[0]
            temproi.stats = dict(zip(statkeys, stats[2:10]))
            if temproi.main:
                main_.append(temproi)
            else:
                sec.append(temproi)
            rle = []
            for p in crs.execute(
                    "SELECT * FROM points WHERE hash = ?",
                    (entry[0],)
            ).fetchall():
                rle.append((p[1], p[2], p[3]))
            temproi.add_to_area(rle)
            if temproi.main:
                center = re.search(r"\((\d*)\D*(\d*)\)?", stats[10])
                center = (int(center.group(1)), int(center.group(2)))
                major = stats[11]
                minor = stats[12]
                angle = stats[13]
                area = stats[14]
                ov = re.search(r"\((-?\d*\.\d*)\D*(-?\d*\.\d*)\)?", stats[15])
                ov = (float(ov.group(1)), float(ov.group(2)))
                ellip = stats[16]
            else:
                center = (None, None)
                major = None
                minor = None
                angle = None
                area = None
                ov = None, None
                ellip = None
            ellp = (center, major, minor, angle, ov, area, ellip)
            temproi.ell_params = dict(zip(ellkeys, ellp))
            temproi.id = entry[0]
            rois.add_roi(temproi)
            ind += 1
        for m in main_:
            for s in sec:
                if s.associated == hash(m):
                    s.associated = m
        print(f"Loaded {len(rois)} roi from database")
        return rois

    def create_result_table_from_list(self, handler: ROIHandler, experiment: str = None) -> None:
        """
        Method to create the result table from a list of rois

        :param handler: The handler containing the rois
        :param experiment: The experiment to load
        :return: None
        """
        self.prg_signal.emit(f"Create Result Table",
                             0, 100, "")
        self.res_table_model.setRowCount(0)
        connection = sqlite3.connect(Paths.database)
        cursor = connection.cursor()
        # Get all available channels for the image
        chans = cursor.execute("SELECT * FROM channels WHERE md5=?", (handler.ident, )).fetchall()
        # Remove main channel and  from list
        chans = [x[2] for x in chans if not bool(x[4]) and bool(x[3])]
        # Sort channel list
        chans = sorted(chans)
        # Create header
        header = ["Image Name", "Image Identifier", "Group", "ROI Identifier", "Center[(y, x)]", "Area [px]",
                  "Ellipticity[%]", "Or. Angle [deg]", "Maj. Axis", "Min. Axis"]
        if experiment:
            # Get all assigned images
            num_imgs = cursor.execute("SELECT COUNT(md5) FROM images WHERE experiment=?",
                                      (experiment, )).fetchall()[0][0]
            # Load data for experiment
            rows = self.get_table_data_from_database(experiment, chans, cursor)
            # Sort rows according to group
            rows = sorted(rows, key=lambda x: x[1])
            self.set_experiment_status_label_text(
                f"Experiment: {experiment}\nImages: {num_imgs}"
            )
            self.cur_exp = experiment
            # Get channel names
            channel_names = cursor.execute("SELECT DISTINCT name FROM channels WHERE md5"
                                           " IN (SELECT md5 FROM images WHERE experiment=?) and main=?",
                                           (experiment, 0)).fetchall()
            channel_names = sorted([x[0] for x in channel_names])
            header.extend(channel_names)
        else:
            rows = self.get_table_data_for_image((handler.ident,), chans, cursor)
            # Get channel names
            channel_names = cursor.execute("SELECT name FROM channels WHERE md5=? AND main=?",
                                           (handler.ident, 0)).fetchall()
            channel_names = sorted([x[0] for x in channel_names])
            header.remove("Group")
            header.extend(channel_names)
            self.set_experiment_status_label_text(
                f"Experiment: None\nImages: 1"
            )
        self.create_table_rows(rows)
        # Set header of table
        self.res_table_model.setHorizontalHeaderLabels(header)
        header = [header]
        header.extend(rows)
        self.data = header

    def create_table_rows(self, rows: List[List[str]], append: bool = True) -> Union[None, List[List[QStandardItem]]]:
        """
        Method to create multiple table rows

        :param rows: The rows to create
        :param append: If true, the row will be directly appended to the results table
        :return: None if append, else the created row
        """
        item_rows = []
        for row in rows:
            item_rows.append(self.create_table_row(row, append))
        if not append:
            return item_rows

    def create_table_row(self, cells: List[str], append: bool = True) -> Union[None, List[QStandardItem]]:
        """
        Method to create a table row

        :param cells: The text each cell of the row should contain
        :param append: If true, the row will be directly appended to the results table
        :return: None if append, else the created row
        """
        # Iterate over created rows
        item_row = []
        # Create an QStandardItem for each cell in the row
        for cell in cells:
            item = QStandardItem()
            item.setText(cell)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            item.setSelectable(False)
            item.setEditable(False)
            item_row.append(item)
        if append:
            # Append the item row to the table model
            self.res_table_model.appendRow(item_row)
            return
        return item_row

    def get_table_data_from_database(self, experiment: str, channel_names: List[str],
                                     cursor: sqlite3.Cursor) -> List[List[str]]:
        """
        Method to load the data of an experiment from the database

        :param experiment: The name of the experiment to get the data for
        :param channel_names: Names of the active channels without the main channel
        :param cursor: Cursor pointing to the database
        :return: List of row to created for display
        """
        # Get images associated with experiment
        imgs = cursor.execute("SELECT md5 FROM images WHERE experiment=?",
                              (experiment,)).fetchall()
        rows: List[List[str]] = []
        # Iterate over all images
        for img in imgs:
            row = self.get_table_data_for_image(img, channel_names, cursor)
            # Check if the image was assigned to a group
            group = cursor.execute("SELECT name FROM groups WHERE image=?", img).fetchall()
            if group:
                group = group[0][0]
            else:
                group = "No Group"
            for row_ in row:
                row_.insert(2, group)
            rows.extend(row)
        return rows

    def get_table_data_for_image(self, img: Tuple[str], channel_names: List[str],
                                 cursor: sqlite3.Cursor) -> List[List[str]]:
        """
        Method to get the table data for the specified image

        :param img: The image to get the data for
        :param channel_names: Names of the active channels without the main channel
        :param cursor: Cursor pointing to the database
        :return: List of rows created for display
        """
        rows: List[List[str]] = []
        # Get all associated nuclei for the image
        nucs = cursor.execute("SELECT hash FROM roi WHERE associated IS NULL AND image=?", img).fetchall()
        nuclen = len(nucs)
        ind = 1
        # Convert key to file name
        name = self.hash_to_name[img[0]]
        # Get all information for each focus
        for nuc in nucs:
            self.prg_signal.emit(f"Creating table for image {img[0]}: {ind}/{nuclen}",
                                 (ind/nuclen) * 100, 100, "")
            # Get statistics of nucleus
            stats = cursor.execute("SELECT * FROM statistics WHERE hash=?", nuc).fetchall()[0]
            row = [name, img[0], str(nuc[0]), stats[10], str(stats[2]), f"{float(stats[16]) * 100:.2f}",
                   f"{float(stats[13]):.2f}", f"{float(stats[11]):.2f}", f"{float(stats[12]):.2f}"]
            # Count available foci
            for channel in channel_names:
                count = cursor.execute("SELECT COUNT(hash) FROM roi WHERE associated=? AND channel=?",
                                       (nuc[0], channel)).fetchall()
                row.append(str(count[0][0]))
            rows.append(row)
            ind += 1
        return rows

    def set_experiment_status_label_text(self, status: str) -> None:
        """
        Method to display information about experiment details on screen

        :param status: The details to display
        :return: None
        """
        self.ui.lbl_exp_details.setText(status)

    def enable_buttons(self, state: bool = True, ana_buttons: bool = True) -> None:
        """
        Method to disable or enable the GUI buttons

        :param state: The state the buttons will set into
        :param ana_buttons: Indicates if the status of the analysis buttons also should be changed
        :return: None
        """
        if ana_buttons:
            self.ui.btn_analyse.setEnabled(state)
            self.ui.btn_analyse_all.setEnabled(state)
            self.ui.btn_clear_list.setEnabled(state)
            self.ui.btn_delete_from_list.setEnabled(state)
            self.ui.btn_reload.setEnabled(state)
        self.ui.btn_load.setEnabled(state)
        self.ui.btn_save.setEnabled(state)
        self.ui.btn_images.setEnabled(state)
        self.ui.btn_statistics.setEnabled(state)
        self.ui.btn_categories.setEnabled(state)
        self.ui.btn_modify.setEnabled(state)

    def _select_next_image(self, first: bool = False) -> None:
        """
        Method to select the next image in the list of loaded images. Selects the first image if no image is selected

        :param first: Indicates if the first image in the list should be selected
        :return: None
        """
        max_ind = self.img_list_model.rowCount()
        cur_ind = self.ui.list_images.currentIndex()
        if cur_ind.row() < max_ind and not first:
            nex = self.img_list_model.index(cur_ind.row() + 1, 0)
            self.ui.list_images.selectionModel().select(nex, QItemSelectionModel.Select)
            self.ui.list_images.setCurrentIndex(nex)
        else:
            first = self.img_list_model.index(0, 0)
            self.ui.list_images.selectionModel().select(first, QItemSelectionModel.Select)
            self.ui.list_images.setCurrentIndex(first)

    def _set_progress(self, text: str, progress: Union[int, float], maxi: Union[int, float], symbol: str) -> None:
        """
        Method to control the progress bar. Should not be called directly, emit the progress signal instead

        :param text: The text to show above the bar
        :param progress: The value of the bar
        :param maxi: The max value of the bar
        :param symbol: The symbol printed after the displayed values
        :return: None
        """
        self.ui.lbl_status.setText(f"{text} -- {(progress / maxi) * 100:.2f}% {symbol}")
        self.ui.prg_bar.setMaximum(maxi)
        self.ui.prg_bar.setValue(progress)

    def show_result_image(self) -> None:
        """
        Method to open an dialog to show the analysis results as plt plot

        :return: None
        """
        image_dialog = ImgDialog(image=Detector.load_image(self.cur_img["path"]), handler=self.roi_cache)
        image_dialog.setWindowTitle(f"Result Images for {self.cur_img['file_name']}")
        image_dialog.setWindowIcon(QtGui.QIcon('logo.png'))
        image_dialog.setWindowFlags(image_dialog.windowFlags() |
                                    QtCore.Qt.WindowSystemMenuHint |
                                    QtCore.Qt.WindowMinMaxButtonsHint |
                                    QtCore.Qt.Window)
        image_dialog.exec_()

    def save_results(self) -> None:
        """
        Method to export the analysis results as csv file

        :return: None
        """
        save = Thread(target=self._save_results)
        self.prg_signal.emit("Saving Results", 0, 100, "")
        save.start()

    def _save_results(self) -> None:
        """
        Method to export the analysis results as csv file

        :return: None
        """
        self.prg_signal.emit("Saving Results", 50, 100, "")
        # Create name for file
        name = f"{self.cur_exp if self.cur_exp else self.roi_cache.ident}.csv"
        with open(os.path.join(Paths.result_path, name), 'w', newline='') as file:
            writer = csv.writer(file, delimiter=";",
                                quotechar="|", quoting=csv.QUOTE_MINIMAL)
            for row in self.data:
                writer.writerow(row)
        self.prg_signal.emit("Results saved -- Program ready", 100, 100, "")
        self.unsaved_changes = False

    def on_config_change(self, config, section, key: str, value: Union[str, int, float]) -> None:
        """
        Will be called if changed occur in the program settings

        :param config: The changed config
        :param section: The section in which the change occured
        :param key: The identifier of the changed field
        :param value: The value of the changed field
        :return: None
        """
        # TODO Implement & test
        print(f"Config:\n{config}")
        if section == "Analysis":
            self.detector.settings[key] = value

    def show_statistics(self) -> None:
        """
        Method to open a dialog showing various statistics

        :return: None
        """
        # Check if experiments were defined
        exps = self.cursor.execute("SELECT * FROM experiments").fetchall()
        if not exps:
            msg = QMessageBox()
            msg.setWindowIcon(Icon.get_icon("LOGO"))
            msg.setIcon(QMessageBox.Information)
            msg.setStyleSheet(open("messagebox.css", "r").read())
            msg.setWindowTitle("Warning")
            msg.setText("No experiments were defined")
            msg.setInformativeText("Statistics can only be displayed, if images are assigned to an experiment")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        # Create dialog window
        stat_dialog = QDialog()
        stat_dialog.ui = uic.loadUi(Paths.ui_stat_dial, stat_dialog)
        # Open dialog to select an experiment
        exp_sel_dial = ExperimentSelectionDialog()
        code = exp_sel_dial.exec()
        if code == QDialog.Accepted:
            exp = exp_sel_dial.sel_exp
            active_channels = exp_sel_dial.active_channels
            stat_dialog = StatisticsDialog(experiment=exp,
                                           active_channels=active_channels)
            stat_dialog.exec()
        else:
            return

    def show_categorization(self) -> None:
        """
        Method to open a dialog to enable the user to categories the loaded image

        :return: None
        """
        cl_dialog = QDialog()
        cl_dialog.ui = uic.loadUi(Paths.ui_class_dial, cl_dialog)
        cl_dialog.setWindowTitle(f"Classification of {self.cur_img['file_name']}")
        cl_dialog.setWindowIcon(QtGui.QIcon('logo.png'))
        hash_ = self.cur_img["key"]
        categories = self.cursor.execute(
            "SELECT category FROM categories WHERE image = ?",
            (hash_,)
        ).fetchall()
        cate = ""
        for cat in categories:
            cate += str(cat[0]) + "\n"
        cl_dialog.ui.te_cat.setPlainText(cate)
        code = cl_dialog.exec()
        if code == QDialog.Accepted:
            self._categorize_image(cl_dialog.ui.te_cat.toPlainText())

    def _categorize_image(self, categories: str) -> None:
        """
        Method to save image categories to the database

        :param categories: The categories to save as str, individually separated by \n
        :return: None
        """
        if categories != "":
            categories = categories.split('\n')
            hash_ = self.cur_img["key"]
            self.cursor.execute(
                "DELETE FROM categories WHERE image = ?",
                (hash_,)
            )
            for cat in categories:
                if cat:
                    self.cursor.execute(
                        "INSERT INTO categories VALUES(?, ?)",
                        (hash_, cat)
                    )
            self.connection.commit()

    def show_settings(self) -> None:
        """
        Method to open the settings dialog

        :return: None
        """
        sett = SettingsDialog()
        sett.initialize_from_file(os.path.join(os.getcwd(), "settings/settings.json"))
        sett.setWindowTitle("Settings")
        sett.setModal(True)
        sett.setWindowIcon(QtGui.QIcon("logo.png"))
        code = sett.exec()
        if code == QDialog.Accepted:
            if sett.changed:
                for key, value in sett.changed.items():
                    self.settings[key] = value[0]
                    self.cursor.execute(
                        "UPDATE OR IGNORE settings SET value = ? WHERE key_ = ?",
                        (key, value[0])
                    )
            sett.save_menu_settings()
            self.connection.commit()

    def show_modification_window(self) -> None:
        """
        Method to open the modification dialog, allowing the user to modify automatically determined results

        :return: None
        """
        editor = Editor(image=Detector.load_image(self.cur_img["path"]),
                        roi=self.roi_cache, size_factor=self.settings["size_factor"],
                        img_name=self.cur_img['file_name'])
        editor.setWindowFlags(editor.windowFlags() |
                              QtCore.Qt.WindowSystemMenuHint |
                              QtCore.Qt.WindowMinMaxButtonsHint |
                              QtCore.Qt.Window)
        code = editor.exec()
        if code == QDialog.Accepted:
            self.create_result_table_from_list(self.roi_cache)

    def on_close(self) -> None:
        """
        Will be called if the program window closes

        :return:
        """
        self.connection.close()


class TableFilterModel(QSortFilterProxyModel):
    """
    Model used to enable tuple sorting
    """

    def __init__(self, parent):
        super(TableFilterModel, self).__init__(parent)

    def lessThan(self, ind1, ind2):
        ldat = self.sourceModel().itemData(ind1)[257]
        rdat = self.sourceModel().itemData(ind2)[257]
        if isinstance(ldat, tuple):
            if ldat[1] == rdat[1]:
                return ldat[0] < rdat[0]
            return ldat[1] < rdat[1]
        return ldat < rdat


def exception_hook(exc_type, exc_value, traceback_obj) -> None:
    """
    General exception hook to display error message for user
    :param exc_type: Type of the exception
    :param exc_value: Value of the exception
    :param traceback_obj: The traceback object associated with the exception
    :return: None
    """
    # Show error message in GUI
    time_string = time.strftime("%Y-%m-%d, %H:%M:%S")
    title = "An error occured during execution"
    info = f"An {exc_type.__name__} occured at {time_string}"
    text = "During the execution of the program, following error occured:\n" \
           f"{''.join(traceback.format_exception(exc_type, exc_value, traceback_obj))}"
    print(text)
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowIcon(QtGui.QIcon('logo.png'))
    msg.setStyleSheet(open("messagebox.css", "r").read())
    msg.setText(text)
    msg.setInformativeText(info)
    msg.setWindowTitle(title)
    msg.exec_()


def main() -> None:
    """
    Function to start the program

    :return: None
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sys.excepthook = exception_hook
        app = QtWidgets.QApplication(sys.argv)
        pixmap = QPixmap("banner_norm.png")
        splash = QSplashScreen(pixmap)
        splash.show()
        splash.showMessage("Loading...")
        main_win = NucDetect()
        splash.finish(main_win)
        main_win.show()
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()
