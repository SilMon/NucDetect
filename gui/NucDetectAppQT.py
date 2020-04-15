from __future__ import annotations

import copy
import datetime
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
import traceback
import pyqtgraph as pg
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from threading import Thread
from typing import Union, Dict, List, Tuple, Any

from gui import Paths, Util
from gui.Definitions import Icon

import PyQt5
import numpy as np
import qtawesome as qta
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QRectF, QItemSelectionModel, QSortFilterProxyModel, QItemSelection
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIcon, QPixmap, QColor, QBrush, QPen, QResizeEvent, \
    QKeyEvent, QMouseEvent, QPainter, QTransform, QFont
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QHeaderView, QDialog, QSplashScreen, QSizePolicy, QWidget, \
    QVBoxLayout, QScrollArea, QMessageBox, QGraphicsScene, QGraphicsEllipseItem, QGraphicsView, QGraphicsItem, \
    QGraphicsPixmapItem, QLabel, QGraphicsLineItem, QStyleOptionGraphicsItem, QInputDialog, QFrame
from gui.Dialogs import ExperimentDialog
from skimage.draw import ellipse

from core.Detector import Detector
from core.ROI import ROI
from core.ROIHandler import ROIHandler
from gui.Plots import BoxPlotWidget, PoissonPlotWidget
from gui.settings.Settings import SettingsShowWidget, SettingsSlider, SettingsText, SettingsComboBox, \
    SettingsCheckBox, SettingsDial, SettingsSpinner, SettingsDecimalSpinner

PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, False)
PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, False)
pg.setConfigOptions(imageAxisOrder='row-major')


class NucDetect(QMainWindow):
    """
    Created on 11.02.2019
    @author: Romano Weiss
    """
    prg_signal = pyqtSignal(str, int, int, str)
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
        self.detector = Detector(settings=[].extend(list(self.settings.values())),
                                 logging=self.settings["logging"])
        # Initialize needed variables
        self.reg_images = []
        self.cur_img = None
        self.roi_cache = None
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
            CREATE TABLE IF NOT EXISTS "groups" (
                "image"	INTEGER,
                "experiment"	INTEGER,
                "name"	INTEGER,
                PRIMARY KEY("image","experiment")
            ) WITHOUT ROWID ;
            CREATE TABLE IF NOT EXISTS "experiments" (
                "name"	TEXT,
                "details"	TEXT,
                "notes"	TEXT,
                PRIMARY KEY("name")
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
                "group"	TEXT,
                PRIMARY KEY("md5")
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS "points" (
                "hash"	INTEGER,
                "x"	INTEGER,
                "y"	INTEGER,
                "intensity"	INTEGER,
                PRIMARY KEY("hash","x","y")
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS "settings" (
                "key_"	TEXT,
                "value"	TEXT,
                PRIMARY KEY("key_")
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS "categories" (
                "image"	INTEGER,
                "category"	TEXT,
                PRIMARY KEY("image","category")
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS "channels" (
                "md5"	INTEGER,
                "index"	INTEGER,
                "name"	INTEGER,
                PRIMARY KEY("md5","index")
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
            CREATE TABLE IF NOT EXISTS "statistics" (
                "hash"	INTEGER,
                "image"	INTEGER,
                "area"	INTEGER,
                "intensity_average"	INTEGER,
                "intensity_median"	INTEGER,
                "intensity_maximum"	INTEGER,
                "intensity_minimum"	INTEGER,
                "intensity_std"	INTEGER,
                "ellipse_center"	INTEGER,
                "ellipse_major_axis_p0"	INTEGER,
                "ellipse_major_axis_p1"	INTEGER,
                "ellipse_major_axis_slope"	INTEGER,
                "ellipse_major_axis_length"	INTEGER,
                "ellipse_major_axis_angle"	INTEGER,
                "ellipse_minor_axis_p0"	INTEGER,
                "ellipse_minor_axis_p1"	INTEGER,
                "ellipse_minor_axis_length"	INTEGER,
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
            INSERT OR IGNORE INTO settings (key_, value) VALUES ("logging", 1);
            INSERT OR IGNORE INTO settings (key_, value) VALUES ("res_path", "./results");
            INSERT OR IGNORE INTO settings (key_, value) VALUES ("names", "Blue;Red;Green");
            INSERT OR IGNORE INTO settings (key_, value) VALUES ("main_channel", 2);
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
            os.mkdir(Paths.thumb_path)
        if not os.path.isdir(Paths.nuc_detect_dir):
            os.mkdir(Paths.nuc_detect_dir)
        if not os.path.isdir(Paths.result_path):
            os.mkdir(Paths.result_path)
        if not os.path.isdir(Paths.images_path):
            os.mkdir(Paths.images_path)
            shutil.copy2(os.path.join(os.pardir, "demo.tif"), os.path.join(Paths.images_path, "demo.tif"))

    def load_settings(self) -> Dict:
        """
        Method to load the saved Settings
        :return: None
        """
        self.cursor.execute(
            "SELECT key_, value FROM settings"
        )
        return dict(self.cursor.fetchall())

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
        self.res_table_model.setHorizontalHeaderLabels(["Image", "Center[(y, x)]", "Area [px]", "Ellipticity [%]",
                                                        "Foci"])
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
        self.add_images_from_folder(Paths.images_path)

    def on_image_selection_change(self) -> None:
        """
        Will be called if a new image is selected

        :return: None
        """
        for index in self.ui.list_images.selectionModel().selectedIndexes():
            self.cur_img = self.img_list_model.item(index.row()).data()
        if self.cur_img:
            # TODO Settings mit einbeziehen
            ana = self.cur_img["analysed"]
            if ana:
                self.prg_signal.emit(f"Loading data from database for {self.cur_img['file_name']}",
                                     0, 100, "")
                thread = Thread(target=self.load_saved_data)
                thread.start()
            else:
                self.ui.lbl_status.setText("Program ready")
                self.res_table_model.setRowCount(0)
                self.enable_buttons(False, ana_buttons=False)
                self.ui.btn_analyse.setEnabled(True)
        else:
            self.ui.btn_analyse.setEnabled(False)

    def load_saved_data(self) -> None:
        """
        Method to load saved data from the database

        :return: None
        """
        # Disable Buttons and list during loading
        self.enable_buttons(state=False)
        self.ui.list_images.setEnabled(False)
        # Load saved data from databank
        self.roi_cache = self.load_rois_from_database(self.cur_img["key"])
        # Create the result table from loaded data
        self.create_result_table_from_list(self.roi_cache)
        # Re-enable buttons and list
        self.ui.list_images.setEnabled(True)
        self.enable_buttons()
        # Disable analysis button -> Useless if image was already analysed
        self.ui.btn_analyse.setEnabled(False)
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
            self.add_image_to_list(file_name.replace("/", os.sep))

    def add_images_from_folder(self, url: str) -> None:
        """
        Method to load a whole folder of images

        :param url: The path of the folder
        :return: None
        """
        paths = []
        for t in os.walk(url):
            for file in t[2]:
                paths.append(os.path.join(t[0], file))
        items = Util.create_image_item_list_from(paths, indicate_progress=True)
        for item in items:
            self.add_item_to_list(item)

    def add_item_to_list(self, item: QStandardItem) -> None:
        """
        Utility method to add an item to the image list

        :param item: The item to add
        :return: None
        """
        if item is not None:
            path = item.data()["path"]
            key = item.data()["key"]
            d = Detector.get_image_data(path)
            self.img_list_model.appendRow(item)
            self.reg_images.append((key, path))
            if not self.cursor.execute(
                    "SELECT * FROM images WHERE md5 = ?",
                    (key,)
            ).fetchall():
                self.cursor.execute(
                    "INSERT OR IGNORE INTO images VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (key, d["datetime"], d["channels"], d["width"], d["height"],
                     str(d["x_res"]), str(d["y_res"]), d["unit"], 0, -1, None)
                )
            self.connection.commit()

    def remove_image_from_list(self) -> None:
        """
        Method to remove an loaded image from the file list.

        :return: None
        """
        cur_ind = self.ui.list_images.currentIndex()
        # TODO Update because of change in reg_images
        del self.reg_images[self.img_list_model.item(cur_ind.row()).tdata["key"]]
        self.img_list_model.removeRow(cur_ind.row())
        if cur_ind.row() < self.img_list_model.rowCount():
            self.ui.list_images.selectionModel().select(cur_ind, QItemSelectionModel.Select)
            self.ui.list_images.setCurrentIndex(cur_ind)
        else:
            nex = self.img_list_model.index(cur_ind.row() - 1, 0)
            self.ui.list_images.selectionModel().select(nex, QItemSelectionModel.Select)
            self.ui.list_images.setCurrentIndex(nex)

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
        self.res_table_model.setRowCount(0)
        if not self.cur_img:
            self.ui.list_images.select(self.img_list_model.index(0, 0))
        self.prg_signal.emit(f"Analysing {self.cur_img['file_name']}",
                             0, 100, "")
        thread = Thread(target=self.analyze_image,
                        args=(self.cur_img["path"],
                              "Analysis finished in {} -- Program ready",
                              100, 100,))
        thread.start()

    def analyze_image(self, path: str, message: str,
                      percent: Union[int, float], maxi: Union[int, float]) -> None:
        """
        Method to analyse the image given by path

        :param path: The path leading to the image
        :param message: The message to display above the progress bar
        :param percent: The value of the progress bar
        :param maxi: The maximum of the progress bar
        :return: None
        """
        self.enable_buttons(False)
        self.ui.list_images.setEnabled(False)
        start = time.time()
        self.prg_signal.emit("Starting analysis", 0, maxi, "")
        self.unsaved_changes = True
        self.prg_signal.emit("Analysing image", maxi * 0.05, maxi, "")
        data = self.detector.analyse_image(path)
        self.roi_cache = data["handler"]
        s0 = time.time()
        self.prg_signal.emit(f"Ellipse parameter calculation", maxi * 0.75, maxi, "")
        with ThreadPoolExecutor(max_workers=None) as e:
            for roi in self.roi_cache:
                if roi.main:
                    e.submit(roi.calculate_ellipse_parameters)
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
        self.enable_buttons()
        self.ui.btn_analyse.setEnabled(False)
        self.ui.list_images.setEnabled(True)

    def save_rois_to_database(self, data: Dict[str, Union[int, float, str]], all: bool = False) -> None:
        """
        Method to save the data stored in the ROIHandler rois to the database

        :param data: The data dict returned by the Detector class
        :param all: Deactivates printing to console
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
                curs.execute(
                    "DELETE FROM points where hash = ?",
                    (h[0],)
                )
            curs.execute(
                "DELETE FROM roi WHERE image = ?",
                (key,)
            )
        for name in data["handler"].idents:
            curs.execute(
                "INSERT OR IGNORE INTO channels VALUES (?, ?, ?)",
                (key, data["handler"].idents.index(name), name)
            )
        roidat = []
        pdat = []
        elldat = []
        for roi in data["handler"].rois:
            dim = roi.calculate_dimensions()
            ellp = roi.calculate_ellipse_parameters()
            stats = roi.calculate_statistics()
            asso = hash(roi.associated) if roi.associated is not None else None
            roidat.append((hash(roi), key, True, roi.ident, str(dim["center"]), dim["width"], dim["height"], asso))
            for p in roi.points:
                pdat.append((hash(roi), p[0], p[1], roi.inten[p]))
            elldat.append(
                (hash(roi), key, stats["area"], stats["intensity average"], stats["intensity median"],
                 stats["intensity maximum"], stats["intensity minimum"], stats["intensity std"],
                 str(ellp["center"]), str(ellp["major_axis"][0]), str(ellp["major_axis"][1]),
                 ellp["major_slope"], ellp["major_length"], ellp["major_angle"], str(ellp["minor_axis"][0]),
                 str(ellp["minor_axis"][1]), ellp["minor_length"], ellp["shape_match"])
            )
        curs.executemany(
            "INSERT OR IGNORE INTO roi VALUES (?, ?, ?, ?, ?, ?, ?,?)",
            roidat
        )
        curs.executemany(
            "INSERT OR IGNORE INTO points VALUES (?, ?, ?, ?)",
            pdat
        )
        curs.executemany(
            "INSERT OR IGNORE INTO statistics VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            elldat
        )
        curs.execute(
            "UPDATE images SET analysed = ? WHERE md5 = ?",
            (True, key)
        )
        con.commit()
        con.close()
        if not all:
            print("ROI saved to database")

    def create_result_table_from_list(self, handler: ROIHandler) -> None:
        """
        Method to create the result table from a list of rois

        :param handler: The handler containing the rois
        :return: None
        """
        tabdat = handler.get_data_as_dict()
        self.res_table_model.setRowCount(0)
        self.res_table_model.setHorizontalHeaderLabels(tabdat["header"])
        self.res_table_model.setColumnCount(len(tabdat["header"]))
        for x in range(len(tabdat["data"])):
            row = []
            for dat in tabdat["data"][x]:
                item = QStandardItem()
                show_text = str(dat)
                if isinstance(dat, tuple):
                    show_text = f"{int(dat[1]):#>5d} | {int(dat[0]):#<5d}"
                elif isinstance(dat, float):
                    show_text = f"{dat * 100:.3f}"
                item.setText(show_text)
                item.setData(dat)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setSelectable(False)
                row.append(item)
            self.res_table_model.appendRow(row)

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

    def analyze_all(self) -> None:
        """
        Method to perform concurrent batch analysis of registered images

        :return: None
        """
        self.enable_buttons(False)
        self.ui.list_images.setEnabled(False)
        self.unsaved_changes = True
        thread = Thread(target=self._analyze_all)
        thread.start()

    def _analyze_all(self, batch_size=20) -> None:
        """
        Method to perform concurrent batch analysis of registered images

        :return: None
        """
        start = time.time()
        with ProcessPoolExecutor(max_workers=None) as e:
            logstate = self.detector.logging
            self.detector.logging = False
            self.prg_signal.emit("Starting multi image analysis", 0, 100, "")
            paths = []
            for ind in range(self.img_list_model.rowCount()):
                data = self.img_list_model.item(ind).data()
                if not bool(data["analysed"]):
                    paths.append(data["path"])
            ind = 1
            cur_batch = 1
            curind = 0
            # Iterate over all images in batches
            for b in range(batch_size + 1 if batch_size < len(paths) else len(paths),
                           len(paths) if len(paths) > batch_size else len(paths) + 1, batch_size):
                s2 = time.time()
                tpaths = paths[curind:b if b < len(paths) else len(paths) - 1]
                res = e.map(self.detector.analyse_image, tpaths)
                maxi = len(paths)
                for r in res:
                    self.prg_signal.emit(f"Analysed images: {ind}/{maxi}",
                                         ind, maxi, "")
                    self.save_rois_to_database(r, all=True)
                    self.roi_cache = r["handler"]
                    self.create_result_table_from_list(r["handler"])
                    ind += 1
                print(f"Analysed batch {cur_batch} in {time.time() - s2} secs\tTotal: {time.time() - start} secs")
                curind = b
                cur_batch += 1
            self.roi_cache = list(res)[:-1]
            self.enable_buttons()
            self.ui.list_images.setEnabled(True)
            self.detector.logging = logstate
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
        print(f"Total analysis time: {time.time() - start} secs")

    def load_rois_from_database(self, md5: int) -> ROIHandler:
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
                    "intensity minimum", "intensity std")
        ellkeys = ("center", "major_axis", "major_slope", "major_length",
                   "major_angle", "minor_axis", "minor_length", "shape_match")
        ind = 1
        max = len(entries)
        for entry in entries:
            self.prg_signal.emit(f"Loading ROI:  {ind}/{max}",
                                 ind, max, "")
            temproi = ROI(channel=entry[3], main=entry[7] is None, associated=entry[7])
            temproi.id = entry[0]
            stats = crs.execute(
                "SELECT * FROM statistics WHERE hash = ?",
                (entry[0],)
            ).fetchall()[0]
            temproi.stats = dict(zip(statkeys, stats[2:8]))
            if temproi.main:
                main_.append(temproi)
            else:
                sec.append(temproi)
            for p in crs.execute(
                    "SELECT * FROM points WHERE hash = ?",
                    (entry[0],)
            ).fetchall():
                temproi.add_point((p[1], p[2]), p[3])
            if temproi.main:
                center = re.search(r"\((\d*)\D*(\d*)\)?", stats[8])
                maj = re.search(r"\((\d*)\D*(\d*)\)?", stats[9]), re.search(r"\((\d*)\D*(\d*)\)?", stats[10])
                mino = re.search(r"\((\d*)\D*(\d*)\)?", stats[14]), re.search(r"\((\d*)\D*(\d*)\)?", stats[15])
                center = (int(center.group(1)), int(center.group(2)))
                major = (int(maj[0].group(1)), int(maj[0].group(2))), (int(maj[1].group(1)), int(maj[1].group(2)))
                minor = (int(mino[0].group(1)), int(mino[0].group(2))), (int(mino[1].group(1)), int(mino[1].group(2)))
            else:
                center = (None, None)
                major = (None, None), (None, None)
                minor = (None, None), (None, None)
            ellp = (center, major, stats[11], stats[12], stats[13], minor, stats[16], stats[17])
            temproi.ell_params = dict(zip(ellkeys, ellp))
            rois.add_roi(temproi)
            ind += 1
        for m in main_:
            for s in sec:
                if s.associated == hash(m):
                    s.associated = m
        print("Loaded roi from database")
        return rois

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
        self.roi_cache.export_data_as_csv(path=Paths.result_path, ident=self.cur_img["file_name"])
        self.prg_signal.emit("Saving Results", 50, 100, "")
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
        # Create dialog window
        stat_dialog = QDialog()
        stat_dialog.ui = uic.loadUi(Paths.ui_stat_dial, stat_dialog)
        # Load available experiments
        exps = self.cursor.execute(
            "SELECT * FROM experiments"
        ).fetchall()
        # Open dialog to select an experiment
        exp = QInputDialog.getItem(self,
                                   "Select an experiment to analyse",
                                   "Experiment: ",
                                   [x[0] for x in exps])[0]
        # If no experiment was selected, return
        if not exp:
            return
        stat_dialog.setWindowTitle(f"Statistics for {exp}")
        stat_dialog.setWindowIcon(QtGui.QIcon('logo.png'))
        # Get the groups associated with the experiment
        groups_raw = self.cursor.execute(
            "SELECT * FROM groups WHERE experiment=?",
            (exp,)
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
            (exp,)
        ).fetchall()
        # Get main channel
        main = self.cursor.execute(
            "SELECT DISTINCT channel FROM roi WHERE associated IS NULL AND image"
            " IN (SELECT image FROM groups WHERE experiment=?)",
            (exp,)
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
        # Create plots
        for i in range(len(channels)):
            # Get the data for this channel
            data = {key: value[i] for key, value in group_data.items()}
            # Create PlotWidget for channel
            d = list(data.values())
            g = list(data.keys())
            pw = BoxPlotWidget(data=d, groups=g)
            pw.setTitle(f"{channels[i]} Analysis")
            pw.laxis.setLabel("Foci/Nucleus")
            stat_dialog.ui.vl_vals.addWidget(pw)
            # Create the line to add
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            # Get plotting data of BoxPlot
            p_data = pw.p_data
            # Create Scroll Area
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            central_widget = QWidget()
            layout = QVBoxLayout()
            central_widget.setLayout(layout)
            sa.setWidget(central_widget)
            # Iterate over groups
            for j in range(len(g)):
                layout.addWidget(QLabel(
                    f"<strong>Group: {g[j]}</strong>"
                ))
                layout.addWidget(QLabel(
                    f"Values (w/o Outliers): {p_data[j]['number']}"
                ))
                layout.addWidget(QLabel(
                    f"Average: {p_data[j]['average']:.2f}"
                ))
                layout.addWidget(QLabel(
                    f"Median: {p_data[j]['median']}"
                ))
                layout.addWidget(QLabel(
                    f"IQR: {p_data[j]['iqr']}"
                ))
                layout.addWidget(QLabel(
                    f"Outliers: {len(p_data[j]['outliers'])}"
                ))
                layout.addSpacing(10)
            stat_dialog.ui.val_par.addWidget(sa)
            if i < len(channels) - 1:
                stat_dialog.ui.val_par.addWidget(line)

            # Add bool to check if a line was already created
            check = False
            # Create Poisson Plot for channel
            for group, values in data.items():
                poiss = PoissonPlotWidget(data=values, label=group)
                poiss.setTitle(f"{group} - Comparison to Poisson Distribution")
                # Create the line to add
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setFrameShadow(QFrame.Sunken)
                # Add poisson plot
                stat_dialog.ui.vl_poisson.addWidget(poiss)
                # Add additional information
                stat_dialog.ui.dist_par.addWidget(QLabel(f"Values: {len(values)}"))
                stat_dialog.ui.dist_par.addWidget(QLabel(f"Average: {np.average(values):.2f}"))
                stat_dialog.ui.dist_par.addWidget(QLabel(f"Min.: {np.amin(values)}"))
                stat_dialog.ui.dist_par.addWidget(QLabel(f"Max.: {np.amax(values)}"))
                # Add line and stretch
                stat_dialog.ui.dist_par.addStretch(1)
                if i == len(channels) - 1:
                    if not check:
                        stat_dialog.ui.dist_par.addWidget(line)
                        check = True
                else:
                    stat_dialog.ui.dist_par.addWidget(line)
                stat_dialog.ui.dist_par.addStretch(1)
        stat_dialog.setWindowFlags(stat_dialog.windowFlags() |
                                   QtCore.Qt.WindowSystemMenuHint |
                                   QtCore.Qt.WindowMinMaxButtonsHint)
        code = stat_dialog.exec()

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
        if categories is not "":
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
        # TODO
        sett = SettingsDialog()
        sett.initialize_from_file(os.path.join(os.getcwd(), "settings/settings.json"))
        sett.setWindowTitle("Settings")
        sett.setModal(True)
        sett.setWindowIcon(QtGui.QIcon("logo.png"))
        code = sett.exec()
        if code == QDialog.Accepted:
            if sett.changed:
                for key, value in sett.changed.items():
                    self.detector.settings[key] = value
                    print("Key: {} Value: {}".format(key, value))
                    # TODO
                    """
                    self.cursor.execute(
                        "INSERT INTO settings VALUES(?, ?)",
                        (key, str(value))
                    )
                    """
            sett.save_menu_settings()

    def show_modification_window(self) -> None:
        """
        Method to open the modification dialog, allowing the user to modify automatically determined results

        :return: None
        """
        mod = ModificationDialog(image=Detector.load_image(self.cur_img["path"]), handler=self.roi_cache)
        mod.setWindowTitle(f"Modification Dialog for {self.cur_img['file_name']}")
        mod.setWindowIcon(QtGui.QIcon("logo.png"))
        mod.setWindowFlags(mod.windowFlags() |
                           QtCore.Qt.WindowSystemMenuHint |
                           QtCore.Qt.WindowMinMaxButtonsHint |
                           QtCore.Qt.Window)
        code = mod.exec()
        if code == QDialog.Accepted:
            self.create_result_table_from_list(self.roi_cache)
        elif code == QDialog.Rejected:
            if mod.changed:
                self.load_saved_data()

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
                c = params["center"][1], params["center"][0]
                d1 = params["minor_length"]
                d2 = params["major_length"]
                slope = params["major_slope"]
                item = QGraphicsEllipseItem(-d1 / 2, -d2 / 2, d1, d2)
                # Get the angle of the major axis
                angle = params["major_angle"]
                item.setData(0, self.nuc_pen)
                item.setData(1, roi.main)
                # Rotate the ellipse according to the angle
                item.setTransformOriginPoint(item.sceneBoundingRect().center())
                item.setRotation(-90 + angle if slope > 0 else 90 - angle)
                item.setPos(c[0], c[1])
                major = params["major_axis"]
                maj = QGraphicsLineItem(major[0][1], major[0][0], major[1][1], major[1][0])
                maj.setData(0, self.nuc_pen)
                minor = params["minor_axis"]
                min_ = QGraphicsLineItem(minor[0][1], minor[0][0], minor[1][1], minor[1][0])
                min_.setData(0, self.nuc_pen)
                self.items[ind].append(maj)
                self.items[ind].append(min_)
                self.plot_item.addItem(maj)
                self.plot_item.addItem(min_)
            else:
                c = dims["minX"], dims["minY"]
                d2 = dims["height"]
                d1 = dims["width"]
                item = QGraphicsEllipseItem(c[0], c[1], d1, d2)
                item.setData(0, self.MARKERS[ind])
                item.setData(1, roi.main)
            self.items[ind].append(item)
            self.plot_item.addItem(item)
        # Add information
        stats = self.handler.calculate_statistics()
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
                number = [x for _, x in roinum[ident].items()]
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

    def on_button_click(self) -> None:
        self.save_image()

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
        self.json = None
        self.url = None
        self._initialize_ui()

    def _initialize_ui(self) -> None:
        self.ui = uic.loadUi(Paths.ui_settings_dial, self)

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
        :param menupoint: The menupoint
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


class ModificationDialog(QDialog):

    def __init__(self, image: np.ndarray = None, handler: ROIHandler = None, parent: QWidget = None) -> None:
        super(ModificationDialog, self).__init__(parent)
        self.handler = handler
        self.image = image
        self.show = True
        self.last_index = 0
        self.cur_index = 0
        self.cur_channel = 3
        self.changed = False
        self.max = 2
        self.mp = None
        self.ui = None
        self.view = None
        self.lst_nuc_model = None
        self.commands = []
        self.conn = sqlite3.connect(Paths.database)
        self.curs = self.conn.cursor()
        self.btn_col = QColor(47, 167, 212)
        self.initialize_ui()

    def accept(self) -> None:
        for comm in self.commands:
            self.curs.execute(
                comm[0],
                comm[1]
            )
        self.conn.commit()
        self.conn.close()
        self.changed = True
        super(ModificationDialog, self).accept()

    def reject(self) -> None:
        super(ModificationDialog, self).reject()

    def initialize_ui(self) -> None:
        self.ui = uic.loadUi(Paths.ui_modification_dial, self)
        # Initialize channel selector
        chan_num = len(self.handler.idents)
        self.max = chan_num - 1
        self.ui.sb_channel.setMaximum(chan_num)
        self.view = NucView(self.image, self.handler, self.commands,
                            self.cur_channel, self.show, True, self.max, self.curs, self)
        self.ui.graph_par.insertWidget(0, self.view, 3)
        self.lst_nuc_model = QStandardItemModel(self.ui.lst_nuc)
        self.ui.lst_nuc.setModel(self.lst_nuc_model)
        self.ui.lst_nuc.setIconSize(QSize(75, 75))
        self.ui.lst_nuc.selectionModel().selectionChanged.connect(self.on_selection_change)
        self.set_list_images(self.view.images)
        self.update_list_indices()
        # Initialize buttons
        self.ui.sb_channel.valueChanged.connect(self.on_nucleus_selection_change)
        self.ui.btn_split.clicked.connect(self.on_button_click)
        self.ui.btn_split.setIcon(qta.icon("fa5s.ruler", color=self.btn_col))
        self.ui.btn_show.clicked.connect(self.on_button_click)
        self.ui.btn_show.setIcon(qta.icon("fa5.eye", color=self.btn_col))
        self.ui.btn_merge.clicked.connect(self.on_button_click)
        self.ui.btn_merge.setIcon(qta.icon("fa5.object-group", color=self.btn_col))
        self.ui.btn_remove.clicked.connect(self.on_button_click)
        self.ui.btn_remove.setIcon(qta.icon("fa5.trash-alt", color=self.btn_col))
        self.ui.btn_edit.clicked.connect(self.on_button_click)
        self.ui.btn_edit.setIcon(qta.icon("fa5.edit", color=self.btn_col))
        # Initialize interactivity of graphics view
        self.set_current_image()

    def set_list_images(self, images: List[np.ndarray]) -> None:
        self.lst_nuc_model.clear()
        for image in images:
            item = QStandardItem()
            item.setTextAlignment(QtCore.Qt.AlignLeft)
            pmap = QPixmap()
            pmap.convertFromImage(NucView.get_qimage_from_numpy(image[...,
                                                                      self.handler.idents.index(self.handler.main)]
                                                                ))
            ic = QIcon(pmap)
            item.setIcon(ic)
            self.lst_nuc_model.appendRow(item)

    def on_nucleus_selection_change(self) -> None:
        self.cur_channel = self.ui.sb_channel.value()
        self.set_current_image()

    def on_button_click(self) -> None:
        """
        Method to handle button clicks

        :return: None
        """
        ident = self.sender().objectName()
        if ident == "btn_show":
            self.show = self.ui.btn_show.isChecked()
            if self.show:
                self.ui.btn_show.setIcon(qta.icon("fa5.eye", color=self.btn_col))
            else:
                self.ui.btn_show.setIcon(qta.icon("fa5.eye-slash", color=self.btn_col.darker()))
            self.view.show = self.show
        elif ident == "btn_edit":
            self.view.edit = self.ui.btn_edit.isChecked()
            if self.view.edit:
                self.ui.btn_edit.setIcon(qta.icon("fa5.edit", color=self.btn_col))
            else:
                self.ui.btn_edit.setIcon(qta.icon("fa5.edit", color=self.btn_col.darker()))
        elif ident == "btn_remove":
            selection = self.ui.lst_nuc.selectionModel().selectedIndexes()
            if selection:
                sel = [x.row() for x in selection]
                code = QMessageBox.question(self, "Remove Nuclei...",
                                            f"Do you really want to remove following nuclei: {sel}",
                                            QMessageBox.Yes | QMessageBox.No)
                if code == QMessageBox.Yes:
                    offset = 0
                    for ind in sorted(sel):
                        nuc = self.view.main[ind + offset]
                        self.handler.remove_roi(nuc, cascade=True)
                        self.view.main.remove(nuc)
                        self.lst_nuc_model.removeRow(ind + offset)
                        del self.view.images[ind + offset]
                        offset -= 1
                        self.commands.extend(
                            (("DELETE FROM roi WHERE hash = ? OR associated = ?",
                              (hash(nuc), hash(nuc))),
                             ("DELETE FROM points WHERE hash = ?",
                              (hash(nuc),)))
                        )
                    self.view.cur_ind = 0
                    self.ui.lst_nuc.selectionModel().select(self.lst_nuc_model.createIndex(0, 0),
                                                            QItemSelectionModel.ClearAndSelect)
                    self.update_list_indices()
        elif ident == "btn_merge":
            selection = self.ui.lst_nuc.selectionModel().selectedIndexes()
            sel = [x.row() for x in selection]
            code = QMessageBox.question(self, "Merge Nuclei...",
                                        f"Do you really want to merge following nuclei: {sel}",
                                        QMessageBox.Yes | QMessageBox.No)
            if code == QMessageBox.Yes:
                seed = self.view.main[sel[0]]
                offset = 0
                ass_list = []
                mergehash = [hash(seed)]
                rem_list = []
                for x in range(1, len(sel)):
                    ind = sel[x]
                    merger = self.view.main[ind + offset]
                    mergehash.append(hash(merger))
                    seed.merge(merger)
                    ass_list.append(merger)
                    rem_list.append(ind + offset)
                    self.handler.rois.remove(merger)
                    del self.view.images[ind + offset]
                    del self.view.main[ind + offset]
                    offset -= 1
                for nuc in ass_list:
                    for foc in self.view.assmap[nuc]:
                        foc.associated = seed
                nuc_dims = seed.calculate_dimensions()
                stats = seed.calculate_statistics()
                ellp = seed.calculate_ellipse_parameters()
                imghash = self.handler.ident
                self.commands.append(
                    ("UPDATE roi SET hash = ?, auto = ?, center = ?, width = ?, height = ? WHERE hash = ?",
                     (hash(seed), False, str(nuc_dims["center"]), nuc_dims["width"], nuc_dims["height"], mergehash[0]))
                )
                self.commands.append(
                    ("INSERT OR IGNORE INTO statistics VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (hash(seed), imghash, stats["area"], stats["intensity average"], stats["intensity median"],
                      stats["intensity maximum"], stats["intensity minimum"], stats["intensity std"],
                      str(ellp["center"]), str(ellp["major_axis"][0]), str(ellp["major_axis"][1]),
                      ellp["major_slope"], ellp["major_length"], ellp["major_angle"], str(ellp["minor_axis"][0]),
                      str(ellp["minor_axis"][1]), ellp["minor_length"], ellp["shape_match"]))
                )
                for h in mergehash:
                    self.commands.extend(
                        (("UPDATE roi SET associated = ? WHERE associated = ?",
                          (hash(seed), h)),
                         ("UPDATE points SET hash = ? WHERE hash = ?",
                          (hash(seed), h)),
                         ("DELETE FROM roi WHERE hash = ?",
                          (h,)),
                         ("DELETE FROM statistics WHERE hash = ?",
                          (h,)))
                    )
                self.view.assmap = Detector.create_association_map(self.handler.rois)
                for rem in rem_list:
                    self.lst_nuc_model.removeRow(rem)
                self.ui.lst_nuc.selectionModel().select(selection[0], QItemSelectionModel.Select)
                pmap = QPixmap()
                pmap.convertFromImage(NucView.get_qimage_from_numpy(seed.get_as_numpy()
                                                                    ))
                ic = QIcon(pmap)
                self.lst_nuc_model.itemFromIndex(selection[0]).setIcon(ic)
                self.update_list_indices()
        elif ident == "btn_split":
            if self.ui.btn_split.isChecked():
                self.ui.btn_remove.setEnabled(False)
                self.ui.btn_edit.setEnabled(False)
                self.ui.btn_show.setEnabled(False)
                self.ui.btn_merge.setEnabled(False)
                self.view.split = True
            else:
                self.ui.btn_remove.setEnabled(True)
                self.ui.btn_edit.setEnabled(True)
                self.ui.btn_show.setEnabled(True)
                if self.ui.lst_nuc.selectionModel().selectedIndexes():
                    self.ui.btn_merge.setEnabled(True)
                self.view.split = False
        self.set_current_image()

    def update_nucleus_list(self) -> None:
        """
        Method to update the interface list after changes
        :return: None
        """
        self.set_list_images(self.view.images)
        self.cur_index = len(self.view.images) - 1
        self.update_list_indices()
        self.set_current_image()

    def update_list_indices(self) -> None:
        """
        Method to change the displayed indices in the interface list after changes
        :return: None
        """
        for a in range(len(self.view.main)):
            self.lst_nuc_model.item(a, 0).setText("Index: {}\nHash: {}".format(a, hash(self.view.main[a])))

    def on_selection_change(self) -> None:
        """
        Method to handle selection changes
        :return: None
        """
        index = self.ui.lst_nuc.selectionModel().selectedIndexes()
        self.ui.btn_merge.setEnabled(False)
        if index:
            self.last_index = self.cur_index
            self.cur_index = index[0].row()
            self.set_current_image()
            if len(index) > 1:
                self.ui.btn_merge.setEnabled(True)

    def set_current_image(self) -> None:
        """
        Method to change the displayed image
        :return: None
        """
        if self.cur_index < len(self.view.main):
            self.view.show_nucleus(self.cur_index, self.cur_channel)
            self.update_counting_label()

    def update_counting_label(self) -> None:
        """
        Method to update the counting label
        :return:
        """
        self.ui.lbl_number.setText("Foci: {}".format(self.view.cur_foc_num))


class NucView(QGraphicsView):

    def __init__(self, image: np.ndarray, handler: ROIHandler, commands: List[Tuple[str, Tuple[Any]]],
                 cur_channel: int = None, show: bool = True, edit: bool = False, max_channel: int = None,
                 db_curs: sqlite3.Cursor = None, parent: QWidget = None):
        super(NucView, self).__init__()
        self.par = parent
        self.setMouseTracking(True)
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        self.setMinimumSize(
            400,
            400
        )
        self.image = image
        self.handler = handler
        self.assmap = Detector.create_association_map(handler.rois)
        self.main = list(self.assmap.keys())
        self.main_channel = self.handler.idents.index(self.main[0].ident)
        self.cur_ind = 0
        self.cur_nuc = self.main[0]
        self.channel = cur_channel
        self.max_channel = max_channel
        self.curs = db_curs
        self.show = show
        self.edit = edit
        self.split = False
        self.temp_split = None
        self.pos = None
        self.temp_foc = None
        self.images = []
        self.foc_group = []
        self.map = {}
        self.commands = commands
        self.cur_foc_num = 0
        scene = QGraphicsScene(self)
        scene.setSceneRect(0, 0, self.width(), self.height())
        self.setScene(scene)
        for nuc in self.main:
            self.images.append(self.convert_roi_to_numpy(nuc))
        # Initialization of the background image
        self.sc_bckg = self.scene().addPixmap(QPixmap())
        self.show_nucleus(self.cur_ind, self.channel)

    def show_nucleus(self, cur_ind: int, channel: int) -> None:
        """
        Method to show a channel of the nucleus specified by index
        :param cur_ind: The index of the nucleus
        :param channel: The channel to show
        :return: None
        """
        self.cur_ind = cur_ind
        self.cur_nuc = self.main[cur_ind]
        self.channel = channel
        self.scene().setSceneRect(0, 0, self.width() - 5, self.height() - 5)
        pmap = QPixmap()
        pmap.convertFromImage(NucView.get_qimage_from_numpy(
            self.convert_roi_to_numpy(self.cur_nuc), mode="RGB" if self.channel > self.max_channel else "L"))
        tempmap = pmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
        self.sc_bckg.setPixmap(tempmap)
        x_scale = tempmap.width() / pmap.width()
        y_scale = tempmap.height() / pmap.height()
        x_trans = self.scene().width() / 2 - tempmap.width() / 2
        y_trans = self.scene().height() / 2 - tempmap.height() / 2
        self.sc_bckg.setPos(self.scene().width() / 2 - tempmap.width() / 2,
                            self.scene().height() / 2 - tempmap.height() / 2)
        self.clear_scene()
        self.cur_foc_num = 0
        if self.show and self.channel != self.main_channel:
            nuc_dat = self.cur_nuc.calculate_dimensions()
            x_offset = nuc_dat["minX"]
            y_offset = nuc_dat["minY"]
            for focus in self.assmap[self.cur_nuc]:
                c_ind = self.handler.idents.index(focus.ident)
                if c_ind == self.channel or self.channel > len(self.handler.idents) - 1:
                    foc = QGraphicsFocusItem(color_index=self.handler.idents.index(focus.ident))
                    temp = focus.calculate_dimensions()
                    dim = (temp["width"], temp["height"])
                    c = temp["center"]
                    ulp = ((c[0] - dim[0] / 2 - x_offset) * x_scale + x_trans,
                           (c[1] - dim[1] / 2 - y_offset) * y_scale + y_trans)
                    bbox = QRectF(ulp[0], ulp[1], dim[0] * x_scale, dim[1] * y_scale)
                    foc.setRect(bbox)
                    self.map[foc] = focus
                    self.foc_group.append(foc)
                    self.scene().addItem(foc)
                    self.cur_foc_num += 1

    @staticmethod
    def get_qimage_from_numpy(numpy: np.ndarray, mode: str = None) -> ImageQt:
        """
        Method to convert a numpy array to an QImage

        :param numpy: The array to convert
        :param mode: The mode to use for conversion
        :return: The QImage
        """
        img = Image.fromarray(numpy, mode)
        qimg = ImageQt(img)
        return qimg

    def clear_scene(self) -> None:
        """
        Method to remove all displayed foci from the screen
        :return: None
        """
        for item in self.foc_group:
            self.scene().removeItem(item)
        self.foc_group.clear()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.show_nucleus(self.cur_ind, self.channel)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        super(NucView, self).keyPressEvent(event)
        if event.key() == Qt.Key_Delete and not self.split:
            rem = []
            for item in self.foc_group:
                if item.isSelected():
                    self.handler.remove_roi(self.map[item])
                    self.commands.extend((("DELETE FROM roi WHERE hash=?",
                                           (hash(self.map[item]),)),
                                          ("DELETE FROM points WHERE hash=?",
                                           (hash(self.map[item]),))))
                    self.assmap[self.map[item].associated].remove(self.map[item])
                    del self.map[item]
                    rem.append(item)
                    self.scene().removeItem(item)
                    self.cur_foc_num -= 1
                    self.par.update_counting_label()
            for item in rem:
                self.foc_group.remove(item)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super(NucView, self).mousePressEvent(event)
        if self.edit and event.button() == Qt.LeftButton and \
                self.channel < self.handler.idents.index(self.handler.main) and not self.split:
            point = self.mapToScene(event.pos())
            p = self.itemAt(point.x(), point.y())
            if isinstance(p, QGraphicsPixmapItem):
                self.pos = event.pos()
                self.temp_foc = QGraphicsFocusItem(color_index=self.channel)
                self.scene().addItem(self.temp_foc)
        elif self.split and event.button() == Qt.LeftButton:
            point = self.mapToScene(event.pos())
            p = self.itemAt(point.x(), point.y())
            if isinstance(p, QGraphicsPixmapItem):
                self.pos = event.pos()
                self.temp_split = QGraphicsLineItem()
                pen = QPen()
                pen.setStyle(Qt.DashDotLine)
                pen.setWidth(3)
                pen.setBrush(QBrush(QColor(207, 255, 4)))
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                self.temp_split.setPen(pen)
                self.scene().addItem(self.temp_split)
        else:
            self.pos = None
            if self.temp_split is not None:
                self.scene().removeItem(self.temp_split)
            if self.temp_foc is not None:
                self.scene().removeItem(self.temp_foc)
            self.temp_foc = None
            self.temp_split = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        super(NucView, self).mouseMoveEvent(event)
        if self.temp_foc is not None:
            tw = max(event.pos().x(), self.pos.x()) - min(self.pos.x(), event.pos().x())
            th = max(event.pos().y(), self.pos.y()) - min(self.pos.y(), event.pos().y())
            width = max(tw, th)
            height = max(tw, th)
            x = self.pos.x() - width
            y = self.pos.y() - height
            bbox = QRectF(
                x,
                y,
                width * 2,
                height * 2
            )
            self.temp_foc.setRect(bbox)
        elif self.temp_split is not None:
            self.temp_split.setLine(self.pos.x(), self.pos.y(), event.pos().x(), event.pos().y())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        super(NucView, self).mouseReleaseEvent(event)
        if self.temp_foc is not None:
            cur_nump = self.convert_roi_to_numpy(self.main[self.cur_ind])
            offset_factor = self.sc_bckg.boundingRect().height() / len(cur_nump)
            hard_offset = self.sc_bckg.pos()
            bbox = self.temp_foc.boundingRect()
            tx = bbox.x() + 1 / 2 * bbox.width()
            ty = bbox.y() + 1 / 2 * bbox.height()
            x_center = (tx - hard_offset.x()) / offset_factor
            y_center = (ty - hard_offset.y()) / offset_factor
            height = bbox.height() / offset_factor / 2
            width = bbox.width() / offset_factor / 2
            mask = np.zeros(shape=(len(cur_nump), len(cur_nump[0])))
            rr, cc = ellipse(y_center, x_center, height, width, shape=mask.shape)
            mask[rr, cc] = 1
            cur_roi = ROI(main=False, auto=False, channel=self.handler.idents[self.channel],
                          associated=self.cur_nuc)
            nuc_dat = self.cur_nuc.calculate_dimensions()
            x_offset = nuc_dat["minX"]
            y_offset = nuc_dat["minY"]
            for y in range(len(mask)):
                for x in range(len(mask[0])):
                    if mask[y][x] > 0:
                        inten = cur_nump[y][x]
                        cur_roi.add_point((x + x_offset, y + y_offset), inten)
                        self.commands.append(("INSERT INTO points VALUES(?, ?, ?, ?)",
                                              (-1, x + x_offset, y + y_offset, np.int(inten))))
            self.handler.rois.append(cur_roi)
            roidat = cur_roi.calculate_dimensions()
            stats = cur_roi.calculate_statistics()
            ellp = cur_roi.calculate_ellipse_parameters()
            imghash = self.handler.ident
            self.commands.extend(
                (("INSERT INTO roi VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                  (hash(cur_roi), imghash, False, cur_roi.ident, str(roidat["center"]), roidat["width"],
                   roidat["height"], hash(self.cur_nuc))),
                 ("UPDATE points SET hash=? WHERE hash=-1",
                  (hash(cur_roi),)),
                 ("INSERT OR IGNORE INTO statistics VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (hash(cur_roi), imghash, stats["area"], stats["intensity average"], stats["intensity median"],
                   stats["intensity maximum"], stats["intensity minimum"], stats["intensity std"],
                   str(ellp["center"]), str(ellp["major_axis"][0]), str(ellp["major_axis"][1]),
                   ellp["major_slope"], ellp["major_length"], ellp["major_angle"], str(ellp["minor_axis"][0]),
                   str(ellp["minor_axis"][1]), ellp["minor_length"], ellp["shape_match"])))
            )
            self.foc_group.append(self.temp_foc)
            self.map[self.temp_foc] = cur_roi
            self.pos = None
            self.temp_foc = None
            self.scene().update()
            self.assmap = Detector.create_association_map(self.handler.rois)
            self.cur_foc_num += 1
            self.par.update_counting_label()
        elif self.temp_split is not None:
            cur_nump = self.main[self.cur_ind].get_as_numpy()
            offset_factor = self.sc_bckg.boundingRect().height() / len(cur_nump)
            hard_offset = self.sc_bckg.pos()
            nuc_dat = self.cur_nuc.calculate_dimensions()
            x_offset = nuc_dat["minX"]
            y_offset = nuc_dat["minY"]
            start_x = (self.pos.x() - hard_offset.x()) / offset_factor + x_offset
            start_y = (self.pos.y() - hard_offset.y()) / offset_factor + y_offset
            stop_x = (event.pos().x() - hard_offset.x()) / offset_factor + x_offset
            stop_y = (event.pos().y() - hard_offset.y()) / offset_factor + y_offset
            # Calculate line equation y = mx + n
            m = (stop_y - start_y) / (stop_x - start_x)
            n = stop_y - stop_x * m
            # Compare each point of nucleus with line
            aroi = ROI(channel=self.cur_nuc.ident)
            broi = ROI(channel=self.cur_nuc.ident)
            # Compare each center of foci with line
            for p in self.cur_nuc.points:
                ly = m * p[0] + n
                if ly > p[1]:
                    aroi.add_point(p, self.cur_nuc.inten[p])
                else:
                    broi.add_point(p, self.cur_nuc.inten[p])
            c = (aroi.calculate_dimensions()["center"], broi.calculate_dimensions()["center"])
            for foc in self.assmap[self.cur_nuc]:
                fc = foc.calculate_dimensions()["center"]
                d1 = math.sqrt((c[0][0] - fc[0]) ** 2 + (c[0][1] - fc[1]) ** 2)
                d2 = math.sqrt((c[1][0] - fc[0]) ** 2 + (c[1][1] - fc[1]) ** 2)
                if d1 < d2:
                    foc.associated = aroi
                else:
                    foc.associated = broi
            # Remove line
            self.scene().removeItem(self.temp_split)
            self.handler.rois.remove(self.cur_nuc)
            self.handler.rois.extend((aroi, broi))
            self.assmap = Detector.create_association_map(self.handler.rois)
            adat = aroi.calculate_dimensions()
            bdat = broi.calculate_dimensions()
            astat = aroi.calculate_statistics()
            bstat = broi.calculate_statistics()
            aell = aroi.calculate_ellipse_parameters()
            bell = broi.calculate_ellipse_parameters()
            imghash = self.handler.ident
            self.commands.extend((
                ("INSERT INTO roi VALUES (?, ?, ?, ?, ? ,?, ?, ?)",
                 (hash(aroi), imghash, False, self.cur_nuc.ident, str(adat["center"]), adat["width"],
                  adat["height"], None)),
                ("INSERT INTO statistics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 (hash(aroi), imghash, astat["area"], astat["intensity average"], astat["intensity median"],
                  astat["intensity maximum"], astat["intensity minimum"], astat["intensity std"],
                  str(aell["center"]), str(aell["major_axis"][0]), str(aell["major_axis"][1]), aell["major_slope"],
                  aell["major_length"], aell["major_angle"], str(aell["minor_axis"][0]), str(aell["minor_axis"][1]),
                  aell["minor_length"], aell["shape_match"])),
                ("INSERT INTO roi VALUES (?, ?, ?, ?, ? ,?, ?, ?)",
                 (hash(broi), imghash, False, self.cur_nuc.ident, str(bdat["center"]), bdat["width"],
                  bdat["height"], None)),
                ("INSERT INTO statistics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 (hash(broi), imghash, bstat["area"], bstat["intensity average"], bstat["intensity median"],
                  bstat["intensity maximum"], bstat["intensity minimum"], bstat["intensity std"],
                  str(bell["center"]), str(bell["major_axis"][0]), str(bell["major_axis"][1]), bell["major_slope"],
                  bell["major_length"], bell["major_angle"], str(bell["minor_axis"][0]), str(bell["minor_axis"][1]),
                  bell["minor_length"], bell["shape_match"])),
                ("DELETE FROM roi WHERE hash=?",
                 (hash(self.cur_nuc),)),
                ("DELETE FROM points WHERE hash=?",
                 (hash(self.cur_nuc),)),
            ))
            for p, inten in aroi.inten.items():
                self.commands.append(
                    ("INSERT INTO points VALUES (?, ?, ?, ?)",
                     (hash(aroi), p[0], p[1], inten))
                )
            for p, inten in broi.inten.items():
                self.commands.append(
                    ("INSERT INTO points VALUES (?, ?, ?, ?)",
                     (hash(broi), p[0], p[1], inten))
                )
            for foc in self.assmap[aroi]:
                self.commands.append(
                    ("UPDATE roi SET associated=? WHERE hash=?",
                     (hash(aroi), hash(foc)))
                )
            for foc in self.assmap[broi]:
                self.commands.append(
                    ("UPDATE roi SET associated=? WHERE hash=?",
                     (hash(broi), hash(foc)))
                )
            self.main.remove(self.cur_nuc)
            self.main.extend((aroi, broi))
            del self.images[self.cur_ind]
            self.images.extend([self.convert_roi_to_numpy(x, True) for x in (aroi, broi)])
            self.cur_nuc = aroi
            self.temp_split = None
            self.par.update_nucleus_list()
            self.scene().update()

    def convert_roi_to_numpy(self, roi, full=False):
        dims = roi.calculate_dimensions()
        y_dist = dims["maxY"] - dims["minY"] + 1
        x_dist = dims["maxX"] - dims["minX"] + 1
        if self.channel > self.max_channel or full:
            channel = self.image
            numpy = np.zeros((y_dist, x_dist, 3), dtype=np.uint8)
        else:
            channel = self.image[..., self.channel]
            numpy = np.zeros((y_dist, x_dist), dtype=np.uint8)
        for p in roi.points:
            numpy[p[1] - dims["minY"], p[0] - dims["minX"]] = channel[p[1]][p[0]]
        return numpy


class QGraphicsFocusItem(QGraphicsEllipseItem):
    COLORS = [
        QColor(255, 50, 0),  # Red
        QColor(50, 255, 0),  # Green
        QColor(255, 255, 0),  # Yellow
        QColor(255, 0, 255),  # Magenta
        QColor(0, 255, 255),  # Cyan
    ]

    def __init__(self, color_index: int = 0):
        super(QGraphicsFocusItem, self).__init__()
        col_num = len(QGraphicsFocusItem.COLORS)
        self.main_color = QGraphicsFocusItem.COLORS[color_index if color_index < col_num else col_num % color_index]
        self.hover_color = self.main_color.lighter(150)
        self.sel_color = self.main_color.lighter(200)
        self.cur_col = self.main_color
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

    def hoverEnterEvent(self, *args, **kwargs):
        self.cur_col = self.hover_color
        self.update()

    def hoverLeaveEvent(self, *args, **kwargs):
        self.cur_col = self.main_color
        self.update()

    def paint(self, painter: QPainter, style: QStyleOptionGraphicsItem, widget: QWidget = None) -> None:
        if self.isSelected():
            painter.setPen(QPen(self.sel_color, 6))
        else:
            painter.setPen(QPen(self.cur_col, 3))
        painter.drawEllipse(self.rect())
        self.scene().update()


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
    msg.setText(text)
    msg.setInformativeText(info)
    msg.setWindowTitle(title)
    msg.exec_()


def main() -> None:
    """
    Function to start the program

    :return: None
    """
    sys.excepthook = exception_hook
    app = QtWidgets.QApplication(sys.argv)
    pixmap = QPixmap("banner_norm.png")
    splash = QSplashScreen(pixmap)
    splash.show()
    splash.showMessage("Loading")
    main_win = NucDetect()
    splash.finish(main_win)
    main_win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
