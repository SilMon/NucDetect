from __future__ import annotations

import traceback

import io
import copy
import datetime
import json
import os
import sqlite3
import sys
import time
from threading import Thread
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Union, Dict, List, Tuple, Any

import PyQt5
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qtawesome as qta
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtCore, QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import QSize, Qt, pyqtSignal, QRectF, QItemSelectionModel
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIcon, QPixmap, QColor, QBrush, QPen, QResizeEvent, \
    QKeyEvent, QMouseEvent, QPainter
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QHeaderView, QDialog, QSplashScreen, QSizePolicy, QWidget, \
    QVBoxLayout, QScrollArea, QMessageBox, QGraphicsScene, QGraphicsEllipseItem, QGraphicsView, QGraphicsItem, \
    QGraphicsPixmapItem, QLabel, QGraphicsLineItem, QStyleOptionGraphicsItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from qtconsole.qt import QtGui
from skimage.draw import ellipse

from Nucleus.core.Detector import Detector
from Nucleus.core.ROI import ROI
from Nucleus.core.ROIHandler import ROIHandler
from Nucleus.gui.settings.Settings import SettingsShowWidget, SettingsSlider, SettingsText, SettingsComboBox, \
    SettingsCheckBox, SettingsDial, SettingsSpinner, SettingsDecimalSpinner

PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, False)
PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, False)

ui_main = os.path.join(os.getcwd(), "nucdetect.ui")
ui_result_image_dialog = os.path.join(os.getcwd(), "result_image_dialog.ui")
ui_class_dial = os.path.join(os.getcwd(), "classification_dialog.ui")
ui_stat_dial = os.path.join(os.getcwd(), "statistics_dialog.ui")
ui_settings_dial = os.path.join(os.getcwd(), "settings_dialog.ui")
ui_modification_dial = os.path.join(os.getcwd(), "modification_dialog.ui")
database = os.path.join(os.pardir, f"database{os.sep}nucdetect.db")
tablescript = os.path.join(os.pardir, f"database{os.sep}nucdetect.sql")
settingsscript = os.path.join(os.pardir, f"database{os.sep}settings.sql")
result_path = os.path.join(os.pardir, "results")


class NucDetect(QMainWindow):
    """
    Created on 11.02.2019
    @author: Romano Weiss
    """
    prg_signal = pyqtSignal(str, int, int, str)
    selec_signal = pyqtSignal(bool)
    aa_signal = pyqtSignal(int, int)
    executor = Thread()

    def __init__(self):
        """
        Constructor of the main window
        """
        QMainWindow.__init__(self)
        # Connect to database
        self.connection = sqlite3.connect(database)
        self.cursor = self.connection.cursor()
        # Create tables if they do not exists
        tscript = open(tablescript, "r").read()
        self.cursor.executescript(tscript)
        # Insert standard settings into table if not already
        setscript = open(settingsscript, "r").read()
        self.cursor.executescript((setscript))
        # Load the settings from database
        self.settings = self.load_settings()
        # Create detector for analysis
        self.detector = Detector(settings=[].extend(list(self.settings.values())),
                                 logging=self.settings["logging"])
        # Initialize needed variables
        self.reg_images = {}
        self.sel_images = []
        self.cur_img = None
        self.roi_cache = None
        self.unsaved_changes = False
        # Setup UI
        self._setup_ui()
        self.setWindowTitle("NucDetect")
        self.setWindowIcon(QtGui.QIcon('logo.png'))

    def load_settings(self) -> None:
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
        self.ui = uic.loadUi(ui_main, self)
        # Initialization of the image list
        self.img_list_model = QStandardItemModel(self.ui.list_images)
        self.ui.list_images.setModel(self.img_list_model)
        self.ui.list_images.selectionModel().selectionChanged.connect(self.on_image_selection_change)
        self.ui.list_images.setWordWrap(True)
        self.ui.list_images.setIconSize(QSize(75, 75))
        # Initialization of the result table
        self.res_table_model = QStandardItemModel(self.ui.table_results)
        self.res_table_model.setHorizontalHeaderLabels(["Index", "Width", "Height", "Center", "Foci"])
        self.ui.table_results.setModel(self.res_table_model)
        self.ui.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Addition of on click listeners
        self.ui.btn_load.clicked.connect(self._show_loading_dialog)
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
        # Add button icons
        btn_col = QColor(47, 167, 212)
        self.ui.btn_load.setIcon(qta.icon("fa5.folder-open", color=btn_col))
        self.ui.btn_save.setIcon(qta.icon("fa5.save", color=btn_col))
        self.ui.btn_images.setIcon(qta.icon("fa5s.microscope", color=btn_col))
        self.ui.btn_statistics.setIcon(qta.icon("fa5.chart-bar", color=btn_col))
        self.ui.btn_categories.setIcon(qta.icon("fa5s.list-ul", color=btn_col))
        self.ui.btn_settings.setIcon(qta.icon("fa.cogs", color=btn_col))
        self.ui.btn_modify.setIcon(qta.icon("fa5s.tools", color=btn_col))
        self.ui.btn_analyse.setIcon(qta.icon("fa5s.hat-wizard", color=btn_col))
        self.ui.btn_analyse_all.setIcon(qta.icon("fa5s.hat-wizard", color="red"))
        self.ui.btn_delete_from_list.setIcon(qta.icon("fa5s.times", color=btn_col))
        self.ui.btn_clear_list.setIcon(qta.icon("fa5s.trash-alt", color=btn_col))
        # Create signal for thread-safe gui updates
        self.prg_signal.connect(self._set_progress)
        self.selec_signal.connect(self._select_next_image)
        pardir = os.getcwd()
        imgdir = os.path.join(os.path.dirname(pardir),
                              r"images")
        self.add_images_from_folder(imgdir)

    def on_image_selection_change(self) -> None:
        """
        Will be called if a new image is selected
        :return: None
        """
        self.sel_images.clear()
        for index in self.ui.list_images.selectionModel().selectedIndexes():
            self.sel_images.append(self.reg_images[self.img_list_model.item(index.row()).text()])
        if len(self.sel_images) > 0:
            hash_ = Detector.calculate_image_id(self.reg_images[self.img_list_model.item(index.row()).text()])
            # TODO Settings mit einbeziehen
            ana = self.cursor.execute(
                "SELECT analysed FROM images WHERE md5 = ?",
                (hash_, )
            ).fetchall()[0][0]
            if ana:
                self.roi_cache = self.load_rois_from_database(hash_)
                self.create_result_table_from_list(self.roi_cache)
                self.enable_buttons()
                self.ui.btn_analyse.setEnabled(False)
                self.cur_img = self.sel_images[0]
                self.ui.lbl_status.setText("Loaded analysis results from database")
            else:
                self.ui.lbl_status.setText("Program ready")
                self.res_table_model.setRowCount(0)
                self.enable_buttons(False, ana_buttons=False)
                self.ui.btn_analyse.setEnabled(True)
        else:
            self.ui.btn_analyse.setEnabled(False)

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

    def add_image_to_list(self, path: str) -> None:
        """
        Method to add an image to the list of loaded files. The image will be processed, added and loaded.
        :param path: The path leading to the file
        :return: None
        """
        # Fix loading of duplicate files
        temp = os.path.split(path)
        folder = temp[0].split(sep=os.sep)[-1]
        file = temp[1]
        if os.path.splitext(file)[1] in Detector.FORMATS:
            d = Detector.get_image_data(path)
            date = d["datetime"]
            t = date.decode("ascii").split(" ") if not isinstance(date, datetime.datetime) \
                else date.strftime("%d/%m/%Y, %H:%M:%S")
            item = QStandardItem()
            item_text = f"Name: {file}\nFolder: {folder}\nDate: {t[0]}\nTime: {t[1]}"
            item.setText(item_text)
            item.setTextAlignment(QtCore.Qt.AlignLeft)
            icon = QIcon()
            icon.addFile(path)
            item.setIcon(icon)
            self.img_list_model.appendRow(item)
            self.reg_images[item_text] = path
            key = Detector.calculate_image_id(path)
            if not self.cursor.execute(
                "SELECT * FROM images WHERE md5 = ?",
                    (key, )
            ).fetchall():
                self.cursor.execute(
                    "INSERT INTO images VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (key, d["datetime"], d["channels"], d["width"], d["height"],
                     str(d["x_res"]), str(d["y_res"]), d["unit"], 0, -1)
                )
            self.connection.commit()

    def add_images_from_folder(self, url: str) -> None:
        """
        Method to load a whole folder of images

        :param url: The path of the folder
        :return: None
        """
        for t in os.walk(url):
            for file in t[2]:
                self.add_image_to_list(os.path.join(t[0], file))

    def remove_image_from_list(self) -> None:
        """
        Method to remove an loaded image from the file list.

        :return: None
        """
        cur_ind = self.ui.list_images.currentIndex()
        del self.reg_images[self.img_list_model.item(cur_ind.row()).text()]
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
        if not self.sel_images:
            self.ui.list_images.select(self.img_list_model.index(0, 0))
        self.prg_signal.emit(f"Analysing {str(self.sel_images[0])}",
                             0, 100, "")
        self.cur_img = self.sel_images[0]
        self.sel_images.remove(self.sel_images[0])
        thread = Thread(target=self.analyze_image,
                        args=(self.cur_img,
                              "Analysis finished in {} -- Program ready",
                              100, 100,))
        thread.start()

    def analyze_image(self, path: str, message: str, percent: Union[int, float], maxi: Union[int, float]) -> None:
        """
        Method to analyse the image given by path

        :param path: The path leading to the image
        :param message: The message to display above the progress bar
        :param percent: The value of the progress bar
        :param maxi: The maximum of the progress bar
        :return:
        """
        self.enable_buttons(False)
        self.ui.list_images.setEnabled(False)
        start = time.time()
        self.prg_signal.emit("Starting analysis", 0, maxi, "")
        self.unsaved_changes = True
        self.prg_signal.emit("Analysing image", maxi*0.05, maxi, "")
        data = self.detector.analyse_image(path)
        self.roi_cache = data["handler"]
        s0 = time.time()
        self.prg_signal.emit(f"Ellipse parameter calculation", maxi * 0.9, maxi, "")
        with ThreadPoolExecutor(max_workers=None) as e:
            for roi in self.roi_cache:
                if roi.main:
                    e.submit(roi.calculate_ellipse_parameters)
        self.prg_signal.emit("Creating result table", maxi * 0.65, maxi, "")
        print(f"Calculation of ellipse parameters: {time.time() - s0:.4f}")
        self.create_result_table_from_list(data["handler"])
        print(f"Creation result table: {time.time()-s0:.4f} secs")
        self.prg_signal.emit("Checking database", maxi * 0.75, maxi, "")
        s1 = time.time()
        self.save_rois_to_database(data)
        print(f"Writing to database: {time.time() - s1:.4f} secs")
        self.prg_signal.emit(message.format(f"{time.time()-start:.2f} secs"),
                             percent, maxi, "")
        self.enable_buttons()
        self.ui.btn_analyse.setEnabled(False)
        self.ui.list_images.setEnabled(True)

    def save_rois_to_database(self, data: Dict[str, Union[int, float, str]]) -> None:
        """
        Method to save the data stored in the ROIHandler rois to the database

        :param data: The data dict returned by the Detector class
        :return: None
        """
        con = sqlite3.connect(database)
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
        for roi in data["handler"].rois:
            dim = roi.calculate_dimensions()
            # Calculate ellipse parameters if roi is main, else use template
            ellp = roi.calculate_ellipse_parameters()
            stats = roi.calculate_statistics()
            asso = hash(roi.associated) if roi.associated is not None else None
            curs.execute(
                "INSERT OR IGNORE INTO roi VALUES (?, ?, ?, ?, ?, ?, ?,?)",
                (hash(roi), key, True, roi.ident, str(dim["center"]), dim["width"], dim["height"], asso)
            )
            for p in roi.points:
                curs.execute(
                    "INSERT OR IGNORE INTO points VALUES (?, ?, ?, ?)",
                    (hash(roi), p[0], p[1], roi.inten[p])
                )
            curs.execute(
                "INSERT OR IGNORE INTO statistics VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (hash(roi), key, stats["area"], stats["intensity average"], stats["intensity median"],
                 stats["intensity maximum"], stats["intensity minimum"], stats["intensity std"],
                 str(ellp["center"]), str(ellp["major_axis"][0]), str(ellp["major_axis"][1]),
                 ellp["major_slope"], ellp["major_length"], str(ellp["minor_axis"][0]), str(ellp["minor_axis"][1]),
                 ellp["minor_length"])
            )
        curs.execute(
            "UPDATE images SET analysed = ? WHERE md5 = ?",
            (True, key)
        )
        con.commit()
        con.close()
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
            for text in tabdat["data"][x]:
                item = QStandardItem()
                item.setText(str(text))
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
        self.ui.lbl_status.setText(f"{text} -- {(progress/maxi)*100:.2f}% {symbol}")
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

    def _analyze_all(self) -> None:
        """
        Method to perform concurrent batch analysis of registered images

        :return: None
        """
        with ProcessPoolExecutor(max_workers=None) as e:
            logstate = self.detector.logging
            self.detector.logging = False
            self.prg_signal.emit("Starting multi image analysis", 0, 100, "")
            res = e.map(self.detector.analyse_image, self.reg_images.values())
            ind = 1
            maxi = len(self.reg_images)
            for r in res:
                self.prg_signal.emit(f"Analysed images: {ind}/{maxi}",
                                     ind, maxi, "")
                self.save_rois_to_database(r)
                self.roi_cache = r["handler"]
                self.create_result_table_from_list(r["handler"])
                ind += 1
            self.roi_cache = list(res)[:-1]
            self.enable_buttons()
            self.ui.list_images.setEnabled(True)
            self.detector.logging = logstate
            self.prg_signal.emit("Analysis finished -- Program ready",
                                 100,
                                 100, "")
            self.selec_signal.emit(True)
            
    def load_rois_from_database(self, md5: int) -> ROIHandler:
        """
        Method to load all rois associated with this image

        :param md5: The md5 hash of the image
        :return: A ROIHandler containing all roi
        """
        print("Loaded roi from database")
        rois = ROIHandler(ident=md5)
        entries = self.cursor.execute(
            "SELECT * FROM roi WHERE image = ?",
            (md5, )
        ).fetchall()
        stats = self.cursor.execute(
            "SELECT * FROM statistics WHERE image = ?",
            (md5, )
        ).fetchall()
        names = self.cursor.execute(
            "SELECT * FROM channels WHERE md5 = ?",
            (md5, )
        ).fetchall()
        for name in names:
            rois.idents.insert(name[1], name[2])
        main_ = []
        sec = []
        statkeys = ("area", "intensity average",
                    "intensity median", "intensity maximum",
                    "intensity minimum", "intensity std")
        ellkeys = ("center", "major_axis", "major_length", "major_slope", "minor_axis", "minor_length")
        for entry in entries:
            temproi = ROI(channel=entry[3], main=entry[7] is None, associated=entry[7])
            temproi.id = entry[0]
            temproi.stats = dict(zip(statkeys, stats[2:8]))
            if temproi.main:
                main_.append(temproi)
                major = tuple(stats[9]), tuple(stats[10])
                minor = tuple(stats[13]), tuple(stats[14])
                ellp = (stats[8], major, stats[14:16], minor, stats[16])
                temproi.ell_params = dict(zip(ellkeys, ellp))
            else:
                sec.append(temproi)
            for p in self.cursor.execute(
                "SELECT * FROM points WHERE hash = ?",
                    (entry[0], )
            ).fetchall():
                temproi.add_point((p[1], p[2]), p[3])
            rois.add_roi(temproi)
        for m in main_:
            for s in sec:
                if s.associated == hash(m):
                    s.associated = m
        return rois

    def show_result_image(self) -> None:
        """
        Method to open an dialog to show the analysis results as plt plot

        :return: None
        """
        image_dialog = ImgDialog(image=Detector.load_image(self.cur_img), handler=self.roi_cache)
        image_dialog.setWindowTitle(f"Result Images for {self.cur_img}")
        image_dialog.setWindowIcon(QtGui.QIcon('logo.png'))
        image_dialog.setWindowFlags(image_dialog.windowFlags() |
                                    QtCore.Qt.WindowSystemMenuHint |
                                    QtCore.Qt.WindowMinMaxButtonsHint|
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
        self.roi_cache.export_data_as_csv(path=result_path)
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
        # TODO Poisson überprüfen
        stat_dialog = QDialog()
        stat_dialog.ui = uic.loadUi(ui_stat_dial, stat_dialog)
        stat_dialog.setWindowTitle("Statistics")
        stat_dialog.setWindowIcon(QtGui.QIcon('logo.png'))
        # Add statistics to list
        stat = self.roi_cache.calculate_statistics()
        assmap = self.detector.create_association_map(self.roi_cache)
        # Add labels to first tab
        stat_dialog.ui.dist_par.addWidget(QLabel(f"Detected nuclei: {len(assmap)}"))
        empty = [x for x in assmap.values() if len(x) > 0]
        stat_dialog.ui.dist_par.addWidget(QLabel(f"Thereof empty: {len(assmap) - len(empty)}"))
        colmarks = ["ro", "go", "co", "mo", "yo", "ko"]
        roinum = {}
        poiss_plots = []
        int_plots = []
        val_plots = [[], []]
        valint_plots = []
        colors = ["red", "green", "blue", "cyan", "magenta", "yellow", "black", "white"]
        # Add foci related labels
        for roi in self.roi_cache:
            if not roi.main:
                if roi.ident not in roinum:
                    roinum[roi.ident] = {roi.associated: 1}
                elif roi.associated in roinum[roi.ident]:
                    roinum[roi.ident][roi.associated] += 1
                else:
                    roinum[roi.ident][roi.associated] = 1
        for x in self.roi_cache.idents:
            if x != self.roi_cache.main:
                stat_dialog.ui.dist_par.addWidget(QLabel(f"Detected foci ({x}): {stat['sec stats'][x]['number']}"))
                stat_dialog.ui.dist_par.addWidget(QLabel(f"Std. Dev. ({x}): {np.std(list(roinum[x].values())):.2f}"))
                stat_dialog.ui.int_par.addWidget(QLabel(f"Average Intensity ({x}):"
                                                        f" {stat['sec stats'][x]['intensity average']:.2f}"))
                stat_dialog.ui.int_par.addWidget(QLabel(f"Std. Intensity ({x}): "
                                                        f"{stat['sec stats'][x]['intensity std']:.2f}"))
                stat_dialog.ui.val_par.addWidget(QLabel(f"Max. number ({x}): {max(roinum[x].values())}"))
                stat_dialog.ui.val_par.addWidget(QLabel(f"Min. number ({x}): {min(roinum[x].values())}"))
                stat_dialog.ui.val_par.addWidget(QLabel(f"Max. intensity ({x}):"
                                                        f" {stat['sec stats'][x]['intensity maximum']:.2f}"))
                stat_dialog.ui.val_par.addWidget(QLabel(f"Min. intensity ({x}):"
                                                        f" {stat['sec stats'][x]['intensity minimum']:.2f}"))
                # Preparation of plots
                poiss_plots.append(PoissonCanvas(np.average(list(roinum[x].values())),
                                                 max(roinum[x].values()),
                                                 list(roinum[x].values()),
                                                 name=f"{x} channel poisson - {self.cur_img}",
                                                 title=f"{x} Channel"))
                vals = []
                for key, value in assmap.items():
                    temp = []
                    for val in value:
                        if val.ident == x:
                            tempstats = val.calculate_statistics()
                            temp.append(tempstats["intensity average"])
                    if len(temp) != 0:
                        vals.append(sum(temp)/len(temp))
                    else:
                        # TODO otherwise division by zero on second image
                        vals.append(0)
                int = BarChart(name=f"{x} channel int - {self.cur_img}",
                               title=f"{x} Channel - Average Focus Intensity",
                               y_title="Average Intensity", x_title="Nucleus Index", x_label_rotation=45,
                               values=[vals],
                               colors=[colors[self.roi_cache.idents.index(x)]]*len(vals),
                               labels=[np.arange(len(vals))])
                int.setToolTip((f"Shows the average {x} foci intensity for the nucleus with the given index.\n"
                                f"255 is the maximal possible value. If no intensity is shown, no {x} foci were\n"
                                "detected in the respective nucleus"))
                int_plots.append(int)
                val_plots[0].append((np.arange(len(roinum[x].values()))))
                val_plots[1].append(roinum[x].values())
                valint_plots.append(stat["sec stats"][x]["intensity list"])

        chans = self.roi_cache.idents.copy()
        chans.remove(self.roi_cache.main)
        cnvs_num = XYChart(x_values=val_plots[0], y_values=val_plots[1], col_marks=colmarks[:len(chans)],
                           dat_labels=chans, name=f"numbers - {self.cur_img}",
                           title="Foci Number", x_title="Nucleus Index", y_title="Foci")
        ind = 0
        x_values = []
        y_values = []
        for key, value in assmap.items():
            for focus in value:
                chan_ind = self.roi_cache.idents.index(focus.ident)
                if len(x_values)-1 < chan_ind:
                    x_values.append([ind])
                    y_values.append([focus.calculate_statistics()["intensity average"]])
                else:
                    x_values[chan_ind].append(ind)
                    y_values[chan_ind].append(focus.calculate_statistics()["intensity average"])
            ind += 1
        colm = colmarks[:len(self.roi_cache.idents)-1]
        labels = self.roi_cache.idents[:len(self.roi_cache.idents)-1]
        cnvs_int = XYChart(x_values=x_values, y_values=y_values, col_marks=colm,
                           dat_labels=labels,
                           name=f"intensities - {key}", title="Intensity", x_title="Nucleus Index",
                           y_title="Average Intensity")
        stat_dialog.ui.vl_vals.addWidget(NavigationToolbar(cnvs_num, stat_dialog))
        stat_dialog.ui.vl_vals.addWidget(cnvs_num)
        stat_dialog.ui.vl_vals.addWidget(NavigationToolbar(cnvs_int, stat_dialog))
        stat_dialog.ui.vl_vals.addWidget(cnvs_int)
        for plot in poiss_plots:
            stat_dialog.ui.vl_poisson.addWidget(NavigationToolbar(plot, stat_dialog))
            stat_dialog.ui.vl_poisson.addWidget(plot)
        for plot in int_plots:
            stat_dialog.ui.vl_int.addWidget(NavigationToolbar(plot, stat_dialog))
            stat_dialog.ui.vl_int.addWidget(plot)
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
        cl_dialog.ui = uic.loadUi(ui_class_dial, cl_dialog)
        cl_dialog.setWindowTitle("Classification")
        cl_dialog.setWindowIcon(QtGui.QIcon('logo.png'))
        # categories = self.detector.get_categories(self.cur_img["key"])
        hash_ = Detector.calculate_image_id(self.cur_img)
        categories = self.cursor.execute(
            "SELECT category FROM categories WHERE image = ?",
            (hash_,)
        )
        cate = ""
        for cat in categories:
            cate += str(cat) + "\n"
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
            hash_ = Detector.calculate_image_id(self.cur_img)
            self.cursor.execute(
                "DELETE FROM categories WHERE image = ?",
                (hash_,)
            )
            for cat in categories:
                self.cursor.execute(
                    "INSERT INTO categories VALUES(?, ?)",
                    (hash_, cat)
                )

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
        mod = ModificationDialog(image=Detector.load_image(self.cur_img), handler=self.roi_cache)
        mod.setWindowTitle("Modification")
        mod.setWindowIcon(QtGui.QIcon("logo.png"))
        mod.setWindowFlags(mod.windowFlags() |
                           QtCore.Qt.WindowSystemMenuHint |
                           QtCore.Qt.WindowMinMaxButtonsHint |
                           QtCore.Qt.Window)
        code = mod.exec()
        if code == QDialog.Accepted:
            self.create_result_table_from_list(self.roi_cache)
        elif code == QDialog.Rejected:
            self.roi_cache = mod.handler

    def on_close(self) -> None:
        """
        Will be called if the program window closes

        :return:
        """
        self.connection.close()


class ResultFigure(FigureCanvas):

    def __init__(self, name: str, width: Union[int, float] = 4,
                 height: Union[int, float] = 4, dpi: Union[int, float] = 65, parent: QWidget = None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.name = name
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)

    def show_image(self, image: np.ndarray, image_title: str = "", show_axis: str = "On") -> None:
        """
        Method to show an image in the ResultFigure

        :param image: The image to show as numpy array
        :param image_title: The title to display for the image
        :param show_axis: Indicates if the axis should be shown
        :return: None
        """
        ax = self.figure.add_subplot(111)
        ax.imshow(image)
        ax.axis(show_axis)
        ax.set_title(image_title)
        ax.set_ylabel("Height")
        ax.set_xlabel("Width")
        self.draw()

    def save(self) -> None:
        """
        Method to save the figure to file

        :return: None
        """
        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"results/images/statistics")
        os.makedirs(pathpardir, exist_ok=True)
        pathresult = os.path.join(pathpardir,
                                  "result - {}.png".format(self.name))
        self.fig.set_size_inches(30, 15)
        self.fig.set_dpi(450)
        self.fig.savefig(pathresult)


class MPLPlot(FigureCanvas):

    def __init__(self, name: str, width: Union[int, float] = 4, height: Union[int, float] = 4,
                 dpi: Union[int, float] = 65, parent: QWidget=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.name = name
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)

    def save(self) -> None:
        """
        Method to save the plot as image

        :return: None
        """
        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"results/images/statistics")
        os.makedirs(pathpardir, exist_ok=True)
        pathresult = os.path.join(pathpardir,
                                  "result - {}.png".format(self.name))
        self.fig.set_size_inches(30, 15)
        self.fig.set_dpi(450)
        self.fig.savefig(pathresult)


class PoissonCanvas(MPLPlot):

    def __init__(self, _lambda: Union[int, float], k: int, values: List[Union[int, float]],
                 title: str = "", name: str = "", parent: QWidget = None, width: Union[int, float] = 4,
                 height: Union[int, float] = 4, dpi: Union[int, float] = 65):
        super(PoissonCanvas, self).__init__(name, width, height, dpi, parent)
        self.title = title
        self.plot(_lambda, k, values)

    def plot(self, _lambda: Union[int, float], k: int, values: List[Union[int, float]]) -> None:
        poisson = np.random.poisson(_lambda, k)
        ax = self.figure.add_subplot(111)
        objects = np.arange(k)
        x_pos = np.arange(max(values))
        conv_values = np.zeros(max(values))
        for val in values:
            conv_values[val - 1] += 1
        s = sum(conv_values)
        conv_values = [(x/s)*100 for x in conv_values]
        ax.set_title("Poisson Distribution - " + self.title)
        ax.bar(x_pos, poisson, align="center", alpha=0.5, label="Poisson Distribution")
        ax.bar(x_pos, conv_values, align="center", alpha=0.5, label="Actual Distribution")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(objects, rotation=45)
        ax.set_ylabel("Probability [%]")
        ax.set_xlabel("Foci number [N]")
        ax.legend()
        self.draw()


class XYChart(MPLPlot):

    def __init__(self, x_values: List[Union[int, float]], y_values: List[Union[int, float]], dat_labels: List[str],
                 col_marks: List[str] = ("ro",), parent: QWidget = None, name: str = "", title: str = "",
                 x_title: str = "", y_title: str = "", width: Union[int, float] = 4, height: Union[int, float] = 4,
                 dpi: Union[int, float] = 65, x_label_max_num: int = 20, y_label_max_num: int = 20,
                 x_label_rotation: Union[int, float] = 0, y_label_rotation: Union[int, float] = 0):
        super(XYChart, self).__init__(name, width, height, dpi, parent)
        self.name = name
        self.title = title
        self.x_label_max_num = x_label_max_num
        self.y_label_max_num = y_label_max_num
        self.x_title = x_title
        self.y_title = y_title
        self.x_label_rotation = x_label_rotation
        self.y_label_rotation = y_label_rotation
        self.x_values = x_values
        self.y_values = y_values
        self.dat_label = dat_labels
        self.colmarks = col_marks
        self.lines = []
        self.plot()

    def plot(self) -> None:
        ax = self.figure.add_subplot(111)
        ax.set_title(self.title)
        ax.set_ylabel(self.y_title)
        ax.set_xlabel(self.x_title)
        x_ticks = 0
        y_ticks = 0
        for x_val in self.x_values:
            for x in x_val:
                if x > x_ticks:
                    x_ticks = x
        for y_val in self.y_values:
            for y in y_val:
                if y > y_ticks:
                    y_ticks = y
        ax.xaxis.set_major_locator(plt.MaxNLocator(self.x_label_max_num))
        ax.yaxis.set_major_locator(plt.MaxNLocator(self.y_label_max_num))
        for ind in range(len(self.x_values)):
            if len(self.colmarks) is len(self.x_values):
                ax.plot(self.x_values[ind], self.y_values[ind], self.colmarks[ind])
            else:
                ax.plot(self.x_values[ind], self.y_values[ind])
        if self.dat_label:
            ax.legend(self.dat_label)
        self.draw()


class BarChart(MPLPlot):

    def __init__(self, values: List[int], labels: List[str], colors: List[str] = (), parent: QWidget = None,
                 overlay: bool = True, name: str = "", title: str = "", x_title: str = "", y_title: str = "",
                 width: Union[int, float] = 4, height: Union[int, float] = 4, dpi: Union[int, float] = 65,
                 x_label_rotation: Union[int, float] = 0, y_label_rotation: Union[int, float] = 0):
        super(BarChart, self).__init__(name, width, height, dpi, parent)
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.y_label_rotation = y_label_rotation
        self.x_label_rotation = x_label_rotation
        self.labels = labels
        self.values = values
        self.colors = colors
        self.overlay = overlay
        self.plot()

    def plot(self) -> None:
        ax = self.figure.add_subplot(111)
        ax.set_title(self.title)
        ax.set_ylabel(self.y_title)
        ax.set_xlabel(self.x_title)
        bar_width = max(0.8/len(self.values), 0.25)
        if not self.overlay:
            x_pos = np.arange(len(self.values))
            for lst in self.values:
                ax.bar(x_pos, lst, width=bar_width, align="center", alpha=1)
                x_pos = np.arange(x + bar_width for x in x_pos)
        else:
            alph = max(1/len(self.values), 0.2)
            x_pos = np.arange(len(self.values[0]))
            for lst in self.values:
                ax.bar(x_pos, lst, width=bar_width, align="center", alpha=alph, color=self.colors)
        if len(self.values) > 1:
            x_ticks_lst = [r + bar_width for r in range(len(self.values[0]))]
        else:
            x_ticks_lst = np.arange(len(self.values[0]))
        ax.set_xticks(x_ticks_lst)
        ax.set_xticklabels(x_ticks_lst, rotation=self.x_label_rotation)
        self.draw()


class ImgDialog(QDialog):
    MARKERS = [
        "r",  # Red
        "g",  # Green
        "b",  # Blue
        "c",  # Cyan
        "m",  # Magenta
        "y",  # Yellow
        "k",  # Black
        "w"   # White
    ]

    def __init__(self, image: np.ndarray, handler: ROIHandler, parent: QWidget = None):
        super(ImgDialog, self).__init__(parent)
        self.orig = image.copy()
        self.image = image
        self.handler = handler
        self.ui = uic.loadUi(ui_result_image_dialog, self)
        self.figure = Figure()
        self.figure.patch.set_alpha(0.1)
        self.canvas = FigureCanvas(self.figure)
        self.nav = NavigationToolbar(self.canvas, self)
        self.initialize_ui()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def initialize_ui(self) -> None:
        self.canvas.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        for ident in self.handler.idents:
            self.ui.cbx_channels.addItem(ident)
        self.ui.cbx_channels.addItem("Composite")
        self.ui.cbx_channels.setCurrentText("Composite")
        self.ui.cbx_channels.currentIndexChanged.connect(self.on_channel_selection_change)
        self.ui.navbar.insertWidget(0, self.nav, 3)
        self.layout().addWidget(self.canvas)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super(ImgDialog, self).resizeEvent(event)
        self.set_current_image()

    def on_channel_selection_change(self) -> None:
        if self.ui.cbx_channels.currentIndex() < len(self.handler.idents):
            self.image = self.orig[..., self.ui.cbx_channels.currentIndex()]
        else:
            self.image = self.orig.copy()
        self.set_current_image()

    def on_button_click(self) -> None:
        self.save_image()

    def set_current_image(self) -> None:
        cur_ind = self.ui.cbx_channels.currentIndex()
        # create an axis
        ax = self.figure.add_subplot(111)
        # Discard old graph
        ax.clear()
        ax.imshow(self.image, cmap="gray" if cur_ind < len(self.handler.idents) else matplotlib.rcParams["image.cmap"])
        dots = [[], [], [], []]
        for roi in self.handler.rois:
            center = roi.calculate_dimensions()["center"]
            ind = self.handler.idents.index(roi.ident)
            mark = self.MARKERS[ind]
            if cur_ind == len(self.handler.idents):
                if roi.main:
                    params = roi.calculate_ellipse_parameters()
                    c = params["center"]
                    cadj = c[1], c[0]
                    d1 = params["major_length"]
                    d2 = params["minor_length"]
                    p0, p1 = params["major_axis"]
                    p00, p10 = params["minor_axis"]
                    slope = params["major_slope"]
                    angle = params["major_angle"]
                    ell = Ellipse(cadj, d1, d2, angle=angle if slope > 0 else 360 - angle,
                                  color="gold", fill=None, linewidth=2, linestyle="--")
                    ax.add_patch(ell)
                    # Draw major axis
                    x = (p0[1], p1[1])
                    y = (p0[0], p1[0])
                    ax.plot(x, y, "gold")
                    # Draw minor axis
                    x = (p00[1], p10[1])
                    y = (p00[0], p10[0])
                    ax.plot(x, y, "goldenrod")
                else:
                    dots[0].append(center[0])
                    dots[1].append(center[1])
                    dots[2].append(mark)
            elif cur_ind == self.handler.idents.index(self.handler.main):
                if roi.main:
                    params = roi.calculate_ellipse_parameters()
                    c = params["center"]
                    cadj = c[1], c[0]
                    p0, p1 = params["major_axis"]
                    p00, p10 = params["minor_axis"]
                    d1 = params["major_length"]
                    d2 = params["minor_length"]
                    slope = params["major_slope"]
                    angle = params["major_angle"]
                    # Draw calculated ellipse
                    ell = Ellipse(cadj, d1, d2, angle=angle if slope > 0 else 360 - angle,
                                  color="gold", fill=None, linewidth=2, linestyle="--")
                    ax.add_patch(ell)
                    # Draw major axis
                    x = (p0[1], p1[1])
                    y = (p0[0], p1[0])
                    ax.plot(x, y, "gold")
                    # Draw minor axis
                    x = (p00[1], p10[1])
                    y = (p00[0], p10[0])
                    ax.plot(x, y, "goldenrod")
            elif cur_ind == ind:
                dots[0].append(center[0])
                dots[1].append(center[1])
                dots[2].append(mark)
        ax.scatter(dots[0], dots[1], marker="o", c=dots[2], s=16)
        ax.set_ylim(0, len(self.image))
        ax.set_xlim(0, len(self.image[0]))
        self.figure.tight_layout()
        self.canvas.draw()


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
        self.ui = uic.loadUi(ui_settings_dial, self)

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

    def add_menu_point(self, section:str, menupoint: Dict[str, Union[str, float, int]]) -> None:
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
        self.original = copy.copy(handler)
        self.show = True
        self.last_index = 0
        self.cur_index = 0
        self.cur_channel = 3
        self.max = 2
        self.mp = None
        self.ui = None
        self.view = None
        self.lst_nuc_model = None
        self.commands = []
        self.conn = sqlite3.connect(database)
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
        super(ModificationDialog, self).accept()

    def reject(self) -> None:
        self.handler = self.original
        super(ModificationDialog, self).reject()

    def initialize_ui(self) -> None:
        self.ui = uic.loadUi(ui_modification_dial, self)
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
            # TODO test3.tif
            # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
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
                                            "Do you really want to remove following nuclei: {}".format(sel),
                                            QMessageBox.Yes | QMessageBox.No)
                if code == QMessageBox.Yes:
                    offset = 0

                    for ind in sorted(sel):
                        nuc = self.view.main[ind + offset]
                        self.handler.rois.remove(nuc)
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
                                        "Do you really want to merge following nuclei: {}".format(sel),
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
                nuc_stat = seed.calculate_dimensions()
                self.commands.append(
                     ("UPDATE roi SET hash = ?, auto = ?, center = ?, width = ?, height = ? WHERE hash = ?",
                      (hash(seed), False, str(nuc_stat["center"]), nuc_stat["width"], nuc_stat["height"], mergehash[0]))
                )
                for h in mergehash:
                    self.commands.extend(
                        (("UPDATE roi SET associated = ? WHERE associated = ?",
                         (hash(seed), h)),
                         ("UPDATE points SET hash = ? WHERE hash = ?",
                         (hash(seed), h)),
                         ("DELETE FROM roi WHERE hash = ?",
                         (h, )))
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
            for item in self.foc_group:
                if item.isSelected():
                    # TODO Fehlerhaft
                    self.handler.remove_roi(self.map[item])
                    self.commands.extend((("DELETE FROM roi WHERE hash=?",
                                         (hash(self.map[item]),)),
                                         ("DELETE FROM points WHERE hash=?",
                                          (hash(self.map[item]),))))
                    del self.map[item]
                    self.scene().removeItem(item)
                    self.cur_foc_num -= 1
                    self.par.update_counting_label()

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
            cur_nump = self.main[self.cur_ind].get_as_numpy()
            offset_factor = self.sc_bckg.boundingRect().height() / len(cur_nump)
            hard_offset = self.sc_bckg.pos()
            bbox = self.temp_foc.boundingRect()
            tx = bbox.x() + 1/2 * bbox.width()
            ty = bbox.y() + 1/2 * bbox.height()
            x_center = (tx - hard_offset.x()) / offset_factor
            y_center = (ty - hard_offset.y()) / offset_factor
            height = bbox.height() / offset_factor / 2
            width = bbox.width() / offset_factor / 2
            mask = np.zeros(shape=cur_nump.shape)
            rr, cc = ellipse(y_center, x_center, height, width, shape=mask.shape)
            mask[rr, cc] = 1
            cur_roi = ROI(auto=False, channel=self.handler.idents[self.channel], associated=self.cur_nuc)
            nuc_dat = self.cur_nuc.calculate_dimensions()
            x_offset = nuc_dat["minX"]
            y_offset = nuc_dat["minY"]
            for y in range(len(mask)):
                for x in range(len(mask[0])):
                    if mask[y][x] > 0:
                        inten = cur_nump[y][x]
                        cur_roi.add_point((x + x_offset, y + y_offset), inten)
                        self.commands.append(
                            ("INSERT INTO points VALUES(?, ?, ?, ?)",
                             (-1, x + x_offset, y + y_offset, np.int(inten)))
                        )
            self.handler.rois.append(cur_roi)
            roidat = cur_roi.calculate_dimensions()
            imghash = self.handler.ident
            self.commands.extend(
                (("INSERT INTO roi VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                 (hash(cur_roi), imghash, False, cur_roi.ident, str(roidat["center"]), roidat["width"],
                  roidat["height"], hash(self.cur_nuc))),
                 ("UPDATE points SET hash=? WHERE hash=-1",
                 (hash(cur_roi),)))
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
                d1 = math.sqrt((c[0][0] - fc[0])**2 + (c[0][1] - fc[1])**2)
                d2 = math.sqrt((c[1][0] - fc[0])**2 + (c[1][1] - fc[1])**2)
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
            imghash = self.handler.ident
            self.commands.extend((
                ("INSERT INTO roi VALUES (?, ?, ?, ?, ? ,?, ?, ?)",
                 (hash(aroi), imghash, False, self.cur_nuc.ident, str(adat["center"]), adat["width"],
                  adat["height"], None)),
                ("INSERT INTO roi VALUES (?, ?, ?, ?, ? ,?, ?, ?)",
                 (hash(broi), imghash, False, self.cur_nuc.ident, str(bdat["center"]), bdat["width"],
                  bdat["height"], None)),
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
    # Print the traceback to console
    tb_infofile = io.StringIO()
    traceback.print_tb(traceback_obj, None, tb_infofile)
    # Show error message in GUI
    time_string = time.strftime("%Y-%m-%d, %H:%M:%S")
    title = "An error occured during execution"
    info = f"An {exc_type.__name__} occured at {time_string}"
    text = "During the execution of the program, following error occured:\n" \
           f"{''.join(traceback.format_exception(exc_type, exc_value, traceback_obj))}"
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(text)
    msg.setInformativeText(info)
    msg.setWindowTitle(title)
    msg.exec_()


def main():
    sys.excepthook = exception_hook
    app = QtWidgets.QApplication(sys.argv)
    pixmap = QPixmap("banner_norm.png")
    splash = QSplashScreen(pixmap)
    splash.show()
    splash.showMessage("Loading...")
    mainWin = NucDetect()
    splash.finish(mainWin)
    mainWin.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
        main()
