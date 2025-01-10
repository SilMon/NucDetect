from __future__ import annotations

import multiprocessing
import os
import shutil
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from datetime import datetime
from functools import partial
from threading import Thread
from typing import Union, Dict, Iterable, List, Tuple, Any

import PyQt5
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5.QtCore import QSize, pyqtSignal, QItemSelectionModel, QSortFilterProxyModel, QModelIndex, \
    QAbstractListModel, QTimer, Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QHeaderView, QDialog, QSplashScreen, QMessageBox

from core.Detector import Detector
from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler
from database.connections import Connector, Requester, Inserter
from definitions.icons import Icon, Color
from detector_modules.ImageLoader import ImageLoader
from dialogs.data import Editor, ExperimentDialog, StatisticsDialog, DataExportDialog
from dialogs.selection import ExperimentSelectionDialog
from dialogs.settings import AnalysisSettingsDialog, SettingsDialog
from gui import Paths, Util

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
    check_timer = QTimer()
    STANDARD_TABLE_HEADER = ["Image Name", "Image Identifier",
                             "ROI Identifier", "Center Y",
                             "Center X", "Area [px]", "Ellipticity[%]",
                             "Or. Angle [deg]", "Maj. Axis", "Min. Axis",
                             "match"]

    def __init__(self):
        """
        Constructor of the main window
        """
        QMainWindow.__init__(self)
        # Create working directories
        self.create_required_dirs()
        # Connect to database
        self.connector = Connector()
        # Create needed tables if necessary
        self.connector.create_tables()
        # Create standard settings if necessary
        self.connector.create_standard_settings()
        self.req_connector = Connector(protected=False)
        self.requester = Requester(self.req_connector)
        self.inserter = Inserter(self.connector)
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
        self.showMaximized()

    @staticmethod
    def create_required_dirs() -> None:
        """
        Method to create the working dirs of this program

        :return: None
        """
        # Create the NucDetect folder
        if not os.path.isdir(Paths.nuc_detect_dir):
            os.makedirs(Paths.nuc_detect_dir)
        # Create the thumbnail folder
        if not os.path.isdir(Paths.thumb_path):
            os.makedirs(Paths.thumb_path)
        # Create the results folder
        if not os.path.isdir(Paths.result_path):
            os.makedirs(Paths.result_path)
        # Create the images folder
        if not os.path.isdir(Paths.images_path):
            os.makedirs(Paths.images_path)
            shutil.copy2(os.path.join(os.pardir, "demo.tif"), os.path.join(Paths.images_path, "demo.tif"))
        # Create the log folder
        if not os.path.isdir(Paths.log_dir_path):
            os.makedirs(Paths.log_dir_path)

    def load_settings(self) -> Dict:
        """
        Method to load the saved Settings

        :return: None
        """
        settings_sql = self.requester.get_all_settings()
        settings = {}
        for row in settings_sql:
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
        self._initialize_window()
        # Initialization of the image list
        self._initialize_image_list()
        # Initialization of the result table
        self._initialize_result_table()
        # Addition of on click listeners
        self._connect_buttons()
        # Add button icons
        self._set_button_icons()
        self._connect_signals()

    def _initialize_window(self) -> None:
        """
        Method to initialize the window

        :return: None
        """
        self.ui = uic.loadUi(Paths.ui_main, self)
        self.ui.setStyleSheet(open(os.path.join(Paths.css_dir, "main.css")).read())
        # General Window Initialization
        self.setWindowTitle("NucDetect - Focus Analysis Software")
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.ui.lbl_logo.setPixmap(QPixmap(os.path.join(Paths.logo_dir, "banner.png")))

    def _initialize_image_list(self) -> None:
        """
        Method to initialize the image list

        :return: None
        """
        self.add_images_from_folder(Paths.images_path)
        self.img_list_model = ImageListModel(self.ui.list_images, paths=self.loaded_files)
        self.ui.list_images.setModel(self.img_list_model)
        self.ui.list_images.selectionModel().selectionChanged.connect(self.on_image_selection_change)
        self.ui.list_images.setWordWrap(True)
        self.ui.list_images.setIconSize(QSize(75, 75))
        self.ui.list_images.verticalScrollBar().valueChanged.connect(self.fetch_more_images_if_needed)

    def _initialize_result_table(self) -> None:
        """
        Method to initialize the result table

        :return: None
        """
        self.res_table_model = QStandardItemModel(self.ui.table_results)
        # Initialize the header
        self.res_table_model.setHorizontalHeaderLabels(NucDetect.STANDARD_TABLE_HEADER)
        # Enable sorting
        self.res_table_sort_model = TableFilterModel(self)
        self.res_table_sort_model.setSourceModel(self.res_table_model)
        self.ui.table_results.setModel(self.res_table_sort_model)
        self.ui.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def _connect_buttons(self) -> None:
        """
        Method to connect the buttons to their respective functions

        :return: None
        """
        self.ui.btn_load.clicked.connect(self._show_loading_dialog)
        self.ui.btn_experiments.clicked.connect(self.show_experiment_dialog)
        self.ui.btn_save.clicked.connect(self.save_results)
        self.ui.btn_analyse.clicked.connect(self.analyze)
        self.ui.btn_statistics.clicked.connect(self.show_statistics)
        self.ui.btn_settings.clicked.connect(self.show_settings)
        self.ui.btn_modify.clicked.connect(self.show_modification_window)
        self.ui.btn_analyse_all.clicked.connect(self.analyze_all)
        self.ui.btn_delete_from_list.clicked.connect(self.remove_image_from_list)
        self.ui.btn_clear_list.clicked.connect(self.clear_image_list)
        self.ui.btn_reload.clicked.connect(self.reload)

    def _set_button_icons(self) -> None:
        """
        Method to give the buttons their respective icons

        :return: None
        """
        self.ui.btn_load.setIcon(Icon.get_icon("FOLDER_OPEN"))
        self.ui.btn_experiments.setIcon(Icon.get_icon("FLASK"))
        self.ui.btn_save.setIcon(Icon.get_icon("SAVE"))
        self.ui.btn_statistics.setIcon(Icon.get_icon("CHART_BAR"))
        self.ui.btn_settings.setIcon(Icon.get_icon("COGS"))
        self.ui.btn_modify.setIcon(Icon.get_icon("TOOLS"))
        self.ui.btn_analyse.setIcon(Icon.get_icon("HAT_WIZARD_BLUE"))
        self.ui.btn_analyse_all.setIcon(Icon.get_icon("HAT_WIZARD_RED"))
        self.ui.btn_delete_from_list.setIcon(Icon.get_icon("TIMES"))
        self.ui.btn_clear_list.setIcon(Icon.get_icon("TRASH_ALT"))
        self.ui.btn_reload.setIcon(Icon.get_icon("SYNC"))

    def _connect_signals(self) -> None:
        """
        Method to connect the used signals

        :return: None
        """
        # Create signal for thread-safe gui updates
        self.prg_signal.connect(self._set_progress)
        self.selec_signal.connect(self._select_next_image)
        self.add_signal.connect(self.add_item_to_list)

    def reload(self) -> None:
        """
        Method to reload the images folder

        :return: None
        """
        self.loaded_files = []
        self.add_images_from_folder(Paths.images_path, reload=True)
        self.img_list_model.set_paths(self.loaded_files)

    def fetch_more_images_if_needed(self, value: int, threshold: float = 0.75):
        """
        Method to check if more items need to be fetched

        :param value: The current value of the scroll bar
        :param threshold: The threshold used to determine, if more items should be fetched
        :return: None
        """
        max_ = self.ui.list_images.verticalScrollBar().maximum()
        if value > max_ * threshold:
            self.img_list_model.fetchMore(QModelIndex())

    def on_image_selection_change(self) -> None:
        """
        Will be called if a new image is selected

        :return: None
        """
        for index in self.ui.list_images.selectionModel().selectedIndexes():
            self.cur_img = self.img_list_model.get_item_at_index(index.row()).data()
        if self.cur_img:
            ana = self.cur_img["analysed"]
            if ana:
                # Get information for this image
                experiment = self.show_experiment_loading_warning_dialog()
                self.prg_signal.emit(f"Loading data from database for {self.cur_img['file_name']}",
                                     0, 100, "")
                self.load_saved_data(experiment)
            else:
                self.ui.lbl_status.setText("Program ready")
                self.res_table_model.setRowCount(0)
                self.enable_buttons(False, ana_buttons=False)
        else:
            self.ui.btn_analyse.setEnabled(False)

    def show_experiment_loading_warning_dialog(self) -> Union[str, None]:
        """
        Method to show the experiment loading warning dialog for the current image

        :return: The name of the associated experiment, if any. None if the image is not associated
        """
        # Get the associated experiment
        exp = self.requester.get_experiment_for_image(self.cur_img["key"])
        if not exp:
            return
        # Get number of attached images
        num_imgs = len(self.requester.get_associated_images_for_experiment(exp))
        exit_code = self.open_two_choice_dialog(
            "Experiment attached!",
            "",
            f"The selected image is assigned to the experiment {exp}, "
            f"with {num_imgs} attached images. Loading it can take up"
            f" to approx. {num_imgs / 60:.2f} min ({num_imgs} secs). ",
            ("Load Experiment", "Load Image Data")
        )
        return exp if exit_code == QMessageBox.Ok else None

    @staticmethod
    def open_two_choice_dialog(title: str = "", info: str = "", text: str = "",
                               button_texts: Tuple[str, str] = ("", "")) -> int:
        """
        Method to open a two choice dialog

        :param title: The window title
        :param info: The info text
        :param text: The general text
        :param button_texts: The texts for the buttons
        :return: The exit code of the dialog
        """
        msg = QMessageBox()
        msg.setStyleSheet(open(os.path.join(Paths.css_dir, "messagebox.css"), "r").read())
        msg.setWindowIcon(Icon.get_icon("LOGO"))
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setInformativeText(info)
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.button(QMessageBox.Ok).setText(button_texts[0])
        msg.button(QMessageBox.Cancel).setText(button_texts[1])
        return msg.exec_()

    def load_saved_data(self, experiment: str = None) -> None:
        """
        Method to load saved data from the database

        :param experiment: Name of the experimental data to load. None if only image data should be loaded
        :return: None
        """
        # Disable Buttons and list during loading
        self.enable_buttons(state=False)
        self.ui.list_images.setEnabled(False)
        load_thread = threading.Thread(target=self._load_saved_data,
                                       args=(experiment,),
                                       daemon=True)
        load_thread.start()

    def _load_saved_data(self, experiment: str) -> None:
        """
        Private method to load the experiment data concurrently

        :param experiment: Name of the experiment
        :return: None
        """
        self.prg_signal.emit(f"Loading data from database for {self.cur_img['file_name']}, please wait...",
                             0, 100, "")
        # Load saved data from databank
        self.roi_cache = self.load_rois_from_database(self.cur_img["key"])
        # Create the result table from loaded data
        self.create_result_table_from_list(self.roi_cache, experiment)
        # Re-enable buttons and list
        self.ui.list_images.setEnabled(True)
        self.enable_buttons()
        self.prg_signal.emit(f"Data loaded from database for {self.cur_img['file_name']}",
                             100, 100, "")
        self.enable_buttons(state=True)
        self.ui.list_images.setEnabled(True)

    def show_experiment_dialog(self) -> None:
        """
        Method to show the experiment dialog

        :return: None
        """
        # Create data for dialog
        data = {"keys": [], "paths": []}
        for path in self.loaded_files:
            data["keys"].append(ImageLoader.calculate_image_id(path))
            data["paths"].append(path)
        exp_dialog = ExperimentDialog(data=data)
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
        paths = []
        for t in os.walk(url):
            tpaths = [os.path.join(t[0], x) for x in t[2]]
            paths.extend([x for x in tpaths if x not in self.loaded_files])
        # If no images where found, open a file dialog to add images
        if not paths:
            files = str(QFileDialog.getExistingDirectory(self, "Select Directory to load images from"))
            # Walk the folder to find all files inside it
            for t in os.walk(files):
                tpaths = [os.path.join(t[0], x) for x in t[2]]
                paths.extend([x for x in tpaths if x not in self.loaded_files])
        self.loaded_files.extend(sorted(paths, key=lambda x: os.path.basename(x)))
        # Add new paths to database
        self.add_images_to_database(self.loaded_files)

    def add_images_to_database(self, images: List[str]) -> None:
        """
        Method to add the given list of images to the database

        :param images: List of image paths
        :return: None
        """
        for image in images:
            self.add_image_information_to_database(image)

    def add_image_information_to_database(self, path: str) -> None:
        """
        Method to add the information about the given image to the database

        :param path: The file path for this image
        :return: None
        """
        # Get md5 hash of this image
        md5 = ImageLoader.calculate_image_id(path)
        # Check if the image is already registered
        if self.requester.check_if_image_is_registered(md5):
            return
        # Get the required information
        d = ImageLoader.get_image_data(path)
        # Add the data to the database
        self.inserter.add_new_image(md5, d["year"], d["month"], d["day"], d["hour"], d["day"],
                                    d["channels"], d["width"], d["height"], d["x_res"],
                                    d["y_res"], d["unit"])
        self.inserter.register_image_filename(path)
        self.connector.commit_changes()

    def add_item_to_list(self, item: QStandardItem) -> None:
        """
        Utility method to add an item to the image list

        :param item: The item to add
        :return: None
        """
        if item is not None:
            path = item.data()["path"]
            if path in self.loaded_files:
                return
            self.img_list_model.appendRow(item)
            self.loaded_files.append(path)
            self.add_image_information_to_database(path)

    def remove_image_from_list(self) -> None:
        """
        Method to remove a loaded image from the file list.

        :return: None
        """
        cur_ind = self.ui.list_images.currentIndex()
        data = self.img_list_model.item(cur_ind.row()).data()
        self.loaded_files.remove(data["path"])
        self.img_list_model.removeRow(cur_ind.row())

    def clear_image_list(self) -> None:
        """
        Method to clear the list of loaded images

        :return: None
        """
        self.img_list_model.clear()
        self.loaded_files.clear()

    def show_analysis_settings_dialog(self, show_redo_option: bool = False) -> Union[Dict, None]:
        """
        Method to show the analysis settings dialog

        :return: Bool which signifies if the dialog was confirmed or cancelled
        """
        anal_sett_dial = AnalysisSettingsDialog(settings=self.settings,
                                                all_=show_redo_option)
        code = anal_sett_dial.exec()
        if code == QDialog.Accepted:
            settings = anal_sett_dial.get_data()
            an_sett = settings["analysis_settings"]
            settings["analysis_settings"].update({x: y for (x, y) in self.settings.items() if x not in an_sett})
            return settings
        else:
            # If the dialog was rejected, abort analysis
            self.ui.list_images.setEnabled(True)
            self.enable_buttons(True)
            return None

    def analyze(self) -> None:
        """
        Method to analyze a loaded image

        :return: None
        """
        if not self.cur_img:
            self.selec_signal.emit(True)
        # Get settings for this analysis
        self.ui.list_images.setEnabled(False)
        self.enable_buttons(False)
        settings = self.show_analysis_settings_dialog()
        if not settings:
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
        data = self.detector.analyse_image(path, settings=analysis_settings, save_log=True)
        self.roi_cache = data["handler"]
        self.prg_signal.emit(f"Ellipse parameter calculation", maxi * 0.75, maxi, "")
        for roi in self.roi_cache:
            if roi.main:
                roi.calculate_ellipse_parameters()
        self.prg_signal.emit("Creating result table", maxi * 0.65, maxi, "")
        #print(f"Calculation of ellipse parameters: {time.time() - s0:.4f}")
        self.prg_signal.emit("Checking database", maxi * 0.9, maxi, "")
        s1 = time.time()
        self.save_rois_to_database(data)
        #print(f"Writing to database: {time.time() - s1:.4f} secs")
        self.prg_signal.emit(message.format(f"{time.time() - start:.2f} secs"),
                             percent, maxi, "")
        self.create_result_table_from_list(self.roi_cache)
        #print(f"Creation result table: {time.time() - s0:.4f} secs")
        self.enable_buttons()
        self.ui.list_images.setEnabled(True)
        self.reflect_item_status_changes()

    def analyze_all(self) -> None:
        """
        Method to perform concurrent batch analysis of registered images

        :return: None
        """
        self.enable_buttons(False)
        self.ui.list_images.setEnabled(False)
        self.unsaved_changes = True
        # Get settings for this analysis
        settings = self.show_analysis_settings_dialog(show_redo_option=True)
        if not settings:
            return
        thread = Thread(target=self._analyze_all, args=(settings,))
        thread.start()

    def _analyze_all(self, settings: Dict[str, Union[int, float, str, Iterable]], batch_size: int = 10) -> None:
        """
        Method to perform concurrent batch analysis of registered images

        :param settings: The settings for this analysis, e.g. channel names, active channels ect.
        :param batch_size: The number of images that are loaded parallel
        :return: None
        """
        start_time = time.time()
        # Use all available threads except 4
        with ProcessPoolExecutor(max_workers=round(multiprocessing.cpu_count() * 0.25)) as e:
            self.res_table_model.setRowCount(0)
            self.res_table_model.setColumnCount(2)
            self.res_table_model.setHorizontalHeaderLabels(["Image Name", "Image Hash",
                                                            "Number of Nuclei", "Number of Foci",
                                                            "Foci per Nucleus"])
            logstate = settings["analysis_settings"]["logging"]
            settings["analysis_settings"]["logging"] = False
            self.prg_signal.emit("Starting multi image analysis", 0, 100, "")
            paths = []
            for image in self.loaded_files:
                # Get md5 hash of file
                md5 = ImageLoader.calculate_image_id(image)
                if not self.requester.check_if_image_was_analysed(md5) or settings["re-analyse"]:
                    paths.append(image)
            self.write_to_log(f"Batch analysis of {len(paths)} images")
            ind = 1
            cur_batch = 1
            curind = 0
            # Define needed batch variables
            start = batch_size + 1 if batch_size < len(paths) else len(paths) + 1
            stop = len(paths) + batch_size if len(paths) > batch_size else len(paths) + 2
            step = batch_size
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
                    # Get the image hash and file name
                    name = self.requester.get_image_filename(r["handler"].ident)
                    name_item = QStandardItem(name)
                    ident_item = QStandardItem(r["handler"].ident)
                    mnum = len([x for x in r["handler"] if x.main])
                    fnum = len([x for x in r["handler"] if not x.main])
                    fpn = ((fnum / mnum) if mnum > 0 else 0)
                    main_item = QStandardItem(str(mnum))
                    focus_item = QStandardItem(str(fnum))
                    fpn_item = QStandardItem(str(f"{fpn:.2f}"))
                    name_item.setTextAlignment(QtCore.Qt.AlignCenter)
                    ident_item.setTextAlignment(QtCore.Qt.AlignCenter)
                    main_item.setTextAlignment(QtCore.Qt.AlignCenter)
                    focus_item.setTextAlignment(QtCore.Qt.AlignCenter)
                    fpn_item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.res_table_model.appendRow(
                        [name_item, ident_item,
                         main_item, focus_item,
                         fpn_item]
                    )
                    ind += 1
                    self.ui.table_results.scrollToBottom()
                images_left = maxi - ind
                time_per_image = int((time.time() - start_time) / ind)
                eta = images_left * time_per_image
                h = eta // 3600
                m = eta % 3600 // 60
                s = eta % 3600 % 60
                msg = f"Analysed batch {cur_batch: 02d}/{maxi // step: 02d} in {time.time() - s2: 09.3f} secs\t\t"\
                      f"Total: {time.time() - start_time: 09.3f} secs\t\t"\
                      f"ETA: {h:02d}h:{m:02d}m:{s:02d}s"
                print(msg)
                self.write_to_log(msg)
                curind = b
                cur_batch += 1
                self.detector.save_log_messages(Paths.log_path, True)
            self.enable_buttons()
            self.ui.list_images.setEnabled(True)
            settings["analysis_settings"]["logging"] = logstate
            self.prg_signal.emit("Analysis finished -- Program ready",
                                 100,
                                 100, "")
            # Change the status of list items to reflect that they were analysed
            for ind in range(self.img_list_model.rowCount()):
                item = self.img_list_model.get_item_at_index(ind)
                data = item.data()
                data["analysed"] = True
                item.setData(data)
            self.selec_signal.emit(True)
            self.check_all_item_statuses()
        msg = f"Total analysis time: {time.time() - start_time:.3f} secs"
        print(msg)
        self.write_to_log(msg)

    @staticmethod
    def write_to_log(msg: str) -> None:
        """
        Method to write something to the log file

        :param msg: The message to write to the log file
        :return: None
        """
        with open(Paths.log_path, "a+") as lf:
            lf.write("#" * 20 + "\n")
            lf.write(datetime.today().strftime("%Y-%m-%d") + "\n")
            lf.write(datetime.today().strftime("%H:%M:%S") + "\n")
            lf.write(msg + "\n")
            lf.write("#" * 20 + "\n")

    @staticmethod
    def save_rois_to_database(data: Dict[str, Union[str, ROIHandler, np.ndarray, Dict[str, str]]],
                              all_: bool = False) -> None:
        """
        Method to save the data stored in the ROIHandler rois to the database

        :param data: The data dict returned by the Detector class
        :param all_: Deactivates printing to console
        :return: None
        """
        key = data["id"]
        # Establish new connector
        req = Requester()
        ins = Inserter()
        # Get info for image and check if image was analysed already
        if req.get_info_for_image(key)[8]:
            # Delete saved data
            ins.delete_existing_image_data(key)
        # Check if image should be added to experiment
        if data["add to experiment"]:
            exp_data = data["experiment details"]
            ins.add_image_to_experiment(key, exp_data["name"], exp_data["details"],
                                        exp_data["notes"], "Standard")
        # Update channel info
        for ind in range(len(data["names"])):
            ins.add_channel(key, ind, data["names"][ind],
                            data["active channels"][ind], data["main channel"] == ind)
        # Save scale and scale unit
        ins.set_image_scale(key, data["x_scale"], data["y_scale"])
        ins.set_image_scale_unit(key, data["scale_unit"])
        # Save data for detected ROI
        roidat, pdat, elldat = NucDetect.prepare_roihandler_for_database(data["handler"], data["channels"])
        # Check if there is any data to save
        if roidat:
            # Save data to database
            ins.save_roi_data_for_image(key, roidat, pdat, elldat)
        ins.commit_and_close()
        req.commit_and_close()
        if not all_:
            print("ROI saved to database")

    @staticmethod
    def prepare_roihandler_for_database(handler: ROIHandler, channels: List[np.ndarray]) -> Tuple[List, List, List]:
        """
        Function to get the necessary data to save the given ROI to the database

        :param handler: The roi handler holding the ROI
        :param channels: List of the channels the roi are derived from
        :return: General ROI data, ROI area data
        """
        roidat = []
        pdat = []
        elldat = []
        # Collect data
        for roi in handler.rois:
            dim = roi.calculate_dimensions()
            ellp = roi.calculate_ellipse_parameters()
            # Get the channel of the roi
            stats = roi.calculate_statistics(channels[handler.idents.index(roi.ident)])
            asso = hash(roi.associated) if roi.associated else None
            roidat.append((hash(roi), handler.ident, True, roi.ident,
                           str(dim["center_x"]), str(dim["center_y"]),
                           dim["width"], dim["height"], asso, roi.detection_method, roi.match, roi.colocalized))
            for p in roi.area:
                pdat.append((hash(roi), p[0], p[1], p[2]))
            elldat.append(
                (hash(roi), handler.ident, stats["area"], stats["intensity average"], stats["intensity median"],
                 stats["intensity maximum"], stats["intensity minimum"], stats["intensity std"],
                 ellp["eccentricity"], ellp["roundness"], ellp["center_x"], ellp["center_y"],
                 ellp["major_axis"], ellp["minor_axis"], ellp["angle"], ellp["area"],
                 ellp['orientation_y'], ellp['orientation_x'], ellp["shape_match"])
            )
        return roidat, pdat, elldat

    def load_rois_from_database(self, md5: str) -> ROIHandler:
        """
        Method to load all rois associated with this image

        :param md5: The md5 hash of the image
        :return: A ROIHandler containing all roi
        """
        self.prg_signal.emit(f"Loading data",
                             0, 100, "")
        # Get requester
        rois = ROIHandler(ident=md5)
        entries = self.requester.get_associated_roi(md5)
        names = self.requester.get_channels(md5)
        for name in names:
            rois.idents.insert(name[1], name[2])
        processed_roi = self.process_roi_database_entries(entries)
        rois.add_rois(processed_roi)
        print(f"Loaded {len(rois)} roi of image {self.cur_img['file_name']} from database")
        return rois

    def process_roi_database_entries(self, entries: List[Tuple], ) -> List[ROI]:
        """
        Method to process the stored ROI
        :param entries: The entries to convert
        :return: List of created ROI objects
        """
        main_ = []
        sec = []
        statkeys = ("area", "intensity average", "intensity median", "intensity maximum",
                    "intensity minimum", "intensity std", "eccentricity", "roundness")
        ellkeys = ("center_x", "center_y", "major_axis", "minor_axis", "angle", "orientation_x",
                   "orientation_y", "area", "shape_match")
        ind = 1
        max_ = len(entries)
        roi = []
        for entry in entries:
            self.prg_signal.emit(f"Loading ROI:  {ind}/{max_}",
                                 ind, max_, "")
            temproi = ROI(channel=entry[3], main=entry[8] is None,
                          auto=bool(entry[2]), associated=entry[8], method=entry[9], match=entry[10])
            stats = self.requester.get_statistics_for_roi(entry[0])
            temproi.stats = dict(zip(statkeys, stats[2:10]))
            if temproi.main:
                main_.append(temproi)
            else:
                sec.append(temproi)
            rle = []
            for p in self.requester.get_points_for_roi(entry[0]):
                rle.append((p[1], p[2], p[3]))
            temproi.set_area(rle)
            ellp = self.extract_statistics_for_roi(stats, temproi.main)
            temproi.ell_params = dict(zip(ellkeys, ellp))
            temproi.id = entry[0]
            ind += 1
            roi.append(temproi)
        for m in main_:
            for s in sec:
                if s.associated == hash(m):
                    s.associated = m
        return roi

    @staticmethod
    def extract_statistics_for_roi(statistics: Tuple, is_main: bool = False) -> Tuple:
        """
        Method to extract the statistics from database results

        :param statistics: The statistics to extract
        :param is_main: Is the roi a main roi?
        :return: The extracted statistics
        """
        if not is_main:
            return None, None, None, None, None, None, None, None, None
        center_x = statistics[10]
        center_y = statistics[11]
        major = statistics[12]
        minor = statistics[13]
        angle = statistics[14]
        area = statistics[15]
        ov_x = statistics[16]
        ov_y = statistics[17]
        ellip = statistics[18]
        return center_x, center_y, major, minor, angle, area, ov_x, ov_y, ellip

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
        # Get all available channels for the image
        chans = Requester().get_channels(handler.ident)
        # Remove main channel and  from list
        chans = [x[2] for x in chans if not bool(x[4]) and bool(x[3])]
        # Sort channel list
        chans = sorted(chans)
        # Create header
        header = copy(NucDetect.STANDARD_TABLE_HEADER)
        header.extend(chans)
        if experiment:
            header.insert(2, "Group")
        rows = self.prepare_main_table_rows(experiment)
        self.create_table_rows(rows)
        if rows:
            self.res_table_model.setColumnCount(len(rows[0]))
        # Set header of table
        self.res_table_model.setHorizontalHeaderLabels(header)
        header = [header]
        header.extend(rows)
        self.data = header

    def prepare_main_table_rows(self, experiment: Union[str, None] = None) -> List[List[str]]:
        """
        Method to prepare the rows of the result table on the main UI

        :param experiment: Name of the experiment to show. None if only the current image should be shown
        :return: The prepared rows
        """
        if experiment:
            # Get all assigned images
            num_imgs = self.requester.get_number_of_associated_images_for_experiment(experiment)
            # Load data for experiment
            rows = self.get_table_data_from_database(experiment)
            # Sort rows according to group
            rows = sorted(rows, key=lambda x: x[1])
            self.set_experiment_status_label_text(
                f"Experiment: {experiment}\nImages: {num_imgs}"
            )
            self.cur_exp = experiment
        else:
            rows = self.get_table_data_for_image(self.cur_img["key"])
            self.set_experiment_status_label_text(
                f"Experiment: None\nImages: 1"
            )
        return rows

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

    def get_table_data_from_database(self, experiment: str) -> List[List[str]]:
        """
        Method to load the data of an experiment from the database

        :param experiment: The name of the experiment to get the data for
        :return: List of row to created for display
        """
        # Get images associated with experiment
        imgs = self.requester.get_associated_images_for_experiment(experiment)
        rows: List[List[str]] = []
        # Iterate over all images
        for img in imgs:
            # Check if the image is already analysed
            if not self.requester.check_if_image_was_analysed(img):
                continue
            row = self.get_table_data_for_image(img)
            # Check if the image was assigned to a group
            group = self.requester.get_associated_group_for_image(img, experiment)
            for row_ in row:
                row_.insert(2, group)
            rows.extend(row)
        return rows

    def get_table_data_for_image(self, img: str) -> List[List[str]]:
        """
        Method to get the table data for the specified image

        :param img: The md5 hash of the image to get the data for
        :return: List of rows created for display
        """
        # Convert key to file name
        name = self.requester.get_image_filename(img)
        self.prg_signal.emit(f"Creating result table for image {name}", 0, 100, "")
        rows = self.requester.get_table_data_for_image(img, name)
        self.prg_signal.emit(f"Creating result table for image {name}", 100, 100, "")
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
        self.ui.btn_statistics.setEnabled(state)
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
        self.ui.prg_bar.setMaximum(int(maxi))
        self.ui.prg_bar.setValue(int(progress))

    def save_results(self) -> None:
        """
        Method to export the analysis results as csv file

        :return: None
        """
        cur = self.cur_img if self.cur_img else {}
        dial = DataExportDialog(cur.get("key", None),
                                cur.get("file_name", None))
        code = dial.exec()
        if code == QDialog.Accepted:
            self.check_timer.setInterval(500)
            start_time = time.time()
            self.check_timer.timeout.connect(
                partial(self.check_for_running_threads,
                        dial.threads,
                        "Background tasks executing, please wait...",
                        start_time)
            )

    def check_for_running_threads(self,
                                  threads: List[threading.Thread],
                                  display_msg: str = "",
                                  starting_time: int = None) -> None:
        """
        Function to keep the program locked until all given threads are finished

        :param threads: List of threads to check
        :param display_msg: The message to display in the progress bar
        :param starting_time: The time this function was called
        :return:None
        """
        time_string = ""
        if starting_time:
            current_runtime = time.time() - starting_time
            time_string = f"Runtime: {current_runtime/1000: .2f} sec"
        msg = display_msg + "Current " + time_string
        if [x for x in threads if x.is_alive()]:
            self.enable_buttons(False)
            print(msg)
            self.prg_signal.emit(msg,
                                 0, 100, "")
        else:
            self.enable_buttons(True)
            self.prg_signal.emit(f"Background tasks finished {time_string}",
                                 0, 100, "")

    def show_statistics(self) -> None:
        """
        Method to open a dialog showing various statistics

        :return: None
        """
        # Check if experiments were defined
        exps = self.requester.get_all_experiments()
        if not exps:
            msg = QMessageBox()
            msg.setWindowIcon(Icon.get_icon("LOGO"))
            msg.setIcon(QMessageBox.Information)
            msg.setStyleSheet(open(os.path.join(Paths.css_dir, "messagebox.css"), "r").read())
            msg.setWindowTitle("Warning")
            msg.setText("No experiments were defined")
            msg.setInformativeText("Statistics can only be displayed, if images are assigned to an experiment")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
            return
        # Open dialog to select an experiment
        exp_sel_dial = ExperimentSelectionDialog()
        code = exp_sel_dial.exec()
        if code == QDialog.Accepted:
            exp = exp_sel_dial.sel_exp
            active_channels = exp_sel_dial.active_channels
            stat_dialog = StatisticsDialog(experiment=exp,
                                           active_channels=active_channels)
            stat_dialog.exec()

    def show_settings(self) -> None:
        """
        Method to open the settings dialog

        :return: None
        """
        sett = SettingsDialog(self.inserter)
        sett.initialize_from_file(os.path.join(Paths.settings_path, "settings.json"))
        code = sett.exec()
        if code == QDialog.Accepted:
            if sett.changed:
                for key, value in sett.changed.items():
                    self.settings[key] = value[0]
                    self.inserter.update_setting(key, value[0])
            sett.save_menu_settings()
            self.inserter.commit()
        self.check_all_item_statuses()
        # TODO check
        self.settings = self.load_settings()

    def show_modification_window(self) -> None:
        """
        Method to open the modification dialog, allowing the user to modify automatically determined results

        :return: None
        """
        # Load channels for image from database
        channels = [(x[1], x[2]) for x in self.requester.get_channels(self.cur_img["key"])]
        editor = Editor(image=ImageLoader.load_image(self.cur_img["path"]),
                        active_channels=channels,
                        roi=self.roi_cache, size_factor=self.settings["size_factor"],
                        img_name=self.cur_img['file_name'],
                        x_scale=self.cur_img["x_scale"], y_scale=self.cur_img["y_scale"])
        editor.setWindowFlags(editor.windowFlags() |
                              QtCore.Qt.WindowSystemMenuHint |
                              QtCore.Qt.WindowMinMaxButtonsHint |
                              QtCore.Qt.Window)
        code = editor.exec()
        if code == QDialog.Accepted:
            self.create_result_table_from_list(self.roi_cache)
            self.check_all_item_statuses()

    def reflect_item_status_changes(self) -> None:
        """
        Method to change the image list items if the underlying image was analysed

        :return: None
        """
        # Check if image was modified
        analysed, modified = Util.check_if_image_was_analysed_and_modified(self.cur_img["key"])
        self.cur_img["analysed"] = analysed
        self.cur_img["modified"] = modified
        item = None
        # Save the data changes to the items data
        for index in self.ui.list_images.selectionModel().selectedIndexes():
            item = self.img_list_model.get_item_at_index((index.row()))
            item.setData(self.cur_img)
        if analysed and item:
            if modified:
                item.setBackground(Color.ITEM_MODIFIED)
            else:
                item.setBackground(Color.ITEM_ANALYSED)

    def check_all_item_statuses(self) -> None:
        """
        Method to change the image list items if the underlying image was analysed

        :return: None
        """
        model = self.ui.list_images.model()
        for index in range(model.rowCount()):
            item = model.get_item_at_index(index)
            data = item.data()
            analysed, modified = Util.check_if_image_was_analysed_and_modified(data["key"])
            data["analysed"] = analysed
            data["modified"] = modified
            item.setData(data)
            if analysed:
                if modified:
                    item.setBackground(Color.ITEM_MODIFIED)
                else:
                    item.setBackground(Color.ITEM_ANALYSED)
            else:
                item.setBackground(Color.STANDARD)

    def on_close(self) -> None:
        """
        Will be called if the program window closes

        :return:
        """
        self.connector.close_connection()


class TableFilterModel(QSortFilterProxyModel):
    """
    Model used to enable tuple sorting
    """

    def __init__(self, parent):
        super(TableFilterModel, self).__init__(parent)

    def lessThan(self, ind1, ind2):
        ldat = self.sourceModel().itemData(ind1)[0]
        rdat = self.sourceModel().itemData(ind2)[0]
        # Check if text can be converted to digit
        if ldat.isdigit():
            return int(ldat) < int(rdat)
        if isinstance(ldat, tuple):
            if ldat[1] == rdat[1]:
                return ldat[0] < rdat[0]
            return ldat[1] < rdat[1]
        return ldat < rdat


class ImageListModel(QAbstractListModel):
    """
    Class to lazy load needed image list items
    """
    __slots__ = (
        "current_index",
        "page_size",
        "_paths",
        "_current_paths",
        "_cache",
    )

    def __init__(self, parent=None, paths: List[str] = (), page_size: int = 30):
        """
        :param paths: The image paths that are the basis of the items
        """
        super().__init__(parent)
        self.set_paths(paths)
        self.page_size = min(page_size, len(paths))
        self.current_index = 0
        self._cache = {}

    def set_paths(self, paths: List[str]):
        self.modelReset.emit()
        self._paths = paths
        self._current_paths = paths
        self.current_index = 0
        self._cache = {}

    def filter_paths(self, keyword: str) -> None:
        """
        Method to filter the paths list via keyword search

        :param keyword: The keyword to search for
        :return: None
        """
        # Reset current index
        self.current_index = 0
        # Clear the current model
        self.clear_data()
        # Filter paths
        self._current_paths = [
            x for x in self._current_paths if keyword in os.path.splitext(x)[0].split(os.sep)[:-1]
        ]
        # Fetch new items
        # TODO
        self.fetchMore()

    def clear_data(self) -> None:
        """
        Method to clear the stored data

        :return: None
        """
        self.beginResetModel()
        self.removeRows(0, self.rowCount())
        self.endResetModel()

    def canFetchMore(self, parent: QModelIndex) -> bool:
        if parent.isValid():
            return False
        return self.current_index * self.page_size < len(self._current_paths)

    def fetchMore(self, parent: QModelIndex) -> None:
        if parent.isValid():
            return
        # Get the number of items to fetch
        remainder = len(self._current_paths) - self.current_index
        items_to_fetch = min(remainder, self.page_size)
        if items_to_fetch == 0:
            return
        self.beginInsertRows(QModelIndex(), self.current_index, self.current_index + items_to_fetch - 1)
        self.current_index += items_to_fetch
        self.endInsertRows()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self.current_index

    def data(self, index: QModelIndex, role: int = ...) -> Any:
        # Check if the index is valid
        if not index.isValid():
            return 0
        # Check if the row is inside teh available boundaries
        row = index.row()
        if row > len(self._paths) or row < 0:
            return 0
        return self.get_item_at_index(row).data(role)

    def setData(self, index: QModelIndex, value, role=Qt.EditRole):
        if role == Qt.EditRole and index.isValid():
            item = self._cache[index.row()]
            item.setData(value)

    def get_item_at_index(self, index: int) -> QStandardItem:
        """
        Method to get the item at the specified index

        :param index: The index of the item
        :return: The item
        """
        if index not in self._cache:
            self._cache[index] = Util.create_list_item(self._paths[index])
        return self._cache[index]


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
    msg.setWindowIcon(Icon.get_icon("LOGO"))
    msg.setStyleSheet(open(os.path.join(Paths.css_dir, "messagebox.css"), "r").read())
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
        pixmap = QPixmap(os.path.join(Paths.logo_dir, "banner_norm.png"))
        splash = QSplashScreen(pixmap)
        splash.show()
        splash.showMessage("Checking for thumbnails...")
        print("Check files for thumbnails...")
        # Count number of available images
        total = 0
        for root, dirs, files in os.walk(Paths.images_path):
            total += len(files)
        file_index = 1
        for root, dirs, files in os.walk(Paths.images_path):
            for file in files:
                msg = f"{file_index: 04d}:{total: 04d} checked..."
                os.system('cls' if os.name == 'nt' else 'clear')
                print(msg)
                splash.showMessage(msg)
                Util.create_thumbnail(os.path.join(root, file))
                file_index += 1
        os.system('cls' if os.name == 'nt' else 'clear')
        print("All files checked for thumbnails, starting...")
        main_win = NucDetect()
        splash.finish(main_win)
        main_win.show()
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()
