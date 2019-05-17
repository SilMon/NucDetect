import os
import sqlite3
import sys
import time
import json
from threading import Thread

import PyQt5
import numpy as np
import piexif
import qtawesome as qta
import matplotlib.pyplot as plt
import copy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtCore, QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import QSize, Qt, pyqtSignal, pyqtProperty, QRectF, QItemSelectionModel, QModelIndex
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIcon, QPixmap, QColor, QTransform, QPainter, QBrush, QPen, \
    QImage
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QHeaderView, QDialog, QSplashScreen, QSizePolicy, QWidget, \
    QVBoxLayout, QSpacerItem, QScrollArea, QMessageBox, QGraphicsScene, QGraphicsEllipseItem, QGraphicsItemGroup, \
    QGraphicsView, QGraphicsItem, QGraphicsPixmapItem
from qtconsole.qt import QtGui
from skimage import img_as_ubyte
from skimage.draw import ellipse, circle

from Nucleus.core.Detector import Detector
from Nucleus.gui.settings.Settings import SettingsShowWidget, SettingsSlider, SettingsText, SettingsComboBox, \
    SettingsCheckBox, SettingsDial, SettingsSpinner, SettingsDecimalSpinner
from Nucleus.core.Detector import Detector
from Nucleus.core.ROI import ROI
from Nucleus.core.ROIHandler import ROIHandler

PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, False)
PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, False)

ui_main = os.path.join(os.getcwd(), "nucdetect.ui")
ui_result_image_dialog = os.path.join(os.getcwd(), "result_image_dialog.ui")
ui_class_dial = os.path.join(os.getcwd(), "classification_dialog.ui")
ui_stat_dial = os.path.join(os.getcwd(), "statistics_dialog.ui")
ui_settings_dial = os.path.join(os.getcwd(), "settings_dialog.ui")
ui_modification_dial = os.path.join(os.getcwd(), "modification_dialog.ui")
database = os.path.join(os.pardir, "database{}nucdetect.db".format(os.sep))
result_path = os.path.join(os.pardir, "results")


class NucDetect(QMainWindow):
    """
    Created on 11.02.2019
    @author: Romano Weiss
    """
    prg_signal = pyqtSignal(str, int, int, str)
    selec_signal = pyqtSignal()
    aa_signal = pyqtSignal(int, int)
    executor = Thread()

    def __init__(self):
        QMainWindow.__init__(self)
        self.connection = sqlite3.connect(database)
        self.cursor = self.connection.cursor()
        self.settings = self.load_settings()
        self.detector = Detector(settings=None)
        self.reg_images = {}
        self.sel_images = []
        self.cur_img = None
        self.roi_cache = None
        self.unsaved_changes = False
        self._setup_ui()
        self.setWindowTitle("NucDetect")
        self.setWindowIcon(QtGui.QIcon('logo.png'))

    def load_settings(self):
        self.cursor.execute(
            "SELECT key_, value FROM settings"
        )
        return dict(self.cursor.fetchall())

    def closeEvent(self, event):
        self.on_close()
        event.accept()

    def _setup_ui(self):
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

    def on_image_selection_change(self):
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
                self.ui.btn_analyse.setEnabled(False)
                self.ui.btn_statistics.setEnabled(True)
                self.ui.btn_images.setEnabled(True)
                self.ui.btn_save.setEnabled(True)
                self.ui.btn_modify.setEnabled(True)
                self.ui.btn_categories.setEnabled(True)
                self.cur_img = self.sel_images[0]
                self.ui.lbl_status.setText("Loaded analysis results from database")
            else:
                self.ui.lbl_status.setText("Program ready")
                self.res_table_model.setRowCount(0)
                self.ui.btn_analyse.setEnabled(True)
                self.ui.btn_statistics.setEnabled(False)
                self.ui.btn_images.setEnabled(False)
                self.ui.btn_save.setEnabled(False)
                self.ui.btn_modify.setEnabled(False)
                self.ui.btn_categories.setEnabled(False)
        else:
            self.ui.btn_analyse.setEnabled(False)

    def _show_loading_dialog(self):
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
            self.add_image_to_list(file_name)

    def add_image_to_list(self, path):
        """
        Method to add an image to the list of loaded files. The image will be processed, added and loaded.
        :param path: The path leading to the file
        :return: None
        """
        temp = os.path.split(path)
        folder = temp[0].split(sep=os.sep)[-1]
        file = temp[1]
        if os.path.splitext(file)[1] in Detector.FORMATS:
            d = Detector.get_image_data(path)
            t = d["datetime"].decode("ascii").split(" ")
            item = QStandardItem()
            item_text = "Name: {}\nFolder: {}\nDate: {}\nTime: {}".format(file, folder, t[0], t[1])
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
                     str(d["x_res"]), str(d["y_res"]), d["unit"], 0, -1)  # TODO settngs hashen
                )
            self.connection.commit()

    def add_images_from_folder(self, url):
        """
        Method to load a whole folder of images
        :param url: The path of the folder
        :return: None
        """
        for t in os.walk(url):
            for file in t[2]:
                self.add_image_to_list(os.path.join(t[0], file))

    def remove_image_from_list(self):
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

    def clear_image_list(self):
        """
        Method to clear the list of loaded images
        :return: None
        """
        self.img_list_model.clear()
        self.reg_images.clear()

    def analyze(self):
        """
        Method to analyze an loaded image
        :return: None
        """
        self.res_table_model.setRowCount(0)
        if not self.sel_images:
            self.ui.list_images.select(self.img_list_model.index(0, 0))
        self.prg_signal.emit("Analysing " + str(self.sel_images[0]),
                             0, 100, "")
        self.cur_img = self.sel_images[0]
        self.sel_images.remove(self.sel_images[0])
        thread = Thread(target=self.analyze_image,
                        args=(self.cur_img,
                              "Analysis finished in {} -- Program ready",
                              100, 100,))
        thread.start()

    def analyze_image(self, path, message, percent, maxi, all_=False):
        if not all_:
            self.ui.list_images.setEnabled(False)
            self.ui.btn_analyse.setEnabled(False)
            self.ui.btn_analyse_all.setEnabled(False)
            self.ui.btn_clear_list.setEnabled(False)
            self.ui.btn_delete_from_list.setEnabled(False)
        start = time.time()
        self.prg_signal.emit("Connecting to database", 0 if not all_ else percent, maxi, "")
        con = sqlite3.connect(database)
        curs = con.cursor()
        self.unsaved_changes = True
        self.prg_signal.emit("Analysing image", maxi*0.05 if not all_ else percent, maxi, "")
        data = self.detector.analyse_image(path)
        key = data["id"]
        self.roi_cache = data["handler"]
        s0 = time.time()
        self.prg_signal.emit("Creating result table", maxi * 0.65 if not all_ else percent, maxi, "")
        self.create_result_table_from_list(data["handler"])
        print("Creation result table: {:.4f}".format(time.time()-s0))
        self.prg_signal.emit("Checking database", maxi * 0.75 if not all_ else percent, maxi, "")
        s1 = time.time()
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
        self.prg_signal.emit("Analysing nuclei", maxi * 0.85 if not all_ else percent, maxi, "")
        for name in data["handler"].idents:
            curs.execute(
                "INSERT INTO channels VALUES (?, ?, ?)",
                (key, data["handler"].idents.index(name), name)
            )
        for roi in data["handler"].rois:
            dim = roi.calculate_dimensions()
            asso = hash(roi.associated) if roi.associated is not None else None
            curs.execute(
                "INSERT INTO roi VALUES (?, ?, ?, ?, ?, ?, ?,?)",
                (hash(roi), key, True, roi.ident, str(dim["center"]), dim["width"], dim["height"], asso)
            )
            for p in roi.points:
                curs.execute(
                    "INSERT INTO points VALUES (?, ?, ?, ?)",
                    (hash(roi), p[0], p[1], roi.inten[p])
                )
        self.prg_signal.emit("Writing to database", maxi * 0.95 if not all_ else percent, maxi, "")
        curs.execute(
            "UPDATE images SET analysed = ? WHERE md5 = ?",
            (True, key)
        )
        con.commit()
        con.close()
        print("Writing to database: {:.4f}".format(time.time() - s1))
        self.prg_signal.emit(message.format("{:.2f} secs".format(time.time()-start)), percent, maxi, "")
        self.ui.btn_save.setEnabled(True)
        self.ui.btn_images.setEnabled(True)
        self.ui.btn_statistics.setEnabled(True)
        self.ui.btn_categories.setEnabled(True)
        self.ui.btn_modify.setEnabled(True)
        if not all_:
            self.ui.list_images.setEnabled(True)
            self.ui.btn_analyse.setEnabled(True)
            self.ui.btn_analyse_all.setEnabled(True)
            self.ui.btn_clear_list.setEnabled(True)
            self.ui.btn_delete_from_list.setEnabled(True)
        # TODO Multiselection implementieren
        ''' 
        if len(self.sel_images) is not 0:
            self.analyze()
            '''

    def create_result_table_from_list(self, handler):
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

    def _select_next_image(self):
        max_ind = self.img_list_model.rowCount()
        cur_ind = self.ui.list_images.currentIndex()
        if cur_ind.row() < max_ind:
            nex = self.img_list_model.index(cur_ind.row() + 1, 0)
            self.ui.list_images.selectionModel().select(nex, QItemSelectionModel.Select)
            self.ui.list_images.setCurrentIndex(nex)
        else:
            first = self.img_list_model.index(0, 0)
            self.ui.list_images.selectionModel().select(first, QItemSelectionModel.Select)
            self.ui.list_images.setCurrentIndex(first)

    def _set_progress(self, text, progress, maxi, symbol):
        self.ui.lbl_status.setText("{} -- {:.2f}% {}".format(text, progress, symbol))
        self.ui.prg_bar.setValue((progress/maxi)*100)

    def analyze_all(self):
        self.ui.list_images.setEnabled(False)
        self.ui.btn_analyse.setEnabled(False)
        self.ui.btn_analyse_all.setEnabled(False)
        self.ui.btn_clear_list.setEnabled(False)
        self.ui.btn_delete_from_list.setEnabled(False)
        self.unsaved_changes = True
        self.selec_signal.emit()
        thread = Thread(target=self._analyze_all, args=(
            0, len(self.reg_images)-1))
        thread.start()

    def _analyze_all(self, percent=0, maxi=0):
        self.ui.list_images.setEnabled(False)
        self.ui.btn_analyse.setEnabled(False)
        self.ui.btn_analyse_all.setEnabled(False)
        self.ui.btn_clear_list.setEnabled(False)
        self.ui.btn_delete_from_list.setEnabled(False)
        self.analyze_image(self.img_keys[self.sel_images[0]],
                           message="Analysing " + self.sel_images[0],
                           percent=percent, maxi=maxi, all_=True)
        if percent < maxi:
            self.selec_signal.emit()
            self._analyze_all(percent=percent + 1, maxi=maxi)
        if percent == maxi:
            self.ui.list_images.setEnabled(True)
            self.ui.btn_analyse.setEnabled(True)
            self.ui.btn_analyse_all.setEnabled(True)
            self.ui.btn_clear_list.setEnabled(True)
            self.ui.btn_delete_from_list.setEnabled(True)
            self.prg_signal.emit("Analysis finished -- Program ready",
                                maxi,
                                maxi, "")
            self.selec_signal.emit()
            
    def load_rois_from_database(self, md5):
        """
        Method to load all rois associated with this image
        :param md5: The md5 hash of the image
        :return: A ROIHandler containing all roi
        """
        # TODO
        rois = ROIHandler(ident=md5)
        entries = self.cursor.execute(
            "SELECT * FROM roi WHERE image = ?",
            (md5, )
        ).fetchall()
        names = self.cursor.execute(
            "SELECT * FROM channels WHERE md5 = ?",
            (md5, )
        ).fetchall()
        for name in names:
            rois.idents.insert(name[1], name[2])
        main = []
        sec = []
        for entry in entries:
            temproi = ROI(channel=entry[3], main=entry[7] is None, associated=entry[7])
            if temproi.main:
                main.append(temproi)
            else:
                sec.append(temproi)
            for p in self.cursor.execute(
                "SELECT * FROM points WHERE hash = ?",
                    (entry[0], )
            ).fetchall():
                temproi.add_point((p[1], p[2]), p[3])
            rois.add_roi(temproi)
        for m in main:
            for s in sec:
                if s.associated == hash(m):
                    s.associated = m
        return rois

    def show_result_image(self):
        # TODO Zoom implementieren: http://doc.qt.io/qt-5/qtwidgets-widgets-imageviewer-example.html
        image_dialog = ImgDialog(image=Detector.load_image(self.cur_img), handler=self.roi_cache)
        image_dialog.setWindowTitle("Result Images for " + self.cur_img)
        image_dialog.setWindowIcon(QtGui.QIcon('logo.png'))
        image_dialog.setWindowFlags(image_dialog.windowFlags() |
                                    QtCore.Qt.WindowSystemMenuHint |
                                    QtCore.Qt.WindowMinMaxButtonsHint|
                                    QtCore.Qt.Window)

        image_dialog.exec_()

    def save_results(self):
        save = Thread(target=self._save_results)
        self.prg_signal.emit("Saving Results", 0, 100, "")
        save.start()

    def _save_results(self):
        self.roi_cache.export_data_as_csv(path=result_path)
        self.prg_signal.emit("Saving Results", 50, 100, "")
        self.prg_signal.emit("Results saved -- Program ready", 100, 100, "")
        self.unsaved_changes = False

    def on_config_change(self, config, section, key, value):
        if section == "Analysis":
            self.detector.settings["key"] = value

    def show_statistics(self):
        stat_dialog = QDialog()
        stat_dialog.ui = uic.loadUi(ui_stat_dial, stat_dialog)
        stat_dialog.setWindowTitle("Statistics")
        stat_dialog.setWindowIcon(QtGui.QIcon('logo.png'))
        # Add statistics to list
        key = self.cur_img["key"]
        stat = self.detector.get_statistics(key)
        # TODO Create stat array to fill charts
        stat_dialog.ui.lbl_num_nuc.setText(
            "Detected nuclei: {:>}".format(stat["number"]))
        stat_dialog.ui.lbl_num_nuc_empt.setText(
            "Thereof empty: {:>}".format(stat["empty"]))
        stat_dialog.ui.lbl_num_red.setText(
            "Detected red foci: {:>}".format(np.sum(stat["number red"])))
        stat_dialog.ui.lbl_num_green.setText(
            "Detected green foci: {:>}".format(np.sum(stat["number green"])))
        stat_dialog.ui.lbl_num_red_std.setText(
            "Std. dev. red foci number: {:>}".format(str(round(stat["number red std"], 2))))
        stat_dialog.ui.lbl_num_green_std.setText(
            "Std. dev. green foci number: {:.2f}".format(stat["number green std"]))
        stat_dialog.ui.lbl_int_red.setText(
            "Average red intensity: {:>10.2f}".format(stat["intensity red average"]))
        stat_dialog.ui.lbl_std_red.setText(
            "Std. dev. red: {:>10.2f}".format(stat["intensity red std"]))
        stat_dialog.ui.lbl_int_green.setText(
            "Average green intensity: {:>10.2f}".format(stat["intensity green average"]))
        stat_dialog.ui.lbl_std_green.setText(
            "Std. dev. green: {:>.2f}".format(stat["intensity green std"]))
        stat_dialog.ui.lbl_num_red_max.setText(
            "Max. Red Number: {:>}".format(max(stat["number red"])))
        stat_dialog.ui.lbl_num_red_min.setText(
            "Min. Red Number: {:>}".format(min([x for x in stat["number red"] if x > 0])))
        stat_dialog.ui.lbl_num_green_max.setText(
            "Max. Green Number: {:>}".format(max(stat["number green"])))
        stat_dialog.ui.lbl_num_green_min.setText(
            "Min. Green Number: {:>}".format(min([x for x in stat["number green"] if x > 0])))
        stat_dialog.ui.lbl_int_red_max.setText(
            "Max. Red Intensity: {:>.2f}".format(max(stat["intensity red"])))
        stat_dialog.ui.lbl_int_red_min.setText(
            "Min. Red Intensity: {:>.2f}".format(min([x for x in stat["intensity red"] if x > 0])))
        stat_dialog.ui.lbl_int_green_max.setText(
            "Max. Green Intensity: {:>.2f}".format(max(stat["intensity green"])))
        stat_dialog.ui.lbl_int_green_min.setText(
            "Min. Green Intensity: {:>.2f}".format(min([x for x in stat["intensity green"] if x > 0])))
        cnvs_red_poisson = PoissonCanvas(stat["number red average"],
                                         max(stat["number red"])+1,
                                         stat["number red"], name="red channel poisson - {}".format(key),
                                         title="Red Channel",)
        cnvs_green_poisson = PoissonCanvas(stat["number red average"],
                                           max(stat["number green"]) + 1,
                                           stat["number green"], name="green channel poisson - {}".format(key),
                                           title="Green Channel")
        cnvs_red_int = BarChart(name="red channel int - {}".format(key), title="Red Channel - Average Focus Intensity",
                                y_title="Average Intensity", x_title="Nucleus Index", x_label_rotation=45,
                                values=[stat["intensity red"]], labels=[np.arange(len(stat["intensity red"]))])
        cnvs_red_int.setToolTip(("Shows the average red foci intensity for the nucleus with the given index.\n"
                                 "255 is the maximal possible value. If no intensity is shown, no red foci were\n"
                                 "detected in the respective nucleus"))
        cnvs_green_int = BarChart(name="green channel int - {}".format(key),
                                  title="Green Channel - Average Focus Intensity", y_title="Average Intensity",
                                  x_title="Nucleus Index", x_label_rotation=45, values=[stat["intensity red"]],
                                  labels=[np.arange(len(stat["intensity red"]))])
        cnvs_green_int.setToolTip(("Shows the average green foci intensity for the nucleus with the given index.\n" 
                                   "255 is the maximal possible value. If no intensity is shown, no green foci were\n"
                                   "detected in the respective nucleus"))
        x_val = []
        x_val.append(np.arange(len(stat["number red"])))
        x_val.append(np.arange(len(stat["number green"])))
        y_val = []
        y_val.append(stat["number red"])
        y_val.append(stat["number green"])
        cnvs_num = XYChart(x_values=x_val, y_values=y_val, col_marks=["ro", "go"],
                           dat_labels=["Green Channel", "Red Channel"], name="numbers - {}".format(key),
                           title="Foci Number", x_title="Nucleus Index", y_title="Foci")
        ind_red = 0
        x_values = []
        y_values = []
        colmarks = []
        for inten in stat["intensity red total"]:
            x_t = np.zeros(len(inten))
            x_t.fill(ind_red)
            x_values.append(x_t)
            y_values.append(inten)
            colmarks.append("ro")
            ind_red += 1
        ind_green = 0
        for inten in stat["intensity green total"]:
            x_t = np.zeros(len(inten))
            x_t.fill(ind_green)
            x_values.append(x_t)
            y_values.append(inten)
            colmarks.append("go")
            ind_green += 1
        cnvs_int = XYChart(x_values=x_values, y_values=y_values, col_marks=colmarks,
                           dat_labels=["Green Channel", "Red Channel"],
                           name="intensities - {}".format(key), title="Foci Intensities", x_title="Nucleus Index",
                           y_title="Average Intensity")
        stat_dialog.ui.vl_poisson.addWidget(cnvs_red_poisson)
        stat_dialog.ui.vl_poisson.addWidget(cnvs_green_poisson)
        stat_dialog.ui.vl_int.addWidget(cnvs_red_int)
        stat_dialog.ui.vl_int.addWidget(cnvs_green_int)
        stat_dialog.ui.vl_vals.addWidget(cnvs_num)
        stat_dialog.ui.vl_vals.addWidget(cnvs_int)
        stat_dialog.setWindowFlags(stat_dialog.windowFlags() |
                                   QtCore.Qt.WindowSystemMenuHint |
                                   QtCore.Qt.WindowMinMaxButtonsHint)
        code = stat_dialog.exec()
        if code == QDialog.Accepted:
            cnvs_red_poisson.save()
            cnvs_green_poisson.save()
            cnvs_red_int.save()
            cnvs_green_int.save()
            cnvs_num.save()
            cnvs_int.save()

    def show_categorization(self):
        cl_dialog = QDialog()
        cl_dialog.ui = uic.loadUi(ui_class_dial, cl_dialog)
        cl_dialog.setWindowTitle("Classification")
        cl_dialog.setWindowIcon(QtGui.QIcon('logo.png'))
        # categories = self.detector.get_categories(self.cur_img["key"])
        categories = self.cursor.execute(
            "SELECT category FROM categories WHERE image = ?",
            (self.cur_img["key"],)
        )
        cate = ""
        for cat in categories:
            cate += str(cat) + "\n"
        cl_dialog.ui.te_cat.setPlainText(cate)
        code = cl_dialog.exec()
        if code == QDialog.Accepted:
            self._categorize_image(cl_dialog.ui.te_cat.toPlainText())

    def _categorize_image(self, categories):
        if categories is not "":
            categories = categories.split('\n')
            self.detector.categorize_image(self.cur_img["key"], categories)
            self.cursor.execute(
                "DELETE FROM categories WHERE image = ?",
                (self.cur_img["key"],)
            )
            for cat in categories:
                self.cursor.execute(
                    "INSERT INTO categories VALUES(?, ?)",
                    (self.cur_img["key"], cat)
                )

    def show_settings(self):
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
                    self.cursor.execute(
                        "INSERT INTO settings VALUES(?, ?)",
                        (key, value)
                    )
            sett.save_menu_settings()

    def show_modification_window(self):
        mod = ModificationDialog(image=Detector.load_image(self.cur_img), handler=self.roi_cache)
        mod.setWindowTitle("Modification")
        mod.setWindowIcon(QtGui.QIcon("logo.png"))
        mod.setWindowFlags(mod.windowFlags() |
                           QtCore.Qt.WindowSystemMenuHint |
                           QtCore.Qt.WindowMinMaxButtonsHint |
                           QtCore.Qt.Window)
        mod.exec()

    def on_close(self):
        self.connection.close()


class MPLPlot(FigureCanvas):

    def __init__(self, name, width=4, height=4, dpi=65, parent=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.name = name
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)

    def save(self):
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

    def __init__(self, _lambda, k, values, title="", name="", parent=None, width=4, height=4, dpi=65):
        super(PoissonCanvas, self).__init__(name, width, height, dpi, parent)
        self.title = title
        self.plot(_lambda, k, values)

    def plot(self, _lambda, k, values):
        poisson = np.random.poisson(_lambda, k)
        ax = self.figure.add_subplot(111)
        objects = np.arange(k)
        x_pos = np.arange(len(poisson))
        conv_values = np.zeros(len(poisson))
        for val in values:
            conv_values[val] += 1
        ax.set_title("Poisson Distribution - " + self.title)
        ax.bar(x_pos, poisson, align="center", alpha=0.5)
        ax.bar(x_pos, conv_values, align="center", alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(objects, rotation=45)
        ax.set_ylabel("Probability [%]")
        ax.set_xlabel("Foci number [N]")
        self.draw()


class XYChart(MPLPlot):

    def __init__(self, x_values, y_values, dat_labels, col_marks=["ro"], parent=None, name="", title="", x_title="",
                 y_title="", width=4, height=4, dpi=65, x_label_max_num=20, y_label_max_num=20, x_label_rotation=0,
                 y_label_rotation=0):
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

    def plot(self):
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
            if len(self.colmarks) == len(self.x_values):
                ax.plot(self.x_values[ind], self.y_values[ind], self.colmarks[ind])
            else:
                ax.plot(self.x_values[ind], self.y_values[ind])
        if self.dat_label:
            ax.legend(self.dat_label)
        self.draw()


class BarChart(MPLPlot):

    def __init__(self, values, labels, colors=[], parent=None, overlay=True, name="",
                 title="", x_title="", y_title="", width=4, height=4, dpi=65,
                 x_label_rotation=0, y_label_rotation=0):
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

    def plot(self):
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
                ax.bar(x_pos, lst, width=bar_width, align="center", alpha=alph)
        if len(self.values) > 1:
            x_ticks_lst = [r + bar_width for r in range(len(self.values[0]))]
        else:
            x_ticks_lst = np.arange(len(self.values[0]))
        ax.set_xticks(x_ticks_lst)
        ax.set_xticklabels(x_ticks_lst, rotation=self.x_label_rotation)
        self.draw()


class ImgDialog(QDialog):
    # TODO Umbauen

    def __init__(self, image, handler, parent=None):
        super(ImgDialog, self).__init__(parent)
        self.image = image
        self.handler = handler
        self.ui = uic.loadUi(ui_result_image_dialog, self)
        self.view = QGraphicsView()
        self.initialize_ui()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        scene = QGraphicsScene(self)
        scene.setSceneRect(0, 0, self.view.width(), self.view.height())
        self.view.setScene(scene)
        # Initialization of the background image
        self.sc_bckg = self.view.scene().addPixmap(QPixmap())
        self.pmap = QPixmap()

    def initialize_ui(self):
        self.view.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        self.view.setMinimumSize(
            400,
            400
        )
        for ident in self.handler.idents:
            self.ui.cbx_channels.addItem(ident)
        self.ui.cbx_channels.addItem("Composite")
        self.ui.cbx_channels.setCurrentText("Composite")
        self.ui.cbx_channels.currentIndexChanged.connect(self.on_channel_selection_change)
        self.ui.btn_save.clicked.connect(self.on_button_click)
        self.ui.image_view.insertWidget(0, self.view, 3)

    def resizeEvent(self, event):
        super(ImgDialog, self).resizeEvent(event)
        self.set_current_image()

    def on_channel_selection_change(self):
        if self.ui.cbx_channels.currentIndex() < len(self.handler.idents):
            nump = img_as_ubyte(self.image[..., self.ui.cbx_channels.currentIndex()])
            tempImg = NucView.get_qimage_from_numpy(nump, mode="L")
        else:
            tempImg = NucView.get_qimage_from_numpy(self.image, "RGB")
        self.pmap.convertFromImage(tempImg)
        self.set_current_image()

    def on_button_click(self):
        self.save_image()

    def set_current_image(self):
        # TODO fertigstellen
        cur_ind = self.ui.cbx_channels.currentIndex()
        if cur_ind > len(self.handler.idents):
            pass
        else:
            pass
        self.view.scene().setSceneRect(0, 0, self.view.width() - 5, self.view.height() - 5)
        tempmap = self.pmap.scaled(self.view.width(), self.view.height(), Qt.KeepAspectRatio)
        self.sc_bckg.setPixmap(tempmap)
        # TODO
        """ 
        x_scale = tempmap.width() / self.pmap.width()
        y_scale = tempmap.height() / self.pmap.height()
        x_trans = self.view.scene().width() / 2 - tempmap.width() / 2
        y_trans = self.view.scene().height() / 2 - tempmap.height() / 2
        self.sc_bckg.setPos(self.view.scene().width() / 2 - tempmap.width() / 2,
                            self.view.scene().height() / 2 - tempmap.height() / 2)
        #self.view.clear_scene()
        for roi in self.handler.rois:
            roiI = QGraphicsFocusItem(color_index=self.handler.idents.index(roi.ident))
            temp = roi.calculate_dimensions()
            dim = (temp["width"], temp["height"])
            c = temp["center"]
            ulp = ((c[0] - dim[0] / 2) * x_scale + x_trans,
                   (c[1] - dim[1] / 2) * y_scale + y_trans)
            bbox = QRectF(ulp[0], ulp[1], dim[0] * x_scale, dim[1] * y_scale)
            roiI.setRect(bbox)
            self.view.scene().addItem(roiI)
        """

    def save_image(self):
        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"results/images")
        os.makedirs(pathpardir, exist_ok=True)
        pathresult = os.path.join(pathpardir,
                                  "result - {}.png".format(self.img_data["id"]))
        self.img_data["result_qt"].save(pathresult)
        # TODO


class SettingsDialog(QDialog):
    """
    Class to display a settings window, dynamically generated from a JSON file
    """

    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.data = {}
        self.changed = {}
        self.json = None
        self.url = None
        self._initialize_ui()

    def _initialize_ui(self):
        self.ui = uic.loadUi(ui_settings_dial, self)

    def initialize_from_file(self, url):
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

    def add_section(self, section):
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

    def add_menu_point(self, section, menupoint):
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

    def menupoint_changed(self, _id=None, value=None):
        """
        Method to detect value changes of the settings widgets
        :param _id: The id of the widget as str
        :param value: The value of the widget. Types depends on widget type
        :return: None
        """
        self.changed[_id] = value
        self.data[_id] = value

    def save_menu_settings(self):
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

    def __init__(self, image=None, handler=None, parent=None):
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
        self.initialize_ui()

    def accept(self):
        for comm in self.commands:
            self.curs.execute(
                comm[0],
                comm[1]
            )
        self.conn.commit()
        self.conn.close()
        super(ModificationDialog, self).accept()

    def reject(self):
        self.handler = self.original
        super(ModificationDialog, self).reject()

    def initialize_ui(self):
        self.ui = uic.loadUi(ui_modification_dial, self)
        # Initialize channel selector
        chan_num = len(self.handler.idents)
        self.max = chan_num - 1
        self.ui.sb_channel.setMaximum(chan_num)
        self.view = NucView(self.image, self.handler, self.commands,
                            self.cur_channel, self.show, True, self.max, self.curs)
        self.ui.graph_par.insertWidget(0, self.view, 3)
        self.lst_nuc_model = QStandardItemModel(self.ui.lst_nuc)
        self.ui.lst_nuc.setModel(self.lst_nuc_model)
        self.ui.lst_nuc.setIconSize(QSize(75, 75))
        self.ui.lst_nuc.selectionModel().selectionChanged.connect(self.on_selection_change)
        self.set_list_images(self.view.images)
        # Initialize buttons
        self.ui.sb_channel.valueChanged.connect(self.on_nucleus_selection_change)
        self.ui.btn_show.clicked.connect(self.on_button_click)
        self.ui.btn_merge.clicked.connect(self.on_button_click)
        self.ui.btn_remove.clicked.connect(self.on_button_click)
        self.ui.btn_edit.clicked.connect(self.on_button_click)
        # Initialize interactivity of graphics view
        self.set_current_image()

    def set_list_images(self, images):
        self.lst_nuc_model.clear()
        for image in images:
            item = QStandardItem()
            item_text = "Index: {}".format(images.index(image))
            item.setText(item_text)
            item.setTextAlignment(QtCore.Qt.AlignLeft)
            pmap = QPixmap()
            pmap.convertFromImage(NucView.get_qimage_from_numpy(image[...,
                                                                      self.handler.idents.index(self.handler.main)]
                                                                ))
            ic = QIcon(pmap)
            item.setIcon(ic)
            self.lst_nuc_model.appendRow(item)

    def get_roi_images(self):
        if self.handler is not None:
            self.lst_nuc_model.clear()
            self.view.images.clear()
            for nuc in self.view.main:
                self.view.images.append(nuc.get_as_numpy())
            for image in self.view.images:
                item = QStandardItem()
                item_text = "Index: {}".format(image[5])
                item.setText(item_text)
                item.setTextAlignment(QtCore.Qt.AlignLeft)
                item.setFlag(QGraphicsItem.ItemIsSelectable)
                pmap = QPixmap()
                pmap.convertFromImage(NucView.get_qimage_from_numpy(image[0]))
                ic = QIcon(pmap)
                item.setIcon(ic)
                self.lst_nuc_model.appendRow(item)

    def on_nucleus_selection_change(self):
        self.cur_channel = self.ui.sb_channel.value()
        self.set_current_image()

    def on_button_click(self):
        ident = self.sender().objectName()
        if ident == "btn_show":
            self.show = self.ui.btn_show.isChecked()
            self.view.show = self.show
        elif ident == "btn_edit":
            self.view.edit = self.ui.btn_edit.isChecked()
        elif ident == "btn_remove":
            selection = self.ui.lst_nuc.selectionModel().selectedIndexes()
            if selection:
                sel = [x.row() for x in selection]
                code = QMessageBox.question(self, "Remove Nuclei...",
                                            "Do you really want to remove following nuclei: {}".format(sel),
                                            QMessageBox.Yes | QMessageBox.No)
                if code == QMessageBox.Yes:
                    offset = 0
                    for ind in sel:
                        nuc = self.view.main[ind]
                        self.handler.rois.remove(nuc)
                        self.view.main.remove(nuc)
                        self.lst_nuc_model.removeRow(ind + offset)
                        del self.view.main[ind]
                        del self.view.images[ind + offset]
                        offset -= 1
                        self.commands.extend(
                            (("DELETE FROM roi WHERE hash = ? OR associated = ?",
                             (hash(nuc), hash(nuc))),
                             ("DELETE FROM points WHERE hash = ?",
                             (hash(nuc),)))
                        )
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
        self.set_current_image()

    def update_list_indices(self):
        for a in range(len(self.view.main)):
            self.lst_nuc_model.item(a, 0).setText("Index: {}".format(a))

    def on_selection_change(self):
        index = self.ui.lst_nuc.selectionModel().selectedIndexes()
        self.ui.btn_merge.setEnabled(False)
        if index:
            self.last_index = self.cur_index
            self.cur_index = index[0].row()
            self.set_current_image()
            if len(index) > 1:
                self.ui.btn_merge.setEnabled(True)

    def set_current_image(self):
        self.view.show_nucleus(self.cur_index, self.cur_channel)


class NucView(QGraphicsView):

    def __init__(self, image, handler, commands, cur_channel=None, show=True, edit=False, max_channel=None,
                 db_curs=None, parent=None):
        super(NucView, self).__init__(parent)
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
        self.pos = None
        self.temp_foc = None
        self.images = []
        self.foc_group = []
        self.map = {}
        self.commands = commands
        scene = QGraphicsScene(self)
        scene.setSceneRect(0, 0, self.width(), self.height())
        self.setScene(scene)
        for nuc in self.main:
            self.images.append(self.convert_roi_to_numpy(nuc))
        # Initialization of the background image
        self.sc_bckg = self.scene().addPixmap(QPixmap())
        self.show_nucleus(self.cur_ind, self.channel)

    def show_nucleus(self, cur_ind, channel):
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

    @staticmethod
    def get_qimage_from_numpy(numpy, mode=None):
        """
        Method to convert a numpy array to an QImage
        :param numpy: The array to convert
        :param mode: The mode to use for conversion
        :return: The QImage
        """
        img = Image.fromarray(numpy, mode)
        qimg = ImageQt(img)
        return qimg

    def clear_scene(self):
        for item in self.foc_group:
            self.scene().removeItem(item)
        self.foc_group.clear()

    def resizeEvent(self, event):
        self.show_nucleus(self.cur_ind, self.channel)

    def keyPressEvent(self, event):
        super(NucView, self).keyPressEvent(event)
        if event.key() == Qt.Key_Delete:
            for item in self.foc_group:
                if item.isSelected():
                    self.handler.rois.remove(self.map[item])
                    self.commands.extend((("DELETE FROM roi WHERE hash=?",
                                          (hash(self.map[item]),)),
                                         ("DELETE FROM points WHERE hash=?",
                                          (hash(self.map[item]),))))
                    del self.map[item]
                    self.scene().removeItem(item)

    def mousePressEvent(self, event):
        super(NucView, self).mousePressEvent(event)
        if self.edit and event.button() == Qt.LeftButton and \
                self.channel < self.handler.idents.index(self.handler.main):
            point = self.mapToScene(event.pos())
            p = self.itemAt(point.x(), point.y())
            if isinstance(p, QGraphicsPixmapItem):
                self.pos = event.pos()
                self.temp_foc = QGraphicsFocusItem(color_index=self.channel)
                self.scene().addItem(self.temp_foc)
        elif self.temp_foc is not None:
            self.scene().removeItem(self.temp_foc)
            self.pos = None
            self.temp_foc = None

    def mouseMoveEvent(self, event):
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

    def mouseReleaseEvent(self, event):
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
            imghash = self.curs.execute(
                "SELECT image FROM roi WHERE hash=?",
                (hash(self.cur_nuc), )
            ).fetchall()[0][0]
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

    def convert_roi_to_numpy(self, roi):
        dims = roi.calculate_dimensions()
        y_dist = dims["maxY"] - dims["minY"] + 1
        x_dist = dims["maxX"] - dims["minX"] + 1
        if self.channel > self.max_channel:
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

    def __init__(self, color_index=0):
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

    def paint(self, painter, style, widget=None):
        if self.isSelected():
            painter.setPen(QPen(self.sel_color, 6))
        else:
            painter.setPen(QPen(self.cur_col, 3))
        painter.drawEllipse(self.rect())
        self.scene().update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    pixmap = QPixmap("banner_norm.png")
    splash = QSplashScreen(pixmap)
    splash.show()
    splash.showMessage("Loading...")
    mainWin = NucDetect()
    splash.finish(mainWin)
    mainWin.show()
    sys.exit(app.exec_())
