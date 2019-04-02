import os
import sqlite3
import sys
import time
import json
from threading import Thread

import PyQt5
import numpy as np
import qtawesome as qta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtCore, QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import QSize, Qt, pyqtSignal, pyqtProperty, QRectF, QItemSelectionModel
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
from Nucleus.image import Channel
from Nucleus.image.ROI import ROI

PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, False)
PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, False)

ui_main = os.path.join(os.getcwd(), "nucdetect.ui")
ui_result_image_dialog = os.path.join(os.getcwd(), "result_image_dialog.ui")
ui_class_dial = os.path.join(os.getcwd(), "classification_dialog.ui")
ui_stat_dial = os.path.join(os.getcwd(), "statistics_dialog.ui")
ui_settings_dial = os.path.join(os.getcwd(), "settings_dialog.ui")
ui_modification_dial = os.path.join(os.getcwd(), "modification_dialog.ui")
database = os.path.join(os.pardir, "database{}nucdetect.db".format(os.sep))


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
        self.detector = Detector()
        self.connection = sqlite3.connect(database)
        self.cursor = self.connection.cursor()
        self.settings = self.load_settings()
        self.detector.settings = self.settings
        self.reg_images = {}
        self.sel_images = []
        self.img_keys = {}
        self.cur_img = {}
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
        self.res_table_model.setHorizontalHeaderLabels(["Index", "Width", "Height", "Center", "Green Foci", "Red Foci"])
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
            self.ui.btn_analyse.setEnabled(True)
        else:
            self.ui.btn_analyse.setEnabled(True)

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
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", imgdir,
                                                   "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)",
                                                   options=options)
        if file_name:
            self.add_image_to_list(file_name)

    def add_image_to_list(self, name):
        """
        Method to add an image to the list of loaded files. The image will be processed, added and loaded.
        :param name: The path leading to the file
        :return: None
        """
        temp = os.path.split(name)
        folder = temp[0].split(sep=os.sep)[-1]
        file = temp[1]
        if os.path.splitext(file)[1] in Detector.FORMATS:
            t = time.strftime('%d.%m.%Y', time.gmtime(os.path.getctime(name)))
            item = QStandardItem()
            item_text = "Name: {}\nFolder: {}\nDate: {}".format(file, folder, str(t))
            item.setText(item_text)
            item.setTextAlignment(QtCore.Qt.AlignLeft)
            icon = QIcon()
            icon.addFile(name)
            item.setIcon(icon)
            self.img_list_model.appendRow(item)
            self.reg_images[item_text] = name
            self.img_keys[name] = self.detector.load_image(name, self.settings["chan_names"])

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
        self.img_keys.clear()
        self.detector.clear()

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
        key = self.img_keys[self.sel_images[0]]
        self.cur_img["path"] = self.sel_images[0]
        self.cur_img["key"] = key
        del self.sel_images[0]
        thread = Thread(target=self.analyze_image,
                        args=(key,
                              "Analysis finished in {} -- Program ready",
                              100, 100,))
        thread.start()

    def analyze_image(self, key, message, percent, maxi):
        con = sqlite3.connect(database)
        curs = con.cursor()
        self.unsaved_changes = True
        self.detector.analyse_image(key)
        data = self.detector.get_output(key)
        self.res_table_model.setRowCount(0)
        self.res_table_model.setHorizontalHeaderLabels(data["header"])
        self.res_table_model.setColumnCount(len(data["data"][0]))
        if not curs.execute(
            "SELECT * FROM imgIndex WHERE md5 = ?",
            (key,)
        ).fetchall():
            curs.execute(
                "DELETE FROM results WHERE md5 = ?",
                (key,)
            )
            curs.execute(
                "DELETE FROM nuclei WHERE image = ?",
                (key,)
            )
            curs.execute(
                "DELETE FROM foci WHERE image = ?",
                (key,)
            )
            curs.execute(
                "DELETE FROM focStat WHERE image = ?",
                (key,)
            )
            curs.execute(
                "DELETE FROM nucStat WHERE image = ?",
                (key,)
            )
        for nucleus in self.detector.snaps[key]["handler"].nuclei:
            nucdat = nucleus.get_data()
            nucstat = nucleus.calculate_statistics()
            curs.execute(
                "INSERT INTO nuclei VALUES (?, ?, ?, ?, ?, ?, ?)",
                (nucdat["id"], key, nucdat["center"][0], nucdat["center"][1], nucdat["width"], nucdat["height"], True)
            )
            curs.execute(
                "INSERT INTO nucStat VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? , ?, ?, ?, ? ,?)",
                (
                    nucdat["id"], nucstat["area"], nucstat["red_roi"], nucstat["green_roi"], nucstat["red_av_int"],
                    nucstat["red_med_int"], nucstat["green_av_int"], nucstat["green_med_int"], nucstat["red_low_int"],
                    nucstat["red_high_int"], nucstat["green_low_int"], nucstat["green_high_int"], nucstat["red_av_int"],
                    nucstat["green_av_int"],nucstat["red_low_int"], nucstat["red_high_int"], nucstat["green_low_int"],
                    nucstat["green_high_int"], key
                )
            )
            for red in nucleus.red:
                focdat = red.get_data()
                focstat = red.calculate_statistics()
                curs.execute(
                    "INSERT INTO foci VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (focdat["id"], key, nucdat["id"], focdat["center"][0], focdat["center"][1], focdat["width"],
                     focdat["height"])
                )
                curs.execute(
                    "INSERT INTO focStat VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (focdat["id"], nucdat["id"], 1, focstat["low_int"], focstat["high_int"], focstat["av_int"],
                     focstat["med_int"], focstat["area"], key)
                )
            for green in nucleus.green:
                focdat = green.get_data()
                focstat = green.calculate_statistics()
                curs.execute(
                    "INSERT INTO foci VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (focdat["id"], key, nucdat["id"], focdat["center"][0], focdat["center"][1], focdat["width"],
                     focdat["height"])
                )
                curs.execute(
                    "INSERT INTO focStat VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (focdat["id"], nucdat["id"], 2, focstat["low_int"], focstat["high_int"], focstat["av_int"],
                     focstat["med_int"], focstat["area"], key)
                )
        for x in range(len(data["data"])):
            row = []
            row_cop = data["data"][x].copy()
            row_cop.insert(0, key)
            row_cop[4] = str(row_cop[4])
            curs.execute(
                "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)",
                row_cop
            )
            for text in data["data"][x]:
                item = QStandardItem()
                item.setText(str(text))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setSelectable(False)
                row.append(item)
            self.res_table_model.appendRow(row)
        curs.execute(
            "UPDATE imgIndex SET analysed = ? WHERE md5 = ?",
            (True, key)
        )
        con.commit()
        con.close()
        self.prg_signal.emit(message.format(self.detector.snaps[key]["time"]), percent, maxi, "")
        self.ui.btn_save.setEnabled(True)
        self.ui.btn_images.setEnabled(True)
        self.ui.btn_statistics.setEnabled(True)
        self.ui.btn_categories.setEnabled(True)
        self.ui.btn_modify.setEnabled(True)
        '''
        if len(self.sel_images) is not 0:
            self.analyze()
            '''

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
        self.unsaved_changes = True
        self.selec_signal.emit()
        thread = Thread(target=self._analyze_all, args=(
            0, len(self.reg_images)-1))
        thread.start()

    def _analyze_all(self, percent=0, maxi=0):
        self.analyze_image(self.img_keys[self.sel_images[0]],
                           message="Analysing " + self.sel_images[0],
                           percent=percent, maxi=maxi)
        if percent < maxi:
            self.selec_signal.emit()
            # TODO Optimales sleep Intervall herausfinden + Detector Bug fixen -> Löschen von Snaps
            time.sleep(0.3)
            self._analyze_all(percent=percent + 1, maxi=maxi)
        if percent == maxi:
            self.prg_signal.emit("Analysis finished -- Program ready",
                                maxi,
                                maxi, "")
            self.selec_signal.emit()

    def show_result_image(self):
        # TODO Zoom implementieren: http://doc.qt.io/qt-5/qtwidgets-widgets-imageviewer-example.html
        image_dialog = ImgDialog(img_data=self.detector.get_snapshot(self.cur_img["key"]))
        image_dialog.setWindowTitle("Result Images for " + self.cur_img["path"])
        image_dialog.setWindowIcon(QtGui.QIcon('logo.png'))
        image_dialog.setWindowFlags(image_dialog.windowFlags() |
                                    QtCore.Qt.WindowSystemMenuHint |
                                    QtCore.Qt.WindowMinMaxButtonsHint|
                                    QtCore.Qt.Window)

        image_dialog.exec_()

    def save_results(self):
        key = self.cur_img["key"]
        save = Thread(target=self._save_results, args=(key,))
        self.prg_signal.emit("Saving Results", 0, 100, "")
        save.start()

    def _save_results(self, key):
        self.detector.create_ouput(key)
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
        mod = ModificationDialog(handler=self.detector.snaps[self.cur_img["key"]]["handler"])
        mod.setWindowTitle("Modification")
        mod.setWindowIcon(QtGui.QIcon("logo.png"))
        mod.setWindowFlags(mod.windowFlags() |
                           QtCore.Qt.WindowSystemMenuHint |
                           QtCore.Qt.WindowMinMaxButtonsHint |
                           QtCore.Qt.Window)
        mod.exec()

    def on_close(self):
        self.detector.save_all_snaps()
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

    def __init__(self, img_data, parent=None):
        super(ImgDialog, self).__init__(parent)
        self.ui = uic.loadUi(ui_result_image_dialog, self)
        self.initialize_ui()
        self.img_data = img_data
        c_img = self.img_data["result_qt"]
        pmap = QPixmap()
        pmap.convertFromImage(c_img)
        self.ui.image_view.setPixmap(pmap.scaled(self.width(), self.height(), QtCore.Qt.KeepAspectRatio))
        self.img = None
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def initialize_ui(self):
        self.ui.btn_original.clicked.connect(self.on_button_click)
        self.ui.btn_ch_blue.clicked.connect(self.on_button_click)
        self.ui.btn_ch_red.clicked.connect(self.on_button_click)
        self.ui.btn_ch_green.clicked.connect(self.on_button_click)
        self.ui.btn_thr_blue.clicked.connect(self.on_button_click)
        self.ui.btn_thr_red.clicked.connect(self.on_button_click)
        self.ui.btn_thr_green.clicked.connect(self.on_button_click)
        self.ui.btn_result.clicked.connect(self.on_button_click)
        self.ui.btn_edge_red.clicked.connect(self.on_button_click)
        self.ui.btn_edge_green.clicked.connect(self.on_button_click)
        self.ui.btn_edge_blue.clicked.connect(self.on_button_click)
        self.ui.btn_save.clicked.connect(self.on_button_click)

    def resizeEvent(self, event):
        super(ImgDialog, self).resizeEvent(event)
        self.set_current_image(self.img)

    def on_button_click(self):
        ident = self.sender().objectName()
        c_img = None
        if ident == "btn_original":
            c_img = self.img_data["original"]
        elif ident == "btn_ch_blue":
            c_img = self.img_data["channel"][0]
        elif ident == "btn_ch_red":
            c_img = self.img_data["channel"][1]
        elif ident == "btn_ch_green":
            c_img = self.img_data["channel"][2]
        elif ident == "btn_edge_blue":
            c_img = img_as_ubyte(self.img_data["edges"][0])
        elif ident == "btn_edge_red":
            c_img = img_as_ubyte(self.img_data["edges"][1])
        elif ident == "btn_edge_green":
            c_img = img_as_ubyte(self.img_data["edges"][2])
        elif ident == "btn_thr_blue":
            c_img = img_as_ubyte(self.img_data["binarized"][0])
        elif ident == "btn_thr_red":
            c_img = img_as_ubyte(self.img_data["binarized"][1])
        elif ident == "btn_thr_green":
            c_img = img_as_ubyte(self.img_data["binarized"][2])
        elif ident == "btn_save":
            self.save_image()
            return
        self.set_current_image(c_img)

    def set_current_image(self, c_img):
        self.img = c_img
        if c_img is not None:
            qimg = self.convert_numpy_to_qimage(c_img)
            pmap = QPixmap()
            pmap.convertFromImage(qimg)
            self.ui.image_view.setPixmap(pmap.scaled(self.width(), self.height(), QtCore.Qt.KeepAspectRatio))
        else:
            c_img = self.img_data["result_qt"]
            pmap = QPixmap()
            pmap.convertFromImage(c_img)
            self.ui.image_view.setPixmap(pmap.scaled(self.width(), self.height(), QtCore.Qt.KeepAspectRatio))

    def convert_numpy_to_qimage(self, numpy):
        img = Image.fromarray(numpy)
        qimg = ImageQt(img)
        return qimg

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

    def __init__(self, handler=None, parent=None):
        super(ModificationDialog, self).__init__(parent)
        self.handler = handler
        self.original = handler.nuclei.copy()
        self.last_index = 0
        self.cur_index = 0
        self.cur_channel = 3
        self.mp = None
        self.initialize_ui()

    def reject(self):
        self.handler.nuclei = self.original
        super(ModificationDialog, self).reject()

    def initialize_ui(self):
        self.ui = uic.loadUi(ui_modification_dial, self)
        self.view = NucView(self.handler, self.cur_index, self.cur_channel, self.show, True)
        self.ui.graph_par.insertWidget(0, self.view, 3)
        self.lst_nuc_model = QStandardItemModel(self.ui.lst_nuc)
        self.ui.lst_nuc.setModel(self.lst_nuc_model)
        self.ui.lst_nuc.setIconSize(QSize(75, 75))
        self.ui.lst_nuc.selectionModel().selectionChanged.connect(self.on_selection_change)
        self.set_list_images(self.view.images)
        # Initialize buttons
        self.ui.btn_comp.clicked.connect(self.on_button_click)
        self.ui.btn_blue.clicked.connect(self.on_button_click)
        self.ui.btn_red.clicked.connect(self.on_button_click)
        self.ui.btn_green.clicked.connect(self.on_button_click)
        self.ui.btn_show.clicked.connect(self.on_button_click)
        self.ui.btn_merge.clicked.connect(self.on_button_click)
        self.ui.btn_remove.clicked.connect(self.on_button_click)
        self.ui.btn_edit.clicked.connect(self.on_button_click)
        # Initialize interactivity of graphics view
        self.set_current_image()

    def set_list_images(self, rois):
        for image in rois:
            item = QStandardItem()
            item_text = "Index: {}".format(image[5])
            item.setText(item_text)
            item.setTextAlignment(QtCore.Qt.AlignLeft)
            pmap = QPixmap()
            pmap.convertFromImage(self.get_qimage_from_numpy(image[0]))
            ic = QIcon(pmap)
            item.setIcon(ic)
            self.lst_nuc_model.appendRow(item)

    def get_roi_images(self):
        if self.handler is not None:
            self.lst_nuc_model.clear()
            self.view.images.clear()
            for nuc in self.view.handler.nuclei:
                self.view.images.append(self.handler.get_roi_as_image(nuc))
            for image in self.view.images:
                item = QStandardItem()
                item_text = "Index: {}".format(image[5])
                item.setText(item_text)
                item.setTextAlignment(QtCore.Qt.AlignLeft)
                item.setFlag(QGraphicsItem.ItemIsSelectable)
                pmap = QPixmap()
                pmap.convertFromImage(self.get_qimage_from_numpy(image[0]))
                ic = QIcon(pmap)
                item.setIcon(ic)
                self.lst_nuc_model.appendRow(item)

    def on_button_click(self):
        ident = self.sender().objectName()
        if ident == "btn_comp":
            self.cur_channel = 3
        elif ident == "btn_blue":
            self.cur_channel = 0
        elif ident == "btn_red":
            self.cur_channel = 1
        elif ident == "btn_green":
            self.cur_channel = 2
        elif ident == "btn_show":
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
                        self.lst_nuc_model.removeRow(ind + offset)
                        del self.view.images[ind + offset]
                        del self.view.handler.nuclei[ind + offset]
                        offset -= 1
                    self.handler.calculate_statistics()
                    self.cur_index = 0
                    self.set_current_image()
        elif ident == "btn_merge":
            selection = self.ui.lst_nuc.selectionModel().selectedIndexes()
            sel = [x.row() for x in selection]
            code = QMessageBox.question(self, "Merge Nuclei...",
                                        "Do you really want to merge following nuclei: {}".format(sel),
                                        QMessageBox.Yes | QMessageBox.No)
            if code == QMessageBox.Yes:
                seed = self.view.handler.nuclei[sel[0]]
                offset = 0
                for x in range(1, len(sel)):
                    ind = sel[x]
                    merger = self.view.handler.nuclei[ind + offset]
                    seed.merge(merger)
                    self.lst_nuc_model.removeRow(ind + offset)
                    del self.view.images[ind + offset]
                    del self.view.handler.nuclei[ind + offset]
                    offset -= 1
                self.view.handler.calculate_statistics()
                self.get_roi_images()
                self.ui.lst_nuc.selectionModel().select(selection[0], QItemSelectionModel.Select)
        self.set_current_image()

    def on_selection_change(self):
        index = self.ui.lst_nuc.selectionModel().selectedIndexes()
        self.ui.btn_merge.setEnabled(False)
        if index:
            self.last_index = self.cur_index
            self.cur_index = index[0].row()
            self.set_current_image()
            if len(index) > 1:
                self.ui.btn_merge.setEnabled(True)

    def get_qimage_from_numpy(self, numpy):
        img = Image.fromarray(numpy)
        qimg = ImageQt(img)
        return qimg

    def set_current_image(self):
        self.view.show_nucleus(self.cur_index, self.cur_channel)


class NucView(QGraphicsView):

    def __init__(self, handler, cur_index=3, cur_channel=0, show=True, edit=False, parent=None):
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
        self.index = cur_index
        self.channel = cur_channel
        self.show = show
        self.edit = edit
        self.pos = None
        self.temp_foc = None
        self.handler = handler
        self.images = []
        self.foc_group = []
        self.map = {}
        scene = QGraphicsScene(self)
        scene.setSceneRect(0, 0, self.width(), self.height())
        self.setScene(scene)
        for nuc in self.handler.nuclei:
            self.images.append(self.handler.get_roi_as_image(nuc))
        # Initialization of the background image
        self.sc_bckg = self.scene().addPixmap(QPixmap())
        self.show_nucleus(self.index, self.channel)

    def show_nucleus(self, index, channel):
        self.index = index
        self.channel = channel
        self.scene().setSceneRect(0, 0, self.width() - 5, self.height() - 5)
        pmap = QPixmap()
        pmap.convertFromImage(self.get_qimage_from_numpy(self.images[self.index][self.channel]))
        tempmap = pmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
        self.sc_bckg.setPixmap(tempmap)
        x_scale = tempmap.width() / pmap.width()
        y_scale = tempmap.height() / pmap.height()
        x_trans = self.scene().width() / 2 - tempmap.width() / 2
        y_trans = self.scene().height() / 2 - tempmap.height() / 2
        self.sc_bckg.setPos(self.scene().width() / 2 - tempmap.width() / 2,
                            self.scene().height() / 2 - tempmap.height() / 2)
        self.clear_scene()
        if self.show:
            if self.channel == 1:
                nuc_dat = self.handler.nuclei[self.index].get_data()["minmax"]
                x_offset = nuc_dat[0]
                y_offset = nuc_dat[2]
                for red in self.handler.nuclei[self.index].red:
                    foc = QGraphicsFocusItem()
                    temp = red.get_data()
                    dim = (temp["width"], temp["height"])
                    c = temp["center"]
                    ulp = ((c[0] - dim[0] / 2 - x_offset) * x_scale + x_trans,
                           (c[1] - dim[1] / 2 - y_offset) * y_scale + y_trans)
                    bbox = QRectF(ulp[0], ulp[1], dim[0] * x_scale, dim[1] * y_scale)
                    foc.setRect(bbox)
                    self.map[foc] = red
                    self.foc_group.append(foc)
                    self.scene().addItem(foc)
            elif self.channel == 2:
                nuc_dat = self.handler.nuclei[self.index].get_data()["minmax"]
                x_offset = nuc_dat[0]
                y_offset = nuc_dat[2]
                for green in self.handler.nuclei[self.index].green:
                    foc = QGraphicsFocusItem(channel=Channel.GREEN)
                    temp = green.get_data()
                    dim = (temp["width"], temp["height"])
                    c = temp["center"]
                    ulp = ((c[0] - dim[0] / 2 - x_offset) * x_scale + x_trans,
                           (c[1] - dim[1] / 2 - y_offset) * y_scale + y_trans)
                    bbox = QRectF(ulp[0], ulp[1], dim[0] * x_scale, dim[1] * y_scale)
                    foc.setRect(bbox)
                    self.map[foc] = green
                    self.foc_group.append(foc)
                    self.scene().addItem(foc)
            elif self.channel == 3:
                nuc_dat = self.handler.nuclei[self.index].get_data()["minmax"]
                x_offset = nuc_dat[0]
                y_offset = nuc_dat[2]
                for red in self.handler.nuclei[self.index].red:
                    foc = QGraphicsFocusItem()
                    temp = red.get_data()
                    dim = (temp["width"], temp["height"])
                    c = temp["center"]
                    ulp = ((c[0] - dim[0] / 2 - x_offset) * x_scale + x_trans,
                           (c[1] - dim[1] / 2 - y_offset) * y_scale + y_trans)
                    bbox = QRectF(ulp[0], ulp[1], dim[0] * x_scale, dim[1] * y_scale)
                    foc.setRect(bbox)
                    self.map[foc] = red
                    self.foc_group.append(foc)
                    self.scene().addItem(foc)
                for green in self.handler.nuclei[self.index].green:
                    foc = QGraphicsFocusItem(channel=Channel.GREEN)
                    temp = green.get_data()
                    dim = (temp["width"], temp["height"])
                    c = temp["center"]
                    ulp = ((c[0] - dim[0] / 2 - x_offset) * x_scale + x_trans,
                           (c[1] - dim[1] / 2 - y_offset) * y_scale + y_trans)
                    bbox = QRectF(ulp[0], ulp[1], dim[0] * x_scale, dim[1] * y_scale)
                    foc.setRect(bbox)
                    self.map[foc] = green
                    self.foc_group.append(foc)
                    self.scene().addItem(foc)

    def get_qimage_from_numpy(self, numpy):
        img = Image.fromarray(numpy)
        qimg = ImageQt(img)
        return qimg

    def clear_scene(self):
        for item in self.foc_group:
            self.scene().removeItem(item)
        self.foc_group.clear()

    def resizeEvent(self, event):
        self.show_nucleus(self.index, self.channel)

    def keyPressEvent(self, event):
        super(NucView, self).keyPressEvent(event)
        if event.key() == Qt.Key_Delete:
            for item in self.foc_group:
                if item.isSelected():
                    if item.channel == Channel.RED:
                        self.handler.nuclei[self.index].red.remove(self.map[item])
                    else:
                        self.handler.nuclei[self.index].green.remove(self.map[item])
                    del self.map[item]
                    self.scene().removeItem(item)

    def mousePressEvent(self, event):
        super(NucView, self).mousePressEvent(event)
        if self.edit and event.button() == Qt.LeftButton and 0 < self.channel < 3:
            point = self.mapToScene(event.pos())
            p = self.itemAt(point.x(), point.y())
            if isinstance(p, QGraphicsPixmapItem):
                self.pos = event.pos()
                if self.channel == 1:
                    self.temp_foc = QGraphicsFocusItem()
                elif self.channel == 2:
                    self.temp_foc = QGraphicsFocusItem(channel=Channel.GREEN)
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
            cur_nuc = self.images[self.index][self.channel]
            offset_factor = self.sc_bckg.boundingRect().height() / len(cur_nuc)
            hard_offset = self.sc_bckg.pos()
            bbox = self.temp_foc.boundingRect()
            tx = bbox.x() + 1/2 * bbox.width()
            ty = bbox.y() + 1/2 * bbox.height()
            x_center = (tx - hard_offset.x()) / offset_factor
            y_center = (ty - hard_offset.y()) / offset_factor
            height = bbox.height() / offset_factor / 2
            width = bbox.width() / offset_factor / 2
            mask = np.zeros(shape=cur_nuc.shape)
            rr, cc = ellipse(y_center, x_center, height, width, shape=mask.shape)
            mask[rr, cc] = 1
            cur_roi = ROI(auto=False, chan=Channel.RED if self.channel is not 2 else Channel.GREEN)
            nuc_dat = self.handler.nuclei[self.index].get_data()["minmax"]
            x_offset = nuc_dat[0]
            y_offset = nuc_dat[2]
            for y in range(len(mask)):
                for x in range(len(mask[0])):
                    if mask[y][x] > 0:
                        inten = cur_nuc[y][x]
                        cur_roi.add_point((x + x_offset, y + y_offset), inten)
            if self.channel is 1:
                self.handler.nuclei[self.index].red.append(cur_roi)
            elif self.channel is 2:
                self.handler.nuclei[self.index].green.append(cur_roi)
            self.foc_group.append(self.temp_foc)
            self.pos = None
            self.temp_foc = None
            self.scene().update()


class QGraphicsFocusItem(QGraphicsEllipseItem):

    def __init__(self, channel=Channel.RED, parent=None):
        super(QGraphicsFocusItem, self).__init__(parent=parent)
        if channel is Channel.RED:
            self.main_color = QColor(255, 50, 0)
            self.hover_color = QColor(255, 150, 0)
            self.sel_color = QColor(255, 75, 150)
        else:
            self.main_color = QColor(50, 255, 0)
            self.hover_color = QColor(150, 255, 0)
            self.sel_color = QColor(75, 255, 150)
        self.cur_col = self.main_color
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.channel = channel

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