import os
import sys
import time
import json
import ast
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
from PyQt5.QtCore import QSize, Qt, pyqtSignal, pyqtProperty
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIcon, QPixmap, QColor
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QHeaderView, QDialog, QSplashScreen, QSizePolicy, QWidget, \
    QVBoxLayout
from qtconsole.qt import QtGui
from skimage import img_as_ubyte

from NucDetect.core.Detector import Detector
from NucDetect.gui.settings.Settings import SettingsShowWidget, SettingsSlider, SettingsTextWidget, SettingsComboBox

PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

ui_main = os.path.join(os.getcwd(), "nucdetect.ui")
ui_result_image_dialog = os.path.join(os.getcwd(), "result_image_dialog.ui")
ui_class_dial = os.path.join(os.getcwd(), "classification_dialog.ui")
ui_stat_dial = os.path.join(os.getcwd(), "statistics_dialog.ui")
ui_settings_dial = os.path.join(os.getcwd(), "settings_dialog.ui")

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
        self.reg_images = {}
        self.sel_images = []
        self.img_keys = {}
        self.cur_img = {}
        self.unsaved_changes = False
        self._setup_ui()
        self.setWindowTitle("NucDetect")
        self.setWindowIcon(QtGui.QIcon('logo.png'))

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
                                                  "All Files (*);;TIFF Files (*.tiff);;PNG Files (*.png)",
                                                  options=options)
        if file_name:
            self.add_image_to_list(file_name)

    def add_image_to_list(self, name):
        """
        Method to add an image to the list of loaded files. The image will be processed, added and loaded.
        :param name: The path leading to the file
        :return: None
        """
        if name.rfind("\\") is -1:
            file = name[name.rfind("/") + 1:]
            folder = os.path.dirname(name)
            folder = folder[folder.rfind("/") + 1:]
        else:
            file = name[name.rfind("\\") + 1:]
            folder = os.path.dirname(name)
            folder = folder[folder.rfind("\\") + 1:]
        t = time.strftime('%d.%m.%Y, %H:%M', time.gmtime(os.path.getctime(name)))
        item = QStandardItem()
        item_text = "Name: " + file + "\nFolder: " + folder + "\nDate: " + str(t)
        item.setText(item_text)
        item.setTextAlignment(QtCore.Qt.AlignLeft)
        icon = QIcon()
        icon.addFile(name)
        item.setIcon(icon)
        self.img_list_model.appendRow(item)
        self.reg_images[item_text] = name
        self.img_keys[name] = self.detector.load_image(name)

    def add_images_from_folder(self, url):
        """
        Method to load a whole folder of images
        :param url: The path of the folder
        :return: None
        """
        files = os.listdir(url)
        for file in files:
            path = os.path.join(url, file)
            if os.path.isfile(path):
                self.add_image_to_list(path)

    def remove_image_from_list(self, name):
        """
        Method to remove an loaded image from the file list.
        :param name: The file name of the image to remove
        :return:
        """
        # TODO
        pass

    def clear_image_list(self):
        pass

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
                              "Analysis finished -- Program ready",
                              100, 100,))
        thread.start()

    def analyze_image(self, key, message, percent, maxi):
        self.unsaved_changes = True
        self.detector.analyse_image(key)
        data = self.detector.get_output(key)
        self.res_table_model.setRowCount(0)
        self.res_table_model.setHorizontalHeaderLabels(data["header"])
        self.res_table_model.setColumnCount(len(data["data"][0]))
        for x in range(len(data["data"])):
            row = []
            for text in data["data"][x]:
                item = QStandardItem()
                item.setText(str(text))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setSelectable(False)
                row.append(item)
            self.res_table_model.appendRow(row)
        self.prg_signal.emit(message, percent, maxi, "")
        self.ui.btn_save.setEnabled(True)
        self.ui.btn_images.setEnabled(True)
        self.ui.btn_statistics.setEnabled(True)
        self.ui.btn_categories.setEnabled(True)
        '''
        if len(self.sel_images) is not 0:
            self.analyze()
            '''

    def _select_next_image(self):
        max_ind = self.img_list_model.rowCount()
        cur_ind = self.ui.list_images.currentIndex()
        if cur_ind.row() < max_ind:
            nex = self.img_list_model.index(cur_ind.row() + 1, 0)
            self.ui.list_images.setCurrentIndex(nex)
        else:
            first = self.img_list_model.index(0, 0)
            self.ui.list_images.setCurrentIndex(first)

    def _set_progress(self, text, progress, maxi, symbol):
        self.ui.lbl_status.setText(text + " -- " + str(progress) + "% " + symbol)
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
        self.detector.save_result_image(key)
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
        stat = self.detector.get_statistics(self.cur_img["key"])
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
            "Min. Red Number: {:>}".format(min([ x for x in stat["number red"] if x > 0])))
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
        '''
        model = QStandardItemModel(stat_dialog.ui.lv_data)
        genItem = QStandardItem()
        genItem.setSelectable(False)
        genItem.setText(
            "Detected nuclei: \t" + str(stat["number"]) + "\n" +
            "Empty nuclei: \t" + str(stat["empty"]) + "\n" +
            "Detected red foci: \t" + str(np.sum(stat["number red"])) + "\n" +
            "Detected green foci: \t" + str(np.sum(stat["number green"]))
        )
        genItem.setTextAlignment(QtCore.Qt.AlignLeft)
        nucItem = QStandardItem()
        nucItem.setSelectable(False)
        nucItem.setText(
            "Average nucleus area: \t" + str(stat["area average"]) + "\n" +
            "Median nucleus area: \t" + str(stat["area median"]) + "\n" +
            "Std nucleus area: \t" + str(stat["area std"]) + "\n" +
            "Average red foci number: \t" + str(stat["number red average"]) + "\n" +
            "Std red foci number: \t" + str(stat["number red std"]) + "\n" +
            "Average red foci number: \t" + str(stat["number green average"]) + "\n" +
            "Std red foci number: \t" + str(stat["number green std"]) + "\n" +
            "Average red foci intensity: \t" + str(stat["intensity red average"]) + "\n" +
            "Average green foci intensity: \t" + str(stat["intensity green average"])
        )
        nucItem.setTextAlignment(QtCore.Qt.AlignLeft)
        focItem = QStandardItem()
        focItem.setSelectable(False)
        focItem.setText(
            "Average red foci intensity: \t" + str(stat["intensity red average"]) + "\n" +
            "Std red foci intensity: \t" + str(stat["intensity red std"]) + "\n" +
            "Average green foci intensity: \t" + str(stat["intensity green average"]) + "\n" +
            "Std green foci intensity: \t" + str(stat["intensity green std"])
        )
        focItem.setTextAlignment(QtCore.Qt.AlignLeft)
        model.appendRow(genItem)
        model.appendRow(nucItem)
        model.appendRow(focItem)
        stat_dialog.ui.lv_data.setModel(model)
        '''
        cnvs_red_poisson = PoissonCanvas(stat["number red average"],
                                         max(stat["number red"])+1,
                                         stat["number red"], name="red channel")
        cnvs_green_poisson = PoissonCanvas(stat["number red average"],
                                           max(stat["number green"]) + 1,
                                           stat["number green"], name="green channel")
        cnvs_red_int = BarChart(name="red channel", title="Red Channel - Average Focus Intensity",
                                y_title="Average Intensity", x_title="Nucleus Index", x_label_rotation=45)
        cnvs_red_int.values.append(stat["intensity red"])
        cnvs_red_int.labels.append(np.arange(len(stat["intensity red"])))
        cnvs_red_int.plot()
        cnvs_red_int.setToolTip(("Shows the average red foci intensity for the nucleus with the given index.\n"
                                   "255 is the maximal possible value. If no intensity is shown, no red foci were\n"
                                   "detected in the respective nucleus"))
        cnvs_green_int = BarChart(name="green channel", title="Green Channel - Average Focus Intensity",
                                y_title="Average Intensity", x_title="Nucleus Index", x_label_rotation=45)
        cnvs_green_int.values.append(stat["intensity green"])
        cnvs_green_int.labels.append(np.arange(len(stat["intensity green"])))
        cnvs_green_int.plot()
        cnvs_green_int.setToolTip(("Shows the average green foci intensity for the nucleus with the given index.\n" 
                                   "255 is the maximal possible value. If no intensity is shown, no green foci were\n"
                                   "detected in the respective nucleus"))
        cnvs_num = XYChart(name="numbers", title="Foci Number", x_title="Nucleus Index", y_title="Foci")
        cnvs_num.x_values.append(np.arange(len(stat["number red"])))
        cnvs_num.x_values.append(np.arange(len(stat["number green"])))
        cnvs_num.y_values.append(stat["number red"])
        cnvs_num.y_values.append(stat["number green"])
        cnvs_num.colmarks = ["ro", "go"]
        cnvs_num.dat_label = ["Green Channel", "Red Channel"]
        cnvs_num.plot()
        cnvs_int = XYChart(name="intensities", title="Foci Intensities", x_title="Nucleus Index",
                           y_title="Average Intensity")
        ind_red = 0
        for inten in stat["intensity red total"]:
            x_t = np.zeros(len(inten))
            x_t.fill(ind_red)
            cnvs_int.x_values.append(x_t)
            cnvs_int.y_values.append(inten)
            cnvs_int.colmarks.append("ro")
            ind_red += 1
        ind_green = 0
        for inten in stat["intensity green total"]:
            x_t = np.zeros(len(inten))
            x_t.fill(ind_green)
            cnvs_int.x_values.append(x_t)
            cnvs_int.y_values.append(inten)
            cnvs_int.colmarks.append("go")
            ind_green += 1
        cnvs_int.plot()
        stat_dialog.ui.vl_poisson.addWidget(cnvs_red_poisson)
        stat_dialog.ui.vl_poisson.addWidget(cnvs_green_poisson)
        stat_dialog.ui.vl_int.addWidget(cnvs_red_int)
        stat_dialog.ui.vl_int.addWidget(cnvs_green_int)
        stat_dialog.ui.vl_vals.addWidget(cnvs_num)
        stat_dialog.ui.vl_vals.addWidget(cnvs_int)
        stat_dialog.setWindowFlags(stat_dialog.windowFlags() |
                                   QtCore.Qt.WindowSystemMenuHint |
                                   QtCore.Qt.WindowMinMaxButtonsHint)
        stat_dialog.exec()

    def show_categorization(self):
        cl_dialog = QDialog()
        cl_dialog.ui = uic.loadUi(ui_class_dial, cl_dialog)
        cl_dialog.setWindowTitle("Classification")
        cl_dialog.setWindowIcon(QtGui.QIcon('logo.png'))
        categories = self.detector.get_categories(self.cur_img["key"])
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

    def show_settings(self):
        sett = SettingsDialog()
        print(os.path.join(os.getcwd(), "settings\settings.json"))
        sett.initialize_from_file(os.path.join(os.getcwd(), "settings/settings.json"))
        sett.setWindowTitle("Settings")
        sett.setModal(True)
        sett.setWindowIcon(QtGui.QIcon("logo.png"))
        code = sett.exec()
    def on_close(self):
        self.detector.save_all_snaps()


class PoissonCanvas(FigureCanvas):

    def __init__(self, _lambda, k, values, name="", parent=None, width=4, height=4, dpi=65):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.name = name
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        self.plot(_lambda, k, values)

    def plot(self, _lambda, k, values):
        poisson = np.random.poisson(_lambda, k)
        ax = self.figure.add_subplot(111)
        objects = np.arange(k)
        x_pos = np.arange(len(poisson))
        conv_values = np.zeros(len(poisson))
        for val in values:
            conv_values[val] += 1
        ax.set_title("Poisson distribution - " + self.name)
        ax.bar(x_pos, poisson, align="center", alpha=0.5)
        ax.bar(x_pos, conv_values, align="center", alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(objects, rotation=45)
        ax.set_ylabel("Probability [%]")
        ax.set_xlabel("Foci number [N]")
        self.draw()


class XYChart(FigureCanvas):

    def __init__(self, parent=None, name="", title="", x_title="",
                 y_title="", width=4, height=4, dpi=65, x_label_max_num=20, y_label_max_num=20, x_label_rotation=0,
                 y_label_rotation=0):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.name = name
        self.title = title
        self.x_label_max_num = x_label_max_num
        self.y_label_max_num = y_label_max_num
        self.x_title = x_title
        self.y_title = y_title
        self.x_label_rotation = x_label_rotation
        self.y_label_rotation = y_label_rotation
        self.x_values = []
        self.y_values = []
        self.dat_label = []
        self.colmarks = []
        self.lines = []
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)

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


class BarChart(FigureCanvas):

    def __init__(self, parent=None, overlay=True, name="", title="", x_title="", y_title="", width=4, height=4, dpi=65,
                 x_label_rotation=0, y_label_rotation=0):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.name = name
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.y_label_rotation = y_label_rotation
        self.x_label_rotation = x_label_rotation
        self.labels = []
        self.values = []
        self.colors = []
        self.overlay = overlay
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)

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
        self.ui.btn_max.clicked.connect(self.change_window_size)
        self.ui.btn_max.setIcon(qta.icon("fa5s.window-maximize", color="white"))
        self.ui.btn_min.clicked.connect(self.change_window_size)
        self.ui.btn_min.setIcon(qta.icon("fa5s.window-minimize", color="white"))

    def change_window_size(self):
        ident = self.sender().objectName()
        if ident == "btn_max":
            self.setWindowState(Qt.WindowFullScreen)
            self.set_current_image(self.img)
        else:
            self.setWindowState(Qt.WindowNoState)
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
        elif ident == "btn_thr_blue":
            c_img = img_as_ubyte(self.img_data["binarized"][0])
        elif ident == "btn_thr_red":
            c_img = img_as_ubyte(self.img_data["binarized"][1])
        elif ident == "btn_thr_green":
            c_img = img_as_ubyte(self.img_data["binarized"][2])
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
        t = len(numpy.shape)
        if t is 3:
            img = Image.fromarray(numpy, mode="RGB")
        else:
            img = Image.fromarray(numpy, mode="L")
        qimg = ImageQt(img)
        return qimg


class SettingsDialog(QDialog):
    data = {}
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self._initialize_ui()

    def _initialize_ui(self):
        self.ui = uic.loadUi(ui_settings_dial, self)

    def initialize_from_file(self, url):
        if not url.lower().endswith(".json"):
            raise ValueError("Only JSON files can be loaded!")
        with open(url) as json_file:
            j_dat = json.load(json_file)
            for section, p in j_dat.items():
                self.add_menu_point(section, p)

    def add_section(self, section):
        try:
            self.data[section]
        except KeyError:
            self.data[section] = []
            tab = QWidget()
            layout = QVBoxLayout()
            layout.setObjectName("base")
            tab.setLayout(layout)
            self.ui.settings.addTab(tab, section)

    def add_menu_point(self, section, menupoint):
        self.add_section(section)
        for ind in range(self.ui.settings.count()):
            if self.ui.settings.tabText(ind) == section:
                for mp in menupoint:
                    t = mp["type"].lower()
                    p = None
                    if t == "show":
                        # TODO
                        p = SettingsShowWidget(
                            mp["widget"]
                        )
                    elif t == "slider":
                        p = SettingsSlider(
                            title=mp["title"],
                            desc=mp["desc"],
                            min_val=mp["values"]["min"],
                            max_val=mp["values"]["max"],
                            step=mp["values"]["step"],
                            value=mp["value"]
                        )
                    elif t == "text":
                        p = SettingsTextWidget(
                            title=mp["title"],
                            desc=mp["desc"],
                            value=mp["value"]
                        )
                    elif t == "combo":
                        dat = mp["values"].split(",")
                        p = SettingsComboBox(
                            title=mp["title"],
                            desc=mp["desc"],
                            data=dat,
                            value=mp["value"]
                        )
                    self.changed.connect(p.changed)
                    tab = self.ui.settings.widget(ind)
                    base = tab.findChildren(QtGui.QVBoxLayout, "base")
                    print(base)
                    base[0].addWidget(p)

    def remove_menu_point(self, section, name):
        for ind in range(self.ui.settings.count()):
            if self.ui.settings.tabText(ind) == section:
                pass
                # TODO
                #self.ui.settings.widget(ind).


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
