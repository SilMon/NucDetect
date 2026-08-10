import math
import multiprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import re
import shutil
import sys
import threading
import time
import traceback
import warnings
# TODO Remove before final release
# Ensure the project root (parent of this file's directory) is importable,
# so this file can be run directly regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from threading import Thread
from typing import Union, Dict, Iterable, List, Tuple, Any, Callable

# --- Import order below is load-bearing: TensorFlow MUST be imported before PyQt5 ---------
# On Windows, loading Qt's DLLs first exhausts the process' static TLS budget. TensorFlow's
# native runtime is then unable to initialise and the import fails with:
#     ImportError: DLL load failed while importing _pywrap_tensorflow_internal
#     _pywrap_tensorflow_common.dll: INITIALIZATION FAILED (0x45A / ERROR_DLL_INIT_FAILED)
# The accompanying diagnostic blames AVX2/the Visual C++ Redistributable; both are red
# herrings. Importing TensorFlow first claims its TLS slots and the conflict disappears.
# Do NOT move this below the PyQt5 imports, and do not let an import sorter reorder it.
import tensorflow  # noqa: F401  (unused by name; imported solely to fix DLL load order)

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
from core.logging_config import configure_logging, get_logger, init_worker_logging, log_messages
from core.progress import ProgressReporter, stage_bounds, ELLIPSE, DATABASE, TABLE
from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler
from core.database.connections import Connector, Requester, Inserter
from gui.definitions.icons import Icon, Color
from core.detector_modules.ImageLoader import ImageLoader
from gui.dialogs.data import Editor, ExperimentDialog, StatisticsDialog, DataExportDialog
from gui.dialogs.selection import ExperimentSelectionDialog
from gui.dialogs.settings import AnalysisSettingsDialog, SettingsDialog
from gui import Paths as gpaths
from gui import Util
PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, False)
PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, False)
pg.setConfigOptions(imageAxisOrder='row-major')
# Reference to the main window, needed to report errors of worker threads. Set by main()
_MAIN_WINDOW = None
LOGGER = get_logger(__name__)
# If set, a ui operation running off the GUI thread raises instead of only being logged. Meant for
# tests and debug runs -- in production a wrong thread should not turn into a hard crash by itself
STRICT_THREAD_AFFINITY = os.environ.get("NUCDETECT_STRICT_THREAD_AFFINITY", "") == "1"
# Steps the progress bar is driven at during an analysis. _set_progress truncates the bar value to
# an int, so the maximum of 100 used elsewhere is too coarse for the sub-stage emits inside nucleus
# extraction -- several of them would land on the same integer and the bar would still look stuck
PRG_RESOLUTION = 1000
# Role the precomputed sort key of a result table cell is stored under
SORT_ROLE = Qt.UserRole
# The result table holds one row per CHANNEL per nucleus. These two columns are the only ones whose
# value belongs to the channel rather than to the nucleus -- Requester.get_table_data_for_image
# builds one nucleus-level row and appends (channel, focus count) to a copy of it per channel, so
# everything else, Co-Localization included, is computed once per nucleus and merely repeated.
# Columns are classified by NAME and never by index: the experiment view inserts a "Group" column at
# index 2, which shifts every index after it.
CHANNEL_LEVEL_COLUMNS = frozenset({"Channel", "Foci"})
# The column whose repeated value identifies a nucleus, used to decide which rows may be merged
NUCLEUS_KEY_COLUMN = "ROI ID"
# Splits a text into its digit and non-digit runs
_DIGIT_RUN = re.compile(r"(\d+)")
# Sort keys are tuples of uniform (kind, number, text) triples. The uniform shape is what keeps a
# numeric run from ever being compared against a text run, which would raise a TypeError
_NUMERIC_PART = 0
_TEXT_PART = 1
# The Detector a worker process analyses with. One per process, built by _init_worker and reused for
# every image that process handles -- see _analyse_in_worker for why this is a module global
_WORKER_DETECTOR = None


def _init_worker() -> None:
    """
    Function to prepare a batch-analysis worker process

    Installed as the ProcessPoolExecutor initializer, so it runs exactly once per process. It does
    two things, both of which have to happen there rather than per task:

    * silences logging in this process -- the parent owns the log file and worker messages come back
      with the result instead (see core.logging_config.init_worker_logging);
    * builds the one Detector this process will use.

    :return: None
    """
    init_worker_logging()
    global _WORKER_DETECTOR
    _WORKER_DETECTOR = Detector()


def _analyse_in_worker(task: Tuple[str, Dict, bool]) -> Dict:
    """
    Function to analyse one image inside a worker process

    A module-level function taking a single tuple, which is what makes this arrangement work at all:
    passing a *bound* method to ``map`` pickles the object it is bound to, so the previous
    ``e.map(self.detector.analyse_image, ...)`` shipped a whole Detector across the process boundary
    once per image. A module-level function pickles as a name, so nothing but the arguments travels.

    Reusing one Detector for every task is safe because ``analyse_image`` already leaves no state
    behind: it calls ``release_transient_state`` at both ends, and clears its message buffer when
    ``save_log`` is False. Those two were written for the per-task copy and are what make a
    process-lifetime instance equivalent.

    :param task: The image path, the analysis settings, and whether the worker writes the log itself
    :return: The analysis result, as returned by Detector.analyse_image
    """
    path, settings, save_log = task
    return _WORKER_DETECTOR.analyse_image(path, settings, save_log)


def create_sort_key(text: Union[str, None]) -> Tuple[Tuple[int, float, str], ...]:
    """
    Function to create the key a result table cell is sorted by.

    Every cell of the result table holds a preformatted string, so comparing the displayed text
    would order "100.00" before "9.00" and the group "10 mikM" before "5 mikM". A cell that is a
    number as a whole -- including decimals and signs -- therefore compares numerically, every
    other cell compares run by run: digit runs numerically, the rest case-insensitively.

    :param text: The displayed text of the cell
    :return: The sort key, comparable against every other key created by this function
    """
    text = str(text) if text is not None else ""
    number = _to_finite_float(text)
    if number is not None:
        key = [(_NUMERIC_PART, number, "")]
    else:
        key = []
        for part in _DIGIT_RUN.split(text):
            if not part:
                continue
            if part.isdigit():
                key.append((_NUMERIC_PART, float(part), ""))
            else:
                key.append((_TEXT_PART, 0.0, part.casefold()))
    # Last resort tiebreak. Without it "img_1" and "img_01" -- or "5.0" and "5.00" -- share a key,
    # and since the sorting is stable, equal keys keep their source order in BOTH directions, i.e.
    # reversing the sort would not reverse those rows
    key.append((_TEXT_PART, 0.0, text.casefold()))
    return tuple(key)


def _to_finite_float(text: str) -> Union[float, None]:
    """
    Function to convert the given text to a float, if it describes one in full

    :param text: The text to convert
    :return: The converted number, None if the text is not a finite number
    """
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    # nan compares False against everything, which would silently destroy the ordering. Neither it
    # nor inf are values this table can hold, so both are sorted as text
    return number if math.isfinite(number) else None


class NucDetect(QMainWindow):
    """
    Created on 11.02.2019
    @author: Romano Weiss
    """
    prg_signal = pyqtSignal(str, float, float, str)
    selec_signal = pyqtSignal(bool)
    # Wired to add_item_to_list in _connect_signals but currently emitted by nothing -- kept as the
    # thread-safe entry point for adding an image from a worker, which is the only correct way to do
    # it. Signature matches the slot; emit it rather than calling add_item_to_list off the GUI thread
    add_signal = pyqtSignal(str)
    # Signals used to report and recover from errors inside worker threads
    err_signal = pyqtSignal(str, str)
    enable_signal = pyqtSignal(bool)
    # Signals used to apply results computed in a worker thread to the ui. Qt requires all widget
    # and model access to happen on the GUI thread, so workers compute plain data and hand it over
    # here instead of touching the models themselves
    table_signal = pyqtSignal(list, list)
    row_signal = pyqtSignal(list)
    status_signal = pyqtSignal(bool)
    # Labels are kept SHORT on purpose. The table has 13 columns, and a header section only shows
    # its sort indicator if the label leaves room for it -- measured before this was shortened,
    # every one of the 13 sections was ~50 px wide against labels needing 52-238 px, so Qt elided
    # every label and clipped every arrow. The user could sort but had no way to see that they had.
    STANDARD_TABLE_HEADER = ["Image Name", "Image ID",
                             "ROI ID", "Center Y",
                             "Center X", "Area [px]", "Ellipt. [%]",
                             "Angle [°]", "Maj. Axis", "Min. Axis",
                             "Co-Loc. [%]", "Channel", "Foci"]

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
        # Highest bar fraction shown so far during the running analysis, or None when no analysis
        # is in progress. See _set_progress for why the monotonicity clamp is opt-in
        self._prg_floor: Union[float, None] = None
        # Timer which polls running data exports. Instance attribute on purpose: as a class
        # attribute it was shared between windows and accumulated one connected slot per export
        self.check_timer = QTimer()
        self.check_timer.setInterval(500)
        # The export dialog whose threads are currently awaited, and when the wait started
        self.export_dialog = None
        self.export_start = None
        self._closing = False
        # Setup UI
        self._setup_ui()
        self.showMaximized()

    @staticmethod
    def create_required_dirs() -> None:
        """
        Method to create the working dirs of this program

        :return: None
        """
        # The directories themselves are created by the module that declares them, so the core
        # works without the GUI. What stays here is the part that is genuinely GUI start-up:
        # seeding a brand-new images folder with the demo image
        created = gpaths.ensure_directories()
        if gpaths.images_path in created:
            shutil.copy2(gpaths.demo_image, os.path.join(gpaths.images_path, "demo.tif"))

    def load_settings(self) -> Dict:
        """
        Method to load the saved Settings

        :return: The loaded settings, keyed by setting name
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
            return str(value).strip().lower() in ("1", "true", "yes")
        else:
            return value

    def closeEvent(self, event) -> None:
        """
        Will be called if the program window closes

        :param event: The closing event
        :return: None
        """
        # Guard against re-entry: on_close() pumps the event loop while waiting for exports
        if self._closing:
            event.ignore()
            return
        self._closing = True
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
        self.ui = uic.loadUi(gpaths.ui_main, self)
        with open(os.path.join(gpaths.css_dir, "main.css"), "r", encoding="utf-8") as f:
            self.ui.setStyleSheet(f.read())
        # General Window Initialization
        self.setWindowTitle("NucDetect - Focus Analysis Software")
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.ui.lbl_logo.setPixmap(QPixmap(os.path.join(gpaths.logo_dir, "banner.png")))

    def _initialize_image_list(self) -> None:
        """
        Method to initialize the image list

        :return: None
        """
        self.add_images_from_folder(gpaths.images_path)
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
        # NOT QHeaderView.Stretch. Stretch divides the width evenly over all 13 columns, which left
        # every section around 50 px -- narrower than every single header label, so Qt elided the
        # text and clipped the sort indicator off the end. Sorting worked and looked like it did
        # not. Sizing to content gives each header the room it needs and lets the table scroll
        # sideways instead, with the last column taking up any slack
        # Interactive rather than ResizeToContents: the widths are set once per fill (see
        # _apply_result_table) instead of being recomputed on every data change, and the user can
        # still drag a column, which ResizeToContents forbids outright
        header = self.ui.table_results.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        # Row banding is the second half of the same problem: without it a reordered table is hard
        # to read as reordered, because nothing marks where one row ends and the next begins
        self.ui.table_results.setAlternatingRowColors(True)
        # Spans are VIEW state and Qt never moves them when the sorting reorders the rows, so they
        # have to be rebuilt on every sort. QTableView connects its own sort slot to this signal
        # first -- setSortingEnabled runs from the .ui, before this method -- and Qt calls slots in
        # connection order, so by the time this one runs the rows are already in their new order
        header.sortIndicatorChanged.connect(self._update_result_table_spans)

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
        self.ui.btn_about.clicked.connect(self.show_about_window)

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
        self.ui.btn_about.setIcon(Icon.get_icon("QUESTION"))

    def _connect_signals(self) -> None:
        """
        Method to connect the used signals

        :return: None
        """
        # Create signal for thread-safe gui updates
        self.prg_signal.connect(self._set_progress)
        self.selec_signal.connect(self._select_next_image)
        self.add_signal.connect(self.add_item_to_list)
        self.err_signal.connect(self._show_worker_error)
        self.enable_signal.connect(self._set_ui_enabled)
        self.table_signal.connect(self._apply_result_table)
        self.row_signal.connect(self._append_result_row)
        self.status_signal.connect(self._apply_item_status)
        # Connected exactly once, here. The parameters the slot needs live on the instance, so
        # save_results() does not have to (re-)connect a partial on every export
        self.check_timer.timeout.connect(self.check_for_running_threads)

    def _run_guarded(self, worker: Callable, *args) -> None:
        """
        Method to run a worker thread body, ensuring that errors are reported and that the ui is
        always re-enabled. Meant to be used as target of every thread started by this window

        :param worker: The method to execute in this thread
        :param args: The arguments to pass to the worker
        :return: None
        """
        try:
            worker(*args)
        except Exception:
            # sys.excepthook does not cover worker threads, so the error has to be routed to the
            # main thread manually
            self.err_signal.emit(worker.__name__, traceback.format_exc())
        finally:
            # An analysis that raised never reaches its own disarm, and a still-armed clamp would
            # silently hold every later progress bar at the value the failed run had reached
            self._prg_floor = None
            # Re-enable the ui via signal, since it must not be touched from this thread
            self.enable_signal.emit(True)

    def _show_worker_error(self, source: str, text: str) -> None:
        """
        Method to inform the user about an error which occured inside a worker thread. Connected to
        err_signal, thus always executed on the main thread

        :param source: The name of the worker the error originated from
        :param text: The formatted traceback of the error
        :return: None
        """
        LOGGER.error("Error during %s:\n%s", source, text)
        self.prg_signal.emit(f"Error during {source} -- Program ready", 100, 100, "")
        show_error_message(title="An error occured during execution",
                           info=f"An error occured at {time.strftime('%Y-%m-%d, %H:%M:%S')} "
                                f"during {source}",
                           text=f"During the execution of the program, following error occured:\n{text}")

    def _set_ui_enabled(self, state: bool) -> None:
        """
        Method to set the state of the buttons and the image list. Connected to enable_signal, thus
        always executed on the main thread

        :param state: The state to set the ui into
        :return: None
        """
        self.enable_buttons(state)
        self.ui.list_images.setEnabled(state)

    def _assert_main_thread(self, operation: str) -> None:
        """
        Method to verify that a ui operation is running on the GUI thread

        Touching a widget or a model from a worker thread does not fail immediately -- it corrupts
        Qt's internal state and surfaces later as an unreproducible freeze or crash. This turns
        that into a loud, deterministic failure at the point of the violation.

        :param operation: Name of the operation, used in the message
        :return: None
        :raises RuntimeError: If called off the GUI thread and strict checking is enabled
        """
        if QtCore.QThread.currentThread() is self.thread():
            return
        msg = (f"{operation} was called from thread "
               f"'{QtCore.QThread.currentThread().objectName() or threading.current_thread().name}'"
               f" instead of the GUI thread -- route it through a signal")
        LOGGER.critical(msg)
        if STRICT_THREAD_AFFINITY:
            raise RuntimeError(msg)

    def _apply_result_table(self, header: List[str], rows: List[List[str]]) -> None:
        """
        Method to fill the result table with rows prepared by a worker thread. Connected to
        table_signal, thus always executed on the main thread

        :param header: The column labels of the table
        :param rows: The prepared rows, as lists of cell texts
        :return: None
        """
        self._assert_main_thread("_apply_result_table")
        self.res_table_model.setRowCount(0)
        self.create_table_rows(rows)
        if rows:
            self.res_table_model.setColumnCount(len(rows[0]))
        # Set header of table
        self.res_table_model.setHorizontalHeaderLabels(header)
        # Size the columns to what they now hold. Done here rather than by a ResizeToContents
        # resize mode so the cost is paid once per fill instead of on every data change, and so
        # the widths stay draggable afterwards
        self.ui.table_results.resizeColumnsToContents()
        # setRowCount(0) above drops every span, so they are rebuilt for the rows just added
        self._update_result_table_spans()
        data = [header]
        data.extend(rows)
        self.data = data

    def _update_result_table_spans(self, *_) -> None:
        """
        Method to merge the result-table cells that repeat within one nucleus

        The table holds one row per channel per nucleus, so a nucleus with two focus channels
        occupies two rows and repeats all of its own values across both. Ten rows then read as ten
        nuclei when there are five. Merging the repeated cells makes the real count visible: five
        merged blocks are five nuclei.

        Which cells may be merged depends on how the rows are currently ordered, because merging is
        only honest while the rows it merges are adjacent:

        * ordered by a nucleus-level column, every nucleus keeps its rows together (they share the
          value and the sort is stable), so all nucleus-level columns merge across them;
        * ordered by a channel-level column, ``TableFilterModel.lessThan`` groups the rows into one
          block per channel, and each block holds exactly one row per nucleus -- nothing repeats
          except the channel itself, which is merged into a block heading instead.

        **The choice is made from the row order itself, never from the sort column.** Those two
        disagree whenever the header changes shape under an active sort: switching between the
        single-image and experiment views changes the column count, which leaves ``sortColumn()``
        at -1 while the rows keep the order the old sort gave them. Reading the sort column there
        selects nucleus merging for channel-grouped rows and silently merges nothing.

        Takes and ignores the arguments of ``sortIndicatorChanged``.

        :return: None
        """
        self._assert_main_thread("_update_result_table_spans")
        table = self.ui.table_results
        model = table.model()
        table.clearSpans()
        if model is None or not model.rowCount():
            return
        nucleus_runs = self._get_result_table_runs(model, NUCLEUS_KEY_COLUMN)
        if any(length > 1 for _, length in nucleus_runs):
            runs = nucleus_runs
            merged = [column for column in range(model.columnCount())
                      if model.column_name(column) not in CHANNEL_LEVEL_COLUMNS]
        else:
            # One row per nucleus per block: only the channel repeats, as a block heading
            runs = self._get_result_table_runs(model, "Channel")
            merged = [model.column_index("Channel")]
        # An order that groups by neither -- which a stale sort can produce -- merges nothing.
        # Spanning unrelated neighbours would state a grouping that is not there
        if not merged or merged[0] < 0:
            return
        # One repaint for the whole rebuild rather than one per setSpan
        table.setUpdatesEnabled(False)
        try:
            for start, length in runs:
                if length > 1:
                    for column in merged:
                        table.setSpan(start, column, length, 1)
        finally:
            table.setUpdatesEnabled(True)

    @staticmethod
    def _get_result_table_runs(model, column_name: str) -> List[Tuple[int, int]]:
        """
        Method to find the runs of consecutive rows sharing a value in the given column

        :param model: The result table's proxy model
        :param column_name: The header label of the column to group by
        :return: The runs, as (first row, number of rows). Empty if the table has no such column
        """
        column = model.column_index(column_name)
        if column < 0:
            return []
        values = [model.data(model.index(row, column), Qt.DisplayRole)
                  for row in range(model.rowCount())]
        runs = []
        start = 0
        for row in range(1, len(values) + 1):
            if row == len(values) or values[row] != values[start]:
                runs.append((start, row - start))
                start = row
        return runs

    def _append_result_row(self, cells: List[str]) -> None:
        """
        Method to append a single row to the result table. Connected to row_signal, thus always
        executed on the main thread

        QStandardItems belong to the thread of the model they are added to, so they are constructed
        here and not by the worker that produced the cell texts.

        :param cells: The text each cell of the row should contain
        :return: None
        """
        self._assert_main_thread("_append_result_row")
        items = []
        for cell in cells:
            item = QStandardItem(cell)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            # Derive the sort key once here, not on every comparison the sorting performs
            item.setData(create_sort_key(cell), SORT_ROLE)
            items.append(item)
        self.res_table_model.appendRow(items)
        self.ui.table_results.scrollToBottom()

    def _apply_item_status(self, all_items: bool) -> None:
        """
        Method to update the image list items after an analysis. Connected to status_signal, thus
        always executed on the main thread

        :param all_items: If true, every item of the list is checked, else only the selected one
        :return: None
        """
        self._assert_main_thread("_apply_item_status")
        if all_items:
            self.check_all_item_statuses()
        else:
            self.reflect_item_status_changes()

    def reload(self) -> None:
        """
        Method to reload the images folder

        :return: None
        """
        self.loaded_files = []
        self.add_images_from_folder(gpaths.images_path, reload=True)
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
        selected = self.ui.list_images.selectionModel().selectedIndexes()
        for index in selected:
            self.cur_img = self.img_list_model.get_item_at_index(index.row()).data()
        if not selected:
            # The list is ExtendedSelection, so Ctrl-clicking the selected row deselects it and
            # fires this handler with an empty selection. The loop above then does not run, and
            # without this cur_img kept pointing at the deselected image while btn_analyse stayed
            # enabled -- Analyse would have run against an image the user had just deselected.
            #
            # This does NOT cover clearing or reloading the list: those reset the model, and Qt
            # drops the selection on a model reset without emitting selectionChanged, so this
            # handler is never called for them.
            self.cur_img = None
        # Analysis needs a selected image, so the button follows that state. btn_analyse starts
        # disabled in the .ui, as btn_modify already did: before this, a freshly launched window
        # offered an enabled Analyse button with nothing loaded, and clicking it opened the settings
        # dialog before failing on cur_img
        self.ui.btn_analyse.setEnabled(self.cur_img is not None)
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
        with open(os.path.join(gpaths.css_dir, "messagebox.css"), "r", encoding="utf-8") as f:
            msg.setStyleSheet(f.read())
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
        load_thread = threading.Thread(target=self._run_guarded,
                                       args=(self._load_saved_data, experiment),
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
        # Re-enable buttons and list. Runs on this worker thread, so it has to go through the
        # signal -- the two duplicated in-body enable calls this replaces did not
        self.enable_signal.emit(True)
        self.prg_signal.emit(f"Data loaded from database for {self.cur_img['file_name']}",
                             100, 100, "")

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
        file_name, _ = QFileDialog.getOpenFileName(self, "Load images..", gpaths.images_path,
                                                   "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)",
                                                   options=options)
        if file_name:
            self.add_item_to_list(file_name)

    def add_images_from_folder(self, url: str, reload: bool = False) -> None:
        """
        Method to load a whole folder of images

        :param url: The path of the folder
        :param reload: Indicator if a reload occurs
        :return: None
        """
        paths = []
        loaded_set = set(self.loaded_files)
        for t in os.walk(url):
            tpaths = [os.path.join(t[0], x) for x in t[2]]
            paths.extend([x for x in tpaths if x not in loaded_set])
        # If no images where found, open a file dialog to add images
        if not paths:
            files = str(QFileDialog.getExistingDirectory(self, "Select Directory to load images from"))
            # Walk the folder to find all files inside it
            for t in os.walk(files):
                tpaths = [os.path.join(t[0], x) for x in t[2]]
                paths.extend([x for x in tpaths if x not in loaded_set])
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
        # Passed by keyword: the signature takes eleven same-looking values in a row, which is how
        # the minute slot silently received the day for as long as it did
        self.inserter.add_new_image(md5,
                                    year=d["year"], month=d["month"], day=d["day"],
                                    hour=d["hour"], minute=d["minute"],
                                    channels=d["channels"], width=d["width"], height=d["height"],
                                    xres=d["x_res"], yres=d["y_res"], res_unit=d["unit"])
        self.inserter.register_image_filename(path)
        self.connector.commit_changes()

    def add_item_to_list(self, path: str) -> None:
        """
        Utility method to add an image to the image list

        The model creates the list item itself, so only the path is needed here

        :param path: The path of the image to add
        :return: None
        """
        if not path:
            return
        # Let the model own the insertion, so it can emit the required Qt signals
        if not self.img_list_model.add_path(path):
            return
        self.loaded_files.append(path)
        self.add_image_information_to_database(path)

    def remove_image_from_list(self) -> None:
        """
        Method to remove a loaded image from the file list.

        :return: None
        """
        cur_ind = self.ui.list_images.currentIndex()
        # Without a selection currentIndex() is invalid (row -1), which would otherwise
        # resolve to the *last* image and delete the wrong entry
        if not cur_ind.isValid():
            return
        path = self.img_list_model.get_item_at_index(cur_ind.row()).data()["path"]
        if self.img_list_model.removeRow(cur_ind.row()):
            self.loaded_files.remove(path)

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
        # The emit above selects the first image, but there may not be one -- an empty list leaves
        # cur_img as None. Checked before the settings dialog opens, so the user is not asked to
        # configure an analysis that cannot run. The button is also disabled in this state; this
        # guard is the safeguard for the two staying out of step
        if not self.cur_img:
            self.prg_signal.emit("No image selected -- load and select an image first",
                                 0, 100, "")
            return
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
        thread = Thread(target=self._run_guarded,
                        args=(self.analyze_image,
                              self.cur_img["path"],
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
        self.enable_signal.emit(False)
        start = time.time()
        # Weights for the three stages that run here, after the Detector has returned. The Detector
        # gets the same table for the stages it owns, so the two halves cannot drift apart
        bounds = stage_bounds(analysis_settings["analysis_settings"]["method"])
        # Arm the monotonicity clamp for the duration of this analysis
        self._prg_floor = 0.0
        reporter = ProgressReporter(self._report_analysis_progress)
        reporter(0.0, "Starting analysis")
        data = self.detector.analyse_image(path, settings=analysis_settings, save_log=True,
                                           progress=reporter)
        self.roi_cache = data["handler"]
        reporter.sub(*bounds[ELLIPSE])(0.0, "Calculating ellipse parameters")
        for roi in self.roi_cache:
            if roi.main:
                roi.calculate_ellipse_parameters()
        reporter.sub(*bounds[DATABASE])(0.0, "Writing results to database")
        self.save_rois_to_database(data)
        reporter.sub(*bounds[TABLE])(0.0, "Creating result table")
        self.create_result_table_from_list(self.roi_cache)
        # Only now is the analysis actually over. The previous version announced completion before
        # building the result table, so the bar read 100 % while work was still running
        self._prg_floor = None
        self.prg_signal.emit(message.format(f"{time.time() - start:.2f} secs"),
                             percent, maxi, "")
        self.enable_signal.emit(True)
        self.status_signal.emit(False)

    def _report_analysis_progress(self, fraction: float, message: str) -> None:
        """
        Method to forward a progress fraction from the analysis onto the progress bar

        Called from the analysis thread by ProgressReporter, hence the signal rather than a direct
        widget call. Single-image analysis runs the Detector in-process, which is what allows a
        plain callback here; batch analysis runs it in a ProcessPoolExecutor, where Qt signals
        cannot cross the process boundary and an IPC queue would be needed instead.

        The bar is driven at PRG_RESOLUTION steps rather than 100 because `_set_progress` truncates
        the value to an int: at a maximum of 100 the sub-stage emits inside nucleus extraction would
        collapse onto the same handful of integers and the bar would still look frozen.

        :param fraction: Progress through the whole analysis, 0..1
        :param message: The text to show above the bar
        :return: None
        """
        self.prg_signal.emit(message, fraction * PRG_RESOLUTION, PRG_RESOLUTION, "")

    def analyze_all(self) -> None:
        """
        Method to perform concurrent batch analysis of registered images

        :return: None
        """
        self.enable_buttons(False)
        self.ui.list_images.setEnabled(False)
        # Get settings for this analysis
        settings = self.show_analysis_settings_dialog(show_redo_option=True)
        if not settings:
            return
        thread = Thread(target=self._run_guarded, args=(self._analyze_all, settings))
        thread.start()

    def _analyze_all(self, settings: Dict[str, Union[int, float, str, Iterable]], batch_size: int = 10) -> None:
        """
        Method to perform concurrent batch analysis of registered images

        :param settings: The settings for this analysis, e.g. channel names, active channels ect.
        :param batch_size: The number of images that are loaded parallel
        :return: None
        """
        start_time = time.time()
        # Use a quarter of the available cores, but never less than 1
        workers = max(1, round(multiprocessing.cpu_count() * 0.25))
        # The initializer runs once per process: it silences logging there (the parent owns the log
        # file) and builds the single Detector that process analyses with
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker) as e:
            # Clear the table and set its header on the GUI thread. The rows follow one by one via
            # row_signal as the results come in, so no rows are passed here
            self.table_signal.emit(["Image Name", "Image Hash", "Number of Nuclei",
                                    "Number of Foci", "Foci per Nucleus"], [])
            logstate = settings["analysis_settings"]["logging"]
            settings["analysis_settings"]["logging"] = False
            self.prg_signal.emit("Starting multi image analysis", 0, 100, "")
            paths = []
            for image in self.loaded_files:
                # Get md5 hash of file
                md5 = ImageLoader.calculate_image_id(image)
                if not self.requester.check_if_image_was_analysed(md5) or settings["re-analyse"]:
                    paths.append(image)
            LOGGER.info("Batch analysis of %d images", len(paths))
            maxi = len(paths)
            # Number of batches, rounded up -- the last one is short unless the count divides evenly
            total_batches = math.ceil(maxi / batch_size) if batch_size > 0 else 0
            # Counts images actually finished. Kept separate from the 1-based display counter
            # below: reusing one variable for both is what made the ETA undercount by one and go
            # negative on the final batch of every run
            done = 0
            # A plain slice loop. The previous start/stop/step arithmetic made the first batch
            # batch_size + 1 images long, and executed once even when there was nothing to analyse
            for batch_start in range(0, maxi, batch_size):
                s2 = time.time()
                tpaths = paths[batch_start:batch_start + batch_size]
                t_setts = [settings for _ in range(len(tpaths))]
                # save_log=False: the workers hand their messages back with the result instead of
                # appending to the log file themselves, so the entries stay in image order and
                # several processes never write the file at the same time
                t_savelog = [False for _ in range(len(tpaths))]
                # One tuple per image against a module-level function: the Detector stays in the
                # worker process instead of being pickled with every task
                res = e.map(_analyse_in_worker, zip(tpaths, t_setts, t_savelog))
                for r in res:
                    self.prg_signal.emit(f"Analysed images: {done + 1}/{maxi}",
                                         done + 1, maxi, "")
                    # Replay the log of the worker that analysed this image
                    log_messages(r.get("log", ()))
                    self.save_rois_to_database(r, all_=True)
                    # Get the image hash and file name
                    name = self.requester.get_image_filename(r["handler"].ident)
                    mnum = len([x for x in r["handler"] if x.main])
                    fnum = len([x for x in r["handler"] if not x.main])
                    fpn = ((fnum / mnum) if mnum > 0 else 0)
                    # Only the cell texts are produced here -- the QStandardItems are built by the
                    # slot, because model items belong to the thread of the model they enter
                    self.row_signal.emit([name, r["handler"].ident,
                                          str(mnum), str(fnum), f"{fpn:.2f}"])
                    done += 1
                images_left = maxi - done
                # Not int(): truncating to whole seconds reports an ETA of zero for anything
                # faster than a second per image
                time_per_image = (time.time() - start_time) / done if done else 0
                eta = int(images_left * time_per_image)
                h = eta // 3600
                m = eta % 3600 // 60
                s = eta % 3600 % 60
                cur_batch = batch_start // batch_size + 1
                msg = f"Analysed batch {cur_batch: 02d}/{total_batches: 02d} in {time.time() - s2: 09.3f} secs\t\t"\
                      f"Total: {time.time() - start_time: 09.3f} secs\t\t"\
                      f"ETA: {h:02d}h:{m:02d}m:{s:02d}s"
                LOGGER.info(msg)
            self.enable_signal.emit(True)
            settings["analysis_settings"]["logging"] = logstate
            self.prg_signal.emit("Analysis finished -- Program ready",
                                 100,
                                 100, "")
            self.selec_signal.emit(True)
            # Change the status of list items to reflect that they were analysed. The loop that
            # used to precede this set "analysed" on every item from this thread; it was redundant,
            # since check_all_item_statuses re-derives the flag from the database and overwrites it
            self.status_signal.emit(True)
        LOGGER.info("Total analysis time: %.3f secs", time.time() - start_time)

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
        try:
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
            # Only commit once all writes succeeded, so a failed save doesn't persist a partial state
            ins.commit()
            req.commit()
        finally:
            # Always release both connections, even if an error interrupted the writes above
            ins.connector.close_connection()
            req.connector.close_connection()
        if not all_:
            LOGGER.info("ROI saved to database")

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
            # TODO
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
        LOGGER.info("Loaded %d roi of image %s from database", len(rois), self.cur_img["file_name"])
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

        Safe to call from a worker thread: the database queries stay on the calling thread and only
        the finished rows are handed to the GUI thread via table_signal. Emitting from the GUI
        thread itself keeps the old synchronous behaviour, since Qt connects a same-thread emit
        directly.

        :param handler: The handler containing the rois
        :param experiment: The experiment to load
        :return: None
        """
        self.prg_signal.emit(f"Create Result Table",
                             0, 100, "")
        # Create header
        header = copy(NucDetect.STANDARD_TABLE_HEADER)
        if experiment:
            header.insert(2, "Group")
        rows = self.prepare_main_table_rows(experiment)
        self.table_signal.emit(header, rows)

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
            # Derive the sort key once here, not on every comparison the sorting performs
            item.setData(create_sort_key(cell), SORT_ROLE)
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
        self._assert_main_thread("enable_buttons")
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

        Note: nothing currently asks for the *next* image -- both selec_signal emits pass
        first=True -- so the advance path below is dead in production. Before wiring it up, settle
        what "next" means on the last image: it wraps to the first here only because that is what
        the code did before, which is not the same as it being right.

        :param first: Indicates if the first image in the list should be selected
        :return: None
        """
        max_ind = self.img_list_model.rowCount()
        cur_ind = self.ui.list_images.currentIndex()
        # An empty list has nothing to select, and index(0, 0) would be invalid
        if not max_ind:
            return
        # rowCount() is a count, currentIndex().row() an index. Comparing them directly asked the
        # model for index(rowCount, 0) while on the last row -- one past the end. Qt does not raise
        # for that, it returns an invalid index, and select()/setCurrentIndex() read an invalid
        # index as "no item", so the selection was silently cleared instead of wrapping round
        wrap = first or not cur_ind.isValid() or cur_ind.row() >= max_ind - 1
        nex = self.img_list_model.index(0 if wrap else cur_ind.row() + 1, 0)
        self.ui.list_images.selectionModel().select(nex, QItemSelectionModel.Select)
        self.ui.list_images.setCurrentIndex(nex)

    def _set_progress(self, text: str, progress: Union[int, float], maxi: Union[int, float], symbol: str) -> None:
        """
        Method to control the progress bar. Should not be called directly, emit the progress signal instead

        While an analysis is running the bar is held monotonic: `_prg_floor` is armed by
        analyze_image and the value is never allowed to fall below the highest one already shown.
        The stage weights driving it are measured, but measured on one image on one machine, and
        stage shares move with nucleus and foci count -- so the estimate *will* be wrong on some
        images. The clamp is what makes a wrong estimate degrade into "the bar pauses" rather than
        "the bar goes backwards", which is what users actually reported.

        The clamp is opt-in rather than global because this method also serves loading, export and
        ROI progress, which legitimately restart at low values without passing through zero.

        :param text: The text to show above the bar
        :param progress: The value of the bar
        :param maxi: The max value of the bar
        :param symbol: The symbol printed after the displayed values
        :return: None
        """
        if self._prg_floor is not None:
            progress = max(progress, self._prg_floor * maxi)
            self._prg_floor = progress / maxi if maxi else 0.0
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
        if code == QDialog.Accepted and dial.threads:
            # Lock the ui until the export threads are done. Without this the user can close the
            # program while an export is still writing, which truncates the file
            self.export_dialog = dial
            self.export_start = time.time()
            self._set_ui_enabled(False)
            self.check_timer.start()

    def check_for_running_threads(self) -> None:
        """
        Method to keep the program locked until the running data exports are finished. Connected to
        check_timer, thus executed on the main thread every 500 ms while an export runs

        :return: None
        """
        dial = self.export_dialog
        if dial is None:
            self.check_timer.stop()
            return
        runtime = time.time() - self.export_start
        running = [x for x in dial.threads if x.is_alive()]
        if running:
            self.prg_signal.emit(f"Exporting data, please wait... "
                                 f"({len(dial.threads) - len(running)}/{len(dial.threads)} done, "
                                 f"{runtime:.1f} secs)", 0, 100, "")
            return
        # All exports finished. Stopping the timer is not optional: it would otherwise keep firing
        # for the rest of the session, re-enabling the buttons twice a second
        self.check_timer.stop()
        errors = list(dial.errors)
        self.export_dialog = None
        self.export_start = None
        self._set_ui_enabled(True)
        if errors:
            self.prg_signal.emit(f"Data export failed after {runtime:.1f} secs -- Program ready",
                                 100, 100, "")
            self._show_worker_error("data export", "\n".join(errors))
        else:
            self.prg_signal.emit(f"Data export finished in {runtime:.1f} secs -- Program ready",
                                 100, 100, "")

    def wait_for_exports(self, timeout: float = 30) -> bool:
        """
        Method to wait for still running data exports, called before the program shuts down

        The export threads are daemons, so without this wait the interpreter would kill them
        mid-write on exit and leave truncated result files behind

        :param timeout: The maximum time to wait in seconds
        :return: True if all exports finished, False if the wait timed out
        """
        self.check_timer.stop()
        dial = self.export_dialog
        if dial is None:
            return True
        deadline = time.time() + timeout
        running = [x for x in dial.threads if x.is_alive()]
        while running and time.time() < deadline:
            self.prg_signal.emit(f"Waiting for {len(running)} running export(s) to finish...",
                                 0, 100, "")
            # Keep the window painting while waiting. Re-entrancy is not a concern here: the ui
            # was disabled when the export started and _closing blocks a second close
            QtWidgets.QApplication.processEvents()
            running[0].join(timeout=0.1)
            running = [x for x in dial.threads if x.is_alive()]
        if running:
            LOGGER.warning("Shutdown with %d export(s) still running after %s secs "
                           "-- exported files may be incomplete", len(running), timeout)
            show_error_message(title="Data export unfinished",
                               info=f"{len(running)} export(s) did not finish within {timeout:.0f} "
                                    f"seconds",
                               text="The program is closing while data is still being exported.\n"
                                    "The affected files may be incomplete and should be exported "
                                    "again.")
            return False
        return True

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
            with open(os.path.join(gpaths.css_dir, "messagebox.css"), "r", encoding="utf-8") as f:
                msg.setStyleSheet(f.read())
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
        sett.initialize_from_file(os.path.join(gpaths.settings_path, "settings.json"))
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
        # btn_modify is disabled in the .ui until results exist, so this should be unreachable --
        # which is exactly why the guard is cheap insurance rather than dead weight. Without it the
        # method raises TypeError on a None cur_img
        if not self.cur_img:
            self.prg_signal.emit("No image selected -- nothing to modify", 0, 100, "")
            return
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

    def show_about_window(self) -> None:
        """
        Method to show the about message box

        :return: None
        """
        # Load the about text
        with open(gpaths.about_txt_path, "r", encoding="utf-8") as f:
            # Define new message box
            msg = QMessageBox()
            msg.setWindowIcon(Icon.get_icon("LOGO"))
            msg.setIcon(QMessageBox.Information)
            msg.setTextFormat(Qt.RichText)
            msg.setText(f.read())
            msg.setWindowTitle("About NucDetect")
            with open(os.path.join(gpaths.css_dir, "main.css"), "r", encoding="utf-8") as cf:
                msg.setStyleSheet(cf.read())
            msg.exec()

    def reflect_item_status_changes(self) -> None:
        """
        Method to change the image list items if the underlying image was analysed

        :return: None
        """
        self._assert_main_thread("reflect_item_status_changes")
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
        self._assert_main_thread("check_all_item_statuses")
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
        # Let running exports finish writing before the interpreter kills them
        self.wait_for_exports()
        # __init__ opens TWO connections -- self.connector, and self.req_connector backing
        # self.requester -- and only the first used to be closed here. They are closed
        # independently so that a failure on one cannot leak the other, and neither can stop the
        # window from closing: nothing useful remains to be done with a connection at this point
        for name, connector in (("connector", self.connector),
                                ("req_connector", self.req_connector)):
            try:
                connector.close_connection()
            except Exception:
                LOGGER.exception("Failed to close %s during shutdown", name)


class TableFilterModel(QSortFilterProxyModel):
    """
    Model used to sort the result table by value instead of by displayed text
    """

    def __init__(self, parent):
        super(TableFilterModel, self).__init__(parent)

    def column_name(self, column: int) -> str:
        """
        Method to get the header label of a source column

        :param column: The index of the column
        :return: The label, empty if there is no such column
        """
        source = self.sourceModel()
        if source is None or column < 0 or column >= source.columnCount():
            return ""
        return source.headerData(column, Qt.Horizontal, Qt.DisplayRole) or ""

    def column_index(self, name: str) -> int:
        """
        Method to find a source column by its header label

        :param name: The label to look for
        :return: The index of the column, -1 if the table does not have it
        """
        source = self.sourceModel()
        if source is None:
            return -1
        for column in range(source.columnCount()):
            if self.column_name(column) == name:
                return column
        return -1

    def lessThan(self, ind1, ind2):
        # A channel-level column describes one channel of a nucleus, so ordering by it alone
        # interleaves the channels of different nuclei -- the rows of one nucleus end up scattered
        # down the table. Sorting by the channel FIRST turns the table into one block per channel,
        # each holding exactly one row per nucleus, which is both readable and countable.
        sort_column = self.column_name(self.sortColumn())
        source = self.sourceModel()
        if source is not None and sort_column in CHANNEL_LEVEL_COLUMNS and sort_column != "Channel":
            channel = self.column_index("Channel")
            if channel >= 0:
                key1 = self.get_sort_key(source.index(ind1.row(), channel))
                key2 = self.get_sort_key(source.index(ind2.row(), channel))
                if key1 != key2:
                    # Qt applies the sort ORDER to whatever this returns, so a descending sort would
                    # reverse the blocks along with their contents. Undoing it here keeps the blocks
                    # in one order while the chosen column sorts both ways inside them
                    if self.sortOrder() == Qt.DescendingOrder:
                        return key1 > key2
                    return key1 < key2
        return self.get_sort_key(ind1) < self.get_sort_key(ind2)

    def get_sort_key(self, index: QModelIndex) -> Tuple[Tuple[int, float, str], ...]:
        """
        Method to get the sort key of the given source index

        :param index: The index of the cell to get the key for
        :return: The sort key stored for the cell, derived from its text if it has none
        """
        source = self.sourceModel()
        key = source.data(index, SORT_ROLE)
        # Rows created outside create_table_row carry no precomputed key
        if key is None:
            key = create_sort_key(source.data(index, Qt.DisplayRole))
        return key


class ImageListModel(QAbstractListModel):
    """
    Class to lazy load needed image list items
    """

    def __init__(self, parent=None, paths: List[str] = (), page_size: int = 30):
        """
        :param paths: The image paths that are the basis of the items
        """
        super().__init__(parent)
        self.page_size = page_size
        self.set_paths(paths)
        self._cache = {}

    def set_paths(self, paths: List[str]):
        # Store a copy: the model owns its path list, so callers mutating their own list
        # (e.g. NucDetect.loaded_files) cannot silently desynchronise the model behind its back
        self.beginResetModel()
        self._paths = list(paths)
        self._current_paths = self._paths
        self.current_index = 0
        self._cache = {}
        self.endResetModel()

    def clear(self) -> None:
        """
        Method to remove all paths and cached items from the model

        :return: None
        """
        self.beginResetModel()
        self._paths = []
        self._current_paths = self._paths
        self.current_index = 0
        self._cache = {}
        self.endResetModel()

    def add_path(self, path: str) -> bool:
        """
        Method to append a single path to the model

        :param path: The path to append
        :return: True if the path was added, False if it was already present
        """
        if path in self._paths:
            return False
        # Append only, never insert: _cache is keyed by absolute row index, so inserting
        # mid-list would make every cached item beyond it map to the wrong path
        if self.current_index == len(self._paths):
            # Everything is revealed, so reveal the new row as well
            row = self.current_index
            self.beginInsertRows(QModelIndex(), row, row)
            self._paths.append(path)
            self.current_index += 1
            self.endInsertRows()
        else:
            # Still paginating: only extend the backing list. Announcing an insert beyond
            # rowCount() would contradict what the view currently believes.
            self._paths.append(path)
        return True

    def removeRows(self, row: int, count: int = 1, parent: QModelIndex = QModelIndex()) -> bool:
        if parent.isValid() or count <= 0 or row < 0 or row + count > len(self._paths):
            return False
        self.beginRemoveRows(QModelIndex(), row, row + count - 1)
        del self._paths[row:row + count]
        # _cache is keyed by absolute row index, so every entry from `row` on now points at
        # the wrong path -> drop them and let them be recreated lazily
        self._cache = {k: v for k, v in self._cache.items() if k < row}
        self.current_index = max(0, self.current_index - count)
        self.endRemoveRows()
        return True

    def clear_data(self) -> None:
        """
        Method to clear the stored data

        :return: None
        """
        # Delegates to clear(): nesting beginRemoveRows (inside removeRows) within a
        # beginResetModel block would be an invalid Qt call sequence
        self.clear()

    def canFetchMore(self, parent: QModelIndex) -> bool:
        if parent.isValid():
            return False
        # current_index is a count of revealed items, not a page number
        return self.current_index < len(self._current_paths)

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

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        """
        Method to provide the data of the item at the given index for the given role

        Returns None, not 0, when there is nothing to provide. Qt queries this for *every* role --
        decoration, size hint, tooltip, check state and so on -- and None is how PyQt spells an
        invalid QVariant, i.e. "no value for this role". A 0 is a valid answer to most of those
        questions and would be acted on: a zero size hint, an unchecked check box, a decoration.

        :param index: The index of the item
        :param role: The role to provide data for
        :return: The data for the given role, or None if there is none
        """
        # Check if the index is valid
        if not index.isValid():
            return None
        # Check if the row is inside the available boundaries. The comparison is >=, not >: at
        # exactly len(self._paths) the row is one past the end, and get_item_at_index would raise
        # an IndexError on self._paths[index] -- the case the guard exists to prevent.
        #
        # It is deliberately against _paths and NOT against rowCount(): rowCount returns
        # current_index, the lazy-loading cursor, which is how many rows have been revealed so far,
        # not how many exist. Guarding against it would reject valid rows that fetchMore is about
        # to reveal
        row = index.row()
        if row >= len(self._paths) or row < 0:
            return None
        return self.get_item_at_index(row).data(role)

    def setData(self, index: QModelIndex, value, role=Qt.EditRole) -> bool:
        """
        Method to set the data of the item at the given index

        :param index: The index of the item to change
        :param value: The value to set
        :param role: The role to set the value for
        :return: True if the value was set, False otherwise
        """
        if role != Qt.EditRole or not index.isValid():
            return False
        row = index.row()
        if row >= len(self._paths) or row < 0:
            return False
        # Via get_item_at_index, not self._cache[row]: the cache is populated lazily, so a row that
        # has not been displayed yet is simply absent and a direct lookup raises KeyError
        self.get_item_at_index(row).setData(value, role)
        # Without this, views keep showing the old value until something else forces a repaint
        self.dataChanged.emit(index, index, [role])
        return True

    def get_item_at_index(self, index: int) -> QStandardItem:
        """
        Method to get the item at the specified index

        :param index: The index of the item
        :return: The item
        """
        if index not in self._cache:
            self._cache[index] = Util.create_list_item(self._paths[index])
        return self._cache[index]


def show_error_message(title: str, info: str, text: str) -> None:
    """
    Function to display an error message. Must only be called from the main thread

    :param title: The title of the message box
    :param info: The informative text of the message box
    :param text: The text of the message box
    :return: None
    """
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowIcon(Icon.get_icon("LOGO"))
    with open(os.path.join(gpaths.css_dir, "messagebox.css"), "r", encoding="utf-8") as f:
        msg.setStyleSheet(f.read())
    msg.setText(text)
    msg.setInformativeText(info)
    msg.setWindowTitle(title)
    msg.exec_()


def exception_hook(exc_type, exc_value, traceback_obj) -> None:
    """
    General exception hook to display error message for user. Only covers the main thread, worker
    threads are covered by thread_exception_hook
    :param exc_type: Type of the exception
    :param exc_value: Value of the exception
    :param traceback_obj: The traceback object associated with the exception
    :return: None
    """
    # Show error message in GUI
    time_string = time.strftime("%Y-%m-%d, %H:%M:%S")
    text = "During the execution of the program, following error occured:\n" \
           f"{''.join(traceback.format_exception(exc_type, exc_value, traceback_obj))}"
    # Must reach the log file: started via pythonw.exe or as a packaged build there is no console,
    # so a printed traceback would be lost and the crash could not be diagnosed afterwards
    LOGGER.critical("Unhandled exception on the main thread:\n%s", text)
    show_error_message(title="An error occured during execution",
                       info=f"An {exc_type.__name__} occured at {time_string}",
                       text=text)


def thread_exception_hook(args) -> None:
    """
    Exception hook for worker threads. Routes the error into the main window, which displays it on
    the main thread. Acts as backstop for threads which are not covered by NucDetect._run_guarded

    :param args: The named tuple provided by threading.excepthook
    :return: None
    """
    text = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    thread_name = getattr(args.thread, "name", "a background task")
    LOGGER.critical("Unhandled exception in thread %s:\n%s", thread_name, text)
    # The message box must not be created from this thread, so the error is emitted as signal
    if _MAIN_WINDOW is not None:
        _MAIN_WINDOW.err_signal.emit(thread_name, text)


def main() -> None:
    """
    Function to start the program

    :return: None
    """
    global _MAIN_WINDOW
    # Configure logging before anything else, so even a failure during start-up is recorded
    configure_logging()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sys.excepthook = exception_hook
        threading.excepthook = thread_exception_hook
        app = QtWidgets.QApplication(sys.argv)
        app.setWindowIcon(Icon.get_icon("LOGO"))
        pixmap = QPixmap(os.path.join(gpaths.logo_dir, "banner_norm.png"))
        splash = QSplashScreen(pixmap)
        splash.show()
        splash.showMessage("Checking for thumbnails...")
        LOGGER.info("Check files for thumbnails...")
        # Count number of available images
        total = 0
        for root, dirs, files in os.walk(gpaths.images_path):
            total += len(files)
        file_index = 1
        for root, dirs, files in os.walk(gpaths.images_path):
            for file in files:
                msg = f"{file_index: 04d}:{total: 04d} checked..."
                # Deliberately not logged: this is a transient progress indicator that overwrites
                # itself on one console line, not a diagnostic worth a line in the log file
                print(msg, end="\r", flush=True)
                splash.showMessage(msg)
                Util.create_thumbnail(os.path.join(root, file))
                file_index += 1
        # Close the in-place progress line before anything else writes to the console
        print()
        LOGGER.info("All files checked for thumbnails, starting...")
        main_win = NucDetect()
        _MAIN_WINDOW = main_win
        splash.finish(main_win)
        main_win.show()
        sys.exit(app.exec_())


if __name__ == '__main__':
    # Necessary for packaging
    multiprocessing.freeze_support()
    main()
