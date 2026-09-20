import copy
import os
import threading
import traceback

from matplotlib import font_manager
from threading import Thread
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Iterable

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import uic, QtCore, QtGui
from PyQt5.QtCore import QRectF, Qt, QItemSelection, QAbstractTableModel, QVariant, pyqtSignal, QTimer
from PyQt5.QtGui import QKeyEvent, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (
                             QDialog, QInputDialog, QSizePolicy, QMessageBox, QSpinBox,
                             QHBoxLayout, QVBoxLayout, QHeaderView, QMenuBar, QMenu, QAction,
                             QComboBox, QListWidget, QAbstractItemView, QListWidgetItem,
                             QAbstractScrollArea, QWidget)
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from gui import Plots
from gui import Util
from core.DataProcessing import perform_statistical_analysis_on_groups
from gui.Plots import PlotCanvas
from gui.Util import create_image_item_list_from
from core.database.connections import Inserter, Requester
from core.logging_config import get_logger
from gui.definitions.icons import Icon
from gui.dialogs.GraphicsItems import EditorView, ROIDrawer, ROIItem
from gui.dialogs.selection import ImageSelectionDialog, ExperimentSelectionDialog
from gui import Paths
from gui.loader import Loader
from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler

# Excel's own limit on a worksheet title. openpyxl only warns above it, but the workbook is then
# unreadable for some applications, so the export truncates rather than relying on the warning
MAX_SHEET_NAME_LENGTH = 31
#: Longest file STEM an export writes. Not a filesystem limit -- NTFS allows 255 per component --
#: but a cap that keeps the full path clear of MAX_PATH once the results folder, the extension and
#: a de-duplication suffix are added. Deliberately separate from MAX_SHEET_NAME_LENGTH: Excel's 31
#: is a property of Excel, and applying it to file names would rename files nothing else renames
MAX_FILE_STEM_LENGTH = 120


LOGGER = get_logger(__name__)


class DataExportDialog(QDialog):
    # No __slots__ here: QDialog is a sip type and supplies a __dict__ from its C++ base, so the
    # declaration removed nothing, rejected no undeclared attribute, and cost bytes per instance
    # for the descriptors -- measured on 2026-08-15 for the equivalent declarations on the editor's
    # graphics items. core/roi/ROI.py's ROI is the one place the memory argument holds; it is a
    # plain class and keeps its __slots__
    STANDARD_OPTIONS = (
        "Selected Image",
        "All analysed Images",
        "All defined Experiments"
    )
    # Areas and axes are in micrometres, centres in pixels -- RW, 2026-09-14: a centre is a
    # literal coordinate in the image and converting it would help nobody. An image with no stored
    # conversion factor reports these three columns in pixels instead, and the log says which image.
    # "Edge" was added 2026-09-15, with the column of the same name on the main result table: the
    # rows come from Requester.get_table_data_for_image, so the two headers describe one row shape
    # and adding a cell there without adding it here miscounts every column after it
    STANDARD_HEADER = ["Image Name", "Image Identifier", "ROI Identifier", "Center Y", "Center X",
                       "Area [µm²]", "Ellipticity[%]", "Or. Angle [deg]", "Maj. Axis [µm]",
                       "Min. Axis [µm]", "match", "Edge"]

    def __init__(self, current_image: Union[str, None] = None, display_name: Union[str, None] = None):
        """
        :param current_image: md5 hash of the currently selected image. None if there is no current image
        :param display_name: The name of the currently selected image. None if there is no current image
        """
        super(DataExportDialog, self).__init__()
        self.threads: List[Thread] = []
        # Collects tracebacks of failed exports, read by NucDetect.check_for_running_threads
        self.errors: List[str] = []
        self.cur_img = current_image
        self.disp_name = display_name
        # File stems already handed out by this export run, and the lock that guards them.
        # The non-workbook path starts ONE THREAD PER IMAGE, so reserving a name is a genuine
        # race: two images sharing a file name would otherwise both be told they may use it
        self._used_file_stems = set()
        self._stem_lock = threading.Lock()
        # ONE REQUESTER PER THREAD, handed out by the `req` property below. This used to be a
        # single `Requester(protected=False)` shared by every export thread -- and `protected` is
        # what `connect_to_database` passes to sqlite3's `check_same_thread`, so the guard against
        # exactly this sharing had been switched OFF to make it possible. Several threads then
        # drove one connection and one cursor concurrently, which sqlite does not support: a
        # cursor's result set belongs to whoever last executed on it.
        self._local = threading.local()
        self.ui = self.initialize_ui()

    @property
    def req(self) -> Requester:
        """
        The calling thread's own Requester, created on first use

        A property rather than an argument threaded through eight call sites: every reader below
        keeps saying `self.req` and gets a connection it is allowed to use. `protected` is left at
        its default, so sqlite enforces the one-thread-per-connection rule again instead of being
        told to ignore it.

        Closed by `_run_export`, which is the single funnel every export thread goes through; the
        dialog's own (main-thread) requester lives as long as the dialog.
        """
        requester = getattr(self._local, "requester", None)
        if requester is None:
            requester = Requester()
            self._local.requester = requester
        return requester

    def _release_thread_requester(self) -> None:
        """
        Method to close the calling thread's Requester, if it made one

        Without this an export run leaks one sqlite connection per image -- the non-workbook path
        starts a thread per image -- and they would only be released when the garbage collector
        happened to reach them.

        :return: None
        """
        requester = getattr(self._local, "requester", None)
        if requester is not None:
            requester.connector.close_connection()
            self._local.requester = None

    def accept(self) -> None:
        self.save_data()
        super().accept()

    def initialize_ui(self) -> Any:
        """
        Method to initialize the ui of this dialog

        :return: The loaded ui
        """
        # Annotated Any deliberately: PyQt5 ships no stubs for uic, so a type checker reads
        # loadUi's source and infers "Unknown | None" from its baseinstance parameter. Every
        # attribute access on the loaded ui is then reported as an error on a possibly-None object
        ui: Any = uic.loadUi(Paths.ui_save_dial, self)
        # Set window and style
        self.setWindowFlags(
            self.windowFlags() |
            QtCore.Qt.WindowSystemMenuHint |
            QtCore.Qt.WindowMinMaxButtonsHint |
            QtCore.Qt.Window
        )
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowTitle("Data Saving Dialog")
        self.setStyleSheet(Util.load_stylesheet("main.css"))
        # Connect buttons to functions
        ui.btn_export.clicked.connect(self.accept)
        ui.btn_cancel.clicked.connect(self.close)
        ui.cbx_xlsx.stateChanged.connect(lambda: ui.cbx_xlsx_single.setEnabled(ui.cbx_xlsx.isChecked()))
        # The single-file option combines XLSX and NOTHING else -- csv and html are written once per
        # image whatever it says, because a csv cannot hold 116 tables. That is what "if possible"
        # in the label was carrying on its own, and it carried it invisibly: RW ticked the box,
        # found a file per image in the results folder and reported it as a defect (2026-09-13).
        #
        # Shown only while the option is actually IN FORCE -- ticked AND reachable -- so it explains
        # the state the user is in rather than standing there as permanent small print. Both signals
        # are needed: unticking XLSX leaves this box checked but disabled, and the note must go with
        # the behaviour, not with the tick
        # Zero-argument lambdas, as the line above uses: `stateChanged` emits the Qt check STATE
        # as an int, and connecting the method directly fed that int into its `ui` parameter --
        # `AttributeError: 'int' object has no attribute 'lbl_single_file_note'`
        ui.cbx_xlsx.stateChanged.connect(lambda: self.update_single_file_note())
        ui.cbx_xlsx_single.stateChanged.connect(lambda: self.update_single_file_note())
        # Fill the combobox
        cbx_cont = []
        if self.cur_img:
            cbx_cont.extend(DataExportDialog.STANDARD_OPTIONS)
        else:
            cbx_cont.extend(DataExportDialog.STANDARD_OPTIONS[1:])
        cbx_cont.extend(self.req.get_all_experiments())
        ui.cbx_choice.addItems(cbx_cont)
        self.update_single_file_note(ui)
        return ui

    def update_single_file_note(self, ui: Any = None) -> None:
        """
        Method to show the single-file note exactly while that option is in force

        :param ui: The loaded ui. Only passed from initialize_ui, which runs BEFORE self.ui exists
        :return: None
        """
        ui = ui if ui is not None else self.ui
        # setVisible, not setText: an empty label still occupies its row and the dialog would
        # change height for no visible reason
        ui.lbl_single_file_note.setVisible(ui.cbx_xlsx.isChecked()
                                           and ui.cbx_xlsx_single.isChecked())

    def save_data(self):
        """
        Method to save the selected data

        :return: None
        """
        # A fresh run may legitimately reuse every name, because it overwrites its own previous
        # output. The reservations only have to be unique WITHIN one export
        with self._stem_lock:
            self._used_file_stems.clear()
        # Get the selection
        selection = self.ui.cbx_choice.currentText()
        # Save the selected image
        if selection == DataExportDialog.STANDARD_OPTIONS[0]:  # Selected image
            self.export_image_as_table(self.cur_img)
        # Save all analysed images
        elif selection == DataExportDialog.STANDARD_OPTIONS[1]:  # All analysed images
            # Get the hashes of all images
            # get_all_images returns md5 STRINGS, so the previous `if x[12]` tested the 13th
            # character of the hash -- always truthy, verified over 10 000 hashes: 0 dropped. The
            # predicate had been written for a row tuple whose column 12 is the analysed flag.
            # Requester has no "give me the analysed ones" query; check_if_image_was_analysed is the
            # accessor that exists, and it answers False for an unknown hash as of 2026-08-13
            img_hashes = [x for x in self.req.get_all_images()
                          if self.req.check_if_image_was_analysed(x)]
            # Check if the data should be saved in one file
            if self.export_goes_into_one_workbook():
                self.export_into_one_workbook(img_hashes, "results_all_images",
                                              self._export_image_as_table)
            else:
                for md5 in img_hashes:
                    self.export_image_as_table(md5)
        # Save all defined experiments
        elif selection == DataExportDialog.STANDARD_OPTIONS[2]:  # All defined experiments
            # Get all defined experiments
            exps = self.req.get_all_experiments()
            # Check if the data should be saved in one file
            if self.export_goes_into_one_workbook():
                self.export_into_one_workbook(exps, "results_all_experiments",
                                              self._export_experiment_as_table)
            else:
                for experiment in exps:
                    self.export_experiment_as_table(experiment)
        # Save selected experiment
        else:
            self.export_experiment_as_table(selection)

    def _run_export(self, worker: Callable, *args) -> None:
        """
        Method to run an export body, ensuring that errors are recorded instead of lost. Meant to
        be used as target of every export thread started by this dialog

        This dialog is usually already closed when an export fails, and it must not create widgets
        from a worker thread anyway, so the error is only collected here. The main window picks it
        up while it waits for the exports to finish

        :param worker: The export method to execute in this thread
        :param args: The arguments to pass to the worker
        :return: None
        """
        try:
            worker(*args)
        except Exception:
            self.errors.append(traceback.format_exc())
        finally:
            # In the finally, so a failed export releases its connection too. This runs on the
            # export thread, which is the only thread allowed to close that connection
            self._release_thread_requester()

    def export_goes_into_one_workbook(self) -> bool:
        """
        Method to check whether the xlsx output of this run is a single workbook

        :return: True if one workbook holding one sheet per exported item is wanted
        """
        return self.ui.cbx_xlsx.isChecked() and self.ui.cbx_xlsx_single.isChecked()

    def export_into_one_workbook(self, items: List[str], file_name: str,
                                 exporter: Callable) -> None:
        """
        Method to export several items into a single workbook, one sheet per item

        ONE thread for the whole workbook, not one per item. DataFrame.to_excel against a path
        creates a NEW workbook per call, so the previous shape -- one thread and one to_excel per
        item, every one of them writing the same path -- kept whichever sheet happened to be
        written last and discarded the rest silently. Which one survived was a race, since the
        threads were started without any limit

        :param items: The image hashes or experiment names to export
        :param file_name: The name of the workbook to write, without extension
        :param exporter: The private per-item export method to drive
        :return: None
        """
        export_thread = threading.Thread(target=self._run_export,
                                         args=(self._export_workbook, items, file_name, exporter),
                                         daemon=True)
        export_thread.start()
        self.threads = [x for x in self.threads if x.is_alive()]
        self.threads.append(export_thread)

    def _export_workbook(self, items: List[str], file_name: str, exporter: Callable) -> None:
        """
        Private method to write one workbook holding one sheet per item

        :param items: The image hashes or experiment names to export
        :param file_name: The name of the workbook to write, without extension
        :param exporter: The private per-item export method to drive
        :return: None
        """
        Paths.ensure_directories()
        path = os.path.join(Paths.result_path, f"{file_name}.xlsx")
        with pd.ExcelWriter(path) as writer:
            for item in items:
                exporter(item, writer=writer)

    @staticmethod
    def get_valid_sheet_name(name: str, taken: Iterable[str]) -> str:
        """
        Method to turn the given name into one Excel accepts and that is not taken yet

        Sheet names come from image file names and experiment names, and Excel's rules on them are
        stricter than the file system's: []:*?/\\ are rejected outright -- pandas raises
        ValueError on the whole export -- more than 31 characters makes the workbook unreadable for
        some applications, and a repeated name silently overwrites the earlier sheet. None of that
        mattered while only one sheet survived a run

        :param name: The name to derive the sheet name from
        :param taken: The sheet names already used in this workbook
        :return: A valid, unused sheet name
        """
        cleaned = "".join("_" if c in r"[]:*?/\\" else c for c in str(name)).strip() or "Sheet"
        cleaned = cleaned[:MAX_SHEET_NAME_LENGTH]
        taken = set(taken)
        if cleaned not in taken:
            return cleaned
        # Two images can carry the same file name in different folders, and both are exported
        counter = 2
        while True:
            suffix = f"_{counter}"
            candidate = f"{cleaned[:MAX_SHEET_NAME_LENGTH - len(suffix)]}{suffix}"
            if candidate not in taken:
                return candidate
            counter += 1

    @staticmethod
    def clean_file_stem(name: str) -> str:
        """
        Method to turn the given name into a file stem the file system accepts

        Separate from `get_valid_sheet_name`, and the two must not be merged: Excel rejects
        `[]:*?/\\` and caps names at 31 characters, while Windows rejects `<>:"/\\|?*` plus the
        control characters and allows 255. Cleaning a file name with Excel's rules would mangle
        names the file system is perfectly happy with, and cleaning a sheet name with the file
        system's would let `[` through into a workbook, which raises.

        :param name: The name to derive a file stem from
        :return: A stem safe to write to disk, never empty
        """
        illegal = set('<>:"/\\|?*')
        cleaned = "".join("_" if c in illegal or ord(c) < 32 else c for c in str(name))
        # Trailing dots and spaces are silently dropped by Windows, which would make two distinct
        # stems collide again after they were checked for collision
        cleaned = cleaned.strip().rstrip(". ")
        return cleaned[:MAX_FILE_STEM_LENGTH] or "export"

    def reserve_file_stem(self, name: str) -> str:
        """
        Method to claim a unique file stem for this export run

        **Two images can carry the same file name in different folders**, and both are exported.
        The workbook path has de-duplicated its SHEET names since 2026-08-xx, but the csv and html
        outputs -- which are written per image whatever the single-file box says -- kept using the
        raw image name, so the second image silently overwrote the first.

        Measured on the live database 2026-09-13, exporting "All analysed Images":
        **116 analysed images produced 108 csv and 108 html files.** Eight images from two folders
        sharing file names lost their output with no error of any kind.

        Reserved under a lock because the non-workbook path runs one thread per image.

        :param name: The name to derive the stem from
        :return: A cleaned stem not yet used by this run
        """
        stem = self.clean_file_stem(name)
        with self._stem_lock:
            if stem not in self._used_file_stems:
                self._used_file_stems.add(stem)
                return stem
            counter = 2
            while True:
                suffix = f"_{counter}"
                candidate = f"{stem[:MAX_FILE_STEM_LENGTH - len(suffix)]}{suffix}"
                if candidate not in self._used_file_stems:
                    self._used_file_stems.add(candidate)
                    return candidate
                counter += 1

    def export_image_as_table(self, md5: str,
                              xlsx_name: str = None,
                              include_header: bool = True,
                              sheet_name: str = None) -> None:
        """
        Method to save the given image

        :param md5: The hash of the image
        :param xlsx_name: Optional; If given, the file will use this name
        :param include_header: If true, the table header will also be saved
        :param sheet_name: Name of the sheet
        :return: The thread used for export
        """
        export_thread = threading.Thread(target=self._run_export,
                                         args=(self._export_image_as_table,
                                               md5, xlsx_name, include_header, sheet_name),
                                         daemon=True)
        export_thread.start()
        self.threads = [x for x in self.threads if x.is_alive()]
        self.threads.append(export_thread)

    def _export_image_as_table(self, md5: str,
                               xlsx_name: str = None,
                               include_header: bool = True,
                               sheet_name: str = None,
                               writer: pd.ExcelWriter = None) -> None:
        """
        Private method to concurrently fetch the image data and save the table

        :param md5: The hash of the image
        :param xlsx_name: Optional; If given, the file will use this name
        :param include_header: If true, the table header will also be saved
        :param sheet_name: Name of the sheet
        :param writer: Optional; the open workbook to add a sheet to, for single-file exports
        :return: None
        """
        # Get general table header
        header = copy.copy(self.STANDARD_HEADER)
        # ("Channel", "Foci"), exactly as _export_experiment_as_table does, because both render the
        # SAME rows -- get_table_data_for_image emits one row per nucleus PER CHANNEL, ending in
        # the channel name and that channel's focus count.
        #
        # This used to append one column per non-main channel, a WIDE layout the rows have never
        # had. It matched only by coincidence, when an image has exactly two non-main channels:
        # the two channel NAMES then fill the "Channel" and "Foci" slots and the widths agree.
        # 115 of the 116 analysed images in the testing database have exactly two. The fifth
        # channel of `demo_5channel` gives four, and pandas refused the export with
        # `ValueError: Writing 14 cols but got 16 aliases` -- which in the single-workbook path
        # aborts the WHOLE run, so every image after it is silently missing. Reported from real
        # use 2026-09-15. Reconstructed with the pre-Edge-column constants it was
        # `Writing 13 cols but got 15 aliases`, so the defect predates that column by exactly two
        # aliases and was not caused by it.
        #
        # Whether this export should instead BE the wide table its header promised -- one column
        # per channel, one row per nucleus -- is a separate question for RW, and a different
        # change: it would have to reshape the rows, not the header.
        header.extend(("Channel", "Foci"))
        # Get the data for the given image
        rows = self.get_data_for_image(md5)
        # Try to get the name of the image
        img_name = self.req.get_image_filename(md5)
        self.save_table_to_disk(img_name, rows, header,
                                include_header=include_header,
                                sheet_name=sheet_name if sheet_name else img_name,
                                xlsx_name=xlsx_name,
                                writer=writer)

    def export_experiment_as_table(self,
                                   experiment: str,
                                   xlsx_name: str = None,
                                   include_header: bool = True,
                                   sheet_name: str = None
                                   ) -> None:
        """
        Method to save the given experiment

        :param experiment: Name of the experiment
        :param xlsx_name: Optional; If given, the file will have this name
        :param include_header: If true, the table header will also be saved
        :param sheet_name: Name of the sheet
        :return: The thread used for export
        """
        export_thread = threading.Thread(target=self._run_export,
                                         args=(self._export_experiment_as_table,
                                               experiment, xlsx_name, include_header, sheet_name),
                                         daemon=True)
        export_thread.start()
        self.threads = [x for x in self.threads if x.is_alive()]
        self.threads.append(export_thread)

    def _export_experiment_as_table(self,
                                    experiment: str,
                                    xlsx_name: str = None,
                                    include_header: bool = True,
                                    sheet_name: str = None,
                                    writer: pd.ExcelWriter = None
                                    ) -> None:
        """
        Private method to concurrently fetch the experiment data and save the table

        :param experiment: Name of the experiment
        :param xlsx_name: Optional; If given, the file will have this name
        :param include_header: If true, the table header will also be saved
        :param sheet_name: Name of the sheet
        :param writer: Optional; the open workbook to add a sheet to, for single-file exports
        :return: None
        """
        # Get general table header
        header = copy.copy(self.STANDARD_HEADER)
        header.insert(2, "Group")
        header.extend(("Channel", "Foci"))
        # Get the data for the given image
        rows = self.req.get_table_data_for_experiment(experiment)
        self.save_table_to_disk(experiment,
                                rows, header,
                                include_header=include_header,
                                sheet_name=sheet_name if sheet_name else experiment,
                                xlsx_name=xlsx_name,
                                writer=writer)

    def get_data_for_image(self, image: str) -> List[List]:
        """
        Method to get the data for the specific image

        :param image: The md5 hash of the image
        :return: The extracted data
        """
        return self.req.get_table_data_for_image(image, self.req.get_image_filename(image))

    def save_table_to_disk(self,
                           name: str,
                           rows: List[List],
                           header: List = (),
                           include_header: bool = True,
                           xlsx_name: str = None,
                           sheet_name: str = "Sheet 1",
                           writer: pd.ExcelWriter = None) -> None:
        """
        Method to save the given table to the disk

        :param name: The file name to use
        :param rows: The rows to save
        :param header: Optional: The header of the table
        :param include_header: If true, the header will also be saved
        :param xlsx_name: Optional: The file name to used for the xlsx file. If not given, name will be used
        :param sheet_name: Optional; Name of the sheet to use. Allows to save in the same file,
        if different sheet names are chosen
        :param writer: Optional; an open workbook to add a sheet to instead of writing a file of
        its own. The csv and html outputs stay one file per item either way
        :return: None
        """
        # Create a pandas dataframe
        # columns=header when there are no rows, and this is not defensive tidying. An image can be
        # analysed and carry NO roi -- detection found nothing, or the quality check removed
        # everything -- and `pd.DataFrame([])` is then (0, 0). Handing 13 column aliases to a frame
        # with no columns raises `ValueError: Writing 0 cols but got 13 aliases`, and in the
        # single-workbook path that takes the WHOLE export down: one thread writes every image into
        # one ExcelWriter, so the run stops at the empty image and every later one is silently
        # absent. Measured 2026-09-13 on the live database -- the empty image is number 55 of 116,
        # and 61 images never reached the workbook.
        #
        # A header-only table is written instead of skipping the image, at RW's instruction: a sheet
        # with headers and no rows says "analysed, nothing found", where an absent sheet is
        # indistinguishable from an export that lost it
        df = pd.DataFrame(rows) if len(rows) else pd.DataFrame(columns=list(header))
        # The results folder is not guaranteed to exist: it is created by Paths.ensure_directories,
        # which the GUI calls at start-up -- but a user who deletes it while the program is running,
        # or any entry point that has not called it, would otherwise get a bare FileNotFoundError
        # out of a background export thread, where it is only visible via self.errors. Asking
        # Paths rather than calling makedirs here keeps directory creation in the module that
        # declares the directories
        Paths.ensure_directories()
        # ONE stem for all three outputs of this item, reserved once, so `image.csv`, `image.html`
        # and `image.xlsx` keep matching names -- and so a second image with the same file name
        # gets `image_2` rather than overwriting the first. Claimed only when a FILE is actually
        # written: an item that only contributes a sheet to a shared workbook needs no stem, and
        # reserving one would push the next duplicate to `_3`
        writes_file = (self.ui.cbx_csv.isChecked() or self.ui.cbx_html.isChecked()
                       or (self.ui.cbx_xlsx.isChecked() and writer is None))
        stem = self.reserve_file_stem(xlsx_name if xlsx_name else name) if writes_file else None
        if self.ui.cbx_csv.isChecked():
            df.to_csv(os.path.join(Paths.result_path, f"{stem}.csv"),
                      header=header if include_header else False, index=False)
        if self.ui.cbx_html.isChecked():
            df.to_html(os.path.join(Paths.result_path, f"{stem}.html"),
                       header=header if include_header else False, index=False)
        if self.ui.cbx_xlsx.isChecked():
            if writer is not None:
                # One sheet in the shared workbook. The name has to be checked against the sheets
                # already in it: an illegal character raises and takes the whole export with it,
                # and a repeated name overwrites the earlier sheet without a word
                df.to_excel(writer, header=header if include_header else False, index=False,
                            sheet_name=self.get_valid_sheet_name(sheet_name, writer.sheets))
            else:
                df.to_excel(os.path.join(Paths.result_path, f"{stem}.xlsx"),
                            header=header if include_header else False, index=False,
                            sheet_name=self.get_valid_sheet_name(sheet_name, ()))


class Editor(QDialog):
    # No __slots__ here: QDialog is a sip type and supplies a __dict__ from its C++ base, so the
    # declaration removed nothing, rejected no undeclared attribute, and cost bytes per instance
    # for the descriptors -- measured on 2026-08-15 for the equivalent declarations on the editor's
    # graphics items. core/roi/ROI.py's ROI is the one place the memory argument holds; it is a
    # plain class and keeps its __slots__

    def __init__(self, image: np.ndarray,
                 roi: ROIHandler,
                 active_channels: List[Tuple[int, str, int]],
                 size_factor: float = 1, img_name: str = "",
                 x_scale: float = 1, y_scale: float = 1):
        """
        Constructor

        :param image: The image to edit
        :param roi: The detected roi
        :param active_channels: Index, Name
        :param size_factor: Scaling factor for standard sizes
        :param img_name: Name of the image
        :param x_scale: Scaling factor for x-axis
        :param y_scale: Scaling factor for y-axis
        """
        super(Editor, self).__init__()
        # No self.ui / self.editor placeholders here: initialize_ui() runs at the end of this
        # constructor and assigns both, so a None seeded first is never observable -- and it
        # contradicted the annotations on the real assignments (self.editor: EditorView)
        self.image = image
        self.img_name = img_name
        self.roi = roi
        # Only the channels the loaded array actually HAS. This list comes from the channels table,
        # which can disagree with the file: nothing ever deleted a channel row, so an image
        # registered once with more channels than it has now kept the surplus indices, and picking
        # one raised `IndexError: index 4 is out of bounds for axis 2 with size 4` from inside the
        # combo box's lambda, where nothing could catch it. Filtering here serves both consumers --
        # this dialog's combo box and the EditorView constructed below
        available = self.image.shape[2] if self.image.ndim > 2 else 1
        self.active_channels = [x for x in active_channels if x[0] < available]
        dropped = [x[1] for x in active_channels if x[0] >= available]
        if dropped:
            LOGGER.warning("Image %s has %d channels but the database lists %d -- not offering %s",
                           img_name, available, len(active_channels), dropped)
        self.size_factor = size_factor
        self.temp_items = []
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.initialize_ui()

    def mark_main_channel_in_combo_box(self) -> None:
        """
        Method to mark the main channel in the channel selector, so it is identifiable at a glance

        RW, 2026-09-15: *"The editor channel selection should use color to highlight the main
        channel if possible. This would allow the user to easily identify the main channel."*
        Asked for after a re-analysis on a different channel was read as having run on the old
        one -- nothing on this screen said which channel the nuclei belong to, and nuclei stay
        drawn on EVERY channel (ROIDrawer.change_channel keeps them active while "show additional"
        is on), so the view itself cannot answer it either.

        **THE ITEM TEXT IS NOT TOUCHED, and that is a constraint rather than a preference.**
        `show_channel` is connected to `currentIndexChanged` and looks `currentText()` up in
        `EditorView.active_channels`, which is keyed by the bare channel name -- appending
        "(main)" to the label would raise KeyError on every selection of that entry. The mark is
        therefore colour, weight and a tooltip, all of which live in item DATA.

        The colour is TAKEN FROM the nucleus pen rather than repeated as a literal, so the entry
        matches what a nucleus actually looks like in the view and cannot drift from it if that
        pen is ever restyled.

        :return: None
        """
        main = self.editor.main_channel
        index = self.ui.cbx_channel.findText(main)
        # -1 means the nominated channel is not offered -- the combo is built from the channels
        # the loaded ARRAY has, and Editor.__init__ drops database rows beyond that. It already
        # logs the mismatch; there is simply nothing to mark here
        if index < 0:
            LOGGER.warning("Main channel %s is not among the offered channels %s -- not marking it",
                           main, [self.ui.cbx_channel.itemText(i)
                                  for i in range(self.ui.cbx_channel.count())])
            return
        font = self.ui.cbx_channel.font()
        font.setBold(True)
        colour = ROIDrawer.MARKERS["nucleus_auto"].color()
        self.ui.cbx_channel.setItemData(index, QtGui.QBrush(colour), Qt.ForegroundRole)
        self.ui.cbx_channel.setItemData(index, font, Qt.FontRole)
        self.ui.cbx_channel.setItemData(index, f"{main} is the main channel -- the nuclei were "
                                               f"detected on it", Qt.ToolTipRole)

    def accept(self) -> None:
        # The editor answers False when the user cancels the confirmation for foci that lie outside
        # every nucleus. Closing anyway would discard the very edits they went back to correct
        if not self.editor.apply_all_changes():
            return
        super().accept()

    def initialize_ui(self) -> None:
        """
        Method to initialize the ui of this widget

        :return: None
        """
        # Annotated Any deliberately -- see DataExportDialog.initialize_ui above
        self.ui: Any = uic.loadUi(Paths.ui_editor_dial, self)
        # Load css file
        self.ui.setStyleSheet(Util.load_stylesheet("main.css"))
        self.setWindowTitle(f"Modification Dialog for {self.img_name}")
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint |
                            QtCore.Qt.Window)
        self.editor: EditorView = EditorView(self.image, self.roi,
                                             self,
                                             active_channels=self.active_channels,
                                             size_factor=self.size_factor,
                                             x_scale=self.x_scale,
                                             y_scale=self.y_scale)
        self.ui.view.addWidget(self.editor)
        # Add icons to buttons
        self.ui.btn_view.setIcon(Icon.get_icon("EYE"))
        self.ui.btn_add.setIcon(Icon.get_icon("PLUS_CIRCLE"))
        self.ui.btn_edit.setIcon(Icon.get_icon("EDIT"))
        self.ui.btn_show.setIcon(Icon.get_icon("CIRCLE"))
        self.ui.btn_coords.setIcon(Icon.get_icon("MOUSE"))
        # Explicit ids, so the mapping to EditorView's modes stops depending on the order the
        # buttons happen to appear in the .ui file. Qt numbers them -2, -3, -4 in that order, and
        # change_mode below used to recover the mode with `abs(id_) - 3` -- correct only by
        # coincidence. Same fix as the main-channel button group
        for mode, button in enumerate((self.ui.btn_view, self.ui.btn_add, self.ui.btn_edit)):
            self.ui.btng_mode.setId(button, mode)
        # NOT idClicked -> set_mode: idClicked emits the button ID, while set_mode takes a mode and
        # checks the matching button, so that connection fed ids into the wrong parameter and
        # silently matched none of its branches. idToggled -> change_mode below is the live path,
        # and it fires for keyboard-driven changes too, which idClicked does not
        self.ui.btn_coords.toggled.connect(
            lambda: self.editor.track_mouse_position(self.ui.btn_coords.isChecked())
        )
        self.ui.btn_coords.toggled.connect(
            lambda: self.set_status(f"Coordinate Tracking: {self.ui.btn_coords.isChecked()}")
        )
        # EVERY channel the editor can show, in channel-index order -- not only the ones that
        # already have roi.
        #
        # This used to iterate `self.roi.idents`, which holds only the channels something was
        # DETECTED in. A channel with no detection never appeared, so it could not be selected, so
        # nothing could be drawn on it -- and a channel with nothing on it is exactly the one a user
        # needs to open in order to add the first item by hand. On a 4- or 5-channel image the combo
        # typically offered the first three and Composite, and the extra channels were unreachable.
        # Reported from real use, 2026-08-22.
        #
        # The KeyError this replaced is still guarded, and better: `show_channel` looks the chosen
        # name up in `EditorView.active_channels`, which is built from this same argument, so every
        # entry added here is showable by construction rather than by filtering.
        for _, name in sorted(self.active_channels, key=lambda channel: channel[0]):
            self.ui.cbx_channel.addItem(name)
        self.ui.cbx_channel.addItem("Composite")
        self.mark_main_channel_in_combo_box()
        self.ui.cbx_channel.setCurrentText("Composite")
        self.ui.cbx_channel.currentIndexChanged.connect(
            lambda: self.editor.show_channel(self.ui.cbx_channel.currentText())
        )
        # toggled, not stateChanged: stateChanged emits the Qt check STATE (0/1/2) while both
        # slots are annotated to take a bool. It worked by truthiness, so the annotation was the
        # thing that was wrong -- and PartiallyChecked would have arrived as True
        self.ui.cbx_high_contrast.toggled.connect(self.editor.toggle_high_contrast_mode)
        self.ui.cbx_white_balance.toggled.connect(self.editor.toggle_adjust_white_balance)
        self.ui.cbx_colormap.addItems(self.get_colormaps())
        self.ui.cbx_colormap.setCurrentText("gray")
        self.editor.change_colormap("gray")
        self.ui.cbx_colormap.currentTextChanged.connect(self.editor.change_colormap)
        # React to Draw Ellipsis Button toggle
        self.ui.btn_show.toggled.connect(
            lambda: self.editor.draw_additional_items(self.ui.btn_show.isChecked())
        )
        self.ui.btng_mode.idToggled.connect(self.change_mode)
        # React to changes of the used size factor
        self.ui.spb_sizeFactor.setValue(self.size_factor)
        self.ui.spb_sizeFactor.valueChanged.connect(
            lambda: self.change_size_factor(self.ui.spb_sizeFactor.value())
        )
        # Setup editing boxes
        sy, sx, _ = self.image.shape
        self.connect_spinboxes_to_change_function()
        self.ui.spb_x.setMinimum(0)
        self.ui.spb_x.setMaximum(sx - 1)
        self.ui.spb_y.setMinimum(0)
        self.ui.spb_y.setMaximum(sy - 1)
        self.ui.spb_width.setMinimum(0)
        self.ui.spb_width.setMaximum(sx)
        self.ui.spb_height.setMinimum(0)
        self.ui.spb_height.setMaximum(sy)
        self.ui.spb_opacity.valueChanged.connect(self.change_opacity)

    def enable_white_balance_mode(self) -> None:
        """
        Method to enable white balance mode

        :return: None
        """
        self.ui.cbx_white_balance.setEnabled(True)

    def enable_high_contrast_mode(self) -> None:
        """
        Method to enable high contrast mode

        :return: None
        """
        self.ui.cbx_high_contrast.setEnabled(True)

    def get_colormaps(self) -> List[str]:
        """
        Method to get a list of all available colormaps

        :return: List contraining the names of all available colormaps
        """
        return sorted(pg.colormap.listMaps(source="matplotlib"))

    def connect_spinboxes_to_change_function(self, connect: bool = True) -> None:
        """
        Method to connect all editing spinboxes to the change function

        :param connect: If false, all spinboxes will be disconnected from the change function
        :return: None
        """
        for spin in (self.ui.spb_x, self.ui.spb_y, self.ui.spb_width, self.ui.spb_height, self.ui.spb_angle):
            self.connect_spinbox_to_change_function(spin, connect)

    def connect_spinbox_to_change_function(self, spin: QSpinBox, connect: bool = True) -> None:
        """
        Method to connect the given spinbox to the change function

        :param spin: The spinbox to connect
        :param connect: If false, the spinbox will be disconnected
        :return: None
        """
        # set_changes, not the preview_changes that stood here until 2026-09-13. The two built the
        # same rectangle and called the same method; the duplicate is gone and this is the survivor
        if connect:
            spin.valueChanged.connect(self.set_changes)
        else:
            spin.valueChanged.disconnect(self.set_changes)

    def change_opacity(self, new_value: float) -> None:
        """
        Method to change the opacity of the displayed ROI

        :param new_value: The opacity as value between 0 and 100
        :return: None
        """
        self.editor.set_item_opacity(new_value)

    def change_size_factor(self, new_value: float) -> None:
        """
        Method to change the used size factor for ROI

        :param new_value: The new size factor to use
        :return: None
        """
        self.size_factor = new_value
        self.editor.size_factor = new_value

    def set_changes(self) -> None:
        """
        Method to apply the values in the editing spin boxes to the selected item

        Reached from the **A** hotkey, which is now the only one. The Preview and Accept buttons
        this also served were removed on 2026-09-08 (RW: *"Both can be removed"*) -- they were
        `enabled=false` in the .ui and nothing ever enabled them, so they had never been clickable.
        With them went the `sender() == btn_preview` test, which was the only thing that ever
        decided preview from commit, and the `override` parameter that existed to bypass it.

        **A second hotkey, P, was removed on 2026-09-13.** It called a `preview_changes` that built
        the same rectangle and called the same method, so P had never previewed anything -- it did
        what A does. Two keys for one behaviour, neither documented anywhere, is worse than one, and
        A is the honest name for a commit. In practice the geometry is already applied before either
        key is pressed, because the spin boxes commit on `valueChanged`; the key is a way to apply a
        value typed and left unconfirmed.

        If a REAL preview is ever wanted for typed values, P is the place for it -- the drag path
        already has one, since every step of a drag previews and Escape puts the item back.

        :return: None
        """
        # Define QRect to adjust position of item
        x, y = self.ui.spb_x.value(), self.ui.spb_y.value(),
        width, height = self.ui.spb_width.value(), self.ui.spb_height.value()
        rect = QRectF(x - width / 2, y - height / 2, width, height)
        angle = self.ui.spb_angle.value()
        self.editor.set_changes(rect, angle)

    # ROIItem, not QGraphicsItem: this reads center/width/height/angle/roi_id, all of which
    # ROIItem.__init__ sets and the Qt base class has none of. ROIItem rather than NucleusItem --
    # the caller passes self.selected_item, which is either a NucleusItem or a FocusItem, and
    # every member read below is set on their common base
    def setup_editing(self, item: ROIItem) -> None:
        """
        Method to display the information of the selected item

        Called when the SELECTION changes. A change to the geometry of an already selected item goes
        to update_editing_values instead -- it is the half that has to run on every step of a drag,
        and re-enabling widgets and rewriting the hash label sixty times a second is not free

        :param item: The item to retrieve the information from
        :return: None
        """
        self.update_editing_values(item)
        self.enable_editing_widgets(True)
        self.display_hash(str(item.roi_id))

    def update_editing_values(self, item: ROIItem) -> None:
        """
        Method to write the given item's geometry into the five editing spin boxes

        The spin boxes are disconnected while they are written and reconnected afterwards, because
        setValue emits valueChanged -- without that, filling the boxes would drive set_changes,
        which reads the boxes and pushes the result straight back onto the item

        :param item: The item to read the geometry from
        :return: None
        """
        self.write_editing_values(item.center[0], item.center[1],
                                  item.width, item.height, item.angle)

    def write_editing_values(self, center_x: float, center_y: float,
                             width: float, height: float, angle: float) -> None:
        """
        Method to write an explicit geometry into the five editing spin boxes

        Takes values rather than an item, because a gesture in progress is a PREVIEW: the item's own
        center/width/height are deliberately not written until the mouse is released, so reading
        them during a drag would show the geometry the item had before the gesture started

        :param center_x: The x coordinate of the ellipse center
        :param center_y: The y coordinate of the ellipse center
        :param width: The length of the major axis
        :param height: The length of the minor axis
        :param angle: The clockwise angle of the major axis
        :return: None
        """
        self.connect_spinboxes_to_change_function(False)
        # round, not int: int() truncates, so a centre of 147.996 was shown as 147. That was
        # invisible while geometry could only be typed in, and constant once it can be dragged --
        # and it is not only a display wart, because pressing Accept writes the SPIN BOX value back
        # onto the item, so truncating drifted the item a pixel every time
        self.ui.spb_x.setValue(round(center_x))
        self.ui.spb_y.setValue(round(center_y))
        self.ui.spb_width.setValue(round(width))
        self.ui.spb_height.setValue(round(height))
        self.ui.spb_angle.setValue(angle)
        self.connect_spinboxes_to_change_function()

    def enable_editing_widgets(self, enable: bool = True) -> None:
        """
        Method to enable the widgets necessary for editing

        :param enable: Boolean decider
        :return: None
        """
        self.ui.spb_x.setEnabled(enable)
        self.ui.spb_y.setEnabled(enable)
        self.ui.spb_width.setEnabled(enable)
        self.ui.spb_height.setEnabled(enable)
        self.ui.spb_angle.setEnabled(enable)
        if not enable:
            self.ui.spb_x.setValue(0)
            self.ui.spb_y.setValue(0)
            self.ui.spb_width.setValue(0)
            self.ui.spb_height.setValue(0)
            self.ui.spb_angle.setValue(0)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        super().keyReleaseEvent(event)
        if event.key() == Qt.Key_Shift:
            self.editor.shift_down = False

    def keyPressEvent(self, event: QKeyEvent) -> None:
        super().keyPressEvent(event)
        if event.key() == Qt.Key_1:
            self.set_mode(0)
        elif event.key() == Qt.Key_2:
            self.set_mode(1)
        elif event.key() == Qt.Key_3:
            self.set_mode(2)
        elif event.key() == Qt.Key_4:
            self.ui.btn_coords.setChecked(not self.ui.btn_coords.isChecked())
        elif event.key() == Qt.Key_5:
            self.ui.btn_show.setChecked(not self.ui.btn_show.isChecked())
        elif event.key() == Qt.Key_A:
            self.set_changes()
        elif event.key() == Qt.Key_Shift:
            self.editor.shift_down = True

    def set_mode(self, mode: int) -> None:
        """
        Method to change the displayed mode

        :param mode: The mode to select
        :return: None
        """
        self.enable_editing_widgets(False)
        if mode == 0:
            self.ui.btn_view.setChecked(True)
        elif mode == 1:
            self.ui.btn_add.setChecked(True)
        elif mode == 2:
            self.ui.btn_edit.setChecked(True)

    def display_hash(self, hash_: str) -> None:
        """
        Method to display the hash of the selected ROI

        :param hash_: The hasht to display
        :return: None
        """
        self.ui.lbl_hash.setText(hash_)

    def change_mode(self, id_, checked) -> None:
        """
        Method to change the editor mode

        :param id_: The id of the toggled button, assigned explicitly in initialize_ui:
                    0 view, 1 add, 2 edit
        :param checked: Whether that button became the checked one
        :return: None
        """
        if checked:
            # -1 view, 0 add, 1 edit -- EditorView's numbering, which starts one lower
            self.editor.change_mode(id_ - 1)

    def set_status(self, status: str) -> None:
        """
        Method to display a status in the status bar

        :param status: The status to display
        :return: None
        """
        self.ui.lbl_status.setText(status)


def ask_for_name(parent: QWidget, title: str, label: str) -> Tuple[str, bool]:
    """
    Function to ask the user for a name through a styled input dialog

    Uses a QInputDialog INSTANCE rather than the static QInputDialog.getText: the static call builds
    a dialog of its own, so styling an instance and handing it over as the parent -- which is what
    both call sites used to do -- applied inputbox.css to a widget that was never shown.

    Module level because ExperimentDialog and GroupDialog both need it and neither is the other's
    base class.

    :param parent: The dialog to parent the input box to
    :param title: The window title
    :param label: The prompt shown above the input field
    :return: The entered name, stripped of surrounding whitespace, and whether it was accepted
    """
    dial = QInputDialog(parent)
    dial.setWindowTitle(title)
    dial.setWindowIcon(Icon.get_icon("LOGO"))
    dial.setStyleSheet(Util.load_stylesheet("inputbox.css"))
    dial.setInputMode(QInputDialog.TextInput)
    dial.setLabelText(label)
    ok = dial.exec() == QDialog.Accepted
    return dial.textValue().strip(), ok


class ExperimentDialog(QDialog):
    #: Characters of an experiment's details shown in the list label before it is cut short
    DETAILS_LABEL_LENGTH = 47

    def __init__(self, data: Dict[str, List[str]] = None, *args, **kwargs):
        """
        :param data: Dict containing the keys and paths of the available images
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.data = data
        self.images = []
        # No self.ui / self.img_model / self.exp_model placeholders: initialize_ui() below assigns
        # all three on every path, so a None seeded first is never observable -- it only made their
        # declared type Optional and every later use an error on a possibly-None object
        # Image Loader for lazy loading
        self.update_timer = None
        self.initialize_ui()
        # Create connection to database
        self.inserter = Inserter()
        self.requester = Requester()
        # What each experiment looked like when it was read, so save_changes can write only what
        # the user actually changed
        self.loaded_state: Dict[str, Tuple] = {}
        self.load_experiments()

    def initialize_ui(self):
        # Annotated Any deliberately -- see DataExportDialog.initialize_ui above
        self.ui: Any = uic.loadUi(Paths.ui_exp_dial, self)
        # Define models for used lists
        self.img_model = QStandardItemModel(self.ui.lv_images)
        self.exp_model = QStandardItemModel(self.ui.lv_experiments)
        self.ui.lv_images.setModel(self.img_model)
        self.ui.lv_experiments.setModel(self.exp_model)
        self.ui.lv_images.setIconSize(Icon.get_icon_size("LIST_ITEM"))
        self.ui.lv_experiments.setIconSize(Icon.get_icon_size("LIST_ITEM"))
        # Connect add btn to dialog
        self.ui.btn_add_group.clicked.connect(self.open_group_dialog)
        self.ui.btn_add.clicked.connect(self.add_experiment)
        self.ui.btn_images_add.clicked.connect(self.add_images_to_experiment)
        self.ui.btn_images_remove.clicked.connect(self.remove_images_from_experiment)
        self.ui.btn_images_clear.clicked.connect(self.remove_all_images_from_experiment)
        self.ui.lv_experiments.selectionModel().selectionChanged.connect(self.on_exp_selection_change)
        # The image-selection handler was defined but never connected, so btn_images_remove was
        # never enabled or disabled in response to a selection. Connected after setModel(), which
        # is what creates the selection model
        self.ui.lv_images.selectionModel().selectionChanged.connect(self.on_image_selection_change)
        # ...and start in the state the handler would produce for an empty selection
        self.ui.btn_images_remove.setEnabled(False)
        # Set window title and icon
        self.setWindowTitle("Experiment Dialog")
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint)

    def on_image_selection_change(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        """
        Method to react to changes in the image selection

        :param selected: The selected items
        :param deselected: The deselected items
        :return: None
        """
        if selected:
            self.ui.btn_images_remove.setEnabled(True)
        else:
            self.ui.btn_images_remove.setEnabled(False)

    def add_experiment(self) -> None:
        """
        Method to add a new experiment

        :return: None
        """
        # The dialog that is built here is the one that is SHOWN. It used to be constructed and
        # styled, then handed to the STATIC QInputDialog.getText as a mere parent -- and the static
        # call builds a dialog of its own, so inputbox.css was applied to a widget nobody ever saw
        name, ok = ask_for_name(self, "Add new Experiment...", "Enter experiment name: ")
        if ok:
            existing = {self.exp_model.item(row).data()["name"]
                        for row in range(self.exp_model.rowCount())}
            # Both were accepted before: an empty name produced an experiment that cannot be
            # picked out of the list, and a duplicate produced two rows that add_new_experiment
            # then collapses back into one on save
            if not name or name in existing:
                QMessageBox.information(
                    self, "Add new Experiment...",
                    "Please enter a name." if not name
                    else f"An experiment named '{name}' already exists.")
                return
            add_item = QStandardItem()
            text = f"{name}\nNo Details\nGroups: No groups"
            add_item.setText(text)
            add_item.setData(
                {"name": name,
                 # None, not the name: this experiment has no row in the database yet, so there is
                 # nothing for save_changes to rename FROM
                 "loaded_name": None,
                 "details": "",
                 "notes": "",
                 "groups": {},
                 "keys": [],
                 "image_paths": []}
            )
            add_item.setIcon(Icon.get_icon("CLIPBOARD"))
            self.exp_model.appendRow(add_item)

    def add_images_to_experiment(self) -> None:
        """
        Method to add the selected images to the selected experiment

        :return: None
        """
        # Get selected images
        # The selection is the COMPLETE intended set: the dialog is pre-seeded with the images the
        # experiment already has and get_selected_images returns everything selected, so assigning
        # it is right and a union would make deselection impossible. What was wrong is the guard --
        # `if keys and paths` discarded an empty selection, so removing the last image silently did
        # nothing. An accepted dialog with nothing selected is now honoured; only a cancel is not
        selection = self.open_image_selection_dialog()
        if selection is not None:
            keys, paths = selection
            # Get selected experiment
            selected_exp = self.exp_model.item(self.ui.lv_experiments.selectionModel().selectedIndexes()[0].row())
            data = selected_exp.data()
            # Clear img_model
            self.img_model.clear()
            data["keys"] = keys
            data["image_paths"] = paths
            selected_exp.setData(data)
            self.enable_experiment_buttons(False)
            if len(keys) > 25:
                self.update_timer = Loader(paths,
                                           feedback=self.add_image_items,
                                           processing=create_image_item_list_from)
            else:
                # Get items
                items = create_image_item_list_from(paths)
                for item in items:
                    self.img_model.appendRow(item)
                self.enable_experiment_buttons()

    def save_changes(self):
        """
        Method to write the dialog's pending changes back

        Named save_changes, NOT accepted: QDialog already has an `accepted` SIGNAL, and defining a
        method of that name over it makes the signal unreachable -- `dialog.accepted.connect(...)`
        would silently connect to nothing. This is invoked directly by the caller, which only ever
        worked because of that shadowing.
        """
        # Change the information for the last selected experiment
        sel = self.ui.lv_experiments.selectionModel().selectedIndexes()
        if sel:
            item = self.exp_model.itemFromIndex(sel[0])
            data = item.data()
            data["name"] = self.ui.le_name.text()
            data["details"] = self.ui.te_details.toPlainText()
            data["notes"] = self.ui.te_notes.toPlainText()
            item.setData(data)
        for ind in range(self.exp_model.rowCount()):
            item = self.exp_model.item(ind)
            data = item.data()
            # The name this experiment was READ under, or None for one added in this dialog. A
            # rename used to be indistinguishable from a new experiment: add_new_experiment is an
            # INSERT OR REPLACE keyed on the name, so it wrote a second row and left the first
            # standing with every group and image association still pointing at the old name
            loaded_name = data.get("loaded_name")
            if loaded_name is not None and loaded_name != data["name"]:
                data = self.apply_pending_rename(item, data, loaded_name)
            # Only what changed. add_new_experiment is an INSERT OR REPLACE, so an untouched
            # experiment had its details and notes rewritten from this model on every OK, along
            # with a group row and an image association per image it holds. An experiment absent
            # from loaded_state is new and always written.
            # Looked up under the name it was LOADED under, not its current one -- otherwise a
            # rename finds nothing in the snapshot and the comparison is meaningless. The
            # fingerprint includes the name, so a renamed experiment differs from its snapshot and
            # is written
            if self.get_experiment_fingerprint(data) == self.loaded_state.get(loaded_name or data["name"]):
                continue
            # Add experiment to database
            self.inserter.add_new_experiment(data["name"], data["details"], data["notes"])
            # REPLACED, not merged. add_image_to_experiment_group is an INSERT OR REPLACE and
            # nothing deleted from the groups table, so a removal made in this dialog or in the
            # group dialog was undone by this very loop -- the stored row survived, and
            # get_associated_images_for_experiment reads that table before images.experiment.
            # Deleting the experiment's rows first makes what the model holds authoritative, which
            # is also what preserves images that are not currently LOADED: they are still in
            # data["groups"], so they are written straight back
            self.inserter.remove_group_associations_for_experiment(data["name"])
            for group, values in data["groups"].items():
                for img in values:
                    self.inserter.add_image_to_experiment_group(img, data["name"], group)
            # Update data for images
            for key in data["keys"]:
                self.inserter.update_image_experiment_association(key, data["name"])
        self.inserter.commit_and_close()
        self.requester.connector.close_connection()

    def reject(self) -> None:
        """
        Method to close the dialog without writing anything

        The connection is closed on BOTH exit paths now. save_changes closes it on OK, but a cancel
        used to drop it with an open write transaction: remove_image_from_experiment and
        remove_all_images_from_experiment execute while the dialog is open, and sqlite holds a
        write lock from the first of those until the connection is committed, rolled back or
        closed. Measured on a scratch database -- a second writer gets
        `OperationalError: database is locked` while it is held, and the deletions are correctly
        discarded once it goes. CPython's refcounting made the window short rather than absent

        :return: None
        """
        self.inserter.connector.close_connection()
        self.requester.connector.close_connection()
        super().reject()

    def remove_images_from_experiment(self) -> None:
        """
        Method to remove the selected images from the selected experiment

        :return: None
        """
        # Get selected experiment
        exp = self.exp_model.itemFromIndex(self.ui.lv_experiments.selectionModel().selectedIndexes()[0])
        # Get selected images
        sel_images = self.ui.lv_images.selectionModel().selectedIndexes()
        # Highest row first: removeRows() shifts every row after the one it deletes, so the indexes
        # collected above go stale as soon as the first removal happens. Walking upwards means the
        # rows still to be deleted all sit below the one being deleted, and their indexes hold
        for index in sorted(sel_images, key=lambda i: i.row(), reverse=True):
            # Get key stored in item
            item_data = self.img_model.itemFromIndex(index).data()
            exp_data = exp.data()
            # list.remove() mutates in place and returns None -- assigning its result set the
            # experiment's key list to None, and save_changes() then raised TypeError iterating it.
            # Guarded rather than bare: a key that is not in the list is not an error here, the
            # image is simply not associated any more
            if item_data["key"] in exp_data["keys"]:
                exp_data["keys"].remove(item_data["key"])
            # ...and out of the GROUPS too. Experiment membership is read from the groups table,
            # not from images.experiment, so leaving the key here meant save_changes re-inserted
            # the row this method had just NULLed and the image came back on the next load
            for group in exp_data["groups"].values():
                if item_data["key"] in group:
                    group.remove(item_data["key"])
            self.inserter.remove_image_from_experiment(item_data["key"])
            exp.setData(exp_data)
            # Remove item from model
            self.img_model.removeRows(index.row(), 1)

    def remove_all_images_from_experiment(self) -> None:
        """
        Method to remove all assigned images from the selected experiment

        :return: None
        """
        # Remove keys from experiment data
        exp = self.exp_model.itemFromIndex(self.ui.lv_experiments.selectionModel().selectedIndexes()[0])
        exp_data = exp.data()
        exp_data["keys"] = []
        # Emptied for the same reason as in remove_images_from_experiment: the groups table decides
        # membership, so an experiment cleared here came back full on the next load
        exp_data["groups"] = {}
        exp.setData(exp_data)
        self.inserter.remove_all_images_from_experiment(exp_data["name"])
        # Clear image model
        self.img_model.clear()

    def open_image_selection_dialog(self) -> Optional[Tuple[List[str], List[str]]]:
        """
        Method to open the image selection dialog

        :return: The selected (keys, paths), or None if the user cancelled. An accepted dialog with
            nothing selected returns two EMPTY lists, which is a real answer and not the same thing
        """
        # Get hashes of images in img_model
        image_hashs = []
        for row in range(self.img_model.rowCount()):
            # Get item
            img_item = self.img_model.item(row)
            image_hashs.append(img_item.data()["key"])
        sel_dialog = ImageSelectionDialog(images=self.data["paths"],
                                          selected_images=image_hashs)
        code = sel_dialog.exec()
        if code == QDialog.Accepted:
            return sel_dialog.get_selected_images()
        # None, not ([], []): the caller has to tell "cancelled" from "accepted with nothing
        # selected", and both used to arrive as a pair of empty lists
        return None

    def load_experiments(self) -> None:
        """
        Method to load all existings experiments

        :return: None
        """
        exps = self.requester.get_all_experiments()
        # Iterate over all experiments
        for exp in sorted(exps):
            imgs = self.requester.get_associated_images_for_experiment(exp)
            # EVERY experiment is listed. This used to be guarded by
            # `if all(elem in self.data["keys"] for elem in imgs)`, so an experiment with a single
            # image missing from the currently loaded list vanished from the dialog completely --
            # the user saw an empty or short list and nothing saying why
            missing = [x for x in imgs if x not in self.data["keys"]]
            name = exp
            # get_info_for_experiment answers None for an experiment that is not in the
            # table -- unpacking that raises TypeError, and an experiment listed by
            # get_all_experiments but missing its details row is a partially written state, not a
            # reason to refuse to open the dialog
            info = self.requester.get_info_for_experiment(exp)
            details, notes = info if info else ("", "")
            groups = {}
            # Only the loaded images have a path here -- .index() raises for the others. The
            # unloaded ones stay in "keys" and in "groups" regardless, because save_changes
            # re-writes both from those, and dropping them would silently delete the
            # association just because the image was not open
            img_paths = [self.data["paths"][self.data["keys"].index(x)]
                         for x in imgs if x not in missing]
            for key in imgs:
                group = self.requester.get_associated_group_for_image(key, exp)
                if group in groups:
                    groups[group].append(key)
                else:
                    groups[group] = [key]
            add_item = QStandardItem()
            label = self.create_experiment_label(name, details, groups)
            if missing:
                label += f"\n({len(missing)} of {len(imgs)} images not loaded)"
            add_item.setText(label)
            add_item.setData(
                {
                    "name": name,
                    # The name as stored. save_changes compares it against "name" to tell a rename
                    # from an edit, and looks the experiment up in loaded_state by it
                    "loaded_name": name,
                    "details": details,
                    "notes": notes,
                    "groups": groups,
                    "keys": imgs,
                    "image_paths": img_paths
                }
            )
            add_item.setIcon(Icon.get_icon("CLIPBOARD"))
            self.loaded_state[name] = self.get_experiment_fingerprint(add_item.data())
            self.exp_model.appendRow(add_item)

    def apply_pending_rename(self, item: QStandardItem, data: Dict, loaded_name: str) -> Dict:
        """
        Method to carry out a pending rename of the given experiment

        Refuses a name that is empty or already taken and reverts to the loaded one. Inserter.
        rename_experiment updates rows BY NAME, so renaming onto an existing experiment would
        merge the two -- strictly worse than the duplicate this fix removes. add_experiment has
        always refused both for a NEW experiment; the line edit that renames never did, and it
        could not matter while a rename merely created a second row.

        :param item: The list item holding the experiment
        :param data: The item's data dictionary, with the new name already written into it
        :param loaded_name: The name the experiment was read from the database under
        :return: The data dictionary, with the name reverted if the rename was refused
        """
        taken = any(self.exp_model.item(row).data()["name"] == data["name"]
                    for row in range(self.exp_model.rowCount())
                    if self.exp_model.item(row) is not item)
        if not data["name"] or taken:
            QMessageBox.information(
                self, "Rename experiment...",
                "Please enter a name." if not data["name"]
                else f"An experiment named '{data['name']}' already exists.")
            data["name"] = loaded_name
            item.setData(data)
            return data
        self.inserter.rename_experiment(loaded_name, data["name"])
        return data

    @staticmethod
    def get_experiment_fingerprint(data: Dict) -> Tuple:
        """
        Method to reduce an experiment to a value that can be compared for equality

        Used to decide what save_changes has to write. Order-insensitive on both the image keys and
        the groups, because neither carries meaning here and the models rebuild them in whatever
        order the queries returned

        :param data: The data dictionary stored on the experiment's item
        :return: A hashable summary of everything save_changes would write
        """
        groups = tuple(sorted((group, tuple(sorted(images)))
                              for group, images in data["groups"].items()))
        return data["name"], data["details"], data["notes"], tuple(sorted(data["keys"])), groups

    def enable_experiment_buttons(self, enable: bool = True) -> None:
        """
        Method to enable the buttons relevant for experiment changes

        :param enable: The bool to pass to the setEnabled function of the buttons
        :return: None
        """
        self.ui.lv_experiments.setEnabled(enable)
        self.ui.btn_add.setEnabled(enable)
        self.ui.btn_remove.setEnabled(enable)
        self.ui.btn_images_add.setEnabled(enable)
        self.ui.btn_images_clear.setEnabled(enable)
        self.ui.btn_add_group.setEnabled(enable)

    def on_exp_selection_change(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        """
        Function to react to changed experiment selection

        :param selected: The selected item
        :param deselected: The deselected item
        :return: None
        """
        # Get selected experiment
        selected = selected.indexes()
        deselected = deselected.indexes()
        self.enable_experiment_buttons(False)
        # Store the current data to the deselected item
        if deselected:
            item = self.exp_model.item(deselected[0].row())
            self.store_current_information_to_item(item)
        # Clear the image list
        self.img_model.clear()
        if selected:
            self.enable_experiment_buttons(True)
            data = self.exp_model.item(selected[0].row()).data()
            # Insert data into textfields
            self.ui.le_name.setText(data["name"])
            self.ui.te_details.setPlainText(data["details"])
            self.ui.te_notes.setPlainText(data["notes"])
            # Enable text inputs for change
            self.enable_text_inputs()
            groups_str = self.create_groups_string(data["groups"].items())
            self.ui.le_groups.setText(groups_str)
            if len(data["image_paths"]) > 25:
                self.update_timer = Loader(data["image_paths"],
                                           feedback=self.add_image_items,
                                           processing=create_image_item_list_from)
            else:
                # Add the saved image items to img_model
                items = Util.create_image_item_list_from(data["image_paths"])
                for item in items:
                    self.img_model.appendRow(item)
                # Enable buttons for input
                self.ui.btn_images_add.setEnabled(True)
                self.ui.btn_remove.setEnabled(True)
            self.enable_experiment_buttons(True)
        else:
            self.clear_experiment_screen()
            self.enable_experiment_buttons(False)

    def enable_text_inputs(self, enable: bool = True) -> None:
        """
        Method to enable/disable the text inputs

        :param enable: If true, the text inputs will be enabled
        :return: None
        """
        self.ui.le_name.setEnabled(enable)
        self.ui.te_details.setEnabled(enable)
        self.ui.te_notes.setEnabled(enable)

    @staticmethod
    def create_groups_string(groups) -> str:
        """
        Method to create the groups string

        :param groups: The groups, either a mapping of group name to its image keys or an
                       iterable of (name, keys) pairs
        :return: The groups string
        """
        pairs = groups.items() if hasattr(groups, "items") else groups
        # Space-separated. The mapping form used to be built inline with a trailing space while
        # this method produced "A (3)B (2)" with no separator at all, so the same data rendered
        # two different ways depending on which one the caller happened to reach
        return " ".join(f"{name} ({len(keys)})" for name, keys in pairs)

    @classmethod
    def create_experiment_label(cls, name: str, details: str, groups) -> str:
        """
        Method to build the label shown for an experiment in the list

        One helper for what were three hand-built copies of the same f-string, each of which
        appended "..." unconditionally -- including when the details were shorter than the cut-off
        and nothing had actually been truncated.

        :param name: The experiment name
        :param details: The experiment details, truncated for display
        :param groups: The groups, in either form create_groups_string accepts
        :return: The label
        """
        shortened = f"{details[:cls.DETAILS_LABEL_LENGTH]}..." \
            if len(details) > cls.DETAILS_LABEL_LENGTH else details
        return f"{name}\n{shortened}\nGroups: {cls.create_groups_string(groups)}"

    def store_current_information_to_item(self, item: QStandardItem) -> None:
        """
        Method to store the current information to the given item

        :param item: The item to store the information in
        :return: None
        """
        name = self.ui.le_name.text()
        details = self.ui.te_details.toPlainText()
        notes = self.ui.te_notes.toPlainText()
        keys = []
        img_paths = []
        # Iterate over all images
        for row in range(self.img_model.rowCount()):
            # Get item
            img_item = self.img_model.item(row)
            # Get item data and append key
            keys.append(img_item.data()["key"])
            img_paths.append(img_item.data()["path"])
        # UPDATED, not rebuilt. This used to assign a fresh six-key dictionary, which silently
        # dropped "loaded_name" -- the name the experiment was read under, added 2026-09-01 so that
        # save_changes can tell a rename from a new experiment. The method runs on every
        # DESELECTION, so renaming an experiment and then clicking a different one restored the
        # duplicate the rename fix had removed. A literal cannot carry a key added after it
        data = item.data()
        groups = data["groups"]
        data.update(
            {
                "name": name,
                "details": details,
                "notes": notes,
                "groups": groups,
                "keys": keys,
                "image_paths": img_paths
            }
        )
        item.setData(data)
        item.setText(self.create_experiment_label(name, details, groups))

    def clear_experiment_screen(self) -> None:
        """
        Method to restore the experiment screen

        :return:None
        """
        # Clear everything if selection was cleared
        self.ui.le_name.clear()
        self.ui.te_details.clear()
        self.ui.te_notes.clear()
        self.ui.le_groups.clear()
        # Disable text inputs until an experiment was selected
        self.ui.le_name.setEnabled(False)
        self.ui.te_details.setEnabled(False)
        self.ui.te_notes.setEnabled(False)
        # Disable buttons to prevent unnecessary input
        self.enable_experiment_buttons(False)
        self.ui.btn_remove.setEnabled(False)

    def enable_associated_buttons(self, enable: bool) -> None:
        """
        Method to enable/disable group and image buttons
        :param enable: If true, buttons will be enabled
        :return: None
        """
        self.ui.btn_images_add.setEnabled(enable)
        self.ui.btn_images_remove.setEnabled(enable)
        self.ui.btn_images_clear.setEnabled(enable)
        self.ui.btn_add_group.setEnabled(enable)

    def add_image_items(self, items: List[QStandardItem], finished: bool = False) -> None:
        """
        Method to add items to the image list

        :param items: The items to add
        :param finished: True when the loader has no items left. Asked rather than inferred from an
        empty batch -- processing can empty one long before the end
        :return: None
        """
        for item in items:
            self.img_model.appendRow(item)
        self.ui.prg_images.setValue(int(self.update_timer.percentage * 100))
        if finished:
            # Enable buttons for input
            self.enable_experiment_buttons()

    def open_group_dialog(self):
        # Get the selected experiment
        exp_index = self.ui.lv_experiments.selectionModel().selectedIndexes()
        exp = self.exp_model.itemFromIndex(exp_index[0])
        exp_data = exp.data()
        group_dial = GroupDialog(data=exp_data)
        code = group_dial.exec()
        if code == QDialog.Accepted:
            groups = {}
            for row in range(group_dial.group_model.rowCount()):
                item = group_dial.group_model.item(row)
                data = item.data()
                groups[data["name"]] = data["keys"]
            exp_data["groups"] = groups
            exp.setData(exp_data)
            group_str = self.create_groups_string(groups)
            exp.setText(self.create_experiment_label(exp_data["name"], exp_data["details"], groups))
            self.ui.le_groups.setText(group_str)


class StatisticsDialog(QDialog):
    """
    Dialog to show statistical analysis of data
    """
    stat_calculation_finished_signal = pyqtSignal()
    def __init__(self, experiment: str, active_channels: Dict, *args, **kwargs):
        """
        :param experiment: The experiment to show
        :param active_channels: The channels to analyse
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.ui = None
        self.experiment = experiment
        self.active_channels = active_channels
        self.requester = Requester()
        self.data = self.get_group_data()
        self.statistics = None
        self.comparison_groups: List = []
        self.initialize_ui()
        self._add_group_boxes()
        self.settings = copy.copy(Plots.STANDARD_SETTINGS)
        self.general_settings = {
            "show_ns": True,
            "show_title": True,
            "show_xlabel": True,
            "show_ylabel": True,
            "show_legend": True,
            "legend_fontsize": 7,
            "show_grid": True,
            "minor_steps": 5,
            "major_steps": 10,
            "violin_inner": "quartile",
            "violin_split": True,
            "orientation": "vertical",
            "palette": "husl"
        }
        # Connected BEFORE any work is started. It used to follow plot_data(), which is safe only
        # while nothing in plot_data reaches _calculate_statistics -- an invariant nothing enforces
        self.stat_calculation_finished_signal.connect(self._display_calculated_statistics)
        self.plot_data()

    def initialize_ui(self) -> None:
        """
        Method to intialize the ui

        :return: None
        """
        # Annotated Any deliberately -- see DataExportDialog.initialize_ui above
        self.ui: Any = uic.loadUi(Paths.ui_stat_dial, self)
        self.list_widget = None
        self.menu_bar = StatisticsDialogMenuBar(parent=self)
        self.layout().insertWidget(0, self.menu_bar)
        # Set window and style
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowTitle(f"Statistics for {self.experiment}")
        self.setStyleSheet(Util.load_stylesheet("main.css"))
        self._initialize_plot_widgets()
        self.ui.tv_group_data.setModel(DataFrameModel(self.data))
        # Sorting was never switched on, on either view -- a QTableView does not sort on a header
        # click unless it is told to, so the header was inert. Safe because DataFrameModel.set_df
        # stores df.copy(): sorting the view cannot reorder StatisticsDialog.data, and with it every
        # statistic and every export derived from it
        self.ui.tv_group_data.setSortingEnabled(True)
        self.ui.tv_group_data.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.tv_group_data.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.btn_statistics.clicked.connect(self.calculate_and_display_statistics)
        self.ui.btn_statistics.setEnabled(False)
        self.ui.lbl_statistics.setVisible(False)
        self.ui.tv_group_statistics.setVisible(False)
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint)
        self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)

    def _initialize_plot_widgets(self):
        self.canvas = PlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cmbx_type = QComboBox()
        self.cmbx_type.addItems(PlotCanvas.plot_types)
        self.cmbx_type.currentTextChanged.connect(self.change_plot_type)
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.toolbar)
        h_layout.addWidget(self.cmbx_type)
        self.ui.vl_data.addLayout(h_layout)
        self.ui.vl_data.addWidget(self.canvas)

    def initialize_statistics_table(self) -> None:
        """
        Method to populate the data and statistics tables

        :return: None
        """
        if not isinstance(self.ui.tv_group_statistics.model(), DataFrameModel):
            self.ui.tv_group_statistics.setModel(DataFrameModel(self.statistics))
            # See tv_group_data above -- both views take their sorting from the same model class
            self.ui.tv_group_statistics.setSortingEnabled(True)
        else:
            # Reset the view and add the newly calculated data
            self.ui.tv_group_statistics.model().setDataFrame(self.statistics)
        self.ui.tv_group_statistics.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.tv_group_statistics.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def change_experiment(self):
        """
        Method to change the currently viewed experiment

        :return: None
        """
        exp_sel_dial = ExperimentSelectionDialog()
        code = exp_sel_dial.exec()
        if code == QDialog.Accepted:
            self.experiment = exp_sel_dial.get_selected_experiment()
            self.active_channels = exp_sel_dial.get_active_channels()
            # ASSIGNED, not discarded. self.data kept the previous experiment's DataFrame, so the
            # table, the plot and every statistic described the old experiment under the new title
            self.data = self.get_group_data()
            # The layout teardown that stood here was dead: it looked for sub-layouts in vl_groups,
            # but _add_group_boxes puts a single QListWidget there, so the branch never ran -- and
            # its inner loop reused the outer loop variable. The real reset is the
            # list_widget.clear() inside _add_group_boxes
            if self.ui.tv_group_statistics.model() is not None:
                self.ui.tv_group_statistics.model().setDataFrame(pd.DataFrame())
            self._add_group_boxes()
            # Plot the new data
            self.plot_statistics()

    def export_data(self):
        """
        Method to export the group and statistics data

        :return: None
        """
        self.data.to_csv(os.path.join(Paths.result_path, f"{self.experiment}_group_data.csv"))
        if self.statistics is not None:
            self.statistics.to_csv(os.path.join(Paths.result_path, f"{self.experiment}_group_data_statistics.csv"))
        msg = QMessageBox()
        msg.setWindowIcon(Icon.get_icon("LOGO"))
        msg.setIcon(QMessageBox.Information)
        msg.setStyleSheet(Util.load_stylesheet("messagebox.css"))
        msg.setWindowTitle("Data exported!")
        msg.setText("Data successfully exported!")
        msg.setInformativeText(f"CSV files can be found at {Paths.result_path}")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def calculate_and_display_statistics(self) -> None:
        """
        Method to calculate the necessary statistics

        The result is not returned: the work runs on a worker thread and comes back through
        stat_calculation_finished_signal. The annotation used to claim a
        Dict[str, Dict[str, List]] that the method never produced.

        :return: None
        """
        # daemon, so closing the window does not leave the interpreter alive waiting on a
        # statistics run nobody is going to look at
        thread = Thread(target=self._calculate_statistics, daemon=True)
        self.setEnabled(False)
        thread.start()

    def _calculate_statistics(self) -> None:
        self.statistics = perform_statistical_analysis_on_groups(
            data=self.data,
            comparison_groups=self.get_comparison_groups()
        )
        self.stat_calculation_finished_signal.emit()

    def _display_calculated_statistics(self) -> None:
        self.ui.lbl_statistics.setVisible(True)
        self.ui.tv_group_statistics.setVisible(True)
        self.initialize_statistics_table()
        self.plot_statistics()
        self.setEnabled(True)

    def _add_group_boxes(self) -> None:
        """
        Private method to populate the group section

        :return: None
        """
        # Check if the list widget was already created
        if not self.list_widget:
            self.list_widget = QListWidget(self)
            self.list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            self.list_widget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
            # Enable drag&drop reordering
            self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
            self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
            self.ui.vl_groups.addWidget(self.list_widget)
            self.list_widget.itemChanged.connect(self.enable_statistics_button)
            self.list_widget.model().rowsMoved.connect(self.plot_data)
        else:
            self.list_widget.clear()
        # Get the respective groups
        for group in sorted(self.data["Group"].unique()):
            item = QListWidgetItem(group)
            # Make the item checkable and draggable
            flags = item.flags()
            flags |= QtCore.Qt.ItemIsUserCheckable
            flags |= QtCore.Qt.ItemIsDragEnabled
            flags |= QtCore.Qt.ItemIsEnabled
            item.setFlags(flags)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.list_widget.addItem(item)

    def enable_statistics_button(self) -> None:
        """
        Method to chec kif the statistics button should be enabled

        :return: None
        """
        self.ui.btn_statistics.setEnabled(bool(self.get_comparison_groups()))


    def get_comparison_groups(self) -> list[str]:
        """
        Private method to identify active comparison groups

        :return: Dictionary with each group and its activation status
        """
        # list_widget is None only between initialize_ui and _add_group_boxes, both of which run in
        # __init__ with nothing in between -- and _add_group_boxes has no early return, so it always
        # creates the widget. Asserted rather than branched on: a `if self.list_widget` guard here
        # would return an empty group list on a state that cannot occur, which reads as "no groups
        # are selected" and silently disables the statistics button instead of failing
        assert self.list_widget is not None, "_add_group_boxes must run before the groups are read"
        active_groups = []
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item.checkState() == QtCore.Qt.Checked:
                active_groups.append(item.text())
        return active_groups

    def get_group_ordering(self):
        """
        Method to get the current top-to-bottom order of the group list

        :return: The group names, in the order the user has dragged them into
        """
        # Same invariant as get_comparison_groups -- see the note there
        assert self.list_widget is not None, "_add_group_boxes must run before the groups are read"
        ordering = []
        for row in range(self.list_widget.count()):
            ordering.append(self.list_widget.item(row).text())
        return ordering

    def get_group_data(self) -> pd.DataFrame:
        """
        Function to get the data for each group

        :return: None
        """
        data_header = DataExportDialog.STANDARD_HEADER + ["Channel", "Foci"]
        data_header.insert(2, "Group")
        # Load the data
        data = pd.DataFrame(self.requester.get_table_data_for_experiment(self.experiment),
                            columns=data_header)
        # Remove the unnecessary columns
        data = data[["Group", "Channel", "Foci"]]
        # THE CHANNEL SELECTION IS APPLIED HERE, and until 2026-09-20 it was applied nowhere:
        # active_channels was accepted by __init__, reassigned on every experiment change and read
        # by nothing, so ticking the boxes in ExperimentSelectionDialog changed neither the table
        # nor the plot. Reported from real use twice -- UI rows 7 and 8 -- before it was wired up.
        #
        # Filtered at the single point where the frame is built, so the table, the plot, the
        # statistics and the CSV export all see the same rows; every one of them reads self.data.
        # The dialog's boxes carry the non-main channel names, which is exactly what the Channel
        # column holds (get_table_data_for_image takes them from get_channel_names(image, False)).
        active = [name for name, selected in self.active_channels.items() if selected]
        # An empty mapping means "nobody has chosen", not "choose nothing": StatisticsDialog can be
        # built directly with {}, and filtering to nothing there would silently empty a dialog that
        # used to show everything. An empty SELECTION cannot arrive here at all -- the selection
        # dialog refuses to close on one.
        if active:
            data = data[data["Channel"].isin(active)]
        # Tell pandas which dtypes to use
        data["Foci"] = pd.to_numeric(data["Foci"], errors="coerce")
        return data

    def change_plot_type(self, text: str):
        self.plot_data()
        self.plot_statistics()

    def plot_data(self) -> None:
        self.canvas.plot(self.data,
                         title=self.experiment,
                         plot_type=self.ui.cmbx_type.currentText(),
                         ordering=self.get_group_ordering(),
                         settings = self.settings,
                         specific_settings=self.general_settings)

    def plot_statistics(self) -> None:
        self.plot_data()
        self.canvas.display_statistics(statistics=self.statistics,
                                       data=self.data,
                                       ordering=self.get_group_ordering(),
                                       show_ns=self.general_settings["show_ns"])

    def update_settings(self, settings: dict, special_settings: Dict) -> None:
        """
        Method to update the used settings for plotting

        :param settings: The general settings
        :param special_settings: The special settings (aka the settings not used by matplotlib through RCI)
        :return: None
        """
        self.settings = settings
        self.general_settings = special_settings
        self.plot_data()
        self.plot_statistics()


class DataFrameModel(QAbstractTableModel):
    # Class to display pandas DataFrames
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.set_df(df)

    def set_df(self, df) -> None:
        """
        Method to initialize the DataFrameModel

        :param df: The data to display as pandas DataFrame
        :return:None
        """
        # _df, not data: QAbstractTableModel.data() is the model's own read method, and an
        # attribute of that name made model.data(index, role) raise
        # "TypeError: 'DataFrame' object is not callable". Qt's rendering was unaffected -- PyQt
        # resolves the virtual through the type -- so it was a trap for Python callers and readers
        # rather than a rendering bug.
        # copy(), because sort() reorders this frame: it used to be the caller's own object, so
        # clicking a column header silently reordered StatisticsDialog.data as a side effect
        self._df = df.copy()
        self._values = self._df.to_numpy(copy=False)
        self._columns = self._df.columns.to_list()
        self._index = self._df.index.to_list()

    def setDataFrame(self, df):
        self.beginResetModel()
        self.set_df(df)
        self.endResetModel()

    def rowCount(self, parent=None) -> int:
        """
        Method to return the number of rows in the DataFrameModel

        :param parent: Parent of this widget
        :return: The number of rows as int
        """
        return self._values.shape[0]

    def columnCount(self, parent=None) -> int:
        """
        Method to return the number of columns in the DataFrameModel

        :param parent: The parent of this widget
        :return: The number of columns as int
        """
        return self._values.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()

        r, c = index.row(), index.column()
        val = self._values[r, c]

        if role == Qt.DisplayRole:
            # Fast, reasonable formatting
            if pd.isna(val):
                return ""
            if isinstance(val, float):
                return f"{val:.6g}"  # scientific-ish, compact
            return str(val)

        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal:
            return self._columns[section]
        else:
            return str(self._index[section])

    def sort(self,
             column: int,
             order: int) -> None:
        """
        Method to enable sorting

        :param column: The column to use for sorting
        :param order: The ordering to use for sorting
        :return: None
        """
        # Optional: enable sorting; for large DF this is still decent
        self.layoutAboutToBeChanged.emit()
        ascending = order == Qt.AscendingOrder
        # inplace on the model's OWN copy -- see set_df. The frame handed in by the caller is no
        # longer touched, so sorting the view cannot reorder the dialog's data underneath it
        self._df.sort_values(self._columns[column],
                             ascending=ascending,
                             inplace=True,
                             kind="mergesort")
        self._values = self._df.to_numpy(copy=False)
        self._index = self._df.index.to_list()
        self.layoutChanged.emit()


class StatisticsDialogMenuBar(QMenuBar):
    # Class to add a menu bar to the statistics dialog

    def __init__(self, parent: "StatisticsDialog", *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        # Kept as its own reference rather than re-derived from self.parent() at each use. Qt types
        # parent() as QObject and it is reparentable, so the six calls below would become
        # AttributeError inside a slot -- where the traceback goes to stderr and the UI just does
        # nothing. Named _dialog, not _parent: "parent" is a QObject method and assigning it would
        # shadow the accessor
        self._dialog = parent
        self._initialize_selection_menu()
        self._initialize_settings_menu()

    def _initialize_selection_menu(self):
        self.file_menu = QMenu("File", self)
        # Add actions
        self.act_change = QAction("Change experiment", self)
        self.act_export = QAction("Export results…", self)
        self.act_close = QAction("Close", self)
        # Add triggers
        self.act_change.triggered.connect(self._dialog.change_experiment)
        self.act_export.triggered.connect(self._dialog.export_data)
        self.act_close.triggered.connect(self._dialog.close)
        # Add the actions to the menu
        self.file_menu.addAction(self.act_change)
        self.file_menu.addAction(self.act_export)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.act_close)
        self.addMenu(self.file_menu)

    def _initialize_settings_menu(self):
        self.settings_menu = QMenu("Diagram Settings", self)
        self.act_show_ns = QAction("Show non-significant", self, checkable=True, checked=True)
        self.act_show_title = QAction("Show title", self, checkable=True, checked=True)
        self.act_show_xlabel = QAction("Show xlabel", self, checkable=True, checked=True)
        self.act_show_ylabel = QAction("Show ylabel", self, checkable=True, checked=True)
        self.act_open_diag_set = QAction("Expanded Settings...")
        # Add triggers
        self.act_open_diag_set.triggered.connect(self.open_plot_settings)
        self.act_show_ns.triggered.connect(self._update_parent_settings)
        self.act_show_title.triggered.connect(self._update_parent_settings)
        self.act_show_xlabel.triggered.connect(self._update_parent_settings)
        self.act_show_ylabel.triggered.connect(self._update_parent_settings)
        # Add the actions to the settings menu
        self.settings_menu.addAction(self.act_show_ns)
        self.settings_menu.addAction(self.act_show_title)
        self.settings_menu.addAction(self.act_show_xlabel)
        self.settings_menu.addAction(self.act_show_ylabel)
        self.settings_menu.addSeparator()
        self.settings_menu.addAction(self.act_open_diag_set)
        self.addMenu(self.settings_menu)

    def _update_parent_settings(self):
        psett = self._dialog.general_settings
        psett["show_ns"] = self.act_show_ns.isChecked()
        psett["show_title"] = self.act_show_title.isChecked()
        psett["show_xlabel"] = self.act_show_xlabel.isChecked()
        psett["show_ylabel"] = self.act_show_ylabel.isChecked()
        self._dialog.plot_statistics()


    def open_plot_settings(self):
        sett = PlotSettingsDialog()
        code = sett.exec()
        if code == QDialog.Accepted:
            sett.settings["show_ns"] = self.act_show_ns.isChecked()
            sett.settings["show_title"] = self.act_show_title.isChecked()
            sett.settings["show_xlabel"] = self.act_show_xlabel.isChecked()
            sett.settings["show_ylabel"] = self.act_show_ylabel.isChecked()
            self._dialog.update_settings(sett.settings,
                                         sett.specific_settings)


class PlotSettingsDialog(QDialog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(300)
        self.update_timer.timeout.connect(self.update_plots)
        self.initialize_ui()
        self.settings = copy.copy(Plots.STANDARD_SETTINGS)
        self.specific_settings = {
            "show_legend": True,
            "show_title": True,
            "show_xlabel": True,
            "show_ylabel": True,
            "legend_fontsize": 10,
            "show_grid": True,
            "palette": "husl",
            "violin_split": True,
            "violin_inner": "quartile",
            "orientation": "vertical",
            "minor_steps": 5,
            "major_steps": 10
        }
        self.create_mockup_diagram()

    def accept(self):
        self.update_settings()
        super().accept()

    def initialize_ui(self):
        # Annotated Any deliberately -- see DataExportDialog.initialize_ui above
        self.ui: Any = uic.loadUi(Paths.ui_stat_plot_settings_dial, self)
        # Initialize the combo box
        self.ui.cmbx_font.addItems(sorted(self.get_available_fonts()))
        self.ui.cmbx_font.setCurrentText("DejaVu Sans")
        self.ui.cmbx_palette.addItems(Plots.COLOR_PALETTES)
        self.ui.cmbx_palette.setCurrentText("deep")
        self.ui.cmbx_violin_inner.addItems(["quartile", "box", "stick", "point"])
        self.ui.cmbx_orientation.addItems(["vertical", "horizontal"])
        self._connect_widgets_to_update_timer()
        # Set window and style
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowTitle("Diagram Settings")
        self.setStyleSheet(Util.load_stylesheet("main.css"))
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint)

    def _connect_widgets_to_update_timer(self):
        """
        Method to make every settings widget schedule one delayed redraw

        Each connection goes through `_schedule_redraw`, which takes no arguments, rather than
        straight to `QTimer.start`. That indirection is the whole point: `start` accepts an
        optional interval in milliseconds and **overrides the configured one**, so connecting a
        value-carrying signal to it hands the widget's own value in as the delay. Measured against
        real widgets and a timer configured exactly as this one is: a DPI spin box at 2500 made the
        preview wait 2.5 seconds, a step count of 0 fired on the next event-loop pass, and every
        check box and combo box drove the interval to 0 or 2 ms -- re-rendering on every tick,
        which is precisely what the 300 ms debounce exists to prevent.

        The check boxes are the same `stateChanged`-emits-an-int trap that has now produced four
        defects in this project. `toggled` would carry a bool, but the argument is unwanted here
        either way, so all seventeen are uniform.
        """
        self.ui.spb_h_size.valueChanged.connect(self._schedule_redraw)
        self.ui.spb_v_size.valueChanged.connect(self._schedule_redraw)
        self.ui.spb_dpi.valueChanged.connect(self._schedule_redraw)
        self.ui.cmbx_font.currentTextChanged.connect(self._schedule_redraw)
        self.ui.spb_title_size.valueChanged.connect(self._schedule_redraw)
        self.ui.spb_axis_size.valueChanged.connect(self._schedule_redraw)
        self.ui.spb_tick_size.valueChanged.connect(self._schedule_redraw)
        self.ui.cmbx_palette.currentTextChanged.connect(self._schedule_redraw)
        self.ui.cbx_show_legend.stateChanged.connect(self._schedule_redraw)
        self.ui.spb_legend_font_size.valueChanged.connect(self._schedule_redraw)
        self.ui.cbx_grid_show.stateChanged.connect(self._schedule_redraw)
        self.ui.cbx_ticks_minor.stateChanged.connect(self._schedule_redraw)
        self.ui.spb_steps_major.valueChanged.connect(self._schedule_redraw)
        self.ui.spb_steps_minor.valueChanged.connect(self._schedule_redraw)
        self.ui.cmbx_orientation.currentTextChanged.connect(self._schedule_redraw)
        self.ui.cbx_violin_split.stateChanged.connect(self._schedule_redraw)
        self.ui.cmbx_violin_inner.currentTextChanged.connect(self._schedule_redraw)

    def _schedule_redraw(self) -> None:
        """
        Method to restart the redraw debounce at its configured interval

        Takes no arguments on purpose -- see `_connect_widgets_to_update_timer`. Do not connect a
        widget signal to `self.update_timer.start` directly.

        :return: None
        """
        self.update_timer.start()

    #: The system font names, resolved once per process. Registering them is global and
    #: cumulative -- matplotlib's font manager keeps every font ever added -- so doing it on every
    #: open of this dialog paid the same cost repeatedly and grew that manager each time
    _font_names = None

    @classmethod
    def get_available_fonts(cls):
        """
        Method to get the names of the fonts available to matplotlib

        :return: The font names, resolved on the first call and cached afterwards
        """
        if cls._font_names is None:
            for font_file in font_manager.findSystemFonts():
                font_manager.fontManager.addfont(font_file)
            cls._font_names = font_manager.get_font_names()
        return cls._font_names

    def create_mockup_diagram(self):
        self.canvas_violin = PlotCanvas(self)
        self.canvas_box = PlotCanvas(self)
        # A local Generator, not np.random.seed(42): seeding sets the PROCESS-WIDE random state,
        # so merely opening this settings dialog made every later consumer of the global stream
        # deterministic. The mockup stays reproducible, which is all the seed was for
        rng = np.random.default_rng(42)
        # Create mockup data
        group_a = [("Control", "Channel Y", x) for x in rng.integers(0, 30, size=35)]
        group_a.extend([("Control", "Channel X", x) for x in rng.integers(0, 45, size=78)])
        group_b = [("Test", "Channel Y", x) for x in rng.integers(0, 145, size=25)]
        group_b.extend([("Test", "Channel X", x) for x in rng.integers(0, 112, size=44)])
        self.data = pd.DataFrame(group_a + group_b, columns=["Group", "Channel", "Foci"])
        v_lay = QVBoxLayout()
        v_lay.addWidget(self.canvas_violin)
        v_lay.addWidget(self.canvas_box)
        self.ui.hl_data.addLayout(v_lay, stretch=3)
        self.update_plots()

    def update_plots(self):
        self.update_settings()
        self.canvas_violin.plot(data=self.data,
                                plot_type=PlotCanvas.plot_types[0],
                                settings=self.settings,
                                specific_settings=self.specific_settings)
        self.canvas_box.plot(data=self.data,
                             title="Box Plot",
                             plot_type=PlotCanvas.plot_types[1],
                             settings=self.settings,
                             specific_settings=self.specific_settings)

    def update_settings(self):
        self.settings["figure.figsize"] = self.ui.spb_h_size.value(), self.ui.spb_v_size.value()
        self.settings["figure.dpi"] = self.ui.spb_dpi.value()
        self.settings["savefig.dpi"] = self.ui.spb_dpi.value()
        self.settings["font.family"] = self.ui.cmbx_font.currentText()
        self.settings["axes.titlesize"] = self.ui.spb_title_size.value()
        self.settings["axes.labelsize"] = self.ui.spb_axis_size.value()
        self.settings["xtick.labelsize"] = self.ui.spb_tick_size.value()
        self.settings["ytick.labelsize"] = self.ui.spb_tick_size.value()
        self.settings["ytick.minor.visible"] = self.ui.cbx_ticks_minor.isChecked()
        self.specific_settings["show_grid"] = self.ui.cbx_grid_show.isChecked()
        self.specific_settings["show_legend"] = self.ui.cbx_show_legend.isChecked()
        self.specific_settings["legend_fontsize"] = self.ui.spb_legend_font_size.value()
        self.specific_settings["palette"] = self.ui.cmbx_palette.currentText()
        self.specific_settings["major_steps"] = self.ui.spb_steps_major.value()
        self.specific_settings["minor_steps"] = self.ui.spb_steps_minor.value()
        self.specific_settings["violin_split"] = self.ui.cbx_violin_split.isChecked()
        self.specific_settings["violin_inner"] = self.ui.cmbx_violin_inner.currentText()
        self.specific_settings["orientation"] = self.ui.cmbx_orientation.currentText()


class GroupDialog(QDialog):

    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        # No self.ui / self.img_model / self.group_model / self.prg_bar placeholders -- see
        # ExperimentDialog above; initialize_ui() assigns all four unconditionally
        self.update_timer = None
        self.inserter = Inserter()
        self.initialize_ui()
        self.load_groups()

    def initialize_ui(self) -> None:
        """
        Method to initialize the UI

        :return: None
        """
        # Annotated Any deliberately -- see DataExportDialog.initialize_ui above
        self.ui: Any = uic.loadUi(Paths.ui_exp_dial_group_dial, self)
        self.img_model = QStandardItemModel(self.ui.lv_images)
        self.group_model = QStandardItemModel(self.ui.lv_groups)
        self.prg_bar = self.ui.prg_images
        self.ui.lv_images.setModel(self.img_model)
        self.ui.lv_groups.setModel(self.group_model)
        self.ui.lv_images.setIconSize(Icon.get_icon_size("LIST_ITEM"))
        self.ui.lv_groups.setIconSize(Icon.get_icon_size("LIST_ITEM"))
        # Connect UI to functionality
        self.ui.btn_add.clicked.connect(self.add_group)
        self.ui.btn_remove.clicked.connect(self.remove_group)
        self.ui.lv_groups.selectionModel().selectionChanged.connect(self.on_group_selection_change)
        # The three image buttons, all acting on lv_images. btn_clear_images and the selection slot
        # were written but never connected, so Clear did nothing and the other two stayed enabled
        # with nothing selected
        self.ui.btn_add_images.clicked.connect(self.add_images_to_group)
        self.ui.btn_remove_image.clicked.connect(self.remove_selected_image)
        self.ui.btn_clear_images.clicked.connect(self.clear_images)
        self.ui.lv_images.selectionModel().selectionChanged.connect(self.on_img_selection_change)
        # Nothing is selected and the list is empty until a group is chosen
        self.ui.btn_remove_image.setEnabled(False)
        self.ui.btn_clear_images.setEnabled(False)
        self.setStyleSheet(Util.load_stylesheet("main.css"))
        self.setWindowTitle("Group Dialog")
        self.setWindowIcon(Icon.get_icon("LOGO"))
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowSystemMenuHint |
                            QtCore.Qt.WindowMinMaxButtonsHint)

    def load_groups(self) -> None:
        """
        Method to load existing groups from the database

        :return: None
        """
        for group, keys in self.data["groups"].items():
            group_item = QStandardItem()
            data = {
                "name": group,
                "exp": self.data["name"],
                "keys": keys
            }
            group_item.setText(f"{data['name']}:\nImages: {len(keys)}")
            group_item.setData(data)
            self.group_model.appendRow(group_item)

    def add_images_to_group(self) -> None:
        """
        Method to add images to the selected group

        :return: None
        """
        # Get selected images
        # See ExperimentDialog.add_images_to_experiment: the returned selection is the complete
        # intended set, and an accepted-but-empty one must clear the group rather than be ignored
        selection = self.open_image_selection_dialog()
        if selection is not None:
            keys, paths = selection
            # Get selected group
            index = self.ui.lv_groups.selectionModel().selectedIndexes()[0]
            # Change group data
            item = self.group_model.itemFromIndex(index)
            data = item.data()
            data["keys"] = keys
            item.setData(data)
            item.setText(
                f"{data['name']}:\nImages: {len(keys)}"
            )
            self.img_model.clear()
            if len(paths) > 25:
                self.update_timer = Loader(paths, feedback=self.add_image_items,
                                           processing=create_image_item_list_from)
                self.setEnabled(False)
            else:
                # Create image items
                items = Util.create_image_item_list_from(paths, indicate_progress=False)
                for item in items:
                    self.img_model.appendRow(item)
            self.refresh_image_buttons()

    def add_image_items(self, items: List[QStandardItem], finished: bool = False) -> None:
        """
        Method to load the images given by the list paths

        :param items: QStandardItems to add to the images list
        :param finished: True when the loader has no items left. Asked rather than inferred from an
        empty batch -- processing can empty one long before the end
        :return: None
        """
        for img in items:
            self.img_model.appendRow(img)
        self.prg_bar.setValue(int(self.update_timer.percentage * 100))
        if finished:
            self.setEnabled(True)
        self.refresh_image_buttons()

    def open_image_selection_dialog(self) -> Optional[Tuple[List[str], List[str]]]:
        """
        Method to open the image selection dialog

        :return: The selected (keys, paths), or None if the user cancelled. An accepted dialog with
            nothing selected returns two EMPTY lists, which is a real answer and not the same thing
        """
        # Get selected group
        index = self.ui.lv_groups.selectionModel().selectedIndexes()[0]
        item = self.group_model.itemFromIndex(index)
        sel_dialog = ImageSelectionDialog(images=self.data["image_paths"],
                                          selected_images=item.data()["keys"])
        code = sel_dialog.exec()
        if code == QDialog.Accepted:
            return sel_dialog.get_selected_images()
        # None, not ([], []) -- see ExperimentDialog.open_image_selection_dialog
        return None

    def remove_selected_image(self) -> None:
        """
        Method to remove the selected image from the selected group

        :return: None
        """
        # Get selected image
        indices = self.ui.lv_images.selectionModel().selectedIndexes()
        img = self.img_model.itemFromIndex(indices[0])
        key = img.data()["key"]
        # (row, count) -- removeRows requires both, and passing only the row raised TypeError
        # before anything was removed, so this button did nothing at all
        self.img_model.removeRows(indices[0].row(), 1)
        # Get selected group
        indices = self.ui.lv_groups.selectionModel().selectedIndexes()
        group = self.group_model.itemFromIndex(indices[0])
        group_data = group.data()
        group_data["keys"].remove(key)
        group.setData(group_data)
        group.setText(
            f"{group_data['name']}:\nImages: {len(group_data['keys'])}"
        )
        self.refresh_image_buttons()

    def clear_images(self) -> None:
        """
        Method to remove all image from the selected group

        :return: None
        """
        # Get selected group
        indices = self.ui.lv_groups.selectionModel().selectedIndexes()
        group = self.group_model.itemFromIndex(indices[0])
        group_data = group.data()
        name = group_data["name"]
        # Get the number of images
        img_num = self.img_model.rowCount()
        # Check if the user really wants to remove all images
        clk = QMessageBox.question(self, "Remove images from group",
                                   f"Do you really want to remove {img_num} images the group {name}?"
                                   " This action cannot be reversed!",
                                   QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if clk == QMessageBox.Yes:
            self.img_model.clear()
            group_data["keys"] = []
            group.setData(group_data)
            group.setText(
                f"{group_data['name']}:\nImages: 0"
            )
            self.refresh_image_buttons()

    def on_img_selection_change(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        """
        Method to react to the selection of images

        :param selected: The selected items
        :param deselected: The deselected items
        :return: None
        """
        # btn_remove_image, not btn_images_remove -- the latter is ExperimentDialog's name and does
        # not exist in group_dialog.ui, so this slot raised AttributeError whenever it ran. It never
        # ran, because nothing connected it either.
        # btn_clear_images acts on the whole list, so it follows the list being non-empty rather
        # than the selection
        self.ui.btn_remove_image.setEnabled(bool(selected.indexes()))
        self.ui.btn_clear_images.setEnabled(self.img_model.rowCount() > 0)

    def refresh_image_buttons(self) -> None:
        """
        Method to bring the image buttons in line with the current image list

        selectionChanged does not fire when the list is repopulated, so switching group would
        otherwise leave Remove enabled for a selection that no longer exists, and Clear disabled
        for a list that now has images in it.

        :return: None
        """
        self.ui.btn_remove_image.setEnabled(
            bool(self.ui.lv_images.selectionModel().selectedIndexes()))
        self.ui.btn_clear_images.setEnabled(self.img_model.rowCount() > 0)

    def on_group_selection_change(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        """
        Function to react to changed experiment selection

        :param selected: The selected item
        :param deselected: The deselected item
        :return: None
        """
        self.ui.lv_images.setEnabled(False)
        # Delete images of the previously selected group
        self.img_model.clear()
        # Get selected experiment
        selected = selected.indexes()
        if selected:
            data = self.group_model.item(selected[0].row()).data()
            paths = [self.data["image_paths"][self.data["keys"].index(x)] for x in data["keys"]]
            if paths:
                for item in Util.create_image_item_list_from(paths, indicate_progress=False):
                    self.img_model.appendRow(
                        item
                    )
            self.ui.btn_add_images.setEnabled(True)
            self.ui.btn_remove.setEnabled(True)
            self.refresh_image_buttons()
        else:
            self.ui.btn_add_images.setEnabled(False)
            self.ui.btn_clear_images.setEnabled(False)
            self.ui.btn_remove_image.setEnabled(False)
            self.ui.btn_remove.setEnabled(False)
        self.ui.lv_images.setEnabled(True)

    def add_group(self) -> None:
        """
        Method to add a new group to the experiment

        :return: None
        """
        # Same fix as ExperimentDialog.add_experiment: the styled dialog is the one shown, rather
        # than being handed to the static call as a parent while that builds an unstyled one
        name, ok = ask_for_name(self, "Group Dialog", "Enter the new group: ")
        if ok:
            existing = {self.group_model.item(row).data()["name"]
                        for row in range(self.group_model.rowCount())}
            if not name or name in existing:
                QMessageBox.information(
                    self, "Group Dialog",
                    "Please enter a name." if not name
                    else f"A group named '{name}' already exists.")
                return
            # Create item to add to group list
            item = QStandardItem()
            item_data = {
                "name": name,
                "keys": [],
                "exp": self.data["name"]
            }
            item.setData(item_data)
            item.setText(f"{name}:\nImages: 0")
            self.group_model.appendRow(item)

    def remove_group(self) -> None:
        """
        Function to remove a group

        :return: None
        """
        # No sqlite3.connect here: this method opened a connection and a cursor, used neither --
        # every write below goes through self.inserter -- and never closed either, leaving a
        # handle on the database for the garbage collector to deal with
        # Get selected group
        index = self.ui.lv_groups.selectionModel().selectedIndexes()[0].row()
        item = self.group_model.item(index)
        data = item.data()
        clk = QMessageBox.question(self, "Erase group",
                                   f"Do you really want to delete the group {data['name']}?"
                                   " This action cannot be reversed!",
                                   QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if clk == QMessageBox.Yes:
            # THE MODEL, AND ONLY THE MODEL -- RW, 2026-09-16: *"consolidate add and remove"*.
            #
            # This used to delete the group's rows and commit here, while `add_group` and
            # `add_images_to_group` only touch the model and rely on the save path. So Cancel
            # undid every addition and kept every deletion, which is the opposite of what a
            # Cancel button promises.
            #
            # The immediate write was also REDUNDANT: `ExperimentDialog.save_changes` calls
            # `remove_group_associations_for_experiment` and rewrites every association from this
            # model, so a group removed from the model is removed from the database on OK
            # whether or not anything was deleted here. Dropping the write is therefore all that
            # is needed to put both halves on one contract -- the model is the pending state, and
            # OK applies it.
            self.group_model.removeRow(index)
