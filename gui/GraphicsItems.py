import sqlite3
from typing import List, Iterable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QKeyEvent, QMouseEvent
from PyQt5.QtWidgets import QDialog, QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem
from skimage.draw import ellipse

from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler
from gui import Paths
from gui.loader import ROIDrawerTimer


class EditorView(pg.GraphicsView):
    COLORS = [
        QColor(255, 50, 0),  # Red
        QColor(50, 255, 0),  # Green
        QColor(255, 255, 0),  # Yellow
        QColor(255, 0, 255),  # Magenta
        QColor(0, 255, 255),  # Cyan
    ]

    def __init__(self, image: np.ndarray, roi: ROIHandler, parent: QDialog, size_factor: float = 1):
        super(EditorView, self).__init__()
        self.parent = parent
        self.size_factor = size_factor
        self.mode = -1
        self.image = image
        self.active_channel: int = None
        self.roi: ROIHandler = roi
        self.main_channel = roi.idents.index(roi.main)
        self.plot_item = pg.PlotItem()
        self.view = self.plot_item.getViewBox()
        self.view.setAspectLocked(True)
        self.view.invertY(True)
        self.pos_track = True
        self.img_item = pg.ImageItem()
        self.plot_item.addItem(self.img_item)
        self.plot_vb = self.plot_item.vb
        # Set proxy to detect mouse movement
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=45, slot=self.mouse_moved)
        self.mpos: Tuple = None
        # Activate mouse tracking for widget
        self.setMouseTracking(True)
        self.setCentralWidget(self.plot_item)
        self.draw_additional = True
        # List of existing items
        self.loading_timer: ROIDrawerTimer = None
        self.items = []
        self.draw_roi()
        # List for newly created items
        self.temp_items = []
        # List for items that should be removed
        self.delete: List[int] = []
        self.item_changes = {}
        self.selected_item: ROIItem = None
        self.shift_down = False
        self.saved_values: Dict = None
        self.show_channel(3)

    def set_changes(self, rect: QRectF, angle: float, preview: bool = True) -> None:
        """
        Method to apply the changes made by editing

        :param rect: The new bounding box of the currently active item
        :param angle: The angle of the currently active item
        :param preview: When true, the item will save its original orientation and size

        :return: None
        """
        if self.selected_item:
            self.selected_item.update_data(rect, angle, preview)

    def draw_additional_items(self, state: bool = True) -> None:
        """
        Method to signal if additional items besides nuclei and foci should be drawn

        :param state: Boolean decider
        :return: None
        """
        self.draw_additional = state
        ROIDrawer.draw_additional_items(self.items, self.draw_additional)

    def show_channel(self, channel: int) -> None:
        """
        Method to show the specified channel of the image

        :param channel: The index of the channel
        :return: None
        """
        if self.selected_item:
            self.selected_item.enable_editing(False)
            self.selected_item = None
            self.parent.enable_editing_widgets(False)
        self.active_channel = channel
        if channel == self.image.shape[2]:
            self.img_item.setImage(self.image)
        else:
            self.img_item.setImage(self.image[..., channel])
        ROIDrawer.change_channel(self.items, channel, self.draw_additional)

    def change_mode(self, mode: int = 0) -> None:
        """
        Method to change the edit mode

        :param mode: 0 for add new, 1 for edit
        :return: None
        """
        self.mode = mode
        if self.selected_item:
            self.selected_item.enable_editing(False)
            self.selected_item = None

    def track_mouse_position(self, state: bool = True) -> None:
        """
        Enables mouse coordinate tracking

        :param state: Boolean decider
        :return: None
        """
        self.pos_track = state

    def draw_roi(self) -> None:
        """
        Method to draw the roi

        :return: None
        """
        self.loading_timer = ROIDrawerTimer(self.roi, self.plot_item,
                                            feedback=self.update_loading,
                                            processing=ROIDrawer.draw_roi)

    def update_loading(self, items: List[QGraphicsItem]) -> None:
        """
        Method to update the progress bar

        :return: None
        """
        self.parent.ui.prg_loading.setValue(self.loading_timer.percentage * 100)
        self.items.extend(items)

    def get_roi_index(self, roi) -> int:
        """
        Method to get the channel index for the given ROI

        :param roi: The ROI
        :return: The channel index as int
        """
        return self.roi.idents.index(roi.ident)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        super().keyPressEvent(event)
        if event.key() == Qt.Key_Shift:
            self.shift_down = True
        if event.key() == Qt.Key_Delete:
            if self.selected_item:
                # Remove item from scene
                self.selected_item.remove_from_view(self)
                # Add item to deletion list to remove it from the database
                self.delete.append(self.selected_item.roi_id)
        elif event.key() == Qt.Key_1:
            self.change_mode(-1)
        elif event.key() == Qt.Key_2:
            self.change_mode(0)
        elif event.key() == Qt.Key_3:
            self.change_mode(1)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        super().keyReleaseEvent(event)
        if event.key() == Qt.Key_Shift:
            self.shift_down = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if self.mode == 0 and self.active_channel != 3 and event.button() == Qt.LeftButton and self.shift_down:
            # Get click position
            pos = self.mpos
            if self.active_channel == self.main_channel:
                item = NucleusItem(round(pos.x() - 75 * self.size_factor), round(pos.y() - 38 * self.size_factor),
                                   round(150 * self.size_factor), round(76 * self.size_factor),
                                   round(pos.x()), round(pos.y()),
                                   0, (0, 0), self.main_channel, -1)
                item.set_pens(
                    pg.mkPen(color="FFD700", width=3, style=QtCore.Qt.DashLine),
                    pg.mkPen(color="FFD700", width=3, style=QtCore.Qt.DashLine),
                    ROIDrawer.MARKERS[-1]
                )
                item.add_to_view(self.plot_item)
                self.parent.set_mode(2)
                self.change_mode(1)
                self.selected_item = item
                item.enable_editing(True)
                self.parent.setup_editing(item)
            else:
                item = FocusItem(round(pos.x() - 3.5 * self.size_factor), round(pos.y() - 3.5 * self.size_factor),
                                 round(7 * self.size_factor), round(7 * self.size_factor), self.active_channel, -1)
                item.set_pen(
                    ROIDrawer.MARKERS[self.active_channel],
                    ROIDrawer.MARKERS[-1]
                )
            item.changed = True
            self.items.append(item)
            self.temp_items.append(item)
            # Add item to view
            item.add_to_view(self.plot_item)
        if self.mode == 1 and self.active_channel != 3 and event.button() == Qt.LeftButton:
            items = [x for x in self.scene().items(self.mapToScene(event.pos()))
                     if isinstance(x, NucleusItem) or isinstance(x, FocusItem)]
            items = [x for x in items if x.channel_index == self.active_channel]
            if items:
                if self.selected_item:
                    self.selected_item.enable_editing(False)
                self.selected_item = items[-1]
                self.selected_item.enable_editing(True)
                self.parent.setup_editing(self.selected_item)

    def mouse_moved(self, event: QMouseEvent) -> None:
        if self.pos_track:
            pos = event[0]
            if self.plot_item.sceneBoundingRect().contains(pos):
                coord = self.plot_vb.mapSceneToView(pos)
                self.mpos = coord
                if self.selected_item and self.selected_item.preview:
                    self.selected_item.reset_item()
                self.parent.set_status(f"X: {coord.x():.2f} Y: {coord.y():.2f}")

    def apply_all_changes(self) -> None:
        """
        Method to apply all made changes and save them to the database

        :return: None
        """
        # Create connection to database
        conn = sqlite3.connect(Paths.database)
        # Create cursor to do stuff
        curs = conn.cursor()
        # Remove deleted roi from item list
        self.items = [x for x in self.items if x.roi_id not in self.delete]
        # Create list of tuples for sqlite3
        sql_delete = [(x,) for x in self.delete]
        # Delete items marked for it
        self.delete_unassociated_roi(curs, sql_delete)
        # Create list of changed items to ignore during map creation
        ignore = [x.roi_id for x in self.items if x.changed]
        # Also ignore roi that were deleted
        ignore.extend(self.delete)
        # Delete all items that can be ignored from ROIHandler
        self.roi.delete_rois(ignore)
        # Create a hash association maps for each channel
        maps = self.roi.create_hash_association_maps((self.image.shape[0], self.image.shape[1]))
        # Create list for items which will be unassociated due to data changes
        unassociated = []
        # Get all associated foci and add them to list of unassociated foci
        for roi in sql_delete:
            roi_hash = curs.execute("SELECT hash FROM roi WHERE associated=?", roi).fetchall()
            if roi_hash:
                unassociated.extend([x[0] for x in roi_hash])
        # Change the rows of fetched foci
        curs.executemany("UPDATE OR IGNORE roi SET associated=NULL WHERE associated=?", sql_delete)
        for item in self.items:
            if item.changed:
                # Check if item was added
                if item.roi_id != -1:
                    # Delete item from database
                    self.delete_item_from_database(curs, item.roi_id)
                    if isinstance(item, NucleusItem):
                        # Get hash list of associated foci
                        hashes = [x[0] for x in curs.execute("SELECT hash FROM roi WHERE associated=?",
                                                             (item.roi_id,)).fetchall()]
                        unassociated.extend(hashes)
                        curs.execute("UPDATE roi SET associated=? WHERE associated=?",
                                     (None, item.roi_id))
                    else:
                        unassociated.append(item.roi_id)
                # Get coordinates corresponding to the item
                rr, cc = ellipse(item.center[1], item.center[0], item.height / 2, item.width / 2,
                                 self.image.shape, np.deg2rad(-item.angle))
                # Get encoded area for item
                rle = self.encode_new_roi(rr, cc, maps[item.channel_index])
                # Create new ROI instance
                roi = ROI(channel=self.roi.idents[item.channel_index],
                          main=isinstance(item, NucleusItem), auto=False)
                roi.set_area(rle)
                roihash = hash(roi)
                # Foci need to be associated
                if isinstance(item, FocusItem):
                    unassociated.append(roihash)
                self.replace_placeholder(maps[item.channel_index], roihash)
                self.write_item_to_database(curs, item, roi, rle, self.image, self.roi.ident)
                # Add ROI to ROIHandler
                self.roi.rois.append(roi)
        associations = self.create_associations(self.roi.idents.index(self.roi.main), maps, unassociated)
        # Clean unassociated list
        unassociated = [(x, ) for x in unassociated if x not in associations.keys()]
        self.delete_unassociated_roi(curs, unassociated)
        # Create new associations
        for focus, nucleus in associations.items():
            curs.execute("UPDATE roi SET associated=? WHERE hash=?",
                         (int(nucleus), int(focus)))
        # Delete erased and unassociated ROI from ROIHandler
        self.delete.extend([x[0] for x in unassociated])
        self.roi.rois = [x for x in self.roi if x.id not in self.delete]
        # Change image entry to indicate that the image was manually modified
        curs.execute("UPDATE images SET modified=? WHERE md5=?",
                     (1, self.roi.ident)
                     )
        conn.commit()

    @staticmethod
    def write_item_to_database(curs: sqlite3.Cursor, item, roi: ROI,
                               rle: List[Tuple[int, int, int]], image: np.ndarray,
                               image_id: str) -> None:
        """
        Method to write the specified item to the database

        :param curs: Cursor pointing to the database
        :param item: The item to write to the database
        :param roi: The ROI associated with the item
        :param rle: The run length encoded area of this item
        :param image: The image from which the roi is derived
        :param image_id: The id of the image
        :return: None
        """
        # Calculate statistics
        stats = roi.calculate_statistics(image[..., item.channel_index])
        ellp = roi.calculate_ellipse_parameters()
        # Prepare data for SQL statement
        rle = [(hash(roi), x[0], x[1], x[2]) for x in rle]
        # Write item to database
        curs.execute("INSERT OR IGNORE INTO roi VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (hash(roi), image_id, False, roi.ident,
                      f"({item.center[1]:.0f}, {item.center[0]:.0f})", item.edit_rect.width,
                      item.edit_rect.height, None))
        curs.executemany("INSERT OR IGNORE INTO points VALUES (?, ?, ?, ?)",
                         rle)
        curs.execute("INSERT OR IGNORE INTO statistics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (hash(roi), image_id, stats["area"], stats["intensity average"],
                      stats["intensity median"], stats["intensity maximum"], stats["intensity minimum"],
                      stats["intensity std"], ellp["eccentricity"], ellp["roundness"],
                      f"({item.center[1]:.0f}, {item.center[0]:.0f})", item.width / 2, item.height / 2,
                      item.angle, ellp["area"], str(ellp["orientation"]), ellp["shape_match"]))

    @staticmethod
    def delete_item_from_database(curs: sqlite3.Cursor, roihash: int) -> None:
        """
        Method to delete the item, specified by its hash, from the database

        :param curs: Cursor pointing to the database
        :param roihash: The hash of the item
        :return: None
        """
        curs.execute("DELETE FROM roi WHERE hash=?", (roihash,))
        curs.execute("DELETE FROM points WHERE hash=?", (roihash,))
        curs.execute("DELETE FROM statistics WHERE hash=?", (roihash,))

    @staticmethod
    def create_associations(main: int, maps: Iterable[np.ndarray],
                            unassociated: Iterable[int]) -> Dict:
        """
        Method to create associations dictionary to associate nuclei with foci

        :param main: Index of the main channel
        :param maps:
        :param unassociated:
        :return: Dictionary containing the associations
        """
        # Create new associations
        associations = {}
        for c in range(len(maps)):
            if c != main:
                for y in range(maps[0].shape[0]):
                    for x in range(maps[0].shape[1]):
                        if maps[c][y][x] and maps[main][y][x] and (maps[c][y][x] in unassociated):
                            associations[maps[c][y][x]] = maps[main][y][x]
        return associations

    @staticmethod
    def delete_unassociated_roi(curs: sqlite3.Cursor, unassociated: Iterable[Tuple[int]]) -> None:
        """
        Method to delete unassociated roi from the database

        :param curs: Cursor pointing to the database
        :param unassociated: List of hashes from unassociated roi, prepared for executemany
        :return: None
        """
        curs.executemany("DELETE FROM roi WHERE hash=?",
                         unassociated)
        curs.executemany("DELETE FROM points WHERE hash=?",
                         unassociated)
        curs.executemany("DELETE FROM statistics WHERE hash=?",
                         unassociated)

    @staticmethod
    def replace_placeholder(map_: np.ndarray, roihash: int, placeholder: int = -1) -> None:
        """
        Method to replace a placeholder in the given map

        :param map_: The map
        :param roihash: The hash to replace the placeholder with
        :param placeholder: The placeholder to replace
        :return: None
        """
        shape = map_.shape
        # Replace placeholder with real hash
        for y in range(shape[0]):
            for x in range(shape[1]):
                if map_[y][x] == placeholder:
                    map_[y][x] = roihash

    @staticmethod
    def encode_new_roi(rr: List[int], cc: List[int],
                       map_: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Method to run length encode newly created roi

        :param rr: The row indices
        :param cc: The corresponding column indices
        :param map_: The corresponding map for this roi
        :return: The run length encoded area of the given roi
        """
        # Get encoded area for item
        rle = []
        # Get unique rows
        rows = np.unique(rr)
        # Iterate over unique rows
        for row in rows:
            rl = 0
            col = -1
            for index in range(len(rr)):
                if rr[index] == row and map_[rr[index]][cc[index]] == 0:
                    map_[rr[index]][cc[index]] = -1
                    rl += 1
                    if col == -1:
                        col = int(cc[index])
            rle.append((int(row), col, rl))
        return rle


class ROIDrawer:

    __slots__ = ()

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

    @staticmethod
    def change_channel(items: Iterable[QGraphicsItem],
                       active_channel: int = 3,
                       draw_additional: bool = False) -> None:
        """
        Method to change the drawing of foci and nuclei according to the active channel

        :param items: The items that are drawn on the view
        :param active_channel: The active channel
        :param draw_additional: Parameter to draw items for additional information
        :return: None
        """
        for item in items:
            if item.channel_index != active_channel and active_channel != 3:
                if isinstance(item, NucleusItem) and draw_additional:
                    item.is_active(True)
                else:
                    item.is_active(False)
            else:
                item.is_active(True)
            item.update_indicators(draw_additional)

    @staticmethod
    def draw_roi(view: pg.PlotItem, rois: Iterable[ROI], idents: Iterable[str]) -> List[QGraphicsEllipseItem]:
        """
        Method to populate the given plot with the roi stored in the handler

        :param view: The PlotItem to populate
        :param rois: The ROIHandler
        :param idents: List of available channels
        :return: List of all created items
        """
        items = []
        for roi in rois:
            ind = idents.index(roi.ident)
            if roi.main:
                items.append(ROIDrawer.draw_nucleus(view, roi, ind))
            else:
                items.append(ROIDrawer.draw_focus(view, roi, ind))
        return items

    @staticmethod
    def draw_focus(view: pg.PlotItem, roi: ROI, ind) -> QGraphicsEllipseItem:
        """
        Function to draw a focus onto the given view

        :param view: The view to draw on
        :param roi: The focus to draw
        :param ind: The index of the roi channel
        :return: None
        """
        dims = roi.calculate_dimensions()
        pen = ROIDrawer.MARKERS[ind]
        c = dims["minX"], dims["minY"]
        d2 = dims["height"]
        d1 = dims["width"]
        nucleus = FocusItem(c[0], c[1], d1, d2, ind, hash(roi))
        nucleus.set_pen(pen, ROIDrawer.MARKERS[-1])
        nucleus.add_to_view(view)
        return nucleus

    @staticmethod
    def draw_nucleus(view: pg.PlotItem, roi: ROI, ind) -> QGraphicsEllipseItem:
        """
        Function to draw a nucleus onto the given view

        :param view: The view to draw on
        :param roi: The nucleus to draw
        :param ind: The index of the roi channel
        :return: None
        """
        pen = pg.mkPen(color="FFD700", width=3, style=QtCore.Qt.DashLine)
        params = roi.calculate_ellipse_parameters()
        cy, cx = params["center"][0], params["center"][1]
        r1 = params["minor_axis"]
        r2 = params["major_axis"]
        angle = params["angle"]
        ovx, ovy = params["orientation"]
        focus = NucleusItem(cx - r2, cy - r1, r2 * 2, r1 * 2, cx, cy, angle, (ovx, ovy), ind, hash(roi))
        focus.set_pens(
            pen,
            pen,
            ROIDrawer.MARKERS[-1]
        )
        focus.is_active()
        focus.update_indicators()
        focus.add_to_view(view)
        return focus

    @staticmethod
    def draw_additional_items(items: List[QGraphicsItem], draw_additional: bool = True) -> None:
        """
        Method to activate the drawing of additional items

        :param items: The list of items to activate
        :param draw_additional: Bool
        :return: None
        """
        for item in items:
            if isinstance(item, NucleusItem):
                item.is_active(draw_additional)
            item.update_indicators(draw_additional)


class EditingRectangle(QGraphicsRectItem):

    __slots__ = [
        "width",
        "height",
        "center",
        "x",
        "y",
        "pen",
        "ipen",
        "color",
    ]

    def __init__(self, x, y, cx, cy, width, height):
        super().__init__(x, y, width, height)
        self.active = False
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.center = cx, cy
        self.ipen = None
        self.pen = None
        self.color = None
        self.initialize()

    def initialize(self):
        """
        Method to initialize this class

        :return:  None
        """
        self.pen = pg.mkPen(color="#bdff00", width=3, style=QtCore.Qt.DashLine)
        self.ipen = ROIDrawer.MARKERS[-1]
        self.setPen(self.pen)

    def activate(self, enable: bool = True) -> None:
        """
        Method to activate this item

        :param enable: Bool
        :return: None
        """
        if enable:
            self.setPen(self.pen)
        else:
            self.setPen(self.ipen)


class ROIItem(QGraphicsEllipseItem):
    __slots__ = [
        "preview"
        "changed",
        "rect",
        "x",
        "y",
        "width",
        "height",
        "angle",
        "channel_index",
        "roi_id",
        "orientation",
        "pen",
        "center",
        "indicators",
        "pen",
        "iapen",
        "ipen"
    ]

    def __init__(self, x: int, y: int, width: int, height: float, index: int, roi_ident: int):
        super().__init__(x, y, width, height)
        self.preview = False
        self.changed = False
        self.rect = QRectF(x, y, width, height)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.center = int(self.x + self.width / 2), int(self.y + self.height / 2)
        self.angle = 0
        self.channel_index = index
        self.roi_id = roi_ident
        self.pen: pg.mkPen = None
        self.ipen: pg.mkPen = None
        self.main_color = None
        self.hover_color = None
        self.sel_color = None
        self.view: EditorView = None
        self.edit_rect = None
        self.setEnabled(False)

    def update_data(self, rect: QRectF, angle: float, keep_original: bool = True) -> None:
        """
        Method to update position and angle of this item

        :param rect: The new bounding rect of this item
        :param angle: The new angle of this item
        :param keep_original: If true, the position and angle before the change will be stored.
        Used for preview purposes
        :return: None
        """
        if not keep_original:
            self.rect = rect
            self.x = rect.x()
            self.y = rect.y()
            self.width = rect.width()
            self.height = rect.height()
            self.center = rect.center().x(), rect.center().y()
            self.angle = angle
            self.preview = False
            self.changed = True
        else:
            self.preview = True
        self.setRotation(0)
        self.setRect(rect)
        self.setTransformOriginPoint(rect.center())
        self.setRotation(angle)
        self.edit_rect.setRotation(0)
        self.edit_rect.setRect(rect)
        self.edit_rect.setTransformOriginPoint(rect.center())
        self.edit_rect.setRotation(angle)

    def reset_item(self) -> None:
        """
        Method to reset the item if the preview was not applied

        :return: None
        """
        self.update_data(self.rect, self.angle)
        self.preview = False

    def remove_from_view(self, view: EditorView) -> None:
        """
        Method to remove this item from the given view

        :param view: The view to remove the item from
        :return: None
        """
        view.scene().removeItem(self.edit_rect)
        view.scene().removeItem(self)

    def is_active(self, active: bool = True) -> None:
        """
        Method to set the activity of this item

        :param active: Bool
        :return: None
        """
        self.setPen(self.pen if active else self.ipen)

    def update_indicators(self, draw: bool = True) -> None:
        """
        Dummy Method to be compatible with NucleusItem
        """
        pass

    def set_pen(self, pen: pg.mkPen, inactive_pen: pg.mkPen):
        self.pen = pen
        self.ipen = inactive_pen
        # Define needed colors
        self.main_color = pen.color()
        self.hover_color = self.main_color.lighter(100)
        self.sel_color = self.main_color.lighter(150)
        self.setPen(pen)

    def add_to_view(self, view: EditorView) -> None:
        """
        Method to add this item and all associated items to the given view

        :param view: View to add to
        :return: None
        """
        self.view = view
        view.addItem(self)
        rect = EditingRectangle(self.x, self.y, self.center[0], self.center[1], self.width, self.height)
        rect.activate(False)
        self.edit_rect = rect

    def enable_editing(self, enable: bool = True) -> None:
        """
        Method to enable the editing of this item

        :param enable: Bool
        :return: None
        """
        if enable:
            self.setEnabled(enable)
            self.view.addItem(self.edit_rect)
            self.edit_rect.activate(enable)
        else:
            self.setEnabled(enable)
            self.view.removeItem(self.edit_rect)
            self.edit_rect.activate(enable)


class NucleusItem(ROIItem):

    def __init__(self, x: int, y: int, width: int, height: int, center_x: int, center_y: int,
                 angle: float, orientation: Tuple[float, float], index: int, roi_ident: int):
        super().__init__(x, y, width, height, index, roi_ident)
        self.changed = False
        self.rect = None
        self.angle = angle
        self.center = center_x, center_y
        self.orientation = orientation
        self.indicators = []
        self.edit = False
        self.edit_rect: EditingRectangle = None
        self.iapen: pg.mkPen = None
        self.initialize()

    def update_data(self, rect: QRectF, angle: float, keep_original: bool = True) -> None:
        """
        Method to update position and angle of this item

        :param rect: The new bounding rect of this item
        :param angle: The new angle of this item
        :param keep_original: If true, the position and angle before the change will be stored.
        Used for preview purposes
        :return: None
        """
        super().update_data(rect, angle, keep_original)
        # Update indicators to represent new params
        r1 = rect.height() / 2
        r2 = rect.width() / 2
        self.indicators[0].setLine(-r2, 0, r2, 0)
        self.indicators[1].setLine(-r1, 0, r1, 0)
        for indicator in self.indicators:
            indicator.setPos(self.boundingRect().center())

    def is_active(self, active: bool = True) -> None:
        """
        Method to set the activity of this item

        :param active: Bool
        :return: None
        """
        if not active:
            self.edit_rect.activate(active)
        self.setPen(self.iapen if not active else self.pen)

    def update_indicators(self, draw: bool = True) -> None:
        """
        Method update the drawing of indicators

        :param draw: Bool to indicate if the indicators should be drawn
        :return: None
        """
        for indicator in self.indicators:
            indicator.setPen(self.ipen if draw else self.iapen)

    def set_pens(self, pen: pg.mkPen, indicator_pen: pg.mkPen,
                 inactive_pen: pg.mkPen) -> None:
        """
        Method to set the pen to draw this item

        :param pen: The pen to draw this item when active
        :param indicator_pen: The pen to draw the indicators of this item with
        :param inactive_pen: The pen to use if this item is set to inactive
        :return: None
        """
        self.pen = pen
        self.ipen = indicator_pen
        self.iapen = inactive_pen
        self.setPen(self.pen)
        for indicator in self.indicators:
            indicator.setPen(self.ipen)

    def initialize(self) -> None:
        """
        Method to initialize the display of this item

        :return: None
        """
        op = self.sceneBoundingRect().center()
        self.setTransformOriginPoint(op)
        self.setRotation(self.angle)
        cx, cy = self.center
        r1, r2 = self.height / 2, self.width / 2
        # Draw major axis
        major_axis = QGraphicsLineItem(-r2, 0, r2, 0)
        major_axis.setPos(cx, cy)
        major_axis.setParentItem(self)
        # Draw minor axis
        minor_axis = QGraphicsLineItem(-r1, 0, r1, 0)
        minor_axis.setPos(cx, cy)
        minor_axis.setParentItem(self)
        minor_axis.setRotation(90)
        rect = EditingRectangle(self.x, self.y, self.center[0], self.center[1], self.width, self.height)
        rect.setTransformOriginPoint(rect.sceneBoundingRect().center())
        rect.setRotation(self.angle)
        self.indicators.extend([
            major_axis,
            minor_axis,
        ])
        self.edit_rect = rect
        self.rect = self.boundingRect()
        self.edit_rect.activate(False)
        self.setEnabled(False)

    def add_to_view(self, view: EditorView) -> None:
        """
        Method to add this item and all associated items to the given view

        :param view: The view to add to
        :return: None
        """
        self.view = view
        view.addItem(self)

    def __str__(self):
        return f"NucleusItem X:{self.x} Y:{self.y} W:{self.width} H:{self.height} C:{self.center}"


class FocusItem(ROIItem):

    def __str__(self):
        return f"FocusItem X:{self.x} Y:{self.y} W:{self.width} H:{self.height} C:{self.center}"
