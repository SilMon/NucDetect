import sqlite3
import time
from typing import List, Iterable, Dict, Tuple, Callable

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QKeyEvent, QMouseEvent
from PyQt5.QtWidgets import QDialog, QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem, \
    QGraphicsView
from skimage.draw import ellipse

from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler
from gui import Paths
from gui.Util import create_partial_list
from gui.loader import Loader


class EditorView(pg.GraphicsView):
    COLORS = [
        QColor(255, 50, 0),  # Red
        QColor(50, 255, 0),  # Green
        QColor(255, 255, 0),  # Yellow
        QColor(255, 0, 255),  # Magenta
        QColor(0, 255, 255),  # Cyan
    ]

    def __init__(self, image: np.ndarray, roi: ROIHandler, parent: QDialog):
        super(EditorView, self).__init__()
        self.parent = parent
        self.mode = -1
        self.image = image
        self.active_channel = None
        self.roi = roi
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
        self.mpos = None
        # Activate mouse tracking for widget
        self.setMouseTracking(True)
        self.setCentralWidget(self.plot_item)
        self.draw_additional = True
        # List of existing items
        self.loading_timer = None
        self.items = []
        self.draw_roi()
        # List for newly created items
        self.temp_items = []
        # List for items that should be removed
        self.delete = []
        self.item_changes = {}
        self.selected_item = None
        self.shift_down = False
        self.saved_values = None
        self.show_channel(3)

    def set_changes(self, rect: QRectF, angle: float, preview: bool = True) -> None:
        """
        Method to apply the changes made by editing

        :param rect: The new bounding box of the currently active item
        :param angle: The angle of the currently active item
        :param preview: When true, the item will save its original orientation and size

        :return: None
        """
        self.selected_item.update_data(rect, angle, preview)
        # Mark Original ident as delete
        if not preview:
            self.delete.append(self.selected_item.roi_ident)

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
        self.loading_timer = ROIDrawerTimer(self.roi, self.plot_item, feedback=self.update_loading)

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
            item = self.selected_item
            # Remove item from scene
            item.remove_from_view(self)
            # Add item to deletion list to remove it from the database
            self.delete.append(item.roi_ident)
        elif event.key() == Qt.Key_1:
            self.change_mode(-1)
        elif event.key() == Qt.Key_2:
            self.change_mode(0)
        elif event.key() == Qt.Key_3:
            self.change_mode(1)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Shift:
            self.shift_down = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if self.mode == 0 and self.active_channel != 3 and event.button() == Qt.LeftButton and self.shift_down:
            # Get click position
            pos = self.mpos
            if self.active_channel == self.main_channel:
                item = NucleusItem(round(pos.x()) - 75, round(pos.y()) - 38, 150, 76, round(pos.x()), round(pos.y()),
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
                item = FocusItem(round(pos.x() - 3.5), round(pos.y() - 3.5), 7, 7, self.active_channel, -1)
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
                self.parent.set_status(f"X: {coord.x():.2f} Y: {coord.y():.2f}")

    def apply_all_changes(self) -> None:
        """
        Method to apply all made changes and save them to the database

        :return: None
        """
        # Create list of changed items to ignore during map creation
        ignore = [x.roi_ident for x in self.items if x.changed]
        ignore.extend(self.delete)
        # Create a hash asso map
        maps = self.roi.create_hash_association_maps((self.image.shape[0], self.image.shape[1]), ignore)
        # Create connection to database
        conn = sqlite3.connect(Paths.database)
        # Create cursor to do stuff
        curs = conn.cursor()
        # Create list of tuples for sqlite3
        self.delete = [(x, ) for x in self.delete]
        # Create list for items which will be unassociated due to data changes
        unassociated = []
        # Delete items marked for it
        self.delete_unassociated_roi(curs, self.delete)
        # Get all associated roi
        for roi in self.delete:
            roi_hash = curs.execute("SELECT hash FROM roi WHERE associated=?", roi).fetchall()
            if roi_hash:
                unassociated.extend([x[0] for x in roi_hash])
        curs.executemany("UPDATE OR IGNORE roi SET associated=NULL WHERE associated=?", self.delete)
        for item in self.items:
            if item.changed:
                # Check if item was added
                if item.roi_ident != -1:
                    # Delete item from database
                    self.delete_item_from_database(curs, item.roi_ident)
                    if isinstance(item, NucleusItem):
                        # Get hash list of associated foci
                        hashes = [x[0] for x in curs.execute("SELECT hash FROM roi WHERE associated=?",
                                                             (item.roi_ident,)).fetchall()]
                        unassociated.extend(hashes)
                        curs.execute("UPDATE roi SET associated=? WHERE associated=?",
                                     (None, item.roi_ident))
                    else:
                        unassociated.append(item.roi_ident)
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
        self.delete.extend(unassociated)
        self.roi.rois = [x for x in self.roi.rois if hash(x) not in self.delete]
        conn.commit()

    @staticmethod
    def write_item_to_database(curs: sqlite3.Cursor, item, roi: ROI,
                               rle: List[List[int]], image: np.ndarray,
                               image_id: int) -> None:
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
        curs.execute("INSERT INTO roi VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (hash(roi), image_id, False, roi.ident,
                      f"({item.center[1]:.0f}, {item.center[0]:.0f})", item.edit_rect.width,
                      item.edit_rect.height, None))
        curs.executemany("INSERT INTO points VALUES (?, ?, ?, ?)",
                         rle)
        curs.execute("INSERT INTO statistics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
    def create_associations(main: int, maps: List[np.ndarray],
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
                        if maps[c][y][x] and maps[main][y][x] and maps[c][y][x] in unassociated:
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


class NucleusItem(QGraphicsEllipseItem):

    __slots__ = [
        "changed",
        "x",
        "y",
        "width",
        "height",
        "angle",
        "channel_index",
        "roi_ident",
        "orientation",
        "pen",
        "center",
        "indicators"
        "pen",
        "iapen",
        "ipen"
    ]

    def __init__(self, x: int, y: int, width: int, height: int, center_x: int, center_y: int,
                 angle: float, orientation: Tuple[float, float], index: int, roi_ident: int):
        super().__init__(x, y, width, height)
        self.changed = False
        self.x = x
        self.y = y
        self.angle = angle
        self.width = width
        self.height = height
        self.center = center_x, center_y
        self.orientation = orientation
        self.channel_index = index
        self.roi_ident = roi_ident
        self.indicators = []
        self.edit = False
        self.edit_rect: EditingRectangle = None
        self.pen: pg.mkPen = None
        self.ipen: pg.mkPen = None
        self.iapen: pg.mkPen = None
        self.view: EditorView = None
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
        if not keep_original:
            self.x = rect.x()
            self.y = rect.y()
            self.width = rect.width()
            self.height = rect.height()
            self.center = rect.center().x(), rect.center().y()
            self.angle = angle
            self.changed = True
        self.setRotation(0)
        self.setRect(rect)
        # Update indicators to represent new params
        r1 = rect.height() / 2
        r2 = rect.width() / 2
        self.indicators[0].setLine(-r2, 0, r2, 0)
        self.indicators[1].setLine(-r1, 0, r1, 0)
        self.setTransformOriginPoint(rect.center())
        self.setRotation(angle)
        self.edit_rect.setRotation(0)
        self.edit_rect.setRect(rect)
        self.edit_rect.setTransformOriginPoint(rect.center())
        self.edit_rect.setRotation(angle)
        for indicator in self.indicators:
            indicator.setPos(self.boundingRect().center())

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

    def __str__(self):
        return f"NucleusItem X:{self.x} Y:{self.y} W:{self.width} H:{self.height} C:{self.center}"


class FocusItem(QGraphicsEllipseItem):

    __slots__ = [
        "changed",
        "x",
        "y",
        "width",
        "height",
        "center",
        "angle"
        "edit_rect",
        "channel_index",
        "roi_ident",
        "pen",
        "ipen"
        "hover_color",
        "main_color",
        "sel_color"
    ]

    def __init__(self, x: int, y: int, width: int, height: float, index: int, roi_ident: int):
        super().__init__(x, y, width, height)
        self.changed = False
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.center = int(self.x + self.width / 2), int(self.y + self.height / 2)
        self.angle = 0
        self.channel_index = index
        self.roi_ident = roi_ident
        self.pen: pg.mkPen = None
        self.ipen: pg.mkPen = None
        self.main_color = None
        self.hover_color = None
        self.sel_color = None
        self.view = None
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
            self.x = rect.x()
            self.y = rect.y()
            self.width = rect.width()
            self.height = rect.height()
            self.center = rect.center().x(), rect.center().y()
            self.angle = angle
            self.changed = True
            print(f"Changed: {self}")
        self.setRotation(0)
        self.setRect(rect)
        self.setTransformOriginPoint(rect.center())
        self.setRotation(angle)
        self.edit_rect.setRotation(0)
        self.edit_rect.setRect(rect)
        self.edit_rect.setTransformOriginPoint(rect.center())
        self.edit_rect.setRotation(angle)

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

    def __str__(self):
        return f"FocusItem X:{self.x} Y:{self.y} W:{self.width} H:{self.height} C:{self.center}"


class ROIDrawerTimer(Loader):

    def __init__(self, items: ROIHandler, view: QGraphicsView,
                 batch_size: int = 25, batch_time: int = 250, feedback: Callable = None):
        """
        Class to implement lazy roi drawing.

        :param items: The items to draw
        :param view: Graphicsview to draw the ROI on
        :param batch_size: The number of images to load per batch
        :param batch_time: The time between consecutive loading approaches in milliseconds
        :param feedback: The function to call after loading. Has to accept a list of QStandardItems
        """
        super().__init__(items, batch_size, batch_time, feedback)
        self.view = view

    def load_next_batch(self) -> None:
        """
        Function to load the next batch. After loading, the feedback function will be called (will pass an empty list
        to the feedback function to indicate finished loading)

        :return: None
        """
        # Load the next batch of ROIs
        item_list = create_partial_list(self.items, self.last_index, self.batch_size)
        # Draw items
        items = ROIDrawer.draw_roi(self.view, item_list, self.items.idents)
        self.items_loaded += len(items)
        # Check if all items were loaded
        if not items:
            print(f"Timer stop after loading {self.items_loaded} items")
            print(f"Total loading time: {time.time() - self.start_time:.2f}")
            self.stop()
        # Update the last index
        self.last_index += self.batch_size
        # Update the loading percentage
        self.percentage = self.items_loaded / (len(self.items))
        # Check if a feedback function was given
        if self.feedback:
            # Call the feedback function
            self.feedback(items)