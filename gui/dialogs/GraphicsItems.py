from typing import List, Iterable, Dict, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import QRectF, Qt, QPointF
from PyQt5.QtGui import QColor, QKeyEvent, QMouseEvent
from PyQt5.QtWidgets import QDialog, QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem
from matplotlib import pyplot as plt
from skimage.draw import ellipse

from DataProcessing import create_lg_lut
from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler
from database.connections import Requester, Inserter
from gui.loader import ROIDrawerTimer


class EditorView(pg.GraphicsView):
    COLORS = [
        QColor(255, 50, 0),  # Red
        QColor(50, 255, 0),  # Green
        QColor(255, 255, 0),  # Yellow
        QColor(255, 0, 255),  # Magenta
        QColor(0, 255, 255),  # Cyan
    ]

    def __init__(self, image: np.ndarray, roi: ROIHandler,
                 parent: QDialog, active_channels: List[Tuple[int, str]],
                 size_factor: float = 1, high_contrast: bool = False):
        """
        :param image: The background image to display
        :param roi: All roi associated with this image
        :param parent: The EditorDialog incorporating this view
        :param active_channels: List containing the index of the channel and its corresponding name
        :param size_factor: Size factor used for newly added ROI
        :param high_contrast: If true, the channels will be shown in high contrast mode
        """
        super(EditorView, self).__init__()
        self.parent = parent
        self.active_channels = {x[1]: x[0] for x in active_channels}
        self.size_factor = size_factor
        self.high_contrast = high_contrast
        self.mode = -1
        self.image = image
        self.hcimg = self.create_high_contrast_image()
        self.active_channel: int = None
        self.roi: ROIHandler = roi
        self.requester = Requester()
        self.inserter = Inserter()
        self.main_channel = self.requester.get_main_channel(self.roi.ident)
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
        self.mpos: QPointF = None
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
        self.show_channel("Composite")
        self.current_channel = "Composite"

    def set_changes(self, rect: QRectF, angle: float, preview: bool = False) -> None:
        """
        Method to apply the changes made by editing

        :param rect: The new bounding box of the currently active item
        :param angle: The angle of the currently active item
        :param preview: When true, the item will save its original orientation and size

        :return: None
        """
        if self.selected_item:
            if not preview:
                self.temp_items.append(self.selected_item)
            self.selected_item.update_data(rect, angle, preview)

    def draw_additional_items(self, state: bool = True) -> None:
        """
        Method to signal if additional items besides nuclei and foci should be drawn

        :param state: Boolean decider
        :return: None
        """
        self.draw_additional = state
        ROIDrawer.draw_additional_items(self.items, self.draw_additional)

    def show_channel(self, channel: str) -> None:
        """
        Method to show the specified channel of the image

        :param channel: The name of the channel
        :return: None
        """
        self.current_channel = channel
        if self.selected_item:
            self.selected_item.enable_editing(False)
            self.selected_item = None
            self.parent.enable_editing_widgets(False)
        self.active_channel = channel
        if channel == "Composite":
            self.img_item.setImage(self.image)
            index = self.image.shape[2]
        else:
            # Check to which index the name corresponds
            index = self.active_channels[channel]
            if self.high_contrast:
                self.img_item.setImage(self.hcimg[..., index])
            else:
                self.img_item.setImage(self.image[..., index])
        ROIDrawer.change_channel(self.items, index, self.draw_additional)

    def create_high_contrast_image(self) -> np.ndarray:
        """
        Method to create the needed high contrast image

        :return: None
        """
        img = np.zeros(shape=self.image.shape)
        for c in range(img.shape[2]):
            channel = self.image[..., c]
            # Create a lut
            lut = create_lg_lut(np.amax(channel))
            # Iterate over the image
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    img[y][x][c] = lut[channel[y][x]]
        return img

    def toggle_high_contrast_mode(self, toggle: bool):
        """
        Method to toggle high contrast mode

        :param toggle: Toggle
        :return: None
        """
        self.high_contrast = toggle
        self.show_channel(self.current_channel)

    def change_colormap(self, colormap: str) -> None:
        """
        Method to load the given colormap

        :param colormap: Name of the colormap to load
        :return: None
        """
        self.img_item.setColorMap(pg.colormap.get(colormap, source="matplotlib"))

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

    def mark_as_changed(self, idents: List[int]) -> None:
        """
        Method to mark items with the given id as changed

        :param idents: List of ids
        :return: None
        """
        for ident in idents:
            for item in self.items:
                if item.roi_id not in self.delete:
                    if item.roi_id == ident:
                        item.changed = True

    def clear_and_update(self) -> None:
        """
        Method to display new roi

        :param rois: The new roi to show
        :return: None
        """
        # Get list of changed items
        changed = [x.roi_id for x in self.items]
        # Clear item lists
        self.items.clear()
        self.draw_roi()
        self.mark_as_changed(changed)
        self.show_channel("Composite")

    def draw_roi(self) -> None:
        """
        Method to draw the roi

        :return: None
        """
        self.roi.sort_roi_list()
        self.loading_timer = ROIDrawerTimer(self.roi, self.plot_item,
                                            feedback=self.update_loading,
                                            processing=ROIDrawer.draw_roi)

    def update_loading(self, items: List[QGraphicsItem]) -> None:
        """
        Method to update the progress bar

        :return: None
        """
        self.parent.ui.prg_loading.setValue(int(self.loading_timer.percentage * 100))
        self.items.extend(items)
        if round(self.loading_timer.percentage * 100) >= 99:
            for item in self.items:
                item.setVisible(True)

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
                if self.selected_item.roi_id != -1:
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
        if self.mode == 0 and self.active_channel != "Composite" and event.button() == Qt.LeftButton:
            self.create_new_item_at_mouse_position()
        # Set the selected item to the mouse position via middle click
        if self.mode == 1 and self.active_channel != "Composite" and event.button() == Qt.MiddleButton:
            # Check if an item is currently selected
            if self.selected_item:
                # Set the center position of the item to the mouse position
                self.move_selected_item_to_position()
        if self.mode == 1 and self.active_channel != "Composite" and event.button() == Qt.LeftButton:
            self.select_item_at_mouse_position(event)

    def move_selected_item_to_position(self) -> None:
        """
        Method to move the selected item to the specified location

        :return: None
        """
        x = self.mpos.x()
        y = self.mpos.y()
        width = self.selected_item.width
        height = self.selected_item.height
        angle = self.selected_item.angle
        rect = QRectF(x - width/2,
                      y - height/2,
                      width, height)
        self.selected_item.update_data(
            rect, angle, False
        )
        self.parent.setup_editing(self.selected_item)

    def select_item_at_mouse_position(self, event: QMouseEvent) -> None:
        """
        Method to select the clicked item at the mouse position

        :return: None
        """
        items = [x for x in self.scene().items(self.mapToScene(event.pos()))
                 if isinstance(x, NucleusItem) or isinstance(x, FocusItem)]
        items = [x for x in items if x.channel_index == self.active_channels[self.active_channel]]
        if items:
            if self.selected_item:
                self.selected_item.enable_editing(False)
            self.selected_item = items[-1]
            self.selected_item.enable_editing(True)
            self.parent.setup_editing(self.selected_item)

    def create_new_item_at_mouse_position(self) -> None:
        """
        Method to create a new item at the current mouse position

        :return: None
        """
        # Get click position
        pos = self.mpos
        if self.active_channel == self.main_channel:
            item = NucleusItem(round(pos.x() - 75 * self.size_factor), round(pos.y() - 38 * self.size_factor),
                               round(150 * self.size_factor), round(76 * self.size_factor),
                               round(pos.x()), round(pos.y()),
                               0, (0, 0), self.active_channels[self.main_channel], -1)
            item.set_pens(
                pg.mkPen(color="#2A2ABB", width=3, style=QtCore.Qt.DashLine),
                pg.mkPen(color="#2A2ABB", width=3, style=QtCore.Qt.DashLine),
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
                             round(7 * self.size_factor), round(7 * self.size_factor),
                             self.active_channels[self.active_channel], -1)
            item.set_pen(
                ROIDrawer.MARKERS[self.active_channels[self.active_channel]],
                ROIDrawer.MARKERS[-1]
            )
        item.changed = True
        self.items.append(item)
        self.temp_items.append(item)
        # Add item to view
        item.add_to_view(self.plot_item)

    def mouse_moved(self, event: QMouseEvent) -> None:
        if self.pos_track:
            pos = event[0]
            if self.plot_item.sceneBoundingRect().contains(pos):
                coord = self.plot_vb.mapSceneToView(pos)
                self.mpos = coord
                self.parent.set_status(f"X: {coord.x():.2f} Y: {coord.y():.2f}")

    def set_item_opacity(self, opacity: float) -> None:
        """
        Method to set the opacity of all ROIItems

        :param opacity: The opacity value [0-100]
        :return: None
        """
        ROIDrawer.change_opacity(self.items, opacity)

    def delete_items_in_list(self) -> None:
        """
        Method to delete all roi in the self.delete list

        :return: None
        """
        # Remove deleted roi from item list
        self.items = [x for x in self.items if x.roi_id not in self.delete]
        # Delete items marked for it
        self.delete_roi(self.delete)

    def create_association_maps(self) -> List[np.ndarray]:
        """
        Method to get the hash association maps

        :return: List of all created association maps
        """
        # Create list of changed items to ignore during map creation
        ignore = [x.roi_id for x in self.items if x.changed]
        # Also ignore roi that were deleted
        ignore.extend(self.delete)
        # Delete all items that can be ignored from ROIHandler
        self.roi.delete_rois(ignore)
        # Create a hash association maps for each channel
        return self.roi.create_hash_association_maps((self.image.shape[0], self.image.shape[1]))

    def get_unassociated_foci(self) -> List[int]:
        """
        Method to get the now unassociated foci for each nucleus in the self.delete list

        :return: List of focus hashes
        """
        unassociated = []
        # Get all associated foci and add them to list of unassociated foci
        for roi in self.delete:
            roi_hash = self.requester.get_hashes_of_associated_foci(roi)
            if roi_hash:
                unassociated.extend(roi_hash)
        return unassociated

    def process_changed_items(self, unassociated: List[int], maps: List[np.ndarray]) -> None:
        """
        Method to process all items that are marked as changed

        :param unassociated: List of all unassociated focus hashes
        :param maps: Association hash maps for all channels
        :return: None
        """
        for item in self.items:
            if item.changed:
                if item not in self.temp_items:
                    continue
                # Check if item was added
                if item.roi_id != -1:
                    # Delete item from database
                    self.delete_item_from_database(item.roi_id)
                    if isinstance(item, NucleusItem):
                        # Get hash list of associated foci
                        hashes = self.requester.get_hashes_of_associated_foci(item.roi_id)
                        print(item.roi_id)
                        unassociated.extend(hashes)
                        self.inserter.reset_nucleus_focus_association(item.roi_id)
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
                self.write_item_to_database(item, roi, rle, self.image, self.roi.ident)
                # Add ROI to ROIHandler
                self.roi.rois.append(roi)

    def apply_all_changes(self) -> None:
        """
        Method to apply all made changes and save them to the database

        :return: None
        """
        self.delete_items_in_list()
        maps = self.create_association_maps()
        # Create list for items which will be unassociated due to data changes
        unassociated = self.get_unassociated_foci()
        # Change the rows of fetched foci
        self.inserter.reset_nuclei_foci_associations(self.delete)
        # Check for changed items
        self.process_changed_items(unassociated, maps)
        associations = self.create_associations(self.roi.idents.index(self.roi.main), maps, unassociated)
        # Clean unassociated list
        unassociated = [x for x in unassociated if x not in associations.keys()]
        self.delete_roi(unassociated)
        # Create new associations
        for focus, nucleus in associations.items():
            self.inserter.associate_focus_with_nucleus(int(nucleus), int(focus))
        # Change image entry to indicate that the image was manually modified
        self.inserter.mark_image_as_modified(self.roi.ident)
        self.inserter.commit_and_close()

    def write_item_to_database(self, item, roi: ROI,
                               rle: List[Tuple[int, int, int]],
                               image: np.ndarray,
                               image_id: str) -> None:
        """
        Method to write the specified item to the database

        :param item: The item to write to the database
        :param roi: The ROI associated with the item
        :param rle: The run length encoded area of this item
        :param image: The image from which the roi is derived
        :param image_id: The id of the image
        :return: None
        """
        # Calculate statistics
        roidat = (hash(roi), image_id, False, roi.ident,
                  item.center[1], item.center[0], item.edit_rect.width,
                  item.edit_rect.height, None, "manual", -1)
        stats = roi.calculate_statistics(image[..., item.channel_index])
        ellp = roi.calculate_ellipse_parameters()
        stat_data = (hash(roi), image_id, stats["area"], stats["intensity average"],
                     stats["intensity median"], stats["intensity maximum"], stats["intensity minimum"],
                     stats["intensity std"], ellp["eccentricity"], ellp["roundness"],
                     item.center[0], item.center[1], item.width / 2, item.height / 2,
                     item.angle, ellp["area"], ellp["orientation_x"], ellp["orientation_y"],
                     ellp["shape_match"])
        # Prepare data for SQL statement
        rle = [(hash(roi), x[0], x[1], x[2]) for x in rle]
        # Write item to database
        self.inserter.save_roi_data_for_image(image_id, roidat, rle, stat_data)

    def delete_item_from_database(self, roihash: int) -> None:
        """
        Method to delete the item, specified by its hash, from the database

        :param roihash: The hash of the item
        :return: None
        """
        self.inserter.delete_roi_from_database(roihash)

    @staticmethod
    def create_associations(main: int, maps: Iterable[np.ndarray], unassociated: List[int]) -> Dict:
        """
        Method to create associations dictionary to associate nuclei with foci

        :param main: Index of the main channel
        :param maps: Hash maps for each channel
        :param unassociated: List of unassociated ROI hashes
        :return: Dictionary containing the associations
        """
        # Create new associations
        associations = {}
        for c in range(len(maps)):
            if c != main:
                for y in range(maps[0].shape[0]):
                    for x in range(maps[0].shape[1]):
                        if maps[c][y][x] and maps[main][y][x]:
                            associations[maps[c][y][x]] = maps[main][y][x]
        # Clean list
        associations = {x: y for x, y in associations.items() if x in unassociated}
        return associations

    def delete_roi(self, unassociated: Iterable[Tuple[int]]) -> None:
        """
        Method to delete unassociated roi from the database

        :param unassociated: List of hashes from unassociated roi, prepared for executemany
        :return: None
        """
        # Remove roi from handler
        self.roi.remove_rois_by_hash(unassociated)
        for roi_hash in unassociated:
            self.inserter.delete_roi_from_database(roi_hash)

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
    MARKERS = {
        "invisible": pg.mkPen(color=(0, 0, 0, 0)),
        "image processing": pg.mkPen(color="r", width=3),
        "machine learning": pg.mkPen(color="g", width=3),
        "merged": pg.mkPen(color="m", width=3),
        "nucleus": pg.mkPen(color="y", width=3),
        "removed": pg.mkPen(color="w", width=3)
    }
    """MARKERS = [
        pg.mkPen(color="r", width=3),  # Red
        pg.mkPen(color="g", width=3),  # Green
        pg.mkPen(color="b", width=3),  # Blue
        pg.mkPen(color="c", width=3),  # Cyan
        pg.mkPen(color="m", width=3),  # Magenta
        pg.mkPen(color="y", width=3),  # Yellow
        pg.mkPen(color="k", width=3),  # Black
        pg.mkPen(color="w", width=3),  # White
        pg.mkPen(color=(0, 0, 0, 0))  # Invisible
    ]"""

    @staticmethod
    def change_opacity(items: Iterable[QGraphicsItem],
                       opacity: float) -> None:
        """
        Method to change the opacity of the given items

        :param items: The items to change the opacity of
        :param opacity: New value for the opacity. [0-100]
        :return: None
        """
        for item in items:
            item.setOpacity(opacity / 100)
    
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
                items.append(ROIDrawer.draw_nucleus(view, roi, ind, False))
            else:
                items.append(ROIDrawer.draw_focus(view, roi, ind, False))
        return items

    @staticmethod
    def draw_focus(view: pg.PlotItem, roi: ROI, ind: int, visible: bool = True) -> QGraphicsEllipseItem:
        """
        Function to draw a focus onto the given view

        :param view: The view to draw on
        :param roi: The focus to draw
        :param ind: The index of the roi channel
        :param visible: Should the item be drawn visibly?
        :return: None
        """
        dims = roi.calculate_dimensions()
        pen = ROIDrawer.MARKERS[roi.detection_method.lower()]
        c = dims["minX"], dims["minY"]
        d2 = dims["height"]
        d1 = dims["width"]
        focus = FocusItem(c[0], c[1], d1, d2, ind, hash(roi), method=roi.detection_method)
        focus.set_pen(pen, ROIDrawer.MARKERS["invisible"])
        focus.setVisible(visible)
        focus.add_to_view(view)
        return focus

    @staticmethod
    def draw_nucleus(view: pg.PlotItem, roi: ROI, ind: int, visible: bool = True) -> QGraphicsEllipseItem:
        """
        Function to draw a nucleus onto the given view

        :param view: The view to draw on
        :param roi: The nucleus to draw
        :param ind: The index of the roi channel
        :param visible: Should the item be drawn visibly?
        :return: None
        """
        pen = pg.mkPen(color="#191970" if roi.auto else "#2A2ABB",
                       width=3, style=QtCore.Qt.DashLine)
        params = roi.calculate_ellipse_parameters()
        cy, cx = params["center_y"], params["center_x"]
        r1 = params["minor_axis"]
        r2 = params["major_axis"]
        angle = params["angle"]
        ovx, ovy = params["orientation_x"], params["orientation_y"]
        nucleus = NucleusItem(cx - r2, cy - r1, r2 * 2, r1 * 2, cx, cy, angle, (ovx, ovy), ind, hash(roi))
        nucleus.set_pens(
            pen,
            pen,
            ROIDrawer.MARKERS["invisible"]
        )
        nucleus.is_active()
        nucleus.update_indicators()
        nucleus.setVisible(visible)
        nucleus.add_to_view(view)
        return nucleus

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
        self.ipen = ROIDrawer.MARKERS["invisible"]
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
        "method"
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

    def __init__(self, x: int, y: int, width: int, height: float, index: int, roi_ident: int, method: str = "IP"):
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
        self.method = method
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
