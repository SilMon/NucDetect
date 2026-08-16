import threading
from typing import List, Iterable, Dict, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import QRectF, Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QColor, QKeyEvent, QMouseEvent
from PyQt5.QtWidgets import QDialog, QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem
from pyqtgraph import ColorBarItem
from skimage.draw import ellipse

from core.DataProcessing import create_lg_lut, automatic_colorbalance
from core.detector_modules.AreaAndROIExtractor import get_nearest_nucleus
from core.logging_config import get_logger
from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler
from core.database.connections import Requester, Inserter
from gui.Util import assert_main_thread
from gui.loader import ROIDrawerTimer

LOGGER = get_logger(__name__)

# Channel index meaning "the composite view", i.e. no single channel is active. Named
# rather than written as a literal because the producer and the consumer used to disagree:
# show_channel passed image.shape[2] and ROIDrawer.change_channel tested against 3, so the
# two only agreed for a 3-channel image. With four or five channels every ROI failed the
# channel test and the composite view of a 5-channel image drew no foci at all
COMPOSITE_CHANNEL = -1


class EditorView(pg.GraphicsView):
    # Emitted once the white-balance and high-contrast variants have been computed. They are
    # calculated on a background thread, and the two check boxes they enable may only be touched on
    # the GUI thread, so the hand-off goes through a queued connection rather than a direct call
    variants_ready_signal = pyqtSignal()

    COLORS = [
        QColor(255, 50, 0),  # Red
        QColor(50, 255, 0),  # Green
        QColor(255, 255, 0),  # Yellow
        QColor(255, 0, 255),  # Magenta
        QColor(0, 255, 255),  # Cyan
    ]

    def __init__(self, image: np.ndarray, roi: ROIHandler,
                 parent: QDialog, active_channels: List[Tuple[int, str]],
                 size_factor: float = 1, high_contrast: bool = False,
                 adjust_whitebalance: bool = False,
                 x_scale: float = 1, y_scale: float = 1):
        """
        :param image: The background image to display
        :param roi: All roi associated with this image
        :param parent: The EditorDialog incorporating this view
        :param active_channels: List containing the index of the channel and its corresponding name
        :param size_factor: Size factor used for newly added ROI
        :param high_contrast: If true, the channels will be shown in high contrast mode
        :param adjust_whitebalance: If true, white balance will be applied to all channels
        :param x_scale: Scale factor for x-axis
        :param y_scale: Scale factor for y-axis
        """
        super(EditorView, self).__init__()
        # Named _dialog, not parent: "parent" is a QWidget method, and assigning over it made
        # every view.parent() call raise "'Editor' object is not callable". Same convention as
        # StatisticsDialogMenuBar._dialog in gui/dialogs/data.py
        self._dialog = parent
        self.active_channels = {x[1]: x[0] for x in active_channels}
        self.size_factor = size_factor
        self.high_contrast = high_contrast
        self.adjust_whitebalance = adjust_whitebalance
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = -1
        self.image = image
        self.image_adj = None
        self.hcimg = None
        self.hcimg_adj = None
        # A channel NAME, not an index -- it is used as a key into self.active_channels (~:390),
        # which maps name -> index. The previous ": int" annotation named the value on the other
        # side of that lookup
        self.active_channel: Optional[str] = None
        self.roi: ROIHandler = roi
        self.requester = Requester()
        self.inserter = Inserter()
        self.main_channel = self.requester.get_main_channel(self.roi.ident)
        # The editor cannot work without one: it decides which items are nuclei (~:471) and indexes
        # active_channels by it (~:476). get_main_channel answers None rather than raising as of
        # 2026-08-17, so the failure is raised HERE, naming the image, instead of surfacing later
        # as a KeyError on None from inside a drawing routine
        if self.main_channel is None:
            raise ValueError(f"Image {self.roi.ident} has no main channel recorded -- "
                             f"the editor cannot be opened for it")
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
        self.mpos: Optional[QPointF] = None
        # Activate mouse tracking for widget
        self.setMouseTracking(True)
        self.setCentralWidget(self.plot_item)
        self.draw_additional = True
        # List of existing items
        self.loading_timer: Optional[ROIDrawerTimer] = None
        self.roi_items = []
        self.draw_roi()
        # List for newly created items
        self.temp_items = []
        # List for items that should be removed
        self.delete: List[int] = []
        self.selected_item: Optional[ROIItem] = None
        # The item currently under the cursor, highlighted with its hover pen. Tracked here
        # rather than by the items themselves -- see ROIItem.set_hovered for why
        self.hovered_item: Optional["ROIItem"] = None
        self.shift_down = False
        # Add a color bar widget
        self.color_bar = ColorBarItem(values=(np.amin(image[...,0]), np.amax(image[..., 0])))
        # Link the bar to the image
        self.color_bar.setImageItem(self.img_item, insert_in=self.plot_item)
        self.show_channel("Composite")
        self.current_channel = "Composite"
        # Add a scale bar to this view
        self.scale_microns = 10
        self.scale_bar = pg.ScaleBar(size=self.scale_microns * self.x_scale, width=15)
        self.scale_bar.setParentItem(self.plot_item.getViewBox())
        self.scale_bar.text.setPlainText(f"{self.scale_microns} µm")
        self.scale_bar.anchor((1, 1), (1, 1), offset=(-50, -50)) # Position set to bottom right
        self.addItem(self.scale_bar)
        # Connected before the thread is started, or a fast worker could emit into nothing
        self.variants_ready_signal.connect(self.enable_variant_modes)
        self.initialize_wb_and_hc()

    def initialize_wb_and_hc(self) -> None:
        """
        Method to initialize the high-contrast and white balanced mode

        :return: None
        """
        init_thread = threading.Thread(target=self.calculate_hc_and_wb_images, daemon=True)
        init_thread.start()

    def enable_variant_modes(self) -> None:
        """
        Method to enable the two check boxes whose images are prepared in the background

        Connected to variants_ready_signal, thus always executed on the GUI thread. The worker
        used to call the two enable_* methods itself, which is undefined behaviour rather than a
        race producing a stale pixel -- a QWidget may only be touched from the thread it lives in

        :return: None
        """
        assert_main_thread("EditorView.enable_variant_modes")
        self._dialog.enable_white_balance_mode()
        self._dialog.enable_high_contrast_mode()

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
        ROIDrawer.draw_additional_items(self.roi_items, self.draw_additional)

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
            self._dialog.enable_editing_widgets(False)
        self.active_channel = channel
        # Check to which index the name corresponds
        index = COMPOSITE_CHANNEL if channel == "Composite" else self.active_channels[channel]
        displayed = self.get_displayed_image(index)
        self.img_item.setImage(displayed)
        # Recalibrate the colour bar to what is actually on screen. It was built once from
        # channel 0 in __init__ and never updated, so every other channel was read against
        # channel 0's range -- a wrong quantitative readout, not a cosmetic one
        self.color_bar.setLevels((float(np.amin(displayed)), float(np.amax(displayed))))
        ROIDrawer.change_channel(self.roi_items, index, self.draw_additional)

    def get_displayed_image(self, index: int) -> np.ndarray:
        """
        Method to get the image data currently shown for the given channel

        :param index: The channel index, or COMPOSITE_CHANNEL for the composite view
        :return: The image data displayed for that channel
        """
        # The composite branch used to ignore high contrast entirely and always show self.image,
        # so the composite view was the one place the checkbox did nothing. The variants are
        # computed in a background thread and are still None while the editor starts up, hence
        # the fallbacks -- handing None to setImage is not an improvement over ignoring the flag
        image = self.image
        if self.high_contrast and self.hcimg is not None:
            image = self.hcimg
            if self.adjust_whitebalance and self.hcimg_adj is not None:
                image = self.hcimg_adj
        elif self.adjust_whitebalance and self.image_adj is not None:
            image = self.image_adj
        return image if index == COMPOSITE_CHANNEL else image[..., index]

    def calculate_hc_and_wb_images(self):
        """
        Method used for concurrency

        :return: None
        """
        self.image_adj = automatic_colorbalance(self.image)
        self.hcimg = self.create_high_contrast_image()
        self.hcimg_adj = automatic_colorbalance(self.hcimg)
        # Emit, do not call: this runs on a plain threading.Thread, and the two check boxes it
        # enables live on the GUI thread. A queued connection is what carries the hand-off
        self.variants_ready_signal.emit()

    def create_high_contrast_image(self) -> np.ndarray:
        """
        Method to create the needed high contrast image

        :return: None
        """
        # lut[channel], not a per-pixel loop: img[y][x][c] = lut[channel[y][x]] is exactly this
        # expression written out one pixel at a time. Measured, with the numba JIT warmed up
        # first, identical output in every case:
        #     1024x1024x3 uint8   1.35 s -> 0.023 s
        #     2048x2048x5 uint16 10.28 s -> 0.237 s
        # This runs in a background thread, so what it delays is the high-contrast checkbox
        # becoming enabled, not the editor opening
        #
        # The array stays float64. The finding that prompted this also proposed taking the dtype
        # from the source image to save memory -- that is wrong and would corrupt the output:
        # create_lg_lut maps a value n to (n*n + n) // 2, so a 16-bit channel produces entries up
        # to ~2.1e9 and an 8-bit one up to 32640. Neither fits the source dtype
        img = np.zeros(shape=self.image.shape)
        for c in range(img.shape[2]):
            channel = self.image[..., c]
            # Create a lut
            lut = np.asarray(create_lg_lut(np.amax(channel)), dtype=np.float64)
            img[..., c] = lut[channel]
        return img

    def toggle_high_contrast_mode(self, toggle: bool) -> None:
        """
        Method to toggle high contrast mode

        :param toggle: Toggle
        :return: None
        """
        self.high_contrast = toggle
        self.show_channel(self.current_channel)

    def toggle_adjust_white_balance(self, toggle: bool) -> None:
        """
        Method to toggle automatic white balance mode

        :param toggle: Toggle
        :return: None
        """
        self.adjust_whitebalance = toggle
        self.show_channel(self.current_channel)

    def change_colormap(self, colormap: str) -> None:
        """
        Method to load the given colormap

        :param colormap: Name of the colormap to load
        :return: None
        """
        # Update the image item
        self.img_item.setColorMap(pg.colormap.get(colormap, source="matplotlib"))
        # Update the corresponding color bar
        self.color_bar.setColorMap(pg.colormap.get(colormap, source="matplotlib"))

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
            for item in self.roi_items:
                if item.roi_id not in self.delete:
                    if item.roi_id == ident:
                        item.changed = True

    def clear_and_update(self) -> None:
        """
        Method to redraw the roi of this view from the handler

        :return: None
        """
        # Taken out of the SCENE, not just out of the bookkeeping list. items.clear() on its own
        # left every ellipse on the plot, so draw_roi painted a second full set over the first
        for item in self.roi_items:
            item.remove_from_view(self)
        self.roi_items.clear()
        # Dropped with the items it could point at -- the highlight is re-established by the next
        # mouse move
        self.hovered_item = None
        self.draw_roi()
        # Only the items the user actually edited. This used to pass every item's roi_id -- the
        # list was called "changed" but was built from self.roi_items in full -- so a redraw marked the
        # whole image as edited
        self.mark_as_changed([x.roi_id for x in self.temp_items])
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

    def update_loading(self, items: List[QGraphicsItem], finished: bool = False) -> None:
        """
        Method to update the progress bar

        :param items: The items loaded in this batch
        :param finished: True when the loader has no items left. Unused here -- this consumer keys
        off the percentage rather than off the end of the load -- but part of the feedback contract
        :return: None
        """
        self._dialog.ui.prg_loading.setValue(int(self.loading_timer.percentage * 100))
        self.roi_items.extend(items)
        if round(self.loading_timer.percentage * 100) >= 99:
            for item in self.roi_items:
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
                item = self.selected_item
                # Remove item from scene
                item.remove_from_view(self)
                # Add item to deletion list to remove it from the database
                if item.roi_id != -1:
                    self.delete.append(item.roi_id)
                # Drop it from the bookkeeping too, and stop treating it as selected. Without this
                # the removed item stayed in self.roi_items and stayed self.selected_item, so a second
                # Delete appended the same roi_id again and the editing spin boxes went on driving
                # an item that is no longer in the scene
                if item in self.roi_items:
                    self.roi_items.remove(item)
                if item in self.temp_items:
                    self.temp_items.remove(item)
                if item is self.hovered_item:
                    self.hovered_item = None
                self.selected_item = None
                self._dialog.enable_editing_widgets(False)
        # Keys 1/2/3 are deliberately NOT bound here. Editor.keyPressEvent binds them to
        # set_mode(), which checks the corresponding toolbar button and lets the button group
        # drive change_mode -- so the toolbar and the mode stay in step. Binding them here as well
        # called change_mode() directly, leaving the buttons showing the previous mode, and which
        # of the two handlers ran depended on which widget had focus

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
        self._dialog.setup_editing(self.selected_item)

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
            self._dialog.setup_editing(self.selected_item)

    def create_new_item_at_mouse_position(self) -> None:
        """
        Method to create a new item at the current mouse position

        :return: None
        """
        # Get click position
        pos = self.mpos
        is_nucleus = self.active_channel == self.main_channel
        if is_nucleus:
            item = NucleusItem(round(pos.x() - 45 * self.size_factor), round(pos.y() - 23 * self.size_factor),
                               round(90 * self.size_factor), round(46 * self.size_factor),
                               round(pos.x()), round(pos.y()),
                               0, (0, 0), self.active_channels[self.main_channel], -1)
            item.set_pens(
                ROIDrawer.MARKERS["nucleus_manual"],
                ROIDrawer.MARKERS["nucleus_manual"],
                ROIDrawer.MARKERS["invisible"]
            )
        else:
            item = FocusItem(round(pos.x() - 2 * self.size_factor), round(pos.y() - 2 * self.size_factor),
                             round(4 * self.size_factor), round(4 * self.size_factor),
                             self.active_channels[self.active_channel], -1)
            item.set_pen(
                ROIDrawer.MARKERS["manual"],
                ROIDrawer.MARKERS["invisible"]
            )
        item.changed = True
        self.roi_items.append(item)
        self.temp_items.append(item)
        # ONE add, on ONE path. The nucleus branch used to add here as well as in the tail, because
        # enable_editing(True) below needs the item to be in the view already -- it attaches the
        # editing rectangle to the item's scene. Qt refused the second add and printed a warning,
        # so nothing was duplicated, but the two call sites disagreed about whose job the add was.
        item.add_to_view(self.plot_item)
        if is_nucleus:
            self._dialog.set_mode(2)
            # change_mode clears any previously selected item, so it must run BEFORE the assignment
            self.change_mode(1)
            self.selected_item = item
            item.enable_editing(True)
            self._dialog.setup_editing(item)

    def mouse_moved(self, event: QMouseEvent) -> None:
        pos = event[0]
        if self.plot_item.sceneBoundingRect().contains(pos):
            if self.pos_track:
                coord = self.plot_vb.mapSceneToView(pos)
                self.mpos = coord
                self._dialog.set_status(f"X: {coord.x():.2f} Y: {coord.y():.2f}")
            self.update_hovered_item(pos)

    def update_hovered_item(self, scene_pos: QPointF) -> None:
        """
        Method to highlight the item lying under the cursor

        :param scene_pos: The cursor position, in scene coordinates
        :return: None
        """
        # Gated on the same conditions under which a click would select an item, so the highlight
        # shows what the next click would hit -- and so the lookup does not run while the user is
        # only looking at the image. The items cannot report this themselves: all but the one
        # being edited are setEnabled(False), and a disabled QGraphicsItem is sent no hover events
        candidate = None
        if self.mode == 1 and self.active_channel != "Composite":
            active_index = self.active_channels[self.active_channel]
            under_cursor = [x for x in self.scene().items(scene_pos)
                            if isinstance(x, ROIItem) and x.channel_index == active_index]
            if under_cursor:
                # [-1] to match select_item_at_mouse_position, which picks the same one
                candidate = under_cursor[-1]
        if candidate is self.hovered_item:
            return
        if self.hovered_item is not None:
            self.hovered_item.set_hovered(False)
        self.hovered_item = candidate
        if candidate is not None:
            candidate.set_hovered(True)

    def set_item_opacity(self, opacity: float) -> None:
        """
        Method to set the opacity of all ROIItems

        :param opacity: The opacity value [0-100]
        :return: None
        """
        ROIDrawer.change_opacity(self.roi_items, opacity)

    def delete_items_in_list(self) -> None:
        """
        Method to delete all roi in the self.delete list

        :return: None
        """
        # Remove deleted roi from item list
        self.roi_items = [x for x in self.roi_items if x.roi_id not in self.delete]
        if self.hovered_item is not None and self.hovered_item not in self.roi_items:
            self.hovered_item = None
        # Delete items marked for it
        self.delete_roi(self.delete)

    def create_association_maps(self) -> List[np.ndarray]:
        """
        Method to get the hash association maps

        :return: List of all created association maps
        """
        # Create list of changed items to ignore during map creation
        ignore = [x.roi_id for x in self.roi_items if x.changed]
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
        new_roi = []
        for item in self.roi_items:
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
                        unassociated.extend(hashes)
                        self.inserter.reset_nucleus_focus_association(item.roi_id)
                    else:
                        unassociated.append(item.roi_id)
                # Get coordinates corresponding to the item
                # Process width/height
                height = round(item.height)
                width = round(item.width)
                # skimage's ellipse() spans an ODD number of pixels around an integer centre
                # (2r-1 for integer r), so an even requested size cannot be drawn on one. That is
                # what the previous `+ 1` was reaching for -- but inflating the radius OVERSHOOTS by
                # a whole pixel: measured, a size of 4 drew 5 px, 46 drew 47, 90 drew 91, so a
                # default 4 px focus was stored 25 % larger than it was drawn. Halving alone
                # undershoots by one instead (4 -> 3). Offsetting the CENTRE by half a pixel is what
                # actually makes the span even: centre + 0.5 with r = 2.0 draws exactly 4 px.
                # The old form also mixed the rounded local with the unrounded item.height/width
                cy = item.center[1] + (0.5 if height % 2 == 0 else 0.0)
                cx = item.center[0] + (0.5 if width % 2 == 0 else 0.0)
                rr, cc = ellipse(cy, cx, height / 2, width / 2,
                                 self.image.shape, np.deg2rad(-item.angle))
                # Get encoded area for item
                rle = self.encode_new_roi(rr, cc, maps[item.channel_index])
                if rle:
                    # Create new ROI instance
                    # method="manual" rather than the constructor default "Not Set": the row
                    # written to the database below already says "manual", and ROIDrawer.MARKERS
                    # has no "not set" key -- so a redraw of the handler after a manual focus was
                    # added raised KeyError: 'not set' in draw_focus, and the in-memory object
                    # disagreed with its own stored row
                    roi = ROI(channel=self.roi.idents[item.channel_index],
                              main=isinstance(item, NucleusItem), auto=False,
                              method="manual")
                    roi.set_area(rle)
                    roihash = hash(roi)
                    # Foci need to be associated
                    if isinstance(item, FocusItem):
                        unassociated.append(roihash)
                    self.replace_placeholder(maps[item.channel_index], roihash)
                    self.write_item_to_database(item, roi, rle, self.image, self.roi.ident)
                    # Add ROI to ROIHandler
                    new_roi.append(roi)
                else:
                    LOGGER.warning("ROI does not contain any points!")
        self.roi.rois.extend(new_roi)


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
        # The centers come from the ROI themselves rather than from the maps, so both association
        # paths measure the same point -- calculate_dimensions is what associate_roi reads too
        centers = {}
        for roi in self.roi:
            dims = roi.calculate_dimensions()
            centers[hash(roi)] = (dims["center_y"], dims["center_x"])
        associations = self.create_associations(self.roi.idents.index(self.roi.main), maps,
                                                unassociated, centers)
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
        # Check if the roi is valid
        if not roi.is_valid():
            return
        # Calculate statistics
        roidat = (hash(roi), image_id, False, roi.ident,
                  item.center[0], item.center[1], item.edit_rect.width,
                  item.edit_rect.height, None, "manual", -1, roi.colocalized)
        stats = roi.calculate_statistics(image[..., item.channel_index])
        # TODO replace for FOCI
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
    def create_associations(main: int, maps: Iterable[np.ndarray], unassociated: List[int],
                            centers: Dict[int, Tuple[int, int]]) -> Dict:
        """
        Method to create associations dictionary to associate nuclei with foci

        Overlap decides whether a focus is associated, and the distance between the two centers
        decides with which nucleus -- the same rule the detector's associate_roi applies, so that a
        focus does not change owner depending on which of the two paths last wrote it. This used to
        keep whichever nucleus the LAST SCANNED PIXEL belonged to, which is a scan order and not a
        rule: for a focus spanning two nuclei the bottom-right one won

        The scan itself was three nested Python loops over every pixel of every channel -- 2.1 M
        iterations for a 1024x1024 image with three channels, 0.39 s on the GUI thread on every
        save. The masked np.unique below is the same question asked once per channel

        :param main: Index of the main channel
        :param maps: Hash maps for each channel
        :param unassociated: List of unassociated ROI hashes
        :param centers: The center of every roi, as {hash: (y, x)}
        :return: Dictionary containing the associations
        """
        # Every nucleus a focus overlaps, as {focus hash: {nucleus hash}}
        overlaps: Dict[int, set] = {}
        for c in range(len(maps)):
            if c == main:
                continue
            both = (maps[c] != 0) & (maps[main] != 0)
            if not both.any():
                continue
            pairs = np.unique(np.stack([maps[c][both], maps[main][both]]), axis=1)
            for focus, nucleus in zip(pairs[0], pairs[1]):
                overlaps.setdefault(int(focus), set()).add(int(nucleus))
        associations = {}
        for focus, nuclei in overlaps.items():
            if focus not in unassociated or focus not in centers:
                continue
            nearest = get_nearest_nucleus(centers[focus],
                                          {n: centers[n] for n in nuclei if n in centers})
            if nearest:
                associations[focus] = nearest
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
        map_[map_ == placeholder] = roihash

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
            rl = 1
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
        "manual": pg.mkPen(color="b", width=3),
        "nucleus_auto": pg.mkPen(color="#b36920", width=3, style=QtCore.Qt.DashLine),
        "nucleus_manual": pg.mkPen(color="#d67c22", width=3, style=QtCore.Qt.DashLine),
        "removed": pg.mkPen(color="w", width=3)
    }

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
    # ROIItem, not QGraphicsItem: the body reads channel_index/is_active/update_indicators, none
    # of which the Qt base class has. Quoted because ROIItem is declared further down this file
    def change_channel(items: Iterable["ROIItem"],
                       active_channel: int = COMPOSITE_CHANNEL,
                       draw_additional: bool = False) -> None:
        """
        Method to change the drawing of foci and nuclei according to the active channel

        :param items: The items that are drawn on the view
        :param active_channel: The active channel
        :param draw_additional: Parameter to draw items for additional information
        :return: None
        """
        for item in items:
            if item.channel_index != active_channel and active_channel != COMPOSITE_CHANNEL:
                if isinstance(item, NucleusItem) and draw_additional:
                    item.is_active(True)
                else:
                    item.is_active(False)
            else:
                item.is_active(True)
            item.update_indicators(draw_additional)

    @staticmethod
    # idents is a Sequence, not an Iterable: the body calls idents.index(), which no plain
    # iterable has -- a generator or a set would raise AttributeError at that line
    def draw_roi(view: pg.PlotItem, rois: Iterable[ROI], idents: Sequence[str]) -> List[QGraphicsEllipseItem]:
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
        focus = FocusItem(c[0], c[1], d1, d2, ind, hash(roi))
        focus.set_pen(pen, ROIDrawer.MARKERS["invisible"])
        focus.setVisible(visible if roi.detection_method != "removed" else False)
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
        pen = ROIDrawer.MARKERS["nucleus_auto"] if roi.auto else ROIDrawer.MARKERS["nucleus_manual"]
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
    # ROIItem for the same reason as change_channel above
    def draw_additional_items(items: List["ROIItem"], draw_additional: bool = True) -> None:
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

    # No __slots__ here or on ROIItem, deliberately. A sip type provides __dict__ from the C++
    # base, so a __slots__ on a QGraphicsItem subclass cannot remove it and cannot reject an
    # undeclared attribute -- measured: the slotted subclass still had a __dict__, still accepted
    # an undeclared name, and was 160 bytes against 144 without, the descriptors being pure
    # overhead. The one place the memory argument does hold is core/roi/ROI.py's ROI, a plain
    # class whose __slots__ works and is kept
    #
    # pos_x/pos_y/active_pen rather than x/y/pen: QGraphicsItem.x(), .y() and .pen() are real
    # methods, and assigning attributes over them makes the accessors uncallable

    def __init__(self, x, y, cx, cy, width, height):
        super().__init__(x, y, width, height)
        self.pos_x = x
        self.pos_y = y
        self.width = width
        self.height = height
        self.center = cx, cy
        self.inactive_pen = None
        self.active_pen = None
        self.color = None
        self.initialize()

    def initialize(self):
        """
        Method to initialize this class

        :return:  None
        """
        self.active_pen = pg.mkPen(color="#bdff00", width=3, style=QtCore.Qt.DashLine)
        self.inactive_pen = ROIDrawer.MARKERS["invisible"]
        self.setPen(self.active_pen)

    def activate(self, enable: bool = True) -> None:
        """
        Method to activate this item

        :param enable: Bool
        :return: None
        """
        if enable:
            self.setPen(self.active_pen)
        else:
            self.setPen(self.inactive_pen)


class ROIItem(QGraphicsEllipseItem):
    # The __slots__ block that stood here was deleted on 2026-08-15. It had two missing commas,
    # fusing "preview" "changed" and "method" "channel_index" so four names were never declared,
    # and it listed "pen" twice -- but repairing it would have bought nothing: see the note on
    # EditingRectangle above for the measurement. __slots__ does not work on a sip subclass

    def __init__(self, x: int, y: int, width: int, height: float, index: int, roi_ident: int):
        super().__init__(x, y, width, height)
        self.preview = False
        self.changed = False
        self.item_rect = QRectF(x, y, width, height)
        self.pos_x = x
        self.pos_y = y
        self.width = width
        self.height = height
        self.center = int(self.pos_x + self.width / 2), int(self.pos_y + self.height / 2)
        self.angle = 0
        self.channel_index = index
        self.roi_id = roi_ident
        # A `method` parameter and attribute stood here until 2026-08-15. Nothing read it -- the
        # drawing code goes to roi.detection_method on the ROI, not to the item -- and its default
        # was "IP", which is not one of ROIDrawer.MARKERS' keys, so a reader that ever did consult
        # it would have got a value the marker lookup cannot resolve.
        self.active_pen: pg.mkPen = None
        self.inactive_pen: pg.mkPen = None
        self.hover_pen: pg.mkPen = None
        self.main_color = None
        self.hover_color = None
        # Whether this item is currently drawn as active, and whether the cursor is over it.
        # Both are needed because the pen depends on the two together -- see apply_pen
        self.active = True
        self.hovered = False
        self.view: Optional[EditorView] = None
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
            self.item_rect = rect
            self.pos_x = rect.x()
            self.pos_y = rect.y()
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
        self.update_data(self.item_rect, self.angle)
        self.preview = False

    def remove_from_view(self, view: EditorView) -> None:
        """
        Method to remove this item from the given view

        :param view: The view to remove the item from
        :return: None
        """
        # The edit rectangle is built by add_to_view / initialize but only put INTO the scene by
        # enable_editing, so an item the user never selected has one that belongs to no scene.
        # Removing it unconditionally made Qt log a warning per item -- and adding it in
        # add_to_view instead is not the fix: enable_editing would then add the same item twice
        if self.edit_rect is not None and self.edit_rect.scene() is not None:
            view.scene().removeItem(self.edit_rect)
        view.scene().removeItem(self)

    def is_active(self, active: bool = True) -> None:
        """
        Method to set the activity of this item

        :param active: Bool
        :return: None
        """
        self.active = active
        self.apply_pen()

    def set_hovered(self, hovered: bool = True) -> None:
        """
        Method to mark this item as lying under the cursor

        Driven by EditorView.mouse_moved rather than by hoverEnterEvent: every item is
        setEnabled(False) except the one being edited, and a disabled QGraphicsItem receives no
        hover events at all, even with setAcceptHoverEvents(True) -- measured, not assumed.

        :param hovered: Bool
        :return: None
        """
        if hovered == self.hovered:
            return
        self.hovered = hovered
        self.apply_pen()

    def apply_pen(self) -> None:
        """
        Method to draw this item with the pen its current state calls for

        :return: None
        """
        if not self.active:
            self.setPen(self.inactive_pen)
        elif self.hovered and self.hover_pen is not None:
            self.setPen(self.hover_pen)
        else:
            self.setPen(self.active_pen)

    def update_indicators(self, draw: bool = True) -> None:
        """
        Dummy Method to be compatible with NucleusItem
        """
        pass

    def set_pen(self, pen: pg.mkPen, inactive_pen: pg.mkPen):
        self.active_pen = pen
        self.inactive_pen = inactive_pen
        # Define needed colors
        self.main_color = pen.color()
        # lighter(160), not lighter(100): 100 % is the identity, so the "hover colour" was the
        # base colour and hovering could not have looked any different even once something read
        # it. The hover pen keeps the width and style of the active pen so only the colour moves
        self.hover_color = self.main_color.lighter(160)
        self.hover_pen = pg.mkPen(color=self.hover_color, width=pen.width(), style=pen.style())
        self.apply_pen()

    def add_to_view(self, view: EditorView) -> None:
        """
        Method to add this item and all associated items to the given view

        :param view: View to add to
        :return: None
        """
        self.view = view
        view.addItem(self)
        rect = EditingRectangle(self.pos_x, self.pos_y, self.center[0], self.center[1],
                                self.width, self.height)
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
        self.item_rect = None
        self.angle = angle
        self.center = center_x, center_y
        self.orientation = orientation
        self.indicators = []
        self.edit = False
        self.edit_rect: Optional[EditingRectangle] = None
        # The pen for the major/minor axis indicators. It used to be stored in ipen, which on the
        # base class means the INACTIVE pen, while the inactive pen lived here in iapen -- so
        # ROIItem.is_active painted a nucleus with its indicator pen whenever the override below
        # did not catch it first. One name, one meaning, in both halves of the hierarchy
        self.indicator_pen: pg.mkPen = None
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
        # The pen choice itself is now the base class's, because inactive_pen means the same
        # thing on both halves of the hierarchy. Only the edit rectangle is special here
        super().is_active(active)

    def update_indicators(self, draw: bool = True) -> None:
        """
        Method update the drawing of indicators

        :param draw: Bool to indicate if the indicators should be drawn
        :return: None
        """
        for indicator in self.indicators:
            indicator.setPen(self.indicator_pen if draw else self.inactive_pen)

    def set_pens(self, pen: pg.mkPen, indicator_pen: pg.mkPen,
                 inactive_pen: pg.mkPen) -> None:
        """
        Method to set the pen to draw this item

        :param pen: The pen to draw this item when active
        :param indicator_pen: The pen to draw the indicators of this item with
        :param inactive_pen: The pen to use if this item is set to inactive
        :return: None
        """
        # set_pen builds the hover pen and applies the right one for the current state; the
        # indicator pen is the only one specific to this class
        self.set_pen(pen, inactive_pen)
        self.indicator_pen = indicator_pen
        for indicator in self.indicators:
            indicator.setPen(self.indicator_pen)

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
        rect = EditingRectangle(self.pos_x, self.pos_y, self.center[0], self.center[1],
                                self.width, self.height)
        rect.setTransformOriginPoint(rect.sceneBoundingRect().center())
        rect.setRotation(self.angle)
        self.indicators.extend([
            major_axis,
            minor_axis,
        ])
        self.edit_rect = rect
        # rect(), not boundingRect(): a QGraphicsEllipseItem's bounding rect is the item rect
        # adjusted outwards by half the pen width -- 1.5 px per side for the 3 px pens used here.
        # reset_item restores this rect, so cancelling a preview grew the nucleus a little each
        # time it was cancelled
        self.item_rect = self.rect()
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
        return f"NucleusItem X:{self.pos_x} Y:{self.pos_y} W:{self.width} H:{self.height} C:{self.center}"


class FocusItem(ROIItem):

    def __str__(self):
        return f"FocusItem X:{self.pos_x} Y:{self.pos_y} W:{self.width} H:{self.height} C:{self.center}"
