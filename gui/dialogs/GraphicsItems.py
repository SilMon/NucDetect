import threading
from typing import List, Iterable, Dict, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import QRectF, Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QColor, QKeyEvent, QMouseEvent
from PyQt5.QtWidgets import QDialog, QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem, QMessageBox
from pyqtgraph import ColorBarItem
from skimage.draw import ellipse

from core.DataProcessing import create_lg_lut, automatic_colorbalance
from core.detector_modules.AreaAndROIExtractor import get_nearest_nucleus
from core.logging_config import get_logger
from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler
from core.database.connections import Requester, Inserter
from gui import Util
from gui.Util import assert_main_thread, composite_channels
from gui.definitions.icons import Icon
from gui.dialogs.geometry import (HANDLE_SIGNS, angle_from_vector, resize_about_anchor,
                                  to_local)
from gui.loader import ROIDrawerTimer

LOGGER = get_logger(__name__)

# Channel index meaning "the composite view", i.e. no single channel is active. Named
# rather than written as a literal because the producer and the consumer used to disagree:
# show_channel passed image.shape[2] and ROIDrawer.change_channel tested against 3, so the
# two only agreed for a 3-channel image. With four or five channels every ROI failed the
# channel test and the composite view of a 5-channel image drew no foci at all
COMPOSITE_CHANNEL = -1


class DragState:
    """
    The gesture currently in progress in the editor

    A plain class, so __slots__ actually works -- unlike on the QGraphicsItem subclasses below,
    where a sip type supplies a __dict__ the declaration cannot remove
    """

    __slots__ = ("role", "item", "start_rect", "start_angle", "grab_x", "grab_y", "start_vector")

    def __init__(self, role: str, item: "ROIItem", grab_x: float = 0.0, grab_y: float = 0.0,
                 start_vector: float = 0.0):
        self.role = role
        self.item = item
        # The geometry BEFORE the gesture, so Escape can put it back and so a gesture that ends
        # where it started can decline to mark the item changed
        self.start_rect = QRectF(item.item_rect)
        self.start_angle = item.angle
        self.grab_x = grab_x
        self.grab_y = grab_y
        self.start_vector = start_vector


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
        self.discard_unusable_roi()
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
        # The gesture in progress, or None. Held here rather than on the items because every item is
        # setEnabled(False) and therefore receives no mouse events of its own -- the same reason the
        # hover highlight is driven from mouse_moved
        self.drag: Optional[DragState] = None
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
        # No addItem here. setParentItem above is pyqtgraph's documented way of attaching a scale
        # bar, and giving an item a parent already puts it into the parent's scene -- the view box
        # lives in this very scene, so adding it again made Qt print
        # "QGraphicsScene::addItem: item has already been added to this scene" on every editor
        # launch. Qt returned early and nothing was duplicated, so the bar always drew correctly;
        # the cost was the warning, on a console that is a diagnostic surface. That exact message
        # is what a genuine double-add produces -- this file has had one before, see the comment in
        # create_new_item_at_mouse_position.
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

    def set_changes(self, rect: QRectF, angle: float) -> None:
        """
        Method to apply the changes made by editing

        **This always commits.** It used to take a `preview` flag whose only True source was the
        Preview button, and that button was `enabled=false` in the .ui with nothing ever enabling
        it -- so the preview branch was unreachable for as long as it existed. Both buttons were
        removed on 2026-09-08 and the branch went with them.

        Previewing itself is not gone: every step of a DRAG is a preview, through
        `update_data(keep_original=True)` in `drag_to`, and `end_drag(commit=False)` on Escape puts
        the item back. That path is live and is the one the bounding-rect fix is exercised through.

        :param rect: The new bounding box of the currently active item
        :param angle: The angle of the currently active item
        :return: None
        """
        if self.selected_item is None:
            return
        self.commit_item_geometry(self.selected_item, rect, angle)

    def commit_item_geometry(self, item: "ROIItem", rect: QRectF, angle: float) -> None:
        """
        Method to apply a geometry change to an item and mark it to be written to the database

        THE commit path -- every gesture that changes an item's geometry for real goes through here,
        so that "the item is in temp_items" and "the item will be written" cannot come apart.

        They had come apart: process_changed_items skips any item that is not in temp_items, and
        temp_items was appended in only two places, the Accept button and the creation of a new item.
        move_selected_item_to_position set `changed = True` without appending, so the middle-click
        "move here" was SILENTLY DISCARDED on OK for every pre-existing roi -- a newly drawn one
        survived only because creating it had put it in the list for another reason.

        :param item: The item to apply the change to
        :param rect: The new, unrotated bounding box
        :param angle: The new angle, applied about the box's center
        :return: None
        """
        item.update_data(rect, angle, keep_original=False)
        # Guarded rather than appended blindly: Accept used to add the same item again on every
        # press, so a repeatedly adjusted nucleus appeared in the list several times
        if item not in self.temp_items:
            self.temp_items.append(item)
        self._dialog.update_editing_values(item)

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
        # The composite goes through composite_channels, which is a no-op at three channels or
        # fewer and folds a bigger stack onto RGB with false colour above that.
        #
        # Without it the raw array reached ImageItem, and pyqtgraph renders at most four planes --
        # `TypeError: data.shape[2] must be <= 4` on a 5-channel image, reported from real use on
        # 2026-08-22, and a SILENT failure at four, where the fourth plane is read as alpha and a
        # dark fluorescence channel makes the whole picture transparent.
        #
        # The single-channel branch needs nothing: image[..., index] is 2-D at any channel count.
        # This return also serves the high-contrast and white-balance variants chosen above, so
        # they are covered by the same call rather than by three more
        if index == COMPOSITE_CHANNEL:
            return composite_channels(image)
        # Defensive, and it says so rather than raising from inside a signal handler. Editor filters
        # the channel list against the array before this can be reached, so an out-of-range index
        # here means a new caller bypassed that -- worth a message naming both numbers instead of an
        # IndexError from a lambda
        # shape[-1] behind an ndim test, not shape[2]: a 2-D array has no third axis and the
        # bare subscript raised IndexError -- a guard written to prevent an IndexError producing
        # one. Same expression Editor.__init__ filters the channel list with
        available = image.shape[2] if image.ndim > 2 else 1
        if index >= available:
            LOGGER.warning("Channel %d requested for an image with %d channel(s) -- showing "
                           "channel 0", index, available)
            index = 0
        return image[..., index] if image.ndim > 2 else image

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

    def discard_unusable_roi(self) -> None:
        """
        Method to drop stored roi with no area from the handler, before anything reads it

        **Once, here, rather than wherever they are met.** A roi with no points cannot be drawn,
        measured or written: `calculate_dimensions` raises for it, `create_hash_association_maps`
        hands numba an untyped list, and both run on the way to saving. Skipping them at DRAW time
        only -- which is what this class did between 2026-09-05 and 2026-09-08 -- let the editor
        open and then fail on OK, and made the lazy loader's item count disagree with the roi count
        it was measured against.

        **Rebuilt rather than removed through `ROIHandler.remove_rois`**: `hash(roi)` is
        `md5(channel + area)`, so every area-less roi in one channel hashes alike and compares
        equal, and `list.remove` would be removing by an identity they all share. A comprehension
        says what is meant.

        `main` and `idents` are deliberately left alone -- neither is derived from the roi that go,
        because a nucleus with no area could not have been the main nomination in the first place.

        :return: None
        """
        unusable = [roi for roi in self.roi if not roi.is_valid()]
        if not unusable:
            return
        # TEMPORARY (2026-09-08, at RW's instruction): identified by channel and count, NOT by
        # hash. Every area-less roi in a channel produces the SAME hash -- md5 of the channel name
        # and an empty area -- so a per-row hash names nothing and repeats. Naming the row properly
        # needs `hash(roi)` and `roi.id` to stop being two different identifiers, which is a known
        # and separate piece of work; until then a channel and a count is all that can be said
        # honestly.
        per_channel = {}
        for roi in unusable:
            per_channel[roi.ident] = per_channel.get(roi.ident, 0) + 1
        LOGGER.warning("Image %s: discarding %d stored roi with no area (%s). They cannot be "
                       "drawn or saved; the rest of the image is unaffected",
                       self.roi.ident, len(unusable),
                       ", ".join(f"{n} in {channel}" for channel, n in sorted(per_channel.items())))
        # Rebuilt in place rather than through remove_rois, and `idents` and `main` are
        # deliberately NOT recomputed. Discarding a roi does not mean the image stopped having that
        # channel, and `main` is a NOMINATION -- since 2026-09-13 the loader takes it from the
        # channels table, which records it whether or not anything was detected in that channel.
        # Recomputing either from what is left would undo that and reintroduce the row 72 crash on
        # any image whose main channel is empty. Whether the handler should derive these at all is
        # an open question, not a decision to take here.
        self.roi.rois = [roi for roi in self.roi.rois if roi.is_valid()]

    def draw_roi(self) -> None:
        """
        Method to draw the roi

        :return: None
        """
        self.roi.sort_roi_list()
        self.loading_timer = ROIDrawerTimer(self.roi, self.plot_item,
                                            channels=self.active_channels,
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
        # `finished`, NOT the percentage. The percentage is items_loaded / len(items), and
        # items_loaded counts what the PROCESSING returned -- so any batch that drops an item holds
        # the percentage below 100 for the rest of the run and the >= 99 test never fires. Nothing
        # would then be made visible: the editor opened completely empty, with no error, which
        # reads as "nothing was detected". `finished` is computed from what each batch CONSUMED and
        # is correct however many items the processing drops
        if finished:
            for item in self.roi_items:
                item.setVisible(True)

    def _channel_name(self, index: int) -> str:
        """
        Method to get the channel name for a database channel index

        The reverse of `self.active_channels`, which is name -> index. Needed because a hand-drawn
        roi carries the INDEX and `ROI` stores the NAME, and since 2026-09-21 that index is the
        database one rather than a position in `idents` -- so the old `self.roi.idents[index]`
        would now read the wrong name whenever the two spaces disagree.

        :param index: The database/image channel index
        :return: The channel name
        :raises KeyError: if no active channel carries that index
        """
        for name, ind in self.active_channels.items():
            if ind == index:
                return name
        # Loud, and deliberately not a fallback: the name is written into the database row for the
        # roi being created, so guessing here would store an roi against the wrong channel -- which
        # is the defect this whole change exists to remove
        raise KeyError(f"No active channel has index {index}")

    # get_roi_index was removed here on 2026-09-21. It returned `idents.index(roi.ident)` -- the
    # wrong index space, the one this file no longer uses -- and it had NO callers, so it was a
    # trap waiting for someone who needed a channel index and found a method offering one.
    # `self.active_channels[roi.ident]` is the answer now.

    def keyPressEvent(self, event: QKeyEvent) -> None:
        super().keyPressEvent(event)
        if event.key() == Qt.Key_Shift:
            self.shift_down = True
        if event.key() == Qt.Key_Escape and self.drag is not None:
            # Gives ROIItem.reset_item its first caller. It had none, which is why the bounding-rect
            # defect it carried went unnoticed until 2026-08-15
            self.end_drag(commit=False)
            return
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
            # A grab point of the ALREADY selected item wins over selecting something else. The
            # points sit on the bounding box, so they can lie outside the ellipse and over a
            # neighbouring roi -- testing them first is what makes a corner of a rotated nucleus
            # grabbable at all
            if not self.start_handle_drag(event):
                self.select_item_at_mouse_position(event)
                self.start_move_drag(event)

    def image_position(self, event: QMouseEvent) -> QPointF:
        """
        Method to translate a widget position into image coordinates

        :param event: The mouse event to read the position from
        :return: The position in image coordinates
        """
        return self.plot_vb.mapSceneToView(self.mapToScene(event.pos()))

    def handle_at(self, widget_pos: QPointF, tolerance: int = 10) -> Optional["HandleItem"]:
        """
        Method to find the grab point of the selected item under the given position

        Hit-tested in DEVICE pixels rather than through scene().items(): a handle carries
        ItemIgnoresTransformations, so its size in image coordinates depends on the zoom level and a
        scene-space query would demand pixel accuracy of the user when zoomed out. A fixed pixel
        radius is what the user is actually aiming at

        :param widget_pos: The position to test, in widget coordinates
        :param tolerance: The grab radius in device pixels
        :return: The handle under the position, or None
        """
        if self.selected_item is None or self.selected_item.edit_rect is None:
            return None
        nearest, best = None, float("inf")
        for handle in self.selected_item.edit_rect.handles.values():
            if not handle.isVisible():
                continue
            centre = self.mapFromScene(handle.scenePos())
            distance = ((centre.x() - widget_pos.x()) ** 2
                        + (centre.y() - widget_pos.y()) ** 2) ** 0.5
            if distance <= tolerance and distance < best:
                nearest, best = handle, distance
        return nearest

    def begin_drag(self, state: DragState) -> None:
        """
        Method to enter a gesture

        :param state: The gesture to enter
        :return: None
        """
        self.drag = state
        # Otherwise the ViewBox pans the image at the same time as the item moves. Its drag handler
        # re-reads this flag on every event, so disabling it after the press still takes effect
        self.view.setMouseEnabled(x=False, y=False)

    def end_drag(self, commit: bool = True) -> None:
        """
        Method to leave the current gesture

        :param commit: When false the item is put back the way it was, for a cancelled gesture
        :return: None
        """
        state, self.drag = self.drag, None
        self.view.setMouseEnabled(x=True, y=True)
        if state is None:
            return
        item = state.item
        # item.rect() and item.rotation(), NOT item.item_rect and item.angle: every step of a
        # gesture is a preview, and a preview deliberately leaves the item's STORED geometry alone.
        # Reading the stored angle here made every rotation look like a gesture that had not moved,
        # so end_drag discarded it -- a resize survived only because setRect does change item.rect()
        rect = QRectF(item.rect())
        angle = item.rotation()
        # A gesture that ends where it began must not mark the item changed: that would send it
        # through the delete-and-reinsert path in process_changed_items for no reason, and every
        # trip through that path is a chance to strand a focus on a hash that no longer exists
        unchanged = rect == state.start_rect and abs(angle - state.start_angle) < 1e-9
        if not commit or unchanged:
            # reset_item restores item_rect and angle, which the gesture never wrote -- every step
            # of a drag is a PREVIEW, and only this method commits
            item.reset_item()
            self._dialog.update_editing_values(item)
            return
        self.commit_item_geometry(item, rect, angle)

    def start_move_drag(self, event: QMouseEvent) -> bool:
        """
        Method to start dragging the selected item by its body

        :param event: The mouse event that started the gesture
        :return: True if a gesture was started
        """
        item = self.selected_item
        if item is None:
            return False
        # The same shape test that decided the selection, so anything selectable is draggable
        if not item.shape().contains(item.mapFromScene(self.mapToScene(event.pos()))):
            return False
        position = self.image_position(event)
        self.begin_drag(DragState("move", item,
                                  grab_x=position.x() - item.center[0],
                                  grab_y=position.y() - item.center[1]))
        return True

    def start_handle_drag(self, event: QMouseEvent) -> bool:
        """
        Method to start a resize or rotation gesture on a grab point

        :param event: The mouse event that started the gesture
        :return: True if a gesture was started
        """
        handle = self.handle_at(QPointF(event.pos()))
        if handle is None:
            return False
        position = self.image_position(event)
        offset = 0.0
        if handle.role == "rotate":
            # How far the grip lies from the item's current angle. Subtracting it on every step
            # means the gesture turns the item BY the amount the mouse turns, rather than snapping
            # the item's axis onto the cursor the instant the grip is touched
            # handle_at returns None unless BOTH selected_item and its edit_rect are set, and
            # this line is unreachable without a handle -- so the invariant is the caller's, not a
            # condition to test. Asserted rather than guarded: a guard here would silently abandon
            # a rotation the user started, which is the failure mode that made this file's
            # unguarded dereferences worth filing rather than blanket-guarding
            assert self.selected_item is not None, "handle_at returned a handle with no selection"
            centre = self.selected_item.item_rect.center()
            offset = angle_from_vector(centre.x(), centre.y(),
                                       position.x(), position.y()) - self.selected_item.angle
        # The grab point, not the item's centre: a resize follows the DELTA from where the grip was
        # taken hold of, so the box does not jump when the press lands a few pixels off the grip
        self.begin_drag(DragState(handle.role, self.selected_item,
                                  grab_x=position.x(), grab_y=position.y(),
                                  start_vector=offset))
        return True

    def drag_to(self, position: QPointF) -> None:
        """
        Method to advance the gesture in progress to the given image position

        Every step is a PREVIEW: update_data is called with keep_original=True, so the item's stored
        geometry and its `changed` flag are left alone until end_drag commits. That is what lets
        Escape put the item back, and what keeps a plain click from marking a nucleus as edited

        :param position: The cursor position in image coordinates
        :return: None
        """
        state = self.drag
        if state is None or state.item is None:
            return
        item = state.item
        angle = state.start_angle
        if state.role == "move":
            rect = QRectF(position.x() - state.grab_x - state.start_rect.width() / 2,
                          position.y() - state.grab_y - state.start_rect.height() / 2,
                          state.start_rect.width(), state.start_rect.height())
        elif state.role == "rotate":
            # Only the angle moves; the box is untouched, so a rotation cannot change the size
            rect = QRectF(state.start_rect)
            centre = state.start_rect.center()
            angle = angle_from_vector(centre.x(), centre.y(), position.x(), position.y(),
                                      state.start_vector,
                                      snap=15.0 if self.shift_down else 0.0)
        elif state.role in HANDLE_SIGNS:
            # The delta is rotated into the item's OWN frame before the box is resized, because the
            # box is stored unrotated. Dragging the corner of a nucleus turned 40 degrees otherwise
            # stretches it along the image axes rather than along its own
            local_dx, local_dy = to_local(position.x() - state.grab_x,
                                          position.y() - state.grab_y, angle)
            rect = QRectF(*resize_about_anchor(
                (state.start_rect.x(), state.start_rect.y(),
                 state.start_rect.width(), state.start_rect.height()),
                state.role, local_dx, local_dy, angle, lock_aspect=self.shift_down))
        else:
            return
        item.update_data(rect, angle, True)
        self._dialog.write_editing_values(rect.center().x(), rect.center().y(),
                                          rect.width(), rect.height(), angle)

    def move_selected_item_to_position(self) -> None:
        """
        Method to move the selected item to the specified location

        :return: None
        """
        item = self.selected_item
        if item is None or self.mpos is None:
            return
        rect = QRectF(self.mpos.x() - item.width / 2,
                      self.mpos.y() - item.height / 2,
                      item.width, item.height)
        # commit_item_geometry, not update_data + setup_editing: the direct call left the moved item
        # out of temp_items, and process_changed_items then dropped the move on OK
        self.commit_item_geometry(item, rect, item.angle)

    @staticmethod
    def marker_covers(item: "ROIItem", scene_pos: QPointF,
                      pixel: Tuple[float, float]) -> bool:
        """
        Method to test whether the marker DRAWN for an item covers the given position

        `QGraphicsScene.items()` cannot answer this. It hit-tests against
        `QGraphicsEllipseItem.shape()`, which is the ellipse united with its pen stroke -- and Qt
        strokes that outline with `pen.widthF()` in ITEM units, while every marker pen here is
        **cosmetic** (`pg.mkPen` sets it), so it is DRAWN 3 device pixels wide at any zoom. The two
        therefore agree only at 1:1: zoomed in, the hit area keeps its 1.5 image-pixel margin while
        the drawn ring shrinks to a hairline, so the cursor lands well outside a focus and the focus
        still lights up. Reported from real use, 2026-08-24.

        The margin below is the drawn one: half the pen width, in device pixels, converted into item
        units through the current view scale. It follows the zoom, which is what makes the highlight
        agree with what is on screen -- and it keeps a small item aimable when zoomed out, because
        the marker is never drawn thinner than its pen.

        :param item: The item to test
        :param scene_pos: The position to test, in scene coordinates
        :param pixel: The size of one device pixel in item units, as (x, y)
        :return: True if the drawn marker covers the position
        """
        rect = item.rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return False
        # mapFromScene, not arithmetic on the scene position: an item carries a rotation and a
        # transform origin, and the ellipse is axis-aligned only in its OWN frame
        local = item.mapFromScene(scene_pos)
        pen = item.pen()
        half = max(pen.widthF(), 1.0) / 2
        rx = rect.width() / 2 + half * pixel[0]
        ry = rect.height() / 2 + half * pixel[1]
        dx = (local.x() - rect.center().x()) / rx
        dy = (local.y() - rect.center().y()) / ry
        return dx * dx + dy * dy <= 1.0

    def roi_item_at(self, scene_pos: QPointF) -> Optional["ROIItem"]:
        """
        Method to find the topmost item on the active channel whose marker covers a position

        **The one lookup for both hovering and selecting.** They must not diverge: the hover
        highlight exists to show what the next click will hit, and it can only do that if the two
        ask the same question.

        :param scene_pos: The position to test, in scene coordinates
        :return: The item, or None
        """
        # The three mouse handlers that reach this all test `active_channel != "Composite"`
        # first, and the composite view has no entry in active_channels -- so this lookup depends
        # on a caller-side condition rather than on anything visible here. Stated, because the
        # failure would otherwise be a bare KeyError from inside a hit test.
        assert self.active_channel in self.active_channels, (
            f"roi_item_at needs a real channel, got {self.active_channel!r}")
        active_index = self.active_channels[self.active_channel]
        pixel = self.plot_vb.viewPixelSize()
        # scene().items() returns items in DESCENDING stacking order, so the first match is the
        # topmost one -- the one the user can see. Both call sites took items[-1] until 2026-08-24,
        # which is the item furthest BACK: where two markers overlapped, the one picked was the one
        # hidden behind the other. Its inflated hit area is still a useful cheap prefilter
        for item in self.scene().items(scene_pos):
            if not isinstance(item, ROIItem) or item.channel_index != active_index:
                continue
            if self.marker_covers(item, scene_pos, pixel):
                return item
        return None

    def select_item_at_mouse_position(self, event: QMouseEvent) -> None:
        """
        Method to select the clicked item at the mouse position

        :return: None
        """
        item = self.roi_item_at(self.mapToScene(event.pos()))
        if item is not None:
            if self.selected_item:
                self.selected_item.enable_editing(False)
            self.selected_item = item
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
            # Same caller-side condition as roi_item_at, and the consequence is worse here: this
            # index is the channel a NEW roi is written to, so a wrong one stores it against the
            # wrong channel -- the defect the index-space work removed on 2026-09-21
            assert self.active_channel in self.active_channels, (
                f"cannot create an item on channel {self.active_channel!r}")
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

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # An override, where the view previously had none: the cursor position was tracked only
        # through a 45 Hz SignalProxy on the scene, which is fine for a status line and too coarse
        # and too indirect to steer a gesture
        super().mouseMoveEvent(event)
        if self.drag is not None:
            self.drag_to(self.image_position(event))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        if self.drag is not None and event.button() == Qt.LeftButton:
            self.end_drag(commit=True)

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
            # roi_item_at is the same lookup select_item_at_mouse_position uses, so the highlight
            # shows exactly what the next click will hit -- the property this feature exists for
            candidate = self.roi_item_at(scene_pos)
        self.update_cursor(scene_pos, candidate)
        if candidate is self.hovered_item:
            return
        if self.hovered_item is not None:
            self.hovered_item.set_hovered(False)
        self.hovered_item = candidate
        if candidate is not None:
            candidate.set_hovered(True)

    # role -> the cursor that says what dragging it will do. Compass names are in the item's own
    # unrotated frame, so the arrows are only exactly right for an unrotated item; turning a nucleus
    # far enough makes a "vertical" cursor sit on what is now a horizontal edge. Rotating the cursor
    # with the item would need eight more bitmaps to buy very little
    CURSORS = {
        "n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor,
        "e": Qt.SizeHorCursor, "w": Qt.SizeHorCursor,
        "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
        "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
        "rotate": Qt.CrossCursor,
    }

    def update_cursor(self, scene_pos: QPointF, hovered: Optional["ROIItem"]) -> None:
        """
        Method to show what the next press would do

        :param scene_pos: The cursor position, in scene coordinates
        :param hovered: The item under the cursor, if any
        :return: None
        """
        if self.drag is not None:
            return
        handle = self.handle_at(QPointF(self.mapFromScene(scene_pos)))
        if handle is not None:
            self.setCursor(EditorView.CURSORS.get(handle.role, Qt.ArrowCursor))
        elif hovered is not None and hovered is self.selected_item:
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.unsetCursor()

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
        # One map per IMAGE channel, keyed by the database channel index -- self.active_channels is
        # name -> index, which is the space ROIItem.channel_index now uses everywhere. Passing it in
        # is what stopped the maps being indexed by position in `idents`; see the method's docstring
        return self.roi.create_hash_association_maps((self.image.shape[0], self.image.shape[1]),
                                                     self.active_channels)

    def get_unassociated_foci(self) -> List[int]:
        """
        Method to get the now unassociated foci for each nucleus in the self.delete list

        :return: List of focus hashes
        """
        unassociated = []
        # Get all associated foci and add them to list of unassociated foci
        for roi in self.delete:
            # self.roi.ident is the IMAGE md5 (ROIHandler.ident), not a channel name -- ROI.ident
            # is the channel. The image is required: a nucleus hash is not unique across images
            roi_hash = self.requester.get_hashes_of_associated_foci(roi, self.roi.ident)
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
                        hashes = self.requester.get_hashes_of_associated_foci(item.roi_id,
                                                                              self.roi.ident)
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
                    roi = ROI(channel=self._channel_name(item.channel_index),
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
        # add_rois, not rois.extend. Appending to the public list skipped add_roi, which is the
        # only thing that maintains the handler's derived state -- so a nucleus drawn by hand went
        # in without nominating a main channel and without registering its channel in `idents`.
        # That is what made the row 72 crash unrecoverable from inside the editor: the image now had
        # a nucleus, and the save still raised because the handler did not know it did.
        self.roi.add_rois(new_roi)


    def apply_all_changes(self) -> bool:
        """
        Method to apply all made changes and save them to the database

        :return: True if the changes were saved, False if the user cancelled the save
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
        # Guarded, because `main` can legitimately be absent. `ROIHandler.main` is "" until an roi
        # carrying main=True is added, and an image whose analysis found nothing has none -- so
        # `idents.index(self.roi.main)` raised `ValueError: '' is not in list` on OK, after the new
        # roi had already been written. Reported by hand twice, UI row 72, 2026-09-01 and
        # 2026-09-13. The loader now recovers `main` from the channels table, which fixes the real
        # case; this keeps the save from dying on any handler that still has no main channel.
        #
        # With no main channel there is nothing to associate foci WITH, so the association step is
        # skipped rather than faked. Everything else in this method still runs: the drawn roi are
        # written, the deletions are applied, and the image is marked modified
        if self.roi.main in self.active_channels:
            associations = self.create_associations(self.active_channels[self.roi.main], maps,
                                                    unassociated, centers)
        else:
            LOGGER.warning("Image %s declares no main channel -- foci drawn on it cannot be "
                           "associated with a nucleus", self.roi.ident)
            associations = {}
        # Clean unassociated list
        unassociated = [x for x in unassociated if x not in associations.keys()]
        # ASK BEFORE DELETING. A focus that lies inside no nucleus at save time is deleted, which is
        # correct per RW's rule that a focus must always lie within a nucleus -- but until the drag
        # gestures landed, reaching that state took typing coordinates into the spin boxes, and it
        # now takes one slip of the mouse. The deletion was silent.
        #
        # Deliberately NOT a refusal, a snap-back or a check in end_drag: a user who deletes a
        # nucleus by accident must be able to redraw it while its foci wait unassociated, so the
        # forbidden state has to be reachable DURING editing and only rejected at save.
        if unassociated and not self.confirm_focus_deletion(len(unassociated)):
            # Nothing has been committed yet -- every write above is in one open transaction that
            # commit_and_close at the end of this method would close. Rolling back is therefore a
            # true cancel and not a half-applied save, which is the failure this editor has had
            # before: UI row 72 wrote the drawn roi and then raised.
            self.inserter.rollback_and_close()
            LOGGER.info("Save cancelled by the user: %d foci lie outside every nucleus",
                        len(unassociated))
            return False
        self.delete_roi(unassociated)
        # Create new associations
        for focus, nucleus in associations.items():
            self.inserter.associate_focus_with_nucleus(int(nucleus), int(focus))
        # Change image entry to indicate that the image was manually modified
        self.inserter.mark_image_as_modified(self.roi.ident)
        self.inserter.commit_and_close()
        return True

    def confirm_focus_deletion(self, count: int) -> bool:
        """
        Method to ask whether foci that lie inside no nucleus may be deleted

        Split out of apply_all_changes so the save path can be driven without a modal dialog: a
        harness overrides this method rather than having to answer a window that never appears
        under an offscreen platform.

        :param count: The number of foci that would be deleted
        :return: True if the save is to go ahead
        """
        msg = QMessageBox()
        msg.setWindowIcon(Icon.get_icon("LOGO"))
        msg.setIcon(QMessageBox.Warning)
        msg.setStyleSheet(Util.load_stylesheet("messagebox.css"))
        msg.setWindowTitle("Save changes?")
        msg.setText(f"{count} {'focus lies' if count == 1 else 'foci lie'} outside every nucleus.")
        msg.setInformativeText(
            f"Saving deletes {'it' if count == 1 else 'them'}, because a focus has to lie inside a "
            f"nucleus. Cancel to go back and place "
            f"{'it' if count == 1 else 'them'}, or draw the nucleus "
            f"{'it belongs' if count == 1 else 'they belong'} to.")
        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        return msg.exec() == QMessageBox.Save

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
        # item.width/height, NOT item.edit_rect.width/height. The editing rectangle's attributes
        # were assigned once at construction and never refreshed (set_geometry does that now), so
        # roi.width/roi.height held the size the item was CREATED at while the statistics row below
        # stored the edited size from item.width/2 -- the two tables disagreed for every item that
        # was ever resized. Both now read the same source
        # THE STORED CENTRE IS HALF A PIXEL ABOVE AND LEFT OF THE AREA'S CENTROID FOR AN
        # EVEN-SIZED ITEM, and that is accepted rather than a defect (RW, 2026-09-13: "Document and
        # accept"). The area below is rasterised around `centre + 0.5` for an even span -- an even
        # number of pixels cannot be centred on an integer -- while these columns are INTEGER and
        # store the unshifted centre the user actually placed.
        #
        # Measured on two foci drawn by hand, UI row 49: stored (1068, 525) against an area centroid
        # of (1068.50, 525.50), exactly -0.50 on both axes, both times. Detector-written foci of the
        # same size are off by -0.30 to +0.18 in varying directions, which is ordinary rounding
        # against a discrete blob; the editor's offset is systematic because the shift is.
        #
        # Storing the shifted value instead would be off by +0.50 the other way, so nothing is
        # gained without widening the columns. Do not "fix" this by removing the +0.5 in
        # process_changed_items: that is what makes an even requested size draw an even span, which
        # UI row 4 verifies
        roidat = (hash(roi), image_id, False, roi.ident,
                  item.center[0], item.center[1], item.width,
                  item.height, None, "manual", -1, roi.colocalized)
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
        # self.roi.ident, not just the hash: an identical focus in two images shares a hash, so a
        # hash-only delete took the other image's roi with it
        self.inserter.delete_roi_from_database(roihash, self.roi.ident)

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
        # Hoisted out of the loop, which is where it used to sit: `maps[main] != 0` does not depend
        # on c, and recomputing it meant a full pass over the main channel for every OTHER channel.
        # Found while measuring the 2026-09-21 index-space change rather than by reading -- it has
        # been in this shape since the vectorised rewrite
        main_mask = maps[main] != 0
        for c in range(len(maps)):
            if c == main:
                continue
            # EMPTINESS FIRST, and it is not a micro-optimisation. Since 2026-09-21 `maps` holds one
            # array per IMAGE CHANNEL rather than one per channel that carries detections, so a
            # 5-channel image with foci in two channels brings three all-zero maps through here.
            # Building `both` for one of those allocated two boolean arrays and ANDed them --
            # measured at 1.9 ms per empty channel on a 1384x1032 image, against 0.6 ms for this
            # test, and +233 % on the whole method (2.5 ms -> 8.2 ms) before it was added.
            # The comparison is REUSED rather than repeated: testing `maps[c].any()` and then
            # computing `maps[c] != 0` walked the array twice, which cost the populated channels
            # more than it saved on the empty ones. One pass answers both questions, and `any()`
            # on the resulting boolean short-circuits at the first set pixel
            channel_mask = maps[c] != 0
            if not channel_mask.any():
                continue
            both = channel_mask & main_mask
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
            self.inserter.delete_roi_from_database(roi_hash, self.roi.ident)

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
    def encode_new_roi(rr: List[int], cc: List[int], map_: np.ndarray,
                       placeholder: int = -1) -> List[Tuple[int, int, int]]:
        """
        Method to run length encode newly created roi

        Rasterise, then scan -- the same (row, first column, pixel count) convention
        AreaAndROIExtractor.encode_areas produces, so an area written here decodes exactly like one
        written by the detector. The previous implementation got all three parts of that wrong:

        * it seeded `rl = 1` before counting, so EVERY stored run was one pixel too long;
        * it appended a row even when the map had claimed all of that row's pixels, emitting
          `(row, -1, 1)` -- a run starting at column -1 reached the database;
        * it assumed each row is one contiguous span, recording a split row as a single run from its
          first free column to the far side of the gap.

        The run list is what ROI.__hash__ is derived from, so this changes the identity of manual roi
        written from now on. It re-encodes nothing already stored: this encoder runs only on the
        editor's write path.

        :param rr: The row indices
        :param cc: The corresponding column indices
        :param map_: The corresponding map for this roi
        :param placeholder: Stamped into the map for every pixel claimed here, to be swapped for the
        real hash by replace_placeholder once the ROI exists and can be hashed
        :return: The run length encoded area of the given roi
        """
        rr = np.asarray(rr, dtype=np.int64)
        cc = np.asarray(cc, dtype=np.int64)
        if rr.size == 0:
            return []
        # Only the pixels no roi on this channel has claimed yet. The filter and the claim below are
        # not incidental to encoding -- the editor drives overlap handling off this map
        free = map_[rr, cc] == 0
        rr, cc = rr[free], cc[free]
        if rr.size == 0:
            return []
        map_[rr, cc] = placeholder
        # Row-major order, then cut a run wherever the row changes or a column is not the previous
        # one plus one. skimage's ellipse() yields each pixel once, so no de-duplication is needed
        order = np.lexsort((cc, rr))
        rr, cc = rr[order], cc[order]
        starts = np.empty(rr.size, dtype=bool)
        starts[0] = True
        starts[1:] = (rr[1:] != rr[:-1]) | (cc[1:] != cc[:-1] + 1)
        start_indices = np.flatnonzero(starts)
        lengths = np.diff(np.append(start_indices, rr.size))
        return [(int(rr[start]), int(cc[start]), int(length))
                for start, length in zip(start_indices, lengths)]


class ROIDrawer:

    __slots__ = ()
    MARKERS = {
        "invisible": pg.mkPen(color=(0, 0, 0, 0)),
        "handle": pg.mkPen(color="#202020", width=1),
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
    # channels is name -> DATABASE CHANNEL INDEX, not a Sequence of idents. It was the latter
    # until 2026-09-21, and `idents.index(roi.ident)` is what gave a drawn roi a different index
    # space from a hand-drawn one -- the two agree only when every channel carries a detection and
    # the orders match, which stops being true the moment a channel is deactivated for the analysis
    def draw_roi(view: pg.PlotItem, rois: Iterable[ROI],
                 channels: Dict[str, int]) -> List[QGraphicsEllipseItem]:
        """
        Method to populate the given plot with the roi stored in the handler

        :param view: The PlotItem to populate
        :param rois: The ROIHandler
        :param channels: Channel name -> its database/image channel index
        :return: List of all created items
        """
        items = []
        for roi in rois:
            # A stored ROI with no points cannot be drawn -- calculate_dimensions raises for it, and
            # this runs inside the lazy loader's timer, so ONE bad row took the whole editor down
            # with `ValueError: ROI ... does not contain any points!` before the window appeared.
            # 57 such rows exist in the live database across 33 images; the editor has to open on
            # the rest of the image regardless of how they got there.
            #
            # Skipped rather than repaired, and logged rather than swallowed: what is wrong is the
            # stored data, and drawing a placeholder would invent geometry that is not there
            # A backstop only: `EditorView.discard_unusable_roi` removes these before the loader
            # ever runs. It stays because this is a static method any caller can reach, and because
            # `calculate_dimensions` raising inside the loader's timer takes the window down.
            #
            # TEMPORARY message (2026-09-08, at RW's instruction): the channel, not the hash. Every
            # area-less roi in a channel hashes to the same value, so the hash named nothing --
            # naming the row needs the roi identity split described at discard_unusable_roi
            if not roi.is_valid():
                LOGGER.warning("Skipping a roi in channel %s: no points are stored for it. This "
                               "should have been discarded before drawing", roi.ident)
                continue
            ind = channels.get(roi.ident)
            if ind is None:
                # A roi in a channel the editor was not given. Skipped for the same reason the
                # association maps skip it: drawing it under an arbitrary index is what this
                # change exists to prevent
                LOGGER.warning("Skipping a roi in channel %s: it is not one of the editor's "
                               "channels", roi.ident)
                continue
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


class HandleItem(QGraphicsRectItem):
    """
    A grab point drawn on the corners and edge midpoints of an EditingRectangle

    Purely a marker. Like every other item in the editor it is setEnabled(False) and receives no
    mouse events; EditorView hit-tests the handles in DEVICE pixels, because a handle that ignores
    the view transform has no meaningful size in image coordinates
    """

    # The eight resize roles as a fraction of the bounding box, plus the rotate grip. Compass names
    # are used in the UNROTATED frame of the item -- "n" is the top edge of the item's own rect, not
    # whatever is uppermost on screen once the item is turned
    POSITIONS = {
        "nw": (0.0, 0.0), "n": (0.5, 0.0), "ne": (1.0, 0.0),
        "w": (0.0, 0.5), "e": (1.0, 0.5),
        "sw": (0.0, 1.0), "s": (0.5, 1.0), "se": (1.0, 1.0),
    }
    # Roles that change only one dimension, and the pair of box fractions each one holds fixed
    EDGE_ROLES = ("n", "e", "s", "w")
    SIZE = 9
    # How far above the top edge the rotate grip sits, in IMAGE pixels. It cannot be a device
    # offset: a child item's position is expressed in its parent's coordinates, and only the
    # handle's SHAPE ignores the view transform
    ROTATE_DISTANCE = 18

    def __init__(self, role: str):
        half = HandleItem.SIZE / 2
        super().__init__(-half, -half, HandleItem.SIZE, HandleItem.SIZE)
        self.role = role
        # The grab point must stay the same size on screen at every zoom level. Without this a
        # handle is a few thousandths of a pixel across when the image is zoomed out, and a slab
        # covering the nucleus when it is zoomed in
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setPen(ROIDrawer.MARKERS["handle"])
        self.setBrush(QColor("#bdff00") if role != "rotate" else QColor("#00d5ff"))
        self.setEnabled(False)


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
        # Created by create_handles, which is called once the owner knows whether it may be rotated
        self.handles: Dict[str, HandleItem] = {}
        self.initialize()

    def initialize(self):
        """
        Method to initialize this class

        :return:  None
        """
        self.active_pen = pg.mkPen(color="#bdff00", width=3, style=QtCore.Qt.DashLine)
        self.inactive_pen = ROIDrawer.MARKERS["invisible"]
        self.setPen(self.active_pen)

    def create_handles(self, rotatable: bool = False) -> None:
        """
        Method to create the grab points for this rectangle

        :param rotatable: Whether a rotation grip should be created as well. Only nuclei carry one --
        a focus is small and round enough that turning it changes nothing a user can see
        :return: None
        """
        if self.handles:
            return
        roles = list(HandleItem.POSITIONS)
        if rotatable:
            roles.append("rotate")
        for role in roles:
            handle = HandleItem(role)
            handle.setParentItem(self)
            self.handles[role] = handle
        self.layout_handles()

    def layout_handles(self) -> None:
        """
        Method to place the grab points on the current bounding box

        :return: None
        """
        if not self.handles:
            return
        rect = self.rect()
        for role, handle in self.handles.items():
            if role == "rotate":
                handle.setPos(rect.center().x(), rect.top() - HandleItem.ROTATE_DISTANCE)
                continue
            fx, fy = HandleItem.POSITIONS[role]
            handle.setPos(rect.left() + rect.width() * fx, rect.top() + rect.height() * fy)

    def set_geometry(self, rect: QRectF, angle: float) -> None:
        """
        Method to move this rectangle onto the given bounding box

        The four positional attributes are refreshed here rather than only in the constructor. They
        used to be written once and never again while setRect moved the drawn rectangle underneath
        them, so they reported the size this item was CREATED at for the rest of its life --
        EditorView.write_item_to_database read them into the roi table's width/height columns

        :param rect: The new, unrotated bounding box
        :param angle: The angle to apply about the box's center
        :return: None
        """
        self.setRotation(0)
        self.setRect(rect)
        self.setTransformOriginPoint(rect.center())
        self.setRotation(angle)
        self.pos_x = rect.x()
        self.pos_y = rect.y()
        self.width = rect.width()
        self.height = rect.height()
        self.center = rect.center().x(), rect.center().y()
        self.layout_handles()

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
        # The grab points belong to the selection, not to the item -- an unselected roi must not
        # show anything to grab. setVisible rather than a pen swap, because a handle is filled
        for handle in self.handles.values():
            handle.setVisible(enable)


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
        # add_to_view assigns edit_rect, and every caller of this method works on an item that
        # is already on the view -- select_item_at_mouse_position, the drag machinery and
        # reset_item all reach items drawn into the scene. Asserted rather than guarded because
        # skipping the geometry update would leave the editing rectangle behind the item it
        # describes, which looks like a redraw bug rather than a missing precondition
        assert self.edit_rect is not None, "update_data before the item was added to a view"
        self.edit_rect.set_geometry(rect, angle)

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
        Method to set the activity of this item -- its PEN, and nothing else

        **This deliberately does not touch a nucleus's editing rectangle.** `NucleusItem` used to
        override it to call `edit_rect.activate(False)` on the way down, and nothing re-activated it
        on the way up: `is_active(True)` only re-applies the pen. So pressing the Ellipses button off
        and on with a nucleus selected left the selection indicator and all nine grab handles
        invisible while the item was still `selected_item` -- the spin boxes still drove it, a drag
        still moved it, and `handle_at` skipped the now-invisible handles, so resize and rotate
        silently stopped working. Reported by hand twice, UI row 20, 2026-09-01 and 2026-09-13.

        **`enable_editing` owns the editing rectangle**, adds and removes it from the view and
        activates it, and runs on SELECTION -- which is where it belongs, as
        `EditingRectangle.activate`'s own comment says: *"The grab points belong to the selection,
        not to the item."* A display toggle must not reach into it.

        The other caller, `ROIDrawer.change_channel`, is unaffected: `EditorView.show_channel`
        clears `selected_item` and calls `enable_editing(False)` before it runs.

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
        # Before activate(False), which is what hides them again -- a grab point created afterwards
        # would stay visible on an unselected item
        rect.create_handles(isinstance(self, NucleusItem))
        rect.activate(False)
        self.edit_rect = rect

    def enable_editing(self, enable: bool = True) -> None:
        """
        Method to enable the editing of this item

        :param enable: Bool
        :return: None
        """
        # The item must be on the view already -- add_to_view is what creates edit_rect, and the
        # call site at select_item_at_mouse_position says so in its own comment. Asserted rather
        # than guarded: returning quietly would leave an item that looks selected but has no grab
        # points, which is the shape of the 2026-09-13 defect where resize and rotate silently
        # stopped working while the selection looked fine
        assert self.edit_rect is not None, "enable_editing before the item was added to a view"
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
        rect.create_handles(rotatable=True)
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
