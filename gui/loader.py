import time
from typing import Callable, Iterable, Sequence

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QGraphicsView

from core.logging_config import get_logger
from core.roi.ROIHandler import ROIHandler
from gui.Util import create_partial_list

LOGGER = get_logger(__name__)


class Loader(QTimer):

    # Sequence, not Iterable: load_next_batch slices this through create_partial_list and
    # update_progress calls len() on it, so a one-shot iterator raises TypeError on the first batch
    def __init__(self, items: Sequence, batch_size: int = 25,
                 batch_time: int = 100, feedback: Callable = None,
                 processing: Callable = None, autostart: bool = True):
        """
        Base class to implement lazy loading

        :param items: The items to load
        :param batch_size: The number of images to load per batch
        :param batch_time: The time between consecutive loading approaches in milliseconds
        :param feedback: The function to call after loading. Has to accept a list of QStandardItems
        :param processing: The function to process the individual items. Needs to return the items after processing
        :param autostart: Whether to start the timer at the end of this constructor. A subclass that
                          assigns attributes its process_items needs must pass False and call
                          start(self.batch_time) itself once it is fully initialised -- see
                          ROIDrawerTimer
        """
        super().__init__()
        self.items = items
        self.batch_size = batch_size
        self.batch_time = batch_time
        self.feedback = feedback
        self.processing = processing
        # Connect timeout to batch loading method
        self.timeout.connect(self.load_next_batch)
        self.last_index = 0
        # Define variable to indicate the percentage of loaded paths
        self.percentage = 0.0
        self.items_loaded = 0
        self.start_time = time.time()
        # Start timer -- unless a subclass still has work to do. Starting unconditionally here was
        # an ordering hazard: ROIDrawerTimer calls super().__init__() first and assigns self.view
        # afterwards, while its process_items dereferences self.view. That is safe only because Qt
        # timers cannot fire before the event loop runs, so anything that pumped events during
        # construction turned it into an AttributeError
        if autostart:
            self.start(self.batch_time)

    def load_next_batch(self) -> None:
        """
        Function to load the next batch. After loading, the feedback function will be called (will pass an empty list
        to the feedback function to indicate finished loading). Should be overwritten by child classes

        :return: None
        """
        # Get the next batch of items
        items = create_partial_list(self.items, self.last_index, self.batch_size)
        # How many items this batch actually consumed, before any processing that might change the
        # count. The last batch is usually shorter than batch_size
        consumed = len(items)
        # Process items, if a processing function was passed
        if self.processing:
            items = self.process_items(items)
        self.items_loaded += len(items)
        # Check if all items were loaded
        if not items:
            LOGGER.debug("Timer stop after loading %d items, total loading time: %.2f secs",
                         self.items_loaded, time.time() - self.start_time)
            self.stop()
        # Advance by what was consumed, not by a full batch_size: after a short final batch the
        # unconditional += left last_index pointing past the end, so it and self.percentage
        # disagreed with reality until the timer stopped on the following tick
        self.last_index += consumed
        # Update the loading percentage
        self.percentage = self.items_loaded / len(self.items) if self.items else 1
        # Check if a feedback function was given
        if self.feedback:
            # Call the feedback function
            self.feedback(items)

    def process_items(self, items: Iterable):
        """
        Function to process items via the specified processing function

        Can be overwritten to account for additional parameters
        :return: None
        """
        return self.processing(items)


class ROIDrawerTimer(Loader):

    def __init__(self, items: ROIHandler, view: QGraphicsView,
                 batch_size: int = 25, batch_time: int = 50,
                 feedback: Callable = None, processing: Callable = None):
        """
        Class to implement lazy roi drawing.

        :param items: The items to draw
        :param view: Graphicsview to draw the ROI on
        :param batch_size: The number of images to load per batch
        :param batch_time: The time between consecutive loading approaches in milliseconds
        :param feedback: The function to call after loading. Has to accept a list of QStandardItems
        :param processing: The function to process the individual items. Needs to return the items after processing
        """
        # autostart=False, then start below: process_items dereferences self.view, so the timer must
        # not be running until it is assigned
        super().__init__(items, batch_size, batch_time, feedback, processing, autostart=False)
        # Re-declared at the narrower type the base stores it under as a plain Sequence: process_items
        # reads self.items.idents, which only a ROIHandler has
        self.items: ROIHandler = items
        self.view = view
        self.start(self.batch_time)

    def process_items(self, items: ROIHandler):
        """
        Expects self.processing to be ROIDrawer.draw_roi

        :param items: The items to process
        :return: The processed items
        """
        return self.processing(self.view, items, self.items.idents)
