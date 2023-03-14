import time
from typing import Callable, Iterable

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QGraphicsView

from core.roi.ROIHandler import ROIHandler
from gui.Util import create_partial_list


class Loader(QTimer):

    def __init__(self, items: Iterable, batch_size: int = 25,
                 batch_time: int = 100, feedback: Callable = None,
                 processing: Callable = None):
        """
        Base class to implement lazy loading

        :param items: The items to load
        :param batch_size: The number of images to load per batch
        :param batch_time: The time between consecutive loading approaches in milliseconds
        :param feedback: The function to call after loading. Has to accept a list of QStandardItems
        :param processing: The function to process the individual items. Needs to return the items after processing
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
        # Start timer
        self.start(self.batch_time)

    def load_next_batch(self) -> None:
        """
        Function to load the next batch. After loading, the feedback function will be called (will pass an empty list
        to the feedback function to indicate finished loading). Should be overwritten by child classes

        :return: None
        """
        # Get the next batch of items
        items = create_partial_list(self.items, self.last_index, self.batch_size)
        # Process items, if a processing function was passed
        if self.processing:
            items = self.process_items(items)
        self.items_loaded += len(items)
        # Check if all items were loaded
        if not items:
            print(f"\nTimer stop after loading {self.items_loaded} items")
            print(f"Total loading time: {time.time() - self.start_time:.2f}\n")
            self.stop()
        # Update the last index
        self.last_index += self.batch_size
        # Update the loading percentage
        self.percentage = self.items_loaded / (len(self.items))
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
        super().__init__(items, batch_size, batch_time, feedback, processing)
        self.view = view

    def process_items(self, items: ROIHandler):
        """
        Expects self.processing to be ROIDrawer.draw_roi

        :param items: The items to process
        :return: The processed items
        """
        return self.processing(self.view, items, self.items.idents)
