import os
from typing import List, Callable, Iterable

from PyQt5.QtCore import QTimer

from gui.Util import create_partial_image_item_list
import time


class Loader(QTimer):

    def __init__(self, items: Iterable, batch_size: int, batch_time: int, feedback: Callable):
        """
        Base class to implement lazy loading

        :param items: The items to load
        :param batch_size: The number of images to load per batch
        :param batch_time: The time between consecutive loading approaches in milliseconds
        :param feedback: The function to call after loading. Has to accept a list of QStandardItems
        """
        super().__init__()
        self.items = items
        self.batch_size = batch_size
        self.batch_time = batch_time
        self.feedback = feedback
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
        return


class ImageLoader(Loader):

    def __init__(self, paths: List[str], batch_size: int = 25,
                 batch_time: int = 1000, feedback: Callable = None):
        """
        Class to implement lazy image loading.

        :param paths: The paths to load
        :param batch_size: The number of images to load per batch
        :param batch_time: The time between consecutive loading approaches in milliseconds
        :param feedback: The function to call after loading. Has to accept a list of QStandardItems
        """
        super().__init__(sorted(paths, key=os.path.basename), batch_size, batch_time, feedback)

    def load_next_batch(self) -> None:
        """
        Function to load the next batch. After loading, the feedback function will be called (will pass an empty list
        to the feedback function to indicate finished loading)

        :return: None
        """
        # Load the next batch of images
        items = create_partial_image_item_list(self.items, self.last_index, self.batch_size)
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


