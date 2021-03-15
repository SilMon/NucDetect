import datetime
import os
import sqlite3
from os.path import isfile
from typing import List, Tuple, Union

from PyQt5 import QtCore
from PyQt5.QtGui import QStandardItem, QIcon
from PyQt5.QtWidgets import QScrollArea, QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy
from skimage import io, img_as_ubyte
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor

from core.Detector import Detector
from gui import Paths

IMAGE_FORMATS = [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".bmp"
]


def create_scroll_area(layout_type: bool = False,
                       widget_resizable: bool = True) -> Tuple[QScrollArea, Union[QVBoxLayout, QHBoxLayout]]:
    """
    Method to create a scroll area to fill

    :param layout_type: False for QVBoxLayout, True for QHBoxLayout
    :param widget_resizable: True if the central widget should be resizable
    :return:The scroll area and the corresponding layout
    """
    sa = QScrollArea()
    central_widget = QWidget()
    layout = QVBoxLayout() if not layout_type else QHBoxLayout()
    central_widget.setLayout(layout)
    sa.setWidget(central_widget)
    sa.setWidgetResizable(widget_resizable)
    return sa, layout


def create_partial_image_item_list(paths: List[str],
                                   start_index: int,
                                   number: int) -> List[QStandardItem]:
    """
    Function to partially load a list of images. Images between start_index and start_index+number will be loaded.

    :param paths: The paths of the images
    :param start_index: The start index
    :param number: The number of images to load
    :return: The loaded images as QStandardItems
    """
    # Get the max available index
    max_ind = min(start_index + number, len(paths))
    # Extract list of paths to load
    part_paths = paths[start_index:max_ind]
    return create_image_item_list_from(part_paths, indicate_progress=False, sort_items=False)


def create_image_item_list_from(paths: List[str],
                                indicate_progress: bool = False,
                                sort_items: bool = True) -> List[QStandardItem]:
    """
    Function to create a list of QStandardItems from image paths. Useful for display in ListViews

    :param paths: A list containing image paths
    :param indicate_progress: If true, loading progress will be printed to the console
    :param sort_items: If true, items will be sorted
    :return: A list of the created items
    """
    items = []
    if indicate_progress:
        print(f"{len(paths)} to load")
        ind = 1
    if sort_items:
        paths = sorted(paths, key=os.path.basename)
    with ThreadPoolExecutor(max_workers=None) as e:
        res = e.map(create_list_item, paths)
        for q in res:
            items.append(q)
            if indicate_progress:
                print(f"Loading: {ind}/{len(paths)}")
                ind += 1
    return items


def create_list_item(path: str) -> QStandardItem:
    """
    Method to create an image list item

    :param path: The path of the image
    :return: The created item
    """
    temp = os.path.split(path)
    folder = temp[0].split(sep=os.sep)[-1]
    file = temp[1]
    if os.path.splitext(file)[1] in IMAGE_FORMATS:
        d = Detector.get_image_data(path)
        date = d["datetime"]
        if isinstance(date, datetime.datetime):
            t = (date.strftime("%d.%m.%Y"), date.strftime("%H:%M:%S"))
        else:
            t = date.decode("ascii").split(" ")
            temp = t[0].split(":")
            t[0] = f"{temp[2]}.{temp[1]}.{temp[0]}"
        key = Detector.calculate_image_id(path)
        item = QStandardItem()
        item_text = f"Name: {file}\nFolder: {folder}\nDate: {t[0]}\nTime: {t[1]}"
        item.setText(item_text)
        item.setTextAlignment(QtCore.Qt.AlignLeft)
        icon = QIcon()
        icon.addFile(
            create_thumbnail(path)
        )
        item.setIcon(icon)
        item.setData({
            "key": key,
            "path": path,
            "analysed": check_if_image_was_analysed(key),
            "file_name": file,
            "folder": folder,
            "date": t[0],
            "time": t[1],
            "icon": icon
        })
        return item


def create_thumbnail(image_path: str, size: Tuple = (75, 75)) -> str:
    """
    Function to create a thumbnail from an image

    :param image_path: The path leading to the image
    :param size: The size of the thumbnail
    :return: The path leading to the thumbnail
    """
    # Calculate the hash of the image
    ident = Detector.calculate_image_id(image_path)
    # Create path to thumbnail
    thumb_path = os.path.join(Paths.thumb_path, f"{ident}.jpg")
    # Check if the thumbnail already exists
    if isfile(thumb_path):
        return thumb_path
    # Load image as numpy array
    img = io.imread(image_path)
    # Get ratio between height and width
    ratio = img.shape[0] / img.shape[1]
    if ratio >= 1:
        new_shape = size[0], int(size[1] / ratio)
    else:
        new_shape = int(size[0] * ratio), size[1]
    # Scale image
    img = resize(img, new_shape)
    # Save the image
    io.imsave(thumb_path, img_as_ubyte(img))
    return thumb_path


def check_if_image_was_analysed(md5: str) -> bool:
    """
    Function to check if an image was already analysed

    :param md5: The md5 hash of the image
    :return: Boolean to indicate if the image was analysed
    """
    connection = sqlite3.connect(Paths.database)
    cursor = connection.cursor()
    analysed = cursor.execute(
        "SELECT analysed FROM images WHERE md5=?",
        (md5, )
    ).fetchall()
    if analysed:
        return analysed[0][0]
    else:
        return False


