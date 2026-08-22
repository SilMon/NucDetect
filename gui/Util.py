# pyright: reportAttributeAccessIssue=false
# ^ PyQt5's stubs nest enum members inside their enum class (Qt.ItemDataRole.DisplayRole)
# while the C++ runtime also exposes them flat on Qt, which is what this file uses. The
# code is correct PyQt5 and a rewrite to the scoped form was declined -- a PyQt6 migration
# is not planned. Suppressed at FILE level only because every hit of this
# rule here is that stub artefact; measured, not assumed. Re-check with the rule enabled
# before adding attribute access to a non-Qt object in this file.
import datetime
import os
import sqlite3
import threading
from contextlib import closing
from os.path import isfile
from typing import List, Optional, Sequence, Tuple, Union

from PyQt5 import QtCore
from PyQt5.QtGui import QStandardItem, QIcon
from PyQt5.QtWidgets import QScrollArea, QVBoxLayout, QHBoxLayout, QWidget
import numpy as np
from skimage import io, img_as_ubyte
from skimage.transform import resize, downscale_local_mean

from gui.definitions.icons import Color
from core.detector_modules.ImageLoader import ImageLoader, dtype_max
from core.logging_config import get_logger
from gui import Paths

LOGGER = get_logger(__name__)

IMAGE_FORMATS = [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".bmp"
]

# If set, a ui operation running off the GUI thread raises instead of only being logged. Meant for
# tests and debug runs -- in production a wrong thread should not turn into a hard crash by itself
STRICT_THREAD_AFFINITY = os.environ.get("NUCDETECT_STRICT_THREAD_AFFINITY", "") == "1"


def assert_main_thread(operation: str, strict: bool = None, logger=None) -> None:
    """
    Function to verify that a ui operation is running on the GUI thread

    Touching a widget or a model from a worker thread does not fail immediately -- it corrupts
    Qt's internal state and surfaces later as an unreproducible freeze or crash. This turns that
    into a loud, deterministic failure at the point of the violation.

    It lives in Util because the editor needs it too, and a dialog module cannot import
    NucDetectAppQT to get at its copy: that would pull TensorFlow in AFTER PyQt5, which is the
    documented DLL-load failure.

    :param operation: Name of the operation, used in the message
    :param strict: Overrides the module flag. NucDetect passes its own so that flipping the flag
    on that module keeps working
    :param logger: Optional; the logger to report on. Defaults to this module's
    :return: None
    :raises RuntimeError: If called off the GUI thread and strict checking is enabled
    """
    if QtCore.QThread.currentThread() is QtCore.QCoreApplication.instance().thread():
        return
    msg = (f"{operation} was called from thread "
           f"'{QtCore.QThread.currentThread().objectName() or threading.current_thread().name}'"
           f" instead of the GUI thread -- route it through a signal")
    (logger or LOGGER).critical(msg)
    if STRICT_THREAD_AFFINITY if strict is None else strict:
        raise RuntimeError(msg)


#: Placeholder every `url()` in a stylesheet is written against. `load_stylesheet` replaces it with
#: the absolute path of the css directory, because Qt resolves a relative `url()` against the
#: current working directory rather than against the stylesheet's own location -- see the header
#: comment in `main.css` for why this is a placeholder and not a silent rewrite of relative paths.
CSS_DIR_PLACEHOLDER = "@@CSS_DIR@@"


def load_stylesheet(name: str) -> str:
    """
    Method to read a stylesheet from the css directory

    Replaces the bare `open(...).read()` that stood at twelve call sites: no context manager, so on
    Windows the handle survived until the GC ran, and no encoding, so the file was read in the
    platform codepage

    Also substitutes `CSS_DIR_PLACEHOLDER`, which is what makes the images a stylesheet references
    resolve independently of the working directory. **This is the only supported way to read a
    stylesheet** -- a bare `open().read()` returns the placeholder verbatim and the images silently
    do not draw

    :param name: The file name of the stylesheet, e.g. "main.css"
    :return: The stylesheet, or an empty string if it could not be read
    """
    path = os.path.join(Paths.css_dir, name)
    try:
        with open(path, "r", encoding="utf-8") as stylesheet:
            sheet = stylesheet.read()
    except OSError:
        # Styling is cosmetic -- an unreadable css file must not take a dialog down with it
        LOGGER.exception(f"Stylesheet {name} could not be read from {path} -- widget shown unstyled")
        return ""
    # Forward slashes even on Windows: Qt parses the url() itself and a backslash is an escape
    # character there, so a native path would be mangled rather than resolved
    return sheet.replace(CSS_DIR_PLACEHOLDER, Paths.css_dir.replace(os.sep, "/"))


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
    part_paths = create_partial_list(paths, start_index, number)
    return create_image_item_list_from(part_paths, indicate_progress=False, sort_items=False)


def create_image_item_list_from(paths: List[str],
                                indicate_progress: bool = False,
                                sort_items: bool = True) -> List[QStandardItem]:
    """
    Function to create a list of QStandardItems from image paths. Useful for display in ListViews

    :param paths: A list containing image paths
    :param indicate_progress: If true, loading progress will be logged
    :param sort_items: If true, items will be sorted
    :return: A list of the created items
    """
    items = []
    if indicate_progress:
        LOGGER.debug("%d to load", len(paths))
        ind = 1
    if sort_items:
        paths = sorted(paths, key=os.path.basename)
    for path in paths:
        item = create_list_item(path)
        # create_list_item returns None for anything that is not a recognised image format. Skip
        # rather than appending, so a non-image path in the folder cannot put a None into a list of
        # QStandardItems that is handed straight to Qt
        if item is not None:
            items.append(item)
        if indicate_progress:
            LOGGER.debug("Loading: %d/%d", ind, len(paths))
            ind += 1
    return items


def create_partial_list(items: Sequence,
                        start_index: int,
                        number: int) -> Sequence:
    """
    Method to create a partial item list

    Sequence, not Iterable: this calls len() and slices, so a genuine iterator raises TypeError.

    :param items: The item list
    :param start_index:  The start index
    :param number: The length of the partial list
    :return: The partial list
    """
    # Get the max available index
    max_ind = min(start_index + number, len(items))
    # Extract list of paths to load
    part_items = items[start_index:max_ind]
    return part_items


def create_list_item(path: str) -> Optional[QStandardItem]:
    """
    Method to create an image list item

    :param path: The path of the image
    :return: The created item, or None if the path is not a recognised image format. Callers must
             handle the None -- create_image_item_list_from skips it
    """
    temp = os.path.split(path)
    folder = temp[0].split(sep=os.sep)[-1]
    file = temp[1]
    if os.path.splitext(file)[1] in IMAGE_FORMATS:
        # Decode and hash ONCE, then hand both to everything below. This function used to hash the
        # file twice (here and inside create_thumbnail) and decode it twice (here and again on a
        # thumbnail cache miss), for every row of the image list
        img = ImageLoader.load_image(path)
        key = ImageLoader.calculate_image_id(path)
        d = ImageLoader.get_image_data(path, img=img)
        date = d["datetime"]
        # No else branch. ImageLoader.get_image_data sets "datetime" to a datetime.datetime in both
        # of its branches, so the byte-string case that used to live here -- date.decode("ascii"),
        # a leftover from when the raw EXIF DateTime tag was used -- was unreachable. It would also
        # have read t[1] below without checking that the split produced two parts. Restore it
        # deliberately if the EXIF path is ever brought back; the commented-out original is still
        # in ImageLoader.get_image_data
        t = (date.strftime("%d.%m.%Y"), date.strftime("%H:%M:%S"))
        item = QStandardItem()
        item_text = f"Name: {file}\nFolder: {folder}\nDate: {t[0]}\nTime: {t[1]}"
        item.setText(item_text)
        item.setTextAlignment(QtCore.Qt.AlignLeft)
        icon = QIcon()
        icon.addFile(
            create_thumbnail(path, ident=key, img=img)
        )
        item.setIcon(icon)
        analysed, modified = check_if_image_was_analysed_and_modified(key)
        y_scale, x_scale = 1, 1
        if analysed:
            if modified:
                item.setBackground(Color.ITEM_MODIFIED)
            else:
                item.setBackground(Color.ITEM_ANALYSED)
            y_scale, x_scale = get_image_scale(key)
        item.setData({
            "key": key,
            "path": path,
            "analysed": analysed,
            "modified": modified,
            "file_name": file,
            "folder": folder,
            "date": t[0],
            "time": t[1],
            "icon": icon,
            "x_scale": x_scale,
            "y_scale": y_scale
        })
        return item


def check_for_thumbnails(paths: List[str]) -> None:
    """
    Function to check if the given images already have a thumbnail created.
    If not, the thumbnails will be created

    **NOTHING CALLS THIS.** Grepped 2026-08-22: no caller in `core/`, `gui/` or `fcn/`. It is left
    in place rather than deleted because deleting it is a decision, not a clean-up.

    It carries the same hole that crashed the startup walk -- it thumbnails whatever it is given,
    with no format filter -- so **it must not be wired up without one**. The filter now lives in the
    two live callers rather than here, because they are the ones that walk directories.

    :param paths: List of image paths
    :return:None
    """
    for path in paths:
        create_thumbnail(path)


def create_thumbnail(image_path: str, size: Tuple = (75, 75),
                     ident: Optional[str] = None,
                     img: Optional[np.ndarray] = None) -> str:
    """
    Function to create a thumbnail from an image

    `ident` and `img` exist so a caller that has already hashed or decoded this file does not pay
    for it twice. `create_list_item` does both, and used to hash the file twice and decode it twice
    per row. **Both must belong to `image_path`** -- passing another file's hash writes the
    thumbnail under the wrong name, and passing another file's pixels draws the wrong picture.

    :param image_path: The path leading to the image
    :param size: The size of the thumbnail
    :param ident: The md5 of the image, if the caller already has it. Computed when omitted
    :param img: The already-decoded pixels, if the caller has them. Read when omitted -- and NOT
        read at all when the thumbnail is already cached, which is the common case
    :return: The path leading to the thumbnail
    """
    # Calculate the hash of the image
    if ident is None:
        ident = ImageLoader.calculate_image_id(image_path)
    # PNG, not JPEG. JPEG cannot store more than three channels, so a 4-channel fluorescence image
    # failed here. The thumbnails are a private cache in ~/NucDetect/thumbnails and nothing outside
    # this function reads them by extension, so the container is free to change; any .jpg left from
    # an earlier version is simply never looked up again and can be deleted by hand
    thumb_path = os.path.join(Paths.thumb_path, f"{ident}.png")
    # Check if the thumbnail already exists
    if isfile(thumb_path):
        return thumb_path
    # Load image as numpy array. Deliberately AFTER the cache check above, so a hit costs no decode.
    #
    # Through ImageLoader.load_image rather than io.imread directly, so an unsupported file raises
    # the documented ValueError naming the extension instead of imageio's "Could not find a backend
    # to open ... with iomode 'r'". Callers are expected to filter -- and as of 2026-08-22 they do,
    # after a .txt in the user's image folder killed the application during the splash screen --
    # but a utility that reads whatever it is handed should say clearly what it refused
    if img is None:
        img = ImageLoader.load_image(image_path)
    # Get ratio between height and width
    ratio = img.shape[0] / img.shape[1]
    if ratio >= 1:
        new_shape = size[0], int(size[1] / ratio)
    else:
        new_shape = int(size[0] * ratio), size[1]
    # Box-filter down to roughly TWICE the target first, then let `resize` do the final factor of
    # two with its proper anti-aliasing.
    #
    # `resize` alone was 86 % of this function -- 134.7 ms of 157 ms per image, measured over ten
    # real images -- because for a large downscale it runs spline interpolation with anti-aliasing
    # across the whole source. A cheap box filter removes almost all of that work.
    #
    # Stopping at 2x rather than at the target is what keeps the picture faithful, and it was
    # measured rather than assumed. On demo.tif, against the shipped output:
    #
    #     box-filter all the way to the target : 20.2 ms, mean abs difference 2.51/255, worst 144
    #     box-filter to 2x the target          : 23.4 ms, mean abs difference 1.31/255, worst  47
    #     resize(anti_aliasing=False)          : 12.2 ms, mean abs difference 3.40/255, worst 184
    #
    # Three milliseconds halves the error, because the final resize still has a real downscale to
    # anti-alias instead of an almost-identity one. (The filed finding measured 0.25/255 on this
    # project's own fluorescence images, which are mostly dark background; demo.tif is busier, so
    # its figure is higher for the same code.)
    #
    # PIL would be faster still (2.6 ms) and is already a dependency, but it CANNOT represent this
    # project's images: Image.fromarray raises on a 5-channel array, and 4- and 5-channel
    # fluorescence images are exactly what this thumbnails. Taking the first three channels would
    # change what is shown. Do not "simplify" to PIL.
    factors = tuple(max(1, dim // (2 * target)) for dim, target in zip(img.shape[:2], new_shape))
    if factors != (1, 1):
        # The factor tuple must cover every axis, so a colour image needs a trailing 1 -- otherwise
        # the channels would be averaged together into greyscale
        factors = factors + (1,) * (img.ndim - 2)
        # Normalised to 0..1 rather than cast back to the source dtype. downscale_local_mean returns
        # FLOAT in the input's own value range (0..255 for uint8), whereas `resize` normalises an
        # integer input and img_as_ubyte below requires 0..1 -- handing the raw float on raises
        # "Images of type float must be between -1 and 1". Measured identical either way (2.515 vs
        # 2.514), so this is for correctness, not fidelity
        img = downscale_local_mean(img, factors) / dtype_max(img.dtype)
        img = np.clip(img, 0.0, 1.0)
    # Scale image
    img = resize(img, new_shape)
    # Save the image
    io.imsave(thumb_path, img_as_ubyte(img), check_contrast=False)
    return thumb_path


def check_if_image_was_analysed_and_modified(md5: str) -> Tuple[bool, bool]:
    """
    Function to check if an image was already analysed

    :param md5: The md5 hash of the image
    :return: Boolean to indicate if the image was analysed
    """
    # closing(), not `with sqlite3.connect(...)`. The connection object's context manager wraps a
    # TRANSACTION -- it commits or rolls back on exit and leaves the connection itself open -- so
    # the obvious spelling would not have closed anything. This function runs once per row of the
    # image list, so a folder of 500 images leaked 500 connections, and on Windows 500 open file
    # handles on the database, until the garbage collector happened to finalise them.
    with closing(sqlite3.connect(Paths.database)) as connection:
        cursor = connection.cursor()
        analysed = cursor.execute(
            "SELECT analysed FROM images WHERE md5=?",
            (md5, )
        ).fetchall()
        modified = cursor.execute(
            "SELECT modified FROM images WHERE md5=?",
            (md5, )
        ).fetchall()
    if analysed:
        analysed = analysed[0][0]
    else:
        analysed = False
    if modified:
        modified = modified[0][0]
    else:
        modified = False
    return analysed, modified


def get_image_scale(md5: str) -> Tuple[float, float]:
    """
    Function to get the saved scale of this image

    :param md5: The md5 hash of the image
    :return: Tuple containing the scale for the y- and x-axis. (1, 1) if the image has no row,
             matching the neutral scale create_list_item starts from
    """
    # See check_if_image_was_analysed_and_modified above for why this is closing() and not a plain
    # `with sqlite3.connect(...)`
    with closing(sqlite3.connect(Paths.database)) as connection:
        cursor = connection.cursor()
        x_scale = cursor.execute(
            "SELECT x_res FROM images WHERE md5=?",
            (md5,)
        ).fetchall()
        y_scale = cursor.execute(
            "SELECT y_res FROM images WHERE md5=?",
            (md5,)
        ).fetchall()
    # Guarded like its neighbour above, which handles the missing row and this one did not: the
    # unguarded [0][0] was an IndexError for an image with no row. Masked today because the only
    # caller reaches it when `analysed` is truthy, which implies the row exists -- the asymmetry
    # between two functions reading the same table is the defect
    return (y_scale[0][0] if y_scale else 1,
            x_scale[0][0] if x_scale else 1)