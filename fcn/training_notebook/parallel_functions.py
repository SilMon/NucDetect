import hashlib
import os
import sqlite3 as sql
from multiprocessing import Pool
from typing import Iterable, Tuple, List

import numpy as np
from IPython.core.display import clear_output
from numba import njit
from skimage import io


def calculate_image_id(path: str) -> str:
    """
    Function to calculate the md5 hash sum of the image described by path
    :param path: The URL of the image
    :return: The md5 hash sum as hex
    """
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def create_training_images_from_folder(path: str, dest: str, subimage_shape: Tuple[int, int]):
    """
    Function to create the training images from a given folder
    :param path: Path containing the images
    :param dest: Folder to save the created images to
    :param subimage_shape: The shape of the sub images to create
    :return: None
    """
    # Check if label path exists
    os.makedirs(dest, exist_ok=True)
    for root, dirs, files in os.walk(path):
        cfiles = check_if_images_already_exist(files, dest, subimage_shape)
        # Prepare files for processing
        data = [(os.path.join(root, x), subimage_shape, dest, False, "") for x in cfiles]
        with Pool(16) as p:
            res = p.starmap(process_image, data)
            for r in res:
                clear_output(wait=True)


def check_if_images_already_exist(files: List[str], dest: str,
                                  subimage_shape: Tuple[int, int],
                                  is_label: bool = False, multiple_channels: bool = True) -> List[str]:
    """
    Function to check if the given images already exist

    :param files: List of files that need to be checked
    :param dest: The folder that should be checked
    :param subimage_shape: The shape of created sub-images
    :param is_label: Should label or training images be checked
    :param multiple_channels: Were the original images split into multiple images?
    :return: List of files that need to be created
    """
    cfiles = []
    for file in files:
        # Create path
        f_name = f"{os.path.splitext(file)[0]}_{'red_' if multiple_channels else ''}{subimage_shape[0]}" \
                 f"-{subimage_shape[1]}_00{'_label' if is_label else''}.png"
        # Check if file already exists
        if not os.path.isfile(os.path.join(dest, f_name)):
            cfiles.append(file)
        else:
            print(f"Files for {file} already exist!")
            clear_output(wait=True)
    return cfiles


def process_image(file_path: str, subimage_shape: Tuple[int, int],
                  dest: str, is_label: bool = False, db_path: str = "",
                  separate_channels: bool = False) -> str:
    """
    Function to process a given image to use for machine learning

    :param file_path: The path leading to the file to load. Is ignored if is_label is True
    :param subimage_shape: The shape of the created sub-images
    :param dest: The folder to save the resulting images in
    :param is_label: If true, the function tries to create a label from the given database
    :param db_path: Path leading to the database
    :return: The md5 hash of the original image
    """
    # Load the image
    if is_label:
        md5 = calculate_image_id(file_path)
        img = get_label_for_image(md5, db_path)
    else:
        img = io.imread(file_path)
    # Get the subimages
    sf_name = os.path.splitext(file_path)[0].split(os.path.sep)[-1]
    subs = extract_subimages(img, subimage_shape)
    for sind, sub in enumerate(subs):
        if not separate_channels:
            # Create name for the given sub image
            name = f"{sf_name}_{subimage_shape[0]}-{subimage_shape[1]}_{sind:02d}{'_label.png' if is_label else '.png'}"
            # Create a file path
            sfpath = os.path.join(dest, name)
            # Check if sub image already exists
            io.imsave(sfpath, sub.astype("uint8"), check_contrast=False)
        else:
            channels = ("red", "green", "blue", "black", "white")
            for channel_index in range(sind.shape[2]):
                # Create name for the given sub image
                name = f"{sf_name}_{channels[channel_index]}_{subimage_shape[0]}-" \
                       f"{subimage_shape[1]}_{sind:02d}{'_label.png' if is_label else '.png'}"
                # Create a file path
                sfpath = os.path.join(dest, name)
                # Check if sub image already exists
                io.imsave(sfpath, sub.astype("uint8"), check_contrast=False)
    return f"Created training images for:\t{sf_name}" if not is_label else f"Created labels for:\t{sf_name}"


def create_label_data_for_images(path: str, label_path: str,
                                 subimage_shape: Tuple[int, int], db_path: str) -> None:
    """
    Function to get the label data for all images at the given path
    :param path: The path to the folder containing the images
    :param label_path: The folder where the label images should be saved
    :param subimage_shape: The shape of the subimages to create
    :param db_path: Path leading to the database
    :return: A list containing lists of all labels for each channel
    """
    # Check if label path exists
    os.makedirs(label_path, exist_ok=True)
    for root, dirs, files in os.walk(path):
        files = check_if_images_already_exist(files, label_path, subimage_shape, True)
        # Prepare files for processing
        data = [(os.path.join(root, x), subimage_shape, label_path, True, db_path) for x in files]
        with Pool(16) as p:
            res = p.starmap(process_image, data)
            for r in res:
                clear_output(wait=True)


@njit
def extract_subimages(img: np.ndarray, subimage_shape: Tuple[int, int]) -> List[np.ndarray]:
    """
    Function to extract subimages from a given image
    :param img: The image to extract the subimages from
    :param subimage_shape: The shape of each subimage
    :return: List of all extracted subimages
    """
    # Get the number of subimages for each axis
    svert, shor = get_number_of_subimages_per_dimension(img.shape, subimage_shape)
    sub_images = []
    for y in range(svert):
        for x in range(shor):
            extract = img[y * subimage_shape[0]: (y + 1) * subimage_shape[0],
                          x * subimage_shape[1]: (x + 1) * subimage_shape[1]]
            tile = np.zeros(shape=(subimage_shape[0], subimage_shape[1], extract.shape[2]))
            tile[0: extract.shape[0], 0: extract.shape[1]] = extract
            sub_images.append(tile)
    return sub_images


@njit
def get_number_of_subimages_per_dimension(img_shape: Tuple[int, int],
                                          sub_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Method to get the number of sub-images per
    :param img_shape: The shape of the image
    :param sub_shape: The shape of the sub-images to extract
    :return: The vertical and horizontal sub-image count
    """
    hcheck = bool(img_shape[0] % sub_shape[0])
    wcheck = bool(img_shape[1] % sub_shape[1])
    # Get the number of tiles
    hcount = img_shape[0] // sub_shape[0] if not hcheck else img_shape[0] // sub_shape[0] + 1
    wcount = img_shape[1] // sub_shape[1] if not wcheck else img_shape[1] // sub_shape[1] + 1
    return hcount, wcount


@njit
def split_channels(img: np.ndarray) -> List[np.ndarray]:
    """
    Function to split the channels
    :param img: The image to split
    :return: List of all available channels
    """
    return [img[..., x] for x in range(img.shape[2])]


def get_label_for_image(img_hash: str, db_path: str) -> np.ndarray:
    """
    Function to get the label data for the given image
    :param img_hash: The md5 hash of the image
    :param dims: The dimensions of the given image
    :param db_path: The path leading to the database, where the label data is stored
    :return: The labels for each channel of the image
    """
    # Connect to database and create cursor
    db = sql.connect(db_path)
    crs = db.cursor()
    # Get the dimensions of the image
    dims = tuple(crs.execute("SELECT height, width FROM images WHERE md5=?", (img_hash,)).fetchall()[0])
    # Load the channel names of the given image
    channel_names = [x[0] for x in crs.execute("SELECT name FROM channels WHERE md5=?", (img_hash,)).fetchall()]
    binmap = np.zeros(shape=(dims[0], dims[1], len(channel_names)))
    # Fetch all associated ROI for each given channel
    for ind, channel in enumerate(channel_names):
        clear_output()
        print(f"Fetching label for: {img_hash} -> {channel}")
        cmap = np.zeros(shape=dims, dtype="uint8")
        # Fetch the roi
        rois = crs.execute("SELECT hash FROM roi WHERE image=? AND channel=?", (img_hash, channel)).fetchall()
        # Fetch each area
        for roi in rois:
            area = crs.execute("SELECT row, column_, width FROM points WHERE hash=?", roi).fetchall()
            imprint_area_into_array(area, cmap, 255)
        binmap[..., ind] = cmap
    return binmap


def imprint_area_into_array(area: Iterable[Tuple[int, int, int]], array: np.ndarray, ident: int) -> None:
    """
    Function to imprint the specified area into the specified area

    :param area: The run length encoced area to imprint
    :param array: The array to imprint into
    :param ident: The identifier to use for the imprint
    :return: None
    """
    # Get normalization factors
    for ar in area:
        array[ar[0], ar[1]: ar[1] + ar[2]] = ident
