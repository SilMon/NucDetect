import math
from typing import List, Tuple, Dict, Iterable

import numpy as np
from matplotlib import pyplot as plt
from numba.typed import List as nList
from skimage.draw import disk

from core.roi.AreaAnalysis import imprint_area_into_array
from core.roi.ROI import ROI


def extract_nuclei_from_maps(map_: np.ndarray, channel_name: str) -> List[ROI]:
    """
    Function to extract ROI from the given map/maps
    :param map_: Map to extract ROI from
    :param channel_name: The name of the channel
    :return: The extracted roi
    """
    # Encode focus area
    areas = encode_areas(map_)
    # Extract roi
    nuclei = extract_roi_from_areas(areas, channel_name, True)
    return nuclei


def extract_foci_from_maps(map_: np.ndarray, channel_name: str, main: List[ROI]) -> List[ROI]:
    """
    Function to extract ROI from the given map/maps
    :param map_: Map to extract ROI from
    :param channel_name: The name of the channel
    :param main: Extracted nuclei
    :return: The extracted roi
    """
    # Encode focus area
    areas = encode_areas(map_)
    # Extract roi
    foci = extract_roi_from_areas(areas, channel_name, False)
    # Create hash map for association
    assmap = create_nucleus_hash_map(main, shape=map_.shape)
    associate_roi(foci, assmap)
    return foci

def extract_foci_from_blobs(blobs: List[tuple[int, int, int]],
                            channel_name: str,
                            main: List[ROI],
                            image_shape: Tuple[int, ...]) -> List[ROI]:
    """

    :param blobs: The detected roi as blobs
    :param channel_name: The name of the channel
    :param main: Extracted nuclei
    :param image_shape: The shape of the original image
    :return: The extracted ROI
    """
    foci = [encode_blob(x,channel_name, image_shape) for x in blobs]
    # Create hash map for association
    assmap = create_nucleus_hash_map(main, shape=image_shape)
    associate_roi(foci, assmap)
    return foci


def encode_blob(blob: Tuple[int, int, int],
                channel_name: str,
                image_shape: Tuple[int, ...]) -> ROI:
    """
    Method to encode the area of a ROI

    :param blob: The ROI as blob with y,x and sigma
    :param channel_name: The name of the channel
    :param image_shape: The shape of the original image
    :return: The encoded ROI
    """
    # Get the pixel coordinates of the ROI
    yy, xx = disk((blob[0], blob[1]), blob[2] * math.sqrt(2), shape=image_shape)
    yy, xx = list(yy), list(xx)
    yd = {
        int(y): [] for y in np.unique(yy)
    }
    # Get all associated x coordinates
    [yd[int(y)].append(int(x)) for y, x in zip(yy, xx)]
    # Create the rle area
    area = [(y, min(values), len(values)) for y, values in yd.items()]
    # Create a new ROI
    roi = ROI(channel=channel_name, main=False)
    roi.set_area(area)
    return roi


def encode_areas(area_map: np.ndarray) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Method to extract individual areas from the given binary map.

    :param area_map: The map to extract the areas from
    :return: Dictionary containing the label for each area as well as the associated area given bei image row
    and run length
    """
    height, width = area_map.shape
    # Check if the area_map actually contains areas
    if np.amax(area_map) == 0:
        return {}
    # Define dict for detected areas
    areas = {
        x: [] for x in np.unique(area_map)[1:]
    }
    # Iterate over map
    for y in range(height):
        x = 0
        while x < width:
            # Get label at y:x
            label = area_map[y][x]
            if label != 0:
                col = x
                # run length
                rl = 0
                # Iterate over row
                while area_map[y][x] == label:
                    rl += 1
                    x += 1
                    # Break if x reaches border
                    if x == width:
                        break
                areas[label].append((y, col, rl))
            else:
                x += 1
    return areas


def create_nucleus_hash_map(nuclei: Iterable[ROI], shape: Tuple[int, ...]) -> np.ndarray:
    """
    Function to create a map containing all hashes from each extracted nucleus
    :param nuclei: The nuclei as iterable of ROI
    :param shape: The shape of the original image
    :return: The hash map
    """
    map_ = np.zeros(shape=shape, dtype="int64")
    for nucleus in nuclei:
        imprint_area_into_array(nList(nucleus.area), map_, hash(nucleus))
    return map_


def extract_roi_from_areas(areas: Dict[int, List[Tuple[int, int, int]]], name: str, main: bool) -> List[ROI]:
    """
    Function to extract roi from given areas

    :param areas: The areas to use as base for the ROI
    :param name: The name of the channel
    :param main: Are the defined areas nuclei?
    :return: The ROI
    """
    rois: List[ROI] = []
    for _, rl in areas.items():
        # Define focus roi
        roi = ROI(channel=name, main=main)
        roi.set_area(rl)
        rois.append(roi)
    return rois


def associate_roi(rois: Iterable[ROI], main_map: np.ndarray) -> None:
    """
    Function to create associations between nuclei and found
    :param rois: List of all found ROI
    :param main_map: Hash map of detected nuclei
    :return: None
    """
    for roi in rois:
        # Calculate center of roi
        y, x = (min(main_map.shape[0] - 1,
                    roi.calculate_dimensions()["center_y"]),
                min(main_map.shape[1] - 1,
                    roi.calculate_dimensions()["center_x"]))
        # Look if center corresponds to a nucleus
        if main_map[y][x] and main_map[y][x] > 0 and not roi.main:
            roi.associated = main_map[y][x]
