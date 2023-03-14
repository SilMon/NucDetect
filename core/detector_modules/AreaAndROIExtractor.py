from typing import List, Tuple, Dict, Iterable

import numpy as np
from numba.typed import List as nList

from roi.AreaAnalysis import imprint_area_into_array
from roi.ROI import ROI


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


def encode_areas(area_map: np.ndarray) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Method to extract individual areas from the given binary map.

    :param area_map: The map to extract the areas from
    :return: Dictionary containing the label for each area as well as the associated area given bei image row
    and run length
    """
    height, width = area_map.shape
    # Define dict for detected areas
    areas = {}
    # Iterate over map
    for y in range(height):
        x = 0
        while x < width:
            # Get label at y:x
            label = area_map[y][x]
            if label != 0:
                if areas.get(label) is None:
                    areas[label] = []
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


def create_nucleus_hash_map(nuclei: Iterable[ROI], shape: Tuple[int, int]) -> np.ndarray:
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
        y, x = roi.calculate_dimensions()["center_y"], roi.calculate_dimensions()["center_x"]
        # Look if center corresponds to a nucleus
        if main_map[y][x] and main_map[y][x] > 0 and not roi.main:
            roi.associated = main_map[y][x]
