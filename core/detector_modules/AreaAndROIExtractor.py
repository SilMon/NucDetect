import math
from typing import List, Tuple, Dict, Iterable

import numpy as np
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
    associate_roi(foci, assmap, main)
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
    associate_roi(foci, assmap, main)
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


def get_overlapped_nuclei(area: Iterable[Tuple[int, int, int]], main_map: np.ndarray) -> set:
    """
    Function to get the nuclei a roi shares pixels with

    :param area: The area of the roi, as (row, first column, length) runs
    :param main_map: Hash map of detected nuclei
    :return: The hashes of every nucleus the area touches
    """
    overlapped = set()
    for row, column, length in area:
        if not 0 <= row < main_map.shape[0]:
            continue
        segment = main_map[row, max(0, column): column + length]
        overlapped.update(int(x) for x in np.unique(segment) if x)
    return overlapped


def get_nearest_nucleus(focus_center: Tuple[int, int],
                        candidates: Dict[int, Tuple[int, int]]) -> int:
    """
    Function to pick which of the overlapped nuclei a focus belongs to

    **The rule, set by RW on 2026-08-17: overlap decides WHETHER a focus is associated, the
    distance between the two centres decides WITH WHICH.** Both association paths go through this
    function -- the detector's associate_roi and the editor's create_associations -- because a
    focus that changes owner depending on which path last wrote it is the defect the rule exists to
    prevent. They used to disagree: the detector asked whether the focus's centre PIXEL landed in a
    nucleus, and the editor took whichever nucleus the last scanned pixel belonged to

    Compares squared distances, which orders identically to the distance itself and needs no root.
    A tie goes to whichever candidate was found first, which is the scan order of the area

    :param focus_center: The center of the focus, as (y, x)
    :param candidates: The nuclei the focus overlaps, as {hash: (y, x)}
    :return: The hash of the nearest nucleus, or 0 if the focus overlaps none
    """
    if not candidates:
        return 0
    return min(candidates.items(),
               key=lambda item: (item[1][0] - focus_center[0]) ** 2
               + (item[1][1] - focus_center[1]) ** 2)[0]


def associate_roi(rois: Iterable[ROI], main_map: np.ndarray, nuclei: Iterable[ROI]) -> None:
    """
    Function to create associations between nuclei and found roi

    :param rois: List of all found ROI
    :param main_map: Hash map of detected nuclei
    :param nuclei: The nuclei the map was built from, needed for their centers
    :return: None
    """
    centers = {}
    for nucleus in nuclei:
        dims = nucleus.calculate_dimensions()
        centers[hash(nucleus)] = (dims["center_y"], dims["center_x"])
    for roi in rois:
        if roi.main:
            continue
        dims = roi.calculate_dimensions()
        # Overlap is the gate. Testing the centre PIXEL instead, as this did until 2026-08-17,
        # refused every focus whose centre happens to fall just outside the nucleus it lies on --
        # and could not see the case where a focus spans two nuclei at all
        overlapped = get_overlapped_nuclei(roi.area, main_map)
        nearest = get_nearest_nucleus((dims["center_y"], dims["center_x"]),
                                      {h: c for h, c in centers.items() if h in overlapped})
        if nearest:
            roi.associated = nearest
