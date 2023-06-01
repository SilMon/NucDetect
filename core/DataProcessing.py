import math
from typing import Iterable, Tuple, Union, List

import numpy as np
from numba import njit
from numba.typed import List as nList


@njit(cache=True)
def create_lg_lut(m: int) -> List[int]:
    """
    Function to create a little_gauss lookup table for the given m values

    :param m: The max number to calculate the little gauss for
    :return: The created lut
    """
    return [little_gauss(x) for x in range(m + 1)]


@njit(cache=True)
def little_gauss(n: int) -> int:
    """
    Function to calculate the the sum of all numbers between 0 and n

    :param n: The number to calculate the sum for
    :return: The sum
    """
    return (n * n + n) // 2


@njit(cache=True)
def get_region_outlines(binary_map: np.ndarray) -> np.ndarray:
    """
    Function to get the outlines of the given binary map

    :param binary_map: The map to get the outlines from
    :return:The outlines as array
    """
    # Create contour map
    contours = np.zeros(shape=binary_map.shape)
    # Check for alternation of black and white pixels
    for y in range(1, binary_map.shape[0], 1):
        for x in range(1, binary_map.shape[1], 1):
            label = binary_map[y][x]
            # Get previous labels for both axis
            plabel_x = binary_map[y][x - 1]
            plabel_y = binary_map[y - 1][x]
            # Check for alternation
            if label + plabel_x == 1 or label + plabel_y == 1:
                if label:
                    contours[y][x] = 1
                else:
                    if plabel_y:
                        contours[y - 1][x] = 1
                    if plabel_x:
                        contours[y][x] = 1
    return contours


@njit(cache=True)
def automatic_whitebalance(image: np.ndarray, cutoff: float = 0.05) -> np.ndarray:
    """
    Function to perform automatic white balance for an image

    :param image: The image to balance
    :param cutoff: The amount of pixels to go into saturation
    :return: The balanced image
    """
    # Create copy of image
    image = image.copy()
    # Calculate histogram of image
    hist = np.histogram(image, bins=256)
    # Calculate pixel threshold
    thresh = cutoff * image.shape[0] * image.shape[1]
    # Iterate over histogram to get min and max
    amin, amax = 0, 255
    # Counts of pixels
    cmin, cmax = 0, 0
    for ind in range(len(hist[0])):
        cmin += hist[0][ind]
        cmax += hist[0][255 - ind]
        if cmin <= thresh:
            amin += 1
        if cmax <= thresh:
            amax -= 1
    # Calculate balance ratio
    ratio = 255 / (amax - amin)
    # Create iterator for image
    with np.nditer(image, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = ratio * x
    return image


@njit(cache=True)
def eu_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Function to calculate the euclidean distance between two two dimensional points

    :param p1: The first point
    :param p2: The second point
    :return: The distance as float
    """
    return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))


def create_circular_mask(h: Union[int, float], w: Union[int, float],
                         center: Tuple[Union[int, float], Union[int, float]] = None,
                         radius: Union[int, float] = None) -> np.ndarray:
    """
    Function to create a binary, circular mask for image filtering

    :param h: The height of the mask
    :param w: The width of the mask
    :param center: The center of the circle, optional
    :param radius: The radius of the circle, optional
    :return: The created mask as numpy array
    """
    if center is None:
        center = [int(w / 2), int(h / 2)]
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


@njit(cache=True)
def relabel_array(array: np.ndarray) -> None:
    """
    Function to relabel a given binary map

    :param array: The map to relabel
    :return: None
    """
    unique = list(np.unique(array))
    nums = np.arange(len(unique) + 1)
    for y in range(len(array)):
        for x in range(len(array[0])):
            array[y][x] = nums[unique.index(array[y][x])]


@njit(cache=True)
def get_major_axis(points: nList) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Function to get the two points with the highest distance from a list of points

    :param points: The points to check
    :return: The two points with the highest distance
    """
    max_d = 0.0
    p0 = None
    p1 = None
    for r1 in range(len(points)):
        point1 = points[r1]
        for r2 in range(r1, len(points)):
            point2 = points[r2]
            dist = eu_dist(point1, point2)
            if dist > max_d:
                p0 = point1
                p1 = point2
                max_d = dist
    return p0, p1


@njit(cache=True)
def get_minor_axis(points: nList, p0: Tuple[int, int], p1: Tuple[int, int]) -> Tuple[Tuple[int, int],
                                                                                     Tuple[int, int]]:
    """
    Function to get the point which has an angle closest to 90°

    :param points: The points to check
    :param p0: The first point of the major axis
    :param p1: The second point of the major axis
    :return: The determined point
    """
    min_ang = 90
    pmin = None
    # Calculate slope of major axis
    m_maj = (p1[0] - p0[0]) / (p1[1] - p0[1])
    # Calculate center of major axis
    center = int((p0[0] + p1[0]) / 2), int((p0[1] + p1[1]) / 2)
    # Determine minor axis for each nucleus
    for r in range(len(points)):
        c = center
        pm = points[r]
        # Determine slope between point and center
        if c[0] != pm[0] and c[1] != pm[1]:
            m_min = (c[0] - pm[0]) / (c[1] - pm[1])
            a = m_maj - m_min
            b = 1 + m_maj * m_min
            if b != 0:
                angle = math.degrees(math.atan(a / b))
            else:
                angle = 0
        else:
            angle = 0
        # Determine angle between line and major axis
        if angle != 0 and angle / 90 < min_ang:
            pmin = pm
            min_ang = angle / 90
    return center, pmin


@njit(cache=True)
def imprint_data_into_channel(channel: np.ndarray, data: np.ndarray, offset: Union[int, float]) -> None:
    """
    Function to transfer the information stored in data into channel. Works in place

    :param channel: The image channel as ndarray
    :param data: The data to transfer as ndarray
    :param offset: The offset of the data
    :return: None
    """
    for i in range(len(data)):
        for ii in range(len(data[0])):
            if data[i][ii] != 0:
                channel[i + offset[0]][ii + offset[1]] = data[i][ii]
