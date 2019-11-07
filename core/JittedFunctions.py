import math
import numpy as np

from typing import Tuple, Union, List
from numba import jit
from numba.typed import List as nList


@jit(nopython=True, cache=True)
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
        center = [int(w/2), int(h/2)]
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
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
    return None
