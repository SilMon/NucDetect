import math
from typing import Tuple, Union, List

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
    Function to calculate the sum of all numbers between 0 and n

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


def automatic_colorbalance(image: np.ndarray, cutoff: float = 0.05) -> np.ndarray:
    """
    Function to perform automatic white balance for an image

    :param image: The image to balance
    :param cutoff: The amount of pixels to go into saturation
    :return: The balanced image
    """
    image = image.copy()
    if len(image.shape) > 2:
        for c in range(image.shape[2]):
            image[..., c] = automatic_whitebalance(image[..., c], cutoff)
    else:
        image = automatic_whitebalance(image, cutoff)
    return image


def automatic_whitebalance(image: np.ndarray, cutoff: float = 0.05) -> np.ndarray:
    """
    Function to perform automatic white balance for an image

    :param image: The image to balance
    :param cutoff: The amount of pixels to go into saturation
    :return: The balanced image
    """
    # Create copy of image
    image = image.copy()
    if "float" not in str(image.dtype):
        image = image.copy()
    else:
        # Convert image to uint TODO
        image = (((image - np.amin(image)) / np.amax(image)) * 255).astype("uint8")
    imgmin, imgmax = np.iinfo(image.dtype).min, np.iinfo(image.dtype).max
    amin, amax = imgmin, imgmax
    # Calculate histogram of image
    hist = np.histogram(image, bins=imgmax + 1)
    # Suppress shadows
    shadow_index = 0
    index_number = 0
    pixel_number = 0.3 * image.shape[0] * image.shape[1]
    for ind, val in enumerate(hist[0]):
        index_number += val
        if index_number > pixel_number:
            shadow_index = ind + 1
            break
    cut_hist = hist[0][shadow_index:]
    total_pixels = np.sum(cut_hist)
    # Calculate pixel threshold
    thresh = cutoff * total_pixels
    # Counts of pixels
    cmin, cmax = 0, 0
    for ind in range(len(cut_hist)):
        cmin += cut_hist[ind]
        cmax += cut_hist[len(cut_hist) - 1 - ind]
        if cmin <= thresh:
            amin += 1
        if cmax <= thresh:
            amax -= 1
    # Calculate balance ratio
    ratio = (imgmax - imgmin) / (amax - amin)
    # Create a lookup table for pixel values
    lut = []
    for val in range(imgmax):
        lut.append(imgmin + (val - amin) * ratio)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x] <= amin:
                image[y][x] = imgmin
            elif image[y][x] >= amax:
                image[y][x] = imgmax
            else:
                image[y][x] = lut[image[y][x]]
    return image


@njit(cache=True)
def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Function to calculate the Euclidean distance between two two-dimensional points

    :param p1: The first point
    :param p2: The second point
    :return: The distance as float
    """
    return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))


@njit(cache=True)
def get_circle_area(d: int) -> float:
    """
    Function to calculate the area of a circle

    :param d: The diameter of the circle
    :return: The area of the circle as float
    """
    return math.pi * (d / 2) ** 2


@njit(cache=True)
def calculate_overlap_between_two_circles(c1: Tuple[int, int, int, int], c2: Tuple[int, int, int, int]) -> float:
    """
    Function to calculate the overlap between two circles
    Adapted from:   https://stackoverflow.com/questions/4247889/area-of-intersection-between-two-circles
                    http://mathworld.wolfram.com/Circle-CircleIntersection.html

    :param c1: The center, diameter and identifier of the first circle
    :param c2: The center, diameter and identifier of the second circle
    :return: The overlapping area between both circles, as float
    """
    center_1 = (c1[0], c1[1])
    center_2 = (c2[0], c2[1])
    radius_1 = c1[2] / 2
    radius_2 = c2[2] / 2
    dist = euclidean_distance(center_1, center_2)
    # Check if both circles overlap at all
    if dist > radius_1 + radius_2:
        return 0
    # Check if one circle is inside the other
    if dist <= abs(radius_1 - radius_2):
        # Check for circle 2
        if radius_1 >= radius_2:
            return get_circle_area(c2[2])
        # Check for circle 1
        else:
            return get_circle_area(c1[2])
    # If the circles overlap and do not encapsulate each other, calculate the overlap
    # Swap if radius_2 is larger than radius_1
    rs = min(radius_1, radius_2)
    rl = max(radius_1, radius_2)
    # calculate part1
    part1 = rs * rs * math.acos(
        (dist * dist + rs * rs - rl * rl) / (2 * dist * rs)
    )
    # calculate part2
    part2 = rl * rl * math.acos(
        (dist * dist + rl * rl - rs * rs) / (2 * dist * rl)
    )
    # calculate part3
    part3 = 0.5 * math.sqrt(
        (-dist + rs + rl) * (dist + rs - rl) * (dist - rs + rl) * (dist + rs + rl)
    )
    return part1 + part2 - part3


def check_if_two_circles_overlap(c1: Tuple[int, int, int, int], c2: Tuple[int, int, int, int]) -> bool:
    """
    Method to check if the two given circles overlap

    :param c1: The center, diameter and identifier of the first circle
    :param c2: The center, diameter and identifier of the second circle
    :return: True, if both circles are overlapping else False
    """
    return euclidean_distance((c1[0], c1[1]), (c2[0], c2[1])) <= abs(c1[2] / 2 - c2[2] / 2)
        

def calculate_overlap_between_two_circles_as_percentage(c1: Tuple[int, int, int, int],
                                                        c2: Tuple[int, int, int, int]) -> float:
    """
    Function to calculate the overlap between two circles as a percentage of the total area

    :param c1: The center and diameter of the first circle
    :param c2: The center and diameter of the second circle
    :return: The overlapping area between both circles, as float
    """
    # Calculate the areas of  the circles
    area_1 = get_circle_area(c1[2])
    area_2 = get_circle_area(c2[2])
    # Calculate the overlapping area between both circles
    overlapping_area = calculate_overlap_between_two_circles(c1, c2)
    if overlapping_area == 0:
        return 0
    else:
        # Get the total area
        return overlapping_area / (area_1 + area_2 - overlapping_area)


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
            dist = euclidean_distance(point1, point2)
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
