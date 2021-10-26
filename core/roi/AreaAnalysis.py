import math
from typing import Iterable, Tuple

import numpy as np
from numba import njit


@njit(cache=True)
def amax(lst: Iterable[int]) -> int:
    """
    Numba wrapper for np.amax

    :param lst: The list to get the maximum from
    :return: The maximum value of the list
    """
    max_ = -0xffffff
    for x in lst:
        if x > max_:
            max_ = x
    return max_


@njit(cache=True)
def amin(lst: Iterable[int]) -> int:
    """
    Numba wrapper for np.amin

    :param lst: The list to get the minimum from
    :return: The minimum value of the list
    """
    min_ = 0xffffff
    for x in lst:
        if x < min_:
            min_ = x
    return min_


@njit(cache=True)
def convert_area_to_binary_map(area: Iterable[Tuple[int, int]]) -> np.ndarray:
    """
    Function to convert an area to an array representation

    :param area: The area to convert
    :return: The created array
    """
    # Get normalization factors
    minrow, mincol, rows, cols = get_bounding_box(area)
    # Create empty image
    binmap = np.zeros(shape=(rows, cols))
    # Iterate over area
    for ar in area:
        binmap[ar[0] - minrow, ar[1] - mincol: ar[1] - mincol + ar[2]] = 1
    return binmap


@njit(cache=True)
def convert_area_to_array(area: Iterable[Tuple[int, int, int]], channel: np.ndarray) -> np.ndarray:
    """
    Function to extract the given area from the channel

    :param area: The run length encoded area
    :param channel: The channel the area is derived from
    :return: The extracted area
    """
    # Get normalization factors
    minrow, mincol, rows, cols = get_bounding_box(area)
    # Create empty image
    carea = np.zeros(shape=(rows, cols), dtype="uint8")
    # Iterate over area
    for ar in area:
        carea[ar[0] - minrow, ar[1] - mincol: ar[1] - mincol + ar[2]] = channel[ar[0], ar[1]: ar[1] + ar[2]]
    return carea


@njit(cache=True)
def imprint_area_into_array(area: Iterable[Tuple[int, int, int]], array: np.ndarray, ident: int) -> None:
    """
    Method to imprint the specified area into the specified area

    :param area: The run length encoced area to imprint
    :param array: The array to imprint into
    :param ident: The identifier to use for the imprint
    :return: None
    """
    # Get normalization factors
    for ar in area:
        array[ar[0], ar[1]: ar[1] + ar[2]] = ident


@njit(cache=True)
def get_bounding_box(area: Iterable[Tuple], rle=True) -> Tuple[int, int, int, int]:
    """
    Function to calculate the bounding box of the given area

    :param area: The area to get the bounding box of
    :param rle: Indicator if the area is run length encoded
    :return: The bounding box
    """
    xmin = amin([a[1] for a in area])
    yvals = [a[0] for a in area]
    ymin = amin(yvals)
    height = len(area)
    if rle:
        width = amax([a[1] + a[2] for a in area]) - xmin
    else:
        width = amax([a[1] for a in area]) - xmin + 1
    return ymin, xmin, height, width


@njit(cache=True)
def get_surface(area: Iterable[Tuple[int, int, int]]) -> int:
    """
    Function to get the surface of an area

    :param area: The area to get the surface of
    :return: The surface
    """
    s = 0
    for rle in area:
        s += rle[2]
    return s


@njit(cache=True)
def get_center(area: Iterable[Tuple[int, int, int]]) -> Tuple[int, int]:
    """
    Function to get the center of the given area

    :param area: The area to get the center from
    :return: The center as y, x
    """
    cy, cx = 0, 0
    s = len(area)
    for rle in area:
        ys, xs, rl = rle
        cy += ys
        cx += (xs + xs + rl - 1) / 2
    return round(cy / s), round(cx / s)


@njit(cache=True)
def get_moment(area: Iterable[Tuple[int, int, int]],  p: int, q: int) -> float:
    """
    Function to get the moment of this ROI specified by p and q

    :param area: The area to get the moment from
    :param p: First parameter
    :param q: Second parameter
    :return: The calculated area moments
        """
    mom = 0.0
    for rl in area:
        for x in range(rl[1], rl[1] + rl[2] -1, 1):
            mom += (rl[0] ** q) * (x ** p)
    return mom


@njit(cache=True)
def get_central_moment(area: Iterable[Tuple[int, int, int]], p: int, q: int) -> float:
    """
    Function to get the central moment of this ROI

    :param area: The area to get the central moment from
    :param p: First parameter
    :param q: Second parameter
    :return: The calculated central moment
    """
    m10, m01 = get_center(area)
    mom = 0.0
    for rl in area:
        for x in range(rl[1], rl[1] + rl[2] - 1, 1):
            mom += ((rl[0] - m10) ** q) * ((x - m01) ** p)
    return mom


@njit(cache=True)
def get_normalized_central_moment(area: Iterable[Tuple[int, int, int]], p: int, q: int) -> float:
    """
    Function to get the normalized central moment of this ROI

    :param area: The area to get the normalized central moment from
    :param p: First parameter
    :param q: Second parameter
    :return: The normalized central moment
    """
    m00 = get_moment(0, 0)
    norm = m00 ** (p + q + 2)
    return get_central_moment(area, p, q) / norm


@njit(cache=True)
def get_orientation_angle(area: Iterable[Tuple[int, int, int]]) -> float:
    """
    Function to get the angle of the main rotation axis of this roi relative to the main axis

    :param area: The area to get the orientation angle from
    :return: The angle of the rotation axis in radians
    """
    m11 = get_central_moment(area, 1, 1)
    m20 = get_central_moment(area, 2, 0)
    m02 = get_central_moment(area, 0, 2)
    if m20 != m02:
        return 0.5 * math.atan2(m20 - m02, 2 * m11)
    else:
        return 0.0


@njit(cache=True)
def get_orientation_vector(area: Iterable[Tuple[int, int, int]]) -> Tuple[float, float]:
    """
    Function to get the orientation vector of this ROI, relative to the main axis

    :param area: The area to calculate the orientation vector from
    :return: The orientation vector
    """
    a = get_central_moment(area, 1, 1) * 2
    b = get_central_moment(area, 2, 0) - get_central_moment(area, 0, 2)
    if a == b:
        return 0, 0
    else:
        x = (0.5 * (1 + (b / math.sqrt(a * a + b * b)))) ** 0.5
        y = (0.5 * (1 - (b / math.sqrt(a * a + b * b)))) ** 0.5
        return x, y if a >= 0 else -y


@njit(cache=True)
def get_calculation_factors(area: Iterable[Tuple[int, int, int]]) -> Iterable[float]:
    """
    Function to get the central moments m20, m02 and m11 from an area

    :param area: The area to get the moments from
    :return: The factors a1 and a2
    """
    m20 = get_central_moment(area, 2, 0)
    m02 = get_central_moment(area, 0, 2)
    m11 = get_central_moment(area, 1, 1)
    a1 = m20 + m02 + math.sqrt(((m20 - m02) ** 2) + 4 * (m11 ** 2))
    a2 = m20 + m02 - math.sqrt(((m20 - m02) ** 2) + 4 * (m11 ** 2))
    return a1, a2


@njit(cache=True)
def get_ellipse_radii(area: Iterable[Tuple[int, int, int]]) -> Tuple[float, float]:
    """
    Function to get the radii of the enclosing ellipse for this area

    :param area: The area
    :return: The major and minor radius
    """
    a1, a2 = get_calculation_factors(area)
    ar = get_surface(area)
    return math.sqrt(((2 * a1) / ar)), math.sqrt(((2 * a2) / ar))


@njit(cache=True)
def get_ovality(area: Iterable[Tuple[int, int, int]]) -> float:
    """
    Function to calculate the ovality of the given area

    :param area: The area to calculate the ovality from
    :return: The ovality as float. -1 if ovality can not be calculated
    """
    if len(area) < 2:
        return -1.0
    # Get perimeter
    per = get_perimeter(area)
    are = get_surface(area)
    return 4 * math.pi * are / per ** 2


@njit(cache=True)
def get_perimeter(area: Iterable[Tuple[int, int, int]]) -> int:
    """
    Function to get the perimeter of the given area

    :param area: The area to get the perimeter from
    :return: The perimeter of the area
    """
    # Get bounding box
    bb = get_bounding_box(area)
    pmap = np.zeros((bb[2] + 1, bb[3] + 1))
    cy, cx = bb[0], bb[1]
    # Create map containing all points
    for p in area:
        ty, tx = p[0] - cy, p[1] - cx
        pmap[ty][tx] = 1
    perimeter = 0
    # Check for transitions between background and foreground
    for y in range(bb[2]):
        for x in range(bb[3]):
            # If point is on map corner, it is part of the perimeter
            if (y == 0 or y == bb[2] - 1 or x == 0 or x == bb[3] - 1) and pmap[y][x]:
                perimeter += 1
            elif pmap[y][x]:
                if not pmap[y - 1][x] or not pmap[y + 1][x] \
                        or not pmap[y][x - 1] or not pmap[y][x + 1]:
                    perimeter += 1
    return perimeter


@njit(cache=True)
def get_eccentricity(area: Iterable[Tuple[int, int, int]]) -> float:
    """
    Function to get the eccentricity of this roi

    :param area: The area to get the eccentricity from
    :return: The eccentricity as float. -1 if eccentricity can not be calculated
    """
    if len(area) < 2:
        return -1.0
    a1, a2 = get_calculation_factors(area)
    return a1 / a2
