import math
from typing import Iterable, Tuple, List, Union

import numba
import numpy as np
from matplotlib import pyplot as plt
from numba import njit


def merge_rle_areas(area1: List[Tuple[int, int, int]],
                    area2: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """
    Function to merge to two run length encoded areas

    :param area1: The first area
    :param area2: The second area
    :return: The merged area
    """
    # Check if any of the lines contains u-turns
    area1 = check_for_u_turn(area1)
    area2 = check_for_u_turn(area2)
    # Get overlapping rows
    ov_rows = get_potentially_overlapping_lines(area1, area2)
    # Get all lines, that are not overlapping
    no_lines1 = [line for line in area1 if line[0] not in ov_rows]
    no_lines2 = [line for line in area2 if line[0] not in ov_rows]
    # Get all lines that are potentially overlapping
    o_lines1 = sorted([line for line in area1 if line[0] in ov_rows], key=lambda y: y[0])
    o_lines2 = sorted([line for line in area2 if line[0] in ov_rows], key=lambda y: y[0])
    if len(o_lines1) != len(o_lines2):
        raise ValueError("Number of lines not equal!")
    merged = []
    for ind, line in enumerate(o_lines1):
        al1 = line
        al2 = o_lines2[ind]
        merged.append(merge_lines(al1, al2))
    # Return unchanged and merged lines
    return sorted(no_lines1 + no_lines2 + merged)


def check_for_u_turn(lines: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """
    Method to check if the given list of lines contains lines within the same row

    :param lines: The lines to test
    :return: The tested list of lines
    """
    # Test if lines contains multiple lines with the same row
    rows, counts = np.unique([x[0] for x in lines], return_counts=True)
    tested = []
    # Get row with multiple entries
    for ind, count in enumerate(counts):
        row = int(rows[ind])
        test = [x for x in lines if x[0] == row]
        if count > 1:
            # Merge the lines
            sort = sorted(test, key=lambda x: x[1])
            tested.append((row, sort[0][1], int(np.sum([x[2] for x in sort]))))
        else:
            tested.extend(test)
    return tested


def get_potentially_overlapping_lines(area1: List[Tuple[int, int, int]],
                                      area2: List[Tuple[int, int, int]]) -> List[int]:
    """
    Function to get potentially overlapping lines from rle areas

    :param area1: The first area
    :param area2: The second area
    :return: List of rows where overlaps area possible
    """
    # Get unique rows of both areas
    rows1 = np.unique([y[0] for y in area1])
    rows2 = np.unique([y[0] for y in area2])
    return [x for x in rows1 if x in rows2]


def merge_sorted_rle_areas(sort1: List[Tuple[int, int, int]],
                           sort2: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """
    Function to merge two sorted, run length encoded areas

    :param sort1: The area which is high or equally high
    :param sort2: The lower area
    :return: The merged area
    """
    lines = []
    # Get the area that is left -> y, x, rl
    if min([x[1] for x in sort1]) < min([x[1] for x in sort2]):
        lines1 = sort1
        lines2 = sort2
    else:
        lines1 = sort1
        lines2 = sort2
    start = lines1[0][0]
    start_ind = -1
    for ind, line in enumerate(lines2):
        if line[0] == start:
            start_ind = ind
            break
    # If a starting index was found, get potentially overlapping lines
    if start_ind != -1:
        # Add the lines that are not overlapping
        l2 = lines2[start_ind:]
        l1 = lines1[:len(l2)]
        lines.extend(sort2[:start_ind])
        lines.extend(sort1[len(l2):])
        # Merge the lines that are overlapping
        for line1, line2 in zip(l1, l2):
            # Get amount of overlap
            lines.append(merge_lines(line1, line2))
    else:
        lines = sort1 + sort2
    return lines


def merge_lines(line1: Tuple[int, int, int], line2: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Function to merge two run length encoded lines

    :param line1: The first line
    :param line2: The second line
    :return: The merged line
    """
    # Check which of the lines is left
    check = line1[1] < line2[1]
    # Swap if line 2 is left
    l1 = line1 if check else line2
    l2 = line2 if check else line1
    # Get the distance between start1 and start2
    start_dist = l2[1] - l1[1]
    # Is the second line encased by the first line?
    if l1[1] + l1[2] > start_dist + l2[2]:
        return l1
    else:
        return l1[0], l1[1], start_dist + l2[2]


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
def convert_area_to_array(area: Union[List[Tuple[int, int, int]], numba.typed.List], channel: np.ndarray) -> np.ndarray:
    """
    Function to extract the given area from the channel

    :param area: The run length encoded area
    :param channel: The channel the area is derived from
    :return: The extracted area
    """
    # TODO deprecation warning
    # Get normalization factors
    minrow, mincol, rows, cols = get_bounding_box(area)
    # Create empty image
    carea = np.zeros(shape=(rows, cols))
    # Iterate over area
    for ar in area:
        carea[ar[0] - minrow, ar[1] - mincol: ar[1] - mincol + ar[2]] = channel[ar[0], ar[1]: ar[1] + ar[2]]
    return carea


@njit(cache=True)
def imprint_area_into_array(area: Iterable[Tuple[int, int, int]],
                            array: np.ndarray,
                            ident: int) -> None:
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
def get_bounding_box(area: Union[List[tuple[int, int, int]], numba.typed.List], rle=True) -> Tuple[int, int, int, int]:
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
        # -1 because the start point is included in the run-length
        width = amax([a[1] + a[2] - 1 for a in area]) - xmin
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
    return a1 / (a2 + math.e**-12)
