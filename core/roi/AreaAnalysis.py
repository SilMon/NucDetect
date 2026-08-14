import math
from collections import defaultdict
from typing import Iterable, Tuple, List, Union

import numba
import numpy as np
from numba import njit


def get_rle_area_intersection(area1: List[Tuple[int, int, int]],
                              area2: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """
    Function to intersect two run length encoded areas

    :param area1: The first area
    :param area2: The second area
    :return: The area intersection
    """
    # Check if both lists contain anything
    if not area1 or not area2:
        return []
    # Index the second area by row. A row can hold more than one run: the encoder emits one per
    # uninterrupted stretch of pixels, so every row crossing the opening of a ring or horseshoe
    # produces two or more. Pairing the two areas by list position instead would misalign them as
    # soon as either has a row gap, and read past the end of the shorter one. The rows were
    # previously forced to hold a single run each by collapsing every row to (leftmost start,
    # summed lengths), which kept the pixel count but slid the pixels left over the gap and so
    # corrupted every position-derived statistic of a concave ROI
    rows2 = defaultdict(list)
    for y, x, rl in area2:
        rows2[y].append((x, x + rl))
    intersect = []
    # Sorting gives scanline order -- row first, then column -- matching the output of encode_areas
    for y, x, rl in sorted(area1):
        for start2, end2 in rows2.get(y, ()):
            # 1D intersection of [x, x + rl) and [start2, end2), both half open
            start = max(x, start2)
            end = min(x + rl, end2)
            if start < end:
                intersect.append((y, start, end - start))
    return intersect


@njit
def amax(lst: Iterable[int]) -> int:
    """
    Numba wrapper for np.amax

    :param lst: The list to get the maximum from
    :return: The maximum value of the list
    """
    # Seeded from the data rather than from a sentinel. -0xffffff was not a floor: any list whose
    # values all lie below -16777215 reduced to the sentinel instead of to a member of the list.
    # It is also what the numba typing note on amin was about -- an int literal cannot unify with
    # the element type of every list this is called with
    if len(lst) == 0:
        raise ValueError("amax of an empty sequence")
    max_ = lst[0]
    for x in lst:
        if x > max_:
            max_ = x
    return max_


@njit
def amin(lst: Iterable[int]) -> int:
    """
    Numba wrapper for np.amin

    :param lst: The list to get the minimum from
    :return: The minimum value of the list
    """
    # Seeded from the data rather than from a sentinel; see amax. This also retires the TODO that
    # stood here -- "Cannot unify Literal[int](16777215) and readonly bytes(uint8, 1d, C) for
    # 'min_'" was numba objecting to exactly this sentinel, so removing it removes the typing
    # conflict rather than working around it
    if len(lst) == 0:
        raise ValueError("amin of an empty sequence")
    min_ = lst[0]
    for x in lst:
        if x < min_:
            min_ = x
    return min_


@njit
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


@njit
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


@njit
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


@njit
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
    if rle:
        # Both extents are measured across the whole area, not taken from one run. The previous
        # version returned len(area) as the height and the longest run as the width, which is only
        # the bounding box of a convex, gap-free shape with exactly one run per row -- and the
        # encoder emits several runs per row for anything concave. Two callers size numpy arrays
        # with these values, so an undersized box silently dropped pixels and, inside njit where
        # bounds are not checked, wrote past the end of the array
        ymax = amax(yvals)
        # a[1] + a[2] is one past the last pixel of the run, the interval being half open
        xmax = amax([a[1] + a[2] for a in area])
        height = ymax - ymin + 1
        width = xmax - xmin
    else:
        # Untouched: no caller passes rle=False, and for a plain point list the run-length
        # arithmetic above does not apply
        height = len(area)
        width = amax([a[1] for a in area]) - xmin + 1
    return ymin, xmin, height, width


@njit
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


@njit
def get_center(area: List[Tuple[int, int, int]]) -> Tuple[int, int]:
    """
    Function to get the center of the given area

    :param area: The area to get the center from
    :return: The center as y, x
    """
    cy, cx = 0.0, 0.0
    total_pixels = 0
    for rle in area:
        ys, xs, rl = rle
        cy += ys * rl
        cx += (xs + (rl - 1)/2) * rl
        total_pixels += rl
    if not total_pixels:
        return 0, 0
    return round(cy / total_pixels), round(cx / total_pixels)


@njit
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


@njit
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


@njit
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


@njit
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


@njit
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


@njit
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


@njit
def get_ellipse_radii(area: Iterable[Tuple[int, int, int]]) -> Tuple[float, float]:
    """
    Function to get the radii of the enclosing ellipse for this area

    :param area: The area
    :return: The major and minor radius
    """
    a1, a2 = get_calculation_factors(area)
    ar = get_surface(area)
    return math.sqrt(((2 * a1) / ar)), math.sqrt(((2 * a2) / ar))


@njit
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


@njit
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
    # Create map containing all points. The whole run is filled, not just its first pixel: a run is
    # (row, first_column, length), so writing pmap[ty][tx] alone put one point per run into the map
    # and the transition count below then tracked the number of runs rather than the outline -- a
    # disc of radius 12 has 23 runs and was reported with a perimeter of 23 against a true
    # circumference of about 72
    for p in area:
        ty, tx = p[0] - cy, p[1] - cx
        pmap[ty, tx: tx + p[2]] = 1
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


@njit
def get_eccentricity(area: Iterable[Tuple[int, int, int]]) -> float:
    """
    Function to get the eccentricity of this roi

    :param area: The area to get the eccentricity from
    :return: The eccentricity as float. -1 if eccentricity can not be calculated
    """
    # PIXELS, not runs. `len(area) < 2` counted RUNS, so any single-row ROI -- however long --
    # reported "cannot be calculated"; a one-row fragment after a check_for_u_turn merge is not
    # unusual. A shape needs two pixels before it has any orientation at all
    pixels = 0
    for run in area:
        pixels += run[2]
    if pixels < 2:
        return -1.0
    a1, a2 = get_calculation_factors(area)
    # sqrt(1 - a2/a1), not a1/a2. The eigenvalue RATIO is 1 for a circle and unbounded for a line;
    # eccentricity is sqrt(1 - (b/a)^2), bounded 0..1 and 0 for a circle -- a different quantity on
    # a different scale. Measured against rasterised ellipses of known eccentricity: 20x15 -> 0.685
    # (true 0.661), 20x10 -> 0.884 (0.866), 25x5 -> 0.984 (0.980); the ratio returned 1.883, 4.558
    # and 32.351 for the same shapes. a1 >= a2 always, by construction in get_calculation_factors
    # (sum + sqrt vs sum - sqrt), so the division needs no ordering guard -- only a floor against
    # a1 == 0 for a degenerate area, and a clamp for float error driving 1 - a2/a1 slightly negative.
    # NOTE: eccentricity is very steep near 0, so a near-circular ROI still reports a visibly
    # non-zero value from rasterisation alone -- a raster circle of r=15 gives 0.237, not 0.000
    if a1 <= 0:
        return -1.0
    return math.sqrt(max(0.0, 1.0 - a2 / a1))
