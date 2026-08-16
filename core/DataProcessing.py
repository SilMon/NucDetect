import math
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from typing import Tuple, Union, List, Dict

import pandas as pd
from scipy.stats import permutation_test, cramervonmises_2samp
from numba import njit
from numba.typed import List as nList


def convert_p_values(pval: float) -> str:
    """
    Function to convert p-values to *
    p > 0.05 -> n.s.
    p <= 0.05 -> *
    p <= 0.01 -> **
    p <= 0.005 -> ***

    :param pval: The p value to convert
    :return: The converted p-value
    """
    pv_str = "n.s."
    # All three thresholds are inclusive, per the docstring above -- a p of exactly 0.05 is
    # significant. The two lines below already used <=; this one did not.
    if pval <= 0.05:
        pv_str = "*"
    if pval <= 0.01:
        pv_str = "**"
    if pval <= 0.005:
        pv_str = "***"
    return pv_str

def get_unique_pairs(data_a, data_b) -> list:
    """
    Function to create a set of unique pairs from both lists

    :param data_a: The first list of data
    :param data_b: The second list of data
    :return: The set containing the pairs
    """
    return sorted({tuple(sorted((x, y))) for x, y in product(data_a, data_b) if x != y})

def cvm_wrapper(x, y, axis):
    return cramervonmises_2samp(y, x, axis=axis).statistic


def perform_statistical_analysis_on_groups(data: pd.DataFrame,
                                           comparison_groups: List[str]) -> pd.DataFrame:
    """
    Function to compare the given groups to each using a permutation test

    :param data: The relevant group data
    :param comparison_groups: The comparison groups which are tested against

    :return: The calculated data as pandas dataframe
    """
    # Get the unique pairings with each comparison group and channel
    pairs = list(product(get_unique_pairs(comparison_groups, data["Group"].unique()), data["Channel"].unique()))
    # Clean the pairs up to create two parameter lists. Boolean indexing rather than
    # DataFrame.query: group and channel names are user-supplied, and an interpolated name
    # containing an apostrophe breaks the query expression while other query syntax silently
    # changes which rows are selected.
    param_a = [data[data["Channel"] == x[1]] for x in pairs]
    param_b = pairs
    # Start a ProcessPool to calculate the results
    with ProcessPoolExecutor(max_workers=(os.cpu_count() // 2) + 2) as exe:
        res = exe.map(_perform_statistical_analysis_on_group, param_a, param_b)
        rows = []
        [rows.append(x) for x in res]
        stat_data = pd.DataFrame(rows,
                                 columns=("Group", "Channel", "Tested Against", "Statistic", "p-Value", "Significance"))
        # pd.to_numeric is not in-place -- the return value has to be assigned back
        stat_data["Statistic"] = pd.to_numeric(stat_data["Statistic"], errors="coerce")
        stat_data["p-Value"] = pd.to_numeric(stat_data["p-Value"], errors="coerce")
        return stat_data

def _perform_statistical_analysis_on_group(data: pd.DataFrame, pair: Tuple[str, str]) -> Tuple:
    """
    Function to perform a permutation test for the given pair on the given channel

    :param data: The underlying data as pandas DataFrame
    :param pair: The pair to check
    :return: One result row as a tuple -- the caller collects these and builds the DataFrame
    """
    # Boolean indexing rather than DataFrame.query -- see the note in the caller: group names
    # are user-supplied and are not safe to interpolate into a query expression.
    control = data[data["Group"] == pair[0][0]]["Foci"].to_numpy()
    test = data[data["Group"] == pair[0][1]]["Foci"].to_numpy()
    perm_data = permutation_test(data=(control,
                                       test),
                                 rng=42,
                                 statistic=cvm_wrapper,
                                 n_resamples=9999)
    return (pair[0][0], pair[1], pair[0][1], perm_data.statistic,
            perm_data.pvalue, convert_p_values(perm_data.pvalue))


@njit
def create_lg_lut(m: int) -> List[int]:
    """
    Function to create a little_gauss lookup table for the given m values

    :param m: The max number to calculate the little gauss for
    :return: The created lut
    """
    return [little_gauss(x) for x in range(m + 1)]


@njit
def little_gauss(n: int) -> int:
    """
    Function to calculate the sum of all numbers between 0 and n

    :param n: The number to calculate the sum for
    :return: The sum
    """
    return (n * n + n) // 2


@njit
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
        # Stretch a float image onto 0..255. Divide by the RANGE, not by the maximum: image - low
        # already starts at zero, so dividing by the maximum compresses the result by low/high --
        # measured at 90% of the range lost for a channel spanning 3800..3999, and it let a float
        # image holding negative values overflow the uint8 cast (a -1..1 image reached 510 pre-cast
        # and wrapped to 254, rendering the brightest pixels dark). A uniform image has no range to
        # stretch; it already came out all-zero here, so map it to that explicitly rather than
        # dividing by zero and casting a nan -- note that dividing by the range alone would widen
        # that division by zero from "maximum is 0" to "any uniform image".
        low, high = np.amin(image), np.amax(image)
        span = high - low
        image = ((image - low) / span * 255).astype("uint8") if span else np.zeros(image.shape, "uint8")
    imgmin, imgmax = np.iinfo(image.dtype).min, np.iinfo(image.dtype).max
    # np.bincount, not np.histogram: histogram spreads its bins over the DATA range, so bin index
    # i was not pixel value i, while amin/amax were stepped once per bin and then used as pixel
    # values in the lookup table. bincount is indexed BY VALUE, which is what the rest of this
    # function has always assumed.
    #
    # How much this mattered depends on the data, measured both ways: on an 8-bit channel that
    # spans the full 0..255 the bins ARE the values, and the old and new saturation points differ
    # by the shadow offset alone -- 0/38 against 1/38 on a real image here. On a 16-bit channel
    # spanning 3800..3999 the old code produced amin/amax of 2305/63560, values the data never
    # takes, and mapped that image into an output span of 213 out of 65535; this version stretches
    # it across the full range. The defect was latent on this project's images, not harmless
    hist = np.bincount(image.ravel(), minlength=imgmax + 1)
    # Suppress shadows -- skip the darkest 30 % of the pixels
    cumulative = np.cumsum(hist)
    pixel_number = 0.3 * image.shape[0] * image.shape[1]
    shadow_index = int(np.searchsorted(cumulative, pixel_number, side="right")) + 1
    cut_hist = hist[shadow_index:]
    total_pixels = int(cut_hist.sum())
    if not total_pixels:
        # Every pixel fell into the shadow cut -- a flat or near-flat image -- so there is no
        # highlight range to stretch. The previous code carried on and mapped such an image to
        # ALL BLACK: measured on a uniform uint8 image of 7, it produced amin=127, amax=128 and an
        # output whose only value is 0. Returning it unchanged keeps the data instead
        return image
    # Calculate pixel threshold
    thresh = cutoff * total_pixels
    # The saturation points, as VALUES: amin is the value below which at most `thresh` pixels lie,
    # counted upwards from the shadow cut, and amax the same counted downwards from the top
    amin = shadow_index + int(np.searchsorted(np.cumsum(cut_hist), thresh, side="right"))
    amax = imgmax - int(np.searchsorted(np.cumsum(cut_hist[::-1]), thresh, side="right"))
    if amax <= amin:
        # The two saturation points met or crossed: there is no range to map onto the full one.
        # DEFENSIVE, and deliberately recorded as such -- the divide-by-zero this guards against
        # was NOT reproduced, neither on the 110 real images nor across 4000 synthetic ones; the
        # smallest gap observed was 1, on a uniform image. It is cheap and the alternative is an
        # exception out of a background thread, but nobody should read this as a fixed crash
        return image
    # Calculate balance ratio
    ratio = (imgmax - imgmin) / (amax - amin)
    # One entry per possible value, imgmax INCLUDED -- range(imgmax) omitted the top value. The
    # remap is lut[image] rather than a Python loop over every pixel: measured on the real 110
    # image set, 1.6 s -> ~5 ms per channel, and the editor ran this twice per channel on a
    # background thread while both of its check boxes stayed disabled
    values = np.arange(imgmax + 1, dtype="float64")
    lut = np.clip(imgmin + (values - amin) * ratio, imgmin, imgmax).astype(image.dtype)
    return lut[image]


@njit
def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Function to calculate the Euclidean distance between two two-dimensional points

    :param p1: The first point
    :param p2: The second point
    :return: The distance as float
    """
    return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))


# A family of circle-geometry helpers (get_circle_area,
# calculate_overlap_between_two_circles[_as_percentage], check_if_two_circles_overlap,
# check_circles_for_engulfment) was removed here, together with ROI.calculate_overlap, its only
# non-dead caller-of-a-caller. They approximated each ROI by a circle of diameter
# max(width, height) and compared those circles.
#
# Nothing called them. MapComparator imported four of them and used none; ROI.calculate_overlap had
# no caller at all. They were superseded by the cKDTree pass in
# MapComparator.get_overlap_between_lists, which is both cheaper -- a ball query instead of an
# all-pairs circle test -- and, for the merge path, strictly more accurate: the candidate pairs it
# produces are then intersected with get_rle_area_intersection, which uses the ROI's true run-length
# geometry rather than a circular approximation of it.
#
# Do not reintroduce a circle approximation for that purpose. If a size-aware test is ever needed,
# the run-length intersection already available on ROI is the correct primitive.


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


@njit
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


@njit
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


@njit
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


@njit
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
