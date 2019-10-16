import math
import numpy as np

from typing import Tuple
from numba import jit
from numba.types import List


@jit(nopython=True, cache=True)
def eu_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Function to calculate the euclidean distance between two two dimensional points

    :param p1: The first point
    :param p2: The second point
    :return: The distance as float
    """
    return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = [int(w/2), int(h/2)]
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask



@jit(nopython=True, cache=True)
def merge_lists(list1, list2):
    """
    Function to merge to lists. Most suitable for large lists
    :param list1: The first list
    :param list2:  The second list
    :return: The merged list
    """
    lst = List(reflected=False)
    lst.extend(list1)
    lst.extend(list2)
    return lst


@jit(nopython=True, cache=True)
def update_dicts(dict1, dict2):
    pass