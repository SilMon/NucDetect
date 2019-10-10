import math
from typing import Tuple
from numba import jit


@jit(nopython=True)
def eu_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Function to calculate the euclidean distance between two two dimensional points

    :param p1: The first point
    :param p2: The second point
    :return: The distance as float
    """
    return math.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))