"""
Created on 09.04.2019
@author: Romano Weiss
"""
from __future__ import annotations
import hashlib
import json
import math
import warnings
from typing import Union, Dict, List, Tuple, Iterable
from numba.typed import List as numList

import numpy as np

from core.roi.AreaAnalysis import get_bounding_box, get_center, get_surface, get_ellipse_radii, get_orientation_angle, \
    get_orientation_vector, get_eccentricity, get_ovality


class ROI:
    __slots__ = [
        "main",
        "ident",
        "auto",
        "area",
        "dims",
        "stats",
        "ell_params", # ellipse parameters
        "length",
        "associated",
        "id",
        "marked"
    ]

    def __init__(self, main: bool = True, channel: str = "Blue", auto: bool = True,
                 associated: Union[ROI, None] = None, marked: bool = False):
        """
        Constructor of ROI class

        :param main: Indicates that this roi is on the main channel
        :param channel: Name of the channel
        :param auto: Indicates if the roi was automatically generated
        :param associated: The ROI this ROI is associated with
        :param marked: Convenience flag for processing
        """
        self.main = main
        self.ident = channel
        self.auto = auto
        self.dims = {}
        self.area = []
        self.stats = {}
        self.ell_params = {}
        self.length = -1
        self.associated = associated
        self.marked = marked
        self.id = None

    def __add__(self, other):
        if isinstance(other, ROI):
            self.add_to_area(other.area)
        else:
            raise AttributeError("Addition only supported for ROI class!")

    def __eq__(self, other: ROI):
        if isinstance(other, ROI):
            return set(self.area) == set(other.area)
        else:
            return False

    def __ne__(self, other):
        if not isinstance(other, ROI):
            return True
        else:
            return not self.__eq__(other)

    def __gt__(self, other):
        if not isinstance(other, ROI):
            return False
        else:
            if len(self) > len(other):
                return True
            return False

    def __lt__(self, other):
        if not isinstance(other, ROI):
            return False
        else:
            if len(self) < len(other):
                return True
            return False

    def __len__(self):
        if self.length == -1:
            self.length = np.sum([x[2] for x in self.area])
            return self.length
        else:
            return self.length

    def __hash__(self):
        if self.id is None:
            md5 = hashlib.md5()
            ident = f"{self.ident}{self.area}".encode()
            md5.update(ident)
            self.id = int(f"0x{md5.hexdigest()}", 0)
        return self.id

    def merge(self, roi: ROI) -> None:
        """
        Method to merge this roi with another ROI

        :param roi: The roi to merge with this
        :return: None
        """
        if isinstance(roi, ROI):
            if roi.ident == self.ident:
                self.add_to_area(roi.area)
                self.id = None
                self.dims.clear()
                self.stats.clear()
                self.ell_params.clear()
            else:
                warnings.warn(f"The ROI {hash(self)} and  "
                              f"{hash(roi)} have different channel IDs!({self.ident}, {roi.ident})")
        else:
            raise ValueError("Not an ROI")

    def reset_stored_values(self) -> None:
        """
        Method to reset the calculated id, stored dimensions, statistics and ellipse parameters

        :return: None
        """
        self.id = None
        self.dims.clear()
        self.stats.clear()
        self.ell_params.clear()
        self.length = -1
        self.calculate_dimensions()

    def set_area(self, rle: Iterable) -> None:
        """
        Method to define the area of this ROI

        :param rle: run length encoded area
        :return: None
        """
        self.area.clear()
        self.add_to_area(rle)

    def add_to_area(self, rle):
        """
        Method to extend the area of this ROI with the given area

        :param rle: RL encoded area to add to this ROI
        :return: None
        """
        [self.area.append(x) for x in rle]
        self.reset_stored_values()

    def calculate_ellipse_parameters(self) -> Union[Dict[str, Union[int, float, Tuple, None]]]:
        """
        Method to calculate the ellipse parameters of this ROI.

        :return: A dictionary containing the calculated parameters. None, if the ROI is not main
        """

        # Check if the current ROI is main, else warn
        if not self.main:
            warnings.warn(f"Ellipse Parameter Calculation: ROI {hash(self)} is not marked as main")
            return {"center": None, "major_axis": None, "minor_axis": None, "angle": None,
                    "orientation": None, "area": None, "shape_match": None, "eccentricity": None,
                    "roundness": None}
        # Check if the parameters are already calculated
        if not self.ell_params:
            numba_area = numList(self.area)
            r_maj, r_min = get_ellipse_radii(numba_area)
            or_vec = get_orientation_vector(numba_area)
            angle = get_orientation_angle(numba_area)
            center = get_center(numba_area)
            area = get_surface(numba_area)
            self.ell_params["center"] = center
            self.ell_params["major_axis"] = r_maj
            self.ell_params["minor_axis"] = r_min
            self.ell_params["angle"] = - (math.degrees(angle) - 45)
            self.ell_params["orientation"] = or_vec
            self.ell_params["area"] = r_min * r_maj * math.pi
            self.ell_params["shape_match"] = self.ell_params["area"] / area
            self.ell_params["eccentricity"] = get_eccentricity(numba_area)
            self.ell_params["roundness"] = get_ovality(numba_area)
        return self.ell_params

    def calculate_roi_intersection(self, roi: ROI) -> float:
        """
        Method to calculate the intersection ratio to another ROI

        :param roi: The other ROI
        :return: The degree of intersection as float
        """
        max_intersection = min(len(self), len(roi))
        intersection = set(self.area).intersection(set(roi.area))
        return len(intersection) / max_intersection

    def calculate_dimensions(self) -> Dict[str, Union[int, float]]:
        """
        Method to calculate the dimension of this roi

        :return: The calculated dimensions as dict
        """
        if not self.dims:
            if self.area:
                numba_area = numList()
                # Add elements to area
                [numba_area.append(x) for x in self.area]
                y, x, height, width = get_bounding_box(numba_area)
                center = get_center(numba_area)
                area = get_surface(numba_area)
                self.dims["minX"] = x
                self.dims["maxX"] = x + width
                self.dims["minY"] = y
                self.dims["maxY"] = y + height
                self.dims["width"] = width
                self.dims["height"] = height
                self.dims["center"] = center
                self.dims["area"] = area
            else:
                raise Exception(f"ROI {self.id} does not contain any points!")
        return self.dims

    def calculate_statistics(self, channel: np.ndarray) -> Dict[str, Union[int, float]]:
        """
        Method to calculate statistics for this roi

        :param channel: The channel this ROI is derived from
        :return: The calculated statistics
        """
        if not self.stats:
            # Extract values from channel
            vals = []
            for row in self.area:
                # Iterate over saved points
                for x in range(row[2]):
                    vals.append(
                        channel[row[0]][row[1] + x]
                    )
            self.stats = {
                "area": int(np.sum([x[2] for x in self.area])),
                "intensity average": float(np.average(vals)),
                "intensity median": float(np.median(vals)),
                "intensity maximum": int(np.amax(vals)),
                "intensity minimum": int(np.amin(vals)),
                "intensity std": float(np.std(vals))
            }
        return self.stats
