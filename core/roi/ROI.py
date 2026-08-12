"""
Created on 09.04.2019
@author: Romano Weiss
"""
from __future__ import annotations

import hashlib
import math
import warnings
from typing import Union, Dict, List, Tuple, Iterable

import numpy as np
from numba.typed import List as numList

from core.roi.AreaAnalysis import get_bounding_box, get_center, get_surface, get_ellipse_radii, get_orientation_angle, \
    get_orientation_vector, get_eccentricity, get_ovality
from core.roi import AreaAnalysis


class ROI:
    __slots__ = [
        "main",
        "ident",
        "auto",
        "area",
        "dims",
        "stats",
        "ell_params",
        "length",
        "associated",
        "id",
        "marked",
        "detection_method",
        "match",
        "colocalized"
    ]

    def __init__(self, main: bool = True, channel: str = "Blue", auto: bool = True,
                 associated: Union[int, None] = None, marked: bool = False,
                 method: str = "Not Set", match: float = 0):
        """
        Constructor of ROI class

        :param main: Indicates that this roi is on the main channel
        :param channel: Name of the channel
        :param auto: Indicates if the roi was automatically generated
        :param associated: hash() of the nucleus this ROI lies inside, or None for a nucleus.
                           An identifier, NOT a ROI object -- associate_roi reads it out of the
                           nucleus hash map, so it arrives as a numpy int64, and every live
                           consumer treats it as a number: the database column stores it, and
                           MapComparator uses it directly as a dict key. It was annotated as a
                           ROI for years while never holding one; the only two methods that
                           expected an object were a CSV export superseded in 2020 and removed.
                           A focus is always associated with a nucleus -- one that ends up
                           without is a background artefact and is deleted, not kept.
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
        self.detection_method = method
        self.match = match
        self.colocalized = False
        self.id = None

    # __add__ was removed here. It delegated to what is now intersect_with, so `a + b` read as a
    # union while computing an intersection, mutated its left operand in place and evaluated to
    # None. It had no callers. Use intersect_with, whose name states what actually happens.

    def __eq__(self, other: Union[int, ROI]):
        if isinstance(other, ROI):
            return set(self.area) == set(other.area)
        elif isinstance(other, int):
            return self.id == other

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
        if not self.id:
            md5 = hashlib.md5()
            ident = f"{self.ident}{self.area}".encode()
            md5.update(ident)
            self.id = int(f"0x{md5.hexdigest()}", 0)
        return self.id

    def intersect_with(self, roi: ROI) -> ROI:
        """
        Method to reduce this roi to the area it shares with another ROI

        This is deliberately NOT a union -- see intersect_area for the reasoning. The two ROI are
        the same focus as seen by the two detection methods, and what survives is the area both
        methods agree on.

        :param roi: The roi to intersect this one with
        :return: Reference to self
        """
        if isinstance(roi, ROI):
            if roi.ident == self.ident:
                if not self.associated:
                    self.associated = roi.associated
                self.intersect_area(roi.area)
            else:
                warnings.warn(f"The ROI {hash(self)} and  "
                              f"{hash(roi)} have different channel IDs!({self.ident}, {roi.ident})")
            return self
        else:
            raise ValueError(f"{type(roi)} is not a ROI")

    def get_minimal_representation(self) -> Tuple[int, int, int, int]:
        """
        Method to get the minimal representation of this ROI as

        :return: Tuple of  center Y, center X, diameter, Identifier
        """
        # Get the dimensions of this roi
        dims = self.calculate_dimensions()
        return dims["center_y"], dims["center_x"], max(dims["width"], dims["height"]), hash(self)

    # calculate_overlap was removed here. It approximated both ROI as circles of diameter
    # max(width, height) and compared those, and it had no callers. Overlap between two ROI is
    # available exactly, from their run-length areas, via AreaAnalysis.get_rle_area_intersection --
    # which is what intersect_area already uses.

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
        if not rle:
            return
        # Copy rather than store the caller's list by reference, and do not clear() first: the
        # clear() mutated the list this ROI held *previously*, which any other holder of it would
        # have seen emptied, and it was pointless anyway given the rebind on the next line.
        # Same aliasing hazard already fixed in ImageListModel.set_paths.
        self.area = list(rle)
        self.reset_stored_values()

    def intersect_area(self, rle) -> bool:
        """
        Method to reduce the area of this ROI to the part it shares with the given area

        This SHRINKS the ROI -- it is an intersection, not a union, and that is intended. For
        detection_method "combined" the same focus is detected twice, once per method, and what is
        kept is the area both methods agree on. A union was implemented earlier and deliberately
        replaced: it produced non-circular foci, which does not reflect the biology. Foci are
        circular, and a non-circular blob usually means several overlapping foci, which this
        program is not meant to report as one. Do NOT "fix" this back to a union.

        :param rle: RL encoded area to intersect this ROI with
        :return: True, if the two areas overlap and this ROI was reduced to the shared part.
                 False leaves the ROI untouched, including its detection_method.
        """
        # Get the intersecting area
        intersect = AreaAnalysis.get_rle_area_intersection(self.area, rle)
        if intersect:
            self.area = intersect
            # Kept as "Merged" rather than renamed with the methods: the value is persisted in the
            # roi table, so changing it would invalidate stored results.
            self.detection_method = "Merged"
            self.reset_stored_values()
            return True
        else:
            return False

    def is_valid(self) -> bool:
        """
        Method to check if the roi contains valid data

        :return: True, if the roi is valid
        """
        if self.area:
            return True
        return False

    def calculate_ellipse_parameters(self) -> Union[Dict[str, Union[int, float, Tuple, None]]]:
        """
        Method to calculate the ellipse parameters of this ROI.

        :return: A dictionary containing the calculated parameters. None, if the ROI is not main
        """

        # Check if the current ROI is main, else warn
        if not self.main:
            warnings.warn(f"Ellipse Parameter Calculation: ROI {hash(self)} is not marked as main")
            return {"center_x": None, "center_y": None, "major_axis": None, "minor_axis": None, "angle": None,
                    "orientation_x": None, "orientation_y": None, "area": None, "shape_match": None,
                    "eccentricity": None, "roundness": None}
        # Check if the parameters are already calculated
        if not self.ell_params:
            # An empty area cannot be typed as a numba list at all -- get_surface then raises
            # TypeError("invalid operation on untyped list") -- so it is kept away from numba
            # rather than guarded after the fact
            if self.is_valid():
                numba_area = numList(self.area)
                area = get_surface(numba_area)
            else:
                area = 0
            # The measured surface divides in two places -- in get_ellipse_radii, which computes
            # sqrt(2 * factor / surface), and again for shape_match below -- so a ROI whose runs
            # sum to zero raised ZeroDivisionError out of the middle of the analysis. Such a ROI
            # has no shape to describe, so every parameter is reported as unknown, exactly as for
            # a non-main ROI above. The check is here rather than at the division because the
            # first of the two is inside get_ellipse_radii, one frame down.
            if not area:
                self.ell_params.update({"center_x": None, "center_y": None, "major_axis": None,
                                        "minor_axis": None, "angle": None, "orientation_x": None,
                                        "orientation_y": None, "area": None, "shape_match": None,
                                        "eccentricity": None, "roundness": None})
                return self.ell_params
            r_maj, r_min = get_ellipse_radii(numba_area)
            or_vec = get_orientation_vector(numba_area)
            angle = get_orientation_angle(numba_area)
            center = get_center(numba_area)
            self.ell_params["center_x"] = center[1]
            self.ell_params["center_y"] = center[0]
            self.ell_params["major_axis"] = r_maj
            self.ell_params["minor_axis"] = r_min
            self.ell_params["angle"] = - (math.degrees(angle) - 45)
            self.ell_params["orientation_x"] = or_vec[1]
            self.ell_params["orientation_y"] = or_vec[0]
            self.ell_params["area"] = r_min * r_maj * math.pi
            self.ell_params["shape_match"] = self.ell_params["area"] / area
            self.ell_params["eccentricity"] = get_eccentricity(numba_area)
            self.ell_params["roundness"] = get_ovality(numba_area)
        return self.ell_params

    def calculate_dimensions(self) -> Dict[str, Union[int, float]]:
        """
        Method to calculate the dimension of this roi

        :return: The calculated dimensions as dict
        """
        if not self.dims:
            if self.is_valid():
                numba_area = numList()
                # Add elements to area
                for x in self.area:
                    numba_area.append(x)
                # TODO
                y, x, height, width = get_bounding_box(numba_area)
                center = get_center(numba_area)
                area = get_surface(numba_area)
                self.dims["minX"] = x
                self.dims["maxX"] = x + width
                self.dims["minY"] = y
                self.dims["maxY"] = y + height
                self.dims["width"] = width
                self.dims["height"] = height
                self.dims["center_x"] = center[1]
                self.dims["center_y"] = center[0]
                self.dims["area"] = area
            else:
                # ValueError, not a bare Exception -- a caller cannot catch a bare Exception
                # selectively, and is_valid() above is the check this condition was duplicating
                raise ValueError(f"ROI {self.id} associated to {self.associated} does not contain any points!")
        return self.dims

    def extract_area_intensity(self,
                               channel: np.ndarray) -> List[Union[int, float]]:
        """
        Method to extract the intensity values of this roi from the given channel

        :param channel: The channel to extract the values from
        :return: The extracted values as list
        """
        vals = []
        for row in self.area:
            # Iterate over saved points
            for x in range(row[2]):
                vals.append(
                    # Runs are (row, first_col, length) with first_col INCLUSIVE, so the run covers
                    # first_col .. first_col + length - 1. Do not reintroduce a "- 1" here: it shifts
                    # every run one pixel left, and for a run starting at column 0 it indexes -1,
                    # which numpy wraps to the last column of the same row -- a value from the
                    # opposite side of the image silently entering the statistic.
                    channel[row[0]][row[1] + x]
                )
        return vals

    def calculate_statistics(self, channel: np.ndarray) -> Dict[str, Union[int, float]]:
        """
        Method to calculate statistics for this roi

        :param channel: The channel this ROI is derived from
        :return: The calculated statistics
        """
        if not self.stats:
            # Extract values from channel
            vals = self.extract_area_intensity(channel)
            self.stats = {
                "area": int(np.sum([x[2] for x in self.area])),
                "intensity average": float(np.average(vals)),
                "intensity median": float(np.median(vals)),
                "intensity maximum": int(np.amax(vals)),
                "intensity minimum": int(np.amin(vals)),
                "intensity std": float(np.std(vals))
            }
        return self.stats

    def __str__(self):
        return f"ROI {self.id} - Channel: {self.ident} - Main: {self.main}"
