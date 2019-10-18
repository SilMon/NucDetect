"""
Created on 09.04.2019
@author: Romano Weiss
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from typing import Union, Dict, List, Tuple

import numpy as np
from numba.typed import List
from skimage.filters import sobel

from Nucleus.core.JittedFunctions import eu_dist, merge_lists


class ROI:
    __slots__ = [
        "main",
        "ident",
        "auto",
        "dims",
        "points",
        "inten",
        "stats",
        "ell_params",
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
        self.points = []
        self.inten = {}
        self.stats = {}
        self.ell_params = {}
        self.associated = associated
        self.marked = marked
        self.id = None

    def __eq__(self, other: ROI):
        if isinstance(other, ROI):
            return set(self.points) == set(other.points)
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
            if set(self.points) > set(other.points):
                return True
            return False

    def __lt__(self, other):
        if not isinstance(other, ROI):
            return False
        else:
            if set(self.points) < set(other.points):
                return True
            return False

    def __len__(self):
        return len(self.points)

    def __hash__(self):
        if self.id is None:
            md5 = hashlib.md5()
            ident = f"{sorted(self.inten.items(), key=lambda k: [k[0], k[1]])}{self.ident}".encode()
            md5.update(ident)
            self.id = int(f"0x{md5.hexdigest()}", 0)
        return self.id

    def merge(self, roi: ROI) -> None:
        """
        Method to merge this roi with another ROI

        :param roi: The roi to merge with this
        :return: None
        """
        # TODO jitten
        if isinstance(roi, ROI):
            if roi.ident == self.ident:
                self.points.extend(roi.points)
                self.inten.update(roi.inten)
                self.id = None
                self.dims.clear()
                self.stats.clear()
                self.ell_params.clear()
            else:
                warnings.warn(f"The ROI {hash(self)} and  "
                              f"{hash(roi)} have different channel IDs!({self.ident}, {roi.ident})")
        else:
            raise ValueError("Not an ROI")

    def add_point(self, point: Tuple[int, int], intensity: int) -> None:
        """
        Method to add a point to this ROI

        :param point: The point as tuple (x,y)
        :param intensity: The intensity of the image associated with this point (int or float)
        :return: None
        """
        self.points.append(point)
        self.inten[point] = intensity
        self.id = None
        self.dims.clear()
        self.stats.clear()
        self.ell_params.clear()

    def set_points(self, point_list: List[tuple[int, int]],
                   original: Union[np.ndarray, Dict[Tuple[int, int], int]]) -> None:
        """
        Method to initialize this roi from a list of points

        :param point_list: The points to add to this roi
        :param original: Either the image where the points are derived from or a dict of intensities
        :return: None
        """
        if isinstance(original, np.ndarray):
            for p in point_list:
                self.add_point(p, original[p[1]][p[0]])
        elif isinstance(original, dict):
            for p in point_list:
                self.add_point(p, original[p])

    def calculate_ellipse_parameters(self) -> Dict[str, Union[int, float, Tuple]]:
        """
        Method to calculate the ellipse parameters of this ROI.

        :return: a dictionary containing the calculated parameters
        """
        # Check if the current ROI is main, else warn
        if not self.main:
            warnings.warn("Ellipse Parameter Calculation: ROI is not marked as main")
        # Check if the parameters are already calculated
        if not self.ell_params:
            if self.main  and len(self) >= 2:
                # Calculate dimensions of ROI
                dims = self.calculate_dimensions()
                offset = dims["minY"], dims["minX"]
                # Get the area of the ROI
                bin_map = self.get_as_binary_map()
                # Add padding for skimage sobel implementation
                bin_map = np.pad(bin_map, pad_width=1,
                                 mode="constant", constant_values=0)
                # Get the edges of the area
                edge_map = sobel(bin_map)
                # Remove added padding
                edge_map = np.array([x[1:-1] for x in edge_map[1:-1]])
                # Extract edge pixels
                points = []
                for y in range(len(edge_map)):
                    for x in range(len(edge_map[0])):
                        if edge_map[y][x] != 0:
                            points.append((y, x))
                # Calculate longest distance for each nucleus
                max_d = 0.0
                p0 = None
                p1 = None
                # Determine main axis
                for r1 in range(len(points)):
                    point1 = points[r1]
                    for r2 in range(r1, len(points)):
                        point2 = points[r2]
                        dist = eu_dist(point1, point2)
                        if dist > max_d:
                            p0 = point1
                            p1 = point2
                            max_d = dist
                # Determine minor axis
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
                max_dist = (p0, p1)
                min_dist = (center, pmin)
                maj_length = eu_dist(max_dist[0], max_dist[1])
                min_length = eu_dist(min_dist[0], min_dist[1]) * 2
                # Calculate overlap between determined ellipse and actual set
                ell_area = math.pi * maj_length / 2 * min_length / 2
                s_area = len(self.points)
                max_a = max((ell_area, s_area))
                min_a = min((ell_area, s_area))
                self.ell_params["center"] = center[0] + offset[0], center[1] + offset[1]
                self.ell_params["major_axis"] = (max_dist[0][0] + offset[0], max_dist[0][1] + offset[1]), \
                                                (max_dist[1][0] + offset[0], max_dist[1][1] + offset[1])
                self.ell_params["major_length"] = maj_length
                self.ell_params["major_slope"] = m_maj
                self.ell_params["major_angle"] = math.degrees(math.atan(abs(self.ell_params["major_slope"])))
                self.ell_params["minor_axis"] = (min_dist[0][0] + offset[0], min_dist[0][1] + offset[1]), \
                                                (min_dist[1][0] + offset[0], min_dist[1][1] + offset[1])
                self.ell_params["minor_length"] = min_length
                self.ell_params["shape_match"] = min_a / max_a
            else:
                return {"center": (None, None), "major_axis": ((None, None), (None, None)), "major_slope": None,
                        "major_length": None, "major_angle": None, "minor_axis": ((None, None), (None, None)),
                        "minor_length": None, "shape_match": None}
        return self.ell_params

    def calculate_roi_intersection(self, roi: ROI) -> float:
        """
        Method to calculate the intersection ratio to another ROI

        :param roi: The other ROI
        :return: The degree of intersection as float
        """
        selfdat = self.calculate_dimensions()
        otherdat = roi.calculate_dimensions()
        selfc = selfdat["center"]
        otherc = otherdat["center"]
        dist = eu_dist(selfc, otherc)
        if dist <= max(selfdat["width"], selfdat["height"])/2 + max(otherdat["width"], otherdat["height"])/2:
            max_intersection = min(len(self), len(roi))
            intersection = set(self.points).intersection(set(roi.points))
            return len(intersection) / max_intersection
        return 0.0

    def get_as_numpy(self) -> None:
        """
        Method to get this roi as numpy array

        :return: The created numpy array
        """
        self.calculate_dimensions()
        array = np.zeros(shape=(self.dims["height"], self.dims["width"]), dtype="uint8")
        for point in self.points:
            array[point[1] - self.dims["minY"], point[0] - self.dims["minX"]] = self.inten[point]
        return array

    def get_as_binary_map(self) -> np.ndarray:
        """
        Method to get this roi as binary map (ndarray)

        :return: The created binary map
        """
        self.calculate_dimensions()
        array = np.zeros(shape=(self.dims["height"], self.dims["width"]))
        for point in self.points:
            array[point[1] - self.dims["minY"], point[0] - self.dims["minX"]] = 1
        return array

    def calculate_dimensions(self) -> Dict[str, Union[int, float]]:
        """
        Method to calculate the dimension of this roi

        :return: The calculated dimensions as dict
        """
        if not self.dims:
            if self.points:
                vals = self.points
                xvals = [x[0] for x in vals]
                yvals = [y[1] for y in vals]
                self.dims["minX"] = min(xvals)
                self.dims["maxX"] = max(xvals)
                self.dims["minY"] = min(yvals)
                self.dims["maxY"] = max(yvals)
                self.dims["width"] = self.dims["maxX"] - self.dims["minX"] + 1
                self.dims["height"] = self.dims["maxY"] - self.dims["minY"] + 1
                self.dims["center"] = (round(np.average(xvals), 2), round(np.average(yvals), 2))
                self.dims["area"] = len(self.points)
            else:
                raise Exception("ROI does not contain any points!")
        return self.dims

    def calculate_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Method to calculate statistics for this roi

        :return: The calculated statistics
        """
        if not self.stats:
            vals = list(self.inten.values())
            self.stats = {
                "area": len(self.points),
                "intensity average": np.average(vals),
                "intensity median": np.median(vals),
                "intensity maximum": max(vals),
                "intensity minimum": min(vals),
                "intensity std": np.std(vals)
            }
        return self.stats

    def convert_to_json(self) -> str:
        """
        Method to convert this roi to a json str

        :return: The json str
        """
        tempinten = {str(x): value for x, value in self.inten}
        d = {
            "channel": self.ident,
            "points": self.points,
            "inten": tempinten,
            "associated": self.associated
        }
        return json.dumps(d)

    def initialize_from_json(self, json_: str) -> None:
        """
        Method to initialize this roi from a json str

        :param json_: The json str
        :return: None
        """
        d = json.loads(json_)
        tempinten = {tuple(x): value for x, value in d["inten"]}
        self.ident = d["channel"]
        self.points = d["points"]
        self.inten = tempinten
        self.associated = ["associated"]
