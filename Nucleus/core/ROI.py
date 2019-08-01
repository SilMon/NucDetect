"""
Created on 09.04.2019
@author: Romano Weiss
"""
from __future__ import annotations
from typing import Union, Dict, List, Tuple
import numpy as np
import hashlib
import json
from math import sqrt


class ROI:
    __slots__ = [
        "main",
        "ident",
        "auto",
        "dims",
        "points",
        "inten",
        "stats",
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
            ident = "{}{}".format(sorted(self.inten.items(), key=lambda k: [k[0], k[1]]), self.ident).encode()
            md5.update(ident)
            self.id = int("0x" + md5.hexdigest(), 0)
        return self.id

    def merge(self, roi: ROI) -> None:
        """
        Method to merge this roi with another ROI

        :param roi: The roi to merge with this
        :return: None
        """
        if isinstance(roi, ROI):
            if roi.ident == self.ident:
                self.points.extend(roi.points)
                self.inten.update(roi.inten)
                self.id = None
                self.dims.clear()
                self.stats.clear()
            else:
                raise Warning("ROIs have different channel IDs!")
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
        dist = sqrt((otherc[0] - selfc[0]) ** 2 + (otherc[1] - otherc[1]) ** 2)
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

    @staticmethod
    def get_roi_map(roi: ROI, asso: ROI, names: List[str]) -> List[np.ndarray]:
        """
        Creates a map of associated roi for the given ROI

        :param roi: The roi to create the association map for
        :param asso:  List of to roi associated ROI
        :param names: An ordered list of channel names
        :return: A tuple containing the different binary association maps
        """
        if not roi.main:
            raise AttributeError("Only main ROI have associated ROI!")
        else:
            d = roi.calculate_dimensions()
            main = np.zeros(shape=(d["height"], d["width"]), dtype="uint8")
            focs = []
            for i in range(len(names)):
                if names[i] == roi.ident:
                    focs.append(None)
                else:
                    focs.append(np.zeros(shape=(d["height"], d["width"]), dtype="uint8"))
            fp = {name: {} for name in names}
            for foc in asso:
                fp[foc.ident].update(foc.inten)
            for p in roi.points:
                main[p[1] - d["minY"], p[0] - d["minX"]] = roi.inten[p]
            for chan, vals in fp.items():
                if chan is not roi.ident:
                    for p, i in vals.items():
                        focs[names.index(chan)][p[1] - d["minY"], p[0] - d["minX"]] = i
            focs[names.index(roi.ident)] = main
            return focs

    @staticmethod
    def get_binary_roi_map(roi: ROI, asso: ROI, names: List[str]) -> List[np.ndarray]:
        """
        Creates a binary map of associated roi for the given ROI

        :param roi: The roi to create the binary association map for
        :param asso:  List of to roi associated ROI
        :param names: An ordered list of channel names
        :return: A tuple containing the different binary association maps
        """
        if not roi.main:
            raise AttributeError("Only main ROI have associated ROI!")
        else:
            d = roi.calculate_dimensions()
            main = np.zeros(shape=(d["height"], d["width"]), dtype="uint8")
            focs = []
            for i in range(len(names)):
                if names[i] == roi.ident:
                    focs.append(None)
                else:
                    focs.append(np.zeros(shape=(d["height"], d["width"]), dtype="uint8"))
            fp = {name: {} for name in names}
            for foc in asso:
                fp[foc.ident].update(foc.inten)
            for p in roi.points:
                main[p[1] - d["minY"], p[0] - d["minX"]] = 1
            for chan, vals in fp.items():
                if chan is not roi.ident:
                    for p, i in vals.items():
                        focs[names.index(chan)][p[1] - d["minY"], p[0] - d["minX"]] = 1
            focs[names.index(roi.ident)] = main
            return focs

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
            else:
                raise Exception("ROI does not contain any points!")
        return self.dims

    def calculate_statistics(self) -> dict[str, Union[int, float]]:
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

