"""
Created on 09.04.2019
@author: Romano Weiss
"""
import numpy as np
import hashlib
import json


class ROI:
    __slots__ = [
        "main",
        "ident",
        "auto",
        "dims",
        "points",
        "stats",
        "associated"
    ]

    def __init__(self, main=True, channel="DAPI", auto=True, associated=None):
        self.main = main
        self.ident = channel
        self.auto = auto
        self.dims = {}
        self.points = {}
        self.stats = {}
        self.associated = associated

    def __add__(self, other):
        if isinstance(other, ROI):
            if other.ident == self.ident:
                self.points.update(other.points)
            else:
                raise Warning("ROIs have different channel IDs!")
        else:
            raise ValueError("Not an ROI")

    def __eq__(self, other):
        if isinstance(other, ROI):
            return self.points == other.points
        else:
            return False

    def __ne__(self, other):
        if not isinstance(other, ROI):
            return True
        else:
            return not self.__eq__(other)

    def __len__(self):
        return len(self.points)

    def __hash__(self):
        return hashlib.md5("{}{}".format(self.points, self.ident).encode()).hexdigest()

    def add_point(self, point, intensity):
        """
        Method to add a point to this ROI
        :param point: The point as tuple (x,y)
        :param intensity: The intensity of the image associated with this point (int or float)
        :return: None
        """
        if not isinstance(point, tuple) or not isinstance(intensity, (int, float)):
            raise ValueError("Type mismatch!")
        else:
            self.points[point] = intensity
            self.dims.clear()
            self.stats.clear()

    def calculate_roi_intersection(self, roi):
        """
        Method to calculate the intersection ratio to another ROI
        :param roi: The other ROI
        :return: The degree of intersection in %
        """
        max_intersection = max(len(self), len(roi))
        intersection = 0
        for p in self.points.keys():
            for p2 in roi.get_points():
                if p == p2:
                    intersection += 1
        return intersection / max_intersection

    def get_points(self):
        """
        Method to get a list of all stored points
        :return: The points as list
        """
        return self.points.keys()

    def get_intensities(self):
        """
        Method to get a list of all stored intensity values
        :return: The intensity values as list
        """
        return self.points.values()

    def get_as_numpy(self):
        """
        Method to get this roi as numpy array
        :return: The created numpy array
        """
        self.calculate_dimensions()
        vals = self.points.keys()
        array = np.zeros(shape=(self.dims["height"], self.dims["width"]))
        for point in vals:
            array[point[1] - self.dims["minY"], point[0] - self.dims["minX"]] = self.points[point]
        return array

    def calculate_dimensions(self):
        """
        Method to calculate the dimension of this roi
        :return: The calculated dimensions as dict
        """
        if not self.dims:
            vals = self.points.keys()
            xvals = [x for x in vals[0]]
            yvals = [y for y in vals[1]]
            self.dims["minX"] = min(xvals)
            self.dims["maxX"] = max(xvals)
            self.dims["minY"] = min(yvals)
            self.dims["maxY"] = max(yvals)
            self.dims["width"] = self.dims["maxX"] - self.dims["minX"]
            self.dims["height"] = self.dims["maxY"] - self.dims["minY"]
            self.dims["center"] = (np.average(xvals), np.average(yvals))
        return self.dims

    def calculate_statistics(self):
        """
        Method to calculate statistics for this roi
        :return: The calculated statistics as dict
        """
        if not self.stats:
            vals = self.points.values()
            self.stats = {
                "area": len(self.points),
                "intensity average": np.average(vals),
                "intensity median": np.median(vals),
                "intensity maximum": max(vals),
                "intensity minimum": min(vals),
                "intensity std": np.std(vals)
            }
        return self.stats

    def convert_to_json(self):
        """
        Method to convert this roi to a json str
        :return: The json str
        """
        return json.dump(self.__dict__)

    def initialize_from_json(self, json_):
        """
        Method to initialize this roi from a json str
        :param json_: The json str
        :return: None
        """
        self.__dict__ = json.loads(json)

