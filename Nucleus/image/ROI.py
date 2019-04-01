"""
Created on 06.10.2018

@author: Romano Weiss
"""

from Nucleus.image import Channel
from operator import itemgetter
import numpy as np
import hashlib


class ROI:
    """
    Class to handle Regions Of Interest (R.O.I.)
    """
    NO_ENCLOSURE = -1
    PARTIAL_ENCLOSURE = 0
    FULL_ENCLOSURE = 1

    def __init__(self, points=None, chan=Channel.BLUE):
        """
        Constructor to initialize the ROI. Each ROI is initialized with no
        points and assumed to be in the blue channel if not set otherwise
        :param image_id: Tmd5 hash of the image containing the ROI (optional)
        :param points: The points which describe the area of the ROI. Contains 2D tuples in form of (x,y) (2D list)
        :param chan: Describes the channel in which the ROI was found (default: channel.BLUE, optional)
        """
        self.chan = chan
        self.coordinates = [0, 0, 0, 0]
        self.width = None
        self.height = None
        self.center = None
        self.points = []
        self.green = []
        self.red = []
        self.stat = {}
        self.intensities = {}
        self.min_foc_int = 20
        self.min_border = 5
        self.on_edge = False
        if points is not None:
            self.points.append(points)

    def __add__(self, other):
        if isinstance(other, ROI):
            self.points.extend(other.points)
        elif isinstance(other, ()):
            if len(other) != 2:
                self.points.append(other[0], other[1])
            else:
                raise ValueError("Tuple size has to be 2 (point, intensity)!")
        else:
            raise ValueError("Type of other does not match!")

    def __eq__(self, other):
        if type(other) is not ROI:
            return False
        elif self.chan == Channel.BLUE:
            return self.points == other.points
        else:
            selfset = set(self.points)
            otherset = set(other.points)
            if selfset <= otherset or otherset <= selfset:
                return True
            else:
                return False

    def add_point(self, point, intensity):
        """
        Method to add a point to the ROI.

        Keyword arguments:
        point(2D tuple): Point to add to the ROI
        """
        self.points.append(point)
        self.intensities[point] = intensity

    def _calculate_center(self):
        """
        Private method to calculate the center of the ROI
        """
        if self.width is None:
            self._calculate_width()
        if self.height is None:
            self._calculate_height()
        self.center = (self.coordinates[0] + self.width//2,
                       self.coordinates[2] + self.height//2)

    def _calculate_width(self):
        """
        Private method to calculate the width of the ROI
        """
        minX = min(self.points, key=itemgetter(0))[0]
        maxX = max(self.points, key=itemgetter(0))[0]
        self.coordinates[0] = minX
        self.coordinates[1] = maxX
        self.width = maxX - minX

    def _calculate_height(self):
        """
        Private method to calculate the height of the ROI
        """
        minY = min(self.points, key=itemgetter(1))[1]
        maxY = max(self.points, key=itemgetter(1))[1]
        self.coordinates[2] = minY
        self.coordinates[3] = maxY
        self.height = maxY - minY

    def perform_foci_quality_check(self):
        if self.chan is not Channel.BLUE:
            raise Exception("A focus has no foci!")
        for red in self.red:
            stat = red.calculate_statistics()
            if stat["av_int"] < self.min_foc_int or stat["area"] < 5:
                self.red.remove(red)
        for green in self.green:
            stat = green.calculate_statistics()
            if stat["av_int"] < self.min_foc_int or stat["area"] < 5:
                self.green.remove(green)

    def merge(self, roi):
        """
        Method to merge to ROI.

        Keyword arguments:
        roi(ROI): The ROI to merge this instance with
        """
        self.points.extend(roi.points)
        self.green.extend(roi.green)
        self.red.extend(roi.red)
        self._calculate_width()
        self._calculate_height()
        self._calculate_center()

    def add_roi(self, roi):
        """
        Method to add a ROI to this instance. Is different from merge() by only
        adding the red and green points of the given ROI and ignoring its blue
        points

        Keyword arguments:
        roi(ROI): The ROI to add to this instance

        Returns:
        bool -- True if the ROI could be added, False if the roi could not or
                only partially be added
        """
        val = self._determine_enclosure(roi)
        enc = False
        if val == ROI.FULL_ENCLOSURE:
            if roi.chan is Channel.GREEN:
                self.green.append(roi)
            if roi.chan == Channel.RED:
                self.red.append(roi)
            enc = True
        '''
        elif val is ROI.PARTIAL_ENCLOSURE:
            a = set(self.points)
            b = set(roi.points)
            if roi.chan == Channel.GREEN:
                self.green.append(ROI(points=list(a & b), chan=Channel.GREEN))
            elif roi.chan == Channel.RED:
                self.red.append(ROI(points=list(a & b), chan=Channel.RED))
            roi.points = list(a - b)
        '''
        return enc

    def get_data(self):
        """
        Method to access the data stored in this instance.

        Returns:
        dictionary --   A dictionary of the stored information.
                        Keys are: height, width, center, green roi and red roi.
        """
        if self.width is None:
            self._calculate_width()
        if self.height is None:
            self._calculate_height()
        if self.center is None:
            self._calculate_center()
        inf = {
            "minmax": self.coordinates,
            "height": self.height,
            "width": self.width,
            "center": self.center,
            "green roi": self.green,
            "green_intensity": 0,
            "red roi": self.red,
            "red_intensity": 0,
            "id": self.get_id()
        }
        return inf

    def _determine_enclosure(self, roi):
        """
        Method to determine if a ROI is enclosed by this ROI.

        Keyword arguments:
        roi(ROI): The ROI to test enclosure for.

        Returns:
        int --  ROI.FULL_ENCOLURE if the given ROI is completely enclosed.
                ROI.PARTIAL_ENCLOSURE if the given ROI is partially enclosed.
                ROI.NO_ENCLOSURE if the given ROI is not enclosed.
        """
        # Use set intersection to determine enclosure
        a = set(self.points)
        b = set(roi.points)
        if b <= a:
            return ROI.FULL_ENCLOSURE
        elif len(a & b) > 0:
            return ROI.PARTIAL_ENCLOSURE
        else:
            return ROI.NO_ENCLOSURE

    def calculate_statistics(self):
        """
        Method to calculate the statistics regarding this ROI

        :return: dict -- A dict containing the calculated data
        """
        if self.chan == Channel.BLUE:
            red_int = []
            red_area = []
            green_int = []
            green_area = []
            xpos = [x for x in self.points[0]]
            ypos = [y for y in self.points[1]]
            for red in self.red:
                stat = red.calculate_statistics()
                t_int = stat["av_int"]
                t_are = stat["area"]
                red_int.append(t_int)
                red_area.append(t_are)
            for green in self.green:
                stat = green.calculate_statistics()
                t_int = stat["av_int"]
                t_are = stat["area"]
                green_int.append(t_int)
                green_area.append(t_are)
            if not red_int or not red_area:
                red_int.append(-1)
                red_area.append(-1)
            if not green_int or green_area:
                green_int.append(-1)
                green_area.append(-1)
            self.stat = {
                "lowest_y": min(ypos),
                "highest_y": max(ypos),
                "lowest_x": min(xpos),
                "highest_x": max(xpos),
                "area": len(self.points),
                "red_roi": len(self.red),
                "green_roi": len(self.green),
                "red_int": red_int,
                "red_av_int": np.average(red_int),
                "red_med_int": np.median(red_int),
                "red_high_int": max(red_int),
                "red_low_int": min(red_int),
                "red_av_area": np.average(red_area),
                "red_low_area": min(red_area),
                "red_high_area": max(red_area),
                "green_int": green_int,
                "green_av_int": np.average(green_int),
                "green_med_int": np.median(green_int),
                "green_high_int": max(green_int),
                "green_low_int": min(green_int),
                "green_av_area": np.average(green_area),
                "green_low_area": min(green_area),
                "green_high_area": max(green_area)
            }
        else:
            int_ = list(self.intensities.values())
            print(min(int_))
            self.stat = {
                "low_int": int(min(int_)),
                "high_int": int(max(int_)),
                "av_int": np.average(int_),
                "med_int": np.median(int_),
                "area": len(int_)
            }
        return self.stat

    def get_id(self):
        """
        Returns an unique identifier for this roi
        :return: The id as str
        """
        m = hashlib.md5("{}{}".format(self.points, self.chan).encode())
        return m.hexdigest()
