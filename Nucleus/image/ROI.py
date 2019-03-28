"""
Created on 06.10.2018

@author: Romano Weiss
"""

from Nucleus.image import Channel
from operator import itemgetter
import numpy as np


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

        Keyword arguments:
        points(list of 2D tuples, optional): The points which describe the area
        of the ROI. Contains 2D tuples in form of (x,y)

        channel(int, optional): Describes the channel in which the ROI was
        found (default: channel.BLUE)
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
            "red_intensity": 0
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
            lowest_y = 0xffffffff
            lowest_x = 0xffffffff
            highest_y = -1
            highest_x = -1
            red_av_int = 0
            red_low_int = 255
            red_high_int = 0
            red_med_int = []
            red_av_area = 0
            red_low_area = 0
            red_high_area = 0
            green_av_int = 0
            green_low_int = 255
            green_high_int = 0
            green_med_int = []
            green_av_area = 0
            green_low_area = 0
            green_high_area = 0
            for point in self.points:
                if point[0] < lowest_x:
                    lowest_x = point[0]
                if point[0] > highest_x:
                    highest_x = point[0]
                if point[1] < lowest_y:
                    lowest_y = point[0]
                if point[1] > highest_y:
                    highest_y = point[1]
            for red in self.red:
                stat = red.calculate_statistics()
                t_int = stat["av_int"]
                t_are = stat["area"]
                red_av_int += t_int
                red_med_int.append(t_int)
                red_high_int = t_int if t_int > red_high_int else red_high_int
                red_low_int = t_int if t_int < red_low_int else red_low_int
                red_av_area += t_are
                red_high_area = t_are if t_are > red_high_area else red_high_area
                red_low_area = t_are if t_are < red_low_area else red_low_area
            len_red = len(self.red) if len(self.red) > 0 else 1
            red_av_area = red_av_area / len_red
            red_av_int = red_av_int / len_red
            for green in self.green:
                stat = green.calculate_statistics()
                t_int = stat["av_int"]
                t_are = stat["area"]
                green_av_int += t_int
                green_med_int.append(t_int)
                green_high_int = t_int if t_int > green_high_int else green_high_int
                green_low_int = t_int if t_int < green_low_int else green_low_int
                green_av_area += t_are
                green_low_area = t_are if t_are < green_low_area else green_low_area
                green_high_area = t_are if t_are > green_high_area else green_high_area
            len_green = len(self.green) if len(self.green) > 0 else 1
            green_av_area = green_av_area / len_green
            green_av_int = green_av_int / len_green
            self.stat = {
                "lowest_y": lowest_y,
                "highest_y": highest_y,
                "lowest_x": lowest_x,
                "highest_x": highest_x,
                "area": len(self.points),
                "red_roi": len(self.red),
                "green_roi": len(self.green),
                "red_int": red_med_int,
                "red_av_int": red_av_int,
                "red_med_int": np.median(red_med_int),
                "red_high_int": red_high_int,
                "red_low_int": red_low_int,
                "red_av_area": red_av_area,
                "red_low_area": red_low_area,
                "red_high_area": red_high_area,
                "green_int": green_med_int,
                "green_av_int": green_av_int,
                "green_med_int": np.median(green_med_int),
                "green_high_int": green_high_int,
                "green_low_int": green_low_int,
                "green_av_area": green_av_area,
                "green_low_area": green_low_area,
                "green_high_area": green_high_area
            }
        else:
            av_int = 0
            low_int = 255
            high_int = 0
            med_int = []
            for key, val in self.intensities.items():
                av_int += val
                med_int.append(val)
                low_int = val if val < low_int else low_int
                high_int = val if val > high_int else high_int
            self.stat = {
                "low_int": low_int,
                "high_int": high_int,
                "av_int": av_int / len(self.points),
                "med_int": med_int.sort(),
                "area": len(self.points)
            }
        return self.stat
