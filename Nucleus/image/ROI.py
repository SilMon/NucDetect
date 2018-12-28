'''
Created on 06.10.2018

@author: Romano Weiss
'''

from NucDetect.image import Channel
from operator import itemgetter
from skimage.exposure.exposure import intensity_range
from builtins import int


class ROI:
    '''
    Class to handle Regions Of Interest (R.O.I.)
    '''
    NO_ENCLOSURE = -1
    PARTIAL_ENCLOSURE = 0
    FULL_ENCLOSURE = 1

    def __init__(self, points=None, chan=Channel.BLUE):
        '''
        Constructor to initialize the ROI. Each ROI is initialized with no
        points and assumed to be in the blue channel if not set otherwise

        Keyword arguments:
        points(list of 2D tuples, optional): The points which describe the area
        of the ROI. Contains 2D tuples in form of (x,y)

        channel(int, optional): Describes the channel in which the ROI was
        found (default: channel.BLUE)
        '''
        self.chan = chan
        self.coordinates = [0, 0, 0, 0]
        self.width = None
        self.height = None
        self.center = None
        self.points = []
        self.green = []
        self.red = []
        self.intensities = {}
        self.average_intensity = 0
        self.averarage_foci_intensity = []
        if points is not None:
            self.points.append(points)

    def __add__(self, other):
        if isinstance(other, ROI):
            self.points.extend(other.points)
        elif isinstance(other, ()):
            if len(other) < 2 or len(other) > 2:
                self.points.append(other[0], other[1])
            else:
                raise ValueError("Tuple size has to be 2 (point, intensity)!")
        else:
            raise ValueError("Type of other does not match!")

    def add_point(self, point, intensity):
        '''
        Method to add a point to the ROI.

        Keyword arguments:
        point(2D tuple): Point to add to the ROI
        '''
        self.points.append(point)
        self.intensities[point] = intensity

    def _calculate_center(self):
        '''
        Private method to calculate the center of the ROI
        '''
        if self.width is None:
            self.width = self._calculate_width()
        if self.height is None:
            self.height = self._calculate_height()
        self.center = (self.coordinates[0] + self.width//2,
                       self.coordinates[2] + self.height//2)

    def _calculate_width(self):
        '''
        Private method to calculate the width of the ROI
        '''
        minX = min(self.points, key=itemgetter(0))[0]
        maxX = max(self.points, key=itemgetter(0))[0]
        self.coordinates[0] = minX
        self.coordinates[1] = maxX
        self.width = maxX - minX

    def _calculate_height(self):
        '''
        Private method to calculate the height of the ROI
        '''
        minY = min(self.points, key=itemgetter(1))[1]
        maxY = max(self.points, key=itemgetter(1))[1]
        self.coordinates[2] = minY
        self.coordinates[3] = maxY
        self.height = maxY - minY

    def _calculate_average_intensity(self):
        '''
        Private method to calculate the average point intensity of this ROI
        '''
        int_temp = 0
        for key, inten in self.intensities:
            int_temp += inten
        self.average_intensity = int_temp/len(self.points)
        return self.average_intensity

    def _calculate_average_foci_intensity(self):
        green_temp = 0
        red_temp = 0
        for roi in self.green:
            green_temp += roi._calculate_average_intensity()
        for roi in self.red:
            red_temp += roi._calculate_average_intensity()
        self.averarage_foci_intensity.append(green_temp/len(self.green))
        self.averarage_foci_intensity.append(red_temp/len(self.red))
        pass

    def merge(self, roi):
        '''
        Method to merge to ROI.

        Keyword arguments:
        roi(ROI): The ROI to merge this instance with
        '''
        self.points.extend(roi.points)
        self.green.extend(roi.green)
        self.red.extend(roi.red)
        self._calculate_width()
        self._calculate_height()
        self._calculate_center()

    def add_roi(self, roi):
        '''
        Method to add a ROI to this instance. Is different from merge() by only
        adding the red and green points of the given ROI and ignoring its blue
        points

        Keyword arguments:
        roi(ROI): The ROI to add to this instance

        Returns:
        bool -- True if the ROI could be added, False if the roi could not or
                only partially be added
        '''
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
        '''
        Method to access the data stored in this instance.

        Returns:
        dictionary --   A dictionary of the stored information.
                        Keys are: height, width, center, green roi and red roi.
        '''
        if self.width is None:
            self._calculate_width()
        if self.height is None:
            self._calculate_height()
        if self.center is None:
            self._calculate_center()
        inf = {
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
        '''
        Method to determine if a ROI is enclosed by this ROI.

        Keyword arguments:
        roi(ROI): The ROI to test enclosure for.

        Returns:
        int --  ROI.FULL_ENCOLURE if the given ROI is completely enclosed.
                ROI.PARTIAL_ENCLOSURE if the given ROI is partially enclosed.
                ROI.NO_ENCLOSURE if the given ROI is not enclosed.
        '''
        # Use set intersection to determine enclosure
        a = set(self.points)
        b = set(roi.points)
        if b <= a:
            return ROI.FULL_ENCLOSURE
        elif len(a & b) > 0:
            return ROI.PARTIAL_ENCLOSURE
        else:
            return ROI.NO_ENCLOSURE
