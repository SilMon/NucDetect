'''
Created on 06.10.2018

@author: Romano Weiss
'''
from skimage.draw import ellipse_perimeter
from Nucleus.image.ROI import ROI
from Nucleus.image import Channel
from math import sqrt
import os


class ROI_Handler:
    '''
    Class to detect and handle ROIs
    '''

    def __init__(self, blue_ws=None, green_ws=None, red_ws=None):
        '''
        Constructor of the class.

        Keyword arguments:
        blue_ws(2D numpy): Blue channel watershed
        green_ws(2D numpy): Green channel watershed
        red_ws (2D numpy): Red channel watershed
        '''
        self.blue_ws = blue_ws
        self.red_ws = red_ws
        self.green_ws = green_ws
        self.nuclei = [None] * 100
        self.green = [None] * 500
        self.red = [None] * 500

    def set_watersheds(self, watersheds):
        self.blue_ws = watersheds[0]
        self.red_ws = watersheds[1]
        self.green_ws = watersheds[2]

    def analyse_image(self):
        '''
        Method to analyse an image according to the given data
        '''
        # Analysis of the blue channel
        for y in range(len(self.blue_ws)):
            for x in range(len(self.blue_ws[0])):
                blue = self.blue_ws[y][x]
                green = self.green_ws[y][x]
                red = self.red_ws[y][x]
                # Detection of nuclei
                if blue != 0:
                    if self.nuclei[blue] is None:
                        roi = ROI()
                        roi.add_point((x, y))
                        self.nuclei[blue] = roi
                    else:
                        self.nuclei[blue].add_point((x, y))
                # Detection of green foci
                if green != 0:
                    if self.green[green] is None:
                        roi = ROI(chan=Channel.GREEN)
                        roi.add_point((x, y))
                        self.green[green] = roi
                    else:
                        self.green[green].add_point((x, y))
                # Detection of red foci
                if red != 0:
                    if self.red[red] is None:
                        roi = ROI(chan=Channel.RED)
                        roi.add_point((x, y))
                        self.red[red] = roi
                    else:
                        self.red[red].add_point((x, y))
        # Determine the green and red ROIs each nucleus includes
        gre_rem = []
        red_rem = []
        for nuc in self.nuclei:
            gre_rem.clear()
            red_rem.clear()
            if nuc is not None:
                for gre in self.green:
                    if gre is not None:
                        if nuc.add_roi(gre):
                            gre_rem.append(gre)
                for red in self.red:
                    if red is not None:
                        if nuc.add_roi(red):
                            red_rem.append(red)
            self.green = [x for x in self.green if x not in gre_rem]
            self.red = [x for x in self.red if x not in red_rem]

    def _calculate_roi_distance(self, roi1, roi2):
        '''
        Private method to calculate the distance between two given ROI

        Keyword arguments:
        roi1(ROI): The first ROI
        roi2(ROI): The second ROI

        Returns:
        float: The euclidean distance between the centers of the two ROI
        '''
        center1 = roi1.get_data().get("center")
        center2 = roi2.get_data().get("center")
        dist = sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)
        return dist

    def _calculate_average_roi_area(self, roi_list):
        '''
        Private method to calculate the average area of the stored roi.

        Keyword arguments:
        roi_list(list of ROI):  List which contains the ROI to use for the
                                calculation

        Returns:
        int -- The calculated area
        '''
        area = 0
        for roi in roi_list:
            if roi is not None:
                area += len(roi.points)
        return area

    def draw_roi(self, img_array):
        '''
        Method to draw the ROI saved in this handler on the image

        Keyword arguments:
        img_array(ndarray): The image to draw the ROI on.

        Returns:
        ndarray -- The image with the drawn ROI
        '''
        canvas = img_array.copy()
        for roi in self.nuclei:
            if roi is not None:
                self._draw_roi(canvas, roi, (50, 50, 255))
                for green in roi.green:
                    if green is not None:
                        self._draw_roi(canvas, green, (50, 255, 50))
                for red in roi.red:
                    if red is not None:
                        self._draw_roi(canvas, red, (255, 50, 50))
        return canvas

    def _draw_roi(self, img_array, roi, col):
        '''
        Private method to draw rois on a image.

        Keyword arguments:
        img_array(ndarray): The image to draw the ROI on.
        roi(ROI): The roi to draw on the image
        col(3D tuple): The color in which the roi should be highlighted
        '''
        data = roi.get_data()
        rr, cc = ellipse_perimeter(
                                  data.get("center")[1], data.get("center")[0],
                                  data.get("height")//2, data.get("width")//2,
                                  shape=img_array.shape
                                  )
        img_array[rr, cc, :] = col

    def get_data(self, console=True, formatted=True):
        '''
        Method to obtain the data stored in this handler

        Keyword arguments:
        console(bool):Determines if results are printed to the console
        (default:True)
        formatted(bool): Determines if the output should be formatted
        (default:True)

        Returns:
        None -- If printed to console
        str -- The data as .csv string if printed to file
        '''
        if formatted:
            form = "{0:^15};{1:^15};{2:^15};{3:^15};{4:^15};{5:^15}"
        else:
            form = "{0};{1};{2};{3};{4};{5}"
        heading = form.format(
                "Index", "Width", "Height", "Center", "Green Foci", "Red Foci")
        if console:
            print(heading)
            ind = 0
            for roi in self.nuclei:
                if roi is not None:
                    data = roi.get_data()
                    print(form.format(ind, data.get("width"),
                          data.get("height"), str(data.get("center")),
                          len(data.get("green roi")), len(data.get("red roi")))
                          )
                    ind += 1
        else:
            pardir = os.getcwd()
            pathpardir = os.path.join(os.path.dirname(pardir),
                                      r"results")
            os.makedirs(pathpardir, exist_ok=True)
            pathresult = os.path.join(pathpardir,
                                      "result.csv")
            file = open(pathresult, "w")
            ind = 0
            file.write(heading + "\n")
            for roi in self.nuclei:
                if roi is not None:
                    data = roi.get_data()
                    file.write(form.format(ind, data.get("width"),
                               data.get("height"), str(data.get("center")),
                               len(data.get("green roi")),
                               len(data.get("red roi")))+"\n")
                    ind += 1
