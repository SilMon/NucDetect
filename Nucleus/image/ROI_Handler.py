'''
Created on 06.10.2018

@author: Romano Weiss
'''
from skimage.draw import circle_perimeter
from NucDetect.image.ROI import ROI
from NucDetect.image import Channel
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np


class ROI_Handler:
    '''
    Class to detect and handle ROIs
    '''

    def __init__(self, ident=None):
        '''
        Constructor of the class.

        Keyword arguments:
        ident: A unique identifier of the handler
        '''
        self.blue_ws = None
        self.red_ws = None
        self.green_ws = None
        self.id = ident
        self.blue_name = None
        self.red_name = "Red Foci"
        self.green_name = "Green Foci"
        self.nuclei = [None] * 200
        self.green = [None] * 500
        self.red = [None] * 500

    def set_names(self, names):
        '''
        Method to rename the image channels

        Keyword arguments:
        names(tuple): Tuple containing the names of the channels.
        Structure: (blue name, red name, green name)
        '''
        if names is not None:
            if names[0] is not None:
                self.blue_name = names[0]
            if names[1] is not None:
                self.red_name = names[1]
            if names[2] is not None:
                self.green_name = names[2]

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
                        roi.add_point((x, y), blue)
                        self.nuclei[blue] = roi
                    else:
                        self.nuclei[blue].add_point((x, y), blue)
                # Detection of green foci
                if green != 0:
                    if self.green[green] is None:
                        roi = ROI(chan=Channel.GREEN)
                        roi.add_point((x, y), green)
                        self.green[green] = roi
                    else:
                        self.green[green].add_point((x, y), green)
                # Detection of red foci
                if red != 0:
                    if self.red[red] is None:
                        roi = ROI(chan=Channel.RED)
                        roi.add_point((x, y), red)
                        self.red[red] = roi
                    else:
                        self.red[red].add_point((x, y), red)
        '''
        Determination of over-segmentation of nuclei & removal
        of falsely identified nuclei
        '''
        '''                
        av_size = self._calculate_average_roi_area(self.nuclei)
        rem = []
        for roi1 in self.nuclei:
            if roi1 is not None:
                if len(roi1.points) < 0.75 * av_size:
                    for roi2 in self.nuclei:
                        if roi2 is not None and roi2 is not roi1:
                            dist = self._calculate_roi_distance(roi1, roi2)
                            width_diff = abs(roi1.width//2 + roi2.width//2)
                            height_diff = abs(roi1.height//2 + roi2.height//2)
                            if (width_diff >= dist[0]) and (height_diff >= dist[1]):
                                roi1.merge(roi2)
                                rem.append(roi2)
        self.nuclei = [x for x in self.nuclei if x not in rem]
        '''
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
        tuple: The total horizontal and vertical distance in the form (x,y)
        '''
        center1 = roi1.get_data().get("center")
        center2 = roi2.get_data().get("center")
        dist = (abs(center1[0]-center2[0]), abs(center1[1]-center2[1]))
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
        area = []
        for roi in roi_list:
            if roi is not None:
                area.append(len(roi.points))
        return np.median(area)

    def _draw_roi(self, img_array):
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
                self._draw_rois(canvas, roi, (50, 50, 255))
                for green in roi.green:
                    if green is not None:
                        self._draw_rois(canvas, green, (50, 255, 50))
                for red in roi.red:
                    if red is not None:
                        self._draw_rois(canvas, red, (255, 50, 50))
        return canvas

    def _draw_rois(self, img_array, roi, col):
        '''
        Private method to draw rois on a image.

        Keyword arguments:
        img_array(ndarray): The image to draw the ROI on.
        roi(ROI): The roi to draw on the image
        col(3D tuple): The color in which the roi should be highlighted
        '''
        data = roi.get_data()
        rr, cc = circle_perimeter(
                                  data.get("center")[1], data.get("center")[0],
                                  max(data.get("height")//2,
                                      data.get("width")//2, 2),
                                  shape=img_array.shape
                                  )
        img_array[rr, cc, :] = col

    def _annotate_image(self, ax, show=True):
        '''
        Method to annotate the result image

        Keyword arguments:
        fig (figure): The figure to annotate
        '''
        ind = 0
        for roi in self.nuclei:
            if roi is not None:
                tex = "In:{0:>6}\nR:{1:>6}\nG:{2:>6}"
                ax.annotate(tex.format(str(ind),
                                       str(len(roi.red)),
                                       str(len(roi.green))
                                       ),
                            roi.get_data()["center"],
                            color="white",
                            fontsize=6 if show else 1.5,
                            bbox=dict(facecolor='none', edgecolor='yellow',
                                      pad=1, linewidth=1 if show else 0.2)
                            )
                ind += 1

    def create_result_image(self, img_array, show=True):
        '''
        Method to create the final output image.

        Returns:
        figure (matplotlib) -- The created result image
        '''
        result = plt.figure()
        ax = result.add_subplot(111)
        ax.imshow(self._draw_roi(img_array))
        self._annotate_image(ax, show)
        return result

    def get_data(self, console=False, formatted=False):
        '''
        Method to obtain the data stored in this handler

        Keyword arguments:
        console(bool):Determines if results are printed to the console
        (default:True)
        formatted(bool): Determines if the output should be formatted
        (default:True)

        Returns:
        dict: A dictionary containing all stored information
        '''
        if formatted:
            form = "{0:^15};{1:^15};{2:^15};{3:^15};{4:^15};{5:^15}"
        else:
            form = "{0};{1};{2};{3};{4};{5}"
        heading = form.format(
                "Index", "Width", "Height", "Center",
                self.green_name, self.red_name)
        now = datetime.datetime.now()
        # Prepare data
        data = {}
        data["header"] = ["Index", "Width", "Height", "Center",
                          self.green_name, self.red_name]
        data["id"] = self.id
        data["date"] = now
        data["blue channel"] = self.blue_name
        data["data"] = []
        ind = 0
        for roi in self.nuclei:
            if roi is not None:
                temp = roi.get_data()
                data["data"].append([ind, temp.get("width"),
                                     temp.get("height"), temp.get("center"),
                                     len(temp.get("green roi")),
                                     len(temp.get("red roi"))])
                ind += 1
        if console:
            print("File id: " + data["id"])
            print("Date: " + str(data["date"]))
            if self.blue_name is not None:
                print("Blue Channel: " + data["blue channel"] + "\n")
            print(heading)
            ind = 0
            for roi in self.nuclei:
                if roi is not None:
                    roi_data = roi.get_data()
                    print(form.format(ind, roi_data.get("width"),
                          roi_data.get("height"), str(roi_data.get("center")),
                          len(roi_data.get("green roi")), len(roi_data.get("red roi")))
                          )
                    ind += 1
        else:
            pardir = os.getcwd()
            pathpardir = os.path.join(os.path.dirname(pardir),
                                      r"results")
            os.makedirs(pathpardir, exist_ok=True)
            pathresult = os.path.join(pathpardir,
                                      "result - " + self.id + ".csv")
            file = open(pathresult, "w")
            ind = 0
            file.write("File id:;" + data["id"] + "\n")
            file.write("Date:;" + str(data["date"]) + "\n")
            if self.blue_name is not None:
                file.write("Blue Channel:;" + data["blue channel"] + "\n")
            file.write("\n" + heading + "\n")
            for roi in self.nuclei:
                if roi is not None:
                    roi_data = roi.get_data()
                    file.write(form.format(ind, roi_data.get("width"),
                               roi_data.get("height"), str(roi_data.get("center")),
                               len(roi_data.get("green roi")),
                               len(roi_data.get("red roi")))+"\n")
                    ind += 1
        return data
