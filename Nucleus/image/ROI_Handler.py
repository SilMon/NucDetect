"""
Created on 06.10.2018

@author: Romano Weiss
"""
from skimage.draw import circle_perimeter
from NucDetect.image.ROI import ROI
from NucDetect.image import Channel
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np


class ROI_Handler:
    """
    Class to detect and handle ROIs
    """

    def __init__(self, ident=None):
        """
        Constructor of the class.

        Keyword arguments:
        ident: A unique identifier of the handler
        """
        self.blue_ws = None
        self.red_ws = None
        self.green_ws = None
        self.id = ident
        self.blue_name = None
        self.red_name = "Red Foci"
        self.green_name = "Green Foci"
        self.nuclei = [None] * 500
        self.green = [None] * 2000
        self.red = [None] * 2000
        self.stat = {}

    def set_names(self, names):
        """
        Method to rename the image channels

        Keyword arguments:
        names(tuple): Tuple containing the names of the channels.
        Structure: (blue name, red name, green name)
        """
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
        """
        Method to analyse an image according to the given data
        """
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
        for nuc in self.nuclei:
            if nuc is not None:
                nuc.perform_foci_quality_check()
        self.calculate_statistics()
        self._check_roi_for_quality()

    def _check_roi_for_quality(self):
        for nucleus in self.nuclei:
            if nucleus is not None:
                nuc_stat = nucleus.calculate_statistics()
                if nuc_stat["area"] <= self.stat["area average"]*0.2:
                    self.nuclei.remove(nucleus)
                elif nuc_stat["area"] >= self.stat["area average"] * 1.4:
                    pass

    def _calculate_roi_distance(self, roi1, roi2):
        """
        Private method to calculate the distance between two given ROI

        Keyword arguments:
        roi1(ROI): The first ROI
        roi2(ROI): The second ROI

        Returns:
        tuple: The total horizontal and vertical distance in the form (x,y)
        """
        center1 = roi1.get_data().get("center")
        center2 = roi2.get_data().get("center")
        dist = (abs(center1[0]-center2[0]), abs(center1[1]-center2[1]))
        return dist

    def _calculate_average_roi_area(self, roi_list):
        """
        Private method to calculate the average area of the stored roi.

        Keyword arguments:
        roi_list(list of ROI):  List which contains the ROI to use for the
                                calculation

        Returns:
        int -- The calculated area
        """
        area = []
        for roi in roi_list:
            if roi is not None:
                area.append(len(roi.points))
        return np.median(area)

    def _draw_roi(self, img_array):
        """
        Method to draw the ROI saved in this handler on the image

        Keyword arguments:
        img_array(ndarray): The image to draw the ROI on.

        Returns:
        ndarray -- The image with the drawn ROI
        """
        canvas = img_array.copy()
        for roi in self.nuclei:
            if roi is not None:
                green_num = 0
                red_num = 0
                for green in roi.green:
                    if green is not None:
                        self._draw_rois(canvas, green, (50, 255, 50))
                        green_num += 1
                for red in roi.red:
                    if red is not None:
                        self._draw_rois(canvas, red, (255, 50, 50))
                        red_num += 1
                if red_num > 0 or green_num > 0:
                    self._draw_rois(canvas, roi, (50, 50, 255))
                else:
                    self._draw_rois(canvas, roi, (120, 120, 255))

        return canvas

    def _draw_rois(self, img_array, roi, col):
        """
        Private method to draw rois on a image.

        Keyword arguments:
        img_array(ndarray): The image to draw the ROI on.
        roi(ROI): The roi to draw on the image
        col(3D tuple): The color in which the roi should be highlighted
        """
        data = roi.get_data()
        rr, cc = circle_perimeter(
                                  data.get("center")[1], data.get("center")[0],
                                  max(data.get("height")//2,
                                      data.get("width")//2, 2),
                                  shape=img_array.shape
                                  )
        img_array[rr, cc, :] = col

    def _annotate_image(self, ax, show=True):
        """
        Method to annotate the result image

        Keyword arguments:
        fig (figure): The figure to annotate
        """
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
        """
        Method to create the final output image.

        Returns:
        figure (matplotlib) -- The created result image
        """
        result = plt.figure()
        ax = result.add_subplot(111)
        ax.imshow(self._draw_roi(img_array))
        self._annotate_image(ax, show)
        return result

    def calculate_statistics(self):
        """
        Method to calculate statistics about the saved ROIs
        :return: dict -- A dictonary containing the data
        """
        area = []
        av_area = 0
        num = 0
        num_red = []
        av_num_red = 0
        num_empty = 0
        int_red = []
        av_int_red = 0
        num_green = []
        av_num_green = 0
        int_green = []
        av_int_green = 0

        for nucleus in self.nuclei:
            if nucleus is not None:
                temp_stat = nucleus.calculate_statistics()
                tarea = temp_stat["area"]
                area.append(tarea)
                av_area += tarea
                tnum_red = temp_stat["red_roi"]
                num_red.append(tnum_red)
                av_num_red += tnum_red
                tint_red = temp_stat["red_av_int"]
                int_red.append(tint_red)
                av_int_red += tint_red
                tnum_green = temp_stat["green_roi"]
                num_green.append(tnum_green)
                av_num_green += tnum_green
                tint_green = temp_stat["green_av_int"]
                int_green.append(tint_green)
                av_int_green += tint_green
                num += 1
                if tnum_red is 0 and tnum_green is 0:
                    num_empty += 1
        if num > 0:
            av_area /= len(area)
            red = len(num_red)
            av_num_red /= red
            av_int_red /= red
            green = len(num_red)
            av_num_green /= green
            av_int_green /= green

        self.stat = {
            "area": area,
            "area average": av_area,
            "number": num,
            "empty": num,
            "number red": num_red,
            "number green": num_green,
            "number red average": av_num_red,
            "number green average": av_num_green,
            "intensity red": int_red,
            "intensity green": int_green,
            "intensity red average": av_int_red,
            "intensity green average": av_int_green,
        }

    def get_statistics(self):
        return self.stat

    def get_data(self, console=False, formatted=False):
        """
        Method to obtain the data stored in this handler

        Keyword arguments:
        console(bool):Determines if results are printed to the console
        (default:True)
        formatted(bool): Determines if the output should be formatted
        (default:True)

        Returns:
        dict: A dictionary containing all stored information
        """
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
