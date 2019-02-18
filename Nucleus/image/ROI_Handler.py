"""
Created on 06.10.2018

@author: Romano Weiss
"""
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import QPoint, Qt, QRectF
from PyQt5.QtGui import QPainter, QStaticText, QTextOption, QColor, QBrush
from scipy import ndimage as ndi
from skimage.draw import circle_perimeter
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from NucDetect.image import Channel
from NucDetect.image.ROI import ROI


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
        self.blue_markers = None
        self.red_markers = None
        self.green_markers = None
        self.blue_lab_nums = None
        self.red_lab_nums = None
        self.green_lab_nums = None
        self.blue_orig = None
        self.red_orig = None
        self.green_orig = None
        self.id = ident
        self.blue_name = None
        self.red_name = "Red Foci"
        self.green_name = "Green Foci"
        self.nuclei = None
        self.green = None
        self.red = None
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

    def set_data(self, markers, lab_nums, orig):
        self.blue_orig = orig[0]
        self.red_orig = orig[1]
        self.green_orig = orig[2]
        self.blue_markers = markers[0]
        self.red_markers = markers[1]
        self.green_markers = markers[2]
        self.blue_lab_nums = lab_nums[0]
        self.red_lab_nums = lab_nums[1]
        self.green_lab_nums = lab_nums[2]

    def analyse_image(self):
        """
        Method to analyse an image according to the given data
        """
        # Analysis of the blue channel
        self.nuclei = self.extract_roi_from_channel(
            channel=self.blue_markers,
            orig=self.blue_orig,
            num_labels=self.blue_lab_nums
        )
        self.red = self.extract_roi_from_channel(
            channel=self.red_markers,
            orig=self.red_orig,
            num_labels=self.red_lab_nums,
            channel_type=Channel.RED
        )
        self.green = self.extract_roi_from_channel(
            channel=self.green_markers,
            orig=self.green_orig,
            num_labels=self.green_lab_nums,
            channel_type=Channel.GREEN
        )
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
        rem_list = []
        add_list = []
        for nucleus in self.nuclei:
            if nucleus is not None:
                nuc_stat = nucleus.calculate_statistics()
                if nuc_stat["area"] <= self.stat["area average"]*0.2:
                    self.nuclei.remove(nucleus)
        self.calculate_statistics()
        for nucleus in self.nuclei:
            if nucleus is not None:
                nuc_stat = nucleus.calculate_statistics()
                if nuc_stat["area"] >= self.stat["area average"] * 1.2:
                    index = self.nuclei.index(nucleus)
                    points, offset = self.create_numpy_from_point_list(nucleus.points)
                    gr_rois = []
                    re_rois = []
                    for gr in nucleus.green:
                        gr_rois.append(gr)
                    for re in nucleus.red:
                        re_rois.append(re)
                    points_edm = ndi.distance_transform_edt(points)
                    points_loc_max = peak_local_max(points_edm, labels=points, indices=False,
                                                    footprint=np.ones((91, 91)))
                    points_labels, num_labels = ndi.label(points_loc_max)
                    points_ws = watershed(-points_edm, points_labels, mask=points)
                    nucs = self.extract_roi_from_channel(points_ws, self.blue_orig, num_labels, offset=offset)
                    # Check which of the foci belongs to which nucleus
                    gre_rem = []
                    red_rem = []
                    for nuc in nucs:
                        gre_rem.clear()
                        red_rem.clear()
                        if nuc is not None:
                            for gre in gr_rois:
                                if gre is not None:
                                    if nuc.add_roi(gre):
                                        gre_rem.append(gre)
                            for red in re_rois:
                                if red is not None:
                                    if nuc.add_roi(red):
                                        red_rem.append(red)
                        gr_rois = [x for x in gr_rois if x not in gre_rem]
                        re_rois = [x for x in re_rois if x not in red_rem]
                    rem_list.append(nucleus)
                    for nuc in nucs:
                        add_list.append((index, nuc))
        for n in add_list:
            self.nuclei.insert(n[0], n[1])
        for rem_nuc in rem_list:
            self.nuclei.remove(rem_nuc)
        for nucleus in self.nuclei:
            nucleus.perform_foci_quality_check()

    def extract_roi_from_channel(self, channel, orig, num_labels, channel_type=Channel.BLUE, offset=None):
        rois = [None] * (num_labels+1)
        for y in range(len(channel)):
            for x in range(len(channel[0])):
                lab = channel[y][x]
                if lab != 0:
                    if rois[lab] is None:
                        roi = ROI(chan=channel_type)
                        if offset is None:
                            roi.add_point((x, y), orig[y][x])
                        else:
                            roi.add_point((x + offset[1], y + offset[0]), orig[y + offset[0]][x + offset[1]])
                        rois[lab] = roi
                    else:
                        if offset is None:
                            rois[lab].add_point((x, y), orig[y][x])
                        else:
                            rois[lab].add_point((x + offset[1], y + offset[0]), orig[y + offset[0]][x + offset[1]])
        del rois[0]
        return rois

    def create_numpy_from_point_list(self, lst):
        min_y = 0xffffffff
        max_y = 0
        min_x = 0xffffffff
        max_x = 0
        for point in lst:
            if point[0] > max_x:
                max_x = point[0]
            if point[0] < min_x:
                min_x = point[0]
            if point[1] > max_y:
                max_y = point[1]
            if point[1] < min_y:
                min_y = point[1]
        y_dist = max_y - min_y + 1
        x_dist = max_x - min_x + 1
        numpy = np.zeros((y_dist, x_dist), dtype=np.uint8)
        for p in lst:
            numpy[p[1]-min_y, p[0]-min_x] = True
        return numpy, (min_y, min_x)

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

    def create_result_image_as_mplfigure(self, img_array, show=True):
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

    def create_result_image_as_qtimage(self, img_array):
        i = Image.fromarray(self._draw_roi(img_array), mode="RGB")
        img = ImageQt(i)
        painter = QPainter()
        painter.begin(img)
        ind = 0
        for roi in self.nuclei:
            if roi is not None:
                tex = "<font color=\"white\">In:{0:>6}<br>R:{1:>6}<br>G:{2:>6}</font>".format(str(ind),
                                                             str(len(roi.red)),
                                                             str(len(roi.green)))
                dat = roi.get_data()
                c = dat["center"]
                dim = (dat["width"], dat["height"])
                rect = QRectF(c[0]-20, c[1]-20, 40, 40)
                painter.fillRect(rect, QBrush(QColor(100, 100, 0, 40)))
                center = QPoint(c[0]-18, c[1]-18)
                text = QStaticText()
                text.setTextFormat(Qt.RichText)
                opt = QTextOption()
                opt.setAlignment(Qt.AlignLeft)
                opt.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
                text.setTextOption(opt)
                text.setText(tex)
                painter.drawStaticText(center, text)
                ind += 1
        painter.end()
        return img

    def calculate_statistics(self):
        """
        Method to calculate statistics about the saved ROIs
        :return: dict -- A dictonary containing the data
        """
        # TODO aufräumen
        area = []
        num_nuc = 0
        num_red = []
        num_empty = 0
        int_red = []
        int_red_tot = []
        num_green = []
        int_green = []
        int_green_tot = []

        for nucleus in self.nuclei:
            if nucleus is not None:
                temp_stat = nucleus.calculate_statistics()
                area.append(temp_stat["area"])
                num_red.append(temp_stat["red_roi"])
                int_red.append(temp_stat["red_av_int"])
                num_green.append(temp_stat["green_roi"])
                int_green.append(temp_stat["green_av_int"])
                int_red_tot.append(temp_stat["red_int"])
                int_green_tot.append((temp_stat["green_int"]))
                if len(int_red_tot[-1]) is 0 and len(int_green_tot[-1]) is 0:
                    num_empty += 1
        self.stat = {
            "area": area,
            "area average": np.average(area),
            "area median": np.median(area),
            "area std": np.std(area),
            "number": len(self.nuclei),
            "empty": num_empty,
            "number red": num_red,
            "number green": num_green,
            "number red average": np.average(num_red),
            "number red std": np.std(num_red),
            "number green average": np.average(num_green),
            "number green std": np.std(num_green),
            "intensity red": int_red,
            "intensity red total": int_red_tot,
            "intensity green": int_green,
            "intensity green total": int_green_tot,
            "intensity red average": np.average(int_red),
            "intensity red std": np.std(int_red),
            "intensity green average": np.average(int_green),
            "intensity green std": np.std(int_green)
        }

    def get_statistics(self):
        if len(self.stat) == 0:
            self.calculate_statistics()
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
