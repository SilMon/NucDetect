"""
Created 09.04.2019
@author Romano Weiss
"""
from __future__ import annotations

import datetime
import os
import hashlib

import math
import piexif
import numpy as np
import time
from typing import Union, Dict, List, Tuple, Any
from scipy import ndimage as ndi
from skimage import io
from skimage.feature import canny, blob_log
from skimage.filters import sobel
from skimage.draw import circle
from skimage.morphology.binary import binary_opening
from Nucleus.core.ROIHandler import ROIHandler
from Nucleus.core.ROI import ROI
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt


class Detector:
    FORMATS = [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".bmp"
    ]

    def __init__(self, settings: Dict[str, Any] = None, logging: bool = None):
        """
        Constructor of the detector class
        :param settings: The settings of the class
        :param logging: Indicates if analysis messages should be printed to the console
        """
        self.settings = settings if settings is not None else {
            "ass_qual": True,
            "names": "Red;Green;Blue",
            "main_channel": 2,
            "min_foc_area": 9
        }
        self.logging = logging

    def analyse_image(self, path, logging=True) -> Dict[str, Union[ROIHandler, np.ndarray, Dict[str, str]]]:
        """
        Method to extract rois from the image given by path

        :param path: The URL of the image
        :param logging: Enables logging
        :return: The analysis results as dict
        """
        logging = logging if self.logging is None else self.logging
        imgdat = Detector.get_image_data(path)
        imgdat["id"] = Detector.calculate_image_id(path)
        image = Detector.load_image(path)
        names = self.settings["names"].split(";")
        main_channel = self.settings["main_channel"]
        if imgdat["channels"] != 3:
            if imgdat["channels"] == 1:
                raise ValueError("Detector class can only analyse multichannel images, not grayscale!")
            elif imgdat["channels"] > 3:
                names.extend(range(imgdat["channels"]))
        # Channel extraction
        channels = Detector.get_channels(image)
        # Channel thresholding
        thresh_chan = Detector.threshold_channels(channels, main_channel)
        # ROI detection
        rois = Detector.extract_rois(channels, thresh_chan, names, main_map=main_channel, logging=logging)
        handler = ROIHandler(ident=imgdat["id"])
        for roi in rois:
            handler.add_roi(roi)
        imgdat["handler"] = handler
        return imgdat

    @staticmethod
    def extract_rois(channels: List[np.ndarray], bin_maps: List[np.ndarray],
                     names: List[str], main_map: int = 2, logging: bool = True) -> List[ROI]:
        """
        Method to extract ROI objects from the given binary maps

        :param channels: List of the channels to detect rois on
        :param bin_maps: A list of binary maps of the channels
        :param names: The names associated with each channel
        :param main_map: Index of the map containing nuclei
        :param logging: Indicates if messages should be printed to console
        :return: A list of all detected roi
        """
        # First round of ROI detection
        s0 = time.time()
        rois = []
        markers, lab_nums = Detector.perform_labelling(bin_maps)
        main_markers = markers[main_map]
        main = [None] * (lab_nums[main_map] + 1)
        # Extraction of main rois
        for y in range(len(main_markers)):
            for x in range(len(main_markers[0])):
                lab = main_markers[y][x]
                if lab != 0:
                    if main[lab] is None:
                        roi = ROI(channel=names[main_map])
                        roi.add_point((x, y), int(channels[main_map][y][x]))
                        main[lab] = roi
                    else:
                        main[lab].add_point((x, y), int(channels[main_map][y][x]))
        for ind in range(len(markers)):
            if ind != main_map:
                temprois = [None] * (lab_nums[ind] + 1)
                for y in range(len(markers[ind])):
                    for x in range(len(markers[ind][0])):
                        lab = markers[ind][y][x]
                        if lab != 0:
                            if temprois[lab] is None:
                                roi = ROI(channel=names[ind], main=False)
                                roi.add_point((x, y), int(channels[ind][y][x]))
                                temprois[lab] = roi
                                if main_markers[y][x] != 0:
                                    roi.associated = main[main_markers[y][x]]
                            else:
                                if temprois[lab].associated is None:
                                    if main_markers[y][x] != 0:
                                        roi.associated = main[main_markers[y][x]]
                                temprois[lab].add_point((x, y), int(channels[ind][y][x]))
                del temprois[0]
                rois.extend(temprois)
        # Second round of ROI detection
        markers, lab_nums = Detector.detect_blobs(channels, main_channel=main_map)
        for ind in range(len(markers)):
            temprois = [None] * (lab_nums[ind] + 1)
            for y in range(len(markers[ind])):
                for x in range(len(markers[ind][0])):
                    lab = markers[ind][y][x]
                    if lab != 0:
                        if temprois[lab] is None:
                            roi = ROI(channel=names[ind], main=False)
                            roi.add_point((x, y), int(channels[ind][y][x]))
                            temprois[lab] = roi
                            if main_markers[y][x] != 0:
                                roi.associated = main[main_markers[y][x]]
                        else:
                            if temprois[lab].associated is None:
                                if main_markers[y][x] != 0:
                                    roi.associated = main[main_markers[y][x]]
                            temprois[lab].add_point((x, y), int(channels[ind][y][x]))
            del temprois[0]
            rois.extend(temprois)
        del main[0]
        rois.extend(main)
        Detector.log("Analysis time: {:.4f}".format(time.time() - s0), logging)
        Detector.perform_roi_quality_check(rois, logging=logging)
        return rois

    @staticmethod
    def perform_roi_quality_check(rois: List[ROI], max_focus_overlapp: float = .75, min_dist: int = 45,
                                  min_thresh: int = 25, max_thresh: int = 60, min_main_area: int = 400,
                                  min_foc_area: int = 5, max_main_area: int = 16000, ws_line: bool = False,
                                  logging: bool = True) -> None:
        """
        Method to check detected rois for their quality. Changes ROI in place.

        :param min_foc_area: The minimal area of a focus
        :param rois: A list of detected rois
        :param max_focus_overlapp: The threshold used to determine if two rois are considered duplicates
        :param min_dist: The minimal distance between 2 nuclei
        :param min_thresh: The lower percentile to check for oversegmentation
        :param max_thresh: The upper percentile to check for undersegmentation
        :param min_main_area: The mininmal area of a nucleus
        :param max_main_area: The maximal area of a nucleus
        :param ws_line: Should the separation line for ws separated nuclei be drawn?
        :param logging: Enables logging
        :return: None
        """
        rem_list = []
        temp = []
        main = []
        foci = []
        s7 = time.time()
        for roi in rois:
            if roi.main:
                main.append(roi)
                temp.append(len(roi))
            else:
                foci.append(roi)
        Detector.log(f"Detected nuclei:{len(main)}\nDetected foci: {len(foci)}", logging)
        ws_list = []
        Detector.log("Checking for very small nuclei", logging)
        # Remove very small nuclei
        Detector.log(f"Time: {time.time() - s7:4f}\nRemove not associated foci", logging)
        s8 = time.time()
        # Remove foci that are either unassociated or whose nucleus was deleted
        maincop = main.copy()
        maincop.extend(foci)
        ass = {key: value for key, value in Detector.create_association_map(maincop).items() if
               len(key) > min_main_area}
        Detector.log(f"Assmap-Creation: {time.time() - s8}", logging)
        mainlen = len(main)
        main = [x for x, _ in ass.items()]
        Detector.log(f"Removed nuclei: {mainlen - len(main)}", logging)
        foclen = len(foci)
        foci = [x for _, focs in ass.items() for x in focs if x.associated is not None]
        # Remove very small foci
        foci = [x for x in foci if len(x) > min_foc_area]
        Detector.log(f"Removed foci: {foclen - len(foci)}\nTime: {time.time() - s8:4f}\nFocus Quality Check",
                     logging)
        # Focus quality check
        s4 = time.time()
        for ind in range(len(foci)):
            focus = foci[ind]
            focdim = focus.calculate_dimensions()
            maxdist = max(focdim["height"], focdim["width"])
            c = focdim["center"]
            if focus not in rem_list:
                for ind2 in range(ind + 1, len(foci)):
                    focus2 = foci[ind2]
                    focdim2 = focus2.calculate_dimensions()
                    maxdist2 = max(focdim2["height"], focdim2["width"])
                    c2 = focdim2["center"]
                    rdist = math.sqrt((c[0] - c2[0]) ** 2 + (c[1] - c2[1]) ** 2) < maxdist / 2 + maxdist2 / 2
                    if focus.ident == focus2.ident and rdist:
                        if focus.calculate_roi_intersection(focus2) >= max_focus_overlapp:
                            if focus2.points >= focus.points:
                                focus2.marked = True
                            else:
                                focus.marked = True
                                break
        foci = [x for x in foci if not x.marked]
        rem_list.clear()
        Detector.log(f"Time: {time.time() - s4:4f}\nNucleus Quality Check", logging)
        # Nucleus quality check
        s1 = time.time()
        ws_num = 0
        res_nuc = 0
        maincop = main.copy()
        maincop.extend(foci)
        ass = Detector.create_association_map(maincop)
        foc_cop = []
        for nucleus in main:
            if len(nucleus) > max_main_area:
                numpy_bin = nucleus.get_as_binary_map()
                numpy_edm = ndi.distance_transform_edt(numpy_bin)
                points_loc_max = peak_local_max(numpy_edm, labels=numpy_bin, indices=False,
                                                min_distance=min_dist)
                points_labels, num_labels = ndi.label(points_loc_max)
                if num_labels > 1:
                    res_nuc += num_labels
                    index = main.index(nucleus)
                    dims = nucleus.calculate_dimensions()
                    offset = (dims["minX"], dims["minY"])
                    points_ws = watershed(-numpy_edm, points_labels, mask=numpy_bin, watershed_line=ws_line)
                    # Extraction of ROI from watershed
                    lab_num = np.amax(points_ws)
                    nucs = [None] * lab_num
                    for i in range(len(points_ws)):
                        for ii in range(len(points_ws[0])):
                            if points_ws[i][ii] > 0:
                                if nucs[points_ws[i][ii] - 1] is None:
                                    nucs[points_ws[i][ii] - 1] = ROI(channel=nucleus.ident)
                                    p = (ii + offset[0], i + offset[1])
                                    nucs[points_ws[i][ii] - 1].add_point(p, nucleus.inten[p])
                                else:
                                    p = (ii + offset[0], i + offset[1])
                                    nucs[points_ws[i][ii] - 1].add_point(p, nucleus.inten[p])
                    ws_list.append((index, nucleus, nucs))
                    ws_num += 1
                else:
                    foc_cop.extend(ass[nucleus])
            else:
                foc_cop.extend(ass[nucleus])
        Detector.log(f"Watershed applied to {ws_num} nuclei, creating {res_nuc} potential nuclei\n"
                     f"Time: {time.time() - s1:4f}\nAdd newly found nuclei", logging)
        s2 = time.time()
        for t in ws_list:
            centers = []
            for nuc in t[2]:
                centers.append(nuc.calculate_dimensions()["center"])
                main.append(nuc)
            for focus in ass[t[1]]:
                c = focus.calculate_dimensions()["center"]
                min_dist = [math.sqrt((c[0] - c2[0]) ** 2 + (c[1] - c2[1]) ** 2) for c2 in centers]
                focus.associated = t[2][min_dist.index(min(min_dist))]
                foc_cop.append(focus)
            main.remove(t[1])
        rois.clear()
        # TODO Reihenfolge wichtig, ausbessern da undynamisch
        rois.extend(foc_cop)
        rois.extend(main)
        Detector.log(f"Time: {time.time() - s2:4f}\nTotal Quality Check Time: {time.time() - s7}", logging)

    @staticmethod
    def create_association_map(rois: List[ROI]) -> Dict[ROI, List[ROI]]:
        """
        Method to create a dictionary which associates main roi (as keys) with their foci (as list of ROI)

        :param rois: The list of roi to create the association map from
        :return: The association map as dict
        """
        ass = {x: [] for x in rois if x.main}
        for roi in rois:
            if roi.associated is not None:
                ass[roi.associated].append(roi)
        return ass

    @staticmethod
    def detect_blobs(channels: List[np.ndarray], main_channel: int = -1, min_sigma: Union[int, float] = 1,
                     max_sigma: Union[int, float] = 5, num_sigma: int = 10,
                     threshold: float = .1) -> Tuple[List[np.ndarray], List[int]]:
        """
        Method to detect blobs in the given channels using the blob_log method

        :param channels: A list of all channels
        :param main_channel: The index of the main channel
        :param min_sigma: the minimum standard deviation for Gaussian kernel
        :param max_sigma: the maximum standard deviation for Gaussian kernel
        :param num_sigma: The number of intermediate values of standard deviations
        to consider between min_sigma and max_sigma
        :param threshold: The absolute lower bound for scale space maxima
        :return: A tuple of the detected blob-maps and the respective numbers of detected blobs
        """
        blob_maps = []
        blob_nums = []
        for ind in range(len(channels)):
            if ind != main_channel:
                blobs = blob_log(channels[ind], min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                                 threshold=threshold, exclude_border=False)
                blob_map = Detector.create_blob_map(channels[ind].shape, blobs)
                blob_num = len(blobs)
                blob_maps.append(blob_map)
                blob_nums.append(blob_num)
            else:
                blob_maps.append(np.zeros(shape=channels[ind].shape))
                blob_nums.append(0)
        return blob_maps, blob_nums

    @staticmethod
    def create_blob_map(shape: Tuple[int, int], blob_dat: np.ndarray) -> np.ndarray:
        """
        Method to create a binary map of detected blobs.

        :param shape: The shape of the blob map
        :param blob_dat: List of all detected blobs
        :return: The created blob map
        """
        map_ = np.zeros(shape, dtype="uint16")
        for ind in range(len(blob_dat)):
            blob = blob_dat[ind]
            rr, cc = circle(blob[0], blob[1], blob[2] * np.sqrt(2) - 0.5, shape=shape)
            map_[rr, cc] = ind + 1
        return map_

    @staticmethod
    def perform_labelling(local_maxima: List[np.ndarray],
                          main_map: int = -1) -> Tuple[List[Union[np.ndarray, int]], List[Union[int, float]]]:
        """
        Method to label a list of maps of local maxima with unique identifiers

        :param main_map: The main map which should not be altered
        :param local_maxima: List of maps of local maxima
        :return: Two lists containing the labelled maps and the numbers of used labels
        """
        labels = []
        label_nums = []
        for i in range(len(local_maxima)):
            if i != main_map:
                label, lab_num = ndi.label(local_maxima[i])
                labels.append(label)
                label_nums.append(lab_num)
            else:
                labels.append(local_maxima[i])
                label_nums.append(np.max(local_maxima[i]))
        return labels, label_nums

    @staticmethod
    def threshold_channels(channels: List[np.ndarray], main_channel: int = 2,
                           main_threshold: float = .05, pad: int = 1) -> List[np.ndarray]:
        """
        Method to threshold the channels to prepare for nuclei and foci detection

        :param channels: The channels of the image as list
        :param main_channel: Index of the channel associated with nuclei (usually blue -> 2)
        :param main_threshold: Global threshold to apply onto the main channel in %
        :param pad: Padding used to account for edge areas (should be 1)
        :return: The thresholded channels
        """
        thresh: List[Union[None, np.ndarray]] = [None] * len(channels)
        edges_main = np.pad(channels[main_channel], pad_width=pad,
                            mode="constant", constant_values=0)
        # Threshold image
        hmax = np.amax(edges_main)
        hmin = np.amin(edges_main)
        threshold = hmin + round(main_threshold * hmax)
        ch_main_bin = binary_opening(ndi.binary_fill_holes(edges_main > threshold), selem=np.ones((20, 20)))
        det: List[Tuple[int, int]] = []
        numpy_edm = ndi.distance_transform_edt(ch_main_bin)
        points_loc_max = peak_local_max(numpy_edm, labels=ch_main_bin, indices=False,
                                        min_distance=45, exclude_border=False)
        points_ws, num_labels = ndi.label(points_loc_max)
        ch_main_bin = watershed(-numpy_edm, points_ws, mask=ch_main_bin, watershed_line=False)
        # Removal of added padding
        ch_main_bin = np.array([x[pad:-pad] for x in ch_main_bin[pad:-pad]])
        truth_table = np.zeros(shape=ch_main_bin.shape, dtype=bool)
        for i in range(len(ch_main_bin)):
            for ii in range(len(ch_main_bin[0])):
                if ch_main_bin[i][ii] and not truth_table[i][ii]:
                    nuc = Detector.adjusted_flood_fill((i, ii), ch_main_bin, truth_table)
                    if nuc is not None:
                        det.append(nuc)
        thresh[main_channel] = ch_main_bin
        for ind in range(len(channels)):
            if ind != main_channel:
                thresh[ind] = Detector.calculate_local_region_threshold(det, channels[ind])
        return thresh

    @staticmethod
    def calculate_local_region_threshold(nuclei: List[Tuple[int, int]],
                                         channel: np.ndarray) -> np.ndarray:
        """
        Method to threshold nuclei for foci extraction

        :param nuclei: The points of the nucleus as list
        :param channel: The corresponding channel
        :return: The foci map for the nucleus
        """
        chan = np.zeros(shape=channel.shape)
        for nuc in nuclei:
            thresh = []
            for p in nuc:
                thresh.append((p, channel[p[0]][p[1]]))
            if thresh:
                thresh_np, offset = Detector.create_numpy_from_point_list(thresh)
                edges = Detector.detect_edges(thresh_np)
                if np.max(edges) > 0:
                    chan_fill = ndi.binary_fill_holes(edges)
                    chan_open = ndi.binary_opening(chan_fill)
                    if np.max(chan_open) > 0:
                        Detector.imprint_data_into_channel(chan, chan_open, offset)
        return chan

    @staticmethod
    def detect_edges(channel: np.ndarray, sigma: Union[int, float] = 2,
                     low_threshold: Union[int, float, None] = None,
                     high_threshold: Union[int, float, None] = None) -> np.ndarray:
        """
        Privat method to detect the edges of the given channel via the canny operator.

        :param channel: The channel to detect the edges on
        :param sigma: Standard deviation of the gaussian kernel
        :param low_threshold: Lower bound for hysteresis thresholding
        :param high_threshold: Upper bound for hysteresis thresholding
        :return: The edge map
        """
        return canny(channel.astype("float64"), sigma, low_threshold, high_threshold)

    @staticmethod
    def adjusted_flood_fill(starting_point: Tuple[int, int],
                            region_map: np.ndarray,
                            truth_table: List[bool]) -> Union[List[Tuple[int, int]], None]:
        """
        Adjusted implementation of flood fill to extract a list of connected points

        :param starting_point: The point to start flood fill from
        :param region_map: The region to check
        :param truth_table: The table to indicate already checked points
        :param labelled: Indicates if the region map is binary or contain labelled areas
        :return: A list of connected points
        """
        points = [
            starting_point
        ]
        height = len(region_map)
        width = len(region_map[0])
        nuc = []
        while points:
            p = points.pop()
            y = p[0]
            x = p[1]
            if (x >= 0) and (x < width) and (y >= 0) and (y < height):
                if region_map[y][x] and not truth_table[y][x]:
                    truth_table[y][x] = True
                    nuc.append(p)
                    points.append((y + 1, x))
                    points.append((y, x + 1))
                    points.append((y, x - 1))
                    points.append((y - 1, x))
        return nuc if nuc else None

    @staticmethod
    def imprint_data_into_channel(channel: np.ndarray, data: np.ndarray, offset: int) -> None:
        """
        Method to transfer the information stored in data into channel. Works in place

        :param channel: The image channel as ndarray
        :param data: The data to transfer as ndarray
        :param offset: The offset of the data
        :return: None
        """
        for i in range(len(data)):
            for ii in range(len(data[0])):
                if data[i][ii] != 0:
                    channel[i + offset[0]][ii + offset[1]] = data[i][ii]

    @staticmethod
    def get_channels(img: np.ndarray) -> List[np.ndarray]:
        """
        Method to extract the channels of the given image

        :param img: The image as ndarray
        :return: A list of all channels
        """
        channels = []
        for ind in range(img.shape[2]):
            channels.append(img[..., ind])
        return channels

    @staticmethod
    def create_numpy_from_point_list(lst: List[Tuple[int, int]]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Method to create a ndarray from a list of points

        :param lst: The point list
        :return: The created ndarray and the offset as tuple
        """
        min_x = min([x[0][1] for x in lst])
        max_x = max([x[0][1] for x in lst])
        min_y = min([x[0][0] for x in lst])
        max_y = max([x[0][0] for x in lst])
        y_dist = max_y - min_y + 1
        x_dist = max_x - min_x + 1
        numpy = np.zeros((y_dist, x_dist), dtype=np.uint8)
        for p in lst:
            numpy[p[0][0] - min_y, p[0][1] - min_x] = p[1]
        return numpy, (min_y, min_x)

    @staticmethod
    def get_image_data(path: str) -> Dict[str, Union[int, float, str]]:
        """
        Method to extract relevant metadata from an image

        :param path: The URL of the image
        :return: The extracted metadata as dict
        """
        filename, file_extension = os.path.splitext(path)
        img = Detector.load_image(path)
        if file_extension in (".tiff", ".tif", ".jpg"):
            tags = piexif.load(path)
            image_data = {
                "datetime": tags["0th"].get(piexif.ImageIFD.DateTime,
                                            datetime.datetime.fromtimestamp(os.path.getctime(path))),
                "height": tags["0th"].get(piexif.ImageIFD.ImageLength, img.shape[0]),
                "width": tags["0th"].get(piexif.ImageIFD.ImageWidth, img.shape[1]),
                "x_res": tags["0th"].get(piexif.ImageIFD.XResolution, -1),
                "y_res": tags["0th"].get(piexif.ImageIFD.YResolution, -1),
                "channels": tags["0th"].get(piexif.ImageIFD.SamplesPerPixel, 3),
                "unit": tags["0th"].get(piexif.ImageIFD.ResolutionUnit, 2)
            }
        else:
            # TODO Dialog for resolution, number of channels and unit
            image_data = {
                "datetime": os.path.getctime(path),
                "heigth": img.shape[0],
                "width": img.shape[1],
                "x_res": -1,
                "y_res": -1,
                "channels": 1 if len(img.shape) == 2 else 3,
                "unit": 2
            }
        return image_data

    @staticmethod
    def load_image(path:str) -> np.ndarray:
        """
        Method to load an image given by path. Method will only load image formats specified by Detector.FORMATS

        :param path: The URL of the image
        :return: The image as ndarray
        """
        if os.path.splitext(path)[1] in Detector.FORMATS:
            return io.imread(path)
        else:
            raise Warning("Unsupported image format ->{}!".format(os.path.splitext(path)[1]))

    @staticmethod
    def calculate_image_id(path: str) -> int:
        """
        Method to calculate the md5 hash sum of the image described by path

        :param path: The URL of the image
        :return: The md5 hash sum as hex
        """
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def log(message: str, state: bool = True):
        """
        Method to log messages to the console

        :param message: The message to log
        :param state: Enables logging
        :return: None
        """
        if state:
            print(message)
