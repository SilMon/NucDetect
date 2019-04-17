"""
Created 09.04.2019
@author Romano Weiss
"""
import os
import hashlib

import piexif
import numpy as np
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

    def __init__(self, settings=None):
        self.settings = settings if settings is not None else {
            "ass_qual": True,
            "names": "Red;Green;Blue",
            "main_channel": 2
        }

    def analyse_image(self, path, main_channel=2):
        """
        Method to extract rois from the image given by path
        :param path: The URL of the image
        :param main_channel: Index of the channel associated with nuclei
        :return: The analysis results as dict
        """
        imgdat = Detector.get_image_data(path)
        imgdat["id"] = Detector.calculate_image_id(path)
        image = Detector.load_image(path)
        names = self.settings["names"].split(";")
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
        rois = Detector.extract_rois(channels, thresh_chan, names, main_map=main_channel)
        handler = ROIHandler(ident=imgdat["id"])
        for roi in rois:
            handler.add_roi(roi)
        imgdat["handler"] = handler
        return imgdat

    @staticmethod
    def extract_rois(channels, bin_maps, names, main_map=2):
        """
        Method to extract ROI objects from the given binary maps
        :param channels: List of the channels to detect rois on
        :param bin_maps: A list of binary maps of the channels
        :param names: The names associated with each channel
        :param main_map: Index of the map containing nuclei
        :return: A list of all detected roi
        """
        # First round of ROI detection
        markers, lab_nums = Detector.perform_labelling(bin_maps)
        rois = []
        for ind in range(len(markers)):
            temprois = [None] * (lab_nums[ind] + 1)
            for y in range(len(markers[ind])):
                for x in range(len(markers[ind][0])):
                    lab = markers[ind][y][x]
                    if lab != 0:
                        if temprois[lab] is None:
                            # TODO
                            roi = ROI(channel=names[ind], main=True if ind == main_map else False)
                            roi.add_point((x, y), int(channels[ind][y][x]))
                            temprois[lab] = roi
                        else:
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
                        else:
                            temprois[lab].add_point((x, y), int(channels[ind][y][x]))
            del temprois[0]
            rois.extend(temprois)
        Detector.perform_roi_quality_check(rois)
        return rois

    @staticmethod
    def perform_roi_quality_check(rois, max_focus_overlapp=.75, main_threshold=5, min_dist=45,
                                  min_thresh=25, max_thresh=60):
        """
        Method to check detected rois for their quality.
        :param rois: A list of detected rois
        :param max_focus_overlapp: The threshold used to determine if two rois are considered duplicates
        :param main_threshold: The threshold for nucleus detection
        :param min_dist: The minimal distance between 2 nuclei
        :param min_thresh: The lower percentile to check for oversegmentation
        :param max_thresh: The upper percentile to check for undersegmentation
        :return: None
        """
        rem_list = []
        temp = []
        main = []
        foci = []
        for roi in rois:
            if roi.main:
                main.append(roi)
                temp.append(len(roi))
            else:
                foci.append(roi)
        min_main_area = np.percentile(temp, min_thresh)
        max_main_area = np.percentile(temp, max_thresh)
        ws_list = []
        # Nucleus quality check
        for nucleus in main:
            if len(nucleus) > max_main_area:
                numpy_bin = nucleus.get_as_binary_map()
                numpy_edm = ndi.distance_transform_edt(numpy_bin)
                points_loc_max = peak_local_max(numpy_edm, labels=numpy_bin, indices=False,
                                                min_distance=min_dist)
                points_labels, num_labels = ndi.label(points_loc_max)
                if num_labels > 1:
                    index = main.index(nucleus)
                    dims = nucleus.calculate_dimensions()
                    offset = (dims["minX"], dims["minY"])
                    points_ws = watershed(-numpy_edm, points_labels, mask=numpy_bin)
                    # Extraction of ROI from watershed
                    lab_num = np.amax(points_ws)
                    print(lab_num)
                    nucs = [None] * lab_num
                    tmap = np.zeros(shape=points_ws.shape)
                    # TODO Schleife fixen
                    for i in range(len(points_ws)):
                        for ii in range(len(points_ws[0])):
                            if points_ws[i][ii] and not tmap[i][ii]:
                                if nucs[points_ws[i][ii]] is not None:
                                    nucs[points_ws].append((i, ii))
                                else:
                                    nucs[points_ws[i][ii]] = [(i, ii)]
                                nuc = Detector.adjusted_flood_fill((i, ii), points_ws, tmap)
                    if nuc is not None:
                        tnuc = ROI(channel=nucleus.ident)
                        tnuc.derive_from_roi(nucleus, nuc, offset)
                        nucs.append(tnuc)
                    print(nucs)
                    ws_list.append((index, nucleus, nucs))
        print("Length before: {}".format(len(main)))
        for t in ws_list:
            for nuc in t[2]:
                main.insert(t[0], nuc)
            main.remove(t[1])
        print("Length after: {}".format(len(main)))
        for nucleus in main:
            if len(nucleus) < min_main_area:
                rem_list.append(nucleus)
        main = [x for x in main if x not in rem_list]
        for rem in rem_list:
            rois.remove(rem)
        rem_list.clear()
        # Create nucleus-focus associations
        for nucleus in main:
            for focus in foci:
                if focus not in rem_list:
                    intersect = nucleus.calculate_roi_intersection(focus)
                    if intersect > 0.95:
                        focus.associated = nucleus
                        rem_list.append(focus)
        rem_list.clear()
        # Focus quality check
        for ind in range(len(foci)):
            if focus not in rem_list:
                focus = foci[ind]
                for ind2 in range(ind+1, len(foci)):
                    focus2 = foci[ind2]
                    if focus.calculate_roi_intersection(focus2) >= max_focus_overlapp and \
                            focus.ident == focus2.ident:
                        rem_list.append(focus2)
        for rem in rem_list:
            rois.remove(rem)

    @staticmethod
    def detect_blobs(channels, main_channel=-1, min_sigma=1, max_sigma=5, num_sigma=10, threshold=.1):
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
        # TODO
        blob_maps = []
        blob_nums = []
        for ind in range(len(channels)):
            if ind != main_channel:
                blobs = blob_log(channels[ind], min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                                 threshold=threshold)
                blob_map = Detector.create_blob_map(channels[ind].shape, blobs)
                blob_num = len(blobs)
                blob_maps.append(blob_map)
                blob_nums.append(blob_num)
            else:
                blob_maps.append(np.zeros(shape=channels[ind].shape))
                blob_nums.append(0)
        return blob_maps, blob_nums

    @staticmethod
    def create_blob_map(shape, blob_dat):
        """
        Method to create a binary map of detected blobs.
        :param shape: The shape of the blob map as tuple
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
    def perform_labelling(local_maxima):
        """
        Method to label a list of maps of local maxima with unique identifiers
        :param local_maxima: List of maps of local maxima
        :return: Two lists containing the labelled maps and the numbers of used labels
        """
        labels = []
        label_nums = []
        for loc_max in local_maxima:
            label, lab_num = ndi.label(loc_max)
            labels.append(label)
            label_nums.append(lab_num)
        return labels, label_nums

    @staticmethod
    def threshold_channels(channels, main_channel=2, main_threshold=5):
        """
        Method to threshold the channels to prepare for nuclei and foci detection
        :param channels: The channels of the image as list
        :param main_channel: Index of the channel associated with nuclei (usually blue -> 2)
        :param main_threshold: Global threshold to apply onto the main channel
        :return: The thresholded channels
        """
        thresh = [None]*len(channels)
        edges_main = (sobel(channels[main_channel]) * 255).astype("uint8")
        det = []
        ch_main_bin = edges_main > main_threshold
        ch_main_bin = ndi.binary_fill_holes(ch_main_bin)
        ch_main_bin = binary_opening(ch_main_bin)
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
    def calculate_local_region_threshold(nuclei, channel):
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
    def detect_edges(channel, sigma=2, low_threshold=None, high_threshold=None):
        """
        Privat method to detect the edges of the given channel via the canny operator.
        :param channel: The channel to detect the edges on
        :param sigma: Standard deviation of the gaussian kernel
        :param low_threshold: Lower bound for hysteresis thresholding
        :param high_threshold: Upper bound for hysteresis thresholding
        :return: The edge map as ndarray
        """
        return canny(channel.astype("float64"), sigma, low_threshold, high_threshold)

    @staticmethod
    def adjusted_flood_fill(starting_point, region_map, truth_table):
        """
        Adjusted implementation of flood fill to extract a list of connected points
        :param starting_point: The point to start flood fill from
        :param region_map: The region to check
        :param truth_table: The table to indicate already checked points
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
    def imprint_data_into_channel(channel, data, offset):
        """
        Method to transfer the information stored in data into channel
        :param channel: The image channel as ndarray
        :param data: The data to transfer as ndarray
        :param offset: The offset of the data
        :return: None
        """
        for i in range(len(data)):
            for ii in range(len(data[0])):
                if data[i][ii] != 0:
                    channel[i + offset[0]][ii+offset[1]] = data[i][ii]

    @staticmethod
    def get_channels(img):
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
    def create_numpy_from_point_list(lst):
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
            numpy[p[0][0]-min_y, p[0][1]-min_x] = p[1]
        return numpy, (min_y, min_x)

    @staticmethod
    def get_image_data(path):
        """
        Method to extract relevant metadata from an image
        :param path: The URL of the image
        :return: The extracted metadata as dict
        """
        tags = piexif.load(path)
        image_data = {
            "datetime": tags["0th"][piexif.ImageIFD.DateTime],
            "height": tags["0th"][piexif.ImageIFD.ImageLength],
            "width": tags["0th"][piexif.ImageIFD.ImageWidth],
            "x_res": tags["0th"][piexif.ImageIFD.XResolution],
            "y_res": tags["0th"][piexif.ImageIFD.YResolution],
            "channels": tags["0th"][piexif.ImageIFD.SamplesPerPixel]
        }
        if piexif.ImageIFD.ResolutionUnit in tags["0th"]:
            image_data["unit"] = tags["0th"][piexif.ImageIFD.ResolutionUnit]
        else:
            image_data["unit"] = 2
        return image_data

    @staticmethod
    def load_image(path):
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
    def calculate_image_id(path):
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

