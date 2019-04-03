"""
Created on 15.10.2018

@author: Romano Weiss
"""
import hashlib
import os
import pickle
import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import io
from skimage.draw import circle, circle_perimeter
from skimage.exposure import histogram
from skimage.feature import canny, blob_log
from skimage.filters import sobel, laplace
from skimage.morphology.binary import binary_opening

from Nucleus.image import Channel
from Nucleus.image.ROI_Handler import ROI_Handler


class Detector:
    """
    Class to detect intranuclear proteins in provided fluorescence images
    """

    FORMATS = [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".bmp"
    ]

    def __init__(self):
        """
        Constructor to initialize the detector

        Keyword parameters:
        images(dict): A dict containing the images to analyze. It must have
        following structure dict[url_to_image] = image (as 2D array)
        """
        self.images = {}
        self.snaps = {}
        self.save_snapshots = True
        self.settings = {
            "ass_qual": True,
            "snaps_num": 2,
            "snaps_save": 1,
        }
        self.keys = []

    def clear(self):
        """
        Method to clean all loaded images
        :return: None
        """
        self.images.clear()
        self.snaps.clear()
        self.keys.clear()

    def load_image(self, url, names=None):
        """
        Method to add an image to the processing queue.

        Keyword arguments:
        url (str) -- The path to the image file
        names (tuple) -- Used to rename the image channels. Structure:
        (blue name, red name, green name)

        Returns:
        str -- The md5 value used as id for the image. Needed to obtain the
        results of the processing
        """
        if os.path.splitext(url)[1] in Detector.FORMATS:
            key = self._calculate_image_id(url)
            self.images[key] = (io.imread(url), names)
            self.keys.append(key)
            return key

    def load_image_folder(self, direct):
        """
        Method to load all images of a specific directory

        :param direct: The path to the directory (str)
        :return: None
        """
        for t in os.walk(direct):
            for file in t[2]:
                self.load_image(os.path.join(t[0], file))

    def _calculate_image_id(self, url):
        """
        Private method to calculate the md5-id of the given image

        Keyword arguments:
        url (str): The url leading to the image file

        Returns:
        str -- The md5 id of the image
        """
        hash_md5 = hashlib.md5()
        with open(url, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def analyse_image(self, which, save_threshold=2):
        """
        Method to analyse a loaded image
        :param which: The md5 hash of the image
        :param save_threshold: Threshold to decide how many analysis data will be hold directly
        by the detector. Is needed to avoid stack overflow when used to analyse a lot images. Default: 2
        :return: None
        """
        start = time.time()
        snap = self.load_snaps(which)
        if snap is not None:
            self.snaps[which] = snap
        else:
            img = self.images[which]
            names = img[1]
            img_array = img[0]
            start = time.time()
            channels = self._get_channels(img_array)
            if self.settings["ass_qual"]:
                qual = self._estimate_image_quality(channels)
            handler = ROI_Handler(ident=which)
            handler.set_names(names)
            thr_chan = self._get_thresholded_channels(channels)
            markers, lab_nums = self._perform_labelling(thr_chan[0])
            blobs, blob_num = self._detect_blobs(channels)
            handler.set_data(markers, blobs, lab_nums=lab_nums, blob_nums=blob_num, orig=channels)
            handler.analyse_image()
            handler.calculate_statistics()
            result_qt = handler.create_result_image_as_qtimage(img_array)
            result_mpl = handler.create_result_image_as_mplfigure(img_array)
            cur_snaps = {
                "id": which,
                "handler": handler,
                "result_qt": result_qt,
                "result_mpl": result_mpl,
                "original": img_array,
                "settings": self.settings,
                "categories": []
            }
            if self.settings["ass_qual"]:
                cur_snaps["quality"] = qual
            if self.settings["snaps_save"]:
                cur_snaps["channel"] = channels
                cur_snaps["binarized"] = thr_chan[0]
                cur_snaps["edges"] = thr_chan[1]
            cur_snaps["time"] = "{0:.3f} sec".format(time.time() - start)
            if len(self.snaps) == save_threshold:
                for x in range(save_threshold-1):
                    self.save_snaps(self.keys[x])
            self.snaps[which] = cur_snaps

    def save_all_snaps(self):
        for key in self.images:
            self.save_snaps(key, clear=False)

    def save_snaps(self, key, clear=True):
        """
        Method to save all acquired data for a specific image.
        :param key: The md5 hash of the image
        :param clear: If true the current snap dict will be cleared. Needed to avoid stack overflow when anaylsing
        many images
        """
        if key in self.snaps:
            pardir = os.getcwd()
            pathpardir = os.path.join(os.path.dirname(pardir),
                                      r"results/snaps")
            os.makedirs(pathpardir, exist_ok=True)
            pathsnap = os.path.join(pathpardir,
                                      str(key) + ".snap")
            del self.snaps[key]["result_qt"]
            pickle.dump(self.snaps[key], open(pathsnap, "wb"))
            if clear:
                del self.snaps[key]
                del self.images[key]
                self.keys.remove(key)

    def load_snaps(self, key):
        """
        Method to load the saved snaps of an image, identified by key.
        This allows to reuse already acquired data if the settings were not changed
        :param key: The md5 hash of the image
        :return: None if the file does not exist or the settings were changed else +1 if the file could be loaded
        """
        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"results/snaps")
        if not os.path.isdir(pathpardir):
            os.makedirs(pathpardir, exist_ok=True)
            return None
        else:
            pathsnap = os.path.join(pathpardir,
                                    str(key) + ".snap")
            if os.path.isfile(pathsnap):
                snap = pickle.load(open(pathsnap, "rb"))
                if self.settings == snap["settings"]:
                    snap["result_qt"] = snap["handler"].create_result_image_as_qtimage(snap["original"])
                    return snap
                else:
                    return None
            else:
                return None

    def analyse_images(self):
        """
        Method to analyze all images in the processing queue
        """
        if len(self.images) is 0:
            raise ValueError("No images provided!")
        for key, img in self.images.items():
            self.analyse_image(key)

    def show_result(self, which, formatted=True):
        """
        Method to show the result table of the processing of a specific image.
        Table will be shown in the console

        Keyword arguments:
        which (str): The url to the image file
        formatted(bool): Determines if the output should be formatted
        (default:True)
        """
        self.snaps.get(which)["handler"].get_data(console=True, formatted=formatted)

    def create_ouput(self, which, formatted=True):
        """
        Method to show the result table of the processing of a specific image.
        Table will be saved as CSV file

        Keyword arguments:
        which (str): The url to the image file
        formatted(bool): Determines if the output should be formatted
        (default:True)
        """
        self.snaps.get(which)["handler"].get_data(console=False, formatted=formatted)

    def get_output(self, which):
        return self.snaps.get(which)["handler"].get_data()

    def get_statistics(self, which):
        return self.snaps.get(which)["handler"].get_statistics()

    def show_result_image(self, which):
        """
        Method to show the visual representation of the foci detection

        Keyword arguments:
        which (str): The url to the image file
        """
        plt.show(self.snaps.get(which)["result"])

    def get_snapshot(self, which):
        """
        Method to obtain the saved processing data for a specific image

        Keyword arguments:
        which (str): The url to the image file

        Returns:
        dict -- A dict which contains the saved processing snapshots. If no
        snapshots are saved, it will only contain the result image
        """
        return self.snaps.get(which)

    def save_result_image(self, which):
        """
        Method to save the processing image to file

        Keyword arguments:
        which (str): The url to the image file
        """
        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"results/images")
        os.makedirs(pathpardir, exist_ok=True)
        pathresult = os.path.join(pathpardir,
                                  "result - " + str(which) + ".png")
        fig = self.snaps.get(which)["handler"].create_result_image_as_mplfigure(
              self.images.get(which)[0], show=False)
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        fig.savefig(pathresult, dpi=750,
                    bbox_inches="tight",
                    pad_inches=0)
        plt.close()
        return pathresult

    def _get_channels(self, img):
        """
        Private Method to extract the channels of a given image

        Keyword arguments:
        img (2D array): The image to extract the channels from

        Returns:
        tuple  -- A tuple containing the three channels in the order
        blue, red, green
        """
        ch_blue = Channel.extract_channel(img, Channel.BLUE)
        ch_red = Channel.extract_channel(img, Channel.RED)
        ch_green = Channel.extract_channel(img, Channel.GREEN)
        return ch_blue, ch_red, ch_green

    def _get_thresholded_channels(self, channels):
        """
        Private Method to calculate the thresholds for each channel

        Keyword arguments:
        channels (tuple): A tuple containing the three channels in the
        order blue, red, green

        Returns:
        tuple  -- A tuple containing the thresholds for each channel
        in the order blue, red, green
        """
        edges_blue = (sobel(channels[0]) * 255).astype("uint8")
        det = []
        ch_blue_bin = edges_blue > 5
        ch_blue_bin = ndi.binary_fill_holes(ch_blue_bin)
        ch_blue_bin = binary_opening(ch_blue_bin)
        truth_table = np.zeros(shape=ch_blue_bin.shape, dtype=bool)
        for i in range(len(ch_blue_bin)):
            for ii in range(len(ch_blue_bin[0])):
                if ch_blue_bin[i][ii] and not truth_table[i][ii]:
                    nuc = self._flood_fill((i, ii), ch_blue_bin, truth_table)
                    if nuc is not None:
                        det.append(nuc)
        ch_red_bin = self._calculate_local_region_threshold(det, channels[1])
        ch_gre_bin = self._calculate_local_region_threshold(det, channels[2])
        return (ch_blue_bin, ch_red_bin[0], ch_gre_bin[0]), (edges_blue, ch_red_bin[1], ch_gre_bin[1])

    def _flood_fill(self, point, region_map, truth_table):
        points = [
            point
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
                    points.append((y+1, x))
                    points.append((y, x+1))
                    points.append((y, x-1))
                    points.append((y-1, x))
        return nuc if nuc else None

    def _calculate_local_region_threshold(self, nuclei, channel):
        chan = np.zeros(shape=channel.shape)
        edge_map = np.zeros(shape=channel.shape, dtype=bool)
        for nuc in nuclei:
            thresh = []
            for p in nuc:
                thresh.append((p, channel[p[0]][p[1]]))
            if thresh:
                thresh_np, offset = self.create_numpy_from_point_list(thresh)
                edges = self._detect_edges(thresh_np)
                if np.max(edges) > 0:
                    chan_fill = ndi.binary_fill_holes(edges)
                    chan_open = ndi.binary_opening(chan_fill)
                    self._imprint_data_into_channel(chan, chan_open, offset)
                    self._imprint_data_into_channel(edge_map, edges, offset)
        return chan, edge_map

    def _imprint_data_into_channel(self, channel, data, offset):
        for i in range(len(data)):
            for ii in range(len(data[0])):
                if data[i][ii] != 0:
                    channel[i + offset[0]][ii+offset[1]] = data[i][ii]

    def create_numpy_from_point_list(self, lst):
        min_y = 0xffffffff
        max_y = 0
        min_x = 0xffffffff
        max_x = 0
        for point in lst:
            if point[0][1] > max_x:
                max_x = point[0][1]
            if point[0][1] < min_x:
                min_x = point[0][1]
            if point[0][0] > max_y:
                max_y = point[0][0]
            if point[0][0] < min_y:
                min_y = point[0][0]
        y_dist = max_y - min_y + 1
        x_dist = max_x - min_x + 1
        numpy = np.zeros((y_dist, x_dist), dtype=np.uint8)
        for p in lst:
            numpy[p[0][0]-min_y, p[0][1]-min_x] = p[1]
        return numpy, (min_y, min_x)

    def _detect_blobs(self, channels, min_sigma=1, max_sigma=5, num_sigma=10, threshold=.1):
        blobs_red = blob_log(channels[1], min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                             threshold=threshold)
        blobs_green = blob_log(channels[2], min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                               threshold=threshold)
        blobs_red_map = self._create_blob_map(channels[1].shape, blobs_red)
        blobs_green_map = self._create_blob_map(channels[2].shape, blobs_green)
        return (blobs_red_map, blobs_green_map), (len(blobs_red), len(blobs_green))

    def _create_blob_map(self, shape, blob_dat):
        map = np.zeros(shape, dtype="uint8")
        for blob in blob_dat:
            rr, cc = circle(blob[0], blob[1], blob[2]*sqrt(2)-0.5, shape=shape)
            map[rr, cc] = 1
        return map

    def categorize_image(self, key, categories):
        if categories and categories is not self.snaps[key]["categories"]:
            self.snaps[key]["categories"].clear()
            for cat in categories:
                self.snaps[key]["categories"].append(cat)

    def get_categories(self, key):
        try:
            cat = self.snaps[key]["categories"]
            return cat
        except KeyError:
            return ""

    def load_available_categories(self):
        pass

    def _detect_edges(self, channel, sigma=1, low_threshold=None, high_threshold=None):
        """
        Privat method to detect the edges of the given channel via the canny operator.

        Keyword arguments:
        channel: The respective color channel of the image
        sigma: The standard deviation of the
        low_threshold: Lower bound for hysteresis thresholding (linking edges).
        high_threshold:Upper bound for hysteresis thresholding (linking edges).

        Returns:
        array -- The edge map of the given channel
        """
        return canny(channel, sigma, low_threshold, high_threshold)

    def _perform_labelling(self, loc_max):
        """
        Private method to perform labeling of each unique region

        Keyword arguments:
        loc_max (tuple): A tuple containing the calculated local maxima of
        each channel in the order blue, red, green

        Returns:
        tuple -- A tuple containing the labeled local maxima of each channel
        in the order blue, red, green
        """
        markers_blue, lab_num_blue = ndi.label(loc_max[0])
        markers_red, lab_num_red = ndi.label(loc_max[1])
        markers_green, lab_num_green = ndi.label(loc_max[2])
        return (markers_blue, markers_red, markers_green), (lab_num_blue, lab_num_red, lab_num_green)

    def _estimate_image_quality(self, channels):
        """
        Private method to estimate the image quality on basis of blurriness
        and overexposure

        Keyword arguments:
        channels (tuple): A tuple containing the individual channels of the
        image in the order blue, red, green

        Returns:
        2D tuple -- A 2D tuple containing the estimated image quality of each
        channel in the order(blurriness(b,r,g), overexposure(b,r,g))
        """
        return (self._calculate_blurriness(channels),
                self._determine_overexposure(channels))

    def _calculate_blurriness(self, channels):
        """
        Private method to calculate the blurriness factor of each image channel

        Keyword arguments:
        channels (tuple): A tuple containing the individual channels of the
        image in the order blue, red, green

        Returns:
        tuple -- A tuple containing the estimated blurriness of each channel in
        the order blue, red, green
        """
        blue_chan = []
        red_chan = []
        green_chan = []
        blu_th = Channel.get_minmax(channels[0], channel_only=True)[0] + 1/5 * (
            Channel.get_dynamic_range(channels[0],
                                      channel_only=True, in_percent=False))
        red_th = Channel.get_minmax(channels[1], channel_only=True)[0] + 1/5 * (
            Channel.get_dynamic_range(channels[1],
                                      channel_only=True, in_percent=False))
        gre_th = Channel.get_minmax(channels[2], channel_only=True)[0] + 1/5 * (
            Channel.get_dynamic_range(channels[2],
                                      channel_only=True, in_percent=False))
        lapl_blue = laplace(channels[0], 3)
        lapl_red = laplace(channels[1], 3)
        lapl_green = laplace(channels[2], 3)
        height = len(channels[0])
        width = len(channels[0][0])
        for y in range(height):
            for x in range(width):
                if channels[0][y][x] > blu_th:
                    blue_chan.append(lapl_blue[y][x])
                if channels[1][y][x] > red_th:
                    red_chan.append(lapl_red[y][x])
                if channels[2][y][x] > gre_th:
                    green_chan.append(lapl_green[y][x])
        blue_var = np.var(blue_chan, ddof=1) if len(blue_chan) > 100 else -1
        red_var = np.var(red_chan, ddof=1) if len(red_chan) > 100 else -1
        green_var = np.var(green_chan, ddof=1) if len(green_chan) > 100 else -1
        return blue_var, red_var, green_var

    def _determine_overexposure(self, channels):
        """
        Private method to determine if an image is overexposed

        Keyword arguments:
        channels (tuple): A tuple containing the individual channels of the
        image in the order blue, red, green

        Returns:
        tuple -- A tuple containing the estimated overexposure of each channel
        in the order blue, red, green
        """
        blue_hist = histogram(channels[0])
        red_hist = histogram(channels[1])
        green_hist = histogram(channels[2])
        pass
