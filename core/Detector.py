"""
Created 09.04.2019
@author Romano Weiss
"""
from __future__ import annotations

import datetime
import hashlib
import os
import time
from typing import Union, Dict, List, Tuple, Any

import numpy as np
import piexif
from scipy import ndimage as ndi
from skimage import io
from skimage.draw import circle
from skimage.feature import canny, blob_log
from skimage.filters import threshold_local
from skimage.filters.rank import maximum
from skimage.morphology import watershed
from skimage.morphology.binary import binary_opening

from core.JittedFunctions import eu_dist, create_circular_mask, relabel_array, imprint_data_into_channel
from core.ROI import ROI
from core.ROIHandler import ROIHandler


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
        start = time.time()
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
        Detector.log(f"Total analysis time: {time.time()-start}", logging)
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
        markers, lab_nums = Detector.perform_labelling(bin_maps, main_map=main_map)
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
        Detector.log(f"Finished main ROI extraction {time.time()-s0:.4f}", logging)
        s1 = time.time()
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
                                if main_markers[y][x] > 0:
                                    roi.associated = main[main_markers[y][x]]
                            else:
                                if temprois[lab].associated is None:
                                    if main_markers[y][x] > 0:
                                        roi.associated = main[main_markers[y][x]]
                                temprois[lab].add_point((x, y), int(channels[ind][y][x]))
                del temprois[0]
                rois.extend(temprois)
        Detector.log(f"Finished first focus extraction {time.time() - s1:.4f}", logging)
        s2 = time.time()
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
                            if main_markers[y][x] > 0:
                                roi.associated = main[main_markers[y][x]]
                        else:
                            if temprois[lab].associated is None:
                                if main_markers[y][x] > 0:
                                    roi.associated = main[main_markers[y][x]]
                            temprois[lab].add_point((x, y), int(channels[ind][y][x]))
            del temprois[0]
            rois.extend(temprois)
        del main[0]
        rois.extend(main)
        Detector.log(f"Finished second focus extraction {time.time() - s2:.4f}", logging)
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
        Detector.log(f"Removed unassociated foci: {foclen - len(foci)}", logging)
        # Remove very small foci
        foci = [x for x in foci if len(x) > min_foc_area]
        Detector.log(f"Total Removed foci: {foclen - len(foci)}\nTime: {time.time() - s8:4f}\nFocus Quality Check",
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
                    rdist = eu_dist(c, c2) < maxdist / 2 + maxdist2 / 2
                    if focus.ident == focus2.ident and rdist:
                        if focus.calculate_roi_intersection(focus2) >= max_focus_overlapp:
                            if focus2.points >= focus.points:
                                focus2.marked = True
                            else:
                                focus.marked = True
                                break
        foci = [x for x in foci if not x.marked]
        rem_list.clear()
        Detector.log(f"Time: {time.time() - s4:4f}", logging)
        rois.clear()
        # TODO Reihenfolge wichtig, ausbessern da undynamisch
        rois.extend(foci)
        rois.extend(main)
        Detector.log(f"Total Quality Check Time: {time.time() - s7}", logging)

    @staticmethod
    def create_association_map(rois: List[ROI]) -> Dict[ROI, List[ROI]]:
        """
        Method to create a dictionary which associates main roi (as keys) with their foci (as list of ROI)

        :param rois: The list of roi to create the association map from
        :return: The association map as dict
        """
        # TODO fix
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
        iterations = 5
        size = 7
        selem = create_circular_mask(size, size)
        # Load image
        orig = channels[main_channel]
        hmax = np.amax(orig)
        hmin = np.amin(orig)
        threshold = hmin + round(0.05 * hmax)
        ch_main_bin = ndi.binary_fill_holes(orig > threshold)
        # Calculate the euclidean distance map
        edm = ndi.distance_transform_edt(ch_main_bin)
        # Normalize edm
        xmax, xmin = edm.max(), edm.min()
        x = (edm - xmin) / (xmax - xmin)
        # Determine maxima of EDM
        maxi = maximum(x, selem=selem)
        # Iteratively determine maximum
        for _ in range(iterations):
            maxi = maximum(maxi, selem=selem)
        thresh_ = threshold_local(maxi, block_size=size * 8 + 1)
        maxi = ndi.binary_fill_holes(maxi > thresh_)

        # Calculate second threshold
        mini = np.amin(orig) + 0.005 * np.amax(orig)
        for y in range(len(orig)):
            for x in range(len(orig[0])):
                pix = orig[y][x]
                if pix <= mini:
                    maxi[y][x] = 0
        # Open maxi to remove noise
        maxi = binary_opening(maxi, selem=create_circular_mask(size * 2, size * 2))

        # Extract nuclei from map
        area, labels = ndi.label(maxi)
        nucs = [None] * (labels + 1)

        for y in range(len(area)):
            for x in range(len(area[0])):
                pix = area[y][x]
                if nucs[pix] is None:
                    nucs[pix] = [[], []]
                nucs[pix][0].append(y)
                nucs[pix][1].append(x)

        # Remove background
        del nucs[0]
        centers = [(np.average(x[0]), np.average(x[1])) for x in nucs]

        cmask = np.zeros(shape=orig.shape)
        ind = 1
        for c in centers:
            cmask[int(c[0])][int(c[1])] = ind
            ind += 1

        # Create watershed segmentation based on centers
        ws = watershed(-edm, cmask, mask=ch_main_bin, watershed_line=True)
        # Check number of unique watershed labels
        unique = list(np.unique(ws))
        t = time.time()
        relabel_array(ws)
        # TODO
        Detector.log(f"Time relabeling: {time.time() - t}", False)
        thresh[main_channel] = ws
        # Extract nuclei from watershed
        det: List[Tuple[int, int]] = [None] * len(unique)
        detpix = []
        for y in range(len(ws)):
            for x in range(len(ws[0])):
                pix = ws[y][x]
                if pix not in detpix:
                    detpix.append(pix)
                if det[pix] is None:
                    det[pix] = []
                det[pix].append((y, x))
        del det[0]
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
        impr_time = 0
        nfpl_time = 0
        start = time.time()
        chan = np.zeros(shape=channel.shape)
        for nuc in nuclei:
            thresh = []
            for p in nuc:
                thresh.append((p, channel[p[0]][p[1]]))
            if thresh:
                t1 = time.time()
                thresh_np, offset = Detector.create_numpy_from_point_list(thresh)
                nfpl_time += time.time() - t1
                edges = Detector.detect_edges(thresh_np)
                if np.max(edges) > 0:
                    chan_fill = ndi.binary_fill_holes(edges)
                    chan_open = ndi.binary_opening(chan_fill)
                    if np.max(chan_open) > 0:
                        t = time.time()
                        imprint_data_into_channel(chan, chan_open, offset)
                        impr_time += time.time() - t
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
    def get_channels(img: np.ndarray) -> List[np.ndarray]:
        """
        Method to extract the channels of the given image

        :param img: The image as ndarray
        :return: A list of all channels
        """
        t = time.time()
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
    def load_image(path: str) -> np.ndarray:
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
