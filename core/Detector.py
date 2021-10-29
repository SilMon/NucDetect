"""
Created 09.04.2019
@author Romano Weiss
"""
from __future__ import annotations

import datetime
import hashlib
import math
import os
import time
from copy import deepcopy
from typing import Union, Dict, List, Tuple, Iterable

import matplotlib.pyplot as plt
import numpy as np
import piexif
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes, label
from skimage import img_as_ubyte
from skimage import io
from skimage.draw import disk
from skimage.feature import canny, blob_log
from skimage.filters import threshold_local
from skimage.filters.rank import maximum
from skimage.morphology import dilation
from skimage.morphology.binary import binary_opening, binary_erosion
from skimage.segmentation import watershed

from core.JittedFunctions import eu_dist, create_circular_mask, relabel_array, imprint_data_into_channel
from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler
from fcn.FCN import FCN


class Detector:
    FORMATS = [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".bmp"
    ]

    def __init__(self):
        """
        Constructor of the detector class

        :param logging: Indicates if analysis messages should be printed to the console
        """
        self.analyser = None

    def analyse_image(self, path: str, settings: Dict[str, Union[List, bool]],
                      logging: bool = True,
                      ml_analysis: bool = False) -> Dict[str, Union[ROIHandler, np.ndarray, Dict[str, str]]]:
        """
        Method to extract rois from the image given by path

        :param path: The URL of the image
        :param settings: Dictionary containing the necessary information for analysis
        :param logging: Enables logging
        :param ml_analysis: Enable image analysis via U-Net
        :return: The analysis results as dict
        """
        analysis_settings = deepcopy(settings["analysis_settings"])
        start = time.time()
        logging = analysis_settings["logging"]
        imgdat = Detector.get_image_data(path)
        imgdat["id"] = Detector.calculate_image_id(path)
        # Check if only a grayscale image was provided
        if imgdat["channels"] == 1:
            raise ValueError("Detector class can only analyse multichannel images, not grayscale!")
        image = Detector.load_image(path)
        names = settings["names"]
        main_channel = settings["main"]
        # Channel extraction
        channels = Detector.get_channels(image)
        active = settings["activated"]
        # Check if all channels are activated
        analysis_settings["names"] = [names[x] for x in range(len(names)) if active[x]]
        channels = [channels[x] for x in range(len(channels)) if active[x]]
        # Adjust the index of the main channel
        for x in range(main_channel):
            main_channel -= 1 if not active[x] and x < main_channel else 0
        if not settings["type"]:
            # Channel thresholding
            thresh_chan = Detector.threshold_channels(channels, main_channel, analysis_settings=analysis_settings)
            rois = Detector.classic_roi_extraction(channels, thresh_chan, names,
                                                   main_map=main_channel, quality_check=not ml_analysis,
                                                   logging=logging, analysis_settings=analysis_settings)
        else:
            self.analyser = FCN()
            nuclei = self.analyser.predict_image(path, self.analyser.NUCLEI, channels=(main_channel,),
                                                 threshold=analysis_settings["fcn_certainty_nuclei"],
                                                 logging=logging)[0]
            # Get indices of foci channels
            indices = [names.index(x) for x in analysis_settings["names"] if names.index(x) != main_channel]
            foci = self.analyser.predict_image(path, self.analyser.FOCI,
                                               channels=indices, logging=logging,
                                               threshold=analysis_settings["fcn_certainty_foci"])
            if main_channel > len(foci):
                foci.append(nuclei)
            else:
                foci.insert(main_channel, nuclei)
            rois = Detector.ml_roi_extraction(channels, foci, names,
                                              main_map=main_channel,
                                              logging=logging,
                                              analysis_settings=analysis_settings)

        handler = ROIHandler(ident=imgdat["id"])
        for roi in rois:
            handler.add_roi(roi)
        handler.idents = analysis_settings["names"]
        imgdat["handler"] = handler
        imgdat["names"] = analysis_settings["names"]
        imgdat["channels"] = channels
        imgdat["active channels"] = active
        imgdat["main channel"] = main_channel
        imgdat["add to experiment"] = settings["add_to_experiment"]
        imgdat["experiment details"] = settings["experiment_details"]
        imgdat["used_settings"] = analysis_settings
        Detector.log(f"Total analysis time: {time.time() - start: .4f}", logging)
        return imgdat

    @staticmethod
    def ml_roi_extraction(channels: Iterable[np.ndarray], bin_maps: Iterable[np.ndarray],
                          names: Iterable[str], main_map: int = 2,
                          logging: bool = True,
                          analysis_settings: Dict = None) -> Iterable[ROI]:
        """
        Method to extract roi from ROI maps created via FCN analysis

        :param channels: List of all analysed channels
        :param bin_maps: Detection maps for all channels
        :param names: List of names for each channel
        :param main_map: Index of the main map
        :param logging: Enables logging during execution
        :param analysis_settings: The settings to use for the ROI extraction
        :return:
        """
        s0 = time.time()
        if analysis_settings:
            names = analysis_settings.get("names", names)
            main_map = analysis_settings.get("main_channel", main_map)
            logging = analysis_settings.get("logging", logging)
        rois = []
        markers, lab_nums = Detector.perform_labelling(bin_maps, main_map=main_map)
        main_markers, main_nums = Detector.mark_areas(markers[main_map])
        markers[main_map] = main_markers
        lab_nums[main_map] = main_nums
        # Extract nuclei
        main = Detector.extract_roi_from_main_map(
            main_markers,
            main_map,
            names
        )
        Detector.log(f"Finished main ROI extraction {time.time() - s0:.4f}", logging)
        # First round of focus detection
        s1 = time.time()
        rois.extend(
            Detector.extract_rois_from_map(
                markers,
                main_map,
                names,
                main
            )
        )
        # Remove unassociated foci
        rois = [x for x in rois if x.associated is not None]
        Detector.log(f"Finished focus extraction {time.time() - s1:.4f}", logging)
        rois.extend(main)
        rois = [x for x in rois if x is not None and len(x) > 9]
        return rois

    @staticmethod
    def classic_roi_extraction(channels: Iterable[np.ndarray], bin_maps: Iterable[np.ndarray],
                               names: Iterable[str], main_map: int = 2, quality_check: bool = True,
                               logging: bool = True,
                               analysis_settings: Dict = None) -> Iterable[ROI]:
        """
        Method to extract ROI objects from the given binary maps

        :param channels: List of the channels to detect rois on
        :param bin_maps: A list of binary maps of the channels
        :param names: The names associated with each channel
        :param main_map: Index of the map containing nuclei
        :param quality_check: Enables a quality check after ROI extraction
        :param logging: Enables logging during execution
        :param analysis_settings: The settings to use for analysis
        :return: A list of all detected roi
        """
        if analysis_settings:
            names = analysis_settings.get("names", names)
            main_map = analysis_settings.get("main_channel", main_map)
            quality_check = analysis_settings.get("quality_check", quality_check)
        s0 = time.time()
        rois = []
        # Label binary maps
        markers, lab_nums = Detector.perform_labelling(bin_maps, main_map=main_map)
        # Extract nuclei
        main = Detector.extract_roi_from_main_map(
            markers,
            main_map,
            names
        )
        Detector.log(f"Finished main ROI extraction {time.time() - s0:.4f}", logging)
        # First round of focus detection
        s1 = time.time()
        rois.extend(
            Detector.extract_rois_from_map(
                markers,
                main_map,
                names,
                main
            )
        )
        Detector.log(f"Finished first focus extraction {time.time() - s1:.4f}", logging)
        # Second round of focus detection
        s2 = time.time()
        markers, lab_nums = Detector.detect_blobs(
            channels,
            main_channel=main_map,
            min_sigma=analysis_settings.get("blob_min_sigma"),
            max_sigma=analysis_settings.get("blob_max_sigma"),
            num_sigma=analysis_settings.get("blob_num_sigma"),
            threshold=analysis_settings.get("blob_threshold")
        )
        rois.extend(
            Detector.extract_rois_from_map(
                markers,
                main_map,
                names,
                main
            )
        )
        main = [x for x in main if x is not None]
        rois.extend(main)
        Detector.log(f"Finished second focus extraction {time.time() - s2:.4f}", logging)
        if quality_check:
            Detector.perform_roi_quality_check(rois,
                                               channels,
                                               names,
                                               analysis_settings=analysis_settings)
        return rois

    @staticmethod
    def extract_roi_from_main_map(binary_maps: Iterable[np.ndarray],
                                  main_map: int, names: Iterable[str]) -> List[ROI]:
        """
        Method to extract the main roi from an area map

        :param binary_maps: The labelled binary ROI maps for all channels
        :param main_map: The index of the main map
        :param names: The names of each channel
        :return: A list of extracted ROI
        """
        main_markers = binary_maps[main_map] if not isinstance(binary_maps, np.ndarray) else binary_maps
        # Define list for main roi
        main = []
        # Get RLE for map
        areas = Detector.encode_areas(main_markers)
        for _, rl in areas.items():
            roi = ROI(channel=names[main_map], main=True)
            roi.set_area(rl)
            main.append(roi)
        return main

    @staticmethod
    def encode_areas(area_map: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
        """
        Method to extract individual areas from the given binary map.

        :param area_map: The map to extract the areas from
        :return: Dictionary containing the label for each area as well as the associated area given bei image row
        and run length
        """
        height, width = area_map.shape
        # Define dict for detected areas
        areas = {}
        # Iterate over map
        for y in range(height):
            x = 0
            while x < width:
                # Get label at y:x
                label = area_map[y][x]
                if label != 0:
                    if areas.get(label) is None:
                        areas[label] = []
                    col = x
                    # run length
                    rl = 0
                    # Iterate over row
                    while area_map[y][x] == label:
                        rl += 1
                        x += 1
                        # Break if x reaches border
                        if x == width:
                            break
                    areas[label].append((y, col, rl))
                else:
                    x += 1
        return areas

    @staticmethod
    def extract_rois_from_map(binary_maps: List[np.ndarray], main_map: int, names: List[str],
                              main: List[ROI]) -> List[ROI]:
        """
        Method to extract ROIs from given binary maps

        :param binary_maps: The labelled binary ROI maps for all channels
        :param main_map: The index of the main map
        :param names: The names of each channel
        :param main: The extracted main ROI
        :return: A list of extracted ROI
        """
        rois: List[ROI] = []
        for ind in range(len(binary_maps)):
            if ind != main_map:
                areas = Detector.encode_areas(binary_maps[ind])
                for _, rl in areas.items():
                    # Define focus roi
                    roi = ROI(channel=names[ind], main=False)
                    roi.set_area(rl)
                    # Get associated main roi
                    y, x = roi.calculate_dimensions()["center"]
                    if binary_maps[main_map][y][x] > 0:
                        roi.associated = main[binary_maps[main_map][y][x] - 1]
                        rois.append(roi)
        return rois

    @staticmethod
    def perform_roi_quality_check(rois: List[ROI], channels: Iterable[np.ndarray], channel_names: Iterable[str],
                                  max_focus_overlapp: float = .75, min_main_area: int = 1000,
                                  max_main_area: int = 30000, min_foc_area: int = 5, max_foc_area: int = 270,
                                  cutoff: float = 0.03,  size_factor: float = 1.0, logging: bool = True,
                                  analysis_settings: Dict = None) -> None:
        """
        Method to check detected rois for their quality. Changes ROI in place.

        :param min_foc_area: The minimal area of a focus
        :param max_foc_area: The maximal area of a focus
        :param rois: A list of detected rois
        :param channels: List of channels the roi are derived from
        :param channel_names: Identifier for each channel as list
        :param max_focus_overlapp: The threshold used to determine if two rois are considered duplicates
        :param min_dist: The minimal distance between 2 nuclei
        :param min_thresh: The lower percentile to check for oversegmentation
        :param max_thresh: The upper percentile to check for undersegmentation
        :param min_main_area: The mininmal area of a nucleus
        :param max_main_area: The maximal area of a nucleus
        :param ws_line: Should the separation line for ws separated nuclei be drawn?
        :param cutoff: Factor to determine the focus cut-off
        :param sife_factor: Factor to determine the scale of all size related factors like areas
        :param logging: Enables logging
        :param analysis_settings: Settings to use for the quality check
        :return: None
        """
        if analysis_settings:
            max_focus_overlapp = analysis_settings.get("quality_max_foc_overlap", max_focus_overlapp)
            min_main_area = analysis_settings.get("quality_min_nuc_size", min_main_area)
            max_main_area = analysis_settings.get("quality_max_nuc_size", max_main_area)
            min_foc_area = analysis_settings.get("quality_min_foc_size", min_foc_area)
            max_foc_area = analysis_settings.get("quality_max_foc_size", max_foc_area)
            cutoff = analysis_settings.get("cutoff", cutoff)
            size_factor = analysis_settings.get("size_factor", size_factor)
            logging = analysis_settings.get("logging", logging)
        rem_list = []
        temp = []
        main: List[ROI] = []
        foci: List[ROI] = []
        s0 = time.time()
        for roi in rois:
            if roi.main:
                main.append(roi)
                temp.append(len(roi))
            else:
                foci.append(roi)
        Detector.log(f"Detected nuclei:{len(main)}\nDetected foci: {len(foci)}", logging)
        # Remove very small or extremely large main ROI
        main = [x for x in main if min_main_area * size_factor < len(x) < max_main_area * size_factor]
        chan_del = []
        # Check if channel for nucleus can be analysed
        for nuc in main:
            for channel in channel_names:
                i = channel_names.index(channel)
                c = channels[i]
                if channel != nuc.ident:
                    # Get all intensity values for area
                    int_area = nuc.extract_area_intensity(c)
                    # Get AVG and std
                    avg, std = np.average(int_area), np.std(int_area)
                    # Check if std for average is smaller than cutting function
                    if math.exp(avg * cutoff) - 5 > std:
                        chan_del.append((channel, nuc))
        # Remove all foci whose main roi was removed
        foci = [x for x in foci if x.associated in main]
        Detector.log(f"Checking for very small nuclei: {time.time() - s0: .4f} secs\nRemove not associated foci",
                     logging)
        s1 = time.time()
        # Remove foci that are either unassociated or whose nucleus was deleted
        association_roi = deepcopy(main)
        association_roi.extend(foci)
        ass = Detector.create_association_map(association_roi)
        marked_foci = 0
        # Remove foci whose channel was declared invalid
        for foc_rem in chan_del:
            # Get list of associated foci
            foci = ass[foc_rem[1]]
            for focus in foci:
                if focus.ident is foc_rem[0]:
                    marked_foci += 1
                    focus.marked = True
        Detector.log(f"Number of marked foci: {marked_foci}", logging)
        mainlen = len(main)
        main = [x for x, _ in ass.items()]
        Detector.log(f"Removed nuclei: {mainlen - len(main)}", logging)
        foclen = len(foci)
        foci = [x for _, focs in ass.items() for x in focs if x.associated is not None]
        Detector.log(f"Removed unassociated foci: {foclen - len(foci)}", logging)
        # Remove very small foci
        foci = [x for x in foci if len(x) > min_foc_area < len(x) < max_foc_area * size_factor]
        Detector.log(f"Removed foci: {foclen - len(foci)}\nTime: {time.time() - s1: .4f} secs",
                     logging)
        # Focus quality check
        s2 = time.time()
        for ind in range(len(foci)):
            focus = foci[ind]
            focint = focus.calculate_statistics(channels[channel_names.index(focus.ident)])["intensity average"]
            if focint < 20:
                focus.marked = True
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
        Detector.log(f"Focus Quality Check: {time.time() - s2: .4f}", logging)
        rois.clear()
        rois.extend(foci)
        rois.extend(main)
        Detector.log(f"Total Quality Check Time: {time.time() - s0: .4f} secs", logging)

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
                     threshold: float = .1, size_factor: float = 1.0) -> Tuple[List[np.ndarray], List[int]]:
        """
        Method to detect blobs in the given channels using the blob_log method

        :param channels: A list of all channels
        :param main_channel: The index of the main channel
        :param min_sigma: the minimum standard deviation for Gaussian kernel
        :param max_sigma: the maximum standard deviation for Gaussian kernel
        :param num_sigma: The number of intermediate values of standard deviations
        to consider between min_sigma and max_sigma
        :param threshold: The absolute lower bound for scale space maxima
        :param size_factor: Factor to influence the detection limits to accommodate for image resolution
        :return: A tuple of the detected blob-maps and the respective numbers of detected blobs
        """
        blob_maps = []
        blob_nums = []
        for ind in range(len(channels)):
            if ind != main_channel:
                blobs = blob_log(channels[ind],
                                 min_sigma=min_sigma * size_factor,
                                 max_sigma=max_sigma * size_factor,
                                 num_sigma=num_sigma,
                                 threshold=threshold,
                                 exclude_border=False)
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
            rr, cc = disk((blob[0], blob[1]), blob[2] * np.sqrt(2) - 0.5, shape=shape)
            map_[rr, cc] = ind + 1
        return map_

    @staticmethod
    def mark_areas(image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Method to segment clustered areas

        :param image: The image to segment
        :return: The segmented area
        """
        # Define mask
        mask = create_circular_mask(40, 40)
        # Fill image
        fill = binary_fill_holes(image)
        # Perform binary erosion on image
        er = binary_erosion(fill, selem=mask)
        # Label eroded map
        lab, nums = label(er)
        # Dilate eroded map
        lab = dilation(lab, selem=mask)
        return lab, nums

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
                           iterations: int = 5, mask_size: int = 7,
                           percent_hmax: float = 0.05, local_threshold_multiplier: int = 8,
                           maximum_size_multiplier: int = 2,
                           size_factor: float = 1.0,
                           analysis_settings: Dict = None) -> List[np.ndarray]:
        """
        Method to threshold the channels to prepare for nuclei and foci detection

        :param channels: The channels of the image as list
        :param main_channel: Index of the channel associated with nuclei (usually blue -> 2)
        :param iterations: Number of maximum filtering to perform in a row
        :param mask_size: The diameter of the circular mask for the filtering
        :param percent_hmax: The percentage of the histogram maximum to add to the histogram minimum. Used to form
        the detection threshold
        :param local_threshold_multiplier: Multiplier used to increase mask_size for local thresholding
        :param maximum_size_multiplier: Multiplier used to increase mask_size for noise removal
        :param size_factor: Factor to accommodate for different image resolutions
        :param analysis_settings: Settings to use for thresholding
        :return: The thresholded channels
        """
        if analysis_settings:
            main_channel = analysis_settings.get("main_channel", main_channel)
            iterations = analysis_settings.get("thresh_iterations", iterations)
            mask_size = analysis_settings.get("thresh_mask_size", mask_size)
            percent_hmax = analysis_settings.get("thresh_percent_hmax", percent_hmax)
            local_threshold_multiplier = analysis_settings.get("thresh_local_thresh_mult", local_threshold_multiplier)
            maximum_size_multiplier = analysis_settings.get("thresh_max_mult", maximum_size_multiplier)
            size_factor = analysis_settings.get("size_factor", size_factor)
        thresh: List[Union[None, np.ndarray]] = [None] * len(channels)
        # Calculate the circular mask to use for morphological operators
        selem = create_circular_mask(mask_size * size_factor, mask_size * size_factor)
        # Load image
        orig = channels[main_channel]
        # Get maximum value of main channel
        hmax = np.amax(orig)
        # Get minimum value of main channel
        hmin = np.amin(orig)
        # Calculate the used threshold
        threshold = hmin + round(percent_hmax * hmax)
        # Threshold channel globally and fill holes
        ch_main_bin = ndi.binary_fill_holes(orig > threshold)
        # Calculate the euclidean distance map
        edm = ndi.distance_transform_edt(ch_main_bin)
        # Normalize edm
        xmax, xmin = edm.max(), edm.min()
        x = img_as_ubyte((edm - xmin) / (xmax - xmin))
        # Determine maxima of EDM
        maxi = maximum(x, selem=selem)
        # Iteratively determine maximum
        for _ in range(iterations):
            maxi = maximum(maxi, selem=selem)
        # Perform local thresholding
        thresh_ = threshold_local(maxi, block_size=(mask_size * local_threshold_multiplier + 1) * size_factor)
        # Threshold maximum EDM
        maxi = ndi.binary_fill_holes(maxi > thresh_)
        # Perform logical AND to remove areas that were not detected in ch_main_bin
        maxi = np.logical_and(maxi, ch_main_bin)
        # Open maxi to remove noise
        maxi = binary_opening(maxi, selem=create_circular_mask(mask_size * maximum_size_multiplier * size_factor,
                                                               mask_size * maximum_size_multiplier * size_factor))
        # Extract nuclei from map
        area, labels = ndi.label(maxi)
        nucs: List[List, List] = [None] * (labels + 1)
        for y in range(len(area)):
            for x in range(len(area[0])):
                pix = area[y][x]
                if nucs[pix] is None:
                    nucs[pix] = [[], []]
                nucs[pix][0].append(y)
                nucs[pix][1].append(x)
        # Remove background
        del nucs[0]
        # Get nuclei centers
        centers = [(np.average(x[0]), np.average(x[1])) for x in nucs]
        # Create mask with marked nuclei centers -> Used as seed points for watershed
        cmask = np.zeros(shape=orig.shape)
        ind = 1
        for c in centers:
            cmask[int(c[0])][int(c[1])] = ind
            ind += 1
        # Create watershed segmentation based on centers
        ws = watershed(-edm, cmask, mask=ch_main_bin, watershed_line=True)
        # Check number of unique watershed labels
        unique = list(np.unique(ws))
        relabel_array(ws)
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
        # Delete background
        del det[0]
        # Extract foci from channels
        for ind in range(len(channels)):
            if ind != main_channel:
                thresh[ind] = Detector.calculate_local_region_threshold(det,
                                                                        channels[ind],
                                                                        analysis_settings["canny_sigma"],
                                                                        analysis_settings["canny_low_thresh"],
                                                                        analysis_settings["canny_up_thresh"])
        return thresh

    @staticmethod
    def calculate_local_region_threshold(nuclei: List[Tuple[int, int]],
                                         channel: np.ndarray,
                                         sigma: float = 2,
                                         low_threshold: float = 0.1,
                                         high_threshold: float = 0.2) -> np.ndarray:
        """
        Method to threshold nuclei for foci extraction

        :param nuclei: The points of the nucleus as list
        :param channel: The corresponding channel
        :param sigma: Standard deviation of the gaussian kernel
        :param low_threshold: Lower bound for hysteresis thresholding
        :param high_threshold: Upper bound for hysteresis thresholding
        :return: The foci map for the nucleus
        """
        chan = np.zeros(shape=channel.shape)
        for nuc in nuclei:
            thresh = []
            for p in nuc:
                thresh.append((p, channel[p[0]][p[1]]))
            if thresh:
                thresh_np, offset = Detector.create_numpy_from_point_list(thresh)
                edges = Detector.detect_edges(thresh_np, sigma, low_threshold, high_threshold)
                if np.max(edges) > 0:
                    chan_fill = ndi.binary_fill_holes(edges)
                    chan_open = ndi.binary_opening(chan_fill)
                    if np.max(chan_open) > 0:
                        imprint_data_into_channel(chan, chan_open, offset)
        return chan

    @staticmethod
    def detect_edges(channel: np.ndarray, sigma: Union[int, float] = 2,
                     low_threshold: float = 0.1,
                     high_threshold: float = 0.2) -> np.ndarray:
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
            image_data = {
                "datetime": datetime.datetime.fromtimestamp(os.path.getctime(path)),
                "height": img.shape[0],
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
    def calculate_image_id(path: str) -> str:
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
