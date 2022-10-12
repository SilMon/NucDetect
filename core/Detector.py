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

from detector_modules.FocusMapper import FocusMapper
from detector_modules.NucleusMapper import NucleusMapper
from detector_modules.ImageLoader import ImageLoader
from detector_modules.QualityTester import QualityTester
from detector_modules.AreaAndROIDExtractor import extract_nuclei_from_maps, extract_foci_from_maps

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
        self.imageloader = ImageLoader()
        self.focusmapper = FocusMapper()
        self.nucleusmapper = NucleusMapper()
        self.qualitytester = QualityTester()

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
        imgdat = self.imageloader.get_image_data(path)
        imgdat["id"] = self.imageloader.calculate_image_id(path)
        # Check if only a grayscale image was provided
        if imgdat["channels"] == 1:
            raise ValueError("Detector class can only analyse multichannel images, not grayscale!")
        image = self.imageloader.load_image(path)
        names = settings["names"]
        main_channel = settings["main"]
        # Channel extraction
        channels = self.imageloader.get_channels(image)
        active = settings["activated"]
        # Check if all channels are activated
        analysis_settings["names"] = [names[x] for x in range(len(names)) if active[x]]
        analysis_settings["use_pre-processing"] = settings["use_pre-processing"]
        channels = [channels[x] for x in range(len(channels)) if active[x]]
        # Adjust the index of the main channel
        for x in range(main_channel):
            main_channel -= 1 if not active[x] and x < main_channel else 0
        main = channels[main_channel]
        foc_channels = [channels[i] for i in range(len(channels)) if i != main_channel]
        foc_names = [names[i] for i in range(len(names)) if i != main_channel]
        if not settings["type"]:
            s0 = time.time()
            # Map nuclei
            self.nucleusmapper.set_channels((main,))
            self.nucleusmapper.set_settings(analysis_settings)
            nucmap = self.nucleusmapper.map_nuclei()
            Detector.log(f"Finished nuclei extraction {time.time() - s0:.4f}", logging)
            # Map foci
            self.focusmapper.set_channels(foc_channels)
            self.focusmapper.set_settings(analysis_settings)
            foc_maps = self.focusmapper.map_foci()
            Detector.log(f"Finished foci extraction {time.time() - s0:.4f}", logging)
            # Extract roi from maps
            nuclei = extract_nuclei_from_maps(nucmap, names[settings["main"]])
            # Extract foci
            foci = []
            for ind, focmap in enumerate(foc_maps):
                foci.extend(extract_foci_from_maps(focmap, foc_names[ind], nuclei))
            # Check the quality of extracted nuclei/foci
            roi = nuclei + foci
            self.qualitytester.set_channels(channels)
            self.qualitytester.set_channel_names(names)
            self.qualitytester.set_settings(analysis_settings)
            self.log(f"Detected Nuclei: {len(nuclei)}", logging)
            self.log(f"Detected Foci: {len(foci)}", logging)
            self.qualitytester.set_roi(roi)
            nuclei, foci = self.qualitytester.check_roi_quality()
            print(f"Main: {len(nuclei)}")
            print(f"Foc: {len(foci)}")
            rois = nuclei + foci
            """
            # Channel thresholding
            thresh_chan = Detector.threshold_channels(channels, main_channel, analysis_settings=analysis_settings)
            rois = Detector.classic_roi_extraction(channels, thresh_chan, names,
                                                   main_map=main_channel, quality_check=not ml_analysis,
                                                   logging=logging, analysis_settings=analysis_settings)
            """
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
        handler.add_rois(rois)
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
        # TODO fix
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
    def log(message: str, state: bool = True):
        """
        Method to log messages to the console

        :param message: The message to log
        :param state: Enables logging
        :return: None
        """
        if state:
            print(message)
