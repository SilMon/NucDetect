"""
Created 09.04.2019
@author Romano Weiss
"""
from __future__ import annotations

import time
from copy import deepcopy
from typing import Union, Dict, List, Tuple

import numpy as np

from core.detector_modules.AreaAndROIExtractor import extract_nuclei_from_maps, extract_foci_from_maps
from core.detector_modules.FCNMapper import FCNMapper
from core.detector_modules.FocusMapper import FocusMapper
from core.detector_modules.ImageLoader import ImageLoader
from core.detector_modules.MapComparator import MapComparator
from core.detector_modules.NucleusMapper import NucleusMapper
from core.detector_modules.QualityTester import QualityTester
from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler


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
        self.fcnmapper = None
        self.qualitytester = QualityTester()

    def analyse_image(self, path: str,
                      settings: Dict[str, Union[List, bool]]) ->\
            Dict[str, Union[ROIHandler, np.ndarray, Dict[str, str]]]:
        """
        Method to extract rois from the image given by path

        :param path: The URL of the image
        :param settings: Dictionary containing the necessary information for analysis
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
        main_channel: int = settings["main"]
        # Channel extraction
        channels = self.imageloader.get_channels(image)
        active = settings["activated"]
        # Check if all channels are activated
        analysis_settings["names"] = [names[x] for x in range(len(names)) if active[x]]
        analysis_settings["use_pre-processing"] = settings["use_pre-processing"]
        analysis_settings["main_channel_name"] = analysis_settings["names"][main_channel]
        channels = [channels[x] for x in range(len(channels)) if active[x]]
        # Adjust the index of the main channel
        for x in range(main_channel):
            main_channel -= 1 if not active[x] and x < main_channel else 0
        main = channels[main_channel]
        foc_channels = [channels[i] for i in range(len(channels)) if i != main_channel]
        analysis_settings["foci_channel_names"] = [x for x in analysis_settings["names"]
                                                   if x is not analysis_settings["main_channel_name"]]
        # Detect roi via image processing and machine learning
        iproi, maps1 = self.ip_roi_extraction(main, foc_channels, analysis_settings, logging)
        mlroi, maps2 = self.ml_roi_extraction(maps1[0], foc_channels, analysis_settings, logging)
        print(f"Detected IP ROI: {len(iproi)}")
        print(f"Detected ML ROI: {len(mlroi)}")
        rois = []
        main_roi = [x for x in iproi if x.main]
        for channel in analysis_settings["foci_channel_names"]:
            # Define map Comparator
            mapc = MapComparator(main_roi,
                                 [x for x in iproi if not x.main and x.ident == channel],
                                 [x for x in mlroi if not x.main and x.ident == channel],
                                 image.shape[:2])
            # Calculate new nucleus match
            mapc.get_match_for_nuclei()
            rois.extend(mapc.merge_overlapping_foci())
        rois.extend(main_roi)

        # Check for quality of roi
        if rois:
            qroi = self.perform_quality_check(channels, names, analysis_settings, rois)
            print(f"QR: Removed foci: {len(rois) - len(qroi)}")
        else:
            qroi = []
        handler = ROIHandler(ident=imgdat["id"])
        handler.add_rois(qroi)
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

    def ip_roi_extraction(self,  main, foc_channels,
                          analysis_settings, logging) -> Tuple[List[ROI], List[np.ndarray]]:
        """
        Method to detect nuclei and foci viaimage processing

        :param main: The channel which should contain nuclei
        :param foc_channels: All image channel which potentially contain foci
        :param analysis_settings: The analysis settings to apply
        :param logging: Decider if logging should be performed
        :return: The extracted ROI and the used detection maps
        """
        s0 = time.time()
        # Map nuclei
        self.nucleusmapper.set_channels((main,))
        self.nucleusmapper.set_settings(analysis_settings)
        nucmap = self.nucleusmapper.map_nuclei()
        Detector.log(f"Finished nuclei extraction {time.time() - s0:.4f}", logging)
        # Map foci
        self.focusmapper.set_channels(foc_channels)
        self.focusmapper.set_settings(analysis_settings)
        foc_maps = self.focusmapper.map_foci(main=nucmap)
        Detector.log(f"Finished foci extraction {time.time() - s0:.4f}", logging)
        roi = Detector.extract_roi_from_maps(nucmap, foc_maps,
                                             analysis_settings["main_channel_name"],
                                             analysis_settings["foci_channel_names"])
        for r in roi:
            r.detection_method = "Image Processing"
        foc_maps.insert(0, nucmap)
        if roi:
            return roi, foc_maps
        else:
            return [], [np.zeros(shape=nucmap.shape) for _ in range(len(foc_maps) + 1)]

    def ml_roi_extraction(self, main_map: np.ndarray, foc_channels,
                          analysis_settings, logging) -> Tuple[List[ROI], List[np.ndarray]]:
        """
        Method to detect nuclei and foci via machine learning

        :param main_map: All channels of the image
        :param foc_channels: All image channel which potentially contain foci
        :param analysis_settings: The analysis settings to apply
        :param logging: Decider if logging should be performed
        :return: The extracted ROI
        """
        s0 = time.time()
        # Map nuclei
        self.fcnmapper = FCNMapper()
        self.fcnmapper.set_settings(analysis_settings)
        # Map foci
        self.fcnmapper.set_channels(foc_channels)
        foc_maps = self.fcnmapper.get_marked_maps()
        Detector.log(f"Finished foci extraction {time.time() - s0:.4f}", logging)
        # Extract roi from maps
        roi = Detector.extract_roi_from_maps(main_map, foc_maps,
                                             analysis_settings["main_channel_name"],
                                             analysis_settings["foci_channel_names"])
        for r in roi:
            r.detection_method = "Machine Learning"
        return roi, foc_maps

    @staticmethod
    def extract_roi_from_maps(main: np.ndarray, foci_maps: List[np.ndarray],
                              main_name: str, foc_names: List[str]) -> List[ROI]:
        """
        Method to extract nuclei and foci from the given maps

        :param main: Binary map for nuclei
        :param foci_maps: List of maps for foci
        :param main_name: Name assigned to the main channel
        :param foc_names: List of names assigned to the foci channels
        :return: The extracted roi
        """
        nuclei = extract_nuclei_from_maps(main, main_name)
        foci = []
        for ind, focmap in enumerate(foci_maps):
            foci.extend(extract_foci_from_maps(focmap, foc_names[ind], nuclei))
        return nuclei + foci

    def perform_quality_check(self, channels: List[np.ndarray],
                              names: List[str], analysis_settings: Dict, roi: List[ROI]):
        """
        Method to perform a quality check on the given roi

        :param channels: The channels the roi were derived from
        :param names: The names associated with each channel
        :param analysis_settings: The analysis settings to apply
        :param roi: The roi to check
        :return: The checked roi
        """
        self.qualitytester.set_channels(channels)
        self.qualitytester.set_channel_names(names)
        self.qualitytester.set_settings(analysis_settings)
        self.qualitytester.set_roi(roi)
        nuclei, foci = self.qualitytester.check_roi_quality()
        return nuclei + foci

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
