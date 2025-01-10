"""
Created 09.04.2019
@author Romano Weiss
"""
from __future__ import annotations

import datetime
import os.path
import time
from copy import deepcopy
from typing import Union, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import Paths
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
        self.analyser = None
        self.analysis_log = {"Date": datetime.datetime.today().strftime("%Y-%m-%d"),
                             "Time": datetime.datetime.today().strftime("%H:%M:%S"),
                             "Analysed Images": [],
                             "Messages": {}}
        self.imageloader = ImageLoader()
        self.focusmapper = FocusMapper()
        self.nucleusmapper = NucleusMapper()
        self.fcnmapper = None
        self.qualitytester = QualityTester()

    def analyse_images(self, images: List[str], settings: Dict[str, Union[List, bool]]) -> \
            List[Dict[str, Union[ROIHandler, np.ndarray, Dict[str, str]]]]:
        """
        Method to analyse a list of images

        :param images: List of paths for the images
        :param settings: Dictionary containing the necessary information for analysis
        :return: The results as list of dictionaries
        """
        results = []
        start = time.time()
        for path in images:
            results.append(self.analyse_image(path, settings))
            print(f"Analysed image {os.path.basename(path)}")
        self.add_log_message(f"Analysed batch with size {len(images)} in {time.time() - start} seconds")
        self.save_log_messages(Paths.log_path)
        return results

    def analyse_image(self, path: str,
                      settings: Dict[str, Union[List, bool]], save_log: bool = True) -> \
            Dict[str, Union[ROIHandler, np.ndarray, Dict[str, str]]]:
        """
        Method to extract rois from the image given by path

        :param path: The URL of the image
        :param settings: Dictionary containing the necessary information for analysis
        :param save_log: If true, the analysis log will be saved to Paths.log_path
        :return: The analysis results as dict
        """
        analysis_settings = deepcopy(settings["analysis_settings"])
        analysis_settings["log"] = self.add_log_message
        start = time.time()
        imgdat = self.imageloader.get_image_data(path)
        self.analysis_log["Analysed Images"].append(os.path.basename(path))
        self.analysis_log["Messages"][self.analysis_log["Analysed Images"][-1]] = []
        imgdat["id"] = self.imageloader.calculate_image_id(path)
        # Check if only a grayscale image was provided
        if imgdat["channels"] == 1:
            self.add_log_message("Detector class can only analyse multichannel images, not grayscale!")
            raise ValueError("Detector class can only analyse multichannel images, not grayscale!")
        image = self.imageloader.load_image(path)
        names = settings["names"]
        main_channel: int = settings["main"]
        detection_method = analysis_settings["method"]
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
        main_map, main_roi = self.nucleus_extraction(main, names[main_channel], analysis_settings)
        # Define a handler to take the ROI
        handler = ROIHandler(ident=imgdat["id"])
        handler.idents = analysis_settings["names"]
        # Check if nuclei were detected
        if main_roi:
            if detection_method == "image processing" or detection_method == "combined":
                iproi, maps1 = self.ip_roi_extraction(main, main_map, main_roi, foc_channels, analysis_settings)
                self.add_log_message(f"Detected IP ROI: {len(iproi)}")
            if detection_method == "u-net" or detection_method == "combined":
                mlroi, maps2 = self.ml_roi_extraction(main_roi, foc_channels, analysis_settings)
                self.add_log_message(f"Detected ML ROI: {len(mlroi)}")
            rois = []
            if detection_method == "combined":
                # Merge the foci for each channel
                foci = []
                for channel in analysis_settings["foci_channel_names"]:
                    # Define map Comparator
                    mapc = MapComparator(main_roi,
                                         maps1[analysis_settings["foci_channel_names"].index(channel)],
                                         [x for x in iproi if x.ident == channel],
                                         maps2[analysis_settings["foci_channel_names"].index(channel)],
                                         [x for x in mlroi if x.ident == channel],
                                         image.shape[:2],
                                         self.add_log_message)
                    foci.append(mapc.merge_overlapping_foci())
                # Add all foci
                for x in foci:
                    rois.extend(x)
                # Check the foci for co-localisation
                MapComparator.get_match_for_nuclei(main_roi, foci)
            elif detection_method == "image processing":
                rois.extend(iproi)
            else:
                rois.extend(mlroi)
            # Add the detected nuclei to the list
            rois.extend(main_roi)
            # Check for quality of roi
            if rois:
                qroi = self.perform_quality_check(channels, names, analysis_settings, rois)
                self.add_log_message(f"QR: Removed foci: {len(rois) - len(qroi)}")
            else:
                qroi = []
            handler.add_rois(qroi)
        imgdat["x_scale"] = analysis_settings["dots_per_micron"]
        imgdat["y_scale"] = analysis_settings["dots_per_micron"]
        imgdat["scale_unit"] = "µm"
        imgdat["handler"] = handler
        imgdat["names"] = analysis_settings["names"]
        imgdat["channels"] = channels
        imgdat["active channels"] = active
        imgdat["main channel"] = main_channel
        imgdat["add to experiment"] = settings["add_to_experiment"]
        imgdat["experiment details"] = settings["experiment_details"]
        # Remove logging function from settings
        del analysis_settings["log"]
        imgdat["used_settings"] = analysis_settings
        self.add_log_message(f"Total analysis time: {time.time() - start: .4f}")
        if save_log:
            self.save_log_messages(Paths.log_path, True)
            self.clear_log()
        return imgdat

    def nucleus_extraction(self, main_channel: np.ndarray, main_name: str,
                           analysis_settings) -> Tuple[np.ndarray, List[ROI]]:
        """
        Method to extract the nuclei from the main channel

        :param main_channel: The channel containing the nuclei
        :param main_name: The name assigned to the main channel
        :param analysis_settings: The analysis settings to apply
        :return: The main map and the list of detected ROI
        """
        s0 = time.time()
        # Map nuclei
        self.nucleusmapper.set_channels((main_channel,))
        self.nucleusmapper.set_settings(analysis_settings)
        nucmap = self.nucleusmapper.map_nuclei()
        nuclei = extract_nuclei_from_maps(nucmap, main_name)
        for nucleus in nuclei:
            nucleus.detection_method = "Nucleus Detection"
        self.add_log_message(f"Finished nuclei extraction {time.time() - s0:.4f}")
        return nucmap, nuclei

    def ip_roi_extraction(self, main: np.ndarray, main_map: np.ndarray, nuclei: List[ROI],
                          foc_channels: List[np.ndarray], analysis_settings) -> Tuple[List[ROI], List[np.ndarray]]:
        """
        Method to detect nuclei and foci via image processing

        :param main: The channel which should contain nuclei
        :param main_map: The detection map for the main channel
        :param nuclei: List of all detected nuclei
        :param foc_channels: All image channel which potentially contain foci
        :param analysis_settings: The analysis settings to apply
        :return: The extracted ROI and the used detection maps
        """
        s0 = time.time()
        # Map foci
        self.focusmapper.set_channels(foc_channels)
        self.focusmapper.set_settings(analysis_settings)
        foc_maps = self.focusmapper.map_foci(main=main, main_map=main_map)
        self.add_log_message(f"Finished IP foci extraction {time.time() - s0:.4f}")
        roi = Detector.extract_foci_from_maps(nuclei, foc_maps,
                                              analysis_settings["foci_channel_names"])
        for r in roi:
            r.detection_method = "Image Processing"
        if roi:
            return roi, foc_maps
        else:
            return [], [np.zeros(shape=main_map.shape) for _ in range(len(foc_maps))]

    def ml_roi_extraction(self, nuclei: List[ROI], foc_channels,
                          analysis_settings) -> Tuple[List[ROI], List[np.ndarray]]:
        """
        Method to detect nuclei and foci via machine learning

        :param nuclei: List of all detected nuclei
        :param foc_channels: All image channel which potentially contain foci
        :param analysis_settings: The analysis settings to apply
        :return: The extracted ROI
        """
        s0 = time.time()
        # Map nuclei
        self.fcnmapper = FCNMapper()
        self.fcnmapper.set_settings(analysis_settings)
        # Map foci
        self.fcnmapper.set_channels(foc_channels)
        foc_maps = self.fcnmapper.get_marked_maps()
        self.add_log_message(f"Finished ML foci extraction {time.time() - s0:.4f}")
        # Extract roi from maps
        roi = Detector.extract_foci_from_maps(nuclei, foc_maps,
                                              analysis_settings["foci_channel_names"])
        for r in roi:
            r.detection_method = "Machine Learning"
        return roi, foc_maps

    @staticmethod
    def extract_foci_from_maps(nuclei: List[ROI], foci_maps: List[np.ndarray],
                               foc_names: List[str]) -> List[ROI]:
        """
        Method to extract nuclei and foci from the given maps

        :param nuclei: List of detected nuclei
        :param foci_maps: List of maps for foci
        :param foc_names: List of names assigned to the foci channels
        :return: The extracted roi
        """
        foci = []
        for ind, focmap in enumerate(foci_maps):
            foci.extend(extract_foci_from_maps(focmap, foc_names[ind], nuclei))
        return foci

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

    def add_log_message(self, msg: str) -> None:
        """
        Method to add a new log message

        :param msg: The message to log
        :return: None
        """
        self.analysis_log["Messages"][self.analysis_log["Analysed Images"][-1]].append(msg)

    def save_log_messages(self, file_path: str, append: bool = True) -> None:
        """
        Method to save the logs to the given file

        :param file_path: Path leading to the log file
        :param append: If true, the file will be extended instead of overwritten
        :return: None
        """
        with open(file_path, "a+" if append else "rw+") as lf:
            lf.write(f"Date: {self.analysis_log['Date']}\n")
            lf.write(f"Time: {self.analysis_log['Time']}\n")
            lf.write("Analysed Images:\n")
            for img in self.analysis_log["Analysed Images"]:
                lf.write(f"{' ' * 4}{img}\n")
                for msg in self.analysis_log["Messages"][img]:
                    lf.write(f"{' ' * 8}{msg}\n")
            lf.write("#" * 20 + "\n")

    def clear_log(self) -> None:
        """
        Method to clear the internal log

        :return: None
        """
        self.analysis_log["Analysed Images"].clear()
        self.analysis_log["Messages"].clear()
