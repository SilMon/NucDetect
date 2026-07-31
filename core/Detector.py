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

from core.logging_config import get_logger, log_messages
from core.progress import (NO_PROGRESS, NUCLEUS_BOUNDS, FOCI_IP_BOUNDS, ProgressReporter,
                           stage_bounds, LOAD, NUCLEUS, FOCI_IP, FOCI_ML, MERGE, QUALITY)
from core.detector_modules.AreaAndROIExtractor import extract_nuclei_from_maps, extract_foci_from_maps, \
    extract_foci_from_blobs
from core.detector_modules.FCNMapper import FCNMapper
from core.detector_modules.FocusMapper import FocusMapper
from core.detector_modules.ImageLoader import ImageLoader
from core.detector_modules.MapComparator import MapComparator
from core.detector_modules.NucleusMapper import NucleusMapper
from core.detector_modules.QualityTester import QualityTester
from core.roi.ROI import ROI
from core.roi.ROIHandler import ROIHandler

LOGGER = get_logger(__name__)


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
            LOGGER.info("Analysed image %s", os.path.basename(path))
        self.add_log_message(f"Analysed batch with size {len(images)} in {time.time() - start} seconds")
        self.flush_log_messages()
        return results

    def analyse_image(self, path: str,
                      settings: Dict[str, Union[List, bool]], save_log: bool = True,
                      progress: ProgressReporter = NO_PROGRESS) -> \
            Dict[str, Union[ROIHandler, np.ndarray, Dict[str, str]]]:
        """
        Method to extract rois from the image given by path

        :param path: The URL of the image
        :param settings: Dictionary containing the necessary information for analysis
        :param save_log: If true, the buffered log messages are written to the log. Pass False when
            running inside a worker process and replay the returned messages in the parent instead
        :param progress: Reporter for the stages of this analysis. Defaults to a no-op, so callers
            that do not show a progress bar -- batch analysis, the verification harnesses, any
            direct use of this class -- need pass nothing. It is a parameter rather than an entry
            in ``settings`` on purpose: ``settings`` is deep-copied and stored as ``used_settings``,
            and a callable has no business being serialised into the database
        :return: The analysis results as dict
        """
        analysis_settings = deepcopy(settings["analysis_settings"])
        analysis_settings["log"] = self.add_log_message
        # Each stage reports 0..1 within its own slice of the bar and never learns its position in
        # the whole run. Weights are measured, per method -- see core/progress.py
        bounds = stage_bounds(analysis_settings["method"])
        prg = {stage: progress.sub(*bounds[stage]) for stage in bounds}
        start = time.time()
        prg[LOAD](0.0, "Reading image metadata")
        imgdat = self.imageloader.get_image_data(path)
        self.analysis_log["Analysed Images"].append(os.path.basename(path))
        self.analysis_log["Messages"][self.analysis_log["Analysed Images"][-1]] = []
        prg[LOAD](0.3, "Hashing image")
        imgdat["id"] = self.imageloader.calculate_image_id(path)
        # Check if only a grayscale image was provided
        if imgdat["channels"] == 1:
            self.add_log_message("Detector class can only analyse multichannel images, not grayscale!")
            raise ValueError("Detector class can only analyse multichannel images, not grayscale!")
        prg[LOAD](0.6, "Loading image")
        image = self.imageloader.load_image(path)
        names = settings["names"]
        main_channel: int = settings["main"]
        detection_method = analysis_settings["method"]
        # Channel extraction
        prg[LOAD](0.9, "Splitting channels")
        channels = self.imageloader.get_channels(image)
        active = settings["activated"]
        # Check if all channels are activated
        analysis_settings["names"] = [names[x] for x in range(len(names)) if active[x]]
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
        main_map, main_roi = self.nucleus_extraction(main, names[main_channel], analysis_settings,
                                                     prg[NUCLEUS])
        # Define a handler to take the ROI
        handler = ROIHandler(ident=imgdat["id"])
        handler.idents = analysis_settings["names"]
        # Check if nuclei were detected
        if main_roi:
            if detection_method == "image processing" or detection_method == "combined":
                iproi = self.ip_roi_extraction(main_roi, foc_channels, analysis_settings,
                                               prg[FOCI_IP])
                self.add_log_message(f"Detected IP ROI: {len(iproi)}")
            if detection_method == "u-net" or detection_method == "combined":
                mlroi = self.ml_roi_extraction(main_roi, foc_channels, analysis_settings,
                                               prg[FOCI_ML])
                self.add_log_message(f"Detected ML ROI: {len(mlroi)}")
            rois = []
            if detection_method == "combined":
                # Merge the foci for each channel
                foci = []
                foci_names = analysis_settings["foci_channel_names"]
                for ind, channel in enumerate(foci_names):
                    prg[MERGE](ind / max(1, len(foci_names)),
                               f"Merging foci of channel {channel}")
                    # Define map Comparator
                    mapc = MapComparator(main_roi,
                                         [x for x in iproi if x.ident == channel],
                                         [x for x in mlroi if x.ident == channel],
                                         self.add_log_message)
                    foci.append(mapc.merge_overlapping_foci())
                # Add all foci
                for x in foci:
                    rois.extend(x)
                    # Check the foci for co-localisation TODO
                    MapComparator.get_match_for_nuclei(main_roi, foci)
            elif detection_method == "image processing":
                rois.extend(iproi)
            else:
                rois.extend(mlroi)
            # Add the detected nuclei to the list
            rois.extend(main_roi)
            # Check for quality of roi
            if rois:
                prg[QUALITY](0.0, "Checking ROI quality")
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
        # Hand the buffered messages to the caller before the buffer is dropped. This is what lets
        # a ProcessPoolExecutor worker get its log across to the parent process, which owns the
        # log file -- the worker's own copy of this Detector dies with the task
        imgdat["log"] = self.get_log_messages()
        if save_log:
            self.flush_log_messages()
        else:
            # Always clear, even when not writing: without this a worker would accumulate the
            # messages of every image it ever handled and repeat them in each result
            self.clear_log()
        return imgdat

    def nucleus_extraction(self, main_channel: np.ndarray, main_name: str,
                           analysis_settings,
                           progress: ProgressReporter = NO_PROGRESS) -> Tuple[np.ndarray, List[ROI]]:
        """
        Method to extract the nuclei from the main channel

        :param main_channel: The channel containing the nuclei
        :param main_name: The name assigned to the main channel
        :param analysis_settings: The analysis settings to apply
        :param progress: Reporter owning the whole nucleus stage. The mapper reports the first five
            sub-stages of NUCLEUS_BOUNDS, this method reports the sixth ("extract")
        :return: The main map and the list of detected ROI
        """
        s0 = time.time()
        # Map nuclei
        self.nucleusmapper.set_channels((main_channel,))
        self.nucleusmapper.set_settings(analysis_settings)
        self.nucleusmapper.set_progress(progress)
        try:
            nucmap = self.nucleusmapper.map_nuclei()
        finally:
            # The reporter belongs to one analysis, not to the mapper. Clearing it also keeps a
            # live callback -- a bound method of the main window during single-image analysis --
            # from outliving the run on a Detector that batch analysis later tries to pickle
            self.nucleusmapper.set_progress(NO_PROGRESS)
        progress.span("extract", NUCLEUS_BOUNDS)(0.0, "Extracting nuclei")
        nuclei = extract_nuclei_from_maps(nucmap, main_name)
        for nucleus in nuclei:
            nucleus.detection_method = "Nucleus Detection"
        self.add_log_message(f"Finished nuclei extraction {time.time() - s0:.4f}")
        return nucmap, nuclei

    def ip_roi_extraction(self, nuclei: List[ROI],
                          foc_channels: List[np.ndarray], analysis_settings,
                          progress: ProgressReporter = NO_PROGRESS) -> List[ROI]:
        """
        Method to detect nuclei and foci via image processing

        :param nuclei: List of all detected nuclei
        :param foc_channels: All image channel which potentially contain foci
        :param analysis_settings: The analysis settings to apply
        :param progress: Reporter owning the image-processing foci stage. The mapper subdivides it
            per channel; the blob extraction that follows is the tail of each channel's share
        :return: The extracted ROI and the used detection maps
        """
        s0 = time.time()
        # Map foci
        self.focusmapper.set_channels(foc_channels)
        self.focusmapper.set_settings(analysis_settings)
        # The mapper owns everything up to the blob extraction, which happens here
        self.focusmapper.set_progress(progress.sub(0.0, FOCI_IP_BOUNDS["extract"][0]))
        try:
            ip_foci = self.focusmapper.map_foci()
        finally:
            # See nucleus_extraction: the reporter must not outlive the analysis
            self.focusmapper.set_progress(NO_PROGRESS)
        progress.span("extract", FOCI_IP_BOUNDS)(0.0, "Extracting foci")
        roi = Detector.extract_foci_from_blobs(nuclei, ip_foci,
                                               analysis_settings["foci_channel_names"],
                                               image_shape=foc_channels[0].shape)
        self.add_log_message(f"Finished IP foci extraction {time.time() - s0:.4f}")
        for r in roi:
            r.detection_method = "Image Processing"
        if roi:
            return roi
        else:
            return []

    def ml_roi_extraction(self, nuclei: List[ROI], foc_channels,
                          analysis_settings,
                          progress: ProgressReporter = NO_PROGRESS) -> List[ROI]:
        """
        Method to detect nuclei and foci via machine learning

        :param nuclei: List of all detected nuclei
        :param foc_channels: All image channel which potentially contain foci
        :param analysis_settings: The analysis settings to apply
        :param progress: Reporter owning the u-net foci stage. Only the per-channel boundaries are
            reported; the inference itself is a single ``model.predict`` call that this method
            cannot see into. Subdividing it would need a Keras callback and a smaller batch size,
            which trades inference throughput for responsiveness and has not been measured
        :return: The extracted ROI
        """
        s0 = time.time()
        progress(0.0, "Loading detection model")
        # Map nuclei
        self.fcnmapper = FCNMapper()
        self.fcnmapper.set_settings(analysis_settings)
        # Map foci
        self.fcnmapper.set_channels(foc_channels)
        self.fcnmapper.set_progress(progress.sub(0.05, 0.9))
        try:
            foc_maps = self.fcnmapper.get_marked_maps()
        finally:
            # See nucleus_extraction: the reporter must not outlive the analysis
            self.fcnmapper.set_progress(NO_PROGRESS)
        self.add_log_message(f"Finished ML foci extraction {time.time() - s0:.4f}")
        # Extract roi from maps
        progress(0.9, "Extracting foci")
        roi = Detector.extract_foci_from_maps(nuclei, foc_maps,
                                              analysis_settings["foci_channel_names"])
        for r in roi:
            r.detection_method = "Machine Learning"
        return roi

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

    @staticmethod
    def extract_foci_from_blobs(nuclei: List[ROI],
                                foci_blobs: List[List[Tuple[int, int, int]]],
                                foc_names: List[str],
                                image_shape: Tuple[int, ...]) -> List[ROI]:
        """
        Method to extract nuclei and foci from the given maps

        :param nuclei: List of detected nuclei
        :param foci_blobs: List of all detected foci as blobs
        :param foc_names: List of names assigned to the foci channels
        :param image_shape: Shape of the image
        :return: The extracted roi
        """
        foci = []
        for ind, focus_blobs in enumerate(foci_blobs):
            foci.extend(extract_foci_from_blobs(focus_blobs,
                                                foc_names[ind],
                                                nuclei,
                                                image_shape))
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

        Messages are buffered instead of logged straight away: this method also runs inside
        ProcessPoolExecutor workers, and buffering lets the parent process replay them in image
        order via get_log_messages instead of several processes appending to the log file at once.

        :param msg: The message to log
        :return: None
        """
        self.analysis_log["Messages"][self.analysis_log["Analysed Images"][-1]].append(msg)

    def get_log_messages(self) -> List[str]:
        """
        Method to get the buffered log messages as a list of formatted lines

        Returned with the analysis result so the messages of a worker process can be replayed by
        the parent, which owns the log file.

        :return: The formatted log lines, in the order the messages were added
        """
        lines = [f"Date: {self.analysis_log['Date']}",
                 f"Time: {self.analysis_log['Time']}",
                 "Analysed Images:"]
        for img in self.analysis_log["Analysed Images"]:
            lines.append(f"{' ' * 4}{img}")
            for msg in self.analysis_log["Messages"][img]:
                lines.append(f"{' ' * 8}{msg}")
        return lines

    def flush_log_messages(self) -> None:
        """
        Method to write the buffered log messages to the log and to clear the buffer

        Replaces the former save_log_messages, which opened gui.Paths.log_path itself. Output now
        goes through the shared logger configured by core.logging_config, which owns the only
        handle on the log file and applies the UTF-8 encoding the image file names in these
        messages require.

        :return: None
        """
        log_messages(self.get_log_messages())
        self.clear_log()

    def clear_log(self) -> None:
        """
        Method to clear the internal log

        :return: None
        """
        self.analysis_log["Analysed Images"].clear()
        self.analysis_log["Messages"].clear()
