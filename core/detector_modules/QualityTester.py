import logging
import warnings
from typing import Dict, Union, List, Iterable, Tuple

import numpy as np

from roi.ROI import ROI


class QualityTester:
    """
    Class to check the quality of found nuclei and foci
    """
    STANDARD_SETTINGS = {
        "max_channel_intensity": 255,
        "max_focus_overlap": .75,
        "min_main_area": 1000,
        "max_main_area": 30000,
        "min_nucleus_int_perc": .8,
        "min_foc_area": 5,
        "max_foc_area": 270,
        "min_foc_int": .2,
        "cutoff": .03,
        "size_factor": 1.0,
        "logging": False,
    }

    def __init__(self, channels: List[np.ndarray] = None, channel_names: List[str] = None,
                 roi: Iterable[ROI] = None, settings: Dict[str, Union[str, int, float]] = None):
        self.channels = channels
        self.channel_names = channel_names
        self.roi = roi
        self.settings = settings

    def set_channels(self, channels: List[np.ndarray]) -> None:
        self.channels = channels

    def set_channel_names(self, channel_names: Iterable[str]) -> None:
        self.channel_names = channel_names

    def set_roi(self, roi: List[ROI]) -> None:
        self.roi = roi

    def set_settings(self, settings: Dict) -> None:
        self.settings = settings

    def check_roi_quality(self) -> Tuple[List[ROI], List[ROI]]:
        """
        Method to check the quality of the saved ROI

        :return: A list containg both the nuclei and foci
        """
        # Check if channels were set
        if not self.channels:
            raise ValueError("No channels were given for quality check!")
        # Check if the roi were set
        if not self.roi:
            raise ValueError("No roi were given!")
        # Check if settings contain anything
        if not self.settings:
            self.settings = self.STANDARD_SETTINGS
            warnings.warn("No settings found, standard settings used for focus mapping")
        return self.check_quality()

    def check_quality(self) -> Tuple[List[ROI], List[ROI]]:
        """
        Method to check the quality of given nuclei/foci

        :return: The checked roi
        """
        main, foci = self.separate_roi_by_channel()
        # Check size of nuclei
        lower_bound, upper_bound = self.settings["min_main_area"], self.settings["max_main_area"]
        #main = self.check_size_boundaries(main, lower_bound, upper_bound)
        print(f"Quality Check:\nNuclei Size Check: {len(main)}")
        # Delete foci whose nucleus was deleted or which are unassociated to a nucleus
        foci = self.delete_unassociated_foci(main, foci)
        print(f"Focus Association Check: {len(foci)}")
        # Check size of foci
        foci = self.check_size_boundaries(foci, self.settings["min_foc_area"], self.settings["max_foc_area"])
        print(f"Focus Size Check: {len(foci)}")
        # Check foci for intensity
        foci = self.check_intensity_boundaries(foci, self.settings["min_foc_int"], 1)
        print(f"Focus Intensity Check: {len(foci)}")
        return main, foci

    def separate_roi_by_channel(self) -> Tuple[List[ROI], List[ROI]]:
        """
        Method to separate nuclei and foci from an unsorted list of roi

        :return: The sorted roi
        """
        main = []
        foci = []
        for roi in self.roi:
            if roi.main:
                main.append(roi)
            else:
                foci.append(roi)
        return main, foci

    def check_size_boundaries(self, roi: List[ROI], lower_bound: int, upper_bound: int) -> List[ROI]:
        """
        Method to check if the area of a roi lies inside the specified boundaries

        :param roi: List of roi to check
        :param lower_bound: Lower threshold
        :param upper_bound: Upper threshold
        :return: List of ROI that are larger than lower_bound and smaller than upper_bound
        """
        lower_bound *= self.settings["size_factor"]
        upper_bound *= self.settings["size_factor"]
        return [x for x in roi if lower_bound <= x.calculate_dimensions()["area"] <= upper_bound]

    @staticmethod
    def delete_unassociated_foci(nuclei: List[ROI], foci: List[ROI]) -> List[ROI]:
        """
        Method to remove unassiciated foci

        :param nuclei: The detected nuclei
        :param foci: The detected foci
        :return: List of associated foci
        """
        nuclei_hashes = [hash(x) for x in nuclei]
        checked_foci = []
        for focus in foci:
            if hash(focus.associated) in nuclei_hashes:
                checked_foci.append(focus)
        return checked_foci

    def check_intensity_boundaries(self, rois: List[ROI], lower_bound: int, upper_bound: int = None) -> List[ROI]:
        """
        Method to check if the intensity of the ROI lies in the specified boundaries

        :param rois: The ROI to check
        :param lower_bound: The lower boundary
        :param upper_bound: The upper boundary
        :return: The checked ROI
        """
        # TODO implement properly
        lower_bound *= 255#self.settings["max_channel_intensity"]
        upper_bound *= 255#self.settings["max_channel_intensity"]
        # Iterate over the given roi to check if their intensity is inside the bounds
        checked = []
        for roi in rois:
            # Get the corresponding channel
            channel = self.channels[self.channel_names.index(roi.ident)]
            # Calculate average intensity
            intensity = roi.calculate_statistics(channel)["intensity average"]
            if lower_bound <= intensity <= upper_bound:
                checked.append(roi)
        return checked

