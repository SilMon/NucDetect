import time
import warnings
from typing import Dict, Union, List, Iterable, Tuple, Any

import numpy as np
from numpy import ndarray

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
        "min_foc_int": .055,
        "min_foc_cont": .005,
        "cutoff": .03,
        "size_factor": 1.0,
        "logging": False,
        "log": []
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
        self.log = self.settings["log"]
        return self.check_quality()

    def check_quality(self) -> Tuple[List[ROI], List[ROI]]:
        """
        Method to check the quality of given nuclei/foci

        :return: The checked roi
        """
        main, foci = self.separate_nuclei_and_foci()
        # Check size of nuclei
        lower_bound, upper_bound = self.settings["min_main_area"], self.settings["max_main_area"]
        main = self.check_size_boundaries(main, lower_bound, upper_bound)
        self.log("Quality Check:")
        self.log(f"Nuclei Size Check: {len(main)}")
        # Delete foci whose nucleus was deleted or which are unassociated to a nucleus
        self.log(f"Foci to check: {len(foci)}")
        foci = self.delete_unassociated_foci(main, foci)
        self.log(f"Focus Association Check: {len(foci)}")
        # Check size of foci
        foci = self.check_size_boundaries(foci, self.settings["min_foc_area"], self.settings["max_foc_area"])
        self.log(f"Focus Size Check: {len(foci)}")
        # Check foci for intensity
        foci = self.check_intensity_boundaries(foci, self.settings["min_foc_int"], 1)
        self.log(f"Focus Intensity Check: {len(foci)}")
        foci = self.check_focus_contrast(foci, self.settings["min_foc_cont"])
        self.log(f"Focus Contrast Check: {len(foci)}")
        return main, foci

    def separate_nuclei_and_foci(self) -> Tuple[List[ROI], List[ROI]]:
        """
        Method to separate nuclei and foci from an unsorted list of roi

        :return: A list of all nuclei, a list of all foci
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
        # TODO check
        # Size factor gives the pix/mikro m ; area is given in pix
        return [x for x in roi if lower_bound <= x.calculate_dimensions()["area"] /
                self.settings["size_factor"] <= upper_bound]

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

    def _get_values_dict(self) -> dict[str | Any, dict[str, ndarray | int | Any]]:
        """
        Method to get an info dict for alle focus channels

        :return: The created dictionary
        """
        names = self.channel_names[:len(self.channels)]
        return {x: {"Channel": self.channels[names.index(x)],
                    "Lower": np.amin(self.channels[names.index(x)]),
                    "Upper": np.amax(self.channels[names.index(x)]),
                    "Max. Val": np.iinfo(self.channels[names.index(x)].dtype).max}
                for x in names}

    def check_focus_contrast(self,
                             foci: List[ROI],
                             min_contrast: float) -> List[ROI]:
        """
        Method to check the focus contrast

        :param foci: The foci to check
        :param min_contrast: The contrast percentage
        :return: The check ROI
        """
        checked = []
        # Get the values for the foci channels
        values = self._get_values_dict()
        # Check each focus individually
        for roi in foci:
            channel = values[roi.ident]["Channel"]
            # Calculate average intensity
            intensity = roi.calculate_statistics(channel)["intensity average"]
            dims = roi.calculate_dimensions()
            fcy, fcx = dims["center_y"], dims["center_x"]
            # Approximation of radius
            fr = max((dims["maxX"] - dims["minX"]) // 2, dims["maxY"] - dims["minY"])
            arr = 3
            if fcy < fr + arr or fcx < fr + arr:
                continue
            # Get area around center
            area = channel[fcy - fr - arr: fcy + fr + arr,
                   fcx - fr - arr: fcx + fr + arr]
            # Get mask
            mask = np.ones(shape=area.shape)
            # Get area shape
            acy, acx = area.shape
            acy = acy // 2
            acx = acx // 2
            # Set focus area to zero
            mask[acy - fr: acy + fr,
            acx - fr: acx + fr] = 0
            # Calculate the average of the surrounding area
            avg = 0
            num = 0
            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    if mask[y][x]:
                        avg += int(area[y][x])
                        num += 1
            if avg == 0 or num == 0:
                continue
            avg /= num
            # If the focus intensity is smaller than its surroundings, it is no focus
            if intensity < avg:
                continue
            # Check if the contrast
            elif intensity - avg > values[roi.ident]["Max. Val"] * min_contrast:
                checked.append(roi)
        return checked

    def check_intensity_boundaries(self,
                                   foci: List[ROI],
                                   lower_bound: float,
                                   upper_bound: float = None) -> List[ROI]:
        """
        Method to check if the intensity of the ROI lies in the specified boundaries

        :param foci: The foci to check
        :param lower_bound: The lower boundary as percent of image max
        :param upper_bound: The upper boundary as percent of image max
        :return: The checked ROI
        """
        # Iterate over the given roi to check if their intensity is inside the bounds
        checked = []
        values = self._get_values_dict()
        # Set the needed boundaries
        for key in values.keys():
            values[key]["Lower"] = values[key]["Lower"] + values[key]["Upper"] * lower_bound
            values[key]["Upper"] = values[key]["Upper"] * upper_bound
        for roi in foci:
            # Get the corresponding channel
            channel = values[roi.ident]["Channel"]
            lower = values[roi.ident]["Lower"]
            upper = values[roi.ident]["Upper"]
            # Calculate average intensity
            intensity = roi.calculate_statistics(channel)["intensity average"]
            if lower <= intensity <= upper:
                checked.append(roi)
        return checked

