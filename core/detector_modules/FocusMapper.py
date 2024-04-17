import warnings
from math import sqrt
from typing import Iterable, Tuple, Dict, List

import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import disk
from skimage.feature import blob_log

from DataProcessing import create_lg_lut, automatic_whitebalance
from detector_modules.AreaMapper import AreaMapper


class FocusMapper(AreaMapper):
    """
    Class to detect foci on image channels
    """
    ___slots__ = (
        "channels",
        "settings",
        "main"
    )
    STANDARD_SETTINGS = {
        "use_pre-processing": True,
        "dots_per_micron": 1.3938,
        "smoothing": 3,
        "min_sigma": 1.1,
        "max_sigma": 2.5,
        "num_sigma": 10,
        "acc_thresh": .1,
        "overlap": .10,
        "logging": False
    }

    def get_foci_maps(self) -> List[np.ndarray]:
        """
        Method to create foci maps from the given channels
        :return: The created foci maps
        """
        # Check if channels were set
        if not self.channels:
            raise ValueError("No channels were set to map foci on!")
        # Check if settings contain anything
        if not self.settings:
            self.settings = self.STANDARD_SETTINGS
            warnings.warn("No settings found, standard settings used for focus mapping")
        return self.map_foci()

    def map_foci(self, main: np.ndarray = None, main_map: np.ndarray = None) -> List[np.ndarray]:
        """
        Method to detect foci

        :param main: The channel containing the nucleus
        :param main_map: The binary main map
        :return: The foci detection maps
        """
        foci_maps = []
        # Detect foci on each channel
        for channel in self.channels:
            pchannel = channel if not self.settings["use_pre-processing"]\
                else self.preprocess_channel(channel, main, main_map)
            # Detect foci on preprocessed channel
            foci = self.detect_foci_on_acc_map(self.settings, pchannel)
            # Create foci map and append
            foci_maps.append(self.create_foci_map(pchannel.shape, foci))
        return foci_maps

    @staticmethod
    def preprocess_channel(channel: np.ndarray,
                           main: np.ndarray = None,
                           main_map: np.ndarray = None,
                           adjust_color_balance: bool = True) -> np.ndarray:
        """
        Method to prepare a channel for focus detection

        :param channel: The channel to pre-process
        :param main: The channel containing the nuclei
        :param main_map: Binary map of the main channel
        :param adjust_color_balance: Should the image be scaled to use the full range of values?
        :return: The pre-processed channel
        """
        # Get lut for processing
        if main is None or main_map is None:
            warnings.warn("No main map given, increase in processing time expected!")
        else:
            img = channel if not adjust_color_balance else automatic_whitebalance(channel)
            lut = create_lg_lut(np.amax(img))
            # Create empty accumulator map
            acc = np.empty(shape=channel.shape)
            for y in range(channel.shape[0]):
                for x in range(channel.shape[1]):
                    acc[y][x] = lut[img[y][x]]
            return acc

    @staticmethod
    def check_for_preprocessing(main: np.ndarray, channel: np.ndarray) -> Tuple[bool, bool]:
        """
        Method to check if the channel should be pre-processed or not

        :param main: Binary image of the main channel
        :param channel: The focus channel to test
        :return: True if pre-processing should be applied, True if the image should be smoothed beforehand
        """
        hist_raw = []
        for y in range(channel.shape[0]):
            for x in range(channel.shape[1]):
                if main[y][x]:
                    hist_raw.append(channel[y][x])
        # Calculate the histogram
        hist, counts = np.unique(hist_raw, return_counts=True)
        # Calculate the percentage histogram
        sum_ = sum(counts)
        phist = [x / sum_ * 100 for x in counts]
        # Check the first 15% of the histogram
        return sum(phist[:int(len(phist) * 0.15)]) > 45

    @staticmethod
    def detect_foci_on_acc_map(settings: Dict, acc_map: np.ndarray) -> List[Tuple]:
        """
        Method to detect foci on the given binary map

        :param settings: The settings to use for this method
        :param acc_map: The map to detect foci from
        :return: The detected foci
        """
        # Get needed variables
        mmpd = settings["dots_per_micron"]
        # TODO fix conversion via mmpd
        min_sigma = settings["min_sigma"]
        max_sigma = settings["max_sigma"]
        num_sigma = settings["num_sigma"]
        acc_thresh = settings["acc_thresh"]
        overlap = settings["overlap"]
        # TODO fix
        return blob_log(acc_map,
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        num_sigma=num_sigma, threshold=acc_thresh, overlap=overlap)

    @staticmethod
    def create_foci_map(shape: Tuple[int], foci: Iterable) -> np.ndarray:
        """
        Method to create a binary foci map for the given foci

        :param shape: The shape of the original map
        :param foci: The foci to mark on the binary map
        :return: The created foci map
        """
        # Create empty map
        bin_map = np.zeros(shape=shape,
                           dtype=np.uint32)
        tsq = sqrt(2)
        # Iterate over the given foci
        for ind, focus in enumerate(foci):
            # Extract variables
            y, x, r = focus
            # Draw focus into the foci map
            rr, cc = disk((y, x), r * tsq, shape=shape)
            bin_map[rr, cc] = ind + 1
        return bin_map
