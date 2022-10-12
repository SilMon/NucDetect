import numpy as np
import warnings
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, Dict

from detector_modules.AreaMapper import AreaMapper
from JittedFunctions import create_lg_lut
from skimage.feature import blob_log
from skimage.draw import disk


class FocusMapper(AreaMapper):
    """
    Class to detect foci on image channels
    """
    STANDARD_SETTINGS = {
        "use_pre-processing": True,
        "min_sigma": 2,
        "max_sigma": 5,
        "num_sigma": 10,
        "acc_thresh": .1,
        "overlap": .10,
        "logging": False
    }

    def get_foci_maps(self) -> Iterable[np.ndarray]:
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

    def map_foci(self) -> Iterable[np.ndarray]:
        """
        Method to detect foci

        :return: The foci detection maps
        """
        foci_maps = []
        # Detect foci on each channel
        for channel in self.channels:
            pchannel = channel if not self.settings["use_pre-processing"] else self.preprocess_channel(channel)
            # Detect foci on preprocessed channel
            foci = self.detect_foci_on_acc_map(self.settings, pchannel)
            # Create foci map and append
            foci_maps.append(self.create_foci_map(pchannel.shape, foci))
        return foci_maps

    @staticmethod
    def preprocess_channel(channel: np.ndarray) -> np.ndarray:
        """
        Method to prepare a channel for focus detection

        :param channel: The channel to pre-process
        :return: The pre-processed channel
        """
        # Get lut for processing
        lut = create_lg_lut(np.amax(channel))
        # Create empty accumulator map
        acc = np.empty(shape=channel.shape)
        for y in range(channel.shape[0]):
            for x in range(channel.shape[1]):
                acc[y][x] = lut[channel[y][x]]
        return channel

    @staticmethod
    def detect_foci_on_acc_map(settings: Dict, acc_map: np.ndarray) -> Iterable[Tuple]:
        """
        Method to detect foci on the given binary map

        :param settings: The settings to use for this method
        :param acc_map: The map to detect foci from
        :return: The detected foci
        """
        # Get needed variables
        min_sigma = settings["min_sigma"]
        max_sigma = settings["max_sigma"]
        num_sigma = settings["num_sigma"]
        acc_thresh = settings["acc_thresh"] if not settings["use_pre-processing"] else 1 - settings["acc_thresh"]
        overlap = settings["overlap"]
        return blob_log(acc_map, min_sigma=min_sigma, max_sigma=max_sigma,
                        num_sigma=num_sigma, threshold=acc_thresh, overlap=overlap)

    @staticmethod
    def create_foci_map(shape: Iterable[int], foci: Iterable) -> np.ndarray:
        """
        Method to create a binary foci map for the given foci

        :param shape: The shape of the original map
        :param foci: The foci to mark on the binary map
        :return: The created foci map
        """
        # Create empty map
        bin_map = np.zeros(shape=shape)
        # Iterate over the given foci
        for ind, focus in enumerate(foci):
            # Extract variables
            y, x, r = focus
            # Draw focus into the foci map
            rr, cc = disk((y, x), r, shape=shape)
            bin_map[rr, cc] = ind
        return bin_map



