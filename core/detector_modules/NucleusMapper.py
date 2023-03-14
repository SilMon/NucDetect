import warnings
from typing import List

import numpy as np
from scipy import ndimage as ndi
from skimage import img_as_ubyte
from skimage.filters import threshold_local
from skimage.filters.rank import maximum
from skimage.morphology.binary import binary_opening
from skimage.segmentation import watershed

from DataProcessing import create_circular_mask
from detector_modules.AreaMapper import AreaMapper


class NucleusMapper(AreaMapper):
    """
    Class to detect foci on image channels
    """
    STANDARD_SETTINGS = {
        "iterations": 5,
        "mask_size": 7,
        "percent_hmax": 0.05,
        "local_threshold_multiplier": 8,
        "maximum_size_multiplier": 2,
        "size_factor": 1.0,
        "logging": False
    }

    def get_nucleus_maps(self) -> np.ndarray:
        """
        Method to create the nucleus map for the given channel

        :return: The created foci maps
        """
        # Check if channels were set
        if not self.channels:
            raise ValueError("No channel was set to map the nuclei on!")
        if len(self.channels) > 1:
            raise ValueError("Multiple channels given as nucleus channel!")
        # Check if settings contain anything
        if not self.settings:
            self.settings = self.STANDARD_SETTINGS
            warnings.warn("No settings found, standard settings used for nucleus mapping")
        return self.map_nuclei()

    def map_nuclei(self) -> np.ndarray:
        """
        Function to map the nuclei on the given main channel

        :return: The map of detected nuclei
        """
        # Threshold channel
        thresh = self.threshold_map()
        # Calculate normalized euclidean distance map
        edm = self.calculate_edm_and_normalize(thresh)
        # Create iterative maximum map
        it_max = self.get_iterative_max_map(edm, thresh)
        # Get the center mask based on it_max
        cmask = self.create_center_mask(it_max)
        # Perform watershed segmentation and return
        return self.perform_watershed_segmentation(edm, cmask, thresh, True)

    def threshold_map(self) -> np.ndarray:
        """
        Method to threshold the given main channel

        :return: The created binary map
        """
        # Get needed variables
        percent_hmax = self.settings["percent_hmax"]
        # Calculate the threshold to use
        threshold = np.amin(self.channels[0]) + round(percent_hmax * np.amax(self.channels[0]))
        return ndi.binary_fill_holes(self.channels[0] > threshold)

    @staticmethod
    def calculate_edm_and_normalize(bin_map: np.ndarray) -> np.ndarray:
        """
        Method to calculate the euclidean distance map (EDM) of the given binary map and normalize it

        :param bin_map: The binary map to calculate the EDM from
        :return: The EDM
        """
        edm = ndi.distance_transform_edt(bin_map)
        # Normalize edm
        xmax, xmin = edm.max(), edm.min()
        return img_as_ubyte((edm - xmin) / (xmax - xmin))

    def get_iterative_max_map(self, edm: np.ndarray, binary_map: np.ndarray) -> np.ndarray:
        """
        Calculates the iterative maximum map of the given image

        :param edm: The euclidean distance map to calculate the iterative maximum map for
        :param binary_map: The original binary map
        :return: The max map
        """
        mask_size = self.settings["mask_size"]
        size_factor = self.settings["size_factor"]
        iterations = self.settings["iterations"]
        maximum_size_multiplier = self.settings["maximum_size_multiplier"]
        local_threshold_multiplier = self.settings["local_threshold_multiplier"]
        mask = create_circular_mask(mask_size * size_factor, mask_size * size_factor)
        maxi = maximum(edm, footprint=mask)
        ind = 0
        while ind < iterations:
            maxi = maximum(maxi, mask)
            ind += 1
        thresh = threshold_local(maxi, block_size=(mask_size * local_threshold_multiplier + 1) * size_factor)
        maxi = ndi.binary_fill_holes(maxi > thresh)
        maxi = np.logical_and(maxi, binary_map)
        maxi = binary_opening(maxi, footprint=create_circular_mask(mask_size * maximum_size_multiplier * size_factor,
                                                                   mask_size * maximum_size_multiplier * size_factor))
        return maxi

    @staticmethod
    def create_center_mask(max_it: np.ndarray) -> np.ndarray:
        """
        Method to create a center mask for watershed segmentation

        :param max_it: The iterative maximum map
        :return: The nucleus extraction map
        """
        # Label individual areas of max_it
        area_map, labels = ndi.label(max_it)
        # Extract individual areas
        nucs: List[List, List] = [None] * (labels + 1)
        for y in range(len(area_map)):
            for x in range(len(area_map[0])):
                pix = area_map[y][x]
                if nucs[pix] is None:
                    nucs[pix] = [[], []]
                nucs[pix][0].append(y)
                nucs[pix][1].append(x)
        # Remove background
        del nucs[0]
        # Calculate center of each detected nucleus
        centers = [(np.average(x[0]), np.average(x[1])) for x in nucs]
        # Create center map as starting point for watershed segmentation
        cmask = np.zeros(shape=max_it.shape)
        ind = 1
        for c in centers:
            cmask[int(c[0])][int(c[1])] = ind
            ind += 1
        return cmask

    @staticmethod
    def perform_watershed_segmentation(edm: np.ndarray, cmask: np.ndarray,
                                       mask: np.ndarray, line: bool) -> np.ndarray:
        """
        Method to perform watershed segmentation on the given mask
        :param edm: The euclidean distance map of the map
        :param cmask: A map marking all centers
        :param mask: The binary map to segment
        :param line: Toggle to draw a line between segmented areas
        :return: The segmented binary map
        """
        # Create watershed segmentation based on centers
        return watershed(-edm, cmask, mask=mask, watershed_line=line)

