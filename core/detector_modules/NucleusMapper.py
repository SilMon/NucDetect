import warnings
from typing import List

import numpy as np
from scipy import ndimage as ndi
from skimage import img_as_ubyte
from skimage.filters import threshold_local
from skimage.filters.rank import maximum
from skimage.morphology import opening
from skimage.segmentation import watershed

from core.DataProcessing import create_circular_mask
from core.detector_modules.AreaMapper import AreaMapper
from core.progress import ITERMAX_BOUNDS, NUCLEUS_BOUNDS, NO_PROGRESS


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

        Reports the first five sub-stages of NUCLEUS_BOUNDS; the sixth ("extract") belongs to
        Detector.nucleus_extraction. Nucleus extraction is the largest single block of a warm
        analysis -- 61 % of an image-processing run -- so without these emits the bar stands still
        for around seven seconds.

        :return: The map of detected nuclei
        """
        # Threshold channel
        self.progress.span("threshold", NUCLEUS_BOUNDS)(0.0, "Thresholding main channel")
        thresh = self.threshold_map()
        # Calculate normalized euclidean distance map
        self.progress.span("edm", NUCLEUS_BOUNDS)(0.0, "Calculating distance map")
        edm = self.calculate_edm_and_normalize(thresh)
        # Create iterative maximum map
        it_max = self.get_iterative_max_map(edm, thresh)
        # TODO it_max anstelle von edm übergeben
        # Get the center mask based on it_max
        cmask = self.create_center_mask(it_max, self.progress.span("centers", NUCLEUS_BOUNDS))
        # Perform watershed segmentation and return
        self.progress.span("watershed", NUCLEUS_BOUNDS)(0.0, "Segmenting nuclei")
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
        # Check for sufficient dynamic range
        # TODO begründete Lösung finden
        if np.amax(self.channels[0]) - np.amin(self.channels[0]) < 30:
            return np.zeros(shape=self.channels[0].shape, dtype=bool)
        return ndi.binary_fill_holes(self.channels[0] > threshold)

    @staticmethod
    def calculate_edm_and_normalize(bin_map: np.ndarray) -> np.ndarray:
        """
        Method to calculate the Euclidean distance map (EDM) of the given binary map and normalize it

        :param bin_map: The binary map to calculate the EDM from
        :return: The EDM
        """
        edm = ndi.distance_transform_edt(bin_map)
        # Normalize edm
        xmax, xmin = edm.max(), edm.min()
        span = xmax - xmin
        # A constant distance map has no range to stretch, and the caller reaches this on purpose:
        # threshold_map deliberately returns an all-zero binary map for a channel whose dynamic
        # range is under 30, which makes the whole EDM zero and the division 0/0. Unguarded it
        # produced an array of NaN and then cast it to uint8 -- which is UNDEFINED in C. It
        # happens to yield the all-zero map that is in fact correct here, but only by accident of
        # this platform; nothing promises the NaN will not land on 255 and hand the watershed a
        # uniformly white distance map. Say the intended answer instead of relying on the cast.
        # Note a single-pixel nucleus is NOT degenerate: it gives xmin 0, xmax 1 and normalises
        if not span:
            return np.zeros(edm.shape, np.uint8)
        return img_as_ubyte((edm - xmin) / span)

    def get_iterative_max_map(self, edm: np.ndarray, binary_map: np.ndarray) -> np.ndarray:
        """
        Calculates the iterative maximum map of the given image

        This method is the single most expensive thing the application does: measured at 76 % of
        nucleus extraction and 46 % of a whole warm image-processing analysis, of which the
        `maximum` calls below are the great majority. Each costs about the same, so the loop is the
        one place in the analysis that can report smooth, evenly spaced progress -- which is why it
        emits per iteration rather than only at its boundaries.

        :param edm: The euclidean distance map to calculate the iterative maximum map for
        :param binary_map: The original binary map
        :return: The max map
        """
        progress = self.progress.span("itermax", NUCLEUS_BOUNDS)
        mask_size = self.settings["mask_size"]
        size_factor = self.settings["size_factor"]
        iterations = self.settings["iterations"]
        maximum_size_multiplier = self.settings["maximum_size_multiplier"]
        local_threshold_multiplier = self.settings["local_threshold_multiplier"]
        mask = create_circular_mask(mask_size * size_factor, mask_size * size_factor)
        progress.span("seed", ITERMAX_BOUNDS)(0.0, "Calculating maximum map")
        maxi = maximum(edm, footprint=mask)
        # The loop's share is divided by the configured iteration count rather than a fixed number:
        # `iterations` is a user setting, and every pass costs the same
        loop = progress.span("loop", ITERMAX_BOUNDS)
        ind = 0
        while ind < iterations:
            loop(ind / iterations, f"Calculating maximum map ({ind + 1}/{iterations})")
            maxi = maximum(maxi, mask)
            ind += 1
        progress.span("threshold_local", ITERMAX_BOUNDS)(0.0, "Applying local threshold")
        # Scale first, then force the result odd. threshold_local requires an odd integer block
        # size, and the +1 that supplied it used to sit INSIDE the parenthesis, so the scaling
        # applied afterwards undid the property it was there to guarantee. Measured with the
        # default mask_size 7 and multiplier 8: size_factor 2.0 gave 114.0 and raised "block_size
        # must be odd!", while 1.5 gave 85.5 and was accepted silently -- the check is
        # `block_size % 2 == 0`, which a non-integer never satisfies, so a fractional window walks
        # straight past it. `| 1` rounds up to the next odd integer and cannot produce an even or
        # fractional value for any positive factor. At the default size_factor of 1.0 this is 57,
        # exactly what the old expression produced
        block_size = int(mask_size * local_threshold_multiplier * size_factor) | 1
        thresh = threshold_local(maxi, block_size=block_size)
        progress.span("fill", ITERMAX_BOUNDS)(0.0, "Filling holes")
        maxi = ndi.binary_fill_holes(maxi > thresh)
        maxi = np.logical_and(maxi, binary_map)
        progress.span("opening", ITERMAX_BOUNDS)(0.0, "Removing artefacts")
        maxi = opening(maxi, footprint=create_circular_mask(mask_size * maximum_size_multiplier * size_factor,
                                                            mask_size * maximum_size_multiplier * size_factor))
        return maxi

    @staticmethod
    def create_center_mask(max_it: np.ndarray, progress=NO_PROGRESS) -> np.ndarray:
        """
        Method to create a center mask for watershed segmentation

        :param max_it: The iterative maximum map
        :param progress: Reporter owning the "centers" sub-stage. Defaults to a no-op so existing
            callers keep working unchanged
        :return: The nucleus extraction map
        """
        # Label individual areas of max_it
        area_map, labels = ndi.label(max_it)
        # Extract individual areas
        nucs: List[List, List] = [None] * (labels + 1)
        # This pure-Python pass touches every pixel and is ~92 % of this method (0.72 s of 0.78 s
        # on a 1024x1024 image), so it reports per row rather than only at the ends. Emitting on
        # every row would flood the signal queue for no visible gain, hence the step
        rows = len(area_map)
        step = max(1, rows // 20)
        for y in range(rows):
            if y % step == 0:
                progress(y / rows, "Locating nucleus centres")
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
        cmask = np.zeros(shape=max_it.shape, dtype=np.uint32)
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
        :param edm: The Euclidean distance map of the map
        :param cmask: A map marking all centers
        :param mask: The binary map to segment
        :param line: Toggle to draw a line between segmented areas
        :return: The segmented binary map
        """
        # Create watershed segmentation based on centers
        return watershed(-edm, cmask, mask=mask, watershed_line=line)

