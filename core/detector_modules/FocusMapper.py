import warnings
from math import sqrt
from typing import Iterable, Tuple, Dict, List, Any

import numpy as np
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet
from skimage.util import img_as_float
from skimage.draw import disk
from skimage.feature import blob_log
from skimage.filters import gaussian, unsharp_mask, butterworth
from skimage.filters.rank import mean, median
from skimage.morphology import white_tophat

from core.detector_modules.AreaMapper import AreaMapper


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
        "use_smoothing": False,
        "use_background_reduction": False,
        "use_signal_improvement": False,
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

    def map_foci(self) -> List[np.ndarray]:
        """
        Method to detect foci

        :return: The foci detection maps
        """
        foci_maps = []
        # Detect foci on each channel
        for channel in self.channels:
            # Perform pre-processing
            pchannel = self.preprocess_channel(channel, self.settings)
            # Detect foci on preprocessed channel
            foci = self.detect_foci_on_acc_map(self.settings, pchannel)
            # Create foci map and append
            foci_maps.append(self.create_foci_map(pchannel.shape, foci))
        return foci_maps

    @staticmethod
    def preprocess_channel(channel: np.ndarray,
                           settings: Dict[str, Any]) -> np.ndarray:
        """
        Method to prepare a channel for focus detection

        :param channel: The channel to pre-process
        :param settings: The settings to use
        :return: The pre-processed image
        """
        processed = channel
        # Check which pre-processing steps should be applied
        if settings["use_smoothing"]:
            processed = FocusMapper.perform_noise_reduction(processed,
                                                            settings["smoothing_method"],
                                                            settings["filter_radius"],
                                                            settings["gaussian_sigma"],
                                                            settings["denoising_weight"],
                                                            settings["sigma_color"],
                                                            settings["sigma_spatial"])
        if settings["use_background_reduction"]:
            processed= FocusMapper.perform_background_subtraction(processed,
                                                                  settings["background_reduction_method"],
                                                                  settings["bckg_subtr_diameter"],
                                                                  settings["bckg_subtr_feature_min"],
                                                                  settings["bckg_subtr_feature_max"],
                                                                  settings["bckg_subtr_order"],
                                                                  settings["dots_per_micron"])
        return processed

    @staticmethod
    def perform_noise_reduction(channel:np.ndarray,
                                method: str,
                                filter_size: int,
                                sigma: float,
                                weight: float,
                                sigma_color: float,
                                sigma_spatial: float,
                                ) -> np.ndarray:
        """
        Method to smooth the image to reduce noise

        :param channel: The channel to smooth
        :param method: The method to use
        :param filter_size: The filter size to use
        :param sigma: The sigma of the gaussian filter. Only relevant if gaussian filtering is applied.
        :param weight: The denoising weight used for Total Variation denoising
        :param sigma_color: The standard deviation for grayvalue/color distance. Used for bilateral denoising
        :param sigma_spatial: The standard deviation for range distance. USed for bilateral denoising
        :return: The smoothed image.
        """
        # Check the respective method
        match method:
            case "Gaussian":
                # Smooth the image
                smoothed = gaussian(channel, sigma=sigma)
            case "average":
                smoothed = mean(channel, footprint=np.ones(shape=(filter_size, filter_size)))
            case "median":
                smoothed = median(channel, footprint=np.ones(shape=(filter_size, filter_size)))
            case "Total Variation Denoising":
                smoothed = denoise_tv_chambolle(img_as_float(channel), weight=weight)
            case "Bilateral Denoising":
                smoothed = denoise_bilateral(img_as_float(channel),
                                             sigma_color=sigma_color,
                                             sigma_spatial=sigma_spatial)
            case "Wavelet Denoising":
                smoothed = denoise_wavelet(img_as_float(channel), rescale_sigma=True)
            case _:
                warnings.warn(f"Unknown noise reduction method '{method}', unprocessed channel returned!")
                smoothed = channel
        return rescale_intensity(smoothed, in_range="image", out_range=(0, 255)).astype(channel.dtype)

    @staticmethod
    def perform_background_subtraction(channel: np.ndarray,
                                       method: str,
                                       filter_size: int,
                                       min_size: float,
                                       max_size: float,
                                       order: int,
                                       micron_per_pixel: float) -> np.ndarray:
        """
        Method to perform background subtraction

        :param channel: The channel to perform background subtraction on
        :param method: The method to use
        :param filter_size: The filter size to use
        :param min_size: The min focus size to keep. Used for Butterworth filtering
        :param max_size: The max focus size to keep. Used for Butterworth filtering
        :param order: The order to use. Used for Butterworth filtering
        :param micron_per_pixel: Conversion rate between microns and pixels
        :return: The channel with subtracted background
        """
        # Check the respective method
        match method:
            case "White Top-Hat":
                processed = white_tophat(channel, footprint=np.ones(shape=(filter_size, filter_size)))
            case "Unsharp Masking":
                processed = unsharp_mask(channel, filter_size)
            case "Butterworth-Filtering":
                # Convert the size to pixels
                d_min_px = min_size / micron_per_pixel
                d_max_px = max_size / micron_per_pixel
                # Ratios to Nyquist (0…0.5)
                r_hp = np.clip(2.0 / d_max_px, 1e-4, 0.49)  # high-pass cutoff
                r_lp = np.clip(2.0 / d_min_px, r_hp + 1e-4, 0.5)  # low-pass cutoff
                # Load and filter
                butter = img_as_float(channel)
                lp = butterworth(butter, cutoff_frequency_ratio=r_lp, high_pass=False, order=order,
                                 squared_butterworth=False)
                hp = butterworth(butter, cutoff_frequency_ratio=r_hp, high_pass=True, order=order,
                                 squared_butterworth=False)
                processed = lp * hp
            case _:
                warnings.warn(f"Unknown background subtraction method '{method}', unprocessed channel returned!")
                processed = channel
        return rescale_intensity(processed, in_range='image', out_range=(0, 255)).astype(channel.dtype)


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
