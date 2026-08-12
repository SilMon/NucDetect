import time
import warnings
from typing import Dict, Union, List, Iterable, Tuple, Any, Callable

import numpy as np
from numpy import ndarray

from core.logging_config import get_logger
from core.roi.ROI import ROI

LOGGER = get_logger(__name__)


# Both reporting callables below are module level on purpose, and must stay that way. `default_log`
# used to be a lambda in the class body of QualityTester, which made **every Detector instance
# unpicklable** -- class-body lambdas cannot be pickled -- and therefore made batch analysis fail
# before a single image was read: `_analyze_all` hands `self.detector.analyse_image` to a
# ProcessPoolExecutor, which pickles the bound method and with it the whole Detector, including the
# QualityTester built in its constructor. A module-level function pickles by reference and does not.
# Anything bound to `self.log` is subject to the same constraint.
def default_log(message: str) -> None:
    """
    Fallback used when logging is enabled but no reporting callable was injected

    The real flow always injects one -- Detector.analyse_image puts `add_log_message` into
    `analysis_settings["log"]` -- so this is reached only by a caller that builds a QualityTester
    itself. It writes through the shared logger rather than `print`, so the message reaches the log
    file; note that in a worker process the configured NullHandler makes it a no-op by design, which
    is why the injected buffer-and-replay callable is what the real flow uses.

    :param message: The message to report
    :return: None
    """
    LOGGER.info(message)


def no_log(message: str) -> None:
    """
    Bound to `self.log` when the `logging` setting is off, so the seven quality-check messages cost
    a call and nothing else

    A no-op function rather than a falsy attribute checked at each call site: it keeps the guard in
    one place instead of seven, and keeps the call sites reading as plain reporting.

    :param message: Ignored
    :return: None
    """


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
        "log": default_log
    }

    def __init__(self, channels: List[np.ndarray] = None, channel_names: List[str] = None,
                 roi: Iterable[ROI] = None, settings: Dict[str, Union[str, int, float, Callable]] = None):
        self.channels = channels
        self.channel_names = channel_names
        self.roi = roi
        self.log: Callable = no_log
        if settings:
            self.set_settings(settings)
        else:
            warnings.warn("No settings found, standard settings used for focus mapping")
            self.set_settings(self.STANDARD_SETTINGS)

    def set_channels(self, channels: List[np.ndarray]) -> None:
        self.channels = channels

    def set_channel_names(self, channel_names: Iterable[str]) -> None:
        self.channel_names = channel_names

    def set_roi(self, roi: List[ROI]) -> None:
        self.roi = roi

    def set_settings(self, settings: Dict) -> None:
        """
        Method to set the settings, and to rebind the reporting callable that goes with them

        **Rebinding `self.log` here is the point of this method, not a detail.** It used to assign
        only `self.settings`, while `self.log` was bound once in the constructor. Detector builds a
        QualityTester with no settings and calls this later with the real ones, so `self.log` stayed
        on the fallback for the lifetime of that Detector and the injected reporter -- the one that
        buffers per image for the parent to replay -- was never used. The seven quality-check
        messages went to stdout and never reached the log file, in single-image and batch analysis
        alike.

        The `logging` setting is honoured here too, which is what `_analyze_all` has always assumed:
        it forces `logging` off for the duration of a batch and restores it afterwards. Until now no
        module read the flag, so that suppression -- and the user-facing *Analysis - General ->
        Logging* checkbox behind it -- did nothing at all.

        :param settings: The settings to use
        :return: None
        """
        self.settings = settings
        # .get, not [], so a caller passing a partial dict falls back rather than raising; the two
        # keys are guaranteed only in STANDARD_SETTINGS and in Detector's analysis_settings
        self.log = settings.get("log", default_log) if settings.get("logging", False) else no_log

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
        return self.check_quality()

    def check_quality(self) -> Tuple[List[ROI], List[ROI]]:
        """
        Method to check the quality of given nuclei/foci

        :return: The checked roi
        """
        # TODO überprüfen ob die Einstellungen so stimmen
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
        # Pair by position and refuse to guess when the two lists disagree. Truncating the names
        # to the number of channels absorbed the mismatch instead, pairing every name with the
        # wrong channel whenever the missing one was not the trailing entry -- which surfaced as a
        # KeyError on roi.ident two call levels away rather than here, where the cause is. Position
        # also replaces names.index(), which returns the first match and so cross-wires two
        # channels that happen to carry the same name
        if len(self.channel_names) != len(self.channels):
            raise ValueError(f"Got {len(self.channel_names)} channel names for "
                             f"{len(self.channels)} channels: {list(self.channel_names)}")
        return {name: {"Channel": channel,
                       "Lower": np.amin(channel),
                       "Upper": np.amax(channel),
                       "Max. Val": np.iinfo(channel.dtype).max}
                for name, channel in zip(self.channel_names, self.channels)}

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
            # Half-extent of the focus, measured FROM THE CENTRE THE WINDOW IS PLACED ON. It sizes
            # both the sampled window below and the mask that blanks the focus out of it, so it has
            # to be a radius, and it has to be a radius about fcy/fcx specifically.
            #
            # Two things were wrong here. The old max((maxX - minX) // 2, maxY - minY) halved only
            # the X term, so the Y term won for anything not more than twice as wide as tall (i.e.
            # every roughly circular focus) and yielded a diameter -- putting the background ring a
            # full focus diameter from the centre, frequently outside the nucleus altogether. And
            # deriving it from the bounding box at all mixes two centres: fcy/fcx come from
            # get_center, the run-length-weighted centroid, while the box is centred on its own
            # midpoint. Those coincide for a symmetric focus and diverge otherwise -- measured 2.5 px
            # for a two-lobed blob -- and the difference is focus pixels sitting outside the mask,
            # in a ring only arr pixels thick.
            #
            # So take the largest distance from the centroid to the area's extremes. maxX/maxY are
            # one past the last pixel, hence the -1; the +1 is because the mask slice below is
            # half-open, so covering a pixel at distance d needs fr > d. Identical to the bounding
            # box for a symmetric focus, larger only where the two centres actually disagree.
            fr = max(fcy - dims["minY"], dims["maxY"] - 1 - fcy,
                     fcx - dims["minX"], dims["maxX"] - 1 - fcx) + 1
            arr = 3
            if fcy < fr + arr or fcx < fr + arr:
                continue
            # Get area around center
            area = channel[fcy - fr - arr: fcy + fr + arr,
                   fcx - fr - arr: fcx + fr + arr]
            # Get mask
            mask = np.ones(shape=area.shape)
            # Focus centre in window coordinates. The slice above starts at fcy - fr - arr, which
            # the guard guarantees is >= 0, so the focus sits at exactly fr + arr whether or not the
            # far edge was clipped. Taking area.shape // 2 instead was wrong at the bottom and right
            # edges of the image: numpy clips a slice that runs past the end, and the midpoint of
            # the clipped window is no longer the focus, so the hole slid toward that edge and let
            # focus pixels -- the brightest in the window -- into the ring that must sample
            # background only. Harmless while fr was a diameter, because the oversized hole covered
            # the focus anyway; a correct fr exposes it.
            acy = acx = fr + arr
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

