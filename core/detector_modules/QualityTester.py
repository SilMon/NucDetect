import time
import warnings
from typing import Dict, Union, List, Iterable, Tuple, Any, Callable

import numpy as np
from numpy import ndarray

from core.logging_config import get_logger
from core.detector_modules.ImageLoader import dtype_max
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
        # A "max_channel_intensity": 255 entry stood here until 2026-08-15. Nothing in core/ or
        # gui/ ever read it, and its name promised exactly the per-dtype ceiling that
        # _get_values_dict already computes correctly as np.iinfo(channel.dtype).max -- so wiring
        # it up as written would have reintroduced the 8-bit cap that the "16-bit images are
        # silently reduced to 8-bit precision" fix removed. Deleted rather than connected.
        # A "max_focus_overlap": .75 entry stood here until 2026-09-13, alongside a
        # "max_foc_overlap" seeded into the settings table -- two spellings of one parameter, and
        # no reader for either: check_focus_overlap does not exist. Removed by RW's decision
        # rather than wired up. "max_foc_area" below is a DIFFERENT key, it IS read by
        # check_size_boundaries, and it stays.
        # dots_per_micron is REQUIRED by check_size_boundaries and is present in the real
        # settings dict (gui/dialogs/settings.py supplies it from the analysis dialog). It is
        # listed here too because STANDARD_SETTINGS is what harnesses and any caller that does not
        # build a full dict fall back to -- and a key that exists in only one of the two is exactly
        # how "use_signal_improvement" passed a harness and then raised KeyError on a real run.
        "dots_per_micron": 6.412,
        # The four size bounds are in SQUARE MICROMETRES, as the settings dialog has always said
        # they were. Until 2026-09-14 they were compared directly against an area in PIXELS, so the
        # unit in the dialog was decorative. These defaults were pixel counts and are converted:
        # 1000 -> 24.3, 30000 -> 729.8, 5 -> 0.12, 270 -> 6.57 at 6.412 px/um.
        "min_main_area": 24.3,
        "max_main_area": 729.8,
        "min_nucleus_int_perc": .8,
        "min_foc_area": 0.12,
        "max_foc_area": 6.57,
        "min_foc_int": .055,
        "min_foc_cont": .005,
        "cutoff": .03,
        # "size_factor": 1.0 stood here until 2026-09-14. QualityTester no longer reads it:
        # check_size_boundaries used to divide by it as though it were a scale, which it is
        # not -- it is the manual editor's spin box, and NucleusMapper's mask multiplier.
        # Both of those uses are untouched; this declaration was dead once the real
        # conversion went in, and a dead declaration in this dict is what the
        # "use_signal_improvement" finding is about.
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
            warnings.warn("No settings found, standard settings used for quality testing")
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
        self.log("Quality Check:")

        # Every line reports PASSED and DISCARDED against the count that went in, rather than the
        # survivors alone. "Nuclei Size Check: 19" read as "19 nuclei were checked" when it meant
        # "19 of 51 survived" -- and that is what hid a finding for weeks: 63 % of the nuclei were
        # being discarded and the line meant to report it looked like a tally of work done. The
        # input count was only recoverable from "Nuclei segmented:" seven lines earlier, in a
        # different block.
        def _report(name: str, before: int, after: int) -> None:
            self.log(f"{name}: {after} of {before} passed, {before - after} discarded")

        # Check size of nuclei
        before = len(main)
        lower_bound, upper_bound = self.settings["min_main_area"], self.settings["max_main_area"]
        main = self.check_size_boundaries(main, lower_bound, upper_bound)
        _report("Nuclei Size Check", before, len(main))
        # Delete foci whose nucleus was deleted or which are unassociated to a nucleus
        before = len(foci)
        foci = self.delete_unassociated_foci(main, foci)
        _report("Focus Association Check", before, len(foci))
        # Check size of foci
        before = len(foci)
        foci = self.check_size_boundaries(foci, self.settings["min_foc_area"], self.settings["max_foc_area"])
        _report("Focus Size Check", before, len(foci))
        # Check foci for intensity
        before = len(foci)
        foci = self.check_intensity_boundaries(foci, self.settings["min_foc_int"], 1)
        _report("Focus Intensity Check", before, len(foci))
        before = len(foci)
        foci = self.check_focus_contrast(foci, self.settings["min_foc_cont"])
        _report("Focus Contrast Check", before, len(foci))
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

    def check_size_boundaries(self, roi: List[ROI], lower_bound: float,
                              upper_bound: float) -> List[ROI]:
        """
        Method to check if the area of a roi lies inside the specified boundaries

        **The bounds are in square micrometres and the areas are in pixels.** The BOUNDS are
        converted, not the areas: a stored area stays a pixel count -- RW's rule, 2026-09-14 -- and
        converting two numbers per call rather than one per roi keeps the comparison on integers.

        Until 2026-09-14 this divided the area by `size_factor` and compared the result against the
        bound. Three things were wrong with that:

        * **`size_factor` is not a scale.** It is the manual editor's spin box -- settings.json
          files it under "Modification", titled *"Size factor for modification window"* -- and it
          also serves as a mask multiplier in NucleusMapper. It has been removed from this
          expression rather than kept alongside the real conversion.
        * **its default is 1.0**, so the division did nothing and a PIXEL area was compared against
          a bound the dialog declares in um^2;
        * the real scale, `dots_per_micron`, was never read here at all.

        Measured consequence, on the live database's tuned bounds: **roughly half of everything the
        detector returned was discarded**, and the bounds had been hand-tuned until that looked
        right -- which made them pixel counts wearing a um^2 label.

        :param roi: List of roi to check
        :param lower_bound: Lower threshold, in um^2
        :param upper_bound: Upper threshold, in um^2
        :return: List of ROI that are larger than lower_bound and smaller than upper_bound
        """
        # px per um, so px^2 per um^2 is its square
        px_per_um2 = self.settings["dots_per_micron"] ** 2
        lower_px, upper_px = lower_bound * px_per_um2, upper_bound * px_per_um2
        return [x for x in roi
                if lower_px <= x.calculate_dimensions()["area"] <= upper_px]

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
                       "Max. Val": dtype_max(channel.dtype)}
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
            # Calculate the average of the surrounding area. Boolean indexing, not a per-pixel
            # Python loop: the ring is the same set of pixels either way, and the loop cost 13x
            # more -- measured at 0.246 s against 0.019 s over 5000 foci on a 12x12 window, which
            # is per focus per image. The sum is taken in Python ints via .item() so a uint16
            # channel cannot overflow the accumulator, which is what int(area[y][x]) was for
            surrounding = area[mask.astype(bool)]
            num = int(surrounding.size)
            avg = int(surrounding.sum().item()) if num else 0
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

