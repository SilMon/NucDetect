"""
Created 09.04.2019
@author Romano Weiss
"""
from __future__ import annotations

import datetime
import os.path
import time
from copy import deepcopy
from typing import Any, Union, Dict, List, Tuple

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

# The detection methods analyse_image knows how to dispatch. The strings come from the analysis
# settings dialog, where they are the radio buttons' captions lowercased, so this set and those
# captions have to agree -- which is exactly why the value is validated rather than assumed
DETECTION_METHODS = frozenset({"image processing", "u-net", "combined"})

# Thresholds for the per-image plausibility report below. They decide when the line is escalated
# from a record to a warning; they filter NOTHING and change no result.
#
# BOTH ARE SET ABOVE WHAT THIS DETECTOR NORMALLY DOES, measured rather than chosen:
#
# * **border**: over the whole testing database -- 116 analysed images, 1714 stored nuclei -- 19.5 %
#   of nuclei are cut off by an edge, per image a median of 20 % and a maximum of 60 %. 0.4 fires
#   on that tail and not on the norm. (The 2026-09-11 investigation measured 25.3 % on its seven
#   reference images, which agrees.)
# * **size bounds**: the detector's own output runs 36 % to 52 % outside them, and RW ruled on
#   2026-09-15 that the bounds are correctly tuned as they stand. So half the output being
#   discarded is this pipeline working normally, and the threshold has to sit well above it --
#   0.5 would have warned on most images, which is a warning nobody reads.
IMPLAUSIBLE_DISCARD_SHARE = 0.75
IMPLAUSIBLE_BORDER_SHARE = 0.4


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
        # An analysis that raised would otherwise leave its channels and ROI behind until the next
        # one; clearing here as well as at the end keeps the invariant unconditional
        self.release_transient_state()
        analysis_settings = deepcopy(settings["analysis_settings"])
        analysis_settings["log"] = self.add_log_message
        # Checked here, before the image is read, because it is a settings defect and not an image
        # one -- there is no point loading a 3 GB stack to reject the configuration afterwards.
        #
        # The dialog now clears a channel's main-channel radio button when the channel is
        # deactivated, but the dialog is not the only source of a settings dictionary: the
        # verification harnesses build them directly and so could any future caller. Without this,
        # main_index further down is computed for a channel that is not in the filtered list, and
        # the analysis runs on a neighbouring channel with nothing in the log, the result table or
        # the exported data recording that the selection was reinterpreted
        if not 0 <= settings["main"] < len(settings["activated"]):
            raise ValueError(f"Main channel index {settings['main']} is outside the "
                             f"0-{len(settings['activated']) - 1} range of configured channels")
        if not settings["activated"][settings["main"]]:
            raise ValueError(f"Channel {settings['main']} is nominated as the main channel but is "
                             f"not active -- activate it or nominate an active channel")
        # Each stage reports 0..1 within its own slice of the bar and never learns its position in
        # the whole run. Weights are measured, per method -- see core/progress.py
        bounds = stage_bounds(analysis_settings["method"])
        prg = {stage: progress.sub(*bounds[stage]) for stage in bounds}
        start = time.time()
        prg[LOAD](0.0, "Reading image metadata")
        # Copied into a plain dict on purpose. get_image_data returns an ImageData (a TypedDict
        # describing image METADATA); the twelve keys added below turn it into the analysis RESULT,
        # which is a different thing with a different contract -- and ":206" even reuses "channels"
        # for the channel arrays rather than the channel count. Mutating the metadata type in place
        # would make ImageData claim to describe both. The copy is one shallow dict; the result
        # contract itself is still undescribed, which is cause 3 of the type-baseline backlog
        imgdat: Dict[str, Any] = dict(self.imageloader.get_image_data(path))
        self.analysis_log["Analysed Images"].append(os.path.basename(path))
        self.analysis_log["Messages"][self.analysis_log["Analysed Images"][-1]] = []
        prg[LOAD](0.3, "Hashing image")
        imgdat["id"] = self.imageloader.calculate_image_id(path)
        # A per-image conversion factor overrides the run-wide one, for THIS image only.
        #
        # Done here rather than at either dispatch site because this is where the image identity is
        # known: the single-image path and the batch path would otherwise each need their own copy
        # of the same override, and the batch one hands the settings to a worker process.
        #
        # The dict is COPIED before the override. analysis_settings is shared across the run --
        # batch analysis passes one dict to every worker -- so writing into it would give the next
        # image whatever the previous one was set to. That this method mutates its argument at all
        # is an open finding; this line does not add to it.
        per_image = analysis_settings.get("per_image_scale") or {}
        if imgdat["id"] in per_image:
            analysis_settings = dict(analysis_settings)
            analysis_settings["dots_per_micron"] = per_image[imgdat["id"]]
        # Check if only a grayscale image was provided
        if imgdat["channels"] == 1:
            self.add_log_message("Detector class can only analyse multichannel images, not grayscale!")
            raise ValueError("Detector class can only analyse multichannel images, not grayscale!")
        prg[LOAD](0.6, "Loading image")
        image = self.imageloader.load_image(path)
        names = settings["names"]
        main_channel: int = settings["main"]
        detection_method = analysis_settings["method"]
        # Validated here rather than trusted: the value is a lowercased Qt radio-button caption
        # (SettingsDialog.get_detection_method), so re-wording a button in Designer silently
        # produces a method no branch below expects -- and the else further down would then reach
        # `mlroi`, which is only bound for u-net/combined, as an UnboundLocalError mid-analysis.
        # stage_bounds() does not catch it either; it falls back silently on an unknown method
        if detection_method not in DETECTION_METHODS:
            raise ValueError(f"Unknown detection method '{detection_method}' -- expected one of "
                             f"{', '.join(sorted(DETECTION_METHODS))}")
        # Channel extraction
        prg[LOAD](0.9, "Splitting channels")
        channels = self.imageloader.get_channels(image)
        active = settings["activated"]
        # From here on there are TWO index spaces and mixing them is what both defects below were.
        # settings["main"] indexes the FULL channel list -- the main-channel radio buttons are fixed
        # positions -- so it stays valid for names/active, while anything indexing the filtered
        # lists needs the index shifted down by the inactive channels before it
        analysis_settings["names"] = [names[x] for x in range(len(names)) if active[x]]
        channels = [channels[x] for x in range(len(channels)) if active[x]]
        # Read from the RAW list with the RAW index. Reading the FILTERED list with the unadjusted
        # index went out of range as soon as enough channels before the main one were deactivated:
        # 32 of the 80 possible five-channel configurations raised IndexError here
        main_channel_name = names[main_channel]
        analysis_settings["main_channel_name"] = main_channel_name
        # One sum, not a loop that decrements the value its own guard is compared against. The old
        # form stopped counting after the first deduction, so with two or more inactive channels
        # before the main one it under-shifted and selected a neighbouring channel
        main_index = main_channel - sum(1 for x in range(main_channel) if not active[x])
        main = channels[main_index]
        foc_channels = [channels[i] for i in range(len(channels)) if i != main_index]
        # != rather than `is not`: identity holds only while both sides are the same interned
        # string object. A channel name read from the database or built at runtime compares
        # unequal under `is not` and the main channel would be kept as a foci channel
        analysis_settings["foci_channel_names"] = [x for x in analysis_settings["names"]
                                                   if x != analysis_settings["main_channel_name"]]
        # Detect roi via image processing and machine learning
        # main_channel_name, not names[main_index]: that mixed the filtered index into the raw list
        # and named the nucleus channel after a different channel whenever anything was deactivated
        main_map, main_roi = self.nucleus_extraction(main, main_channel_name, analysis_settings,
                                                     prg[NUCLEUS])
        # Reported HERE, on what the detector returned, not on what survives the quality check:
        # the point of the line is to describe the detection, and a result that is implausible
        # because the bounds removed most of it reads as a clean one once they have
        plausibility = self.report_nucleus_plausibility(main_roi, main.shape, analysis_settings)
        # Define a handler to take the ROI
        # The nomination comes from the analysis dialog, by way of settings["main"] -- it is
        # the channel the user pointed at, and it holds whether or not anything was found on it
        handler = ROIHandler(ident=imgdat["id"], main=main_channel_name)
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
                # Check the foci for co-localisation. Called ONCE, outside the loop: it takes the
                # whole per-channel list and was previously invoked once per channel with that same
                # list. Each call rebuilds its own match dictionary, so the repeats produced an
                # identical result rather than a wrong one -- the cost was len(foci) times the work,
                # which is why nothing looked broken. It reports through nucleus.match rather than a
                # return value, which is why discarding the result here is correct
                MapComparator.get_match_for_nuclei(main_roi, foci)
            elif detection_method == "image processing":
                rois.extend(iproi)
            else:
                rois.extend(mlroi)
            # Add the detected nuclei to the list
            rois.extend(main_roi)
            # Check for quality of roi. Gated on the setting: the "Analysis - Quality Check"
            # master switch was written by the settings dialog, stored, and consulted by nothing,
            # so a user who turned it off to save time still paid for it and a user who read its
            # description believed their data had not been filtered when it had.
            # .get with a default, not [..]: analysis_settings comes from a user-editable JSON file
            # and from callers that build it by hand (the harnesses), so an absent key must mean
            # "run the check" rather than KeyError out of the middle of an analysis
            if rois and analysis_settings.get("quality_check", True):
                prg[QUALITY](0.0, "Checking ROI quality")
                # The FILTERED names, to match the filtered channels. Passing the raw list paired
                # each name with the wrong channel array, and since the ROI idents come from the
                # filtered list the lookup missed outright -- a KeyError out of the quality check
                # -- as soon as a deactivated channel was not the trailing one
                qroi = self.perform_quality_check(channels, analysis_settings["names"],
                                                  analysis_settings, rois)
                # RW ruled on 2026-08-24 that this line must report more than it did:
                # *"Both the count of deleted nuclei as well as deleted foci should be provided.
                # For foci, the number per channel should be stated."*
                #
                # It read `len(rois) - len(qroi)` under the label "Removed foci". Both lists hold
                # nuclei AND foci, so the difference is the number of ROI removed, always too high
                # by the nucleus count: on a run RW pasted it said 5211 foci where 5179 foci and 32
                # nuclei had gone. That mattered more than a wording slip, because the number then
                # failed to reconcile with the lines above it -- and reconciling the gap by hand is
                # what exposed the size-bound defect.
                #
                # The counts are taken HERE, from the two lists, because this is the last place the
                # channel of each removed ROI is still known. Identity, not equality: two ROI with
                # the same area on the same channel are equal and hash alike, so a set of hashes
                # would under-count duplicates.
                kept = {id(x) for x in qroi}
                removed = [x for x in rois if id(x) not in kept]
                removed_nuclei = sum(1 for x in removed if x.main)
                per_channel = {}
                for roi in removed:
                    if not roi.main:
                        per_channel[roi.ident] = per_channel.get(roi.ident, 0) + 1
                breakdown = ", ".join(f"{name}: {n}" for name, n in sorted(per_channel.items()))
                self.add_log_message(
                    f"QR: Removed {removed_nuclei} nuclei and {sum(per_channel.values())} foci"
                    + (f" ({breakdown})" if breakdown else ""))
            else:
                # `rois`, NOT an empty list. This read `qroi = []`, so **switching the quality check
                # off discarded every detected ROI** and the analysis produced nothing at all --
                # silently, with a full progress bar and no error. The setting is a user-facing
                # checkbox ("Analysis - Quality Check"), so anyone who turned it off to save time
                # got an empty result and no indication why.
                #
                # Found 2026-08-22 while investigating the empty-nuclei report. The branch also
                # covers `not rois`, where rois is already empty and this is a no-op.
                qroi = rois
            # THE ASSOCIATION RULE IS NOT PART OF THE QUALITY CHECK AND MUST NOT BE OPTIONAL.
            # A focus that lies inside no nucleus is a background artefact, and RW's rule is that
            # one must never reach the database: it is stored with `associated = NULL`, which is
            # also how a NUCLEUS is stored, so it is read back as a nucleus and takes the result
            # table down on its missing ellipse statistics.
            #
            # delete_unassociated_foci already did this, but only from inside check_quality -- so
            # turning off the "Analysis - Quality Check" box, which is a user-facing switch about
            # SIZE and INTENSITY filtering, silently also turned off a data-model invariant. Run
            # here it is unconditional, and it is idempotent when the quality check already ran.
            #
            # The ORDER of qroi is preserved rather than rebuilt as nuclei + foci. `ROIHandler`
            # registers channels in the order roi arrive, and `create_hash_association_maps`
            # indexes its maps by `idents.index` -- so reordering here would renumber the channels
            # under everything that reads a channel index, including the editor.
            nuclei = [x for x in qroi if x.main]
            keep = {id(x) for x in QualityTester.delete_unassociated_foci(
                nuclei, [x for x in qroi if not x.main])}
            checked = [x for x in qroi if x.main or id(x) in keep]
            dropped = len(qroi) - len(checked)
            if dropped:
                self.add_log_message(f"QR: Foci outside every nucleus, deleted: {dropped}")
            handler.add_rois(checked)
        imgdat["x_scale"] = analysis_settings["dots_per_micron"]
        imgdat["y_scale"] = analysis_settings["dots_per_micron"]
        imgdat["scale_unit"] = "µm"
        imgdat["handler"] = handler
        # Travels back with the result so the PARENT can escalate it. A batch worker's own logger
        # is a NullHandler, so the warning has to be raised where the results are collected
        imgdat["plausibility"] = plausibility
        imgdat["names"] = analysis_settings["names"]
        imgdat["channels"] = channels
        imgdat["active channels"] = active
        imgdat["main channel"] = main_channel
        imgdat["add to experiment"] = settings["add_to_experiment"]
        imgdat["experiment details"] = settings["experiment_details"]
        # Remove logging function from settings
        del analysis_settings["log"]
        imgdat["used_settings"] = analysis_settings
        # Returned as a FIELD, not only as log text. The parent's batch loop needs the time this
        # image cost in order to estimate a remaining time that accounts for parallelism, and
        # scraping it back out of the replayed log would be parsing our own prose. Computed once and
        # used for both, so the number in the log and the number the estimate uses cannot diverge
        imgdat["duration"] = time.time() - start
        self.add_log_message(f"Total analysis time: {imgdat['duration']: .4f}")
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
        self.release_transient_state()
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
            # get_nucleus_maps, NOT map_nuclei. It is a thin wrapper that validates before it
            # delegates -- exactly one channel must be set, and settings must be present -- and
            # calling the inner method directly bypassed those checks, which is what made the
            # wrapper look like dead code. The checks are worth having here: clear_state() sets
            # the channel tuple to () between analyses, so an analysis that reached this line
            # without set_channels would otherwise map whatever was left over
            nucmap = self.nucleusmapper.get_nucleus_maps()
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

    def report_nucleus_plausibility(self, nuclei: List[ROI], shape: Tuple[int, int],
                                    analysis_settings: Dict) -> Dict[str, Union[int, float, bool]]:
        """
        Method to describe what the nucleus detection returned, before anything filters it

        **Nothing here changes a result.** Until 2026-09-15 the pipeline reported the two counts
        the mapper logs -- seeds found, nuclei segmented -- and nothing else, so the failures a
        user actually meets were invisible: the segmentation returns 1 px regions (4 of 166 on the
        reference images, against a median of 6188 px), a third to a half of its output is
        discarded downstream by the size bounds, and a quarter of what survives is cut off by the
        image border. Each of those produces a plausible-looking result with a full progress bar.

        The figures were all already computed somewhere; the point of this method is that they are
        reported together, per image, **without the user having to know what the image should
        contain**. The bounds are the same ones QualityTester filters on, converted the same way,
        so the "discarded" figure here and what the quality check actually removes cannot drift
        apart -- and it is reported whether or not the quality check is switched on, because the
        question "is this a sensible detection?" does not depend on that setting.

        Reported through add_log_message rather than LOGGER: in a batch run this executes inside a
        ProcessPoolExecutor worker, whose logger is a NullHandler by design, so a LOGGER call here
        would vanish for exactly the runs that need it most. The parent replays the buffered lines
        and escalates the warning -- see the returned dict, which travels back in the result.

        :param nuclei: The nuclei as the detector produced them, before the quality check
        :param shape: The (height, width) of the main channel
        :param analysis_settings: The settings this analysis runs with
        :return: The figures, for the caller to surface
        """
        # px per um, squared for an area, exactly as QualityTester.check_size_boundaries does it.
        # .get with a default where the quality check uses []: a missing bound must not take a
        # REPORT down, and the three keys are guaranteed present in the dict a real analysis runs
        # with -- it is harnesses and hand-built dicts that omit them
        px_per_um2 = analysis_settings.get("dots_per_micron", 1.0) ** 2
        lower = analysis_settings.get("min_main_area", 0) * px_per_um2
        upper = analysis_settings.get("max_main_area", float("inf")) * px_per_um2
        below = above = border = 0
        for nucleus in nuclei:
            area = nucleus.calculate_dimensions()["area"]
            if area < lower:
                below += 1
            elif area > upper:
                above += 1
            if nucleus.touches_border(shape):
                border += 1
        total = len(nuclei)
        # max(1, total) rather than a guard: total == 0 is itself implausible and is reported as
        # such below, so the shares must stay computable rather than take the report down
        divisor = max(1, total)
        report = {
            "nuclei": total,
            "below_min_area": below,
            "above_max_area": above,
            "border": border,
            "discarded_share": (below + above) / divisor,
            "border_share": border / divisor,
        }
        report["implausible"] = bool(
            total == 0
            or report["discarded_share"] > IMPLAUSIBLE_DISCARD_SHARE
            or report["border_share"] > IMPLAUSIBLE_BORDER_SHARE
        )
        self.add_log_message(
            f"Plausibility: {total} nuclei, {below} below and {above} above the size bounds "
            f"({report['discarded_share']:.1%} discarded), {border} touching the image border "
            f"({report['border_share']:.1%})"
        )
        if report["implausible"]:
            # The reason is spelled out rather than left to be re-derived from the line above: the
            # three conditions look alike in the numbers and mean different things -- an empty
            # result, a channel whose objects are the wrong size, and a field of view that is
            # mostly edge
            if total == 0:
                reason = "no nuclei were detected at all"
            elif report["discarded_share"] > IMPLAUSIBLE_DISCARD_SHARE:
                reason = "most of the detected nuclei lie outside the size bounds"
            else:
                reason = "most of the detected nuclei are cut off by the image border"
            report["reason"] = reason
            self.add_log_message(f"Plausibility: RESULT LOOKS IMPLAUSIBLE -- {reason}")
        return report

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

    def release_transient_state(self) -> None:
        """
        Method to drop the per-image data the mappers hold on to between analyses

        The mappers keep whatever ``set_channels``/``set_roi`` were last given, purely as a side
        effect of their setter-based API -- nothing reads it once ``analyse_image`` has returned.
        Holding it has two costs:

        * **memory** -- a Detector that has run one analysis retains the image channels and every
          detected ROI for as long as it lives, which for the main window means the whole session;
        * **serialisation** -- batch analysis passes the bound ``analyse_image`` to a
          ``ProcessPoolExecutor``, which pickles this object **once per image**. Measured on a
          1024x1024 image: 0.9 KiB when clean, **3 431 KiB** after one image-processing analysis and
          **11 211 KiB** after one u-net analysis, the latter because ``fcnmapper`` holds a loaded
          Keras model. Over a 100-image batch that is the difference between a rounding error and
          roughly a gigabyte of pickling.

        Invariant: outside of a running ``analyse_image``, this object holds no image data. It is
        therefore called at both ends of an analysis -- the leading call covers an analysis that
        raised partway through.

        :return: None
        """
        self.nucleusmapper.set_channels(())
        self.focusmapper.set_channels(())
        self.qualitytester.set_channels(())
        self.qualitytester.set_channel_names(())
        self.qualitytester.set_roi([])
        # Holds a reference to the Keras model, which is the bulk of the u-net figure above.
        #
        # The reason this is free CHANGED on 2026-08-21 and the old one no longer holds. It used to
        # be "it is rebuilt on every ml_roi_extraction call regardless" -- true then, and the reason
        # rebuilding was so expensive. FCNMapper now caches the model at module level, so dropping
        # the mapper no longer forces a reload: the next ml_roi_extraction builds a mapper that
        # picks the cached model straight back up.
        #
        # Dropping it here is therefore still free AND still necessary. Necessary because the model
        # must not be reachable from this object when a ProcessPoolExecutor pickles it once per
        # image -- a module-level cache is not pickled, but `self.fcnmapper.model` would be.
        self.fcnmapper = None

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
