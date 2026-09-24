"""
Created on 09.04.2019
@author: Romano Weiss
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, Union, List, Tuple

import numpy as np
from numba.typed import List as numList

from core.logging_config import get_logger
from core.roi import AreaAnalysis
from core.roi.ROI import ROI

LOGGER = get_logger(__name__)


class ROIHandler(Sequence):
    """Container for the ROI of one image.

    Declares Sequence rather than merely behaving like one. It already implements __len__,
    __getitem__ (slices included, since rois is a list) and __iter__, but Sequence is an ABC and is
    matched nominally, so isinstance(handler, Sequence) was False and no type checker would accept
    a ROIHandler where a Sequence is required -- which gui.loader.Loader requires, and is handed one
    by ROIDrawerTimer. The base class is a declaration, not an implementation: it adds only the
    mixin methods index, count, __contains__ and __reversed__.

    collections.abc ABCs declare __slots__ = (), so this does not give instances a __dict__ and the
    slots below stay effective. That matters because ROIHandler crosses the process boundary --
    workers build one and hand it back through the ProcessPoolExecutor.
    """
    __slots__ = [
        "ident",
        "main",
        "rois",
        "idents",
    ]

    def __init__(self, ident: str = None, main: str = ""):
        """
        :param ident: md5 hash of the image this handler is associated with
        :param main: name of the channel the nuclei are detected on
        """
        self.ident: str = ident
        self.rois: List[ROI] = []
        self.idents: List[str] = []
        # NOMINATED, not derived -- RW, 2026-09-14: *"The user assigns on which channel the program
        # will look for nuclei, so which channel is 'main'."*
        #
        # It used to be set by `add_roi` from each ROI's own `main` flag, which made it a property
        # of what happened to be DETECTED rather than of what the user asked for. Two consequences:
        # an image whose nucleus channel found nothing had `main == ""`, and `idents.index("")`
        # raised on save (UI row 72, twice); and removing the last nucleus would silently have
        # un-nominated the channel, which is why the removal paths could not simply recompute it.
        #
        # `""` still means "nobody has said", and readers still have to handle it -- an in-band
        # sentinel is its own small finding -- but it is now only reachable when the caller supplies
        # nothing, not as a consequence of detection.
        self.main = main

    def __len__(self):
        return len(self.rois)

    def __getitem__(self, item):
        return self.rois[item]

    def __iter__(self):
        return iter(self.rois)

    def sort_roi_list(self):
        """
        Method to sort the internal ROI list according to channel

        :return: None
        """
        self.rois = sorted(self.rois, key=lambda x: x.ident)

    def add_roi(self, roi: ROI) -> None:
        """
        Method to add a ROI to this handler

        :param roi: The ROI to add
        :return: None
        """
        self.rois.append(roi)
        if roi.ident not in self.idents:
            self.idents.append(roi.ident)
        # `if roi.main: self.main = roi.ident` stood here until 2026-09-14. The main channel is a
        # nomination made by the user before the analysis runs, so it is supplied at construction
        # and a detected ROI no longer votes on it. See __init__.
        #
        # The fallback covers a handler built without one: the first main ROI still names the
        # channel, so nothing that used to work stops working, but a supplied nomination is never
        # overwritten by what was detected.
        if roi.main and not self.main:
            self.main = roi.ident

    def add_rois(self, rois: List[ROI]) -> None:
        """
        Method to add new roi to this handler

        Use this rather than `handler.rois.extend(...)`. `rois` is a plain public list, so appending
        to it directly skips `add_roi` -- the only place `idents` and `main` are maintained -- and
        leaves the handler describing itself incorrectly. One caller did exactly that until
        2026-09-13 and it is why a hand-drawn nucleus could not be saved.

        :param rois: List of ROI to add
        :return: None
        """
        for roi in rois:
            self.add_roi(roi)

    def get_roi_by_hash(self, hash_: int) -> Union[int, None]:
        """
        Method to get a ROI by its hash

        :param hash_: The md5 hash of the ROI
        :return: The found ROI if it is contained in this handler else None
        """
        for roi in self:
            if roi == hash_:
                return roi

    def remove_roi(self, roi: ROI, cascade: bool = False) -> None:
        """
        Method to remove a ROI from this handler

        :param roi: The ROI to remove
        :param cascade: If the roi is main, cascade can be used to delete all associated ROI
        :return: None
        """
        self.rois.remove(roi)
        if roi.main and cascade:
            # If cascadian deletion is activated, delete all associated roi.
            # ROI.associated holds hash() of the nucleus, not the nucleus object, so the comparison
            # has to be against the hash. The previous `x.associated is not roi` compared an int
            # against a ROI and was therefore always true, which kept every focus and left them
            # orphaned -- a state the domain forbids, since a focus is always inside a nucleus
            nucleus_hash = hash(roi)
            self.rois = [x for x in self.rois if x.associated != nucleus_hash]

    def remove_roi_by_hash(self, hash_: int, cascade: bool = False) -> None:
        """
        Method to remove the ROI with the given hash

        :param hash_: The md5 hash of the roi
        :param cascade: If the roi is main, cascade can be used to delete all associated ROI
        :return: None
        """
        roi = self.get_roi_by_hash(hash_)
        if roi:
            self.remove_roi(roi, cascade)

    def remove_rois(self, rois: List[ROI]) -> None:
        """
        Method to remove ROI from this handler

        :param rois: List of ROI to remove
        :return: None
        """
        for roi in rois:
            self.remove_roi(roi)

    def remove_rois_by_hash(self, hashes: List[int], cascade: bool = False) -> None:
        """
        Method to remove rois by their hashes

        :param hashes: List of ROI md5 hashes
        :param cascade: If the roi is main, cascade can be used to delete all associated ROI
        :return: None
        """
        for hash_ in hashes:
            self.remove_roi_by_hash(hash_, cascade)

    def create_hash_association_maps(self, shape: Tuple[int, int],
                                     channels: Dict[str, int]) -> List[np.ndarray]:
        """
        Method to create arrays with labelling hashes for each saved ROI

        **One map per IMAGE CHANNEL, indexed by the channel's database index** -- not one per
        entry in `idents`. Until 2026-09-21 it was the latter, and that is what let
        `ROIItem.channel_index` mean two different things: the editor sets it from the database
        index when an item is drawn by hand and read it back as a position in `idents`, which agree
        only when every channel carries a detection AND the orders match. With a channel
        deactivated for the analysis they do not: `idents` is `analysis_settings["names"]`, the
        ACTIVE channels, so a focus stored on one channel was offered under another and could be
        saved there. Measured 2026-09-15: 1260 items on a channel they were not in.

        The caller supplies the mapping because this class does not know it -- `idents` is a list
        of names in arrival order, and only the editor holds name -> database index.

        :param shape: The shape of the original image
        :param channels: Channel name -> its database/image channel index
        :return: One map per channel index, positionally indexed by that index
        """
        # Sized from the mapping, not from len(idents): a channel with no detections still needs
        # its slot, or every index above it shifts down -- which is the defect this signature
        # exists to make impossible
        maps = [np.zeros(shape, dtype="int64") for _ in range(max(channels.values(), default=-1) + 1)]
        for roi in self:
            index = channels.get(roi.ident)
            if index is None:
                # A roi in a channel the editor was not given. Skipped loudly rather than imprinted
                # into an arbitrary map -- the old code would have raised ValueError here, which at
                # least failed; silently choosing a map would corrupt the geometry it writes
                LOGGER.warning("No channel index for %r -- its roi are left out of the association "
                               "maps", roi.ident)
                continue
            # Create numba list
            num_area = numList()
            for x in roi.area:
                num_area.append(x)
            # Create the channel maps using numba
            AreaAnalysis.imprint_area_into_array(num_area, maps[index], hash(roi))
        return maps

    def delete_rois(self, hashes: List[str]) -> None:
        """
        Method to delete roi from this handler based on their hashes

        :param hashes: The hashes of roi to delete
        :return: None
        """
        self.rois = [x for x in self if x.id not in hashes]

