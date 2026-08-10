"""
Created on 09.04.2019
@author: Romano Weiss
"""
from __future__ import annotations

from typing import Union, List, Tuple, Iterable

import numpy as np
from numba.typed import List as numList

from core.roi import AreaAnalysis
from core.roi.ROI import ROI


class ROIHandler:
    __slots__ = [
        "ident",
        "main",
        "rois",
        "idents",
    ]

    def __init__(self, ident: str = None):
        """
        :param ident: md5 hash of the image this handler is associated with
        """
        self.ident: str = ident
        self.rois: List[ROI] = []
        self.idents: List[str] = []
        self.main = ""

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
        if roi.main:
            self.main = roi.ident

    def add_rois(self, rois: List[ROI]) -> None:
        """
        Method to add new roi to this handler

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

    def create_hash_association_maps(self, shape: Tuple[int, int]) -> Iterable[np.ndarray]:
        """
        Method to create arrays with labelling hashes for each saved ROI

        :param shape: The shape of the original image
        :return: The created maps
        """
        maps = []
        # Create empty maps
        for _ in range(len(self.idents)):
            maps.append(np.zeros(shape, dtype="int64"))
        for roi in self:
            # Create numba list
            num_area = numList()
            for x in roi.area:
                num_area.append(x)
            # Create the channel maps using numba
            AreaAnalysis.imprint_area_into_array(num_area, maps[self.idents.index(roi.ident)], hash(roi))
        return maps

    def delete_rois(self, hashes: List[str]) -> None:
        """
        Method to delete roi from this handler based on their hashes

        :param hashes: The hashes of roi to delete
        :return: None
        """
        self.rois = [x for x in self if x.id not in hashes]

