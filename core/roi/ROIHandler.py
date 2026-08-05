"""
Created on 09.04.2019
@author: Romano Weiss
"""
from __future__ import annotations

from typing import Union, Dict, List, Tuple, Iterable

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
        "stats",
    ]

    def __init__(self, ident: str = None):
        """
        :param ident: md5 hash of the image this handler is associated with
        """
        self.ident: str = ident
        self.rois: List[ROI] = []
        self.idents: List[str] = []
        self.stats: Dict[str, Union[int, float]] = {}
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
        self.stats.clear()

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
            # If cascadian deletion is activated, delete all associated roi
            self.rois = [x for x in self.rois if x.associated is not roi]
        self.stats.clear()

    def remove_roi_by_hash(self, hash_: int, cascade: bool = False) -> None:
        """
        Method to remove the ROI with the given hash

        :param hash_: The md5 hash of the roi
        :param cascade: If the roi is main, cascade can be used to delete all associated ROI
        :return: None
        """
        roi = self.get_roi_by_hash(hash_)
        if roi:
            self.remove_roi(roi)

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

    def calculate_statistics(self, img: np.ndarray) -> Dict[str, Union[int, float]]:
        """
        Method to calculate statistics about the saved ROIs

        :param img: The image this handler is associated to
        :return: dict -- A dictonary containing the calculated statistics
        """
        if not self.stats:
            main = {
                "num": 0,
                "num empty": 0,
                "area": [],
                "intensity": [],
                "method": [],
                "match": []
            }
            sec = {}
            channels = [img[..., x] for x in range(img.shape[2])]
            for roi in self.rois:
                temp_stat = roi.calculate_statistics(channels[self.idents.index(roi.ident)])
                if roi.main:
                    main["num"] += 1
                    main["area"].append(temp_stat["area"])
                    main["intensity"].append(temp_stat["intensity average"])
                    main["method"].append(temp_stat["method"])
                    main["match"].append(temp_stat["match"])
                else:
                    if roi.ident not in sec:
                        sec[roi.ident] = {
                            "num": 1,
                            "area": [temp_stat["area"]],
                            "intensity": [temp_stat["intensity average"]]
                        }
                        main["num empty"] -= 1
                    else:
                        sec[roi.ident]["num"] += 1
                        sec[roi.ident]["area"].append(temp_stat["area"])
                        sec[roi.ident]["intensity"].append(temp_stat["intensity average"])

            sec_stat = self._calculate_secondary_statistics(sec)
            area = main["area"]
            match = main["match"]
            method = main["method"]
            inten = main["intensity"]
            self.stats = {
                "number": main["num"],
                "match": np.average(match),
                "method": np.unique(method, return_counts=True),
                "number stats": sec_stat,
                "area list": area,
                "area average": np.average(area),
                "area median": np.median(area),
                "area std": np.std(area),
                "area minimum": min(area),
                "area maximum": max(area),
                "intensity list": inten,
                "sec idents": sec_stat.keys(),
                "sec stats": sec_stat
            }
        return self.stats

    @staticmethod
    def _calculate_secondary_statistics(sec: Dict) -> Dict:
        """
        Private method to calculate the secondary statistics

        :param sec: The dict containing information about all detected foci
        :return: The secondary statistics dict
        """
        sec_stat = {}
        for key, inf in sec.items():
            inten = inf["intensity"]
            area = inf["area"]
            sec_stat[key] = {
                "number": inf["num"],
                "area list": area,
                "area average": np.average(area),
                "area median": np.median(area),
                "area std": np.std(area),
                "area minimum": min(area),
                "area maximum": max(area),
                "intensity list": inten,
                "intensity average": np.average(inten),
                "intensity median": np.median(inten),
                "intensity std": np.std(inten),
                "intensity minimum": min(inten),
                "intensity maximum": max(inten)
            }
        return sec_stat

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
            [num_area.append(x) for x in roi.area]
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
        # Reset statistics
        self.stats.clear()

