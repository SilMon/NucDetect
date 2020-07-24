"""
Created on 09.04.2019
@author: Romano Weiss
"""
from __future__ import annotations

import csv
import datetime
import os
from typing import Union, Dict, List, Tuple

import numpy as np

from core.roi.ROI import ROI


class ROIHandler:
    __slots__ = [
        "ident",
        "main",
        "rois",
        "idents",
        "stats",
    ]

    def __init__(self, ident: int = None):
        """
        :param ident: md5 hash of the image this handler is associated with
        """
        self.ident = ident
        self.rois: List[ROI] = []
        self.idents: List[str] = []
        self.stats: Dict[str, Union[int, float]] = {}
        self.main = ""

    def __len__(self):
        return len(self.rois)

    def __getitem__(self, item):
        return self.rois[item]

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
                "intensity": []
            }
            sec = {}
            channels = [img[..., x] for x in range(img.shape[2])]
            for roi in self.rois:
                temp_stat = roi.calculate_statistics(channels[self.idents.index(roi.ident)])
                if roi.main:
                    main["num"] += 1
                    main["area"].append(temp_stat["area"])
                    main["intensity"].append(temp_stat["intensity average"])
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
                    "area maximum": min(area),
                    "intensity list": inten,
                    "intensity average": np.average(inten),
                    "intensity median": np.median(inten),
                    "intensity std": np.std(inten),
                    "intensity minimum": min(inten),
                    "intensity maximum": max(inten)
                }
            area = main["area"]
            inten = main["intensity"]
            self.stats = {
                "number": main["num"],
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

    def get_data_as_dict(self) -> Dict[str, List[Union[str, int, float, Tuple[int, int]]]]:
        """
        Method to retrieve the stored ROI data as list

        :return: The data as dict
        """
        tempdat = {}
        header = ["Image", "Center[(y, x)]", "Area [px]", "Ellipticity[%]"]
        if self.idents:
            header.extend([x for x in self.idents if x != self.main])
            tempdat["header"] = header
            tempdat["data"] = []
            tempdat["footer"] = []
            stats = [0] * len(self.idents)
            for roi in self.rois:
                stats[self.idents.index(roi.ident)] += 1
                if roi.main:
                    tempstat = roi.calculate_dimensions()
                    tempell = roi.calculate_ellipse_parameters()
                    row = [
                        self.ident,
                        tempstat["center"],
                        tempstat["area"],
                        tempell["shape_match"]
                    ]
                    secstat = {}
                    for roi2 in self.rois:
                        if roi2.associated is roi:
                            if roi2.ident in secstat:
                                secstat[roi2.ident] += 1
                            else:
                                secstat[roi2.ident] = 1
                    for chan in self.idents:
                        if chan in secstat.keys():
                            row.append(secstat[chan])
                        elif chan != self.main:
                            row.append(0)
                    tempdat["data"].append(row)
            if self.main is not "":
                tempdat["footer"].append(("Detected Nuclei:",  stats[self.idents.index(self.main)]))
                for chan in self.idents:
                    if chan != self.main:
                        tempdat["footer"].append((f"Detected {chan} foci:",  stats[self.idents.index(chan)]))
        else:
            tempdat["header"] = header
            tempdat["data"] = ()
            tempdat["footer"] = ()
        return tempdat

    def export_data_as_csv(self, path: str, delimiter: str = ";",
                           quotechar: str = "|", ident: str = None) -> bool:
        """
        Method to save the roi data to a csv table

        :param path: The folder to save the file in
        :param delimiter: The char used to separate cells
        :param quotechar: The char used as a substitute for \"\" or \'\'
        :param ident: Optional identifier used for the file name
        :return: True if the csv could be exported
        """
        with open(os.path.join(path, f"{self.ident if ident is None else ident}.csv"), 'w', newline='') as file:
            writer = csv.writer(file, delimiter=delimiter,
                                quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["File id:", self.ident])
            writer.writerow(["Date:", datetime.datetime.now().strftime("%d.%m.%Y, %H:%M:%S")])
            writer.writerow(["Channels:", self.idents])
            writer.writerow(["Main Channel:", self.main])
            dat = self.get_data_as_dict()
            for data in dat["footer"]:
                writer.writerow(data)
            writer.writerow(dat["header"])
            for data in dat["data"]:
                if ident is not None:
                    data[0] = ident
                writer.writerow(data)
        return True
