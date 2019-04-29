"""
Created on 09.04.2019
@author: Romano Weiss
"""
import numpy as np
import csv
import datetime
import os


class ROIHandler:
    __slots__ = [
        "ident",
        "main",
        "rois",
        "idents",
        "stats",
    ]

    def __init__(self, ident=None):
        self.ident = ident
        self.rois = []
        self.idents = []
        self.stats = {}
        self.main = ""

    def __len__(self):
        return len(self.rois)

    def __getitem__(self, item):
        return self.rois[item]

    def add_roi(self, roi):
        self.rois.append(roi)
        if roi.ident not in self.idents:
            self.idents.append(roi.ident)
        if roi.main:
            self.main = roi.ident
        self.stats.clear()

    def calculate_statistics(self):
        """
        Method to calculate statistics about the saved ROIs
        :return: dict -- A dictonary containing the calculated statistics
        """
        if not self.stats:
            main = {
                "num": 0,
                "num empty": 0,
                "area": []
            }
            sec_ass = []
            sec = {}
            for roi in self.rois:
                temp_stat = roi.calculate_statistics()
                if roi.main:
                    main["num"] += 1
                    main["area"].append(temp_stat["area"])
                else:
                    if roi.ident not in sec:
                        sec[roi.ident] = {
                            "num": 1,
                            "area": [temp_stat["area"]],
                            "intensity": [temp_stat["intensity"]]
                        }
                    else:
                        sec[roi.ident]["num"] += 1
                        sec[roi.ident]["area"].append(temp_stat["area"])
                        sec[roi.ident]["intensity"].append(temp_stat["intensity"])
                        if roi.ident not in sec_ass:
                            sec_ass.append(roi.ident)
            sec_stat = {}
            for key, inf in sec:
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
                "empty": main["num"] - len(sec),
                "area list": area,
                "area average": np.average(area),
                "area median": np.median(area),
                "area std": np.std(area),
                "area minimum": min(area),
                "area maximum": max(area),
                "intensity list": inten,
                "sec idents": sec_stat.keys(),
                "sec statistics": sec_stat
            }
        return self.stats

    def get_data_as_dict(self):
        """
        Method to retrieve the stored ROI data as list
        :return: The data as dict
        """
        tempdat = {}
        header = ["Hash", "Width", "Height", "Center"]
        header.extend(self.idents)
        header.remove(self.main)
        tempdat["header"] = header
        tempdat["data"] = []
        for roi in self.rois:
            if roi.main:
                tempstat = roi.calculate_dimensions()
                row = [
                    hash(roi),
                    tempstat["width"],
                    tempstat["height"],
                    tempstat["center"]
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
                tempdat["data"].append(row)
        return tempdat

    def export_data_as_csv(self, path, delimiter=";", quotechar="|"):
        """
        Method to save the roi data to a csv table
        :param path: The folder to save the file in
        :param delimiter: The char used to separate cells
        :param quotechar: The char used as a substitute for \"\" or \'\'
        :return: True if the csv could be exported
        """
        with open(os.path.join(path, "{}.csv".format(self.ident)), 'w', newline='') as file:
            writer = csv.writer(file, delimiter=delimiter,
                                quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["File id:", self.ident])
            writer.writerow(["Date:", datetime.datetime.now().strftime("%d.%m.%Y, %H:%M:%S")])
            writer.writerow(["Channels:", self.idents])
            writer.writerow(["Main Channel:", self.main])
            row = ["Index", "Width", "Height", "Center"]
            row.extend(self.idents)
            row.remove(self.main)
            writer.writerow(row)
            for roi in self.rois:
                if roi.main:
                    tempstat = roi.calculate_dimensions()
                    row = [
                        hash(roi),
                        tempstat["width"],
                        tempstat["height"],
                        tempstat["center"]
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
                    writer.writerow(row)
        return True