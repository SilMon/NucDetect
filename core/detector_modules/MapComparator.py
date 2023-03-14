import logging
import warnings
from typing import List, Tuple, Dict, Union

import numpy as np

from roi.AreaAnalysis import imprint_area_into_array, convert_area_to_array
from roi.ROI import ROI


class MapComparator:
    __slots__ = [
        "main",
        "foci1",
        "foc1_bin",
        "foci2",
        "foc2_bin",
        "img_shape"
    ]

    def __init__(self, main: List[ROI], foci1: List[ROI], foci2: List[ROI], img_shape: Tuple[int, int]):
        """
        :param main: List of all detected nuclei
        :param foci1: List of all detected foci for method 1
        :param foci2: List of all detected foci for method 2
        :param img_shape: The shape (height, width) of the image the ROI are derived from
        """
        self.main: List[ROI] = main
        self.foci1: List[ROI] = foci1
        self.foci2: List[ROI] = foci2
        self.img_shape: Tuple[int, int] = img_shape
        self.foc1_bin, self.foc2_bin = self.create_hash_maps_for_foci()

    def create_hash_maps_for_foci(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to get the hash maps for both given foci lists

        :return: The hash maps
        """
        # Create maps for all foci
        focmap1 = np.zeros(shape=self.img_shape, dtype="int64")
        focmap2 = np.zeros(shape=self.img_shape, dtype="int64")
        for focus_ip in self.foci1:
            imprint_area_into_array(focus_ip.area, focmap1, hash(focus_ip))
        for focus_ml in self.foci2:
            imprint_area_into_array(focus_ml.area, focmap2, hash(focus_ml))
        return focmap1, focmap2

    def get_match_for_nuclei(self) -> None:
        """
        Method to compare the detected foci per nucleus

        :return: None
        """
        for nucleus in self.main:
            focar1 = convert_area_to_array(nucleus.area, self.foc1_bin)
            focar2 = convert_area_to_array(nucleus.area, self.foc2_bin)
            # Create temporary map
            foc1 = (focar1 > 0).astype(int)
            foc2 = (focar2 > 0).astype(int)
            # Compare both maps for the given nucleus
            comp_area = np.logical_and(foc1, foc2) > 0
            match = np.sum(comp_area) / max(np.sum(foc1), np.sum(foc2), 1)
            if nucleus.match != -1:
                nucleus.match = (nucleus.match + match) / 2
            else:
                nucleus.match = match

    @staticmethod
    def create_overlap_dict(focarea1: np.ndarray, focarea2: np.ndarray) -> Dict[int, List[int]]:
        """
        Method to check the foci areas for overlaps

        :param focarea1: The first focus area
        :param focarea2: The second focus area
        :return: List of overlapping foci
        """
        overlap = {}
        for y in range(focarea1.shape[0]):
            for x in range(focarea1.shape[1]):
                # Get pixel at position y:x
                pix = focarea1[y][x]
                pix2 = focarea2[y][x]
                if pix > 0 and pix2 > 0:
                    if pix not in overlap:
                        overlap[pix] = []
                    if pix2 not in overlap[pix]:
                        overlap[pix].append(pix2)
        return overlap

    def merge_overlapping_foci(self) -> List[ROI]:
        """
        Method to merge overlapping nuclei

        :return: The cleaned list of foci and the percentage of overlap between
        """
        overlap = self.create_overlap_dict(self.foc1_bin, self.foc2_bin)
        # Get a list of all potential ROI
        roi = self.foci1 + self.foci2
        match = []
        focind = 0
        channel = "No channel given"
        for key, values in overlap.items():
            # Get the focus
            foc1 = self.get_foci_via_hash(self.foci1, key)
            if foc1:
                foc1 = foc1[0]
                channel = foc1.ident
                # Remove the large ROI in favor of the smaller roi
                if len(values) > 1:
                    roi.remove(foc1)
                    continue
                # Get list of potentially overlapping roi
                foc2 = self.get_foci_via_hash(self.foci2, values)
                # Calculate the overlap for each given focus
                spec_overlap = self.calculate_overlap(foc1, foc2)
                foc1.match = np.average(spec_overlap)
                # Iterate over all overlapping foci from the second map
                for ind, foc2 in enumerate(foc2):
                    focind += 1
                    # Get specific overlap for this focus
                    sov = spec_overlap[ind]
                    foc2.match = sov
                    if sov > 0.2:
                        if hash(foc2) in match:
                            continue
                        match.append(hash(foc2))
                        roi.remove(foc2)
                        foc2.detection_method = "Removed"
                        # Get overlapping area
                        foc1.set_area(self.get_overlapping_area(foc1.area, foc2.area))
                        foc1.detection_method = "Merged"
            else:
                warnings.warn(f"Focus with hash {key} not found!")
        logging.info(f"Channel: {channel}\t{len(match)} matching foci found and merged")
        return roi

    @staticmethod
    def get_foci_via_hash(foci: List[ROI], hashes: Union[int, List[int]]) -> List[ROI]:
        """
        Method to extract the ROI given by hash

        :param foci: The list of foci to extract from
        :param hashes: Either the hash of the ROI to extract or a list of hashes to extract
        :return: List of extracted ROI
        """
        if not isinstance(hashes, List):
            return [x for x in foci if hash(x) == hashes]
        else:
            foci_ = []
            for hash_ in hashes:
                foci_.extend(MapComparator.get_foci_via_hash(foci, hash_))
            return foci_

    @staticmethod
    def get_overlapping_area(lines1, lines2) -> List[Tuple[int, int, int]]:
        """
        Method to get the overlapping area bewteen two rl-encoded areas

        :param lines1: The first area
        :param lines2: The second area
        :return: The overlapping area
        """
        # Get potentially overlapping lines
        sl1, sl2 = MapComparator.get_potentially_overlapping_lines(lines1, lines2)
        # Check both areas for u-turn
        sl1 = MapComparator.check_for_u_turn(sl1)
        sl2 = MapComparator.check_for_u_turn(sl2)
        if not sl1 or not sl2:
            raise ValueError("Areas not overlapping!")
        else:
            area = []
            # Zip bot lists
            for l1, l2 in zip(sl1, sl2):
                area.append(MapComparator.get_overlap_line(l1, l2))
            return area

    @staticmethod
    def check_for_u_turn(lines: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Method to check if the given list of lines contains lines within the same row
        :param lines: The lines to test
        :return: The tested list of lines
        """
        # Test if lines contains multiple lines with the same row
        rows, counts = np.unique([x[0] for x in lines], return_counts=True)
        tested = []
        # Get row with multiple entries
        for ind, count in enumerate(counts):
            row = int(rows[ind])
            test = [x for x in lines if x[0] == row]
            if count > 1:
                # Merge the lines
                sort = sorted(test, key=lambda x: x[1])
                tested.append((row, sort[0][1], int(np.sum([x[2] for x in sort]))))
            else:
                tested.extend(test)
        return tested

    @staticmethod
    def get_overlap_line(line1: Tuple[int, int, int], line2: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Method to merge both given lines

        :param line1: The first line
        :param line2: The second line
        :return: The merged line
        """
        if line1[0] != line2[0]:
            raise ValueError("Lines are not in the same row!")
        # Get the left line
        if line1[1] < line2[1]:
            return line1[0], line2[1], MapComparator.get_line_overlap(line1, line2)
        else:
            return line1[0], line1[1], MapComparator.get_line_overlap(line1, line2)

    @staticmethod
    def get_potentially_overlapping_lines(lines1, lines2) -> Tuple:
        """
        Method to get tuples of potentially overlapping lines for the given Areas

        :param lines1: The lines of the first area
        :param lines2: The lines of the second area
        :return: The potentially overlapping lines between both areas
        """
        # Sort both lists according to their row
        sort1 = sorted(lines1, key=lambda x: x[0])
        sort2 = sorted(lines2, key=lambda x: x[0])
        # Check which area is higher
        sl1 = sort1 if sort1[0][0] <= sort2[0][0] else sort2
        sl2 = sort2 if sort1[0][0] <= sort2[0][0] else sort1
        # Get index where areas potentially overlap
        start = None
        for ind, line in enumerate(sl1):
            if line[0] == sl2[0][0]:
                start = ind
        if start is not None:
            return sl1[start:], sl2[:len(sl1[start:]) + 1]
        else:
            return (), ()

    @staticmethod
    def get_amount_of_overlapping_pixels(lines1, lines2) -> int:
        """
        Method to get tuples of potentially overlapping lines for the given Areas

        :param lines1: The lines of the first area
        :param lines2: The lines of the second area
        :return: The overall overlap between both areas
        """
        overlap = []
        sl1, sl2 = MapComparator.get_potentially_overlapping_lines(lines1, lines2)
        if not sl1 or not sl2:
            return 0
        for line1, line2 in zip(sl1, sl2):
            overlap.append(MapComparator.get_line_overlap(line1, line2))
        return np.sum(overlap)

    @staticmethod
    def get_line_overlap(line1: Tuple[int, int, int], line2: Tuple[int, int, int]) -> int:
        """
        Method to get the overlap between to run-length encoded lines

        :param line1: The first line
        :param line2: The second line
        :return: The overlap between both lines
        """
        # Check which of the lines is right
        if line1[1] > line2[1]:
            return MapComparator.calculate_line_overlap(line2, line1)
        elif line1[1] < line2[1]:
            return MapComparator.calculate_line_overlap(line1, line2)
        elif line1[1] == line2[1]:
            return min(line1[2], line2[2])
        return 0

    @staticmethod
    def calculate_line_overlap(line1: Tuple[int, int, int], line2: Tuple[int, int, int]) -> int:
        """
        Calculates the overlap between both lines

        :param line1: The line with the lower x value
        :param line2: The line with the higher x value
        :return: The overlap between both lines
        """
        dist = line2[1] - line1[1]
        if line1[2] > dist + line2[2]:
            return line2[2]
        else:
            return line1[2] - dist

    @staticmethod
    def calculate_overlap(focus1: ROI, focus2: List[ROI]) -> List[float]:
        """
        Method to calculate the overlap for the given ROI

        :param focus1: The focus to check overlap for
        :param focus2: List of foci to check
        :return: The overlaps
        """
        overlaps = []
        for foc2 in focus2:
            # Get the size of both areas
            ar1 = focus1.calculate_dimensions()["area"]
            ar2 = foc2.calculate_dimensions()["area"]
            # Get potentially overlapping lines between both foci
            overlap = MapComparator.get_amount_of_overlapping_pixels(focus1.area, foc2.area)
            overlaps.append((overlap / ar1) if ar1 < ar2 else overlap / ar2)
        return overlaps


