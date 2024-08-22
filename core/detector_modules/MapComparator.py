import time
import warnings
from typing import List, Tuple, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from numba.typed import List as NumbaList

from roi.AreaAnalysis import imprint_area_into_array, convert_area_to_array
from DataProcessing import calculate_overlap_between_two_circles_as_percentage, check_if_two_circles_overlap, \
    check_circles_for_engulfment, get_circle_area
from roi.ROI import ROI


class MapComparator:
    __slots__ = [
        "main",
        "foci1",
        "foci1_map",
        "foci2",
        "foci2_map",
        "img_shape",
        "log"
    ]

    def __init__(self,
                 main: List[ROI],
                 foci1_map: np.ndarray,
                 foci1: List[ROI],
                 foci2_map: np.ndarray,
                 foci2: List[ROI],
                 img_shape: Tuple[int, int], log_function):
        """
        :param main: List of all detected nuclei
        :param foci1_map: Map contain all detected foci of method 1 as numerical identifiers
        :param foci1: List of all detected foci for method 1
        :param foci2_map: Map contain all detected foci of method 2 as numerical identifiers
        :param foci2: List of all detected foci for method 2
        :param img_shape: The shape (height, width) of the image the ROI are derived from
        :param log_function: Function to log
        """
        self.main: List[ROI] = main
        # IP foci
        self.foci1_map: np.ndarray = foci1_map
        self.foci1: List[ROI] = foci1
        # ML foci
        self.foci2_map: np.ndarray = foci2_map
        self.foci2: List[ROI] = foci2
        self.img_shape: Tuple[int, int] = img_shape
        self.log = log_function
        self.log("Map Comparator:")

    @staticmethod
    def get_match_for_nuclei(nuclei: List[ROI], foci: List[List[ROI]]) -> None:
        """
        Method to check the given foci for co-localization.

        :param nuclei: List of all detected nuclei
        :param foci: List of all detected foci, subdivided by method
        :return: None
        """
        # Create a conversion dictionary for all methods
        nucleus_focus_association_dict = MapComparator.get_nucleus_focus_association_dictionary(nuclei, foci)
        hash_roi_converter = MapComparator.get_hash_roi_converter(nuclei)
        # Get dict that allows hash to ROI conversion
        focus_conversion_dict = MapComparator.create_focus_conversion_dict(foci)
        # Iterate over all nuclei and check the co-localization for all foci
        for nucleus, foci in nucleus_focus_association_dict.items():
            co_localized = []
            # Get the foci of the first method
            foci1 = foci[0]
            # Get the foci of the second method
            foci2 = foci[1]
            # Iterate over the first foci list and check for overlap
            for focus in foci1:
                for focus2 in foci2:
                    if check_if_two_circles_overlap(focus, focus2):
                        co_localized.append((focus, focus2))
                        break
            # Iterate over the co-localization list and set the foci to co-localized
            for focus_pair in co_localized:
                focus_conversion_dict[focus_pair[0][-1]].colocalized = True
                focus_conversion_dict[focus_pair[1][-1]].colocalized = True
            # Calculate the amount of co-localization for this nucleus
            hash_roi_converter[nucleus].match = len(co_localized) / (len(foci1) + len(foci2))

    @staticmethod
    def get_hash_roi_converter(roi: List[ROI]) -> Dict[int, ROI]:
        """
        Method to get an ROI object from their hash

        :param roi: The roi to get the converter for
        :return: The converter dict
        """
        return {hash(x): x for x in roi}

    def merge_overlapping_foci(self, max_overlap: float = 0.5) -> List[ROI]:
        """
        Method to merge overlapping foci

        :param max_overlap: Max overlap foci should have
        :return: The cleaned list of foci and the percentage of overlap between
        """
        start = time.time()
        # Get a dictionary linking all nuclei with their respective foci
        nucleus_focus_association_dict = self.get_nucleus_focus_association_dictionary(self.main,
                                                                                       [self.foci1, self.foci2])
        # Get dict that allows hash to ROI conversion
        focus_conversion_dict = self.create_focus_conversion_dict([self.foci1, self.foci2])
        # List that contains all foci that should be added
        focus_addition_lst = []
        # List that contains all focus pairs that should be merged
        focus_merge_lst = []
        # Check the overlap for each focus
        for nucleus, foci in nucleus_focus_association_dict.items():
            temp_addition_lst, temp_merge_lst = self.check_focus_overlap(foci[0], foci[1], max_overlap)
            focus_addition_lst.extend(temp_addition_lst)
            focus_merge_lst.extend(temp_merge_lst)
        # Merge the foci marked for it
        merged_roi = self.merge_marked_roi(focus_merge_lst, focus_conversion_dict)
        # Get the roi marked for addition
        # Iterate over the addition foci, that their overlap and create a list containing the ROI objects
        added_roi = []
        for data in focus_addition_lst:
            # Get the actual roi object
            roi = focus_conversion_dict[data[1][-1]]
            # Set the overlap
            roi.match = data[0]
            # Add the roi to the list
            added_roi.append(roi)
        self.log(f"Channel: {added_roi[0].ident}\t{len(focus_merge_lst)} matching foci"
                 f" found and merged in {time.time() - start: .3f} secs")
        return merged_roi + added_roi

    @staticmethod
    def get_nucleus_focus_association_dictionary(nuclei: List[ROI],
                                                 foci: List[List[ROI]]) -> Dict[int, Tuple[List[Tuple], List[Tuple]]]:
        """
        Method to get a dictionary that associates all nuclei with their respective foci for both methods

        :param nuclei: The nuclei to create the checklist for
        :param foci: List of all foci
        :return: The dict linking the idents of the nucleus with the minimal repr. of the foci for both methods
        """
        associations = {}
        # Iterate over each nucleus
        for nucleus in nuclei:
            nucleus_hash = hash(nucleus)
            associations[nucleus_hash] = [], []
            for method, focus_lst in enumerate(foci):
                # Iterate over all foci for this method
                for focus in focus_lst:
                    if hash(focus.associated) == nucleus_hash:
                        associations[nucleus_hash][method].append(focus.get_minimal_representation())
        return associations

    @staticmethod
    def check_for_excessive_overlap(foci1: List[Tuple[int, int, int, int]],
                                    foci2: List[Tuple[int, int, int, int]],
                                    threshold: float = 0.4) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
        """
        Method to check if the foci of list 1 overlap excessively with those of list 2. Results are given as
        Overlap, Minimal Repr. pairs.

        :param foci1: The first list of foci in minimal representation
        :param foci2: The second list of foci in minimal representation
        :param threshold: Threshold for overlap. If the total overlap exceeds this, the focus marked for deletion
        :return: Foci without excessive overlap, foci with excessive overlap, foci without any overlap
        """
        without_excessive_overlap = []
        with_excessive_overlap = []
        without_overlap = []
        for focus in foci1:
            total_overlap = 0
            overlapping_foci = 0
            # Iterate over all foci from foci2 and check for overlap
            for focus2 in foci2:
                overlap = calculate_overlap_between_two_circles_as_percentage(focus, focus2)
                # If the focus overlaps, add it to the list
                if overlap > 0:
                    total_overlap += overlap
                    overlapping_foci += 1
            # Check if the overlap is excessive
            if overlapping_foci > 1 and total_overlap > threshold:
                with_excessive_overlap.append((total_overlap, focus))
            elif overlapping_foci == 0:
                without_overlap.append((0, focus))
            else:
                without_excessive_overlap.append((total_overlap, focus))
        return without_excessive_overlap, with_excessive_overlap, without_overlap

    def check_focus_overlap(self,
                            foci1: List[Tuple[int, int, int, int]],
                            foci2: List[Tuple[int, int, int, int]],
                            max_overlap: float) -> Tuple[List, List]:
        """
        Method to check the overlap of the foci opf list 1 with the foci of list 2

        :param foci1: First list of foci in minimal representation
        :param foci2: Second list of foci in minimal representation
        :param max_overlap: Maximum acceptable overlap 0-1
        :return: A list of all foci that can be directly added and a list of foci that have to be merged
        """
        # List of foci that can be directly added
        focus_addition_lst = []
        """
        List that contains all focus pairs that should be merged
        Contains pairs of foci repr. as Tuple. The first focus is only the repr, the second and later tuples
        contain the overlap with focus 0 and the repr. of the focus
        """
        focus_merge_lst = []
        # Check method 2 for excessive overlap
        foci2, _, foci2_add = self.check_for_excessive_overlap(foci2, foci1, 0)
        foci2 = [x[1] for x in foci2]
        # Iterate over every focus of method 1 and check for the overlap
        for focus in foci1:
            overlapping_foci = []
            area1 = get_circle_area(focus)
            # For each focus of method 2, check the overlap
            for focus2 in foci2:
                overlap = calculate_overlap_between_two_circles_as_percentage(focus, focus2)
                # If the focus overlaps, append it
                if overlap > 0:
                    overlapping_foci.append((overlap, focus2))
            # If the focus overlaps with multiple foci
            if len(overlapping_foci) > 1:
                # Calculate the total overlap
                ovl = [x[0] for x in overlapping_foci]
                # Check if the total overlap is larger than the allowed overlap
                if np.sum(ovl) >= max_overlap:
                    # Check if all items are below the threshold or if multiple items
                    check = [x >= max_overlap for x in ovl]
                    if not all(check) or np.sum(check) > 1:
                        # Add all additional foci to merge
                        focus_addition_lst.extend(overlapping_foci)
                    # Check which focus causes the overshoot, add both to merge, add the rest to addition
                    for checker, focus2 in zip(check, overlapping_foci):
                        # If the focus does not cause the overshoot, add it to the addition list
                        if not checker:
                            focus_addition_lst.append(focus2)
                        else:
                            focus_merge_lst.append((focus, focus2))
            # If the focus overlaps with 1 other focus
            elif 0 < len(overlapping_foci) < 2:
                # Check the overlap
                focus2 = overlapping_foci[0][1]
                focus2_overlap = overlapping_foci[0][0]
                # Merge if threshold is exceeded
                if focus2_overlap > max_overlap:
                    focus_merge_lst.append((focus, (focus2_overlap, focus2)))
                # Check if one focus is engulfed in the other
                elif check_circles_for_engulfment(focus, focus2):
                    # Check if the overlap is larger than half of the max overlap
                    if focus2_overlap > max_overlap / 2:
                        # Merge both foci
                        focus_merge_lst.append((focus, (focus2_overlap, focus2)))
                    # If not, add the smaller focus (most likely to be accurate)
                    else:
                        area1 = get_circle_area(focus[2])
                        area2 = get_circle_area(focus2[2])
                        if area1 >= area2:
                            focus_addition_lst.append((focus2_overlap, focus))
                        else:
                            focus_addition_lst.append((focus2_overlap, focus2))
                # If no conditions are met, add both foci
                else:
                    focus_addition_lst.append((focus2_overlap, focus))
                    focus_addition_lst.append((focus2_overlap, focus2))
            # If the focus does not overlap
            else:
                focus_addition_lst.append((0, focus))
        # Add all foci from the second method that do not overlap
        focus_addition_lst.extend(foci2_add)
        return focus_addition_lst, focus_merge_lst

    @staticmethod
    def create_focus_conversion_dict(foci: List[List[ROI]]) -> Dict[int, ROI]:
        """
        Method to create a dictionary that allows the conversion from hash to ROI object

        :return: The created dictionary
        """
        conv_dict = {}
        # Iterate over all foci
        for data in foci:
            for focus in data:
                conv_dict[hash(focus)] = focus
        return conv_dict

    @staticmethod
    def merge_marked_roi(foci: List[Tuple], conv_dict: Dict[int, ROI]) -> List[ROI]:
        """
        Method to merge the foci

        :param foci: Tuple containing the hashes of ROI to be merged
        :param conv_dict: Dictionary allowing the conversion from hash to ROI object
        :return: List of merged ROI
        """
        rois = []
        for merge_lst in foci:
            # Get the first ROI
            focus1 = conv_dict[merge_lst[0][-1]]
            # Get the total overlap
            total_overlap = 0
            # Merge the given focus with all marked foci
            for focus2 in merge_lst[1:]:
                total_overlap += focus2[0]
                focus1.merge(conv_dict[focus2[1][-1]])
            focus1.match = total_overlap
            rois.append(focus1)
        return rois

    def create_roi_identifier_conversion_list(self) -> Tuple[Dict[int, ROI], Dict[int, ROI]]:
        """
        Method to create a list of ROI identifier conversions

        :return: The list of ROI according to their corresponding numerical identifier
         for the first and second focus map
        """
        conv1 = self.create_roi_identifier_conversion_list_for_area(self.foci1_map, self.foci1)
        conv2 = self.create_roi_identifier_conversion_list_for_area(self.foci2_map, self.foci2)
        return conv1, conv2

    @staticmethod
    def create_roi_identifier_conversion_list_for_area(area_map: np.ndarray, roi: List[ROI]) -> Dict[int, ROI]:
        """
        Method to create a list of ROI according to their corresponding numerical identifier

        :param area_map: The binary map containing the roi as numerical areas
        :param roi: List of all rois
        :return:  List of ROI according to their corresponding numerical identifier
        """
        conv: Dict[int: ROI] = {}
        for roi in roi:
            # Get the first row of the roi
            ident = area_map[roi.area[0][0]][roi.area[0][1]]
            if ident == 0:
                raise ValueError(f"Malformed ROI detected! -> {hash(roi)}")
            conv[ident] = roi
        return conv
