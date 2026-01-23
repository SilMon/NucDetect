import itertools
import time
import warnings
from typing import List, Tuple, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from numba.typed import List as NumbaList
from scipy.spatial import cKDTree

from core.roi.AreaAnalysis import imprint_area_into_array, convert_area_to_array, get_surface
from core.DataProcessing import calculate_overlap_between_two_circles_as_percentage, check_if_two_circles_overlap, \
    check_circles_for_engulfment, get_circle_area
from core.roi.ROI import ROI
from core.roi import AreaAnalysis


class MapComparator:
    __slots__ = [
        "main",
        "foci1",
        "foci2",
        "log"
    ]
    # TODO Größere Änderungen, Detector und sonstige Klassen anpassen
    def __init__(self,
                 main: List[ROI],
                 foci1: List[ROI],
                 foci2: List[ROI],
                 log_function):
        """
        :param main: List of all detected nuclei
        :param foci1: List of all detected foci for method 1
        :param foci2: List of all detected foci for method 2
        :param log_function: Function to log
        """
        # Nuclei
        self.main: List[ROI] = main
        # IP foci/ yH2AX foci
        self.foci1: List[ROI] = foci1
        # ML foci/53BP1 foci
        self.foci2: List[ROI] = foci2
        self.log = log_function
        self.log("Map Comparator:")

    @staticmethod
    def get_match_for_nuclei(nuclei: List[ROI],
                             foci: List[List[ROI]],
                             max_distance: float = 9) -> None:
        """
        Method to check the given foci for co-localization.

        :param nuclei: List of all detected nuclei
        :param foci: List of all detected foci, subdivided by method
        :param max_distance: Maximum distance for two foci centers to be considered co-localized
        :return: None
        """
        # TODO Verwendet derzeit den Wert der Methoden-Übereinstimmung, nicht der Co-Localization!
        # TODO Überprüfen ob die Methode so funktioniert
        start = time.time()
        # Create a dictionary to keep track of matched and unmatched foci
        nucleus_match = {
            hash(x):{
                "Matched": 0,
                "Unmatched": 0,
                "ROI": x
            } for x in nuclei
        }
        # Only the first two channels will determine co-localization
        foci_a, foci_b = foci[:2]
        # Get the overlap between both methods
        pairs, _, _, unmatched_a, unmatched_b = MapComparator.get_overlap_between_lists(foci_a,
                                                                                        foci_b,
                                                                                        max_distance)
        # Mark the foci as co-localized and count the matches for each nucleus
        for index_a, index_b in pairs:
            focus_a, focus_b = foci_a[index_a], foci_b[index_b]
            # Check if focus_a is associated, else ignore the focus
            if not focus_a.associated:
                continue
            focus_a.colocalized = hash(focus_b)
            focus_b.colocalized = hash(focus_a)
            nucleus_match[hash(focus_a.associated)]["Matched"] += 2
        # Add the number of unmatched a foci
        unmatched_foci = list(itertools.compress(foci_a, unmatched_a)) + list(itertools.compress(foci_b, unmatched_b))
        # Set the number of unmatched foci
        for focus in unmatched_foci:
            # Regard only foci that were matched to a nucleus
            if focus.associated:
                nucleus_match[focus.associated]["Unmatched"] += 1
        # Calculate the overlap for each nucleus
        for data in nucleus_match.values():
            matched, unmatched, nucleus = data.values()
            nucleus.match = (matched / (matched + unmatched)) if (matched + unmatched) > 0 else 0

    def merge_overlapping_foci(self, max_distance: float = 5) -> List[ROI]:
        """
        Method to merge overlapping foci

        :param max_distance: Maximum distance for two foci centers to be considered the same focus
        :return: The cleaned list of foci and the percentage of overlap between
        """
        start = time.time()
        foci_a, foci_b = self.foci1, self.foci2
        pairs, matched_a, matched_b, unmatched_a, unmatched_b = self.get_overlap_between_lists(self.foci1,
                                                                                               self.foci2,
                                                                                               max_distance)
        added_a = list(itertools.compress(foci_a, unmatched_a))
        added_b = list(itertools.compress(foci_b, unmatched_b))
        merged_foci = []
        for ind_a, ind_b in pairs:
            focus_a = foci_a[ind_a]
            focus_b = foci_b[ind_b]
            focus_a.merge(focus_b)
            if focus_a.detection_method != "Merged":
                added_a.append(focus_a)
                added_b.append(focus_b)
            else:
                merged_foci.append(focus_a)
        self.log(f"Channel: {foci_a[0].ident}\t{sum(matched_a)} matching foci ({sum(unmatched_a)} unmatched)"
                 f" found and merged in {time.time() - start: .3f} secs")
        return merged_foci + added_a + added_b

    @staticmethod
    def get_overlap_between_lists(foci_a: List[ROI],
                                  foci_b: List[ROI],
                                  max_distance: float) -> Tuple[List[Tuple[int, int]],
                                                                     np.ndarray,
                                                                     np.ndarray,
                                                                     np.ndarray,
                                                                     np.ndarray]:
        """
        Function to get the overlap between both lists of ROI

        :param foci_a: The first list of ROI
        :param foci_b: The second list of ROI
        :param max_distance: The maximum distance between 2 roi centers to be considered the same roi
        :return: The overlapping ROI as pairs of list indices, the matched roi in a, the matched roi in b,
        the unmatched roi in a, the unmatched roi in b
        """
        # TODO checken ob beide Listen auch etwas enthalten
        # Convert the focus list to centroids
        centroids_a = [x.get_minimal_representation()[:2] for x in foci_a]
        centroids_b = [x.get_minimal_representation()[:2] for x in foci_b]
        # Create ckD Trees from both centroid lists
        tree_a = cKDTree(centroids_a)
        tree_b = cKDTree(centroids_b)
        # Get the respective overlap between the foci lists
        matches = tree_a.query_ball_tree(tree_b,
                                         r=max_distance)
        data_a = [(y, x) for y, x in tree_a.data]
        data_b = [(y, x) for y, x in tree_b.data]
        # Get all pairs
        pairs = {
            hash(x): [None, None, None] for x in foci_b
        }
        matched_a = np.zeros(len(data_a), dtype=bool)
        matched_b = np.zeros(len(data_b), dtype=bool)
        for ind, matches in enumerate(matches):
            # Check if any point of B was matched
            if not matches:
                continue
            else:
                # Get the original point
                point_a = data_a[ind]
                # Sort the matches by distance
                points_b = [data_b[x] for x in matches]
                dists = np.linalg.norm(np.asarray(points_b) - np.asarray(point_a), axis=1)
                nearest_neighbor = [matches[x] for x in np.argsort(dists)][0]
                # Get the nearest neighbor as ROI
                focus_b = foci_b[nearest_neighbor]
                # Check if the nearest neighbor was already matched, else match it
                # TODO Edge cases lead sometimes to overlapping foci
                if not matched_b[nearest_neighbor] or dists[0] < pairs[hash(focus_b)][2]:
                    matched_b[nearest_neighbor] = 1
                    matched_a[ind] = 1
                    if pairs[hash(focus_b)][0]:
                        matched_a[pairs[hash(focus_b)][0]] = 0
                    pairs[hash(focus_b)] = [ind, nearest_neighbor, dists[0]]
        unmatched_a = np.invert(matched_a)
        unmatched_b = np.invert(matched_b)
        return ([(x[0], x[1]) for x in pairs.values() if x[0] is not None],
                matched_a, matched_b,
                unmatched_a,
                unmatched_b)
