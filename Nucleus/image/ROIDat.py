"""
Created on 04.04.2019

@author: Romano Weiss
"""


class ROIDat:
    """
    Class to hold ROI objects and provide statistical analysis
    """

    def __init__(self):
        self.channels = {}
        self.channel_ident = []
        self.main = ""
        self.rois = {}

        pass

    def add_channel(self, ident, array):
        if ident not in self.channel_ident:
            self.channel_ident.append(ident)
        self.channels[ident] = array

    def add_roi(self, roi, ident):
        if ident in self.rois:
            self.rois[ident].append[roi]
        else:
            self.rois[ident] = [roi]

    def get_roi_channel(self, roi):
        for key in self.rois.keys():
            if roi in self.rois[key]:
                return key
        return None
