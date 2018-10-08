'''
Created on 06.10.2018

'''

from Nucleus.image import channel

class ROI:
    '''
    classdocs
    '''
    NO_ENCLOSURE = -1
    PARTIAL_ENCLOSURE = 0
    FULL_ENCLOSURE = 1
    
    def __init__(self, points=None):
        '''
        Constructor to initialize the ROI
        '''
        self.width = None
        self.height = None
        self.center = None
        self.points = []
        self.green = []
        self.red = []
        if points != None:
            self.points.append(points) 
        
    def add_point(self, point):
        self.points.append(point)

    def calculate_center(self):
        #TODO
        pass
    
    def calculate_area(self):
        #TODO
        pass
    
    def calculate_width(self):
        #TODO
        pass
    
    def calculate_height(self):
        #TODO
        pass
    
    def merge(self, roi):
        self.points.extend(roi.points)
        self.green.extend(roi.green)
        self.red.extend(roi.red)
    
    def add_roi(self, roi, chan):
        if chan == channel.RED:
            self.red.append(roi)
        elif chan == channel.GREEN:
            self.green.append(roi)
            
    def add_roi_partially(self, roi, chan):
        if chan == channel.GREEN:
            a = set(self.green)
            b = set(roi.green)
            self.green.append(a & b)
            roi.points = list(a - b)
        elif chan == channel.RED:
            a = set(self.red)
            b = set.red
            self.green.append(a & b)
            roi.points = list(a - b)
    
    def determine_enclosure(self, roi, chan):
        #Use set intersection to determine enclosure
        if chan == channel.GREEN:
            a = set(self.green)
            b = set(roi.green)
            if b <= a:
                return ROI.FULL_ENCLOSURE
            elif len(a & b) > 0:
                return ROI.PARTIAL_ENCLOSURE
            else:
                return ROI.NO_ENCLOSURE
                 
        elif chan == channel.RED:
            a = set(self.red)
            b = set(roi.red)
            if b <= a:
                return ROI.FULL_ENCLOSURE
            elif len(a & b) > 0:
                return ROI.PARTIAL_ENCLOSURE
            else:
                return ROI.NO_ENCLOSURE
