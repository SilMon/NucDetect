'''
Created on 06.10.2018

'''

from Nucleus.image import channel

class ROI:
    '''
    Class to handle Regions Of Interest (R.O.I.)
    '''
    NO_ENCLOSURE = -1
    PARTIAL_ENCLOSURE = 0
    FULL_ENCLOSURE = 1
    
    def __init__(self, points=None, chan=channel.BLUE):
        '''
        Constructor to initialize the ROI. Each ROI is initialized with no points and assumed to be in the blue channel if not set otherwise
        
        Keyword arguments:
        points(list of 2D tuples, optional): The points which describe the area of the ROI. Contains 2D tuples in form of (x,y)
        channel(int, optional): Describes the channel in which the ROI was found (default: channel.BLUE)
        '''
        self.chan = channel 
        self.width = None
        self.height = None
        self.center = None
        self.points = []        #Points always describes the area of the ROI in form of tuples (x,y)
        if channel == channel.BLUE:
            self.green = []         #Contains Sub-ROI of the green channel (if any)
            self.red = []           #Contains Sub-ROI of the red channel (if any)
        if points != None:
            self.points.append(points) 
        
    def add_point(self, point):
        '''
        Method to add a point to the ROI.
        
        Keyword arguments:
        point(2D tuple): Point to add to the ROI
        '''
        self.points.append(point)

    def __calculate_center(self):
        '''
        Private method to calculate the center of the ROI
        '''
        if self.width == None:
            self.calculate_width()
        if self.height == None:
            self.calculate_height()
        self.center = (self.width/2,self.height/2)
    
    def __calculate_width(self):
        '''
        Private method to calculate the width of the ROI
        '''
        minX = max(self.points,key=itemgetter(0))[0] 
        maxX = max(self.points,key=itemgetter(0))[0]
        self.width = maxX - minX   
    
    def __calculate_height(self):
        '''
        Private method to calculate the height of the ROI
        '''
        minY = max(self.points,key=itemgetter(1))[0] 
        maxY = max(self.points,key=itemgetter(1))[0]
        self.width = maxY - minY  
    
    def merge(self, roi):
        '''
        Method to merge to ROI.
        
        Keyword arguments:
        roi(ROI): The ROI to merge this instance with
        '''
        self.points.extend(roi.points)
        self.green.extend(roi.green)
        self.red.extend(roi.red)
    
    def add_roi(self, roi):
        '''
        Method to add a ROI to this instance. Is different from merge() by only adding the red and green points of the given ROI and ignoring its blue points
        
        Keyword arguments:
        roi(ROI): The ROI to add to this instance
        
        Returns:
        bool -- True if the ROI could be added, False if the roi could not or only partially be added
        '''
        val = self.__determine_enclosure(roi)
        if val == ROI.FULL_ENCLOSURE:
            if roi.chan == channel.GREEN:
                self.green.append(roi)
            if roi.chan == channel.RED:
                self.red.append(roi)
            return True
        elif val == ROI.PARTIAL_ENCLOSURE:
            a = set(self.points)
            b = set(roi.points)
            if roi.chan == channel.GREEN:
                self.green.append(ROI(list(a & b)))
            elif roi.chan == channel.RED:
                self.red.append(ROI.list(a&b))
            roi.points = list(a - b)
            return False
        else:
            return False
    
    def __determine_enclosure(self, roi):
        '''
        Method to determine if a ROI is enclosed by this ROI.
        
        Keyword arguments:
        roi(ROI): The ROI to test enclosure for.
        
        Returns: 
        int --  ROI.FULL_ENCOLURE if the given ROI is completely enclosed by this instance.
                ROI.PARTIAL_ENCLOSURE if the given ROI is partially enclosed by this instance.
                ROI.NO_ENCLOSURE if the given ROI is not enclosed by this instance
        '''
        #Use set intersection to determine enclosure
        a = set(self.points)
        b = set(roi.points)
        if b <= a:
            return ROI.FULL_ENCLOSURE
        elif len(a & b) > 0:
            return ROI.PARTIAL_ENCLOSURE
        else:
            return ROI.NO_ENCLOSURE
