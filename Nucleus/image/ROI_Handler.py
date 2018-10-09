'''
Created on 06.10.2018

'''
from Nucleus.image.ROI import ROI
from Nucleus.image import channel

class ROI_Handler:
    '''
    Class to detect and handle ROIs
    '''
    

    def __init__(self,blue_ws, green_ws, red_ws):
        '''
        Constructor
        '''
        self.blue_ws = blue_ws
        self.red_ws = red_ws
        self.green_ws = green_ws
        self.nuclei = [None] * 100
        self.green = [None] * 500
        self.red = [None] * 500
            
    def analyse_image(self):
        #Analysis of the blue channel
        for y in range(len(self.blue_ws)):
            for x in range(len(self.blue_ws[0])):
                blue = self.blue_ws[y][x]
                green = self.green_ws[y][x]
                red = self.red_ws[y][x]
                #Detection of nuclei
                if blue != 0:
                    if self.nuclei[blue] == None:
                        roi = ROI()
                        roi.add_point((x,y))
                        self.nuclei[blue] = roi
                    else:
                        self.nuclei[blue].add_point((x,y))
                #Detection of green foci
                if green != 0:
                    if self.green[green] == None:
                        roi = ROI(chan=channel.GREEN)
                        roi.add_point((x,y))
                        self.green[green] = roi
                    else:
                        self.green[green].add_point((x,y))                          
                #Detection of red foci
                if red != 0:
                    if self.red[red] == None:
                        roi = ROI(chan=channel.RED)
                        roi.add_point((x,y))
                        self.red[red] = roi
                    else:
                        self.red[red].add_point((x,y))
        #Determine the green and red ROIs each nucleus includes
        gre_rem = []
        red_rem = []
        for nuc in self.nuclei:
            gre_rem.clear()
            red_rem.clear()
            if nuc != None:
                for gre in self.green:
                    if gre != None:
                        if nuc.add_roi(gre):
                            gre_rem.append(gre)
                for red in self.red:
                    if red != None:
                        if nuc.add_roi(red):
                            red_rem.append(red)
            self.green = [x for x in self.green if x not in gre_rem]
            self.red = [x for x in self.red if x not in red_rem]
