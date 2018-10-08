'''
Created on 02.10.2018

'''
from skimage import io
from Nucleus.image import channel
from skimage.filters import threshold_triangle
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
import time
from Nucleus.image.ROI_Handler import ROI_Handler

def test_func(url):
    '''
    Function to test the general approach of image processing with skimage
    keyword arguments:
    url -- URL of the image file
    '''
    start = time.time()
    img_array = io.imread(url)                                                                                  #3D array of image (color space = RGB)
    
    ch_blue = channel.extract_channel(img_array,channel.BLUE)                                                   #Extraction of the blue color channel (DAPI)
    ch_red = channel.extract_channel(img_array,channel.RED)                                                     #Extraction of the red color channel
    ch_green = channel.extract_channel(img_array, channel.GREEN)                                                #Extraction of the green color channel

    th_blue = threshold_triangle(ch_blue)                                                                       #Threshold calculation for blue channel
    th_red = channel.percentile_threshold(img_array)                                                            #Threshold calculation for red channel
    th_green = channel.percentile_threshold(img_array, channel.GREEN)                                           #Threshold calculation for green channel 
      
    ch_blue_bin = ch_blue > th_blue                                                                             #Actual thresholding of blue channel
    ch_red_bin = ch_red > th_red                                                                                #Actual thresholding of red channel
    ch_green_bin = ch_green > th_green                                                                          #Actual thresholding of green channel
    
    edt_blue = ndi.distance_transform_edt(ch_blue_bin)                                                          #Calculation of eucledian distance maps (EDTs)
    edt_red = ndi.distance_transform_edt(ch_red_bin)                                                                     
    edt_green = ndi.distance_transform_edt(ch_green_bin)                                                                     
                                                                        
    edt_blue_max = peak_local_max(edt_blue,labels=ch_blue_bin,indices=False, footprint=np.ones((91, 91)))       #Calculation of local maxima of EDT
    edt_red_max = peak_local_max(edt_red,labels=ch_red_bin,indices=False, footprint=np.ones((3, 3)))            #Footprint is the min. distance between maxima
    edt_green_max = peak_local_max(edt_green,labels=ch_green_bin,indices=False, footprint=np.ones((3, 3)))
    
    markers_blue = ndi.label(edt_blue_max)[0]
    markers_red = ndi.label(edt_red_max)[0]
    markers_green = ndi.label(edt_green_max)[0]
    
    ws_blue = watershed(-edt_blue, markers_blue,mask=ch_blue_bin)                                               #Watershed of each channel, using determined EDT maxima
    ws_red = watershed(-edt_red, markers_red,mask=ch_red_bin)
    ws_green = watershed(-edt_green, markers_green,mask=ch_green_bin)
    
    handler = ROI_Handler(ws_blue,ws_green,ws_red)
    handler.analyse_image()                                                                                     #Calculation of ROIs
          
    #Displaying the created images
    fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(16, 9))
    fig.canvas.set_window_title('Detection of intranuclear proteins')
    ax = axes.ravel()
    ax[0].imshow(ch_blue,cmap='gray')
    ax[0].set_title("Blue channel")
    ax[1].imshow(ch_red, cmap='gray')
    ax[1].set_title("Red channel")
    ax[2].imshow(ch_green,cmap='gray')
    ax[2].set_title("Green channel")
    ax[3].imshow(ch_blue_bin,cmap='gray')
    ax[3].set_title("Thresholding - triangle")
    ax[4].imshow(ch_red_bin,cmap='gray')
    ax[4].set_title("Thresholding - custom" + " ({0:.2f})".format(th_red))
    ax[5].imshow(ch_green_bin,cmap='gray')
    ax[5].set_title("Thresholding - custom" + " ({0:.2f})".format(th_green))
    ax[6].imshow(edt_blue, cmap="gray")
    ax[6].set_title("Euclidean Distance Transform")
    ax[7].imshow(edt_red, cmap="gray")
    ax[7].set_title("Euclidean Distance Transform")
    ax[8].imshow(edt_green, cmap="gray")
    ax[8].set_title("Euclidean Distance Transform")
    ax[9].imshow(ws_blue,cmap='gray')
    ax[9].set_title("Watershed")
    ax[10].imshow(ws_red,cmap='gray')
    ax[10].set_title("Watershed")
    ax[11].imshow(ws_green,cmap='gray')
    ax[11].set_title("Watershed")
    for a in ax:
        a.axis('off')
    plt.gray()
    stop = time.time()
    print("Time: " + "{0:.2f} secs".format((stop-start)))
    plt.show()
    
def count_regions(img_array):
    pass

def merge_adjacent_regions(img_array, size):
    pass

if __name__ == '__main__':
    test_func(r"PATH/TO/FILE")
    
