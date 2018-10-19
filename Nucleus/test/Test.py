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
    det = Detector()
    det.load_image(url)
    det.analyse_images()
    det.show_result(url)
    det.show_result_image(url)

if __name__ == "__main__":
    file_name = os.path.join(os.path.dirname(__file__), "test_2.tif")
    test_func(file_name)
    
