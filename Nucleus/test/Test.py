'''
Created on 02.10.2018
@author: Romano Weiss
'''

import os
from skimage.feature import canny
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


def test_func(url):
    '''
    Function to test the general approach of image processing with skimage
    keyword arguments:
    url -- URL of the image file
    '''
    image = io.imread(url)
    plt.imshow(image[..., 1], cmap="gray")
    plt.show()
    plt.imshow(image[..., 1].astype("float64"), cmap="gray")
    plt.show()
    plt.imshow(canny(image[..., 1].astype("float64")), cmap="gray")
    plt.show()


if __name__ == "__main__":
    file_name = r""
    test_func(file_name)
