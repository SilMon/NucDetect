'''
Created on 02.10.2018

'''
from skimage.exposure import histogram
RED = 0
GREEN = 1
BLUE = 2

def extract_channel(img_array, channel=2, gray=True):
    '''
    Method to extract the channels of a color image. Image is to be expected to be in the RGB color space
    
    Keyword arguments:
    img_array(ndarray) -- The color image to extract the channel from
    channel(int) -- int to represent which channel should be extracted; use the provided module variables (default:RED)
    gray (bool) -- If True, the channel image will be converted to grayscale, otherwise the resulting image is a color image with all other channels set to 0
    
    Returns:
    ndarray: The color channel as ndarray 
    '''
    channels = [0,1,2]                      #List of all possible color channels
    channels.remove(channel)                #Removal of the color channel to keep from list of av. channels
    if gray:
        ch = img_array[...,channel]    
    else:
        ch = img_array.copy()
        ch[:,:,channels[0]] = 0             #Setting the 2 remaining channels to 0
        ch[:,:,channels[1]] = 0
    return ch

def get_minmax(img_array, percent=0, channel=RED, gray=True):
    '''
    Method to get the minimum and maximum of the given color channel.
    
    Keyword arguments:
    img_array(ndarray) -- The color image to extract the channel from
    percent (float, optional) -- Determines how many pixels (in %) are ignored at either side of the histogram (useful to account for outliers; default: 0)
    channel(int, optional) -- int to represent which channel should be extracted; use the provided module variables (default:RED)
    gray (bool, optional) -- If True, the channel image will be converted to grayscale, otherwise the resulting image is a color image with all other channels set to 0 (default:True)
    
    Returns:
    tuple: (min_value,max_value)
    '''
    if percent < 0 or percent > 1:
        raise ValueError("Invalid percentage provided!")
    hist = get_channel_histogram(img_array, channel, gray)
    total = len(img_array[0]) * len(img_array[1])
    ignore = percent * total +1
    max_pix = 0
    min_pix = 0
    max_val = len(hist[0])
    min_val = 0
    for i in reversed(hist[0]):
        max_pix += i
        if max_pix >= ignore:
            break
        else:
            max_val -= 1
    for i in hist[0]:
        min_pix += i
        if min_pix >= ignore:
            break;
        else:
            min_val += 1
    return (min_val, max_val)

def get_dynamic_range(img_array, channel=RED, gray=True, in_percent=True):
    '''
    Method to obtain the dynamic range of a specific color channel
    
    Keyword arguments:
    img_array(ndarray) -- The color image to extract the channel from
    channel(int, optional) -- int to represent which channel should be extracted; use the provided module variables (default:RED)
    gray (bool, optional) -- If True, the channel image will be converted to grayscale, otherwise the resulting image is a color image with all other channels set to 0 (default:True)
    in_percent(bool, optional) -- If true, the calculated dynamic range will be converted to float (between 0-1)
    
    Returns:
    int: If in_percent=False - The dynamic range of the respective channel (0-255)
    float: If in_percent=True - The dynamic range of the respective channel (0-1)
    '''
    minmax = get_minmax(img_array, channel, gray)
    dr = minmax[1] - minmax[2]
    if in_percent : 
        return dr/255
    else:
        return dr

def get_channel_histogram(img_array, channel=RED,gray=True, nbins=255):
    '''
    Method to obtain the histogram of the respective channel.
    
    Keyword arguments:
    img_array(ndarray) -- The color image to extract the channel from
    channel(int, optional) -- int to represent which channel should be extracted; use the provided module variables (default:RED)
    nbins(int, optional) -- The number of bins used to calculate the histogram (default:255)
    
    Returns:
    array: The calculated channel histogram as 2D array. 
    '''
    color_channel = extract_channel(img_array, channel, gray)
    return histogram(color_channel, nbins)

def percentile_threshold(img_array, channel=RED, p0=0.8, ignore=0.0005):
    '''
    Method to threshold a specific color channel globally. The overall threshold is determined by calculating the the 80. percentile. Thereby, ignore percent are excluded from the calculation to account for outliers.
    
    Keyword arguments:
    img_array(ndarray) -- The color image to extract the channel from
    channel(int, optional) -- int to represent which channel should be extracted; use the provided module variables (default:RED)
    p0(float, optional) -- The percentile to calculate the threshold. (0-1;default:0.8)
    ignore(float, optional) -- The percentage of pixels to ignore in the calculation. Is used to account for outliers. (0-1;default:0.0005)
    
    Returns:
    float: The calculated threshold (0.0-255.0)
    '''
    return get_minmax(img_array, ignore, channel, True)[1] * p0