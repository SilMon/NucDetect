'''
Created on 15.10.2018

@author: Romano Weiss
'''
from skimage import io
from Nucleus.image import Channel
from skimage.filters import threshold_triangle
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from Nucleus.image.ROI_Handler import ROI_Handler
import matplotlib.pyplot as plt
import numpy as np
import os


class Detector:
    '''
    Class to detect intranuclear proteins in provided fluorescence images
    '''

    def __init__(self, images=None):
        '''
        Constructor to initialize the detector

        Keyword parameters:
        images(dict): A dict containing the images to analyze. It must have
        following structure dict[url_to_image] = image (as 2D array)
        '''
        self.images = images if images is not None else {}
        self.snaps = {}
        self.save_snapshots = True

    def load_image(self, url):
        '''
        Method to add an image to the processing queue.

        Keyword arguments:
        url (str) -- The path to the image file
        '''
        self.images[url] = io.imread(url)

    def load_image_folder(self, direct):
        '''
        Method to load all images of a specific directory

        Keyword arguments:
        direct (str): The path to the directory
        '''
        files = os.listdir(direct)
        for file in files:
            path = os.path.join(direct, file)
            if os.path.isfile(path):
                self.load_image(path)

    def add_images(self, **images):
        '''
        Method to add multiple images at once to the processing queue
        '''
        pass

    def analyse_images(self):
        '''
        Method to analyze all images in the processing queue
        '''
        if len(self.images) is 0:
            raise ValueError("No images provided!")
        for key, img in self.images.items():
            channels = self._get_channels(img)
            qual = self._estimate_image_quality(channels)
            thresholds = self._get_thresholds(channels)
            thr_chan = self._perform_thresholding(channels, thresholds)
            edms = self._calculate_edm(thr_chan)
            loc_max = self._calculate_local_maxima(thr_chan, edms)
            markers = self._perform_labelling(loc_max)
            watersheds = self.perform_watershed(edms, markers, thr_chan)
            handler = ROI_Handler()
            handler.set_watersheds(watersheds)
            handler.analyse_image()
            result = handler.draw_roi(img)
            cur_snaps = {}
            cur_snaps["path"] = key
            cur_snaps["handler"] = handler
            cur_snaps["result"] = result
            cur_snaps["quality"] = qual
            if self.save_snapshots:
                cur_snaps["quality"] = qual
                cur_snaps["channel"] = channels
                cur_snaps["threshold"] = thresholds
                cur_snaps["binarized"] = thr_chan
                cur_snaps["edm"] = edms
                cur_snaps["max"] = loc_max
                cur_snaps["marker"] = markers
                cur_snaps["watershed"] = watersheds
            self.snaps[key] = cur_snaps

    def show_snapshot(self, which=None):
        '''
        Method to show the saved processing snapshots of a specific image
        '''
        if self.save_snapshots and len(self.snaps) > 0:
            if which is not None:
                snap = which
                vals = self.snaps.get(which)
                fig, axes = plt.subplots(ncols=3, nrows=5, figsize=(20, 9))
                fig.canvas.set_window_title("Analysis of: " + snap)
                ax = axes.ravel()
                ax[0].imshow(vals.get("channel")[0], cmap='gray')
                ax[0].set_title("Blue channel")
                ax[1].imshow(vals.get("channel")[1], cmap='gray')
                ax[1].set_title("Red channel")
                ax[2].imshow(vals.get("channel")[2], cmap='gray')
                ax[2].set_title("Green channel")
                ax[3].imshow(vals.get("binarized")[0], cmap='gray')
                ax[3].set_title("Thresholding - triangle")
                ax[4].imshow(vals.get("binarized")[1], cmap='gray')
                ax[4].set_title("Thresholding - custom" +
                                " ({0:.2f})".format(vals.get("threshold")[1]))
                ax[5].imshow(vals.get("binarized")[2], cmap='gray')
                ax[5].set_title("Thresholding - custom" +
                                " ({0:.2f})".format(vals.get("threshold")[2]))
                ax[6].imshow(vals.get("edm")[0], cmap="gray")
                ax[6].set_title("Euclidean Distance Transform")
                ax[7].imshow(vals.get("edm")[1], cmap="gray")
                ax[7].set_title("Euclidean Distance Transform")
                ax[8].imshow(vals.get("edm")[2], cmap="gray")
                ax[8].set_title("Euclidean Distance Transform")
                ax[9].imshow(vals.get("watershed")[0], cmap='gray')
                ax[9].set_title("Watershed")
                ax[10].imshow(vals.get("watershed")[1], cmap='gray')
                ax[10].set_title("Watershed")
                ax[11].imshow(vals.get("watershed")[2], cmap='gray')
                ax[11].set_title("Watershed")
                ax[12].set_title("Result")
                ax[12].imshow(vals.get("result"))
                for a in ax:
                    a.axis('off')
                plt.gray()
                plt.show()

    def show_result(self, which, formatted=True):
        '''
        Method to show the result table of the processing of a specific image.
        Table will be shown in the console

        Keyword arguments:
        which (str): The url to the image file
        formatted(bool): Determines if the output should be formatted
        (default:True)
        '''
        self.snaps.get(which)["handler"].get_data(True, formatted)

    def create_ouput(self, formatted=True):
        '''
        Method to show the result table of the processing of a specific image.
        Table will be saved as CSV file

        Keyword arguments:
        which (str): The url to the image file
        formatted(bool): Determines if the output should be formatted
        (default:True)
        '''
        pass

    def show_result_image(self, which):
        '''
        Method to show the visual representation of the foci detection

        Keyword arguments:
        which (str): The url to the image file
        '''
        plt.imshow(self.snaps.get(which)["result"])
        plt.show()

    def get_snapshot(self, which):
        '''
        Method to obtain the saved processing data for a specific image

        Keyword arguments:
        which (str): The url to the image file

        Returns:
        dict -- A dict which contains the saved processing snapshots. If no
        snapshots are saved, it will only contain the result image
        '''
        return self.snaps.get(which)

    def save_result_image(self, which):
        '''
        Method to save the processing image to file

        Keyword arguments:
        which (str): The url to the image file
        '''
        pass  # TODO

    def _get_channels(self, img):
        '''
        Private Method to extract the channels of a given image

        Keyword arguments:
        img (2D array): The image to extract the channels from

        Returns:
        tuple  -- A tuple containing the three channels in the order
        blue, red, green
        '''
        ch_blue = Channel.extract_channel(img, Channel.BLUE)
        ch_red = Channel.extract_channel(img, Channel.RED)
        ch_green = Channel.extract_channel(img, Channel.GREEN)
        return (ch_blue, ch_red, ch_green)

    def _get_thresholds(self, channels):
        '''
        Private Method to calculate the thresholds for each channel

        Keyword arguments:
        channels (tuple): A tuple containing the three channels in the
        order blue, red, green

        Returns:
        tuple  -- A tuple containing the thresholds for each channel
        in the order blue, red, green
        '''
        th_blue = threshold_triangle(channels[0])
        th_red = Channel.percentile_threshold(channels[1], channel_only=True)
        th_gre = Channel.percentile_threshold(channels[2], channel_only=True)
        return (th_blue, th_red, th_gre)

    def _perform_thresholding(self, channels, thresholds):
        '''
        Private Method to perform channel thresholding.

        Keyword arguments:
        channels (tuple): A tuple containing the three channels in the
        order blue, red, green
        thresholds (tuple): A tuple containing the thresholds to apply
        in the order blue, red, green

        Returns:
        tuple -- A tuple containing the thresholded images for each channel
        in the order blue, red, green
        '''
        ch_blue_bin = channels[0] > thresholds[0]
        ch_red_bin = channels[1] > thresholds[1]
        ch_green_bin = channels[2] > thresholds[2]
        return (ch_blue_bin, ch_red_bin, ch_green_bin)

    def _calculate_edm(self, thr_chan):
        '''
        Private method to calculate the euclidean distance maps (EDMs) of each
        channel.

        Keyword arguments:
        thr_chan (tuple): Tuple containing the thresholded channels in the
        order blue, red, green

        Returns:
        tuple -- A tuple containing the calculated EDM for each channel in the
        order blue, red, green
        '''
        edt_blue = ndi.distance_transform_edt(thr_chan[0])
        edt_red = ndi.distance_transform_edt(thr_chan[1])
        edt_green = ndi.distance_transform_edt(thr_chan[2])
        return (edt_blue, edt_red, edt_green)

    def _calculate_local_maxima(self, labels, edms):
        '''
        Private method to calculate the local maxima of euclidean distance maps

        Keyword arguments:
        labels (tuple): A tuple containing the thresholded channels in the
        order blue, red, green
        edms (tuple): A tuple containing the euclidean distance maps of each
        channel in the order blue, red, green

        Returns:
        tuple -- A tuple containing arrays of the calculated local maxima of
        each channel in the order blue, red, green
        '''
        edt_blue_max = peak_local_max(edms[0], labels=labels[0], indices=False,
                                      footprint=np.ones((91, 91)))
        edt_red_max = peak_local_max(edms[1], labels=labels[1], indices=False,
                                     footprint=np.ones((9, 9)))
        edt_gre_max = peak_local_max(edms[2], labels=labels[2], indices=False,
                                     footprint=np.ones((9, 9)))
        return (edt_blue_max, edt_red_max, edt_gre_max)

    def _perform_labelling(self, loc_max):
        '''
        Private method to perform labeling of each unique region

        Keyword arguments:
        loc_max (tuple): A tuple containing the calculated local maxima of
        each channel in the order blue, red, green

        Returns:
        tuple -- A tuple containing the labeled local maxima of each channel
        in the order blue, red, green
        '''
        markers_blue = ndi.label(loc_max[0])[0]
        markers_red = ndi.label(loc_max[1])[0]
        markers_green = ndi.label(loc_max[2])[0]
        return (markers_blue, markers_red, markers_green)

    def perform_watershed(self, edms, markers, thr_chan):
        '''
        Private method to perform watershed segmentation for each channel

        Keyword arguments:
        edms (tuple): A tuple containing the calculated euclidean distance
        maps for each channel in the order blue, red, green
        markers (tuple): A tuple containing the markers used to identify each
        individual region of each channel in the order blue, red, green
        thr_chan: A tuple containing the thresholded channels in the order
        blue, red, green

        Returns:
        tuple -- A tuple of the segmented channels in the order blue, red,
        green
        '''
        ws_blue = watershed(-edms[0], markers[0], mask=thr_chan[0])
        ws_red = watershed(-edms[1], markers[1], mask=thr_chan[1])
        ws_green = watershed(-edms[2], markers[2], mask=thr_chan[2])
        return (ws_blue, ws_red, ws_green)

    def _create_result_image(self, which):
        '''
        Method to save the result image as file.

        Keyword arguments:
        which (str) -- The path to the original image
        '''
        pass

    def _estimate_image_quality(self, channels):
        '''
        Private method to estimate the image quality on basis of blurriness
        and overexposure

        Keyword arguments:
        channels (tuple): A tuple containing the individual channels of the
        image in the order blue, red, green

        Returns:
        2D tuple -- A 2D tuple containing the estimated image quality of each
        channel in the order(blurriness(b,r,g), overexposure(b,r,g))
        '''
        return (self._calculate_blurrines(channels),
                self._determine_overexposure(channels))

    def _calculate_blurrines(self, channels):
        '''
        Private method to calculate the blurriness factor of each image channel

        Keyword arguments:
        channels (tuple): A tuple containing the individual channels of the
        image in the order blue, red, green

        Returns:
        tuple -- A tuple containing the estimated blurriness of each channel in
        the order blue, red, green
        '''
        blue_var = laplace(channels[0], 3).var()
        red_var = laplace(channels[1], 3).var()
        green_var = laplace(channels[2], 3).var()
        return (blue_var, red_var, green_var)
    
    def _determine_overexposure(self, channels):
        '''
        Private method to determine if an image is overexposed

        Keyword arguments:
        channels (tuple): A tuple containing the individual channels of the
        image in the order blue, red, green

        Returns:
        tuple -- A tuple containing the estimated overexposure of each channel in
        the order blue, red, green
        '''
        pass
