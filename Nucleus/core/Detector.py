'''
Created on 15.10.2018

@author: Romano Weiss
'''
from skimage import io
from NucDetect.image import Channel
from skimage.filters import threshold_triangle, laplace
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.exposure import histogram
from scipy import ndimage as ndi
from NucDetect.image.ROI_Handler import ROI_Handler
import matplotlib.pyplot as plt
import numpy as np
import os
import hashlib
import time
from skimage.feature._canny import canny
from skimage.transform.hough_transform import hough_ellipse
from skimage.draw.draw import ellipse
from skimage.morphology.binary import binary_dilation


class Detector:
    '''
    Class to detect intranuclear proteins in provided fluorescence images
    '''

    def __init__(self):
        '''
        Constructor to initialize the detector

        Keyword parameters:
        images(dict): A dict containing the images to analyze. It must have
        following structure dict[url_to_image] = image (as 2D array)
        '''
        self.images = {}
        self.snaps = {}
        self.save_snapshots = True
        self.assessment = False  # settings["assess quality"]
        self.settings = {
            "assess_quality": False,
            "hough": False
        }
        np.set_printoptions(threshold=np.nan)

    def load_image(self, url, names=None):
        '''
        Method to add an image to the processing queue.

        Keyword arguments:
        url (str) -- The path to the image file
        names (tuple) -- Used to rename the image channels. Structure:
        (blue name, red name, green name)

        Returns:
        str -- The md5 value used as id for the image. Needed to obtain the
        results of the processing
        '''
        key = self._calculate_image_id(url)
        self.images[key] = (io.imread(url), names)
        return key

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

    def _calculate_image_id(self, url):
        '''
        Private method to calculate the md5-id of the given image

        Keyword arguments:
        url (str): The url leading to the image file

        Returns:
        str -- The md5 id of the image
        '''
        hash_md5 = hashlib.md5()
        with open(url, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def analyse_image(self, which):
        print("Analysis started")
        img = self.images[which]
        names = img[1]
        img_array = self._add_image_padding(img[0])
        start = time.time()
        channels = self._get_channels(img_array)
        if self.assessment:
            qual = self._estimate_image_quality(channels)
            print("Quality assessed")
        handler = ROI_Handler(ident=which)
        handler.set_names(names)
        if self.settings["hough"]:
            print("Hough started: " + str(time.time() - start))
            thr_chan, params = self._detect_regions(channels, accuracy=20,
                                                    threshold=250, min_size_1=50, min_size_2=2, min_size_3=2,
                                                    max_size_1=300, max_size_2=10, max_size_3=10)
            print("Labelling started: " + str(time.time() - start))
            markers = self._perform_labelling(thr_chan)
            handler.set_watersheds(markers)
        else: 
            thr_chan = self._get_thresholded_channels(channels)
            edms = self._calculate_edm(thr_chan)
            loc_max = self._calculate_local_maxima(thr_chan, edms)
            markers = self._perform_labelling(loc_max)
            watersheds = self.perform_watershed(edms, markers, thr_chan)
            handler.set_watersheds(watersheds)
        print("Handler started: " + str(time.time() - start))
        handler.analyse_image()
        # result = handler.create_result_image(self._remove_image_padding(img_array))
        result = handler.create_result_image(img_array)
        cur_snaps = {}
        cur_snaps["id"] = which
        cur_snaps["handler"] = handler
        cur_snaps["result"] = result
        cur_snaps["original"] = img_array
        if self.assessment:
            cur_snaps["quality"] = qual
        if self.save_snapshots:
            cur_snaps["channel"] = channels
            cur_snaps["binarized"] = thr_chan
            cur_snaps["watershed"] = markers
            if self.settings["hough"]:
                cur_snaps["params"] = params
            else:
                cur_snaps["edm"] = edms
                cur_snaps["max"] = loc_max
                cur_snaps["watershed"] = watersheds
            cur_snaps["time"] = "{0:.2f} sec".format(time.time() - start)
        self.snaps[which] = cur_snaps
        print("Analysis complete in " + str(cur_snaps["time"]))

    def analyse_images(self):
        '''
        Method to analyze all images in the processing queue
        '''
        if len(self.images) is 0:
            raise ValueError("No images provided!")
        for key, img in self.images.items():
            self.analyse_image(key)

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
                ax[4].set_title("Thresholding - custom")
                ax[5].imshow(vals.get("binarized")[2], cmap='gray')
                ax[5].set_title("Thresholding - custom")
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
        self.snaps.get(which)["handler"].get_data(console=True, formatted=formatted)

    def create_ouput(self, which, formatted=True):
        '''
        Method to show the result table of the processing of a specific image.
        Table will be saved as CSV file

        Keyword arguments:
        which (str): The url to the image file
        formatted(bool): Determines if the output should be formatted
        (default:True)
        '''
        self.snaps.get(which)["handler"].get_data(console=False, formatted=formatted)

    def get_output(self, which):
        return self.snaps.get(which)["handler"].get_data()

    def get_result_image_as_figure(self, which):
        '''
        Method to get the result image as figure
        :param which: The md5 hash of the image
        :return: The annotated result image as matplotlib.figure
        '''
        return self.snaps.get(which)["result"]

    def show_result_image(self, which):
        '''
        Method to show the visual representation of the foci detection

        Keyword arguments:
        which (str): The url to the image file
        '''
        plt.show(self.snaps.get(which)["result"])

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
        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"results")
        os.makedirs(pathpardir, exist_ok=True)
        pathresult = os.path.join(pathpardir,
                                  "result - " + str(which) + ".png")
        fig = self.snaps.get(which)["handler"].create_result_image(
              self.images.get(which)[0], show=False)
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        fig.savefig(pathresult, dpi=750,
                    bbox_inches="tight",
                    pad_inches=0)
        plt.close()

    def check_for_key(self, key):
        return self.snaps.__contains__(key)

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
        return ch_blue, ch_red, ch_green

    def _get_thresholded_channels(self, channels):
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
        det = []
        ch_blue_bin = channels[0] > th_blue
        truth_table = np.zeros((len(ch_blue_bin), len(ch_blue_bin[0])), dtype=bool)
        for i in range(len(ch_blue_bin)):
            for ii in range(len(ch_blue_bin[0])):
                if ch_blue_bin[i][ii] and not truth_table[i][ii]:
                    nuc = self._flood_fill((i, ii), ch_blue_bin, truth_table)
                    if nuc is not None:
                        det.append(nuc)
        ch_red_bin = self._calculate_local_region_threshold(det, channels[1])
        ch_gre_bin = self._calculate_local_region_threshold(det, channels[2])
        return ch_blue_bin, ch_red_bin, ch_gre_bin

    def _flood_fill(self, point, region_map, truth_table):
        points = []
        points.append(point)
        height = len(region_map)
        width = len(region_map[0])
        nuc = []
        while points:
            p = points.pop()
            y = p[0]
            x = p[1]
            if (x >= 0) and (x < width) and (y >= 0) and (y < height):
                if region_map[y][x] and not truth_table[y][x]:
                    truth_table[y][x] = True
                    nuc.append(p)
                    points.append((y+1, x))
                    points.append((y, x+1))
                    points.append((y, x-1))
                    points.append((y-1, x))
        return nuc if len(nuc) > 0 else None

    def _calculate_local_region_threshold(self, nuclei, channel,
                                          ign=0.0005, percent=0.2, minimum=10):
        chan = np.zeros(channel.shape, dtype=bool)
        for nuc in nuclei:
            thresh = []
            for p in nuc:
                thresh.append(channel[p[0]][p[1]])
            if len(thresh) != 0:
                hist = np.histogram(thresh, bins=255)
                ignore = ign * len(thresh) + 1
                max_val = 255
                max_pix = 0
                for i in reversed(hist[0]):
                    max_pix += i
                    if max_pix >= ignore:
                        break
                    else:
                        max_val -= 1
                local_threshold = max_val * percent if max_val * percent > minimum else minimum
                for p in nuc:
                    chan[p[0]][p[1]] = channel[p[0]][p[1]] > local_threshold
        return chan
    
    def _detect_regions(self, channels, accuracy, threshold, min_size_1, min_size_2, min_size_3,
                        max_size_1, max_size_2, max_size_3, max_number=200):
        '''
        Private method to detect unique regions inside the channels of the image.
        
        Keyword arguments:
        channels: Tuple containing all channels of the image
        accuracy: Bin size on the minor axis used in the accumulator.
        threshold: Accumulator threshold value.
        min_size: Minimal major axis length.
        max_size: Maximal major axis length.
        max_number: The max number of recognized ellipses
        
        Returns:
        tuple -- A tuple containing the calculated EDM for each channel in the
        order blue, red, green
        '''
        nuclei = self._detect_ellipses(channels[0], accuracy, threshold, min_size_1, max_size_1, max_number)
        red_foci = self._detect_ellipses(channels[1], accuracy, threshold, min_size_2, max_size_2, max_number)
        green_foci = self._detect_ellipses(channels[2], accuracy, threshold, min_size_3, max_size_3, max_number)
        return (nuclei[0], red_foci[0], green_foci[0]), (nuclei[1], red_foci[1], green_foci[1])
    
    def _detect_ellipses(self, channel, accuracy, threshold, min_size, max_size, max_number=200):
        '''
        Private method to detect ellipses via Hough Transform. Is used to detect nuclei and foci.
        
        Keyword arguments:
        channel: The respective channel of the image
        accuracy: Bin size on the minor axis used in the accumulator.
        threshold: Accumulator threshold value.
        min_size: Minimal major axis length.
        max_size: Maximal major axis length.
        max_number: The max number of recognized ellipses
        
        Returns:
        tuple -- A tuple containing the thresholded channel and a list with the parameters of all detected ellipses
        '''
        print("Ellipse detection started")
        print("Edge detection started")
        edges = self._detect_edges(self._add_image_padding(channel))
        # Fill holes to reduce number of edges --> time saving
        edges = binary_dilation(edges)
        edges = binary_dilation(edges)
        filled = ndi.binary_fill_holes(edges)
        filled = self._detect_edges(filled)
        print("Edge detection finished")
        plt.imshow(filled , cmap='gray')
        plt.show()
        print("Hough transform started")
        hough = hough_ellipse(filled, accuracy, threshold, min_size, max_size)
        print("Hough transform finished")
        hough.sort(order="accumulator")
        params = []
        # Detect max_number ellipses in the image
        for y in range(min(max_number, len(hough))):
            x = hough[-y]
            # yc, xc, a, b, orientation
            params.append(int(round(x)) for x in hough[1:6])
        bina = np.zeros(shape=channel)
        # Create binarized image from parameters
        for x in params:
            cy, cx = ellipse(x[0], x[1], x[2], x[3], x[4]) 
            bina[cy, cx] = 1
        return bina, params
    
    def _add_image_padding(self, img, padding=10):
        '''
        Private method to add padding to an image
        
        Keyword arguments:
        padding: The amount of padding added to the image
        
        Returns:
        The padded image
        '''
        pad = np.empty((len(img)+2*padding, len(img[0])+2*padding, 3), dtype=np.uint8)
        for y in range(len(img)):
            for x in range(len(img[0])):
                pad[y + padding, x + padding] = img[y, x]
        return pad

    def _remove_image_padding(self, img, padding=10):
        '''
        Private method to remove padding from an image
        :param img: The image to remove the padding from
        :param padding: The amount of padding to remove (default: 5)
        :return: The image without the padding
        '''
        unpad = np.empty((len(img)-2*padding, len(img[0])-2*padding, 3), dtype=np.uint8)
        for y in range(len(img)-padding):
            for x in range(len(img[0])-padding):
                print(img[y+padding, x+padding])
                unpad[y, x] = img[y+padding, x+padding]
        return unpad

    def _detect_edges(self, channel, sigma=2, low_threshold=0.55, high_threshold=0.8):
        '''
        Privat method to detect the edges of the given channel via the canny operator.
        
        Keyword arguments:
        channel: The respective color channel of the image
        sigma: The standard deviation of the 
        low_threshold: Lower bound for hysteresis thresholding (linking edges).
        high_threshold:Upper bound for hysteresis thresholding (linking edges).
        
        Returns:
        array -- The edge map of the given channel
        '''
        return canny(channel, sigma, low_threshold, high_threshold)

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
        return edt_blue, edt_red, edt_green

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
                                     footprint=np.ones((31, 31)))
        edt_gre_max = peak_local_max(edms[2], labels=labels[2], indices=False,
                                     footprint=np.ones((31, 31)))
        return edt_blue_max, edt_red_max, edt_gre_max

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
        return markers_blue, markers_red, markers_green

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
        return ws_blue, ws_red, ws_green

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
        return (self._calculate_blurriness(channels),
                self._determine_overexposure(channels))

    def _calculate_blurriness(self, channels):
        '''
        Private method to calculate the blurriness factor of each image channel

        Keyword arguments:
        channels (tuple): A tuple containing the individual channels of the
        image in the order blue, red, green

        Returns:
        tuple -- A tuple containing the estimated blurriness of each channel in
        the order blue, red, green
        '''
        blue_chan = []
        red_chan = []
        green_chan = []
        blu_th = Channel.get_minmax(channels[0], channel_only=True)[0] + 1/5 * (
            Channel.get_dynamic_range(channels[0],
                                      channel_only=True, in_percent=False))
        red_th = Channel.get_minmax(channels[1], channel_only=True)[0] + 1/5 * (
            Channel.get_dynamic_range(channels[1],
                                      channel_only=True, in_percent=False))
        gre_th = Channel.get_minmax(channels[2], channel_only=True)[0] + 1/5 * (
            Channel.get_dynamic_range(channels[2],
                                      channel_only=True, in_percent=False))
        lapl_blue = laplace(channels[0], 3)
        lapl_red = laplace(channels[1], 3)
        lapl_green = laplace(channels[2], 3)
        height = len(channels[0])
        width = len(channels[0][0])
        for y in range(height):
            for x in range(width):
                if channels[0][y][x] > blu_th:
                    blue_chan.append(lapl_blue[y][x])
                if channels[1][y][x] > red_th:
                    red_chan.append(lapl_red[y][x])
                if channels[2][y][x] > gre_th:
                    green_chan.append(lapl_green[y][x])
        blue_var = np.var(blue_chan, ddof=1) if len(blue_chan) > 100 else -1
        red_var = np.var(red_chan, ddof=1) if len(red_chan) > 100 else -1
        green_var = np.var(green_chan, ddof=1) if len(green_chan) > 100 else -1
        return blue_var, red_var, green_var

    def _determine_overexposure(self, channels):
        '''
        Private method to determine if an image is overexposed

        Keyword arguments:
        channels (tuple): A tuple containing the individual channels of the
        image in the order blue, red, green

        Returns:
        tuple -- A tuple containing the estimated overexposure of each channel
        in the order blue, red, green
        '''
        blue_hist = histogram(channels[0])
        red_hist = histogram(channels[1])
        green_hist = histogram(channels[2])
        pass
