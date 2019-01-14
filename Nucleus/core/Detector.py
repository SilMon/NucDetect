"""
Created on 15.10.2018

@author: Romano Weiss
"""
import pickle
from skimage import io
from NucDetect.image import Channel
from skimage.filters import sobel, threshold_yen,threshold_isodata, laplace
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
from skimage.morphology.binary import binary_dilation, binary_opening


class Detector:
    """
    Class to detect intranuclear proteins in provided fluorescence images
    """

    def __init__(self):
        """
        Constructor to initialize the detector

        Keyword parameters:
        images(dict): A dict containing the images to analyze. It must have
        following structure dict[url_to_image] = image (as 2D array)
        """
        self.images = {}
        self.snaps = {}
        self.save_snapshots = True
        self.assessment = False  # settings["assess quality"]
        self.settings = {
            "assess_quality": False,
            "save_snaps": True,
            "oversegmentation": False,
            "padding": 10,
            "edge-ignore": True,
            "hough": False,
            "hough_acc": 20,
            "hough_thresh": 250,
            "hough_min_size1": 50,
            "hough_max_size1" : 300,
            "hough_min_size2": 2,
            "hough_max_size2": 10,
            "hough_min_size3": 2,
            "hough_man_size3": 10,
            "hough_num": 200,
            "nucleus_hor": 13,
            "nucleus_vert": 91,
            "foci_hor": 31,
            "foci_vert": 31,
            "foci_perc": 40,
            "foci_min": 10,
            "foci_ign": 0.0005,
            "res_csv": 1,
            "res_cons": 0,
            "res_form": 0,
            "res_img_ann": 1,
            "res_snap": 1
        }
        self.keys = []

    def _check_for_saved_snaps(self):
        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"results/snaps")
        if os.path.isdir(pathpardir):
            for file in os.listdir(pathpardir):
                name = file.split(".")[0]
        pass

    def load_image(self, url, names=None):
        """
        Method to add an image to the processing queue.

        Keyword arguments:
        url (str) -- The path to the image file
        names (tuple) -- Used to rename the image channels. Structure:
        (blue name, red name, green name)

        Returns:
        str -- The md5 value used as id for the image. Needed to obtain the
        results of the processing
        """
        key = self._calculate_image_id(url)
        self.images[key] = (io.imread(url), names)
        self.keys.append(key)
        return key

    def load_image_folder(self, direct):
        """
        Method to load all images of a specific directory

        Keyword arguments:
        direct (str): The path to the directory
        """
        files = os.listdir(direct)
        for file in files:
            path = os.path.join(direct, file)
            if os.path.isfile(path):
                self.load_image(path)

    def _calculate_image_id(self, url):
        """
        Private method to calculate the md5-id of the given image

        Keyword arguments:
        url (str): The url leading to the image file

        Returns:
        str -- The md5 id of the image
        """
        hash_md5 = hashlib.md5()
        with open(url, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def analyse_image(self, which, save_threshold=2):
        """
        Method to analyse a loaded image
        :param which: The md5 hash of the image
        :param save_threshold: Threshold to decide how many analysis data will be hold directly
        by the detector. Is needed to avoid stack overflow when used to analyse a lot images. Default: 2
        :return: None
        """
        print("Analysis started")
        snap = self.load_snaps(which)
        if snap is not None:
            self.snaps[which] = snap
        else:
            img = self.images[which]
            names = img[1]
            img_array = img[0]
            #img_array = self._add_image_padding(img[0], dim=3, padding=self.settings["padding"])
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
            cur_snaps = {
                "id": which,
                "handler": handler,
                "result": result,
                "original": img_array,
                "settings": self.settings,
                "categories": []
            }
            if self.settings["assess_quality"]:
                cur_snaps["quality"] = qual
            if self.settings["save_snaps"]:
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
            if len(self.snaps) == save_threshold:
                for x in range(save_threshold-1):
                    self.save_snaps(self.keys[x])
            else:
                self.snaps[which] = cur_snaps
            print("Analysis complete in " + "{0:.2f} sec".format(time.time() - start))

    def save_all_snaps(self):
        for key in self.images:
            self.save_snaps(key, clear=False)

    def save_snaps(self, key, clear=True):
        """
        Method to save all acquired data for a specific image.
        :param key: The md5 hash of the image
        :param clear: If true the current snap dict will be cleared. Needed to avoid stack overflow when anaylsing
        many images
        """
        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"results/snaps")
        os.makedirs(pathpardir, exist_ok=True)
        pathsnap = os.path.join(pathpardir,
                                  str(key) + ".snap")
        pickle.dump(self.snaps[key], open(pathsnap, "wb"))
        if clear:
            del self.snaps[key]
            del self.images[key]
            self.keys.remove(key)

    def load_snaps(self, key):
        """
        Method to load the saved snaps of an image, identified by key.
        This allows to reuse already acquired data if the settings were not changed
        :param key: The md5 hash of the image
        :return: None if the file does not exist or the settings were changed else +1 if the file could be loaded
        """

        #TODO Fehlerhaft

        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"results/snaps")
        if not os.path.isdir(pathpardir):
            os.makedirs(pathpardir, exist_ok=True)
            return None
        else:
            pathsnap = os.path.join(pathpardir,
                                    str(key) + ".snap")
            if os.path.isfile(pathsnap):
                snap = pickle.load(open(pathsnap, "rb"))
                if self.settings == snap["settings"]:
                    return snap
                else:
                    return None
            else:
                return None

    def export_snaps_as_image(self, key, which=None):
        """
        Method to export the snapshots of an image to file.
        :param key: The md5 hash of the image
        :param which: Defines which snapshots should be exported. If None, all will be exported.
        :return: None
        """
        # TODO

    def analyse_images(self):
        """
        Method to analyze all images in the processing queue
        """
        if len(self.images) is 0:
            raise ValueError("No images provided!")
        for key, img in self.images.items():
            self.analyse_image(key)

    def show_snapshot(self, which):
        """
        Method to show the saved processing snapshots of a specific image using matplotlib

        :param which The md5 hash of the image
        """
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
        """
        Method to show the result table of the processing of a specific image.
        Table will be shown in the console

        Keyword arguments:
        which (str): The url to the image file
        formatted(bool): Determines if the output should be formatted
        (default:True)
        """
        self.snaps.get(which)["handler"].get_data(console=True, formatted=formatted)

    def create_ouput(self, which, formatted=True):
        """
        Method to show the result table of the processing of a specific image.
        Table will be saved as CSV file

        Keyword arguments:
        which (str): The url to the image file
        formatted(bool): Determines if the output should be formatted
        (default:True)
        """
        self.snaps.get(which)["handler"].get_data(console=False, formatted=formatted)

    def get_output(self, which):
        return self.snaps.get(which)["handler"].get_data()

    def get_statistics(self, which):
        return self.snaps.get(which)["handler"].calculate_statistics()

    def get_result_image_as_figure(self, which):
        """
        Method to get the result image as figure
        :param which: The md5 hash of the image
        :return: The annotated result image as matplotlib.figure
        """
        return self.snaps.get(which)["result"]

    def show_result_image(self, which):
        """
        Method to show the visual representation of the foci detection

        Keyword arguments:
        which (str): The url to the image file
        """
        plt.show(self.snaps.get(which)["result"])

    def get_snapshot(self, which):
        """
        Method to obtain the saved processing data for a specific image

        Keyword arguments:
        which (str): The url to the image file

        Returns:
        dict -- A dict which contains the saved processing snapshots. If no
        snapshots are saved, it will only contain the result image
        """
        return self.snaps.get(which)

    def save_result_image(self, which):
        """
        Method to save the processing image to file

        Keyword arguments:
        which (str): The url to the image file
        """
        pardir = os.getcwd()
        pathpardir = os.path.join(os.path.dirname(pardir),
                                  r"results/images")
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

    def _get_channels(self, img):
        """
        Private Method to extract the channels of a given image

        Keyword arguments:
        img (2D array): The image to extract the channels from

        Returns:
        tuple  -- A tuple containing the three channels in the order
        blue, red, green
        """
        ch_blue = Channel.extract_channel(img, Channel.BLUE)
        ch_red = Channel.extract_channel(img, Channel.RED)
        ch_green = Channel.extract_channel(img, Channel.GREEN)
        return ch_blue, ch_red, ch_green

    def _get_thresholded_channels(self, channels):
        """
        Private Method to calculate the thresholds for each channel

        Keyword arguments:
        channels (tuple): A tuple containing the three channels in the
        order blue, red, green

        Returns:
        tuple  -- A tuple containing the thresholds for each channel
        in the order blue, red, green
        """
        # TODO
        edges_blue = (sobel(channels[0]) * 255).astype("uint8")
        det = []
        ch_blue_bin = edges_blue > 5
        ch_blue_bin = ndi.binary_fill_holes(ch_blue_bin)
        ch_blue_bin = binary_opening(ch_blue_bin)
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
        points = [
            point
        ]
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
                                          ign=0.0005, percent=0.8, minimum=40):
        chan = np.zeros(shape=channel.shape)
        for nuc in nuclei:
            thresh = []
            for p in nuc:
                if channel[p[0]][p[1]] > minimum:
                    thresh.append((p, channel[p[0]][p[1]]))
            if thresh:
                thresh_np, offset = self._create_numpy_from_point_list(thresh)
                # edges = (sobel(thresh_np) * 255).astype("uint8")
                edges = canny(thresh_np)
                if np.max(edges) > 0:
                    #th = threshold_yen(edges)
                    #chan_bin = thresh_np > th
                    chan_fill = ndi.binary_fill_holes(edges)
                    chan_open = ndi.binary_opening(chan_fill)
                    self._imprint_data_into_channel(chan, chan_open, offset)
                '''
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
                '''
        return chan

    def _check_threshold_for_values(self, list):
        for row in list:
            for int in row:
                if int > 0:
                    return True
        return False

    def _imprint_data_into_channel(self, channel, data, offset):
        for i in range(len(data)):
            for ii in range(len(data[0])):
                if data[i][ii] is not 0:
                    channel[i + offset[0]][ii+offset[1]] = data[i][ii]

    def _create_numpy_from_point_list(self, list):
        # TODO
        min_y = 0xffffffff
        max_y = 0
        min_x = 0xffffffff
        max_x = 0
        for point in list:
            if point[0][1] > max_x:
                max_x = point[0][1]
            if point[0][1] < min_x:
                min_x = point[0][1]
            if point[0][0] > max_y:
                max_y = point[0][0]
            if point[0][0] < min_y:
                min_y = point[0][0]
        y_dist = max_y - min_y + 1
        x_dist = max_x - min_x + 1
        numpy = np.zeros((y_dist, x_dist), dtype=np.uint8)
        for p in list:
            numpy[p[0][0]-min_y, p[0][1]-min_x] = p[1]
        return numpy, (min_y, min_x)

    def _detect_regions(self, channels, accuracy, threshold, min_size_1, min_size_2, min_size_3,
                        max_size_1, max_size_2, max_size_3, max_number=200):
        """
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
        """
        nuclei = self._detect_ellipses(channels[0], accuracy, threshold, min_size_1, max_size_1, max_number)
        red_foci = self._detect_ellipses(channels[1], accuracy, threshold, min_size_2, max_size_2, max_number)
        green_foci = self._detect_ellipses(channels[2], accuracy, threshold, min_size_3, max_size_3, max_number)
        return (nuclei[0], red_foci[0], green_foci[0]), (nuclei[1], red_foci[1], green_foci[1])
    
    def _detect_ellipses(self, channel, accuracy, threshold, min_size, max_size, max_number=200):
        """
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
        """
        start = time.time()
        print("Ellipse detection started")
        print("Edge detection started")
        edges = self._detect_edges(self._add_image_padding(channel))
        # Fill holes to reduce number of edges --> time saving
        edges = binary_dilation(edges)
        edges = binary_dilation(edges)
        filled = ndi.binary_fill_holes(edges)
        filled = self._detect_edges(filled)
        print("Edge detection finished")
        plt.imshow(filled, cmap='gray')
        plt.show()
        print("Hough transform started")
        hough = hough_ellipse(filled, accuracy, threshold, min_size, max_size)
        print("Hough transform finished")
        hough.sort(order="accumulator")
        print(time.time() - start)
        params = []
        # Detect max_number ellipses in the image
        for y in range(min(max_number, len(hough))):
            best = hough[-y]
            # yc, xc, a, b, orientation
            yc, xc, a, b = [int(round(x)) for x in best[1:5]]
#            yc, xc, a, b = (int(round(x)) for x in best[1:5])
            orientation = best[5]
            params.append((yc, xc, a, b, orientation))
        bina = np.zeros(shape=channel.shape)
        # Create binarized image from parameters
        for x in params:
            cy, cx = ellipse(x[0], x[1], x[2], x[3], x[4]) 
            bina[cy, cx] = 1
        return bina, params
    
    def _add_image_padding(self, img, dim=1, padding=5):
        """
        Private method to add padding to an image

        Keyword arguments:
        padding: The amount of padding added to the image

        Returns:
        The padded image
        """
        if dim is not 1:
            pad = np.empty((len(img)+2*padding, len(img[0])+2*padding, dim), dtype=np.uint8)
        else:
            pad = np.empty((len(img) + 2 * padding, len(img[0]) + 2 * padding), dtype=np.uint8)
        for y in range(len(img)):
            for x in range(len(img[0])):
                pad[y + padding, x + padding] = img[y, x]
        return pad

    def _remove_image_padding(self, img, dim=1, padding=5):
        """
        Private method to remove padding from an image
        :param img: The image to remove the padding from
        :param padding: The amount of padding to remove (default: 5)
        :return: The image without the padding
        """
        '''
        if dim is not 1:
            unpad = np.empty((len(img)- 2 * padding,
                              len(img[0])- 2 * padding, dim), dtype=np.uint8)
        else:
            unpad = np.empty((len(img) - 2 * padding,
                              len(img[0]) - 2 * padding), dtype=np.uint8)
        for y in range(len(img)-padding):
            for x in range(len(img[0])-padding):
                print(img[y+padding, x+padding])
                unpad[y, x] = img[y+padding-1, x+padding-1]
        return unpad
        '''
        return img

    def categorize_image(self, key, categories):
        if categories and categories is not self.snaps[key]["categories"]:
            self.snaps[key]["categories"].clear()
            for cat in categories:
                self.snaps[key]["categories"].append(cat)

    def get_categories(self, key):
        try:
            cat = self.snaps[key]["categories"]
            return cat
        except Exception:
            return ""

    def _detect_edges(self, channel, sigma=2, low_threshold=0.55, high_threshold=0.8):
        """
        Privat method to detect the edges of the given channel via the canny operator.

        Keyword arguments:
        channel: The respective color channel of the image
        sigma: The standard deviation of the
        low_threshold: Lower bound for hysteresis thresholding (linking edges).
        high_threshold:Upper bound for hysteresis thresholding (linking edges).

        Returns:
        array -- The edge map of the given channel
        """
        return canny(channel, sigma, low_threshold, high_threshold)

    def _calculate_edm(self, thr_chan):
        """
        Private method to calculate the euclidean distance maps (EDMs) of each
        channel.

        Keyword arguments:
        thr_chan (tuple): Tuple containing the thresholded channels in the
        order blue, red, green

        Returns:
        tuple -- A tuple containing the calculated EDM for each channel in the
        order blue, red, green
        """
        edt_blue = ndi.distance_transform_edt(thr_chan[0])
        edt_red = ndi.distance_transform_edt(thr_chan[1])
        edt_green = ndi.distance_transform_edt(thr_chan[2])
        return edt_blue, edt_red, edt_green

    def _calculate_local_maxima(self, labels, edms):
        """
        Private method to calculate the local maxima of euclidean distance maps

        Keyword arguments:
        labels (tuple): A tuple containing the thresholded channels in the
        order blue, red, green
        edms (tuple): A tuple containing the euclidean distance maps of each
        channel in the order blue, red, green

        Returns:
        tuple -- A tuple containing arrays of the calculated local maxima of
        each channel in the order blue, red, green
        """
        edt_blue_max = peak_local_max(edms[0], labels=labels[0], indices=False,
                                      footprint=np.ones((91, 91)))
        edt_red_max = peak_local_max(edms[1], labels=labels[1], indices=False,
                                     footprint=np.ones((3, 3)))
        edt_gre_max = peak_local_max(edms[2], labels=labels[2], indices=False,
                                     footprint=np.ones((3, 3)))
        return edt_blue_max, edt_red_max, edt_gre_max

    def _perform_labelling(self, loc_max):
        """
        Private method to perform labeling of each unique region

        Keyword arguments:
        loc_max (tuple): A tuple containing the calculated local maxima of
        each channel in the order blue, red, green

        Returns:
        tuple -- A tuple containing the labeled local maxima of each channel
        in the order blue, red, green
        """
        markers_blue = ndi.label(loc_max[0])[0]
        markers_red = ndi.label(loc_max[1])[0]
        markers_green = ndi.label(loc_max[2])[0]
        return markers_blue, markers_red, markers_green

    def perform_watershed(self, edms, markers, thr_chan):
        """
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
        """
        ws_blue = self._remove_image_padding(watershed(-edms[0], markers[0], mask=thr_chan[0]),
                                             padding=self.settings["padding"])
        ws_red = self._remove_image_padding(watershed(-edms[1], markers[1], mask=thr_chan[1]),
                                            padding=self.settings["padding"])
        ws_green = self._remove_image_padding(watershed(-edms[2], markers[2], mask=thr_chan[2]),
                                              padding=self.settings["padding"])
        return ws_blue, ws_red, ws_green

    def _estimate_image_quality(self, channels):
        """
        Private method to estimate the image quality on basis of blurriness
        and overexposure

        Keyword arguments:
        channels (tuple): A tuple containing the individual channels of the
        image in the order blue, red, green

        Returns:
        2D tuple -- A 2D tuple containing the estimated image quality of each
        channel in the order(blurriness(b,r,g), overexposure(b,r,g))
        """
        return (self._calculate_blurriness(channels),
                self._determine_overexposure(channels))

    def _calculate_blurriness(self, channels):
        """
        Private method to calculate the blurriness factor of each image channel

        Keyword arguments:
        channels (tuple): A tuple containing the individual channels of the
        image in the order blue, red, green

        Returns:
        tuple -- A tuple containing the estimated blurriness of each channel in
        the order blue, red, green
        """
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
        """
        Private method to determine if an image is overexposed

        Keyword arguments:
        channels (tuple): A tuple containing the individual channels of the
        image in the order blue, red, green

        Returns:
        tuple -- A tuple containing the estimated overexposure of each channel
        in the order blue, red, green
        """
        blue_hist = histogram(channels[0])
        red_hist = histogram(channels[1])
        green_hist = histogram(channels[2])
        pass
