import os
from pathlib import Path
from typing import Iterable, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import binary_erosion, binary_opening, binary_closing
from skimage.transform import resize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import models
import Paths
from DataProcessing import automatic_whitebalance
from detector_modules.AreaMapper import AreaMapper


class FCNMapper(AreaMapper):
    """
    Class to detect foci on image channels using machine learning
    """
    ___slots__ = (
        "channels",
        "settings",
        "main",
        "model",
        "fcn",
    )
    STANDARD_SETTING = {
        "fcn_certainty_nuclei": 0.95,
        "fcn_certainty_foci": 0.8
    }

    def __init__(self, channels: Iterable[np.ndarray] = None,
                 settings: Dict = None, main: int = 2):
        super().__init__(channels, settings)
        self.script_dir = Path().resolve().parent / "fcn" / "model"
        self.set_gpu_memory_growth()
        self.model = self.load_model()
        self.main = main
        self.model_type = True

    @staticmethod
    def set_gpu_memory_growth() -> None:
        """
        Method to set the gpu memory growth to dynamic
        :return: None
        """
        # Load the gpus
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    @staticmethod
    def load_model() -> models.Model:
        """
        Method to load the ML models
        :return: None
        """
        path = os.path.join(Paths.model_dir, "detector.h5")
        model = models.load_model(path)
        return model

    def get_marked_maps(self) -> List[np.ndarray]:
        """
        Method to create nucleus and focus maps for the given channels

        :return: The created maps
        """
        # Check if channels were set
        if not self.channels:
            raise ValueError("No channels were set to be analysed!")
        # Check if setting were given
        if not self.settings:
            self.settings = self.STANDARD_SETTING
        pmaps = self.map_channels()
        return self.threshold_maps(pmaps)

    def map_channels(self, adjust_white_balance: bool = True) -> List[np.ndarray]:
        """
        Method to map the given channels

        :param adjust_white_balance: Should the channels be scaled to use the full possible intensity range?
        :return: The prediction maps
        """
        prediction_maps = []
        for channel in self.channels:
            orig_shape = channel.shape
            # Resize the channel to match the training size
            channel = resize(channel, output_shape=(1024, 1024), preserve_range=True, anti_aliasing=True)
            # Split channel images into tiles
            tiles = self.extract_subimages(channel if not adjust_white_balance else automatic_whitebalance(channel),
                                           (256, 256))
            # Predict the individual tiles
            ptiles = self.predict_tiles(tiles, self.model)
            # Merge prediction maps
            pred_map = resize(self.merge_prediction_tiles(ptiles, channel.shape), output_shape=orig_shape,
                              preserve_range=True, anti_aliasing=True)
            prediction_maps.append(pred_map)
        return prediction_maps

    @staticmethod
    def extract_subimages(img: np.ndarray, subimage_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Function to extract subimages from a given image

        :param img: The image to extract the subimages from
        :param subimage_shape: The shape of each subimage
        :return: List of all extracted subimages
        """
        # Get the number of subimages for each axis
        svert, shor = FCNMapper.get_number_of_subimages_per_dimension(img.shape, subimage_shape)
        sub_images = []
        for y in range(svert):
            for x in range(shor):
                extract = img[y * subimage_shape[0]: (y + 1) * subimage_shape[0],
                              x * subimage_shape[1]: (x + 1) * subimage_shape[1]]
                tile = np.zeros(shape=subimage_shape)
                tile[0: extract.shape[0], 0: extract.shape[1]] = extract
                sub_images.append(tile)
        return sub_images

    @staticmethod
    def get_number_of_subimages_per_dimension(img_shape: Tuple[int, int],
                                              sub_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Method to get the number of sub-images per

        :param img_shape: The shape of the image
        :param sub_shape: The shape of the sub-images to extract
        :return: The vertical and horizontal sub-image count
        """
        hcheck = bool(img_shape[0] % sub_shape[0])
        wcheck = bool(img_shape[1] % sub_shape[1])
        # Get the number of tiles
        hcount = img_shape[0] // sub_shape[0] if not hcheck else img_shape[0] // sub_shape[0] + 1
        wcount = img_shape[1] // sub_shape[1] if not wcheck else img_shape[1] // sub_shape[1] + 1
        return hcount, wcount

    @staticmethod
    def predict_tiles(tiles: List[np.ndarray],
                      model: models.Model) -> List[np.ndarray]:
        """
        Method to predict a list of tiles

        :param tiles: The tiles to predict
        :param model: The model to use for the prediction
        :return: Predictions for all tiles
        """
        tiles = np.asarray(tiles).astype("float32")
        tiles /= 255
        tiles = tiles.reshape(-1, tiles.shape[1], tiles.shape[2], 1)
        predictions = model.predict(tiles)#, use_multiprocessing=True)
        masks = []
        for prediction in predictions:
            masks.append(FCNMapper.create_prediction_mask(prediction))
        return masks

    @staticmethod
    def create_prediction_mask(pred: tf.Tensor) -> np.ndarray:
        """
        Method to create the prediction mask from the prediction tensor

        :param pred: The prediction tensor
        :return: The created mask
        """
        mask = pred[:, :, 0]
        # Add a new axis to the prediction mask
        return mask

    @staticmethod
    def merge_prediction_tiles(masks: List[np.ndarray],
                               orig_shape: Tuple[int, int]) -> np.ndarray:
        """
        Method to merge created prediction masks into one large image

        :param masks: A list containing the created prediciton masks
        :param orig_shape: A tuple specifying the shape of the original image
        :return: The merged prediction mask
        """
        hcount, wcount = FCNMapper.get_number_of_subimages_per_dimension(orig_shape, masks[0].shape)
        img = np.zeros(shape=(orig_shape[0], orig_shape[1]))
        height, width = orig_shape
        ts = masks[0].shape
        for y in range(hcount):
            for x in range(wcount):
                # Define the mask position
                y1 = ts[0] * y
                y2 = ts[0] * (y + 1)
                x1 = ts[1] * x
                x2 = ts[1] * (x + 1)
                mask = masks[y * wcount + x]
                # Cut mask to boundaries
                mask = mask[0:min(ts[0], height - y1), 0:min(ts[1], width - x1)]
                # Merge mask with image
                img[y1:y2, x1:x2] = mask[...]
        return img

    def threshold_maps(self, prediction_maps: List[np.ndarray]) -> List[np.ndarray]:
        """
        Method to threshold the given prediction maps

        :param prediction_maps: The prediction maps to threshold
        :return: The thresholded prediction maps
        """
        if len(prediction_maps) > 1:
            bin_maps = []
            for pmap in prediction_maps:
                bmap = binary_fill_holes(pmap >= self.settings["fcn_certainty_foci"])
                #bmap = binary_opening(bmap, footprint=np.ones(shape=(5, 5)))
                bmap = label(bmap)[0]
                bin_maps.append(bmap)
            return bin_maps
        else:
            return label(binary_opening(prediction_maps[0] >= self.settings["fcn_certainty_nuclei"],
                                        footprint=np.ones(shape=(5, 5))))[0]


