import os
from pathlib import Path
from typing import Iterable, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.signal.windows import hann
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu, threshold_minimum, threshold_local
from skimage.filters.rank import maximum
from skimage.segmentation import watershed
from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import opening
from skimage.transform import resize
from skimage.util import view_as_windows

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import models
import gui.Paths as gpaths
from core.detector_modules.AreaMapper import AreaMapper


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
        path = os.path.join(gpaths.model_dir, "detector.keras")
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

    def map_channels(self) -> List[np.ndarray]:
        """
        Method to map the given channels

        :return: The prediction maps
        """
        prediction_maps = []
        for channel in self.channels:
            orig_shape = channel.shape
            orig_dtype = channel.dtype
            # Resize the channel to match the training size
            # TODO resizen von feature größe abhängig machen, tilen übernimmt den Rest
            channel = resize(channel, output_shape=(1024, 1024),
                             preserve_range=True, anti_aliasing=True).astype(orig_dtype)
            # Split channel images into tiles
            tiles = self.extract_subimages(channel,(256, 256))
            # Predict the individual tiles
            ptiles = self.predict_tiles(tiles, self.model)
            # Merge prediction maps and bring it back to the original size
            pred_map = resize(self.merge_prediction_tiles(ptiles,
                                                          orig_shape,
                                                          orig_dtype=orig_dtype),
                              output_shape=orig_shape,
                              preserve_range=True, anti_aliasing=True)
            prediction_maps.append(pred_map)
        return prediction_maps

    @staticmethod
    def extract_subimages(img: np.ndarray,
                          subimage_shape: Tuple[int, int],
                          overlap: float = 0.5) -> List[np.ndarray]:
        """
        Function to extract subimages from a given image

        :param img: The image to extract the subimages from
        :param subimage_shape: The shape of each subimage
        :param overlap: The overlap between two subimages
        :return: List of all extracted subimages
        """
        tile_height, tile_width = subimage_shape
        # Defines the stride for each axis
        step_height = step_width = int(tile_height * (1 - overlap))
        # Create tiles and show them
        return view_as_windows(img,
                               (tile_height, tile_width),
                               step=(step_height, step_width))


    @staticmethod
    def predict_tiles(tiles: List[np.ndarray],
                      model: models.Model) -> List[np.ndarray]:
        """
        Method to predict a list of tiles

        :param tiles: The tiles to predict
        :param model: The model to use for the prediction
        :return: Predictions for all tiles
        """
        orig_max = np.iinfo(tiles[0].dtype).max
        tiles = np.asarray(tiles).astype("float32")
        tiles /= orig_max
        tiles = tiles.reshape(-1, 256, 256, 1)
        return [pred[:, :, 0] for pred in model.predict(tiles)]

    @staticmethod
    def merge_prediction_tiles(masks: List[np.ndarray],
                               orig_shape: Tuple[int, int],
                               overlap: float = 0.5,
                               orig_dtype = np.uint8) -> np.ndarray:
        """
        Method to merge created prediction masks into one large image

        :param masks: A list containing the created prediciton masks
        :param overlap: The overlap between prediction masks
        :param orig_shape: A tuple specifying the shape of the original image
        :param orig_dtype: The dtype of the original image
        :return: The merged prediction mask
        """
        # TODO Overlap einstellbar machen
        # Create an accumulator map as well as a weights map
        accum = np.zeros(orig_shape, np.float32)
        weights = np.zeros(orig_shape, np.float32)
        tile_height, tile_width = masks[0].shape[0], masks[0].shape[1]
        step_height = step_width = int(tile_height * (1 - overlap))
        n_tiles_vert = int(((orig_shape[0] - tile_height) / step_height)) + 1
        n_tiles_hor = int(((orig_shape[1] - tile_width) / step_width)) + 1
        # Create the 1D weighting function
        weight1d = hann(masks[0].shape[0], sym=False)
        # Create the 2D weighting array
        weight2d = np.outer(weight1d, weight1d)
        for y in range(n_tiles_vert):
            for x in range(n_tiles_hor):
                accum[y * step_height: y * step_height + tile_height,
                x * step_width: x * step_width + tile_width] += masks[y * n_tiles_hor + x] * weight2d
                weights[y * step_height: y * step_height + tile_height,
                x * step_width: x * step_width + tile_width] += weight2d
        return (np.divide(accum, weights, out=np.zeros_like(accum),
                          where=weights!=0) * np.iinfo(orig_dtype).max).astype(orig_dtype)

    @staticmethod
    def threshold_maps(prediction_maps: List[np.ndarray]) -> List[np.ndarray]:
        """
        Method to threshold the given prediction maps

        :param prediction_maps: The prediction maps to threshold
        :return: The thresholded prediction maps
        """
        bin_maps = []
        for inference in prediction_maps:
            # Threshold the image
            # TODO threshold als einstellung ermöglichen
            threshold = threshold_otsu(inference)
            binary = opening(inference >= threshold)
            # Extract the individual areas using watershed segmentation
            seed_points = peak_local_max(inference,
                                         threshold_abs=threshold,
                                         footprint=np.ones((3, 3)))
            mask = np.zeros(inference.shape, dtype=bool)
            mask[tuple(seed_points.T)] = True
            labeled = label(mask)[0]
            bin_maps.append(watershed(image=-inference,
                                      markers=labeled,
                                      mask=binary,
                                      watershed_line=False))
        return bin_maps

