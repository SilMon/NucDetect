import time

import tensorflow as tf
from numba import jit
from skimage import io
from tensorflow.keras import models
from typing import List, Tuple, Union
import numpy as np
from pathlib import Path
from skimage.segmentation import flood_fill


class FCN:
    FOCI = 1
    NUCLEI = 2
    script_dir = Path().resolve().parent / "fcn" / "model"

    def __init__(self, limit_gpu_growth=True):
        """
        Constructor

        :param limit_gpu_growth: Should the use of GPU-RAM be limited?
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and limit_gpu_growth:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        m = self.load_model()
        self.nuc_model: models.Model = m[0]
        self.foc_model: models.Model = m[1]

    def show_model_summary(self, model: int = NUCLEI) -> None:
        """
        Method to print the summary of a model

        :param model: The model to show the summary of
        :return: None
        """
        if model == self.FOCI:
            self.foc_model.summary()
        else:
            self.nuc_model.summary()

    def predict_image(self, path: Union[str, np.ndarray],
                      model: int,
                      channels: List[int],
                      threshold: float = 0.35,
                      logging=True) -> List[np.ndarray]:
        """
        Method to create the prediction mask of an image, specified by path

        :param path: The path pointing to the image file or the loaded image as ndarray
        :param model: The model to use for the prediction
        :param channels: A list containing the indices of channels to analyse
        :param threshold: The minimal certainty of the prediction
        :param logging: Enables logging
        :return: The prediction mask
        """
        # Load the image
        if path is isinstance(path, np.ndarray):
            img = path
        else:
            img = io.imread(path)
        start = time.time()
        # Split the image into tiles
        tiles = self.split_image(img, channels)
        # Create predictions for all tiles
        pred_tiles = []
        threshs = []
        model = self.nuc_model if model == self.NUCLEI else self.foc_model
        for i in range(len(channels)):
            pred_tiles.append(FCN.predict_tiles(tiles[i], model))
        FCN.log(f"Prediction finished: {time.time() - start} secs", logging)
        maps = []
        # Merge predictions for the tiles to create the prediction map
        for p in pred_tiles:
            maps.append(FCN.merge_prediction_masks(p, img.shape))
        FCN.log(f"Merging finished: {time.time() - start:.4f} secs", logging)
        # Threshold maps
        for m in maps:
            threshs.append(FCN.threshold_prediction_mask(m, threshold=threshold))
        FCN.log(f"Thresholding finished: {time.time() - start:.4f} secs", logging)
        return threshs

    @staticmethod
    def load_model() -> Tuple[models.Model]:
        """
        Method to load the needed models

        :return: A tuple of the loaded models
        """
        nuc_path: Path = FCN.script_dir / "nucleus_detector.h5"
        foc_path: Path = FCN.script_dir / "focus_detector.h5"
        nuc_model = models.load_model(nuc_path.resolve())
        foc_model = models.load_model(foc_path.resolve())
        return nuc_model, foc_model

    @staticmethod
    def split_image(img: np.ndarray,
                    channels: List[int]) -> List[List[np.ndarray]]:
        """
        Method to split the image into 256x256 tiles

        :param img: The image to split
        :param channels: The channels to extract
        :return: A list containing the tiles
        """
        # square dimension of the tiles
        ts = 256
        # Get the dimensions of the image
        height = img.shape[0]
        width = img.shape[1]
        # Check if the dims can be divided by 256 without a remainder
        hcheck = bool(height % ts)
        wcheck = bool(width % ts)
        # Get the number of tiles
        hcount = height // ts if not hcheck else height // ts + 1
        wcount = width // ts if not wcheck else width // ts + 1
        # Create the tiles
        tiles = []
        for _ in channels:
            tiles.append([])
        for y in range(hcount):
            for x in range(wcount):
                # Define the tile dimensions
                y1 = ts * y
                y2 = ts * (y + 1)
                x1 = ts * x
                x2 = ts * (x + 1)
                for channel in range(len(channels)):
                    # Extract the channel
                    c = img[..., channels[channel]]
                    extract = c[y1:y2, x1:x2]
                    tile = np.zeros(shape=(ts, ts))
                    tile[0:extract.shape[0], 0:extract.shape[1]] = extract
                    tiles[channel].append(tile)
        return tiles

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
    def merge_prediction_masks(masks: List[np.ndarray],
                               orig_shape: Tuple[int, int]) -> np.ndarray:
        """
        Method to merge created prediction masks into one large image

        :param masks: A list containing the created prediciton masks
        :param orig_shape: A tuple specifying the shape of the original image
        :return: The merged prediction mask
        """
        # square dimension of the tiles
        ts = 256
        # Get the dimensions of the image
        height = orig_shape[0]
        width = orig_shape[1]
        # Check if the dims can be divided by 256 without a remainder
        hcheck = bool(height % ts)
        wcheck = bool(width % ts)
        # Get the number of tiles
        hcount = height // ts if not hcheck else height // ts + 1
        wcount = width // ts if not wcheck else width // ts + 1
        img = np.zeros(shape=(orig_shape[0], orig_shape[1]))
        for y in range(hcount):
            for x in range(wcount):
                # Define the mask position
                y1 = ts * y
                y2 = ts * (y + 1)
                x1 = ts * x
                x2 = ts * (x + 1)
                mask = masks[y * wcount + x]
                # Cut mask to boundaries
                mask = mask[0:min(ts, height - y1), 0:min(ts, width - x1)]
                # Merge mask with image
                img[y1:y2, x1:x2] = mask[...]
        return img

    @staticmethod
    def threshold_prediction_mask(prediction_mask: np.ndarray,
                                  threshold: float = 0.98) -> np.ndarray:
        """
        Method to threshold the created prediction mask

        :param prediction_mask: The mask to threshold
        :param threshold: The threshold to apply in percent
        :param logging: Enables logging
        """
        height = prediction_mask.shape[0]
        width = prediction_mask.shape[1]
        mask = prediction_mask > threshold
        mask = mask.astype("uint8")
        # Label the individual areas of the map
        label = 2
        for y in range(height):
            for x in range(width):
                if mask[y][x] == 1:
                    FCN.flood_fill(mask, (y, x), label)
                    label += 1
        return mask

    @staticmethod
    @jit(nopython=True)
    def flood_fill(mask: np.ndarray, p: Tuple[int, int], label: int):
        """
        Implementation of iterative flood fill, 8-connected

        :param mask: The mask to fill
        :param p: The starting point
        :param label: The label to use
        :return: The filled mask
        """
        # Create the stack
        stack = [p]
        # Repeat action until stack is empty
        while stack:
            y, x = stack.pop()
            if mask[y][x] == 1:
                mask[y][x] = label
                stack.append((y, x + 1))
                stack.append((y, x - 1))
                stack.append((y + 1, x))
                stack.append((y - 1, x))
                stack.append((y + 1, x + 1))
                stack.append((y - 1, x - 1))
                stack.append((y + 1, x - 1))
                stack.append((y - 1, x + 1))

    @staticmethod
    def predict_tiles(tiles: List[np.ndarray],
                      model: tf.keras.Model) -> List[np.ndarray]:
        """
        Method to predict a list of tiles

        :param tiles: The tiles to predict
        :param model: The model to use for the prediction
        :return: Predictions for all tiles
        """
        tiles = np.asarray(tiles).astype("float32")
        tiles /= 255
        tiles = tiles.reshape(-1, 256, 256, 1)
        predictions = model.predict(tiles, use_multiprocessing=True)
        masks = []
        for prediction in predictions:
            masks.append(FCN.create_prediction_mask(prediction))
        return masks

    @staticmethod
    def log(message: str, state: bool = True):
        """
        Method to log messages to the console

        :param message: The message to log
        :param state: Enables logging
        :return: None
        """
        if state:
            print(message)
