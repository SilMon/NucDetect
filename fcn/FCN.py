import tensorflow as tf
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

    def __init__(self):
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
                      threshold: float = 0.5) -> List[np.ndarray]:
        """
        Method to create the prediction mask of an image, specified by path

        :param path: The path pointing to the image file or the loaded image as ndarray
        :param model: The model to use for the prediction
        :param channels: A list containing the indices of channels to analyse
        :param threshold: The minimal certainty of the prediction
        :return: The prediction mask
        """
        # Load the image
        if path is isinstance(path, np.ndarray):
            img = path
        else:
            img = io.imread(path)
        # Split the image into tiles
        tiles = self.split_image(img, channels)
        # Create predictions for all tiles
        pred_tiles = []
        threshs = []
        model = self.nuc_model if model == self.NUCLEI else self.foc_model
        for i in range(len(channels)):
            pred_tiles.append(FCN.predict_tiles(tiles[i], model))
        maps = []
        # Merge predictions for the tiles to create the prediction map
        for p in pred_tiles:
            maps.append(FCN.merge_prediction_masks(p, img.shape))
        # Threshold maps
        for m in maps:
            threshs.append(FCN.threshold_prediction_mask(m, threshold=threshold))
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
                    mask = flood_fill(mask, (y, x), label)
                    label += 1
        return mask

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
        predictions = model.predict(tiles)
        masks = []
        for prediction in predictions:
            mask = FCN.create_prediction_mask(prediction)
            masks.append(mask)
        return masks

