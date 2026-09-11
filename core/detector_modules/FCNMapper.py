import os
import threading
from typing import Iterable, Dict, List, Tuple
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
from core.logging_config import get_logger

LOGGER = get_logger(__name__)

# The process-wide Keras model, loaded at most once. See FCNMapper.load_model for why this is a
# module global rather than a class or instance attribute -- in short, so that it is per-process and
# so that it never enters the pickle batch analysis makes of Detector.
_MODEL_CACHE = None
_MODEL_CACHE_LOCK = threading.Lock()
from core.detector_modules.AreaMapper import AreaMapper
from core.detector_modules.ImageLoader import dtype_max


class FCNMapper(AreaMapper):
    """
    Class to detect foci on image channels using machine learning
    """
    # No __slots__ -- see the note on AreaMapper. This class was the reason: its declaration named
    # neither `script_dir` nor `model_type`, both of which __init__ assigned, so the list could
    # never have been activated as written. Both attributes have since been deleted (2026-08-08 and
    # 2026-08-15), so the original obstacle is gone; whether to declare __slots__ here is now the
    # open question on AreaMapper rather than a defect in this class.
    STANDARD_SETTING = {
        "fcn_certainty_nuclei": 0.95,
        "fcn_certainty_foci": 0.8
    }
    # The resolution every channel is resized to before inference. This is SCALE NORMALISATION, not
    # a performance or memory compromise: the network was trained on 1024x1024 material, so it
    # learned nuclei and foci at that scale. Feeding a 2048x2048 image at native resolution would
    # present every feature at twice the size the network recognises, which is a detection-accuracy
    # problem. The prediction is resized back to the image's own shape afterwards.
    #
    # Note this normalises by PIXEL DIMENSIONS, which is only equivalent to normalising by feature
    # scale while every image covers the same field of view at the same magnification. Making it
    # depend on dots_per_micron instead is what the TODO in map_channels asks for; that setting is
    # already carried in analysis_settings and is currently read but unused elsewhere.
    TRAINING_SHAPE = (1024, 1024)
    # The input shape of the saved detector.keras. Tiles are cut to this size and predicted in one
    # batch; a fully convolutional network is theoretically size-agnostic, but this model object is
    # built for a fixed input, so feeding a whole image would mean rebuilding it per image shape.
    TILE_SHAPE = (256, 256)

    def __init__(self, channels: Iterable[np.ndarray] = None, settings: Dict = None):
        # There was a `self.script_dir = Path().resolve().parent / "fcn" / "model"` here. It was
        # never read, and it was wrong: it derived the model directory from the CURRENT WORKING
        # DIRECTORY, which the application mutates at import time, so it pointed somewhere else
        # depending on how the process was started. load_model uses gui.Paths.model_dir, which
        # names the same directory but derives it from the package location. Use that.
        #
        # A `main: int = 2` parameter, `self.main` and `self.model_type` were removed on 2026-08-15.
        # All three were leftovers of the dropped nucleus-detection model: 2 was FCN.NUCLEI, and the
        # deleted FCN class used it to choose between two loaded models. This class loads ONE --
        # detector.keras, the focus detector -- so the parameter selected nothing, and neither
        # attribute was ever read. The signature was actively misleading: a caller could reasonably
        # believe FCNMapper(..., main=1) switched it to focus detection and that it was configured
        # for nuclei, and both readings were wrong.
        super().__init__(channels, settings)
        self.set_gpu_memory_growth()
        self.model = self.load_model()

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
                LOGGER.warning("Could not set dynamic GPU memory growth: %s", e)

    @staticmethod
    def load_model(use_cache: bool = True) -> models.Model:
        """
        Method to load the ML model, reusing the one already loaded in this process

        **The cache is the point of this method, not an optimisation detail.**
        ``Detector.ml_roi_extraction`` constructs a new ``FCNMapper`` for every image, and every
        ``FCNMapper`` used to load its own model. Reading the file again is the small half of that
        cost -- measured at 0.359 s. The large half is that **each freshly loaded model brings its
        own ``predict_function``, so TensorFlow re-traces it**, and inference itself then runs far
        slower: measured 6.29 s with one reused model against 11.10 s with a fresh model per image,
        **+76 %**, with TensorFlow emitting "triggered tf.function retracing" exactly where
        predicted. That is roughly 30 % of the whole u-net stage. Full measurement in the
        2026-08-21 ML pipeline timings.

        **Module level, not instance level, and that matters for two reasons.** A module global is
        per-process, so a ``ProcessPoolExecutor`` worker gets its own and never shares a TensorFlow
        object across processes; and it is not part of any instance's ``__dict__``, so it cannot be
        dragged into the pickle that batch analysis makes of ``Detector`` once per image. The
        existing invariant that ``Detector.release_transient_state`` drops ``self.fcnmapper`` before
        that pickle happens is untouched -- this change only means the next mapper is cheap to
        build rather than expensive.

        **Memory trade, stated plainly:** the model (~11 MiB resident) now lives for the lifetime of
        the process rather than being rebuilt per image. That is the intended exchange, and in a
        batch it replaces N loads with one per worker.

        :param use_cache: False forces a fresh load, bypassing and NOT populating the cache. Exists
            for measurement -- it is what lets the retracing cost above be reproduced on demand --
            and has no use in the application
        :return: The Keras model
        """
        global _MODEL_CACHE
        if not use_cache:
            return models.load_model(os.path.join(gpaths.model_dir, "detector.keras"))
        # Double-checked under the lock. The application analyses one image at a time per process,
        # so contention is not expected, but a Qt worker thread and the main thread can both reach
        # this and a race would build two models and keep whichever was assigned last -- wasting a
        # load and, worse, handing different callers different objects, which is the retracing
        # problem again in a subtler form
        if _MODEL_CACHE is None:
            with _MODEL_CACHE_LOCK:
                if _MODEL_CACHE is None:
                    path = os.path.join(gpaths.model_dir, "detector.keras")
                    LOGGER.debug("Loading detection model from %s", path)
                    _MODEL_CACHE = models.load_model(path)
        return _MODEL_CACHE

    @staticmethod
    def clear_model_cache() -> None:
        """
        Drop the cached model, so the next load rebuilds it

        For tests and measurement. The application has no reason to call it: the model is read-only
        once loaded, and dropping it only means paying the load and the re-trace again.

        :return: None
        """
        global _MODEL_CACHE
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE = None

    def get_marked_maps(self) -> List[np.ndarray]:
        """
        Method to create focus maps for the given channels

        Focus maps only, despite what this said until 2026-08-15. Nucleus detection via FCN was
        dropped: Detector.nucleus_extraction always uses NucleusMapper, and FCNMapper is only ever
        constructed inside the focus branch (detection_method "u-net" or "combined").

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

        Progress is reported per channel only. The inference itself is one `model.predict` call
        covering every tile at once, which is where essentially all of the ~21 s goes and which
        this loop cannot see into; reporting inside it would need a Keras callback, and useful
        granularity would also need a smaller batch size than the default, at some cost in
        inference throughput.

        :return: The prediction maps
        """
        prediction_maps = []
        channels = list(self.channels)
        count = max(1, len(channels))
        for ind, channel in enumerate(channels):
            self.progress(ind / count, f"Detecting foci on channel {ind + 1}/{count} (u-net)")
            orig_shape = channel.shape
            orig_dtype = channel.dtype
            # Resize the channel to match the training size
            # TODO resizen von feature größe abhängig machen, tilen übernimmt den Rest
            channel = resize(channel, output_shape=self.TRAINING_SHAPE,
                             preserve_range=True, anti_aliasing=True).astype(orig_dtype)
            # Split channel images into tiles
            tiles = self.extract_subimages(channel, self.TILE_SHAPE)
            # Predict the individual tiles
            ptiles = self.predict_tiles(tiles, self.model)
            # Merge the tiles back into TRAINING_SHAPE, then resize that to the image's own shape.
            # The merge must be told the shape the tiles were CUT FROM, not the original: it derives
            # the tile grid from that shape, and the tiles came from the resized channel. Passing
            # orig_shape asked for a grid the tile list could not satisfy -- 225 tiles for a 2048
            # image against the 49 that exist (IndexError), and only 9 of 49 for a 512 one, which
            # reconstructed the top-left corner and left the rest of the map at zero.
            #
            # It also made the resize below a no-op, because the merge already returned orig_shape.
            # That resize is the step that brings the prediction back into image coordinates, and it
            # only does real work now that the merge produces TRAINING_SHAPE.
            merged = self.merge_prediction_tiles(ptiles, self.TRAINING_SHAPE,
                                                 orig_dtype=orig_dtype)
            pred_map = resize(merged, output_shape=orig_shape,
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
        orig_max = dtype_max(tiles[0].dtype)
        tiles = np.asarray(tiles).astype("float32")
        tiles /= orig_max
        # Derived from TILE_SHAPE, not written out again. This used to read
        # reshape(-1, 256, 256, 1) while map_channels cut the tiles with its own literal, so the
        # tile size was stated independently in two places -- exactly the shape of the defect this
        # class already had, where merge_prediction_tiles derived its grid from one shape while the
        # tiles came from another. Two statements of the same fact only agree until one is edited.
        tile_height, tile_width = FCNMapper.TILE_SHAPE
        tiles = tiles.reshape(-1, tile_height, tile_width, 1)
        return [pred[:, :, 0] for pred in model.predict(tiles)]

    @staticmethod
    def merge_prediction_tiles(masks: List[np.ndarray],
                               tiled_shape: Tuple[int, int],
                               overlap: float = 0.5,
                               orig_dtype: np.dtype = None) -> np.ndarray:
        """
        Method to merge created prediction masks into one large image

        :param masks: A list containing the created prediciton masks
        :param overlap: The overlap between prediction masks
        :param tiled_shape: The shape the tiles were CUT FROM -- i.e. the resized channel, not the
                            original image. The tile grid is derived from it, so it has to be the
                            shape extract_subimages was given or the grid will not match the number
                            of masks. It was named orig_shape and passed the original image shape,
                            which is what made this raise IndexError for anything above 1024x1024
                            and silently drop most of the prediction below it.
        :param orig_dtype: The dtype of the original image. Required -- see below
        :return: The merged prediction mask, of tiled_shape
        """
        # No default dtype. The masks are float predictions and carry no memory of the bit depth
        # they were derived from, so the previous default of uint8 would quietly scale a 16-bit
        # image's prediction map onto 0..255 for any caller that forgot the argument. Every caller
        # passes it today; this makes forgetting it a failure rather than a silent one
        if orig_dtype is None:
            raise ValueError("orig_dtype is required: the prediction masks are floats and carry "
                             "no source bit depth")
        # TODO Overlap einstellbar machen
        # Create an accumulator map as well as a weights map
        accum = np.zeros(tiled_shape, np.float32)
        weights = np.zeros(tiled_shape, np.float32)
        tile_height, tile_width = masks[0].shape[0], masks[0].shape[1]
        step_height = step_width = int(tile_height * (1 - overlap))
        n_tiles_vert = int(((tiled_shape[0] - tile_height) / step_height)) + 1
        n_tiles_hor = int(((tiled_shape[1] - tile_width) / step_width)) + 1
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
                          where=weights!=0) * dtype_max(orig_dtype)).astype(orig_dtype)

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

