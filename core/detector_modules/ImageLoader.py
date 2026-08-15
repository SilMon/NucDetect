import datetime
import hashlib
import os
from fractions import Fraction
from typing import Dict, Optional, Tuple, TypedDict, Union, List

import numpy as np
import piexif
from skimage import io

# Value written to x_res/y_res when an image declares no usable resolution. Both branches of
# get_image_data use it, so "unknown" has one representation rather than one per file format.
# Deliberately private: a future change replaces it with None/SQL NULL, and publishing a name
# that is meant to be retracted would invite callers to couple to it.
_UNKNOWN_SCALE = -1


def dtype_max(dtype: np.dtype) -> float:
    """
    Largest representable value for an image dtype

    ``np.iinfo`` accepts **integer** dtypes only and raises ``ValueError: Invalid integer data
    type`` for float32/float64, which took the whole quality check and the FCN normalisation down
    for any float image.

    **255.0 for floats, NOT scikit-image's 0..1 convention.** This project already made that
    decision, with its reasoning written out, in ``FocusMapper._rescale_to_channel_range``
    (~:221-226): *"a float channel keeps the previous 0-255 target rather than silently changing
    meaning -- the images this was written for are integer, and a float one is a separate question
    from bit depth."* Returning 1.0 here would put two different float conventions in one pipeline,
    and ``verify_bit_depth`` pins the 0..255 one.

    Takes a dtype rather than an array so the callers that only have a dtype to hand -- the
    re-scaling at the end of ``merge_prediction_tiles`` -- do not have to invent an array.

    :param dtype: The dtype to read, e.g. ``channel.dtype``
    :return: The dtype's maximum as float
    """
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 255.0


class ImageData(TypedDict):
    """
    The metadata dict returned by ImageLoader.get_image_data

    A TypedDict rather than Dict[str, ...]: the value types genuinely differ per key, and the
    previous ``Dict[str, Union[int, float, str]]`` was wrong about the one key that is neither --
    ``datetime`` holds a ``datetime.datetime``, which ``gui/Util.py`` then calls ``strftime`` on.
    A widened union would only move the problem to the callers, every one of which would have to
    narrow before use; ``Inserter.add_new_image`` takes eleven of these values as typed parameters.
    """
    #: File creation time. Both format branches use it, and the six fields below are derived from it
    datetime: datetime.datetime
    height: int
    width: int
    #: Pixels per `unit`, or _UNKNOWN_SCALE when the image declares no usable resolution
    x_res: float
    y_res: float
    channels: int
    #: Resolution unit name, e.g. "Inch" or "Centimeter"
    unit: str
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int


class ImageLoader:
    FORMATS = [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".bmp"
    ]

    @staticmethod
    def get_image_data(path: str) -> ImageData:
        """
        Method to extract relevant metadata from an image

        :param path: The URL of the image
        :return: The extracted metadata, see ImageData for the keys and their types
        """
        filename, file_extension = os.path.splitext(path)
        img = ImageLoader.load_image(path)
        # Hoisted out of the two branches, which computed it identically. It is also what the six
        # calendar fields are derived from, so reading the clock once keeps them consistent with it
        created = datetime.datetime.fromtimestamp(os.path.getctime(path))
        if file_extension in (".tiff", ".tif", ".jpg"):
            tags = piexif.load(path)
            # No default here on purpose -- an absent tag is _rational_to_scale's business.
            # Handing .get() a default is what let the two format branches disagree about
            # what "unknown" means.
            x_res_tag = tags["0th"].get(piexif.ImageIFD.XResolution)
            y_res_tag = tags["0th"].get(piexif.ImageIFD.YResolution)
            unit_tag = tags["0th"].get(piexif.ImageIFD.ResolutionUnit, 2)
            """
            dt = tags["0th"].get(piexif.ImageIFD.DateTime,
                                 datetime.datetime.fromtimestamp(os.path.getctime(path)))
            """
            height = tags["0th"].get(piexif.ImageIFD.ImageLength, img.shape[0])
            width = tags["0th"].get(piexif.ImageIFD.ImageWidth, img.shape[1])
            x_res = ImageLoader._rational_to_scale(x_res_tag)
            y_res = ImageLoader._rational_to_scale(y_res_tag)
            channels = tags["0th"].get(piexif.ImageIFD.SamplesPerPixel, 3)
            unit = ImageLoader._convert_tag_to_unit(unit_tag)
        else:
            height = img.shape[0]
            width = img.shape[1]
            x_res = _UNKNOWN_SCALE
            y_res = _UNKNOWN_SCALE
            channels = 1 if len(img.shape) == 2 else 3
            unit = "Inch"
        # Built in one literal rather than filled in two stages: a TypedDict has to be complete at
        # construction, and the staged version is what let the return annotation drift from the
        # actual contents in the first place
        tt = created.timetuple()
        return ImageData(
            datetime=created,
            height=height,
            width=width,
            x_res=x_res,
            y_res=y_res,
            channels=channels,
            unit=unit,
            year=tt.tm_year,
            month=tt.tm_mon,
            day=tt.tm_mday,
            hour=tt.tm_hour,
            minute=tt.tm_min,
            second=tt.tm_sec,
        )

    @staticmethod
    def _rational_to_scale(value: Optional[Tuple[int, int]]) -> float:
        """
        Method to convert an EXIF RATIONAL (numerator, denominator) into a scale

        Returns _UNKNOWN_SCALE when the tag is absent, malformed, or carries a zero
        denominator. The previous default of (-1, -1) did not survive Fraction, which
        normalises signs: Fraction(-1, -1) is 1, so a missing tag produced a scale of exactly
        1.0 -- both a legal resolution and the multiplicative identity, so no consumer could
        tell it from real metadata and no conversion visibly changed anything. A zero
        denominator (written by some microscope software for "unset") raised ZeroDivisionError
        out of image import instead.

        :param value: The RATIONAL tag value, or None if the tag is absent
        :return: The scale as float, or _UNKNOWN_SCALE if it cannot be determined
        """
        if not value:
            return _UNKNOWN_SCALE
        try:
            num, den = value
        except (TypeError, ValueError):
            return _UNKNOWN_SCALE
        if den == 0:
            return _UNKNOWN_SCALE
        return float(Fraction(num, den))

    @staticmethod
    def _convert_tag_to_unit(unit: int) -> str:
        """
        Method to get the name of the unit from int

        :param unit: The TIFF ResolutionUnit tag value
        :return: The unit as string
        """
        # Indexed by the tag value itself. The previous `[unit - 1]` turned a ResolutionUnit of 0 --
        # which the TIFF specification defines as "no absolute unit" -- into index -1 and silently
        # returned "Centimeter", the most specific answer for the least specific input. An unknown
        # value returns the no-unit case rather than raising, matching how a missing resolution tag
        # is handled above
        units = ("No Unit", "No Unit", "Inch", "Centimeter")
        return units[unit] if 0 <= unit < len(units) else units[0]

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """
        Method to load an image given by path. Method will only load image formats specified by Detector.FORMATS

        :param path: The URL of the image
        :return: The image as ndarray
        """
        if os.path.splitext(path)[1] in ImageLoader.FORMATS:
            return io.imread(path)
        else:
            # ValueError, not Warning. Warning is not an error type -- it reads as advisory to
            # anyone scanning the code, and a caller guarding against bad input with the usual
            # (ValueError, OSError) would not have caught it.
            raise ValueError("Unsupported image format ->{}!".format(os.path.splitext(path)[1]))

    @staticmethod
    def calculate_image_id(path: str) -> str:
        """
        Method to calculate the md5 hash sum of the image described by path

        :param path: The URL of the image
        :return: The md5 hash sum as hex
        """
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def get_channels(img: np.ndarray) -> List[np.ndarray]:
        """
        Method to extract the channels of the given image

        :param img: The image as ndarray
        :return: A list of all channels
        """
        # A grayscale image is 2-D and has no channel axis, so img.shape[2] raises IndexError.
        # get_image_data already treats a 2-D array as one channel, so the two disagreed about
        # what a grayscale image is; this is the same reading of it.
        if img.ndim == 2:
            return [img]
        channels = []
        for ind in range(img.shape[2]):
            channels.append(img[..., ind])
        return channels
