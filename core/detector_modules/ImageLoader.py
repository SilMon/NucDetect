import datetime
import hashlib
import os
from fractions import Fraction
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import piexif
from skimage import io

# Value written to x_res/y_res when an image declares no usable resolution. Both branches of
# get_image_data use it, so "unknown" has one representation rather than one per file format.
# Deliberately private: a future change replaces it with None/SQL NULL, and publishing a name
# that is meant to be retracted would invite callers to couple to it.
_UNKNOWN_SCALE = -1


class ImageLoader:
    FORMATS = [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".bmp"
    ]

    @staticmethod
    def get_image_data(path: str) -> Dict[str, Union[int, float, str]]:
        """
        Method to extract relevant metadata from an image

        :param path: The URL of the image
        :return: The extracted metadata as dict
        """
        filename, file_extension = os.path.splitext(path)
        img = ImageLoader.load_image(path)
        if file_extension in (".tiff", ".tif", ".jpg"):
            tags = piexif.load(path)
            # No default here on purpose -- an absent tag is _rational_to_scale's business.
            # Handing .get() a default is what let the two format branches disagree about
            # what "unknown" means.
            x_res = tags["0th"].get(piexif.ImageIFD.XResolution)
            y_res = tags["0th"].get(piexif.ImageIFD.YResolution)
            unit = tags["0th"].get(piexif.ImageIFD.ResolutionUnit, 2)
            """
            dt = tags["0th"].get(piexif.ImageIFD.DateTime,
                                 datetime.datetime.fromtimestamp(os.path.getctime(path)))
            """
            image_data = {
                "datetime": datetime.datetime.fromtimestamp(os.path.getctime(path)),
                "height": tags["0th"].get(piexif.ImageIFD.ImageLength, img.shape[0]),
                "width": tags["0th"].get(piexif.ImageIFD.ImageWidth, img.shape[1]),
                "x_res": ImageLoader._rational_to_scale(x_res),
                "y_res": ImageLoader._rational_to_scale(y_res),
                "channels": tags["0th"].get(piexif.ImageIFD.SamplesPerPixel, 3),
                "unit": ImageLoader._convert_tag_to_unit(unit)
            }
        else:
            image_data = {
                "datetime": datetime.datetime.fromtimestamp(os.path.getctime(path)),
                "height": img.shape[0],
                "width": img.shape[1],
                "x_res": _UNKNOWN_SCALE,
                "y_res": _UNKNOWN_SCALE,
                "channels": 1 if len(img.shape) == 2 else 3,
                "unit": "Inch"
            }
        # Convert extracted time stamp
        tt = image_data["datetime"].timetuple()
        image_data["year"] = tt.tm_year
        image_data["month"] = tt.tm_mon
        image_data["day"] = tt.tm_mday
        image_data["hour"] = tt.tm_hour
        image_data["minute"] = tt.tm_min
        image_data["second"] = tt.tm_sec
        return image_data

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
            raise Warning("Unsupported image format ->{}!".format(os.path.splitext(path)[1]))

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
        channels = []
        for ind in range(img.shape[2]):
            channels.append(img[..., ind])
        return channels
