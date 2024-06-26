import datetime
import hashlib
import os
from fractions import Fraction
from typing import Dict, Union, List

import numpy as np
import piexif
from skimage import io


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
            x_res = tags["0th"].get(piexif.ImageIFD.XResolution, (-1, -1))
            y_res = tags["0th"].get(piexif.ImageIFD.YResolution, (-1, -1))
            unit = tags["0th"].get(piexif.ImageIFD.ResolutionUnit, 2)
            """
            dt = tags["0th"].get(piexif.ImageIFD.DateTime,
                                 datetime.datetime.fromtimestamp(os.path.getctime(path)))
            """
            image_data = {
                "datetime": datetime.datetime.fromtimestamp(os.path.getctime(path)),
                "height": tags["0th"].get(piexif.ImageIFD.ImageLength, img.shape[0]),
                "width": tags["0th"].get(piexif.ImageIFD.ImageWidth, img.shape[1]),
                "x_res": float(Fraction(x_res[0], x_res[1])),
                "y_res": float(Fraction(y_res[0], y_res[1])),
                "channels": tags["0th"].get(piexif.ImageIFD.SamplesPerPixel, 3),
                "unit": ImageLoader._convert_tag_to_unit(unit)
            }
        else:
            image_data = {
                "datetime": datetime.datetime.fromtimestamp(os.path.getctime(path)),
                "height": img.shape[0],
                "width": img.shape[1],
                "x_res": -1,
                "y_res": -1,
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
    def _convert_tag_to_unit(unit: int) -> str:
        """
        Method to get the name of the unit from int

        :param unit: The index
        :return: The unit as string
        """
        return ["No Unit", "Inch", "Centimeter"][unit-1]

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
