import numpy as np
from typing import Iterable, Dict


class AreaMapper:
    ___slots__ = (
        "channels",
        "settings"
    )

    def __init__(self, channels: Iterable[np.ndarray] = None, settings: Dict = None):
        """
        :param channels: The channels this mapper uses
        :param settings: The settings this mapper uses
        """
        self.channels = channels
        self.settings = settings

    def set_channels(self, channels: Iterable[np.ndarray]) -> None:
        """
        Method to set the channels used by the mapper

        :param channels: The channels to set
        :return: None
        """
        self.channels = channels

    def set_settings(self, settings: Dict) -> None:
        """
        Method to set the settings used by this mapper
        :param settings: The settings to use
        :return: None
        """
        self.settings = settings
