from typing import Iterable, Dict

import numpy as np

from core.progress import NO_PROGRESS, ProgressReporter


class AreaMapper:
    ___slots__ = (
        "channels",
        "settings",
        "progress"
    )

    def __init__(self, channels: Iterable[np.ndarray] = None, settings: Dict = None):
        """
        :param channels: The channels this mapper uses
        :param settings: The settings this mapper uses
        """
        self.channels = channels
        self.settings = settings
        # Defaults to a no-op, so a mapper used without a progress bar needs no special handling
        # and subclasses can report unconditionally
        self.progress: ProgressReporter = NO_PROGRESS

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

    def set_progress(self, progress: ProgressReporter) -> None:
        """
        Method to set the progress reporter used by this mapper

        The reporter owns a slice of the progress bar and is addressed in its own 0..1 space, so a
        mapper never needs to know where in the analysis it runs.

        :param progress: The reporter to report this mapper's progress to
        :return: None
        """
        self.progress = progress
