from typing import Iterable, Dict

import numpy as np

from core.progress import NO_PROGRESS, ProgressReporter


class AreaMapper:
    # NO __slots__ HERE, DELIBERATELY -- and none on the subclasses either.
    #
    # The convention in this project is to slot the classes that are instantiated in bulk, not
    # every class. ROI is the one that pays for it: at 200 bytes saved per instance and a few
    # thousand ROI per image, that is hundreds of KiB. A mapper is constructed three times per
    # analysis run (FocusMapper, NucleusMapper, FCNMapper), so slotting the whole hierarchy saved
    # a measured 840 bytes -- against channels of 2 MiB each.
    #
    # These classes did carry a declaration, misspelled `___slots__` with three underscores, so it
    # was inert and instances kept their __dict__ regardless. Correcting the spelling was not the
    # cheap fix it looked like: the lists had drifted while nothing enforced them, and activating
    # them would have raised AttributeError inside FCNMapper.__init__, which assigns two attributes
    # the list does not name. They were deleted instead.

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
