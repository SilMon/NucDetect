from typing import Callable, Iterable, Dict

import numpy as np

from core.logging_config import get_logger
from core.progress import NO_PROGRESS, ProgressReporter

LOGGER = get_logger(__name__)


# Both reporting callables are module level on purpose, and must stay that way -- the same
# constraint QualityTester's pair carries, for the same reason. `_analyze_all` hands
# `Detector.analyse_image` to a ProcessPoolExecutor, which pickles the bound method and with it the
# whole Detector, including the three mappers built in its constructor. A class-body lambda cannot
# be pickled and would break batch analysis before a single image was read; a module-level function
# pickles by reference. Anything bound to `self.log` is subject to this.
def default_log(message: str) -> None:
    """
    Fallback used when logging is enabled but no reporting callable was injected

    The real flow always injects one -- `Detector.analyse_image` puts `add_log_message` into
    `analysis_settings["log"]` -- so this is reached only by a caller that builds a mapper itself.
    It writes through the shared logger rather than `print`, so the message reaches the log file.
    In a worker process the configured NullHandler makes it a no-op by design, which is why the
    injected buffer-and-replay callable is what the real flow uses.

    :param message: The message to report
    :return: None
    """
    LOGGER.info(message)


def no_log(message: str) -> None:
    """
    Bound to `self.log` when the `logging` setting is off, so a reporting call costs a call and
    nothing else

    A no-op function rather than a falsy attribute tested at each call site: it keeps the guard in
    one place instead of one per message, and keeps the call sites reading as plain reporting.

    :param message: Ignored
    :return: None
    """


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
        # Same shape as `progress`: a no-op until settings arrive, so subclasses report
        # unconditionally and the `logging` setting is honoured in exactly one place. Bound here
        # AND rebound in set_settings -- see the note there for why the constructor alone is not
        # enough
        self.log: Callable[[str], None] = no_log
        if settings:
            self.set_settings(settings)

    def set_channels(self, channels: Iterable[np.ndarray]) -> None:
        """
        Method to set the channels used by the mapper

        :param channels: The channels to set
        :return: None
        """
        self.channels = channels

    def set_settings(self, settings: Dict) -> None:
        """
        Method to set the settings used by this mapper, and to rebind the reporting callable that
        goes with them

        **Rebinding `self.log` here is part of this method's job, not a detail.** `Detector` builds
        its mappers with no settings and calls this later with the real ones, so a binding made
        only in the constructor would stay on the fallback for the lifetime of that Detector and
        the injected reporter -- the one that buffers per image for the parent to replay -- would
        never be used. That is exactly the defect QualityTester carried until 2026-08-15, and it
        cost every one of its messages the trip to the log file.

        Honouring the `logging` setting here is also what makes `_analyze_all`'s suppression real:
        it forces `logging` off for the duration of a batch and restores it afterwards, which does
        nothing unless a module reads the flag.

        :param settings: The settings to use
        :return: None
        """
        self.settings = settings
        # .get, not [], so a caller passing a partial dict falls back rather than raising; the two
        # keys are guaranteed only in a subclass's STANDARD_SETTINGS and in Detector's
        # analysis_settings
        self.log = settings.get("log", default_log) if settings.get("logging", False) else no_log

    def set_progress(self, progress: ProgressReporter) -> None:
        """
        Method to set the progress reporter used by this mapper

        The reporter owns a slice of the progress bar and is addressed in its own 0..1 space, so a
        mapper never needs to know where in the analysis it runs.

        :param progress: The reporter to report this mapper's progress to
        :return: None
        """
        self.progress = progress
